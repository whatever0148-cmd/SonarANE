"""Shared 48 kHz FMCW ultrasonic sonar math for SonarANE.

Everything the CoreML model, the CPU baselines and the live test agree on lives here:
chirp parameters, the transmit waveform, and the fused dechirp + range-DFT kernel that
becomes the weight of the Neural Engine 1x1 conv.
"""
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import get_window
from scipy.signal.windows import tukey

FS = 48_000                      # Hz, built-in speaker + mic native rate
F0, F1 = 18_800.0, 22_300.0      # Hz, sweep band (Phase 1: 18.5-22, Phase 1b: 17.2-20.7 kHz)
B = F1 - F0                      # 3500 Hz sweep bandwidth
FC = 0.5 * (F0 + F1)             # 20.55 kHz centre frequency
T_CHIRP = 0.04                   # s per chirp (PRF = 25 Hz)
N = int(round(FS * T_CHIRP))     # 1920 samples per chirp = conv C_in
SLOPE = B / T_CHIRP              # 87.5 kHz/s
N_CHIRPS = 150                   # 6.0 s of continuous chirps
C_SOUND = 343.0                  # m/s at ~20 C
RANGE_RES_M = C_SOUND / (2 * B)  # 4.90 cm native range resolution
RD_COUPLING_S = FC / SLOPE       # 0.235 s: up-chirp beat sees r_meas = r + RD_COUPLING_S * dr/dt

K_BINS = 128                     # range bins kept
BIN_STEP = 0.5                   # 2x zero-padded range DFT -> 2.45 cm grid, 0..3.1 m
C_OUT = 2 * K_BINS               # [Re | Im] output channels


def chirp_phase(n):
    t = np.asarray(n, dtype=np.float64) / FS
    return 2 * np.pi * (F0 * t + 0.5 * SLOPE * t * t)


def make_chirp(alpha=0.1):
    """One Tukey-windowed linear up-chirp, N samples, unit peak."""
    n = np.arange(N)
    return (np.cos(chirp_phase(n)) * tukey(N, alpha)).astype(np.float32)


def make_tx(n_chirps=N_CHIRPS, alpha=0.15, amp=0.9, lead_s=0.1, tail_s=0.6):
    """Short stream-start guard + continuous chirp train + silent tail (I/O latency, then noise floor)."""
    lead = np.zeros(int(lead_s * FS), np.float32)
    tail = np.zeros(int(tail_s * FS), np.float32)
    train = np.tile(make_chirp(alpha), n_chirps) * amp
    return np.concatenate([lead, train, tail]).astype(np.float32), len(lead)


def range_axis_m(k_bins=K_BINS, bin_step=BIN_STEP):
    """Bin k -> beat f_b = k*step/T -> tau = f_b/SLOPE -> range = c*tau/2 (relative to direct path)."""
    k = np.arange(k_bins) * bin_step
    return k * RANGE_RES_M


def window_gain(window="hann"):
    return float(get_window(window, N, fftbins=True).sum())


def dechirp_dft_kernel(k_bins=K_BINS, bin_step=BIN_STEP, window="hann"):
    """Fused matched-filter (dechirp) + windowed range-DFT as one real [2K, N] matrix.

    For a real received frame x[n]:
        X_k = sum_n h[n] * exp(+j*phi[n]) * exp(-j*2*pi*k*step*n/N) * x[n]
    Mixing with exp(+j*phi) maps an echo of delay tau to the positive beat
    f_b = SLOPE*tau; the sum-frequency image lands at +(37.6..44.6) kHz -> -(10.4..3.4) kHz,
    far above the kept bins (<= 3.2 kHz). Rows [0, K) are Re(X), rows [K, 2K) are Im(X).
    Weights are left un-normalised (|w| <= 1) so FP16 partial sums stay well inside the
    normal range; callers multiply by amp_scale() afterwards.
    """
    n = np.arange(N)
    h = get_window(window, N, fftbins=True)
    k = (np.arange(k_bins) * bin_step)[:, None]
    m = h * np.exp(1j * chirp_phase(n)) * np.exp(-2j * np.pi * k * n / N)
    return np.concatenate([m.real, m.imag], axis=0).astype(np.float32)


def amp_scale(window="hann"):
    """Multiply |X| by this to read echo amplitude in input units (cos of amp a -> a)."""
    return 2.0 / window_gain(window)


MTI_NOISE_GAIN = np.sqrt(6.0)    # white-noise amplitude gain of [1, -2, 1] (power gain 1 + 4 + 1)


def mti2(frames):
    """Second-order slow-time MTI canceller y_m = x_m - 2 x_{m-1} + x_{m-2} on time-domain frames.

    Linear, so identical to applying it after the range-DFT, but doing it first strips the direct
    speaker->mic path before the FP16 cast and leaves the echoes more of FP16's precision.
    Response 4*sin^2(pi*f_d/PRF): nulls static clutter, -12 dB at 2 Hz Doppler, max gain at 12.5 Hz.
    """
    return frames[2:] - 2.0 * frames[1:-1] + frames[:-2]


def to_complex(out):
    """[.., 2K, S] real conv output -> complex [S, K]."""
    out = np.asarray(out, dtype=np.float32).reshape(C_OUT, -1)
    return (out[:K_BINS] + 1j * out[K_BINS:]).T


def fft_pipeline(frames, window="hann"):
    """Classic CPU reference: dechirp then zero-padded rFFT. frames: [S, N] -> complex [S, K]."""
    n = np.arange(N)
    h = get_window(window, N, fftbins=True)
    ref = (h * np.exp(1j * chirp_phase(n))).astype(np.complex64)
    nfft = int(round(N / BIN_STEP))
    return np.fft.fft(frames * ref, n=nfft, axis=-1)[:, :K_BINS]


# ---------------------------------------------------------------- Phase 2: dual-path clutter + tracker
LEAK_FIT_CM = 10.0               # bins used to fit the per-chirp direct-path drift (gain + delay)
PRESENCE_MIN_CM = 15.0           # Path B is blind below this: residual direct-path drift lives there


def smooth_mag(mag, bins=5):
    """Range-smoothed magnitude: merges the multi-lobe response of a moving / extended target."""
    return np.sqrt(uniform_filter1d(mag ** 2, size=bins, axis=-1, mode="nearest"))


def os_cfar(mag, pct=25):
    """Slow-time order-statistic CFAR ratio: each range bin over its own pct-th percentile in time."""
    return mag / np.percentile(mag, pct, axis=0)


def direct_path_basis(window="hann"):
    """[3, K] exact range responses of the direct path (delay 0) and its 1st/2nd delay derivatives.

    These are the true shapes of direct-path leakage (mainlobe + every sidelobe) and of its sub-sample
    delay drift, so fitting them removes drift residual at all ranges without touching real reflectors.
    """
    n = np.arange(N)
    u = [fft_pipeline(np.cos(chirp_phase(n - d))[None], window)[0] * amp_scale(window) for d in (-0.5, 0.0, 0.5)]
    return np.stack([u[1], u[2] - u[0], u[2] - 2 * u[1] + u[0]])


def presence_residual(X, trim=0.3, iters=3, lock="near"):
    """Path B: |X_m - background_m| with a hand-robust, leakage-locked background.

    B0 is a per-bin complex trimmed mean: chirps where a bin deviates most from the current background
    (the hand is probably in it) are dropped and B0 is re-estimated, so a hand that pauses for a while
    does not end up inside its own background. The direct speaker->mic path drifts in gain/phase and by
    fractions of a sample in delay; lock="near" fits that drift per chirp on the direct-path bins
    (< LEAK_FIT_CM) with direct_path_basis() and subtracts it everywhere. lock="none" uses B0 alone.
    """
    fit = range_axis_m() * 100 < LEAK_FIT_CM
    u = direct_path_basis()

    def background(b0):
        if lock == "none":
            return np.broadcast_to(b0, X.shape)
        coef, *_ = np.linalg.lstsq(u[:, fit].T, (X[:, fit] - b0[fit]).T, rcond=None)
        return b0 + coef.T @ u

    b0 = X.mean(0)
    for _ in range(iters):
        dev = np.abs(X - background(b0))
        keep = dev <= np.quantile(dev, 1 - trim, axis=0)
        b0 = b0 + ((X - background(b0)) * keep).sum(0) / keep.sum(0)
    return np.abs(X - background(b0))


def dual_path_scores(X, Y, smooth_bins=5, pct=25, lock="near", presence_min_cm=PRESENCE_MIN_CM):
    """CFAR ratios of Path A (MTI motion, rows of Y) and Path B (presence) on the same chirp grid.

    Y = mti2 spectra; row m is centred on chirp m + 1, so Path B uses X[1:-1].
    """
    a = os_cfar(smooth_mag(np.abs(Y), smooth_bins), pct)
    pres = presence_residual(X, lock=lock)
    b = os_cfar(smooth_mag(pres[1:-1], smooth_bins), pct)
    b[:, range_axis_m() * 100 < presence_min_cm] = 0.0
    return a, b


def tgc_threshold_db(r_cm, near_db=11.0, far_db=8.0, flat_until_cm=35.0, far_from_cm=75.0):
    """Range-adaptive CFAR threshold (the TGC equivalent; a per-bin gain would cancel in the CFAR ratio).

    near_db up to flat_until_cm (direct-path sidelobes, strong near echoes), raised-cosine taper to far_db
    at far_from_cm, flat beyond: two-way spreading makes far hands weak, so they get a lower bar.
    """
    x = np.clip((np.asarray(r_cm, float) - flat_until_cm) / (far_from_cm - flat_until_cm), 0, 1)
    return far_db + (near_db - far_db) * 0.5 * (1 + np.cos(np.pi * x))


def candidates(score, r_cm, gate_cm, thr):
    """Per chirp: local maxima of score above thr (scalar or per range bin) inside gate, sub-bin
    interpolated, sorted near -> far."""
    thr = np.broadcast_to(np.asarray(thr, float), r_cm.shape)
    out = []
    lo, hi = np.searchsorted(r_cm, gate_cm[0]), np.searchsorted(r_cm, gate_cm[1], side="right")
    step = r_cm[1] - r_cm[0]
    ls = np.log(score + 1e-20)
    for row, lrow in zip(score, ls):
        k = np.arange(max(lo, 1), min(hi, len(row) - 1))
        pk = k[(row[k] > thr[k]) & (row[k] >= row[k - 1]) & (row[k] >= row[k + 1])]
        a, b, c = lrow[pk - 1], lrow[pk], lrow[pk + 1]
        den = a - 2 * b + c
        frac = np.clip(0.5 * (a - c) / np.where(den == 0, -1e-9, den), -0.5, 0.5)
        out.append((r_cm[pk] + frac * step, row[pk]))
    return out


def kalman_track(cands, kappa=None, t_chirp=T_CHIRP, sigma_a=600.0, sigma_z=2.5, gate_sigma=3.0,
                 gate_cm=15.0, max_coast=15, init_run=4, init_step=10.0, coast_decay=0.9,
                 init_run_far=6, init_far_cm=35.0):
    """Constant-velocity Kalman tracker on [range cm, range-rate cm/s] with closest-scatterer bias.

    Measurement model z = r + kappa * v absorbs the up-chirp range-Doppler coupling directly.
    Initiation (the Phase 1b continuity gate, multi-hypothesis): every candidate extends any tentative
    chain that ended on the previous chirp within init_step cm; a chain of init_run chirps confirms a
    track, the nearest confirmed chain wins. Chains whose mean range is beyond init_far_cm (where the
    TGC threshold is lower) need init_run_far chirps instead: same track false-alarm rate. Update: among candidates inside the gate (Mahalanobis <=
    gate_sigma and |innovation| <= gate_cm) take the NEAREST in range (palm before forearm / book edge),
    not the strongest. No candidate in gate -> coast on the prediction, velocity multiplied by
    coast_decay per chirp (0.9: pure CV coasting ran a held book 25 -> 54 cm onto another scatterer in the
    Phase 3 capture; Phase 1b replay preferred 1.0, see MEMORY.md); more than
    max_coast misses ->
    drop, and the unconfirmed coast is erased. Returns r, v (nan when no track) and status per chirp:
    2 = measurement update, 1 = coast, 0 = none.
    """
    kappa = RD_COUPLING_S if kappa is None else kappa
    f = np.array([[1.0, t_chirp], [0.0, 1.0]])
    q = sigma_a ** 2 * np.array([[t_chirp ** 4 / 4, t_chirp ** 3 / 2], [t_chirp ** 3 / 2, t_chirp ** 2]])
    h = np.array([1.0, kappa])
    n = len(cands)
    r_out, v_out, status = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, int)
    x = p = None
    misses = 0
    chains = []                                                   # list of [(chirp, z), ...]
    for m, (rng, _) in enumerate(cands):
        if x is None:
            ext = []
            for z in rng:
                prev = [c for c in chains if c[-1][0] == m - 1 and abs(c[-1][1] - z) <= init_step]
                best = max(prev, key=len) if prev else []
                ext.append(best + [(m, z)])
            chains = ext
            def need(c):                                  # longer confirmation where the threshold is lower
                far = init_run_far and np.mean([z for _, z in c[-init_run:]]) > init_far_cm
                return init_run_far if far else init_run

            done = [c for c in chains if len(c) >= need(c)]
            if done:
                chain = min(done, key=lambda c: c[-1][1])
                chain = chain[-need(chain):]
                zs = np.array([c[1] for c in chain])
                v0 = np.polyfit(np.arange(len(zs)) * t_chirp, zs, 1)[0]
                x = np.array([zs[-1] - kappa * v0, v0])
                p = np.diag([sigma_z ** 2 * 2, 30.0 ** 2])
                misses = 0
                for mm, z in chain:
                    status[mm], r_out[mm], v_out[mm] = 2, z - kappa * v0, v0
                chains = []
            continue
        x = f @ x
        p = f @ p @ f.T + q
        s = h @ p @ h + sigma_z ** 2
        nu = rng - h @ x
        ok = (nu ** 2 / s <= gate_sigma ** 2) & (np.abs(nu) <= gate_cm)
        if ok.any():
            z = rng[ok].min()                                    # closest-scatterer bias
            kg = p @ h / s
            x = x + kg * (z - h @ x)
            p = (np.eye(2) - np.outer(kg, h)) @ p
            misses = 0
            status[m] = 2
        else:
            misses += 1
            x[1] *= coast_decay
            if misses > max_coast:
                x = None
                drop = slice(m - max_coast, m)
                status[drop], r_out[drop], v_out[drop] = 0, np.nan, np.nan
                continue
            status[m] = 1
        r_out[m], v_out[m] = x
    if x is not None and misses:                                  # trailing coast never confirmed
        tail = slice(n - misses, n)
        status[tail], r_out[tail], v_out[tail] = 0, np.nan, np.nan
    return r_out, v_out, status


# ---------------------------------------------------------------- Phase 3: respiration interferometry
LAMBDA_C = C_SOUND / FC          # 1.67 cm at 20.55 kHz: 1 mm radial motion = 4*pi/lambda = 0.75 rad
RESP_BAND_HZ = (0.12, 0.60)      # 7-36 breaths/min


def circle_center(z):
    """Algebraic (Kasa) least-squares circle through complex samples -> (centre, radius)."""
    a = np.stack([z.real, z.imag, np.ones(len(z))], 1)
    (d, e, f), *_ = np.linalg.lstsq(a, np.abs(z) ** 2, rcond=None)
    c = complex(d / 2, e / 2)
    return c, float(np.sqrt(max(f + abs(c) ** 2, 0.0)))


def displacement_mm(X, k, ref_bin=0, reference=True, arc_min_rad=1.0):
    """Radial displacement (mm, + = moving away) of range bin k from its slow-time complex phase.

    reference=True rotates every chirp by the direct-path bin phase: a frame-timing error delta (integer
    alignment rounding, clock drift, jitter) adds ~2*pi*fc*delta to every bin alike, so it cancels (whatever
    the direct path itself physically drifts is injected instead). Static clutter sharing bin k offsets the
    IQ arc from the origin, which compresses naive phase; when the arc spans >= arc_min_rad around a fitted
    circle centre, phase is measured around that centre (arc-centre demodulation).
    Returns (displacement_mm, info).
    """
    z = np.asarray(X[:, k], complex)
    if reference:
        z = z * np.exp(-1j * np.angle(X[:, ref_bin]))
    c, rad = circle_center(z)
    phi_c = np.unwrap(np.angle(z - c))
    radial_res = float(np.std(np.abs(z - c) - rad))
    # Only trust the centre when the samples really lie on an arc: a static noisy cluster also "fits" a
    # circle, and phase measured around that arbitrary centre is garbage.
    fit_ok = np.ptp(phi_c) >= arc_min_rad and radial_res < 0.25 * rad and rad < 5 * np.abs(z).max()
    phi = phi_c if fit_ok else np.unwrap(np.angle(z))
    d = (phi - phi[0]) * LAMBDA_C / (4 * np.pi) * 1000
    return d, {"arc_centre_fit": bool(fit_ok), "arc_span_rad": float(np.ptp(phi_c)),
               "arc_radial_residual": radial_res / rad if rad > 0 else None,
               "clutter_to_target_db": float(20 * np.log10(abs(c) / rad)) if fit_ok and rad > 0 else None}


def breathing_bin(X, rows, r_lo_cm=20.0, r_hi_cm=65.0, fs=1 / T_CHIRP):
    """Range bin in [r_lo, r_hi] with the most respiratory-band echo power over chirps `rows`.

    Uses the complex slow-time signal (not its phase), so weak bins with random phase cannot win.
    """
    from scipy.signal import butter, sosfiltfilt
    r = range_axis_m() * 100
    ks = np.flatnonzero((r >= r_lo_cm) & (r <= r_hi_cm))
    z = X[rows][:, ks] * np.exp(-1j * np.angle(X[rows, 0]))[:, None]
    z = z - z.mean(0)
    sos = butter(2, RESP_BAND_HZ[1], "low", fs=fs, output="sos")
    p = (np.abs(sosfiltfilt(sos, z, axis=0)) ** 2).mean(0)
    return int(ks[np.argmax(p)]), p


def respiration(d_mm, fs=1 / T_CHIRP, band=RESP_BAND_HZ):
    """Breathing waveform, rate (three-way consensus) and depth from a displacement trace.

    Windows shorter than 2 periods of band[0] cannot carry a band[0] high-pass, so they are linearly
    detrended and low-passed at band[1] instead. Three independent rate estimates:
      spectrum  - Hann-windowed, zero-padded spectrum peak in band with parabolic (log) interpolation;
      autocorr  - first local maximum of the unbiased autocorrelation in the band's lag range
                  (insensitive to how deep each individual breath is);
      peaks     - inhalation peaks of the waveform divided by its Hilbert envelope, so shallow breaths
                  clear the same prominence bar as deep ones; median peak spacing.
    Rate = median of the available estimates. Confidence: 'low' under 2 cycles, 'high' when all three
    agree within 5 %, 'ok' within 15 %, else 'moderate'.
    """
    from scipy.signal import butter, detrend, find_peaks, hilbert, sosfiltfilt
    n = len(d_mm)
    dur = n / fs
    x = detrend(np.asarray(d_mm, float))
    long_enough = dur >= 2 / band[0]
    sos = (butter(2, band, "bandpass", fs=fs, output="sos") if long_enough
           else butter(2, band[1], "low", fs=fs, output="sos"))
    w = sosfiltfilt(sos, x)

    # 1. spectrum
    nfft = 8192
    spec = np.abs(np.fft.rfft(w * np.hanning(n), nfft))
    f = np.fft.rfftfreq(nfft, 1 / fs)
    ib = np.flatnonzero((f >= band[0]) & (f <= band[1]))
    i = ib[np.argmax(spec[ib])]
    a, b, c = np.log(spec[i - 1:i + 2] + 1e-30)
    f_spec = float(f[i] + 0.5 * (a - c) / (a - 2 * b + c) * (f[1] - f[0])) if a - 2 * b + c < 0 else float(f[i])

    # 2. unbiased autocorrelation, first peak in the band's lag range (needs >= 2 periods of data)
    full = np.correlate(w, w, "full")[n - 1:] / (n - np.arange(n))
    acf = full / full[0]
    lo, hi = int(fs / band[1]), min(int(fs / band[0]), n // 2)
    f_acf = None
    if hi > lo + 2:
        pk_acf, _ = find_peaks(acf[lo:hi + 1])
        # Periodicity test: the peak must rise > 0.3 above the half-period trough before it. With irregular
        # breath intervals the one-period peak itself can be low (0.13 on the 30 s capture) while the
        # trough is deep (-0.51); a fixed height floor would skip to the 2-period lag and halve the rate.
        pk_acf = [k for k in lo + pk_acf if acf[k] > 0 and acf[k] - acf[1:k].min() > 0.3]
        if len(pk_acf):
            k = pk_acf[0]                              # FIRST peak = one period (later ones are its multiples)
            a, b, c = acf[k - 1:k + 2]
            lag = k + (0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c < 0 else 0.0)
            f_acf = float(fs / lag)

    # 3. envelope-normalised peaks
    env = np.abs(hilbert(w))
    env = sosfiltfilt(butter(2, band[0], "low", fs=fs, output="sos"), env) if long_enough else np.full(n, env.mean())
    wn = w / np.maximum(env, 1e-3 * np.abs(w).max() + 1e-12)
    pk, _ = find_peaks(wn, prominence=0.5, distance=int(fs / band[1] * 0.8))
    f_pk = float(1 / np.median(np.diff(pk) / fs)) if len(pk) >= 2 else None

    est = {"spectrum": f_spec, "autocorr": f_acf, "peaks": f_pk}
    ok = [v for v in est.values() if v is not None and band[0] <= v <= band[1]]
    f_est = float(np.median(ok)) if ok else f_spec
    spread = float(max(abs(v - f_est) / f_est for v in ok)) if len(ok) > 1 else None
    cycles = dur * f_est
    if cycles < 2:
        confidence = "low"
    elif len(ok) == 3 and spread < 0.05:
        confidence = "high"
    elif len(ok) >= 2 and spread < 0.15:
        confidence = "ok"
    else:
        confidence = "moderate"
    bpm = lambda v: None if v is None else float(60 * v)
    return w, {"window_s": float(dur), "filter": "bandpass" if long_enough else "detrend + low-pass",
               "bpm": float(60 * f_est), "bpm_from_spectrum": bpm(f_spec), "bpm_from_autocorr": bpm(f_acf),
               "bpm_from_peaks": bpm(f_pk), "estimator_spread_pct": None if spread is None else 100 * spread,
               "cycles_observed": float(cycles), "confidence": confidence, "peak_to_peak_mm": float(np.ptp(w)),
               "inhalation_peaks_s": [float(i / fs) for i in pk]}


def quiet_segment(X, margin_db=6.0, merge_gap=10, r_lo_cm=15.0, r_hi_cm=80.0):
    """Longest motion-free stretch of chirps (no body or hand movement), for breathing analysis.

    Per-chirp [1,-2,1] MTI energy in r_lo..r_hi must stay within margin_db of its median; gaps up to
    merge_gap chirps are bridged. Movement makes phase jump ~pi per chirp, so unwrapping across it
    slips whole wavelengths (8 mm each) and fakes tens of millimetres of "breathing".
    Returns (rows, mti_energy_db per chirp).
    """
    r = range_axis_m() * 100
    sel = (r >= r_lo_cm) & (r <= r_hi_cm)
    y = mti2(X)
    e = 20 * np.log10(np.sqrt((np.abs(y[:, sel]) ** 2).mean(1)) + 1e-30)
    quiet = np.concatenate([[False], e < np.median(e) + margin_db, [False]])   # MTI row m -> chirp m + 1
    edges = np.diff(np.concatenate([[0], quiet.astype(int), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    for a, b in zip(ends[:-1], starts[1:]):                                       # bridge short blips
        if b - a <= merge_gap:
            quiet[a:b] = True
    edges = np.diff(np.concatenate([[0], quiet.astype(int), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if not len(starts):
        return np.arange(len(X)), np.concatenate([[e[0]], e, [e[-1]]])
    i = int(np.argmax(ends - starts))
    return np.arange(starts[i], ends[i]), np.concatenate([[e[0]], e, [e[-1]]])


def breathing_displacement_mm(X, k, ref_bin=0, half_width=1, clamp_rad=1.2, amp_gate=(0.2, 0.5), reference=True):
    """Robust chest displacement (mm, + = away) from a 3-bin coherent neighbourhood around bin k.

    1. Every chirp is referenced to the 0 cm direct-path phase (common frame-timing error cancels).
    2. Bins k-half_width..k+half_width are rotated onto bin k by their mean phase offset and summed, so a
       torso spread over adjacent 2.45 cm bins, or shifting between them, keeps its amplitude and phase.
    3. Arc-centre demodulation of the summed signal when it really lies on an arc (see displacement_mm).
    4. Phase is integrated from per-chirp increments angle(z_t conj z_{t-1}), each in (-pi, pi]:
       clamped to +-clamp_rad (1.2 rad = 1.6 mm per 40 ms = 4 cm/s, above any breathing chest speed) and
       soft-gated by amplitude (weight 0 below amp_gate[0] x median |z|, 1 above amp_gate[1] x median), so
       a dropout holds the phase instead of slipping a whole 8.3 mm half-wavelength.
    Clamping truncates genuinely fast movement (an offset remains), so motion-free gating stays first.
    Simulated 30 s chest at -43 dB with a 15 mm posture shift and sway, 4 seeds: 14.6-15.2 breaths/min
    (truth 15) vs 8.0-14.6 for single-bin unwrap; waveform correlation 0.72-0.88 vs 0.67-0.83.
    Returns (displacement_mm, info).
    """
    rot = np.exp(-1j * np.angle(X[:, ref_bin]))[:, None] if reference else np.ones((len(X), 1))
    ks = np.arange(max(1, k - half_width), min(X.shape[1], k + half_width + 1))
    zs = X[:, ks] * rot
    centre = zs[:, list(ks).index(k)]
    align = np.exp(-1j * np.angle((zs * np.conj(centre)[:, None]).mean(0)))
    z = (zs * align).sum(1)
    c, rad = circle_center(z)
    radial_res = float(np.std(np.abs(z - c) - rad))
    span = float(np.ptp(np.unwrap(np.angle(z - c))))
    fit_ok = span >= 1.0 and radial_res < 0.25 * rad and rad < 5 * np.abs(z).max()
    zc = z - c if fit_ok else z
    a = np.abs(zc)
    dphi = np.angle(zc[1:] * np.conj(zc[:-1]))
    clamped = np.abs(dphi) > clamp_rad
    dphi = np.clip(dphi, -clamp_rad, clamp_rad)
    amp = np.sqrt(a[1:] * a[:-1]) / np.median(a)
    w = np.clip((amp - amp_gate[0]) / (amp_gate[1] - amp_gate[0]), 0, 1)
    phi = np.concatenate([[0.0], np.cumsum(w * dphi)])
    d = phi * LAMBDA_C / (4 * np.pi) * 1000
    return d, {"bins_combined_cm": [float(range_axis_m()[i] * 100) for i in ks],
               "summed_amplitude_vs_centre_bin_db": float(20 * np.log10(np.abs(z).mean() / np.abs(centre).mean())),
               "arc_centre_fit": bool(fit_ok), "arc_span_rad": span,
               "arc_radial_residual": radial_res / rad if rad > 0 else None,
               "clamped_steps": int(clamped.sum()), "clamped_fraction": float(clamped.mean()),
               "amplitude_gated_steps": int((w < 1).sum()),
               "max_phase_step_rad": float(np.abs(np.angle(zc[1:] * np.conj(zc[:-1]))).max())}
