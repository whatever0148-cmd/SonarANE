"""SonarANE Phase 4: pulsed TDM stereo-MIMO FMCW (left / right speaker) waveform, kernels and geometry.

Every 40 ms PRI holds a 16 ms up-chirp on the LEFT speaker (0-16 ms) and a 16 ms up-chirp on the RIGHT
speaker (20-36 ms), each followed by a 4 ms guard. Each speaker therefore runs at 40 % duty: its average
power (what a speaker thermal/RMS limiter integrates) is 4 dB below the Phase 1-3 continuous chirp at the
same peak amplitude. The two transmitters are separated in time, so in each speaker's 16 ms receive window
the other speaker contributes only tails whose beat frequencies fall outside the kept range bins.
Bandwidth is unchanged (18.8-22.3 kHz), so the FFT range resolution is still c / 2B = 4.9 cm; the slope is
2.5x steeper, so range-Doppler coupling drops from 0.235 s to 0.094 s.
"""
import numpy as np
from scipy.signal import get_window
from scipy.signal.windows import tukey

import fmcw

FS = fmcw.FS
F0, F1, B, FC = fmcw.F0, fmcw.F1, fmcw.B, fmcw.FC
PRI = 0.040                           # s per stereo pair (PRF 25 Hz per speaker)
NP = int(round(FS * PRI))             # 1920 samples per PRI
TC = 0.016                            # s per chirp
NC = int(round(FS * TC))              # 768 samples per chirp = conv C_in
SLOT = {"L": 0, "R": NP // 2}         # chirp start inside the PRI (0 ms, 20 ms)
SLOPE = B / TC                        # 218.75 kHz/s
RD_COUPLING_S = FC / SLOPE            # 0.094 s
DUTY = TC / PRI                       # 0.4 per speaker
K_BINS, BIN_STEP = 64, 0.5            # 2.45 cm grid, 0 .. 1.55 m
C_OUT = 2 * K_BINS
BASELINE_CM = 27.0                    # assumed left-right speaker spacing (user-supplied, not measured)


def chirp_phase(n, slope=SLOPE, f0=F0):
    t = np.asarray(n, dtype=np.float64) / FS
    return 2 * np.pi * (f0 * t + 0.5 * slope * t * t)


def make_chirp(alpha=0.15):
    n = np.arange(NC)
    return (np.cos(chirp_phase(n)) * tukey(NC, alpha)).astype(np.float32)


def make_tx(n_pri, amp=0.9, alpha=0.15, lead_s=0.1, tail_s=1.0):
    """Stereo [samples, 2] TDM train: left chirp in slot L, right chirp in slot R of every PRI."""
    lead, tail = int(lead_s * FS), int(tail_s * FS)
    tx = np.zeros((lead + n_pri * NP + tail, 2), np.float32)
    c = make_chirp(alpha) * amp
    for ch, col in (("L", 0), ("R", 1)):
        for m in range(n_pri):
            s = lead + m * NP + SLOT[ch]
            tx[s:s + NC, col] = c
    return tx, lead


def range_axis_cm():
    return np.arange(K_BINS) * BIN_STEP * fmcw.RANGE_RES_M * 100


def dechirp_dft_kernel(window="hann"):
    """[2K, NC] fused dechirp + Hann + zero-padded range DFT for one 16 ms chirp (as fmcw, shorter)."""
    n = np.arange(NC)
    h = get_window(window, NC, fftbins=True)
    k = (np.arange(K_BINS) * BIN_STEP)[:, None]
    m = h * np.exp(1j * chirp_phase(n)) * np.exp(-2j * np.pi * k * n / NC)
    return np.concatenate([m.real, m.imag], axis=0).astype(np.float32)


def amp_scale(window="hann"):
    return 2.0 / float(get_window(window, NC, fftbins=True).sum())


def fft_pipeline(frames, window="hann"):
    """NumPy reference: [M, NC] frames -> complex [M, K]."""
    n = np.arange(NC)
    ref = get_window(window, NC, fftbins=True) * np.exp(1j * chirp_phase(n))
    return np.fft.fft(frames * ref, n=int(round(NC / BIN_STEP)), axis=-1)[:, :K_BINS] * amp_scale(window)


def to_complex(out):
    out = np.asarray(out, dtype=np.float32).reshape(C_OUT, -1)
    return (out[:K_BINS] + 1j * out[K_BINS:]).T


# ---------------------------------------------------------------- geometry
def mic_offset_cm(direct_delay_diff_samples):
    """Mic x-position on the speaker line from (arrival_R - arrival_L) of the direct paths, minus the slot gap.

    With speakers at -/+ BASELINE/2 and the mic between them on the same line, D_L - D_R = 2 x_mic, and the
    right direct path arrives (D_R - D_L)/c later than the slot offset alone predicts.
    """
    d_lr_cm = -direct_delay_diff_samples / FS * fmcw.C_SOUND * 100       # D_L - D_R
    return float(np.clip(d_lr_cm / 2, -BASELINE_CM / 2 + 0.5, BASELINE_CM / 2 - 0.5))


def solve_xy(r_l_cm, r_r_cm, x_mic_cm, grid=None, max_resid_cm=3.0):
    """(X, Y) in cm from the two bistatic ranges (each beyond its own direct path).

    Speaker s at (-/+ BASELINE/2, 0), mic at (x_mic, 0); bistatic path |T - S| + |T - M| = D_s + 2 R_s.
    Two ellipses with a shared focus; the Y > 0 intersection is found by least squares on a 0.5 cm grid.
    The model is planar: the hand's height above the speaker/mic line is folded into Y.
    Pairs whose two ellipses have no common point within max_resid_cm of path length (the two channels
    locked onto different scatterers) are rejected (nan).
    """
    xs = np.arange(-45, 45.01, 0.5) if grid is None else grid[0]
    ys = np.arange(3, 90.01, 0.5) if grid is None else grid[1]
    gx, gy = np.meshgrid(xs, ys)
    sl, sr, mx = -BASELINE_CM / 2, BASELINE_CM / 2, x_mic_cm
    dm = np.hypot(gx - mx, gy)
    pl = np.hypot(gx - sl, gy) + dm - abs(mx - sl)
    pr = np.hypot(gx - sr, gy) + dm - abs(mx - sr)
    out = np.full((len(r_l_cm), 2), np.nan)
    for i, (rl, rr) in enumerate(zip(r_l_cm, r_r_cm)):
        if np.isfinite(rl) and np.isfinite(rr):
            e = (pl - 2 * rl) ** 2 + (pr - 2 * rr) ** 2
            j = np.unravel_index(np.argmin(e), e.shape)
            if np.sqrt(e[j] / 2) <= max_resid_cm and 0 < j[0] < len(ys) - 1 and 0 < j[1] < len(xs) - 1:
                out[i] = gx[j], gy[j]
    return out


# ---------------------------------------------------------------- NE range conv for TDM chirps
def range_model(s=64):
    """Compiled FP16 1x1-conv range model for [1, NC, 1, s] TDM frames (built / rebuilt from its kernel tag)."""
    import hashlib
    import shutil
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    import ane_radar_benchmark as bench
    w = dechirp_dft_kernel()
    tag = f"{F0:.0f}-{F1:.0f}Hz-NC{NC}-" + hashlib.sha1(w.astype(np.float16).tobytes()).hexdigest()[:12]
    pkg = bench.MODEL_DIR / f"sonar_tdm_range_S{s}.mlpackage"
    compiled = pkg.with_suffix(".mlmodelc")
    old = (ct.models.MLModel(str(pkg), skip_model_load=True).user_defined_metadata.get("kernel_tag")
           if pkg.exists() else None)
    if not compiled.exists() or old != tag:
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, NC, 1, s), dtype=types.fp16)],
                    opset_version=ct.target.macOS15)
        def prog(x):
            return mb.conv(x=x, weight=w.astype(np.float16)[:, :, None, None], pad_type="valid", name="range_fft")
        m = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                       compute_precision=ct.precision.FLOAT16)
        m.user_defined_metadata["kernel_tag"] = tag
        if pkg.exists():
            shutil.rmtree(pkg)
        m.save(str(pkg))
        compiled = bench.compile_once(pkg)
    plan = bench.compute_plan(compiled, ct.ComputeUnit.CPU_AND_NE)
    device = next(o["preferred"] for o in plan if o["op"].endswith("conv"))
    return ct.models.CompiledMLModel(str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE), device


def ne_range(model, frames, s=64):
    """[M, NC] frames -> complex [M, K] via the NE model; per-chirp FP16 normalisation undone per column."""
    g = 1.0 / np.abs(frames).max(axis=1, keepdims=True)
    x = (frames * g).astype(np.float16)
    m = len(x)
    xb = np.concatenate([x, np.zeros((-m % s, NC), np.float16)])
    out = [model.predict({"x": np.ascontiguousarray(xb[i:i + s].T[None, :, None, :])})["range_fft"].reshape(C_OUT, s)
           for i in range(0, len(xb), s)]
    return to_complex(np.concatenate(out, axis=1)[:, :m]) * amp_scale() / g


# ---------------------------------------------------------------- per-channel dual-path tracking
def direct_path_basis():
    n = np.arange(NC)
    u = [fft_pipeline(np.cos(chirp_phase(n - d))[None])[0] for d in (-0.5, 0.0, 0.5)]
    return np.stack([u[1], u[2] - u[0], u[2] - 2 * u[1] + u[0]])


def presence_residual(X, trim=0.3, iters=3, fit_cm=10.0):
    """fmcw.presence_residual for TDM spectra: trimmed-mean background + per-chirp direct-path drift lock."""
    fit = range_axis_cm() < fit_cm
    u = direct_path_basis()

    def background(b0):
        coef, *_ = np.linalg.lstsq(u[:, fit].T, (X[:, fit] - b0[fit]).T, rcond=None)
        return b0 + coef.T @ u

    b0 = X.mean(0)
    for _ in range(iters):
        dev = np.abs(X - background(b0))
        keep = dev <= np.quantile(dev, 1 - trim, axis=0)
        b0 = b0 + ((X - background(b0)) * keep).sum(0) / keep.sum(0)
    return np.abs(X - background(b0))


def track_channel(X, gate_cm=(10.0, 90.0), presence_min_cm=15.0):
    """Dual-path (MTI motion + presence) scores, TGC CFAR candidates, Kalman track for one speaker channel."""
    r = range_axis_cm()
    y = X[2:] - 2 * X[1:-1] + X[:-2]
    a = fmcw.os_cfar(fmcw.smooth_mag(np.abs(y)))
    b = fmcw.os_cfar(fmcw.smooth_mag(presence_residual(X)[1:-1]))
    b[:, r < presence_min_cm] = 0.0
    score = np.maximum(a, b)
    cands = fmcw.candidates(score, r, gate_cm, 10 ** (fmcw.tgc_threshold_db(r) / 20))
    rr, vv, st = fmcw.kalman_track(cands, kappa=RD_COUPLING_S, t_chirp=PRI)
    return rr, vv, st, score


# ---------------------------------------------------------------- Phase 4b: joint stereo 2D tracking
GRID_X = np.arange(-40.0, 40.0 + 1e-9, 1.0)
GRID_Y_FULL = np.arange(8.0, 70.0 + 1e-9, 1.0)
# Personal workspace bubble (library: people and desk dividers move at 65-90 cm). Interactive hand
# tracking only looks at y <= 55 cm, and per-channel evidence beyond a bistatic range of 58 cm is ignored.
WORKSPACE_Y_MAX, WORKSPACE_R_MAX = 55.0, 58.0
GRID_Y = np.arange(8.0, WORKSPACE_Y_MAX + 1e-9, 1.0)


def gain_normalise(Xs, win=25, max_rel_db=10.0):
    """Divide each channel by a 1 s rolling median of its direct-path amplitude (speaker gain sag cancels
    in every echo), but only for channels whose direct path is within max_rel_db of the strongest one:
    a weak direct path (here the right speaker, 19 dB below the left at the mic, swinging 3.5 dB) is
    dominated by interference, and dividing by it would inject that into every echo.
    Returns (normalised spectra dict, {channel: applied?})."""
    from scipy.ndimage import median_filter
    lvl = {ch: float(np.median(np.abs(X[:, 0]))) for ch, X in Xs.items()}
    top = max(lvl.values())
    out, applied = {}, {}
    for ch, X in Xs.items():
        ok = 20 * np.log10(lvl[ch] / top) > -max_rel_db
        g = median_filter(np.abs(X[:, 0]), size=win, mode="nearest") / lvl[ch] if ok else np.ones(len(X))
        out[ch], applied[ch] = X / g[:, None], bool(ok)
    return out, applied


def channel_scores(X, presence_min_cm=15.0):
    """max(MTI, presence) CFAR ratio per MTI row [M-2, K] (as in track_channel)."""
    r = range_axis_cm()
    y = X[2:] - 2 * X[1:-1] + X[:-2]
    a = fmcw.os_cfar(fmcw.smooth_mag(np.abs(y)))
    b = fmcw.os_cfar(fmcw.smooth_mag(presence_residual(X)[1:-1]))
    b[:, r < presence_min_cm] = 0.0
    return np.maximum(a, b)


def bistatic_grid(x_mic, ys=None):
    """Bistatic range beyond each speaker's direct path on the (GRID_X, ys) grid (default: workspace)."""
    gx, gy = np.meshgrid(GRID_X, GRID_Y if ys is None else ys)
    out = {}
    for ch, xs in (("L", -BASELINE_CM / 2), ("R", BASELINE_CM / 2)):
        out[ch] = 0.5 * (np.hypot(gx - xs, gy) + np.hypot(gx - x_mic, gy) - abs(xs - x_mic))
    return gx, gy, out


def bistatic(x, y, x_mic):
    """R_L, R_R and their gradients d/dx, d/dy at one point."""
    r, g = {}, {}
    for ch, xs in (("L", -BASELINE_CM / 2), ("R", BASELINE_CM / 2)):
        ds, dm = np.hypot(x - xs, y), np.hypot(x - x_mic, y)
        r[ch] = 0.5 * (ds + dm - abs(xs - x_mic))
        g[ch] = 0.5 * np.array([(x - xs) / ds + (x - x_mic) / dm, y / ds + y / dm])
    return r, g


def backprojection(scores, x_mic, ys=None, r_max=WORKSPACE_R_MAX):
    """Joint map per MTI row: geometric mean sqrt(Z_L(R_L(x,y)) * Z_R(R_R(x,y))) on the Cartesian grid.
    A scatterer elevated in one channel only (an 80 cm ghost on the right while the left sees 20 cm) has
    no partner at the geometrically consistent range, so the product stays low. Also returns the matching
    geometric-mean TGC threshold map."""
    gx, gy, rr = bistatic_grid(x_mic, ys)
    r = range_axis_cm()
    keep = r <= (np.inf if r_max is None else r_max)                # workspace: no evidence beyond r_max
    zl_all, zr_all = scores["L"] * keep, scores["R"] * keep
    maps = np.sqrt(np.stack([np.interp(rr["L"].ravel(), r, zl) * np.interp(rr["R"].ravel(), r, zr)
                             for zl, zr in zip(zl_all, zr_all)]).reshape(-1, *gx.shape))
    thr = 10 ** ((fmcw.tgc_threshold_db(rr["L"]) + fmcw.tgc_threshold_db(rr["R"])) / 40)
    return maps, thr, gx, gy


def map_candidates(m, thr, gx, gy, max_n=5):
    """Local maxima of one backprojection map above threshold -> [(x, y, value)], strongest first."""
    from scipy.ndimage import maximum_filter
    pk = (m == maximum_filter(m, size=5)) & (m > thr)
    iy, ix = np.nonzero(pk)
    order = np.argsort(m[iy, ix])[::-1][:max_n]
    return [(float(gx[iy[i], ix[i]]), float(gy[iy[i], ix[i]]), float(m[iy[i], ix[i]])) for i in order]


class JointEKF:
    """Streaming form of joint_track (same equations, one row per step) for the live app.

    step(t, cands) -> (status, state, events): status 2 update / 1 coast / 0 none for row t, state
    [x, y, vx, vy] or None, and retroactive events for batch callers: ("fill", [(row, x, y, vx, vy)...])
    when a multi-hypothesis chain confirms (its rows become track rows), ("clear", lo, hi) when a coasted
    tail is dropped (those rows were never a track).
    """

    def __init__(self, x_mic, dt=PRI, sigma_a=600.0, sigma_r=1.5, gate_chi2=13.8, max_coast=15, init_run=4,
                 init_step_cm=8.0, coast_decay=0.9, ys=None, init_y=None, init_min_speed=None,
                 init_min_disp=6.0, static_xy=None, hold_y_max=None):
        self.x_mic, self.dt, self.gate_chi2, self.max_coast = x_mic, dt, gate_chi2, max_coast
        self.init_run, self.init_step_cm, self.coast_decay = init_run, init_step_cm, coast_decay
        self.init_y, self.init_min_speed, self.init_min_disp = init_y, init_min_speed, init_min_disp
        self.static_xy, self.hold_y_max = static_xy, hold_y_max
        self.gy_lo, self.gy_hi = (GRID_Y if ys is None else ys)[[0, -1]]
        self.f = np.eye(4)
        self.f[0, 2] = self.f[1, 3] = dt
        q1 = sigma_a ** 2 * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
        self.q = np.zeros((4, 4))
        self.q[np.ix_([0, 2], [0, 2])] = q1
        self.q[np.ix_([1, 3], [1, 3])] = q1
        self.rm = np.eye(2) * sigma_r ** 2
        self.s = self.p = None
        self.misses, self.chain = 0, []

    def _h(self, state):
        k = RD_COUPLING_S
        r, g = bistatic(state[0], state[1], self.x_mic)
        hz = np.array([r["L"] + k * g["L"] @ state[2:], r["R"] + k * g["R"] @ state[2:]])
        hj = np.array([[*g["L"], *(k * g["L"])], [*g["R"], *(k * g["R"])]])
        return hz, hj

    def _initiate(self, t, cs):
        # multi-hypothesis initiation (as fmcw.kalman_track): every candidate extends the longest chain
        # that ended on the previous row within init_step_cm; the strongest confirmed chain wins.
        dt = self.dt
        ext = []
        for cx, cy, val in cs:
            prev = [c for c in self.chain if c[-1][0] == t - 1 and np.hypot(cx - c[-1][1], cy - c[-1][2]) <= self.init_step_cm]
            ext.append((max(prev, key=len) if prev else []) + [(t, cx, cy, val)])
        self.chain = ext
        done = []
        for cc in [c for c in self.chain if len(c) >= self.init_run]:
            arr = np.array(cc[-self.init_run:])[:, :3]
            cvx = np.polyfit(arr[:, 0] * dt, arr[:, 1], 1)[0]
            cvy = np.polyfit(arr[:, 0] * dt, arr[:, 2], 1)[0]
            xl, yl = arr[-1, 1], arr[-1, 2]
            if self.init_y is not None and not (self.init_y[0] <= yl <= self.init_y[1]):
                continue
            if self.init_min_speed is not None:
                moving = np.hypot(cvx, cvy) >= self.init_min_speed
                away = bool(self.static_xy) and min(np.hypot(xl - sx, yl - sy) for sx, sy in self.static_xy) >= self.init_min_disp
                if not (moving or away):
                    continue
            done.append(cc)
        if not done:
            return 0, None, []
        c = np.array(max(done, key=lambda cc: sum(v for *_, v in cc[-self.init_run:]))[-self.init_run:])[:, :3]
        vx = np.polyfit(c[:, 0] * dt, c[:, 1], 1)[0]
        vy = np.polyfit(c[:, 0] * dt, c[:, 2], 1)[0]
        self.s = np.array([c[-1, 1], c[-1, 2], vx, vy])
        self.p = np.diag([4.0, 4.0, 30.0 ** 2, 30.0 ** 2])
        self.chain, self.misses = [], 0
        return 2, self.s.copy(), [("fill", [(int(tt), xx, yy, vx, vy) for tt, xx, yy in c])]

    def step(self, t, cs):
        if self.s is None:
            return self._initiate(t, cs)
        s = self.f @ self.s
        p = self.f @ self.p @ self.f.T + self.q
        hz, hj = self._h(s)
        si = np.linalg.inv(hj @ p @ hj.T + self.rm)
        best, best_v = None, -np.inf
        for cx, cy, val in cs:
            r, _ = bistatic(cx, cy, self.x_mic)
            nu = np.array([r["L"], r["R"]]) - hz
            if nu @ si @ nu <= self.gate_chi2 and val > best_v:
                best, best_v = nu, val
        if best is not None:
            kg = p @ hj.T @ si
            s = s + kg @ best
            p = (np.eye(4) - kg @ hj) @ p
            self.misses, st = 0, 2
        else:
            self.misses += 1
            s[2:] *= self.coast_decay
            if self.misses > self.max_coast:
                self.s = None
                return 0, None, [("clear", t - self.max_coast, t)]
            st = 1
        if self.hold_y_max is not None and s[1] > self.hold_y_max + 2:            # left the hand zone: drop
            lo = t - self.misses + 1 if st == 1 else t
            self.s = None
            return 0, None, [("clear", lo, t + 1)]
        if not (GRID_X[0] - 5 <= s[0] <= GRID_X[-1] + 5 and self.gy_lo - 5 <= s[1] <= self.gy_hi + 5):
            if st == 1:                                                           # coasted off the grid: drop
                lo = t - self.misses + 1
                self.s = None
                return 0, None, [("clear", lo, t + 1)]
            s[0] = np.clip(s[0], GRID_X[0], GRID_X[-1])
            s[1] = np.clip(s[1], self.gy_lo, self.gy_hi)
        self.s, self.p = s, p
        return st, s.copy(), []


def joint_track(cands, x_mic, **kw):
    """4-state Cartesian EKF [x, y, vx, vy] observing both bistatic ranges jointly (batch form of JointEKF).

    Measurement z = (R_L, R_R) of the chosen map peak; model h = R_s(x, y) + kappa * grad R_s . v (the
    up-chirp range-Doppler coupling of each channel). Both ranges come from one state, so the right
    channel cannot sit at 80 cm while the left sits at 20 cm. Gate: Mahalanobis chi2 (2 dof, 99.9 %);
    among gated peaks the strongest wins. Initiation is multi-hypothesis: any map peak may extend any
    chain that moved <= init_step_cm since the previous row; init_run rows confirm (strongest chain wins).
    Coasting decays velocity; > max_coast misses drops the track and erases the coast.
    Kinematic hand gate (optional): a confirmed chain only starts a track if its last point lies in init_y
    and it moves deliberately (fitted speed >= init_min_speed) or sits >= init_min_disp from every
    pre-roll static reflector (static_xy); a started track is dropped once it leaves y <= hold_y_max.
    Returns x, y, vx, vy, status (2 update, 1 coast, 0 none) per row.
    """
    ekf = JointEKF(x_mic, **kw)
    n = len(cands)
    out = np.full((n, 4), np.nan)
    status = np.zeros(n, int)
    for t, cs in enumerate(cands):
        st, s, events = ekf.step(t, cs)
        for ev in events:
            if ev[0] == "fill":
                for tt, xx, yy, vx, vy in ev[1]:
                    out[tt], status[tt] = (xx, yy, vx, vy), 2
            else:
                status[ev[1]:ev[2]], out[ev[1]:ev[2]] = 0, np.nan
        if st:
            out[t], status[t] = s, st
    if ekf.s is not None and ekf.misses:
        status[n - ekf.misses:], out[n - ekf.misses:] = 0, np.nan
    return out[:, 0], out[:, 1], out[:, 2], out[:, 3], status


# ---------------------------------------------------------------- Phase 4d: pre-roll calibration + hand/torso split
PREROLL_S = 2.0
HAND_INIT_Y = (8.0, 45.0)        # a new hand track may only start here ...
HAND_HOLD_Y_MAX = 52.0           # ... and, once started, may hold (down to 0 cm/s) up to here
INIT_MIN_SPEED = 8.0             # cm/s: deliberate motion
INIT_MIN_DISP = 6.0              # cm from every pre-roll static reflector
PRESENCE_Y = (25.0, 55.0)        # seated-torso band: the user's torso sat at y = 34-35 cm in the 2026-10-21
                                 # take-2 capture (leaning in over the notebook), not at 40-58
TRACK_Y_MAX = 58.0               # joint tracker / map grid top (hand zone + torso band)
MICRO_RMS_MM = (0.3, 5.0)        # torso sway + breathing at a static reflector; laptop edge / desk < 0.2 mm
MICRO_BREATH_FRAC = 0.5          # share of micro-displacement power inside the respiration band
MICRO_MIN_ROWS = 50              # >= 2 s of hand-free rows before a micro-motion verdict
PRESENCE_MAX_EXCURSION = 5.0     # cm: micro-motion, not a gesture
FLOOR_MIN_REL = 0.5              # floor bound: >= this x the workspace-median floor
MIN_GESTURE_S = 0.5              # a deliberate 50 cm sweep at 20-60 cm/s takes >= 0.8 s. NOTE: this value
                                 # also happens to remove the one 0.36 s track left in the 2026-10-21
                                 # library baseline, so that baseline is not independent evidence for it.


def presence_fixed(X, b0, fit_cm=10.0):
    """|X - background| with a FIXED background b0 (from the pre-roll) plus the per-chirp direct-path
    drift lock (gain/phase + sub-sample delay basis)."""
    fit = range_axis_cm() < fit_cm
    u = direct_path_basis()
    coef, *_ = np.linalg.lstsq(u[:, fit].T, (X[:, fit] - b0[fit]).T, rcond=None)
    return np.abs(X - (b0 + coef.T @ u))


def calibration_rows(X_pair, t, preroll_s=None, win_s=PREROLL_S):
    """Rows used to calibrate: the hands-in-lap pre-roll if the capture has one, else (legacy captures)
    the quietest win_s window by workspace MTI energy of both channels. Returns (rows, description)."""
    if preroll_s:
        rows = np.flatnonzero((t >= 0.2) & (t < preroll_s - 0.1))
        return rows, f"pre-roll {t[rows[0]]:.2f}-{t[rows[-1]] + PRI:.2f} s (hands in lap)"
    r = range_axis_cm()
    sel = (r >= 15) & (r <= WORKSPACE_R_MAX)
    e = sum(np.concatenate([[0], (np.abs(X[2:] - 2 * X[1:-1] + X[:-2])[:, sel] ** 2).mean(1), [0]])
            for X in X_pair.values())
    n = int(round(win_s / PRI))
    c = np.convolve(e, np.ones(n) / n, "valid")
    i = int(np.argmin(c[1:-1])) + 1
    rows = np.arange(i, i + n)
    return rows, f"quietest {win_s:.0f} s window {t[rows[0]]:.2f}-{t[rows[-1]] + PRI:.2f} s (legacy capture, no pre-roll)"


def calibrate(X, rows, pct=50.0, min_rel=FLOOR_MIN_REL):
    """Static background b0 and locked per-bin floors (MTI and presence paths) from the calibration rows.
    A floor bound (min_rel x the 15-58 cm median) keeps ultra-quiet bins from turning millimetre torso
    sway into 15 dB spikes."""
    r = range_axis_cm()
    ws = (r >= 15) & (r <= WORKSPACE_R_MAX)
    xc = X[rows]
    b0 = xc.mean(0)
    mti = fmcw.smooth_mag(np.abs(xc[2:] - 2 * xc[1:-1] + xc[:-2]))
    pres = fmcw.smooth_mag(presence_fixed(xc, b0))
    out = {"b0": b0}
    for name, m in (("sigma_mti", mti), ("sigma_pres", pres)):
        s = np.percentile(m, pct, axis=0)
        out[name] = np.maximum(s, min_rel * np.median(s[ws]))
        out[name + "_bounded_bins"] = int((s < min_rel * np.median(s[ws]))[ws].sum())
    return out


def channel_scores_cal(X, cal, presence_min_cm=15.0):
    """max(MTI, presence) ratios over the LOCKED pre-roll floors, per MTI row [M-2, K]."""
    r = range_axis_cm()
    a = fmcw.smooth_mag(np.abs(X[2:] - 2 * X[1:-1] + X[:-2])) / cal["sigma_mti"]
    b = fmcw.smooth_mag(presence_fixed(X, cal["b0"])[1:-1]) / cal["sigma_pres"]
    b[:, r < presence_min_cm] = 0.0
    return np.maximum(a, b)


def static_reflectors(cals, x_mic, ys=None, n=3, r_min=10.0):
    """Strongest static reflectors of the pre-roll scene in (x, y): geometric mean of both channels'
    |b0| backprojected (torso, laptop edge, desk...)."""
    gx, gy, rr = bistatic_grid(x_mic, ys)
    r = range_axis_cm()
    m = np.sqrt(np.interp(rr["L"], r, np.abs(cals["L"]["b0"]) * (r >= r_min)) *
                np.interp(rr["R"], r, np.abs(cals["R"]["b0"]) * (r >= r_min)))
    inner = [(x, y) for x, y, _ in map_candidates(m, np.median(m) * 2, gx, gy, max_n=3 * n)
             if gx[0, 0] < x < gx[0, -1] and gy[0, 0] < y < gy[-1, 0]]   # grid-edge peaks are cut-off ridges
    return inner[:n]


def micro_motion(X, k, rows, dt=PRI):
    """Phase-referenced micro-displacement of range bin k over `rows` (3-bin coherent, clamped steps):
    rms / p-p mm and the share of its power in the respiration band. A seated torso gives ~1 mm rms with
    most of the power at 0.12-0.6 Hz; a laptop edge or desk gives < 0.2 mm."""
    from scipy.signal import detrend, welch
    d, _ = fmcw.breathing_displacement_mm(X[rows], k)
    d = detrend(d)
    f, pw = welch(d, 1 / dt, nperseg=min(len(d), 128))
    band = (f >= fmcw.RESP_BAND_HZ[0]) & (f <= fmcw.RESP_BAND_HZ[1])
    frac = float(pw[band].sum() / max(pw[f > 0.05].sum(), 1e-30))
    return {"rms_mm": round(float(d.std()), 2), "pp_mm": round(float(np.ptp(d)), 2),
            "breath_band_fraction": round(frac, 2), "rows": int(len(rows)),
            "torso_like": bool(MICRO_RMS_MM[0] <= d.std() <= MICRO_RMS_MM[1] and frac >= MICRO_BREATH_FRAC)}


def torso_evidence(X, statics, x_mic, jx, jy, clear_cm=12.0):
    """Micro-Doppler check of every pre-roll static reflector in the torso band: both channels' range
    bins at its bistatic ranges, over rows where the joint track is >= clear_cm away (or absent).
    Confirmed when either channel is torso-like (MICRO_RMS_MM, MICRO_BREATH_FRAC)."""
    r = range_axis_cm()
    m = len(X["L"])
    out = []
    for sx, sy in statics:
        if not (PRESENCE_Y[0] <= sy <= PRESENCE_Y[1]):
            continue
        rr, _ = bistatic(sx, sy, x_mic)
        far = np.ones(m, bool)
        far[1:m - 1] = np.hypot(np.nan_to_num(jx, nan=1e3) - sx, np.nan_to_num(jy, nan=1e3) - sy) > clear_cm
        rows = np.flatnonzero(far)
        e = {"xy_cm": [round(sx, 1), round(sy, 1)], "confirmed": False}
        if len(rows) >= MICRO_MIN_ROWS:
            for ch in X:
                e[ch] = micro_motion(X[ch], int(np.argmin(np.abs(r - rr[ch]))), rows)
            e["confirmed"] = any(e[ch]["torso_like"] for ch in X)
        out.append(e)
    return out


def track_segments(x, y, status, dt=PRI):
    """[(start_row, end_row, duration_s, x_excursion, y_excursion, mean_x, mean_y)] of a status trace."""
    on = (status > 0).astype(int)
    d = np.diff(np.concatenate([[0], on, [0]]))
    out = []
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        out.append((int(a), int(b), float((b - a) * dt), float(np.ptp(x[a:b])), float(np.ptp(y[a:b])),
                    float(np.mean(x[a:b])), float(np.mean(y[a:b]))))
    return out


def hand_presence_pipeline(X_pair, x_mic, preroll_s=None):
    """Full Phase 4d analysis on per-channel TDM spectra {'L': [M, K], 'R': [M, K]}.

    gain normalisation -> calibration rows (pre-roll, or quietest 2 s for legacy captures) -> locked floors
    and static background -> scores -> ONE joint EKF tracker on y 8..58 cm, then every track segment is
    classified by its whole-segment kinematics (a 4-frame speed estimate is dominated by peak jitter):
      hand             starts at y <= HAND_INIT_Y[1], moves >= INIT_MIN_DISP and lasts >= MIN_GESTURE_S
                       (rows beyond HAND_HOLD_Y_MAX are not counted as hand rows)
      seated presence  excursion < PRESENCE_MAX_EXCURSION and mean y in PRESENCE_Y
      other motion     anything else (started outside the hand zone, too short, or too small)
    A pre-roll static reflector in the torso band counts as seated presence only when its range bins show
    torso-like micro-motion (torso_evidence): seated presence while sitting still, breathing.
    Rows are MTI rows (row m centred on chirp m + 1).
    """
    m = len(X_pair["L"])
    t = np.arange(m) * PRI
    X, gain_applied = gain_normalise(X_pair)
    rows_cal, cal_desc = calibration_rows(X, t, preroll_s)
    cals = {ch: calibrate(X[ch], rows_cal) for ch in X}
    scores = {ch: channel_scores_cal(X[ch], cals[ch]) for ch in X}
    ys = np.arange(8.0, TRACK_Y_MAX + 1e-9, 1.0)
    statics = static_reflectors(cals, x_mic, ys=ys, n=6)
    maps, thr, gx, gy = backprojection(scores, x_mic, ys=ys, r_max=62.0)
    cands = [map_candidates(mp, thr, gx, gy) for mp in maps]
    jx, jy, jvx, jvy, jst = joint_track(cands, x_mic, ys=ys)
    rows_t = (np.arange(m - 2) + 1) * PRI
    after = rows_t >= (preroll_s or 0.0)
    hand_rows = np.zeros(len(jst), bool)
    kinds = {"hand": [], "seated_presence": [], "other_motion": []}
    for sg in track_segments(jx, jy, jst):
        a, b = sg[0], sg[1]
        if not after[a:b].any():
            continue
        exc = max(sg[3], sg[4])
        speed = float(np.nanmedian(np.hypot(jvx[a:b], jvy[a:b])))
        starts_low = jy[a] <= HAND_INIT_Y[1]
        info = {"t_s": [round(a * PRI + PRI, 2), round(b * PRI + PRI, 2)], "duration_s": round(sg[2], 2),
                "excursion_cm": [round(sg[3], 1), round(sg[4], 1)], "mean_xy_cm": [round(sg[5], 1), round(sg[6], 1)],
                "start_y_cm": round(float(jy[a]), 1), "median_speed_cm_s": round(speed, 1)}
        if starts_low and exc >= INIT_MIN_DISP and sg[2] >= MIN_GESTURE_S:
            kinds["hand"].append(info)
            hand_rows[a:b] = jy[a:b] <= HAND_HOLD_Y_MAX
        elif exc < PRESENCE_MAX_EXCURSION and PRESENCE_Y[0] <= sg[6] <= PRESENCE_Y[1]:
            kinds["seated_presence"].append(info)
        else:
            kinds["other_motion"].append(info)
    torso = torso_evidence(X, statics, x_mic, jx, jy)
    torso_static = [tuple(e["xy_cm"]) for e in torso if e["confirmed"]]
    hx = np.where(hand_rows, jx, np.nan)
    hy = np.where(hand_rows, jy, np.nan)
    hst = np.where(hand_rows, jst, 0)
    return {"rows_t": rows_t, "after": after, "hand": {"x": hx, "y": hy, "status": hst},
            "all_tracks": {"x": jx, "y": jy, "status": jst}, "map_mean": maps[after].mean(0) if after.any() else maps.mean(0),
            "maps": maps, "cals": cals, "torso_evidence": torso,
            "gx": gx, "gy": gy, "statics": statics,
            "summary": {
                "calibration": cal_desc, "calibration_rows": int(len(rows_cal)),
                "floor_bounded_bins": {ch: {k: cals[ch][k] for k in ("sigma_mti_bounded_bins", "sigma_pres_bounded_bins")}
                                       for ch in cals},
                "static_reflectors_xy_cm": [[round(a_, 1), round(b_, 1)] for a_, b_ in statics],
                "classifier": {"hand": f"start y <= {HAND_INIT_Y[1]:.0f} cm, whole-segment excursion >= {INIT_MIN_DISP:.0f} cm, "
                                       f"duration >= {MIN_GESTURE_S} s; "
                                       f"hand rows only while y <= {HAND_HOLD_Y_MAX:.0f} cm",
                               "seated_presence": f"excursion < {PRESENCE_MAX_EXCURSION:.0f} cm, mean y {PRESENCE_Y[0]:.0f}-{PRESENCE_Y[1]:.0f} cm, "
                                                  "or a torso-band pre-roll static reflector whose range bins show "
                                                  f"{MICRO_RMS_MM[0]}-{MICRO_RMS_MM[1]} mm rms micro-motion with >= "
                                                  f"{MICRO_BREATH_FRAC:.0%} of its power in the respiration band",
                               "floors": f"locked from the calibration rows (median), bounded at {FLOOR_MIN_REL} x workspace median"},
                "hand_tracks": len(kinds["hand"]),
                "hand_rows": int(hand_rows[after].sum()), "hand_rows_updated": int(((hst == 2) & after).sum()),
                "rows_after_preroll": int(after.sum()),
                "hand_coverage": float(hand_rows[after].mean()) if after.any() else 0.0,
                "hand_segments": kinds["hand"],
                "hand_x_cm": [float(np.nanmin(hx[after])), float(np.nanmax(hx[after]))] if hand_rows[after].any() else None,
                "hand_y_cm": [float(np.nanmin(hy[after])), float(np.nanmax(hy[after]))] if hand_rows[after].any() else None,
                "seated_presence_detected": bool(kinds["seated_presence"] or torso_static),
                "seated_presence_s": round(float(sum(sg["duration_s"] for sg in kinds["seated_presence"])), 2),
                "seated_presence_segments": kinds["seated_presence"],
                "seated_presence_static_xy_cm": [list(xy) for xy in torso_static],
                "seated_presence_micro_motion": torso,
                "other_motion_segments": kinds["other_motion"],
                "gain_normalised": gain_applied}}
