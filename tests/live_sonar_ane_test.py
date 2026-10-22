"""SonarANE Phase 3 live test: 8 s capture -> NE range-FFT -> TGC dual-path Kalman tracker + respiration.

Plays 200 Tukey-windowed chirps (band from src/fmcw.py, T = 40 ms) on the MacBook Air speakers while
recording the built-in microphone. Protocol: 0-4 s sweep a book/palm 15 -> 55 -> 25 cm, 4-8 s hold still at
25-35 cm and breathe slowly (a spoken "hold still and breathe" cue is mixed into the transmit buffer at 4 s,
low-passed far below the sweep band). Raw and [1,-2,1] MTI frames go through the FP16 range conv on the NE;
the dual-path Kalman tracker uses the range-adaptive (TGC) CFAR; the hold window feeds the direct-path
phase-referenced respiration interferometer; the fused NE graph builds a sliding Range-Doppler cube.

Run:  ~/SonarANE/.venv/bin/python ~/SonarANE/tests/live_sonar_ane_test.py              # live, plays sound
      ~/SonarANE/.venv/bin/python ~/SonarANE/tests/live_sonar_ane_test.py --from-wav phase2
          re-analyse artifacts/phase2_capture.wav silently -> artifacts/phase3_replay_phase2_*
"""
import datetime as dt
import json
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import sounddevice as sd
import coremltools as ct
from scipy.io import wavfile
from scipy.signal import butter, correlate, detrend, find_peaks, hilbert, sosfiltfilt, welch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ane_kernels"))
import fmcw  # noqa: E402
import ane_radar_benchmark as bench  # noqa: E402
import ane_fused_2d_radar as fused  # noqa: E402

ART = ROOT / "artifacts"
PHASE = "phase3"


def out_paths(tag):
    return ART / f"{tag}_ane_sonar_proof.png", ART / f"{tag}_telemetry.json", ART / f"{tag}_capture.wav"


PNG, TELEMETRY, WAV = out_paths(PHASE)

MIC_NAME, SPK_NAME = "MacBook Air Microphone", "MacBook Air Speakers"
TEST_VOLUME = 75
BANK = 64                         # NE-scheduled Range-Doppler bank (see MEMORY.md: S=128 lands on CPU)
FIRST_CONFIG = {"tukey_alpha": 0.15, "amp": 0.90}
MAX_ATTEMPTS = 3                  # later configs come from next_config() based on which gate failed
GATES = {"snr_db_min": 10.0, "align_pmr_db_min": 15.0, "align_jitter_samples_max": 4.0,
         "clip_dbfs_max": -0.5}
CAPTURE_S = 8.0                   # 0-4 s sweep 15 -> 55 -> 25 cm, 4-8 s hold + breathe
N_CAPTURE = int(round(CAPTURE_S / fmcw.T_CHIRP))
HOLD_FROM_S = 4.25                # respiration window: hold phase after a short settle
CUE = ("Hold still and breathe", 4.0)
ANNOUNCE = ("Book or palm at 15 centimeters. Sweep out to 55 and back to 25, then hold still and breathe "
            "slowly. Starting in 3, 2, 1, move now")
RETRY_ANNOUNCE = "Re-testing. Book or palm at 15 centimeters. 3, 2, 1, move now"
HAND_MIN_CM, HAND_MAX_CM = 10.0, 120.0
THRESH_DB = fmcw.tgc_threshold_db(fmcw.range_axis_m() * 100)    # range-adaptive CFAR (TGC), 11 -> 8 dB
RD_HOP = 8                        # sliding Range-Doppler cube: 64-chirp blocks every 8 chirps
STILL_CM_S = 5.0                  # |range rate| below this counts as a zero-velocity hold
STREAM_STATS = ART / "live_radar_stream_stats.json"

# ---------------------------------------------------------------- audio I/O
def osa(script):
    return subprocess.run(["osascript", "-e", script], capture_output=True, text=True).stdout.strip()


def say(text):
    """SILENT by design: the user works in a library. Never call macOS `say` (or a terminal bell).
    Announcements become an on-screen countdown window + terminal banner (src/visual_cue.py)."""
    import visual_cue
    visual_cue.banner(text, countdown=3 if "3, 2, 1" in text else 0)


def find_device(name, kind):
    for d in sd.query_devices():
        if name in d["name"] and d[f"max_{kind}_channels"] > 0:
            return d["index"], d["name"]
    raise RuntimeError(f"{kind} device '{name}' not found: {sd.query_devices()}")


def spoken_cue(text, n_total, at_sample, peak):
    """Formerly `say` speech mixed into the transmit buffer. SILENT now (library): returns zeros, so the
    speakers only ever emit the 18.8-22.3 kHz chirps."""
    print(f"[cue] (silent) at {at_sample / fmcw.FS:.1f} s: {text}", flush=True)
    return np.zeros(n_total)


def _spoken_cue_audible(text, n_total, at_sample, peak):
    """Old audible version, kept for reference only. Do not use in the library."""
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "cue.wav"
        subprocess.run(["say", "-o", str(f), "--file-format=WAVE", "--data-format=LEI16@48000", text],
                       check=True, capture_output=True)
        fs, x = wavfile.read(f)
    x = (x[:, 0] if x.ndim > 1 else x).astype(np.float64) / 32768.0
    x = sosfiltfilt(butter(6, 8000, "low", fs=fs, output="sos"), x)
    x *= peak / np.abs(x).max()
    out = np.zeros(n_total)
    end = min(n_total, at_sample + len(x))
    out[at_sample:end] = x[:end - at_sample]
    return out


def capture(cfg, announce, n_chirps=N_CAPTURE, cue=CUE):
    mic, mic_name = find_device(MIC_NAME, "input")
    spk, spk_name = find_device(SPK_NAME, "output")
    default_out = sd.query_devices(kind="output")["name"]
    tx, lead = fmcw.make_tx(n_chirps=n_chirps, alpha=cfg["tukey_alpha"], amp=cfg["amp"])
    cue_info = None
    if cue:
        peak = max(0.06, 0.98 - cfg["amp"])
        try:
            tx = np.clip(tx + spoken_cue(cue[0], len(tx), lead + int(cue[1] * fmcw.FS), peak), -1, 1).astype(np.float32)
            cue_info = {"text": cue[0], "at_s": cue[1], "peak": peak, "lowpass_hz": 8000}
        except (subprocess.CalledProcessError, OSError, ValueError) as e:
            print(f"[live] spoken cue skipped: {e}", flush=True)
    saved_vol = osa("output volume of (get volume settings)")
    saved_mute = osa("output muted of (get volume settings)")
    try:
        osa(f"set volume output volume {TEST_VOLUME}")
        osa("set volume without output muted")
        say(announce)
        rec = sd.playrec(np.repeat(tx[:, None], 2, axis=1), samplerate=fmcw.FS, channels=1,
                         device=(mic, spk), dtype="float32", blocking=True)[:, 0]
    finally:
        if saved_vol.isdigit():
            osa(f"set volume output volume {saved_vol}")
        if saved_mute == "true":
            osa("set volume with output muted")
    restored_vol = osa("output volume of (get volume settings)")
    say("Calibration complete")
    audio = {"mic": mic_name, "speaker": spk_name, "default_output": default_out,
             "osascript_volume_targets_speaker": default_out == spk_name,
             "saved_volume": saved_vol, "saved_muted": saved_mute, "test_volume": TEST_VOLUME,
             "restored_volume": restored_vol, "tx_samples": len(tx), "lead_samples": lead,
             "announce": announce, "cue": cue_info, "chirps": n_chirps,
             "rec_peak_dbfs": float(20 * np.log10(np.abs(rec).max() + 1e-12))}
    return rec.astype(np.float64), lead, audio


# ---------------------------------------------------------------- DSP
def band_power(f, p, lo, hi):
    m = (f >= lo) & (f <= hi)
    return float(np.mean(p[m]))


OOB_HZ = (15_000.0, fmcw.F0 - 500.0)  # out-of-band reference just below the sweep


def spectrum_snr(rec, starts):
    """Band SNR: chirp-train PSD vs the silent tail (after the last echo + 150 ms reverb guard)."""
    s0, s1 = int(starts[0]), int(starts[-1]) + fmcw.N
    sig = rec[s0:s1]
    noise = rec[s1 + int(0.15 * fmcw.FS):]
    f, ps = welch(sig, fmcw.FS, nperseg=4096)
    _, pn = welch(noise, fmcw.FS, nperseg=min(4096, len(noise)))
    s, n = band_power(f, ps, fmcw.F0, fmcw.F1), band_power(f, pn, fmcw.F0, fmcw.F1)
    snr = 10 * np.log10(max(s - n, 1e-30) / n)
    contrast = 10 * np.log10(s / band_power(f, ps, *OOB_HZ))
    return f, ps, pn, {"band_snr_db": float(snr), "band_vs_out_of_band_db": float(contrast),
                       "band_power_dbfs": float(10 * np.log10(s)),
                       "noise_floor_dbfs": float(10 * np.log10(n)),
                       "noise_ref_seconds": float(len(noise) / fmcw.FS)}


def align(rec, lead, chirp, n_chirps=None):
    """Matched-filter the recording against one chirp; locate the direct path of every chirp."""
    xc = correlate(rec, chirp, mode="valid", method="fft")
    env = np.abs(hilbert(xc))
    search = env[lead:]
    first = lead + int(np.argmax(search > 0.5 * search.max()))
    l0 = first + int(np.argmax(env[first:first + fmcw.N // 4]))
    n_chirps = n_chirps or fmcw.N_CHIRPS
    m = np.arange(n_chirps)
    peaks = np.full(n_chirps, np.nan)
    for i in m:
        c = l0 + i * fmcw.N
        if c + 48 < len(env):
            peaks[i] = c - 48 + np.argmax(env[c - 48:c + 48])
    ok = np.isfinite(peaks)
    b, a = np.polyfit(m[ok], peaks[ok], 1)
    resid = peaks[ok] - (a + b * m[ok])
    starts = a + b * m                                     # fractional: follows the fitted clock exactly
    seg = env[l0:l0 + n_chirps * fmcw.N]
    q = {"latency_samples": int(l0 - lead), "latency_ms": float((l0 - lead) / fmcw.FS * 1e3),
         "period_fit_samples": float(b), "clock_drift_ppm": float((b - fmcw.N) / fmcw.N * 1e6),
         "jitter_samples_rms": float(np.sqrt(np.mean(resid ** 2))),
         "peak_to_median_db": float(20 * np.log10(np.median(env[peaks[ok].astype(int)]) / np.median(seg))),
         "frames_in_bounds": bool(starts[-1] + fmcw.N + 32 <= len(rec))}
    return starts, q


def fractional_frames(rec, starts, pad=32):
    """Chirp frames cut at fractional sample starts via a frequency-domain fractional delay.

    Rounding starts to whole samples makes the frame grid jump one sample every 1/(drift * N) chirps
    (~3.3 s at 6.5 ppm): a 2.7 rad phase step on every bin at once, which MTI and presence read as
    full-range motion. Fractional starts keep the grid on the fitted clock.
    """
    n = fmcw.N + 2 * pad
    ramp = np.exp(2j * np.pi * np.fft.rfftfreq(n))
    out = []
    for s in starts:
        i = int(np.floor(s))
        seg = rec[i - pad:i - pad + n]
        out.append(np.fft.irfft(np.fft.rfft(seg) * ramp ** (s - i), n)[pad:pad + fmcw.N])
    return np.array(out)


def load_model(s):
    pkg = bench.model_path(s)
    compiled = pkg.with_suffix(".mlmodelc")
    weight = fmcw.dechirp_dft_kernel()
    tag = (ct.models.MLModel(str(pkg), skip_model_load=True).user_defined_metadata.get("kernel_tag")
           if pkg.exists() else None)
    if not compiled.exists() or tag != bench.kernel_tag(weight):   # band changed -> rebuild the weights
        print(f"[live] rebuilding S={s} model for {fmcw.F0:.0f}-{fmcw.F1:.0f} Hz (was {tag})", flush=True)
        compiled = bench.compile_once(bench.build_model(s, weight))
    model = ct.models.CompiledMLModel(str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE)
    plan = bench.compute_plan(compiled, ct.ComputeUnit.CPU_AND_NE)
    return model, next(o["preferred"] for o in plan if o["op"].endswith("conv"))


def ne_range_fft(frames, chirps):
    """[M, N] frames -> complex [M, K] range spectra via the CoreML model (batched + streaming)."""
    g = 1.0 / np.abs(frames).max(axis=1, keepdims=True)    # per-chirp: FP16 activations in [-1, 1]
    x = (frames * g).astype(np.float16)
    m = len(frames)
    bank, bank_dev = load_model(BANK)
    single, single_dev = load_model(1)
    pad = -m % BANK
    xb = np.concatenate([x, np.zeros((pad, fmcw.N), np.float16)])
    outs = []
    t0 = time.perf_counter()
    for i in range(0, len(xb), BANK):
        blk = np.ascontiguousarray(xb[i:i + BANK].T[None, :, None, :])
        outs.append(bank.predict({bench.INPUT: blk})[bench.OUTPUT].reshape(fmcw.C_OUT, BANK))
    t_bank = time.perf_counter() - t0
    y = np.concatenate(outs, axis=1)[:, :m]
    t0 = time.perf_counter()
    for i in range(m):
        single.predict({bench.INPUT: np.ascontiguousarray(x[i][None, :, None, None])})
    t_single = time.perf_counter() - t0
    ref = fmcw.dechirp_dft_kernel().astype(np.float64) @ x.astype(np.float64).T
    X = fmcw.to_complex(y) * fmcw.amp_scale() / g          # conv is per-column linear: undo each gain
    timing = {"frames": m, "bank_size": BANK, "bank_conv_device": bank_dev, "bank_calls": len(xb) // BANK,
              "bank_total_ms": t_bank * 1e3, "bank_chirps_per_s": m / t_bank,
              "single_conv_device": single_dev, "single_total_ms": t_single * 1e3,
              "single_us_per_chirp": t_single / m * 1e6,
              "rel_err_vs_fp64": float(np.linalg.norm(y - ref) / np.linalg.norm(ref)),
              "realtime_factor": float(chirps * fmcw.T_CHIRP / t_bank)}
    return X, timing


def analyse(X, Y):
    """Phase 3 dual-path analysis (TGC CFAR). X: raw range spectra [M, K]; Y: [1,-2,1] MTI spectra [M-2, K]."""
    r_cm = fmcw.range_axis_m() * 100
    ref = np.abs(X).mean(0).max()                           # direct path = 0 dB
    a, b = fmcw.dual_path_scores(X, Y)
    score = np.maximum(a, b)
    cands = fmcw.candidates(score, r_cm, (HAND_MIN_CM, HAND_MAX_CM), 10 ** (THRESH_DB / 20))
    r, v, st = fmcw.kalman_track(cands)
    rows = np.arange(len(Y))
    t = (rows + 1.5) * fmcw.T_CHIRP                        # MTI row m is centred on chirp m + 1
    k = np.clip(np.searchsorted(r_cm, np.nan_to_num(r)), 0, len(r_cm) - 1)
    upd, on = st == 2, st > 0
    from_b = upd & (b[rows, k] > a[rows, k])
    pres = fmcw.presence_residual(X)
    static_db = 20 * np.log10(np.abs(X).mean(0) / ref + 1e-12)
    mti_db = 20 * np.log10(np.abs(Y).mean(0) / ref + 1e-12)
    pres_db = 20 * np.log10(pres.mean(0) / ref + 1e-12)
    sp, _ = find_peaks(static_db)
    sp = sp[r_cm[sp] >= 10.0]
    starts = np.flatnonzero(np.diff(np.concatenate([[0], on.astype(int)])) == 1)
    ends = np.flatnonzero(np.diff(np.concatenate([on.astype(int), [0]])) == -1)
    still = on & (np.abs(v) < STILL_CM_S)
    vv = v[on]
    near = (r_cm >= 15) & (r_cm <= 80)
    stats = {
        "static_strongest_reflectors_cm": [round(float(x), 1) for x in r_cm[sp[np.argsort(static_db[sp])[::-1][:3]]]],
        "mti_direct_path_residual_db": float(20 * np.log10(np.median(np.abs(Y[:, 0])) / fmcw.MTI_NOISE_GAIN / ref)),
        "mti_floor_15_80cm_db": float(20 * np.log10(np.median(np.abs(Y[:, near])) / fmcw.MTI_NOISE_GAIN / ref)),
        "presence_direct_path_residual_db": float(20 * np.log10(np.median(pres[:, r_cm < 5]) / ref)),
        "presence_floor_15_80cm_db": float(20 * np.log10(np.median(pres[:, near]) / ref)),
        "cfar_threshold_db": {"near_db": 11.0, "far_db": 8.0, "taper_cm": [35.0, 75.0],
                              "far_confirmation_chirps": 6, "near_confirmation_chirps": 4},
        "chirps_with_candidates": float(np.mean([len(c[0]) > 0 for c in cands])),
        "track_coverage": float(on.mean()),
        "track_updated": float(upd.mean()),
        "track_coasted": float((st == 1).mean()),
        "track_segments": int(len(starts)),
        "longest_segment_s": float((ends - starts + 1).max() * fmcw.T_CHIRP) if len(starts) else 0.0,
        "updates_from_presence_path": float(from_b.sum() / max(1, upd.sum())),
        "zero_velocity_hold_s": float((still & upd).sum() * fmcw.T_CHIRP),
        "zero_velocity_updates_from_presence_path": float((still & from_b).sum() / max(1, (still & upd).sum())),
        "motion_or_presence_snr_db": float(20 * np.log10(np.median(score[rows, k][upd]))) if upd.any() else None,
        "hand_range_cm": {"min": float(np.nanmin(r)) if on.any() else None,
                          "median": float(np.nanmedian(r)) if on.any() else None,
                          "max": float(np.nanmax(r)) if on.any() else None},
        "range_rate_cm_s": {"p95_abs": float(np.percentile(np.abs(vv), 95)) if len(vv) else None,
                            "max_approach": float(vv.min()) if len(vv) else None,
                            "max_recede": float(vv.max()) if len(vv) else None},
        "micro_doppler_shift_hz_p95": float(2 * np.percentile(np.abs(vv), 95) / 100 * fmcw.FC / fmcw.C_SOUND)
        if len(vv) else None,
        "range_doppler_coupling_s": float(fmcw.RD_COUPLING_S),
        "tracker": {"gate_cm": [HAND_MIN_CM, HAND_MAX_CM], "presence_min_cm": fmcw.PRESENCE_MIN_CM,
                    "kalman": "CV [r, v], z = r + kappa*v, sigma_a 600 cm/s^2, sigma_z 2.5 cm, gate 3 sigma & 15 cm, "
                              "nearest-in-gate, coast <= 15 chirps with velocity x0.9/chirp, init 4 chirps "
                              "within 10 cm (6 when beyond 35 cm)"},
    }
    for name, sel in (("sweep_0_4s", t < CUE[1]), ("hold_4s_end", t >= CUE[1])):
        on_w = on & sel
        stats[name] = {"coverage": float(on_w.sum() / max(1, sel.sum())),
                       "updated": float((upd & sel).sum() / max(1, sel.sum())),
                       "range_cm": [float(np.nanmin(r[on_w])), float(np.nanmax(r[on_w]))] if on_w.any() else None}
    return {"r_cm": r_cm, "static_db": static_db, "mti_db": mti_db, "pres_db": pres_db, "score": score,
            "score_p95_db": 20 * np.log10(np.percentile(score, 95, axis=0) + 1e-12),
            "score_max_db": 20 * np.log10(score.max(0) + 1e-12),
            "t": t, "track": r, "v": v, "status": st, "stats": stats}


def settle_time(an, t_cue=CUE[1], still_chirps=10):
    """First time after the hold cue when the tracked range rate stays under STILL_CM_S for 0.4 s."""
    t, v, st = an["t"], an["v"], an["status"]
    still = (st > 0) & (np.abs(v) < STILL_CM_S) & (t >= t_cue)
    run = np.convolve(still.astype(int), np.ones(still_chirps, int), "valid")
    hit = np.flatnonzero(run == still_chirps)
    return float(t[hit[0]]) if len(hit) else HOLD_FROM_S


def respiration_analysis(X, t_from=HOLD_FROM_S):
    """Direct-path phase-referenced displacement of the breathing bin (25-55 cm) over the hold window."""
    after = np.flatnonzero(np.arange(len(X)) * fmcw.T_CHIRP >= t_from)
    quiet, energy_db = fmcw.quiet_segment(X[after])
    rows = after[quiet]                                  # longest motion-free stretch after t_from
    r_cm = fmcw.range_axis_m() * 100
    ref = np.abs(X).mean(0).max()
    k, power = fmcw.breathing_bin(X, rows)
    xh = X[rows]
    d_ref, info = fmcw.breathing_displacement_mm(xh, k, reference=True)      # 3-bin coherent, clamped steps
    d_raw, _ = fmcw.breathing_displacement_mm(xh, k, reference=False)
    wave, est = fmcw.respiration(d_ref)
    # Noise-floor yardstick: the most phase-stable reflector beyond 150 cm (nobody stands there). Strength
    # alone is not enough: the 2.45 m wall return decorrelates chirp to chirp in this room.
    xr = xh * np.exp(-1j * np.angle(xh[:, :1]))
    coh = np.abs(xr.mean(0)) / np.abs(xr).mean(0)
    level = 20 * np.log10(np.abs(xh).mean(0) / ref)
    far = np.flatnonzero((r_cm >= 150) & (r_cm <= 280) & (level > -50))   # not the last bins of the axis
    kw = far[np.argmax(coh[far])]
    # A phase-noise yardstick only means something on a coherent reflector; otherwise report none.
    floor = ({ref_: float(np.std(detrend(fmcw.displacement_mm(xh, kw, reference=ref_)[0]))) for ref_ in (True, False)}
             if coh[kw] >= 0.8 else {True: None, False: None})
    loud = energy_db >= np.median(energy_db) + 6.0
    edges = np.diff(np.concatenate([[0], loud.astype(int), [0]]))
    bursts = [[float((after[0] + a) * fmcw.T_CHIRP), float((after[0] + b) * fmcw.T_CHIRP)]
              for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)) if b - a > 5]
    stats = {"hold_window_s": [float(rows[0] * fmcw.T_CHIRP), float((rows[-1] + 1) * fmcw.T_CHIRP)],
             "motion_bursts_s": bursts,

             "bin_cm": float(r_cm[k]), "target_level_db_re_direct": float(20 * np.log10(np.abs(xh[:, k]).mean() / ref)),
             **est, **info,
             "unreferenced_peak_to_peak_mm": fmcw.respiration(d_raw)[1]["peak_to_peak_mm"],   # same filter
             "static_reflector_cm": float(r_cm[kw]), "static_reflector_coherence": float(coh[kw]),
             "static_reflector_jitter_mm_rms": {"referenced": floor[True], "unreferenced": floor[False]},
             "lambda_mm": float(fmcw.LAMBDA_C * 1000),
             "rad_per_mm": float(4 * np.pi / (fmcw.LAMBDA_C * 1000))}
    t = rows * fmcw.T_CHIRP
    d_all, _ = fmcw.breathing_displacement_mm(X[after], k)                    # whole span, for display
    return {"t": t, "d_ref": d_ref, "d_raw": d_raw - d_raw[0], "wave": wave, "stats": stats,
            "t_all": after * fmcw.T_CHIRP, "d_all": d_all}


def rd_cube(frames):
    """Sliding 64-chirp Range-Doppler maps from ONE fused NE call (B=16); hop widens to fit long captures."""
    b = 16
    hop = max(RD_HOP, int(np.ceil((len(frames) - fused.S) / (b - 1))))
    starts = np.arange(0, len(frames) - fused.S + 1, hop)
    blocks = np.stack([frames[i:i + fused.S] for i in starts[:b]])
    pad = np.zeros((b - len(blocks),) + blocks.shape[1:])
    model, compiled = fused.load(b)
    plan = bench.compute_plan(compiled, ct.ComputeUnit.CPU_AND_NE)
    x, g = fused.to_input(np.concatenate([blocks, pad]) + 1e-9)
    t0 = time.perf_counter()
    out = model.predict({fused.INPUT: x})[fused.OUTPUT]
    call_ms = (time.perf_counter() - t0) * 1e3
    p = out[:len(blocks), :, 0, :].astype(np.float64) * (fmcw.amp_scale() / (fused.STAGE2_SCALE * g[:len(blocks), None, None])) ** 2
    r_cm = fmcw.range_axis_m() * 100
    gate = (r_cm >= 15) & (r_cm <= 80)
    dop = fused.doppler_axis_hz()
    moving = np.abs(dop) >= 1.0
    energy = p[:, moving][:, :, gate].sum(axis=(1, 2))
    best = int(np.argmax(energy))
    pk = np.unravel_index(np.argmax(p[best][:, gate] * moving[:, None]), p[best][:, gate].shape)
    info = {"maps": int(len(blocks)), "block_chirps": fused.S, "hop_chirps": int(hop), "batch_per_call": b,
            "ne_call_ms": call_ms, "all_ops_on_ne": all(o["preferred"] == "NeuralEngine" for o in plan if o["op"] != "const"),
            "doppler_unambiguous_cm_s": float(fmcw.C_SOUND / (4 * fmcw.FC * fmcw.T_CHIRP) * 100),
            "most_active_block_centre_s": float((starts[best] + fused.S / 2) * fmcw.T_CHIRP),
            "most_active_peak_range_cm": float(r_cm[gate][pk[1]]),
            "most_active_peak_doppler_hz": float(dop[pk[0]])}
    return p[best], info


def gates_pass(snr, aq, audio):
    return {"snr": snr["band_snr_db"] >= GATES["snr_db_min"],
            "alignment": aq["peak_to_median_db"] >= GATES["align_pmr_db_min"]
            and aq["jitter_samples_rms"] <= GATES["align_jitter_samples_max"] and aq["frames_in_bounds"],
            "no_clip": audio["rec_peak_dbfs"] <= GATES["clip_dbfs_max"]}


def next_config(cfg, g):
    """Adjust the chirp for the gate that failed: clipping -> less drive; weak -> more energy;
    misaligned -> smoother Tukey edges (less spectral splatter in the matched filter)."""
    if not g["no_clip"]:
        return {**cfg, "amp": round(cfg["amp"] * 0.5, 3)}
    if not g["snr"]:
        return {"tukey_alpha": max(0.05, round(cfg["tukey_alpha"] - 0.05, 3)), "amp": 1.0}
    return {**cfg, "tukey_alpha": min(0.30, round(cfg["tukey_alpha"] + 0.05, 3))}


# ---------------------------------------------------------------- dashboard
SURF, INK, INK2, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#383835"
BLUE, ORANGE, AQUA, YELLOW = "#3987e5", "#d95926", "#199e70", "#c98500"


def style(ax, title):
    ax.set_facecolor(SURF)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8, fontweight="bold")
    ax.tick_params(colors=INK2, labelsize=8.5)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRID, lw=0.6, alpha=0.7)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def legend(ax, **kw):
    return ax.legend(**{"frameon": False, "fontsize": 8.5, "labelcolor": INK2, **kw})


def dashboard(spec, an, resp, rd_map, rd_info, stream, live, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    fig = plt.figure(figsize=(17, 10.5), facecolor=SURF)
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.18, left=0.05, right=0.985, top=0.9, bottom=0.07)
    fig.suptitle("SonarANE Phase 3 — TGC hand tracking + phase-referenced respiration on the M2 Neural Engine",
                 color=INK, fontsize=15, x=0.05, ha="left", fontweight="bold")
    fig.text(0.05, 0.935, meta, color=INK2, fontsize=9.5, ha="left")
    cmap = LinearSegmentedColormap.from_list("seq", [SURF, "#104281", "#2a78d6", "#86b6ef", "#cde2fb"])
    r_cm = an["r_cm"]

    # Panel 1: spectrum | range-adaptive threshold
    sub = gs[0, 0].subgridspec(1, 2, wspace=0.28)
    f, ps, pn, snr = spec
    ax = fig.add_subplot(sub[0])
    style(ax, "1 · Spectrum, SNR & range-adaptive CFAR")
    m = (f >= 15_000) & (f <= 24_000)
    ax.axvspan(fmcw.F0 / 1e3, fmcw.F1 / 1e3, color=BLUE, alpha=0.10, lw=0)
    ax.plot(f[m] / 1e3, 10 * np.log10(pn[m]), color=INK2, lw=1.2, label="Noise (silent tail)")
    ax.plot(f[m] / 1e3, 10 * np.log10(ps[m]), color=BLUE, lw=1.8, label="Chirp train")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("PSD (dB re FS²/Hz)")
    ax.text(0.97, 0.95, f"SNR {snr['band_snr_db']:.1f} dB", transform=ax.transAxes, ha="right", va="top",
            color=INK, fontsize=10, bbox=dict(fc=SURF, ec=GRID, boxstyle="round,pad=0.3"))
    legend(ax, loc="lower left", fontsize=7.5)
    ax = fig.add_subplot(sub[1])
    style(ax, "")
    g = (r_cm >= HAND_MIN_CM) & (r_cm <= HAND_MAX_CM)
    ax.axvspan(35, HAND_MAX_CM, color=AQUA, alpha=0.08, lw=0)
    ax.plot(r_cm[g], an["score_max_db"][g], color=INK2, lw=1.0, label="Score max over capture")
    ax.plot(r_cm[g], an["score_p95_db"][g], color=BLUE, lw=1.6, label="Score 95th pct")
    ax.plot(r_cm[g], THRESH_DB[g], color=ORANGE, lw=2.0, label="TGC threshold 11→8 dB")
    ax.axhline(11, color=ORANGE, lw=1, ls=":", alpha=0.7)
    ax.text(37, 1.5, "far zone: 6-chirp confirm", color=INK2, fontsize=7.5)
    ax.set_xlabel("Range (cm)")
    ax.set_ylabel("dB over per-bin CFAR floor")
    ax.set_ylim(0, max(30, float(np.nanmax(an["score_max_db"][g])) + 2))
    legend(ax, loc="upper right", fontsize=7.5)

    # Panel 2: live stream latency | fused NE Range-Doppler map
    sub = gs[0, 1].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.3)
    ax = fig.add_subplot(sub[0])
    style(ax, "2 · Real-time stream & NE Range-Doppler")
    if stream:
        lat = np.asarray(stream["frame_latency_ms"])
        ax.hist(lat, bins=40, color=BLUE, edgecolor=SURF, lw=0.5)
        ax.axvline(1000 * fmcw.T_CHIRP, color=ORANGE, lw=1.5, ls="--")
        ax.text(0.97, 0.95, f"{stream['source']} · {stream['frames']} frames @ {stream['fps']:.1f} fps\n"
                f"p50 {stream['latency_ms_p50']:.1f} ms · p95 {stream['latency_ms_p95']:.1f} ms\n"
                f"NE call p50 {stream['ne_call_ms_p50']:.2f} ms ({stream['ne_conv_device']})\n"
                f"budget {1000 * fmcw.T_CHIRP:.0f} ms/chirp · dropped {stream['dropped_blocks']}",
                transform=ax.transAxes, ha="right", va="top", color=INK, fontsize=8,
                bbox=dict(fc=SURF, ec=GRID, boxstyle="round,pad=0.3"))
        ax.set_xlabel("Per-chirp processing latency (ms)")
        ax.set_ylabel("Frames")
    else:
        ax.text(0.5, 0.5, "run src/live_radar_app.py --headless 10\nto record stream latency",
                transform=ax.transAxes, ha="center", va="center", color=INK2)
    ax = fig.add_subplot(sub[1])
    style(ax, "")
    rr = r_cm <= 120
    db = 10 * np.log10(rd_map[:, rr] / rd_map[:, rr].max() + 1e-12)
    dop = fused.doppler_axis_hz()
    ax.imshow(db, origin="lower", aspect="auto", cmap=cmap, vmin=-30, vmax=0,
              extent=[0, r_cm[rr][-1], dop[0], dop[-1]], interpolation="nearest")
    ax.set_xlabel("Range (cm)")
    ax.set_ylabel(f"Doppler (Hz, ±{rd_info['doppler_unambiguous_cm_s']:.0f} cm/s)", fontsize=8.5)
    ax.set_title(f"fused NE map @ {rd_info['most_active_block_centre_s']:.1f} s ({rd_info['maps']} maps, "
                 f"{rd_info['ne_call_ms']:.1f} ms)", color=INK2, fontsize=8.5, loc="left")

    # Panel 3: waterfall + Kalman track + range-rate strip
    sub = gs[1, 0].subgridspec(2, 2, height_ratios=[3, 1], width_ratios=[60, 1], hspace=0.08, wspace=0.02)
    ax = fig.add_subplot(sub[0, 0])
    style(ax, "3 · Dual-path detection waterfall & Kalman track (TGC CFAR)")
    rm = r_cm <= 150
    wf = 20 * np.log10(an["score"][:, rm] + 1e-12)
    im = ax.imshow(wf.T, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=30,
                   extent=[an["t"][0], an["t"][-1], 0, r_cm[rm][-1]], interpolation="nearest")
    st = an["status"]
    ax.plot(an["t"], np.where(st == 2, an["track"], np.nan), color=ORANGE, lw=1.8, label="Kalman track (update)")
    ax.plot(an["t"], np.where(st == 1, an["track"], np.nan), color=YELLOW, lw=0, marker="o", ms=2.5,
            label="Kalman coast")
    ax.axvline(CUE[1], color=INK, lw=1, ls="--", alpha=0.7)
    ax.axvspan(resp["t"][0], resp["t"][-1] + fmcw.T_CHIRP, ymin=0, ymax=0.03, color=AQUA, lw=0)
    s_ = an["stats"]
    sw, hd = s_["sweep_0_4s"], s_["hold_4s_end"]
    rng_txt = lambda w: f"{w['range_cm'][0]:.0f}–{w['range_cm'][1]:.0f} cm" if w["range_cm"] else "—"
    note = (f"Track {s_['track_coverage'] * 100:.0f}% ({s_['track_updated'] * 100:.0f}% updated), "
            f"{s_['track_segments']} segment(s) · sweep 0–4 s: {sw['coverage'] * 100:.0f}%, {rng_txt(sw)}\n"
            f"hold 4 s+: {hd['coverage'] * 100:.0f}%, {rng_txt(hd)} · zero-velocity hold {s_['zero_velocity_hold_s']:.1f} s"
            f" · presence path {s_['updates_from_presence_path'] * 100:.0f}% of updates")
    ax.text(0.01, 0.97, note, transform=ax.transAxes, ha="left", va="top", color=INK, fontsize=8.3,
            bbox=dict(fc=SURF, ec=GRID, boxstyle="round,pad=0.3"))
    ax.set_ylim(0, 120)
    ax.set_ylabel("Range (cm)")
    ax.tick_params(labelbottom=False)
    legend(ax, loc="upper right", bbox_to_anchor=(1.0, 0.80), fontsize=8)
    cb = fig.colorbar(im, cax=fig.add_subplot(sub[0, 1]))
    cb.set_label("dB over CFAR floor", color=INK2, fontsize=8.5)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    cb.outline.set_edgecolor(GRID)
    ax2 = fig.add_subplot(sub[1, 0], sharex=ax)
    style(ax2, "")
    ax2.axhspan(-STILL_CM_S, STILL_CM_S, color=AQUA, alpha=0.12, lw=0)
    ax2.axhline(0, color=GRID, lw=1)
    ax2.plot(an["t"], np.where(st > 0, an["v"], np.nan), color=ORANGE, lw=1.6)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Range rate (cm/s)", fontsize=8.5)

    # Panel 4: respiration interferometer
    sub = gs[1, 1].subgridspec(2, 1, hspace=0.45)
    rs = resp["stats"]
    ax = fig.add_subplot(sub[0])
    style(ax, f"4 · Phase-referenced respiration @ {rs['bin_cm']:.1f} cm ({rs['hold_window_s'][0]:.1f}–{rs['hold_window_s'][1]:.1f} s)")
    ax.plot(resp["t"], resp["d_raw"], color=INK2, lw=1.2, label="Unreferenced phase")
    ax.plot(resp["t"], resp["d_ref"], color=ORANGE, lw=1.8, label="Referenced to 0 cm direct path")
    ax.set_ylabel("Displacement (mm)", fontsize=8.5)
    jit = rs["static_reflector_jitter_mm_rms"]
    ax.text(0.99, 0.05, (f"static {rs['static_reflector_cm']:.0f} cm reflector jitter: {jit['referenced']:.3f} mm rms "
                         f"referenced vs {jit['unreferenced']:.3f} unreferenced") if jit["referenced"] is not None
            else f"no coherent static reflector (best coherence {rs['static_reflector_coherence']:.2f})",
            transform=ax.transAxes, ha="right", va="bottom", color=INK2, fontsize=7.8,
            bbox=dict(fc=SURF, ec="none", pad=1.5))
    legend(ax, loc="upper left", fontsize=7.8)
    ax = fig.add_subplot(sub[1], sharex=ax)
    style(ax, "")
    ax.axhline(0, color=GRID, lw=1)
    ax.plot(resp["t"], resp["wave"], color=BLUE, lw=2.0)
    for tp in rs["inhalation_peaks_s"]:
        ax.plot(resp["t"][0] + tp, np.interp(resp["t"][0] + tp, resp["t"], resp["wave"]), "o", color=ORANGE, ms=6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Breathing (mm)", fontsize=8.5)
    ax.text(0.01, 0.04, f"{rs['bpm']:.1f} breaths/min · {rs['peak_to_peak_mm']:.2f} mm p-p · "
            f"{rs['cycles_observed']:.1f} cycles [{rs['confidence']} confidence] · {rs['filter']}\n"
            f"target {rs['target_level_db_re_direct']:.0f} dB re direct · arc-centre fit "
            f"{'on' if rs['arc_centre_fit'] else 'off'} ({rs['arc_span_rad']:.1f} rad) · "
            f"1 mm = {rs['rad_per_mm']:.2f} rad",
            transform=ax.transAxes, ha="left", va="bottom", color=INK, fontsize=8,
            bbox=dict(fc=SURF, ec=GRID, boxstyle="round,pad=0.3"))
    lim = float(np.abs(resp["wave"]).max()) * 1.15 + 1e-6
    ax.set_ylim(-2.2 * lim, lim)
    fig.savefig(PNG, dpi=150, facecolor=SURF)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    global PNG, TELEMETRY, WAV
    ART.mkdir(exist_ok=True)
    fb = json.loads(fused.FUSED_JSON.read_text())
    stream = json.loads(STREAM_STATS.read_text()) if STREAM_STATS.exists() else None
    attempts = []
    src = sys.argv[sys.argv.index("--from-wav") + 1] if "--from-wav" in sys.argv else None
    n_chirps = N_CAPTURE
    if src:                                                # re-analyse a saved capture, no audio played
        _, src_tel, src_wav = out_paths(src)
        if src != PHASE:
            PNG, TELEMETRY, WAV = out_paths(f"{PHASE}_replay_{src}")
        prev = json.loads(src_tel.read_text())
        if (prev["dsp"]["f0"], prev["dsp"]["f1"]) != (fmcw.F0, fmcw.F1):
            sys.exit(f"[live] {src_wav.name} was recorded at {prev['dsp']['f0']:.0f}-{prev['dsp']['f1']:.0f} Hz "
                     f"but fmcw.py is {fmcw.F0:.0f}-{fmcw.F1:.0f} Hz; re-capture instead of --from-wav")
        _, rec = wavfile.read(src_wav)
        rec = rec.astype(np.float64)
        cfg = prev["capture_attempts"][-1]["config"]
        n_chirps = prev["dsp"].get("chirps", fmcw.N_CHIRPS)
        starts, aq = align(rec, prev["audio"]["lead_samples"], fmcw.make_chirp(cfg["tukey_alpha"]), n_chirps)
        spec = spectrum_snr(rec, starts)
        attempts = [{**a, "snr": spec[3], "alignment": aq, "audio": prev["audio"]}
                    for a in prev["capture_attempts"]]
    cfg, announce = dict(FIRST_CONFIG), ANNOUNCE
    for i in range(0 if src else MAX_ATTEMPTS):
        print(f"[live] attempt {i + 1}: {cfg} at volume {TEST_VOLUME}, {fmcw.F0:.0f}-{fmcw.F1:.0f} Hz, "
              f"{CAPTURE_S:.0f} s", flush=True)
        rec, lead, audio = capture(cfg, announce)
        starts, aq = align(rec, lead, fmcw.make_chirp(cfg["tukey_alpha"]), n_chirps)
        spec = spectrum_snr(rec, starts)
        g = gates_pass(spec[3], aq, audio)
        attempts.append({"config": cfg, "snr": spec[3], "alignment": aq, "gates": g, "audio": audio})
        print(f"[live]   SNR {spec[3]['band_snr_db']:.1f} dB, peak {audio['rec_peak_dbfs']:.1f} dBFS, "
              f"align PMR {aq['peak_to_median_db']:.1f} dB, jitter {aq['jitter_samples_rms']:.2f} smp, "
              f"latency {aq['latency_ms']:.1f} ms -> {g}", flush=True)
        if all(g.values()):
            break
        cfg, announce = next_config(cfg, g), RETRY_ANNOUNCE
    if not src:
        wavfile.write(WAV, fmcw.FS, rec.astype(np.float32))

    starts = starts[starts + fmcw.N + 32 <= len(rec)]    # a late output start pushes the last chirps past the end
    frames = fractional_frames(rec, starts)
    mti = fmcw.mti2(frames)
    X_all, live = ne_range_fft(np.concatenate([frames, mti]), len(frames))   # raw + MTI in one NE pass
    X, Y = X_all[:len(frames)], X_all[len(frames):]
    an = analyse(X, Y)
    t_settle = settle_time(an)
    resp = respiration_analysis(X, t_settle)
    rd_map, rd_info = rd_cube(frames)
    cfg = attempts[-1]["config"]
    meta = (f"{'replay of ' + src + '_capture.wav' if src and src != PHASE else 'live capture'} · chirp "
            f"{fmcw.F0 / 1e3:.1f}–{fmcw.F1 / 1e3:.1f} kHz, T = 40 ms, {len(frames)} chirps "
            f"({len(frames) * fmcw.T_CHIRP:.0f} s), Tukey α = {cfg['tukey_alpha']}, amp {cfg['amp']}, vol "
            f"{attempts[-1]['audio']['test_volume']}% · λ = {fmcw.LAMBDA_C * 1000:.1f} mm · FP16 NE range conv + "
            f"fused Range-Doppler · {dt.datetime.now():%Y-%m-%d %H:%M}")
    dashboard(spec, an, resp, rd_map, rd_info, stream, live, meta)

    val_path = ART / "phase2_tracker_validation.json"
    validation = json.loads(val_path.read_text()) if val_path.exists() else None
    final = attempts[-1]
    b16 = fb["batches"]["16"]
    telemetry = {
        "phase": "SonarANE Phase 3",
        "source": ("live capture (re-analysed)" if src == PHASE else f"replay of artifacts/{src}_capture.wav")
        if src else "live capture",
        "captured_at": (prev.get("captured_at", prev["timestamp"]) if src else
                        dt.datetime.now().isoformat(timespec="seconds")),
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "host": {"machine": "MacBook Air M2", "macos": subprocess.run(["sw_vers", "-productVersion"],
                 capture_output=True, text=True).stdout.strip(), "coremltools": ct.__version__},
        "dsp": {"fs": fmcw.FS, "f0": fmcw.F0, "f1": fmcw.F1, "T_chirp_s": fmcw.T_CHIRP, "chirps": len(frames),
                "range_res_cm": fmcw.RANGE_RES_M * 100, "range_grid_cm": fmcw.RANGE_RES_M * fmcw.BIN_STEP * 100,
                "lambda_mm": fmcw.LAMBDA_C * 1000},
        "protocol": {"sweep_s": [0.0, CUE[1]], "hold_s": [CUE[1], len(frames) * fmcw.T_CHIRP],
                     "respiration_window_from_s": t_settle, "settle": "first 0.4 s with |v| < 5 cm/s after cue"},
        "audio": final["audio"],
        "capture_attempts": [{"config": a["config"], "band_snr_db": round(a["snr"]["band_snr_db"], 2),
                              "align_pmr_db": round(a["alignment"]["peak_to_median_db"], 2),
                              "gates": a["gates"]} for a in attempts],
        "spectrum": {k: round(v, 2) for k, v in final["snr"].items()},
        "alignment": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in final["alignment"].items()},
        "tracking": an["stats"],
        "respiration": resp["stats"],
        "ne_range_conv": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in live.items()},
        "ne_fused_range_doppler_cube": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rd_info.items()},
        "fused_benchmark_B16": {"ne_us": round(b16["coreml_cpu_and_ne"]["median_us"], 1),
                                "accelerate_gemm_us": round(b16["numpy_gemm_pipeline"]["median_us"], 1),
                                "ne_speedup_vs_gemm": round(b16["speedup_vs_numpy_gemm_pipeline"], 2),
                                "cpu_offload_vs_gemm_pct": round(b16["cpu_offload_vs_numpy_gemm_pipeline_pct"], 1),
                                "all_ops_on_ne": b16["coreml_cpu_and_ne"]["all_on_ne"]},
        "realtime_stream": None if stream is None else {k: v for k, v in stream.items() if k != "frame_latency_ms"},
        "tracker_validation": validation,
        "artifacts": {k: str(p.relative_to(ROOT)) for k, p in
                      {"dashboard": PNG, "telemetry": TELEMETRY, "capture_wav": src_wav if src else WAV,
                       "fused_benchmark": fused.FUSED_JSON, "tracker_validation": val_path}.items()},
    }
    TELEMETRY.write_text(json.dumps(telemetry, indent=2))
    print(f"[live] wrote {PNG} and {TELEMETRY}")


if __name__ == "__main__":
    main()
