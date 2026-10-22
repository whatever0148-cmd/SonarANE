"""SonarANE Phase 4 live test (library-hardened): silent pulsed TDM stereo MIMO + NE super-resolution
+ joint 2D tracking in the personal workspace bubble.

SILENT: no `say`, no spoken cue, no terminal bell (the user works in a library). Cues are an on-screen
countdown window (src/visual_cue.py) plus terminal prints. The duplex stream is opened before the countdown
("warm"), so the speakers start on time.

Modes:
  (default)            8 s live sweep: move a flat notebook/palm left-right (-25 .. +25 cm) and in-out
                       (15 .. 45 cm) inside the workspace -> artifacts/phase4_physics_rewritten.png + telemetry
  --seconds N          live sweep of N seconds
  --library-baseline   6 s, hands off the keyboard/screen, sitting normally: counts false hand tracks inside
                       the 10-55 cm workspace -> artifacts/phase4_library_baseline_*
  --from-wav [TAG]     re-analyse artifacts/TAG_capture.wav (default phase4; the original 12 s capture is
                       kept as phase4_v1_12s)
Per speaker channel: fractional-sample frames, one combined NE call per 64 frames (range FFT + 80-iteration
FISTA), joint backprojection + 4-state Cartesian EKF gated to y <= 55 cm, bistatic r <= 58 cm.
"""
import datetime as dt
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import correlate, hilbert, welch

ROOT = Path(__file__).resolve().parents[1]
for p in ("src", "ane_kernels", "tests"):
    sys.path.insert(0, str(ROOT / p))
import fmcw  # noqa: E402
import mimo  # noqa: E402
import ane_superres_mimo as sr  # noqa: E402
import live_sonar_ane_test as live  # noqa: E402

ART = ROOT / "artifacts"
WARM_S = 1.0
AMP, ALPHA, VOLUME = 0.9, 0.15, 75
SWEEP_TEXT = ("SWEEP NOTEBOOK NOW", "left ↔ right (−25..+25 cm) and in ↔ out (12..45 cm)")
BASELINE_TEXT = ("KEEP HANDS IN LAP", "sit normally, don't reach in")
CAL_TEXT = "HOLD HANDS IN LAP — CALIBRATING"
YELLOW_BG, GREEN_BG = "#f5c518", "#2ecc71"


def paths(tag):
    """(png, telemetry, wav, meta) for a capture tag; 'phase4' keeps the historical dashboard name."""
    png = ART / ("phase4_physics_rewritten.png" if tag == "phase4" else f"{tag}_dashboard.png")
    return png, ART / f"{tag}_telemetry.json", ART / f"{tag}_capture.wav", ART / f"{tag}_capture_meta.json"


PNG, TELEMETRY, WAV, META = paths("phase4")


# ---------------------------------------------------------------- capture
GUIDE = ("Here is the plan. Rest both hands in your lap. When the window turns yellow, stay still for two seconds "
         "while it calibrates. When it turns green, lift the notebook, hold it flat facing the screen, about twenty "
         "centimetres away, and sweep it slowly from left to right and back, one sweep every two seconds. "
         "After four seconds move it in towards the screen and out again. Keep it smooth. Starting in a moment.")


def capture_warm(tx, lead, preroll_s, seconds, instruction, voice=False):
    """Duplex stream opened first (silence). Silent on-screen cues: rest hands in lap (no chirps), then
    YELLOW 'HOLD HANDS IN LAP - CALIBRATING' while the pre-roll chirps run, then GREEN instruction for
    `seconds`. Cue timing follows the playback position of the transmit buffer.
    Returns (recording, input index where the train's first sample went out, audio info)."""
    import visual_cue
    mic, mic_name = live.find_device(live.MIC_NAME, "input")
    spk, spk_name = live.find_device(live.SPK_NAME, "output")
    rec, st = [], {"play": False, "pos": 0, "start_in": None, "n_in": 0, "done": False, "status": 0}

    def cb(indata, outdata, frames, time_info, status):
        if status:
            st["status"] += 1
        rec.append(indata[:, 0].copy())
        if st["play"]:
            if st["start_in"] is None:
                st["start_in"] = st["n_in"]
            chunk = tx[st["pos"]:st["pos"] + frames]
            outdata[:len(chunk)] = chunk
            outdata[len(chunk):] = 0
            st["pos"] += frames
            st["done"] = st["pos"] >= len(tx)
        else:
            outdata[:] = 0
        st["n_in"] += frames

    cue = visual_cue.Cue("SonarANE - capture", voice=voice)
    saved = (live.osa("output volume of (get volume settings)"), live.osa("output muted of (get volume settings)"))
    t_open = time.perf_counter()
    try:
        live.osa(f"set volume output volume {VOLUME}")
        live.osa("set volume without output muted")
        stream = sd.Stream(samplerate=mimo.FS, blocksize=480, device=(mic, spk), channels=(1, 2),
                           dtype="float32", callback=cb)
        stream.start()
        cue.show("READY", "Rest both hands in your lap — calibration starts next", YELLOW_BG)
        if voice:
            cue.speak(GUIDE)
            cue.pump(24.0)                                     # let the guide finish before the chirps start
        cue.pump(WARM_S + 1.5)
        st["play"] = True
        spoken = set()
        while not st["done"]:
            tt = (st["pos"] - lead) / mimo.FS
            if tt < preroll_s:
                cue.show(f"{np.clip(np.ceil(preroll_s - tt), 1, preroll_s):.0f}", f"{CAL_TEXT} ({preroll_s:.0f}s)", bg=YELLOW_BG)
                if "cal" not in spoken:
                    cue.speak("Hold still. Calibrating.")
                    spoken.add("cal")
            elif tt < preroll_s + seconds:
                cue.show(f"{np.ceil(preroll_s + seconds - tt):.0f}", f"{instruction[0]} ({seconds:.0f}s)\n{instruction[1]}",
                         bg=GREEN_BG)
                if "go" not in spoken:
                    cue.speak("Go. Sweep the notebook left and right." if instruction is SWEEP_TEXT else "Go. Stay as you are.")
                    spoken.add("go")
                elif tt >= preroll_s + seconds / 2 and "half" not in spoken and instruction is SWEEP_TEXT:
                    cue.speak("Now in and out.")
                    spoken.add("half")
            cue.pump(0.05)
        cue.pump(0.3)
        cue.speak("Done. You can relax.")
        stream.stop()
        stream.close()
    finally:
        if saved[0].isdigit():
            live.osa(f"set volume output volume {saved[0]}")
        if saved[1] == "true":
            live.osa("set volume with output muted")
        cue.close("Done", 1.0)
    restored = live.osa("output volume of (get volume settings)")
    audio = {"mic": mic_name, "speaker": spk_name, "saved_volume": saved[0], "restored_volume": restored,
             "test_volume": VOLUME, "stream": "warm duplex (opened before a silent on-screen countdown)",
             "audible_cues": "none (library mode: no say, no spoken cue, no bell)",
             "callback_status_flags": st["status"], "session_s": round(time.perf_counter() - t_open, 2),
             "rec_peak_dbfs": float(20 * np.log10(np.abs(np.concatenate(rec)).max() + 1e-12))}
    return np.concatenate(rec).astype(np.float64), st["start_in"], audio


# ---------------------------------------------------------------- alignment + frames
def align_tdm(rec, t0, lead, n_pri):
    """Direct-path arrival of every left and right chirp; linear clock fit per channel (fractional starts)."""
    env = np.abs(hilbert(correlate(rec, mimo.make_chirp(ALPHA), mode="valid", method="fft")))
    lo = t0 + lead
    win = env[lo:lo + int(0.5 * mimo.FS)]
    first = lo + int(np.argmax(win > 0.5 * win.max()))
    l0 = first + int(np.argmax(env[first:first + mimo.NC // 4]))
    r_guess = l0 + mimo.SLOT["R"]
    r0 = r_guess - 60 + int(np.argmax(env[r_guess - 60:r_guess + 60]))
    out, q = {}, {"latency_ms": float((l0 - lo) / mimo.FS * 1e3), "r_minus_l_direct_samples": int(r0 - r_guess)}
    m = np.arange(n_pri)
    for ch, base in (("L", l0), ("R", r0)):
        pk = np.full(n_pri, np.nan)
        for i in m:
            c = base + i * mimo.NP
            if c + 24 < len(env):
                pk[i] = c - 24 + np.argmax(env[c - 24:c + 24])
        ok = np.isfinite(pk)
        b, a = np.polyfit(m[ok], pk[ok], 1)
        starts = a + b * m
        out[ch] = starts[starts + mimo.NC + 32 <= len(rec)]
        q[ch] = {"clock_drift_ppm": float((b - mimo.NP) / mimo.NP * 1e6),
                 "jitter_samples_rms": float(np.sqrt(np.nanmean((pk[ok] - (a + b * m[ok])) ** 2))),
                 "frames_in_bounds": bool(len(out[ch]) == n_pri), "frames": int(len(out[ch]))}
    return out, q


def frames_at(rec, starts, n=mimo.NC, pad=32):
    """Fractional-sample frames (frequency-domain delay), as live.fractional_frames for any frame length."""
    L = n + 2 * pad
    ramp = np.exp(2j * np.pi * np.fft.rfftfreq(L))
    out = []
    for s in starts:
        i = int(np.floor(s))
        out.append(np.fft.irfft(np.fft.rfft(rec[i - pad:i - pad + L]) * ramp ** (s - i), L)[pad:pad + n])
    return np.array(out)


# ---------------------------------------------------------------- analysis
def level_change(X, t):
    lvl = 20 * np.log10(np.abs(X[:, 0]) + 1e-12)
    base = np.median(lvl[t < 2])
    return lvl - base, [round(float(np.median(lvl[(t >= a) & (t < a + 2)]) - base), 2) for a in np.arange(0, t[-1] - 1, 2)]


def peak_width_cm(profile, grid, k):
    """-3 dB (half-power) width of the peak at index k."""
    half = profile[k] / np.sqrt(2)
    lo, hi = k, k
    while lo > 0 and profile[lo] > half:
        lo -= 1
    while hi < len(profile) - 1 and profile[hi] > half:
        hi += 1
    return float(grid[hi] - grid[lo])


def joint_2d(X, x_mic, sweep, gated=True):
    """Gain-normalised spectra -> per-channel scores -> backprojection -> 4-state EKF (+ summary)."""
    Xn, gain_applied = mimo.gain_normalise(X)
    jscores = {ch: mimo.channel_scores(Xn[ch]) for ch in Xn}
    ys = None if gated else mimo.GRID_Y_FULL
    maps, thr_map, gx, gy = mimo.backprojection(jscores, x_mic, ys=ys, r_max=mimo.WORKSPACE_R_MAX if gated else None)
    jc = [mimo.map_candidates(mp, thr_map, gx, gy) for mp in maps]
    jx, jy, jvx, jvy, jst = mimo.joint_track(jc, x_mic, ys=ys)
    jr = {"L": np.full(len(jx), np.nan), "R": np.full(len(jx), np.nan)}
    for i in np.flatnonzero(jst > 0):
        r_, _ = mimo.bistatic(jx[i], jy[i], x_mic)
        jr["L"][i], jr["R"][i] = r_["L"], r_["R"]
    on = jst[sweep] > 0
    seg_on = np.diff(np.concatenate([[0], (jst > 0).astype(int)]))
    summary = {
        "gate": {"y_max_cm": mimo.WORKSPACE_Y_MAX, "r_max_cm": mimo.WORKSPACE_R_MAX} if gated else "none (y <= 70 cm)",
        "sweep_rows": int(sweep.sum()), "rows_with_2d": int(on.sum()),
        "rows_updated": int((jst[sweep] == 2).sum()), "rows_coasted": int((jst[sweep] == 1).sum()),
        "coverage": float(on.mean()) if len(on) else 0.0,
        "segments": int(np.sum(seg_on == 1)),
        "x_cm": [float(np.nanmin(jx[sweep])), float(np.nanmax(jx[sweep]))] if on.any() else None,
        "y_cm": [float(np.nanmin(jy[sweep])), float(np.nanmax(jy[sweep]))] if on.any() else None,
        "r_left_cm": [float(np.nanmin(jr["L"][sweep])), float(np.nanmax(jr["L"][sweep]))] if on.any() else None,
        "r_right_cm": [float(np.nanmin(jr["R"][sweep])), float(np.nanmax(jr["R"][sweep]))] if on.any() else None,
        "gain_normalised": gain_applied}
    return {"x": jx, "y": jy, "status": jst, "r": jr, "map_mean": maps[sweep].mean(0) if sweep.any() else maps.mean(0),
            "gx": gx, "gy": gy, "thr": thr_map, "summary": summary}


def analyse(rec, t0, lead, n_pri, sweep_s, preroll_s=None):
    starts, q = align_tdm(rec, t0, lead, n_pri)
    m = min(len(starts["L"]), len(starts["R"]))
    raw = {ch: frames_at(rec, starts[ch][:m]) for ch in ("L", "R")}
    bg = {ch: raw[ch] - raw[ch].mean(0) for ch in raw}                      # static background removed
    model, plan = sr.load_mimo()
    a = sr.dictionary()
    X_all, P_all, call_ms = sr.run_mimo(model, np.concatenate([raw["L"], raw["R"]]),
                                        np.concatenate([bg["L"], bg["R"]]), a)
    X = {"L": X_all[:m], "R": X_all[m:]}
    P = {"L": P_all[:m], "R": P_all[m:]}
    t = np.arange(m) * mimo.PRI
    tr = {}
    for ch in ("L", "R"):
        r, v, st_, score = mimo.track_channel(X[ch])
        tr[ch] = {"r": r, "v": v, "status": st_, "score": score}
    rows_t = (np.arange(m - 2) + 1) * mimo.PRI
    x_mic = mimo.mic_offset_cm(q["r_minus_l_direct_samples"])
    p0 = preroll_s or 0.0
    sweep = (rows_t >= p0) & (rows_t < p0 + sweep_s)
    joint = joint_2d(X, x_mic, sweep, gated=True)                    # uncalibrated, for comparison
    joint_ungated = joint_2d(X, x_mic, sweep, gated=False)
    hp = mimo.hand_presence_pipeline(X, x_mic, preroll_s)             # pre-roll floors + hand/torso split
    both = (tr["L"]["status"] > 0) & (tr["R"]["status"] > 0)
    rl = np.where(both, tr["L"]["r"], np.nan)
    rr = np.where(both, tr["R"]["r"], np.nan)
    xy = mimo.solve_xy(rl, rr, x_mic)
    # super-resolution vs FFT on real frames: main peak in 10-90 cm while the left track is up
    grid = sr.GRID_CM
    g = (grid >= 10) & (grid <= 90)
    fft_fine = sr.fft_fine(bg["L"])
    widths = {"fft": [], "fista_ne": []}
    for i in np.flatnonzero((tr["L"]["status"][:m - 2] > 0) & sweep) + 1:
        for key, prof in (("fft", fft_fine[i]), ("fista_ne", P["L"][i])):
            kk = np.flatnonzero(g)[np.argmax(prof[g])]
            widths[key].append(peak_width_cm(prof, grid, kk))
    sag = {ch: level_change(X[ch], t) for ch in X}
    f, ps = welch(rec[int(starts["L"][0]):int(starts["L"][m - 1])], mimo.FS, nperseg=4096)
    _, pn = welch(rec[int(0.1 * mimo.FS):int((WARM_S - 0.1) * mimo.FS)], mimo.FS, nperseg=4096)
    band = (f >= mimo.F0) & (f <= mimo.F1)
    snr = float(10 * np.log10(max(ps[band].mean() - pn[band].mean(), 1e-30) / pn[band].mean()))
    # hold window breathing on the left channel, only if the protocol had a hold of >= 2 s
    hold = np.flatnonzero(t >= sweep_s + 0.25)
    est_hold = None
    if len(hold) * mimo.PRI >= 2.0:
        kb = int(np.argmax(np.abs(X["L"][hold][:, (mimo.range_axis_cm() >= 20) & (mimo.range_axis_cm() <= 65)]).mean(0))
                 + np.flatnonzero(mimo.range_axis_cm() >= 20)[0])
        d_hold, _ = fmcw.breathing_displacement_mm(X["L"][hold], kb)
        _, est_hold = fmcw.respiration(d_hold, fs=1 / mimo.PRI)
    ops = [o for o in plan if o["op"] != "const"]
    return {
        "t": t, "rows_t": rows_t, "X": X, "P": P, "tr": tr, "xy": xy, "sweep": sweep, "sweep_s": sweep_s,
        "x_mic": x_mic, "joint": joint, "hp": hp, "preroll_s": p0,
        "fft_fine": fft_fine, "sag": sag, "spec": (f, ps, pn),
        "stats": {
            "alignment": q, "band_snr_db": snr,
            "sag_db_per_2s": {ch: sag[ch][1] for ch in sag},
            "sag_db_at_end": {ch: sag[ch][1][-1] for ch in sag},
            "ne": {"graph_ops": len(ops), "all_on_ne": all(o["preferred"] == "NeuralEngine" for o in ops),
                   "calls": len(call_ms), "call_ms": [round(x, 2) for x in call_ms],
                   "frames_per_call": sr.S, "fista_iterations": sr.ITERS, "grid_cm": 0.6},
            "geometry": {"baseline_cm_assumed": mimo.BASELINE_CM, "mic_x_cm_from_direct_paths": x_mic,
                         "model": "planar: speakers at -/+ baseline/2 on the hinge line, mic on the same line"},
            "tracking": {ch: {"coverage": float((tr[ch]["status"] > 0).mean()),
                              "updated": float((tr[ch]["status"] == 2).mean()),
                              "sweep_range_cm": [float(np.nanmin(tr[ch]["r"][sweep])), float(np.nanmax(tr[ch]["r"][sweep]))]
                              if np.isfinite(tr[ch]["r"][sweep]).any() else None} for ch in tr},
            "hand_tracker": hp["summary"],
            "joint_2d": joint["summary"],
            "joint_2d_ungated": joint_ungated["summary"],
            "xy_sweep": {"points": int(np.isfinite(xy[sweep, 0]).sum()),
                         "x_cm": [float(np.nanmin(xy[sweep, 0])), float(np.nanmax(xy[sweep, 0]))]
                         if np.isfinite(xy[sweep, 0]).any() else None,
                         "y_cm": [float(np.nanmin(xy[sweep, 1])), float(np.nanmax(xy[sweep, 1]))]
                         if np.isfinite(xy[sweep, 1]).any() else None},
            "superres_live_peak_width_cm": {k: (float(np.median(v)) if v else None) for k, v in widths.items()},
            "superres_live_chirps": len(widths["fft"]),
            "hold_breathing_left": None if est_hold is None else
            {k: est_hold[k] for k in ("window_s", "bpm", "confidence", "cycles_observed", "peak_to_peak_mm")},
        }}


def saved_respiration():
    """Three-way consensus on the saved 30 s continuous-chirp capture (book on chest)."""
    tel = json.loads((ART / "respiration_telemetry.json").read_text())
    _, rec = wavfile.read(ART / "respiration_capture.wav")
    rec = rec.astype(np.float64)
    starts, _ = live.align(rec, tel["audio"]["lead_samples"], fmcw.make_chirp(0.15), tel["dsp"]["chirps"])
    starts = starts[starts + fmcw.N + 32 <= len(rec)]
    X = fmcw.fft_pipeline(live.fractional_frames(rec, starts)) * fmcw.amp_scale()
    return live.respiration_analysis(X, 1.0), X


# ---------------------------------------------------------------- dashboard
def dashboard(an, resp, srb, cont_sag, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S_, I_, I2, G_ = live.SURF, live.INK, live.INK2, live.GRID
    B_, O_, A_, Y_ = live.BLUE, live.ORANGE, live.AQUA, live.YELLOW
    st = an["stats"]
    fig = plt.figure(figsize=(17, 10.5), facecolor=S_)
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.2, left=0.05, right=0.985, top=0.9, bottom=0.07)
    fig.suptitle("SonarANE Phase 4 — pulsed TDM stereo MIMO · NE super-resolution · 2D tracking · respiration consensus",
                 color=I_, fontsize=15, x=0.05, ha="left", fontweight="bold")
    fig.text(0.05, 0.935, meta, color=I2, fontsize=9.5, ha="left")

    # 1: sag + latency
    ax = fig.add_subplot(gs[0, 0])
    live.style(ax, "1 · Speaker level over time: continuous chirp vs pulsed TDM (40 % duty)")
    ax.axhline(0, color=G_, lw=1)
    tc, dc = cont_sag
    ax.plot(tc, dc, color=I2, lw=1.4, label="Phase 3 continuous chirp (30 s capture)")
    body = slice(0, -2)                                   # last frames hold the train's end edge
    ax.plot(an["t"][body], an["sag"]["L"][0][body], color=B_, lw=1.6, label="TDM left speaker")
    ax.plot(an["t"][body], an["sag"]["R"][0][body], color=O_, lw=1.6, label="TDM right speaker")
    ax.set_xlim(0, max(14, an["t"][-1]))
    ax.set_ylim(-7, 4.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Direct-path level vs first 2 s (dB)")
    q = st["alignment"]
    ax.text(0.99, 0.97, f"loop latency {q['latency_ms']:.0f} ms (warm stream; cold playrec was 2874 ms) · "
            f"frames L {q['L']['frames']}, R {q['R']['frames']} (all in bounds: {q['L']['frames_in_bounds'] and q['R']['frames_in_bounds']})\n"
            f"{an['t'][-1]:.0f} s level change: L {st['sag_db_at_end']['L']:+.2f} dB, R {st['sag_db_at_end']['R']:+.2f} dB "
            f"· SNR {st['band_snr_db']:.1f} dB", transform=ax.transAxes, ha="right", va="top", color=I_, fontsize=8.5,
            bbox=dict(fc=S_, ec=G_, boxstyle="round,pad=0.3"))
    live.legend(ax, loc="lower left", fontsize=8)

    # 2: super-resolution
    sub = gs[0, 1].subgridspec(1, 2, wspace=0.3)
    ax = fig.add_subplot(sub[0])
    live.style(ax, "2 · Two-target resolution, 30 dB")
    rows = srb["two_target"]["snr_30dB"]
    seps = [float(s) for s in rows]
    for key, col, lab in (("fft", I2, "Hann FFT"), ("fista_fp64", A_, "FISTA fp64"), ("fista_ne_fp16", O_, "FISTA on NE (FP16)")):
        ax.plot(seps, [rows[s][key] * 100 for s in rows], color=col, lw=2 if "ne" in key else 1.4, marker="o", ms=4, label=lab)
    ax.axhline(90, color=G_, lw=1, ls="--")
    ax.set_xlabel("Separation (cm)")
    ax.set_ylabel("Resolved (%)")
    ax.set_ylim(-3, 103)
    live.legend(ax, loc="lower right", fontsize=7.5)
    ax = fig.add_subplot(sub[1])
    live.style(ax, "")
    grid = sr.GRID_CM
    rows_ok = np.flatnonzero((an["tr"]["L"]["status"][:len(an["t"]) - 2] > 0) & an["sweep"]) + 1
    gz = (grid >= 8) & (grid <= 90)                       # beyond the direct-path residual
    if len(rows_ok):
        i = rows_ok[np.argmax([an["P"]["L"][j][gz].max() for j in rows_ok])]
        for prof, col, lab in ((an["fft_fine"][i], I2, "FFT"), (an["P"]["L"][i], O_, "NE FISTA")):
            ax.plot(grid[gz], prof[gz] / prof[gz].max(), color=col, lw=1.8, label=lab)
        ax.set_title(f"live, left ch. @ {an['t'][i]:.1f} s · median -3 dB width\nFFT "
                     f"{st['superres_live_peak_width_cm']['fft']:.1f} cm vs NE "
                     f"{st['superres_live_peak_width_cm']['fista_ne']:.1f} cm (sparsity sharpening)",
                     color=I2, fontsize=8, loc="left")
    ax.set_xlim(8, 90)
    ax.set_xlabel("Range beyond direct path (cm)")
    ax.set_ylabel("Normalised")
    live.legend(ax, loc="upper right", fontsize=7.5)

    # 3: calibrated hand tracker (pre-roll floors, kinematic hand/torso split)
    sub = gs[1, 0].subgridspec(1, 2, width_ratios=[1.25, 1], wspace=0.42)
    ax = fig.add_subplot(sub[0])
    hp = an["hp"]
    hs = st["hand_tracker"]
    w0, w1 = an["preroll_s"], an["preroll_s"] + an["sweep_s"]
    live.style(ax, f"3 · Hand tracker {w0:.0f}–{w1:.0f} s: {hs['hand_tracks']} hand track(s), "
                   f"{hs['hand_rows']}/{hs['rows_after_preroll']} rows · seated presence: {hs['seated_presence_detected']}")
    hm = 20 * np.log10(hp["map_mean"] + 1e-6)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", [S_, "#104281", "#2a78d6", "#86b6ef", "#cde2fb"])
    ax.imshow(hm, origin="lower", aspect="auto", cmap=cmap, vmin=float(np.percentile(hm, 5)),
              vmax=float(np.percentile(hm, 99.5)),
              extent=[hp["gx"][0, 0] - 0.5, hp["gx"][0, -1] + 0.5, hp["gy"][0, 0] - 0.5, hp["gy"][-1, 0] + 0.5])
    at = hp["all_tracks"]
    other = (at["status"] > 0) & (hp["hand"]["status"] == 0) & hp["after"]
    ax.scatter(at["x"][other], at["y"][other], s=8, color=I2, alpha=0.6, label="other motion / presence")
    upd = hp["after"] & (hp["hand"]["status"] == 2)
    cst = hp["after"] & (hp["hand"]["status"] == 1)
    sc = ax.scatter(hp["hand"]["x"][upd], hp["hand"]["y"][upd], c=hp["rows_t"][upd], cmap="autumn", s=16, vmin=w0,
                    vmax=w1, label="hand (EKF update)")
    ax.scatter(hp["hand"]["x"][cst], hp["hand"]["y"][cst], facecolors="none", edgecolors=Y_, s=12, lw=0.8, label="hand (coast)")
    for (sx, sy) in hp["statics"]:
        ax.plot(sx, sy, "x", color=A_, ms=7, mew=2)
    ax.plot([], [], "x", color=A_, ms=7, mew=2, label="pre-roll static reflectors")
    ax.axhline(mimo.HAND_INIT_Y[1], color=Y_, lw=1, ls=":")
    ax.axhline(mimo.HAND_HOLD_Y_MAX, color=Y_, lw=1, ls="--")
    ax.text(-39, mimo.HAND_INIT_Y[1] + 0.8, "hand start ≤ 45", color=Y_, fontsize=7)
    ax.text(-39, mimo.HAND_HOLD_Y_MAX + 0.8, "hand hold ≤ 52 · torso band 40–58", color=Y_, fontsize=7)
    ax.plot([-mimo.BASELINE_CM / 2, mimo.BASELINE_CM / 2], [0, 0], "s", color=B_, ms=8, clip_on=False)
    ax.plot([an["x_mic"]], [0], "^", color=O_, ms=8, clip_on=False)
    ax.set_xlim(-40, 40)
    ax.set_ylim(0, 72)
    ax.set_xlabel("X (cm, left − / right +) · ■ speakers (assumed) ▲ mic")
    ax.set_ylabel("Y (cm)")
    cb = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.05)
    cb.set_label("time (s)", color=I2)
    cb.ax.tick_params(colors=I2, labelsize=8)
    live.legend(ax, loc="upper right", fontsize=7, ncol=2)
    ax = fig.add_subplot(sub[1])
    live.style(ax, "")
    if an["preroll_s"]:
        ax.axvspan(0, an["preroll_s"], color=Y_, alpha=0.15, lw=0)
        ax.text(0.1, 85, "pre-roll\n(calibration)", color=Y_, fontsize=7.5, va="top")
    rl = np.full(len(hp["rows_t"]), np.nan)
    rr_ = np.full(len(hp["rows_t"]), np.nan)
    for i in np.flatnonzero(hp["hand"]["status"] > 0):
        r_, _ = mimo.bistatic(hp["hand"]["x"][i], hp["hand"]["y"][i], an["x_mic"])
        rl[i], rr_[i] = r_["L"], r_["R"]
    for ch, col, rv in (("L", B_, rl), ("R", O_, rr_)):
        ax.plot(an["rows_t"], np.where(an["tr"][ch]["status"] > 0, an["tr"][ch]["r"], np.nan), color=col, lw=0.8,
                alpha=0.35, label=f"{ch} independent 1D (old)")
        ax.plot(hp["rows_t"], rv, color=col, lw=2.0, label=f"{ch} from hand track")
    ax.axhline(mimo.WORKSPACE_R_MAX, color=Y_, lw=1, ls="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Bistatic range (cm)")
    ax.set_ylim(0, 90)
    live.legend(ax, loc="upper right", fontsize=7)

    # 4: respiration consensus on the saved capture
    ax = fig.add_subplot(gs[1, 1])
    rs = resp["stats"]
    live.style(ax, f"4 · Respiration consensus — saved 30 s capture, {rs['bin_cm']:.0f} cm bin")
    ax.axhline(0, color=G_, lw=1)
    ax.plot(resp["t"], resp["wave"], color=B_, lw=2)
    for tp in rs["inhalation_peaks_s"]:
        ax.plot(resp["t"][0] + tp, np.interp(resp["t"][0] + tp, resp["t"], resp["wave"]), "o", color=O_, ms=6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Chest displacement (mm)")
    ax.text(0.01, 0.04, f"{rs['bpm']:.1f} breaths/min [{rs['confidence']}] · spectrum {rs['bpm_from_spectrum']:.2f} · "
            f"autocorr {rs['bpm_from_autocorr']:.2f} · env-normalised peaks {rs['bpm_from_peaks']:.2f} · spread "
            f"{rs['estimator_spread_pct']:.1f} %\n{rs['peak_to_peak_mm']:.1f} mm p-p · {rs['cycles_observed']:.1f} cycles · "
            f"motion-free {rs['hold_window_s'][0]:.1f}–{rs['hold_window_s'][1]:.1f} s · 3-bin coherent, clamped phase steps",
            transform=ax.transAxes, ha="left", va="bottom", color=I_, fontsize=8.3,
            bbox=dict(fc=S_, ec=G_, boxstyle="round,pad=0.3"))
    lim = float(np.abs(resp["wave"]).max()) * 1.15
    ax.set_ylim(-2.0 * lim, lim)
    fig.savefig(PNG, dpi=150, facecolor=S_)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    global PNG, TELEMETRY, WAV, META
    import argparse
    import shutil
    ap = argparse.ArgumentParser(description="SonarANE Phase 4 (silent, library-hardened)")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--library-baseline", action="store_true")
    ap.add_argument("--from-wav", nargs="?", const="phase4", default=None, metavar="TAG")
    ap.add_argument("--voice", action="store_true", help="spoken guidance via macOS say (NOT for the library)")
    args = ap.parse_args()
    ART.mkdir(exist_ok=True)
    # keep the original 12 s capture (6 s sweep + 6 s hold) before a new live run overwrites phase4_*
    _, _, w0, m0 = paths("phase4")
    _, _, w1, m1 = paths("phase4_v1_12s")
    if w0.exists() and m0.exists() and "sweep_s" not in json.loads(m0.read_text()) and not w1.exists():
        shutil.copy(w0, w1)
        old = json.loads(m0.read_text())
        old.update(seconds=12.0, sweep_s=6.0, n_pri=300, protocol="6 s sweep + 6 s hold (spoken cues, pre-library)")
        m1.write_text(json.dumps(old, indent=2))
        print(f"[p4] backed up the original 12 s capture as {w1.name}", flush=True)
    if args.from_wav:
        tag = args.from_wav
        PNG, TELEMETRY, WAV, META = paths(tag)
        meta = json.loads(META.read_text())
        _, rec = wavfile.read(WAV)
        rec, t0, lead, audio = rec.astype(np.float64), meta["tx_start_in"], meta["lead"], meta["audio"]
        seconds = meta.get("seconds", 12.0)
        sweep_s = meta.get("sweep_s", 6.0)
        preroll_s = meta.get("preroll_s")
        baseline = meta.get("mode") == "library_baseline"
    else:
        baseline = args.library_baseline
        tag = "phase4_library_baseline" if baseline else "phase4"
        PNG, TELEMETRY, WAV, META = paths(tag)
        seconds = 6.0 if baseline else args.seconds
        sweep_s = seconds
        preroll_s = mimo.PREROLL_S
        n_pri = int(round((preroll_s + seconds) / mimo.PRI))
        tx, lead = mimo.make_tx(n_pri, amp=AMP, alpha=ALPHA)
        print(f"[p4] SILENT {'library baseline' if baseline else 'live sweep'}: TDM {n_pri} PRIs "
              f"({preroll_s:.0f} s pre-roll + {seconds:.0f} s), "
              f"duty {mimo.DUTY:.0%} per speaker, warm stream, on-screen countdown", flush=True)
        rec, t0, audio = capture_warm(tx, lead, preroll_s, seconds, BASELINE_TEXT if baseline else SWEEP_TEXT, voice=args.voice)
        wavfile.write(WAV, mimo.FS, rec.astype(np.float32))
        meta = {"tx_start_in": int(t0), "lead": int(lead), "audio": audio, "seconds": seconds, "sweep_s": sweep_s,
                "preroll_s": preroll_s,
                "n_pri": n_pri, "mode": "library_baseline" if baseline else "sweep",
                "protocol": f"{preroll_s:.0f} s hands-in-lap pre-roll, then {seconds:.0f} s "
                            + ": ".join(BASELINE_TEXT if baseline else SWEEP_TEXT) + (" (spoken guidance)" if args.voice else ""),
                "captured_at": dt.datetime.now().isoformat(timespec="seconds")}
        META.write_text(json.dumps(meta, indent=2))
    n_pri = meta.get("n_pri", int(round(seconds / mimo.PRI)))
    an = analyse(rec, t0, lead, n_pri, sweep_s, preroll_s)
    resp, Xc = saved_respiration()
    tcont = np.arange(len(Xc)) * fmcw.T_CHIRP
    dcont, _ = level_change(Xc, tcont)
    keep = tcont <= 14
    srb = json.loads(sr.SR_JSON.read_text())
    st = an["stats"]
    kind = "library baseline (hands off)" if baseline else ("live sweep" if not args.from_wav else f"replay of {WAV.name}")
    cues = "spoken guidance (say) + window" if "spoken" in str(meta.get("protocol", "")) else "SILENT (no say / spoken cue)"
    meta_line = (f"{kind}, {seconds:.0f} s, {cues} · TDM 16 ms up-chirps, L @ 0 / R @ 20 ms per "
                 f"40 ms, {fmcw.F0 / 1e3:.1f}–{fmcw.F1 / 1e3:.1f} kHz, amp {AMP}, vol {VOLUME}% · NE: range FFT + "
                 f"{sr.ITERS}-it FISTA ({st['ne']['graph_ops']} ops, all on NE: {st['ne']['all_on_ne']}) · "
                 f"{dt.datetime.now():%Y-%m-%d %H:%M}")
    dashboard(an, resp, srb, (tcont[keep], dcont[keep]), meta_line)
    rs = resp["stats"]
    load = lambda name: json.loads((ART / name).read_text()) if (ART / name).exists() else None
    base = load("phase4_library_baseline_telemetry.json")
    v1 = load("phase4_v1_12s_telemetry.json")
    telemetry = {
        "phase": "SonarANE Phase 4 (library-hardened)",
        "mode": "library_baseline" if baseline else "sweep",
        "source": f"re-analysed {WAV.name}" if args.from_wav else "live capture",
        "captured_at": meta["captured_at"],
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "protocol": meta.get("protocol", "6 s sweep + 6 s hold"), "seconds": seconds, "preroll_s": preroll_s,
        "analysis_window_s": [preroll_s or 0.0, (preroll_s or 0.0) + sweep_s],
        "waveform": {"type": "pulsed TDM stereo FMCW", "pri_ms": mimo.PRI * 1e3, "chirp_ms": mimo.TC * 1e3,
                     "slots_ms": {"L": 0.0, "R": mimo.SLOT["R"] / mimo.FS * 1e3}, "duty_per_speaker": mimo.DUTY,
                     "band_hz": [mimo.F0, mimo.F1], "amp": AMP, "range_res_cm_fft": fmcw.RANGE_RES_M * 100,
                     "range_doppler_coupling_s": mimo.RD_COUPLING_S, "pris": n_pri},
        "audio": audio,
        "alignment": st["alignment"],
        "band_snr_db": round(st["band_snr_db"], 2),
        "speaker_level_change_db": {"tdm_per_2s": st["sag_db_per_2s"], "tdm_at_end": st["sag_db_at_end"],
                                    "continuous_30s_capture_per_2s": level_change(Xc, tcont)[1]},
        "stereo_isolation_sim_db": {"tdm_max": -37.7, "tdm_median": -53.4, "updown_cdm_max": -14.2,
                                    "note": "R leakage into L's 10-80 cm bins re L direct path, with a 30 ms room tail"},
        "right_channel": {"direct_path_db_re_left": -19.8, "far_static_echo_db_re_left": [-4.1, -3.4],
                          "cfar_15_55cm_median_db": {"L": 8.0, "R": 7.8},
                          "note": "from phase4_v1_12s: the right speaker radiates ~4 dB below the left; only its "
                                  "direct bleed to the mic is in a ~16 dB null. Per-bin OS-CFAR already normalises "
                                  "each channel to its own floor, so the null does not penalise right-channel scores."},
        "geometry": st["geometry"],
        "tracking_independent_1d": st["tracking"],
        "hand_tracker": st["hand_tracker"],
        "joint_2d_workspace_uncalibrated": st["joint_2d"],
        "joint_2d_ungated_uncalibrated": st["joint_2d_ungated"],
        "independent_2d": st["xy_sweep"],
        "library_baseline": None if base is None else
        {"captured_at": base["captured_at"], "seconds": base["seconds"], "hand_tracker": base.get("hand_tracker")},
        "replay_v1_12s_capture": None if v1 is None else
        {"hand_tracker": v1.get("hand_tracker"), "independent_2d": v1["independent_2d"]},
        "joint_2d_validation_sim": load("phase4_joint2d_validation.json"),
        "ne_mimo_graph": st["ne"],
        "superres_benchmark": {k: srb[k] for k in ("iters", "lambda", "ops", "all_on_ne", "ne_rel_err_vs_fp64",
                                                    f"ne_us_per_{sr.S}_chirps", f"numpy_fp32_accelerate_us_per_{sr.S}_chirps",
                                                    "ne_speedup_vs_numpy", "resolution_90pct_cm_snr_30dB",
                                                    "resolution_90pct_cm_snr_20dB")},
        "superres_live_peak_width_cm": st["superres_live_peak_width_cm"],
        "superres_live_chirps": st["superres_live_chirps"],
        "respiration_saved_30s": {k: rs[k] for k in ("hold_window_s", "bin_cm", "bpm", "bpm_from_spectrum",
                                                     "bpm_from_autocorr", "bpm_from_peaks", "estimator_spread_pct",
                                                     "confidence", "cycles_observed", "peak_to_peak_mm")},
        "respiration_live_hold_left": st["hold_breathing_left"],
        "artifacts": {k: str(v.relative_to(ROOT)) if v.is_relative_to(ROOT) else str(v) for k, v in
                      {"dashboard": PNG, "telemetry": TELEMETRY, "capture_wav": WAV, "superres_benchmark": sr.SR_JSON}.items()},
    }
    TELEMETRY.write_text(json.dumps(telemetry, indent=2))
    print(f"[p4] wrote {PNG} and {TELEMETRY}")


if __name__ == "__main__":
    main()
