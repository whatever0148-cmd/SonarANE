"""SonarANE contactless respiration test: direct-path phase-referenced chest-motion interferometry.

Sit still 25-55 cm in front of the screen and breathe normally. The script plays a continuous chirp train
(default 30 s, ~6-7 breaths at 15/min) on the MacBook Air speakers, records the built-in mic, cuts
fractional-sample frames, runs them through the FP16 range conv on the Neural Engine, picks the range bin
with the most respiratory-band echo power, and turns its slow-time phase, referenced to the 0 cm direct
path, into displacement: dr = lambda / (4 pi) * dphi (1 mm = 0.75 rad at 20.55 kHz).

Run:  ~/SonarANE/.venv/bin/python ~/SonarANE/tests/live_respiration_ane_test.py [--duration 30]
      ~/SonarANE/.venv/bin/python ~/SonarANE/tests/live_respiration_ane_test.py --from-wav phase3
          silent: analyse the hold part of an existing capture (from the tracker's settle time)
      ... --from-wav respiration   re-analyse the last capture in place
Only the longest motion-free stretch is analysed (fmcw.quiet_segment): body movement makes the phase jump
~pi per chirp and unwrapping then slips whole 8 mm wavelengths.
Outputs artifacts/respiration_proof.png and artifacts/respiration_telemetry.json.
"""
import argparse
import datetime as dt
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import fmcw  # noqa: E402
import live_sonar_ane_test as live  # noqa: E402

ART = ROOT / "artifacts"
PNG, TELEMETRY, WAV = ART / "respiration_proof.png", ART / "respiration_telemetry.json", ART / "respiration_capture.wav"
ANNOUNCE = ("Press the flat book against your chest, lean back in your chair, and take steady breaths for 30 "
            "seconds. Starting in 3, 2, 1")


def plot(resp, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rs = resp["stats"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.5), facecolor=live.SURF, sharex=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.1, hspace=0.25)
    fig.suptitle("SonarANE — contactless respiration (direct-path phase-referenced interferometry)",
                 color=live.INK, fontsize=13, x=0.08, ha="left", fontweight="bold")
    fig.text(0.08, 0.905, meta, color=live.INK2, fontsize=9, ha="left")
    live.style(a1, f"Displacement of the {rs['bin_cm']:.1f} cm bin ({rs['target_level_db_re_direct']:.0f} dB re direct path)"
                   f" · shaded = body movement (excluded), analysed {rs['hold_window_s'][0]:.1f}–{rs['hold_window_s'][1]:.1f} s")
    for b0, b1 in rs.get("motion_bursts_s", []):
        a1.axvspan(b0, b1, color=live.YELLOW, alpha=0.18, lw=0)
    a1.plot(resp["t_all"], resp["d_all"], color=live.INK2, lw=1.0, label="Referenced phase, whole capture")
    off = np.interp(resp["t"][0], resp["t_all"], resp["d_all"])
    a1.plot(resp["t"], resp["d_ref"] + off, color=live.ORANGE, lw=2.0, label="Analysed motion-free window")
    a1.set_ylabel("mm")
    live.legend(a1, loc="upper left")
    live.style(a2, "Breathing waveform")
    a2.axhline(0, color=live.GRID, lw=1)
    a2.plot(resp["t"], resp["wave"], color=live.BLUE, lw=2)
    for tp in rs["inhalation_peaks_s"]:
        a2.plot(resp["t"][0] + tp, np.interp(resp["t"][0] + tp, resp["t"], resp["wave"]), "o", color=live.ORANGE, ms=6)
    a2.set_ylabel("mm")
    a2.set_xlabel("Time (s)")
    a2.text(0.01, 0.95, f"{rs['bpm']:.1f} breaths/min · {rs['peak_to_peak_mm']:.2f} mm p-p · {rs['cycles_observed']:.1f} "
            f"cycles [{rs['confidence']}] · {rs['filter']}\nspectrum {rs['bpm_from_spectrum']:.1f} / peaks "
            f"{'—' if rs['bpm_from_peaks'] is None else format(rs['bpm_from_peaks'], '.1f')} per min · "
            + (f"static {rs['static_reflector_cm']:.0f} cm reflector jitter "
               f"{rs['static_reflector_jitter_mm_rms']['referenced']:.3f} mm rms"
               if rs['static_reflector_jitter_mm_rms']['referenced'] is not None
               else f"no coherent static reflector (best {rs['static_reflector_coherence']:.2f})"),
            transform=a2.transAxes, va="top", color=live.INK, fontsize=8.5,
            bbox=dict(fc=live.SURF, ec=live.GRID, boxstyle="round,pad=0.3"))
    fig.savefig(PNG, dpi=150, facecolor=live.SURF)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seconds", "--duration", dest="seconds", type=float, default=30.0)
    ap.add_argument("--from-wav", dest="src", default=None, help="analyse artifacts/<src>_capture.wav silently")
    args = ap.parse_args()
    if args.src:
        tel = json.loads((ART / f"{args.src}_telemetry.json").read_text())
        if (tel["dsp"]["f0"], tel["dsp"]["f1"]) != (fmcw.F0, fmcw.F1):
            sys.exit(f"{args.src} was recorded in another band; re-capture")
        _, rec = wavfile.read(ART / f"{args.src}_capture.wav")
        rec, n = rec.astype(np.float64), tel["dsp"]["chirps"]
        cfg, lead, audio = tel["capture_attempts"][-1]["config"], tel["audio"]["lead_samples"], tel["audio"]
        source = f"artifacts/{args.src}_capture.wav"
    else:
        n = int(round(args.seconds / fmcw.T_CHIRP))
        cfg = dict(live.FIRST_CONFIG)
        rec, lead, audio = live.capture(cfg, ANNOUNCE, n_chirps=n, cue=None)
        wavfile.write(WAV, fmcw.FS, rec.astype(np.float32))
        source = "live capture"
    starts, aq = live.align(rec, lead, fmcw.make_chirp(cfg["tukey_alpha"]), n)
    starts = starts[starts + fmcw.N + 32 <= len(rec)]    # a late output start pushes the last chirps past the end
    frames = live.fractional_frames(rec, starts)
    X_all, ne = live.ne_range_fft(np.concatenate([frames, fmcw.mti2(frames)]), len(frames))
    X, Y = X_all[:len(frames)], X_all[len(frames):]
    # Phase 3 captures: start after the tracker sees the hand settle; sit-still captures: skip the first second
    t_from = live.settle_time(live.analyse(X, Y)) if args.src and args.src.startswith("phase") else 1.0
    resp = live.respiration_analysis(X, t_from)
    meta = (f"{source} · {fmcw.F0 / 1e3:.1f}–{fmcw.F1 / 1e3:.1f} kHz · λ = {fmcw.LAMBDA_C * 1000:.1f} mm · window "
            f"{resp['stats']['hold_window_s'][0]:.1f}–{resp['stats']['hold_window_s'][1]:.1f} s · NE range conv on "
            f"{ne['bank_conv_device']} · {dt.datetime.now():%Y-%m-%d %H:%M}")
    plot(resp, meta)
    out = {"source": source, "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
           "dsp": {"f0": fmcw.F0, "f1": fmcw.F1, "chirps": n, "chirps_recorded": len(frames)},
           "capture_attempts": [{"config": cfg}],
           "alignment": aq,
           "audio": {k: audio.get(k) for k in ("saved_volume", "restored_volume", "test_volume", "rec_peak_dbfs",
                                               "lead_samples")},
           "ne_range_conv": {k: ne[k] for k in ("bank_conv_device", "bank_total_ms", "rel_err_vs_fp64")},
           "respiration": resp["stats"]}
    TELEMETRY.write_text(json.dumps(out, indent=2))
    print(json.dumps(resp["stats"], indent=1))
    print(f"[resp] wrote {PNG} and {TELEMETRY}")


if __name__ == "__main__":
    main()
