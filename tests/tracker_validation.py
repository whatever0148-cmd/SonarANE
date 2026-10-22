"""SonarANE Phase 2 tracker validation (no audio played).

Runs the dual-path Kalman tracker (src/fmcw.py) on
  * a simulated palm (15 -> 55 cm, 0.8 s pause, 55 -> 15 cm, rest) with a forearm scatterer 12 cm behind
    it, a drifting direct path (1.2 dB gain + 0.05-sample delay wobble) and static room reflectors,
    at -28 dB and -36 dB re the direct path;
  * ten no-hand scenes (same room and drift, different noise) -> must give 0 false tracks;
  * the saved capture artifacts/<src>_capture.wav, when it was recorded in the current band.
Results are merged per band into artifacts/phase2_tracker_validation.json.

Run:  ~/SonarANE/.venv/bin/python ~/SonarANE/tests/tracker_validation.py [src=phase1]
"""
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

OUT = ROOT / "artifacts" / "phase2_tracker_validation.json"
R_CM = fmcw.range_axis_m() * 100
GATE_CM = (10.0, 120.0)
THR = 10 ** (fmcw.tgc_threshold_db(R_CM) / 20)    # range-adaptive CFAR (Phase 3)
NO_HAND_SEEDS = range(1, 11)


def palm_path(t):
    """Palm range (m): 15 -> 55 cm in 2.2 s, pause 0.8 s, 55 -> 15 cm in 2.2 s, rest; raised-cosine moves."""
    up = 15 + 40 * 0.5 * (1 - np.cos(np.pi * np.clip(t / 2.2, 0, 1)))
    down = 55 - 40 * 0.5 * (1 - np.cos(np.pi * np.clip((t - 3.0) / 2.2, 0, 1)))
    return np.where(t < 3.0, up, down) / 100


def sim_frames(hand_db=-28.0, seed=1, hand=True):
    rng = np.random.default_rng(seed)
    n = np.arange(fmcw.N)
    frames = []
    for m in range(fmcw.N_CHIRPS):
        t = (m * fmcw.N + n) / fmcw.FS
        g = 0.2 * 10 ** (0.6 * np.sin(2 * np.pi * 0.15 * t) / 20)
        x = g * np.cos(fmcw.chirp_phase(n - 0.05 * np.sin(2 * np.pi * 0.1 * t)))
        for r_wall, a in ((0.9, 0.02), (2.45, 0.01)):
            x += a * np.cos(fmcw.chirp_phase(n - 2 * r_wall / fmcw.C_SOUND * fmcw.FS))
        if hand:
            rp = palm_path(t) + 0.001 * np.sin(2 * np.pi * 3.1 * t + seed)        # +-1 mm tremor
            a = 0.2 * 10 ** (hand_db / 20)
            for offset, rel in ((0.0, 1.0), (0.12, 0.7)):                          # palm, forearm
                x += a * rel * np.cos(fmcw.chirp_phase(n - 2 * (rp + offset) / fmcw.C_SOUND * fmcw.FS))
        frames.append(x + 3e-4 * rng.standard_normal(fmcw.N))
    return np.array(frames)


def track(frames):
    X = fmcw.fft_pipeline(frames) * fmcw.amp_scale()
    Y = fmcw.fft_pipeline(fmcw.mti2(frames)) * fmcw.amp_scale()
    a, b = fmcw.dual_path_scores(X, Y)
    r, v, st = fmcw.kalman_track(fmcw.candidates(np.maximum(a, b), R_CM, GATE_CM, THR))
    k = np.clip(np.searchsorted(R_CM, np.nan_to_num(r)), 0, len(R_CM) - 1)
    rows = np.arange(len(st))
    from_b = (st == 2) & (b[rows, k] > a[rows, k])
    starts = np.flatnonzero(np.diff(np.concatenate([[0], (st > 0).astype(int)])) == 1)
    return r, v, st, {"coverage": float((st > 0).mean()), "updated": float((st == 2).mean()),
                      "coasted": float((st == 1).mean()), "segments": int(len(starts)),
                      "updates_from_presence_path": float(from_b.sum() / max(1, (st == 2).sum()))}


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "phase1"
    band = f"{fmcw.F0:.0f}-{fmcw.F1:.0f}Hz"
    res = {}
    t_rows = (np.arange(fmcw.N_CHIRPS - 2) + 1.5) * fmcw.T_CHIRP     # MTI row m is centred on chirp m + 1
    truth = palm_path(t_rows) * 100
    pause = (t_rows >= 2.2) & (t_rows < 3.0)
    for db in (-28.0, -36.0):
        r, v, st, m = track(sim_frames(db))
        ok = st > 0
        err = r[ok] - truth[ok]
        m.update(range_rms_cm=float(np.sqrt(np.mean(err ** 2))), range_bias_cm=float(err.mean()),
                 frac_err_gt_8cm=float((np.abs(err) > 8).mean()),
                 pause_coverage=float((st[pause] > 0).mean()), pause_updated=float((st[pause] == 2).mean()))
        res[f"sim_hand_{db:.0f}dB"] = m
    for seed in NO_HAND_SEEDS:
        _, _, st, m = track(sim_frames(seed=seed, hand=False))
        res[f"sim_no_hand_seed{seed}"] = {"false_track_segments": m["segments"], "coverage": m["coverage"]}
    tel_path, wav_path = ROOT / "artifacts" / f"{src}_telemetry.json", ROOT / "artifacts" / f"{src}_capture.wav"
    if tel_path.exists() and wav_path.exists():
        tel = json.loads(tel_path.read_text())
        if (tel["dsp"]["f0"], tel["dsp"]["f1"]) == (fmcw.F0, fmcw.F1):
            import live_sonar_ane_test as live
            _, rec = wavfile.read(wav_path)
            rec = rec.astype(np.float64)
            alpha = tel["capture_attempts"][-1]["config"]["tukey_alpha"]
            starts, _ = live.align(rec, tel["audio"]["lead_samples"], fmcw.make_chirp(alpha))
            r, v, st, m = track(live.fractional_frames(rec, starts))
            m["range_cm"] = [float(np.nanmin(r)), float(np.nanmedian(r)), float(np.nanmax(r))]
            res[f"{src}_wav_replay"] = m
    allres = json.loads(OUT.read_text()) if OUT.exists() else {}
    allres[band] = res
    OUT.write_text(json.dumps(allres, indent=2))
    print(json.dumps({band: res}, indent=1))
    fails = [k for k, v in res.items() if k.startswith("sim_no_hand") and v["false_track_segments"]]
    if fails:
        sys.exit(f"false tracks in {fails}")


if __name__ == "__main__":
    main()
