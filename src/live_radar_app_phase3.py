"""SonarANE live radar: continuous FMCW sonar with the Neural Engine range conv running at 25 Hz.

A sounddevice duplex Stream plays one 40 ms chirp per callback block on the MacBook Air speakers and
hands every recorded block to the processing loop. After a 1 s direct-path calibration (re-checked every
5 s for clock drift), each new chirp frame goes through the FP16 range conv on the NE (the S=8 bank;
CoreML places S=1 on the CPU), then the dual-path TGC Kalman tracker and the phase-referenced
respiration estimator run on sliding windows.

Run:  ~/SonarANE/.venv/bin/python ~/SonarANE/src/live_radar_app.py                # window, plays ultrasound
      ~/SonarANE/.venv/bin/python ~/SonarANE/src/live_radar_app.py --term         # terminal bar
      ~/SonarANE/.venv/bin/python ~/SonarANE/src/live_radar_app.py --sim          # synthetic scene, no audio
      ~/SonarANE/.venv/bin/python ~/SonarANE/src/live_radar_app.py --headless 10  # no display, write stats
Ctrl-C (or closing the window) stops the stream and restores the speaker volume.
"""
import argparse
import collections
import json
import queue
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import coremltools as ct
from scipy.signal import correlate, hilbert

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ane_kernels"))
import fmcw  # noqa: E402
import ane_radar_benchmark as bench  # noqa: E402

MIC_NAME, SPK_NAME = "MacBook Air Microphone", "MacBook Air Speakers"
VOLUME, AMP, ALPHA = 75, 0.9, 0.15
BANK = 8                          # smallest bank CoreML schedules on the NE
TRACK_WINDOW = 100                # chirps (4 s) for CFAR floors, presence background, Kalman re-run
RESP_WINDOW = 150                 # chirps (6 s) for breathing
RESP_MAX_PP_MM = 15.0             # larger than this is a hand / body moving, not breathing
GATE_CM = (10.0, 120.0)
THRESH = 10 ** (fmcw.tgc_threshold_db(fmcw.range_axis_m() * 100) / 20)
STATS_JSON = ROOT / "artifacts" / "live_radar_stream_stats.json"


def osa(script):
    return subprocess.run(["osascript", "-e", script], capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------- sources
class AudioSource:
    """Duplex stream: every callback block is exactly one chirp out and N samples in."""

    def __init__(self):
        import sounddevice as sd
        self.sd = sd
        self.mic = self._find(MIC_NAME, "input")
        self.spk = self._find(SPK_NAME, "output")
        self.chirp = np.repeat((fmcw.make_chirp(ALPHA) * AMP)[:, None], 2, axis=1).astype(np.float32)
        self.q = queue.Queue()
        self.dropped = 0
        self.saved = None
        self.stream = None

    def _find(self, name, kind):
        for d in self.sd.query_devices():
            if name in d["name"] and d[f"max_{kind}_channels"] > 0:
                return d["index"]
        raise RuntimeError(f"{kind} device {name!r} not found")

    def _callback(self, indata, outdata, frames, time_info, status):
        outdata[:] = self.chirp
        if status:
            self.dropped += 1
        self.q.put((time.perf_counter(), indata[:, 0].copy()))

    def start(self):
        self.saved = (osa("output volume of (get volume settings)"), osa("output muted of (get volume settings)"))
        osa(f"set volume output volume {VOLUME}")
        osa("set volume without output muted")
        self.stream = self.sd.Stream(samplerate=fmcw.FS, blocksize=fmcw.N, device=(self.mic, self.spk),
                                     channels=(1, 2), dtype="float32", callback=self._callback)
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
        if self.saved:
            if self.saved[0].isdigit():
                osa(f"set volume output volume {self.saved[0]}")
            if self.saved[1] == "true":
                osa("set volume with output muted")


class SimSource:
    """Synthetic scene in real time: 110 ms I/O latency, direct path, wall, a hand sweeping 15-55 cm for
    4 s, then a chest at 30 cm breathing 4 mm p-p at 15 breaths/min. No audio."""

    def __init__(self, seed=0):
        self.q = queue.Queue()
        self.dropped = 0
        self.rng = np.random.default_rng(seed)
        self.latency = 5281.3
        self.chirp = fmcw.make_chirp(ALPHA).astype(np.float64)
        self._stop = threading.Event()

    def _wave(self, n_abs, delay):
        x = (n_abs - delay) % fmcw.N
        return np.interp(x, np.arange(fmcw.N), self.chirp, period=fmcw.N)

    def _run(self):
        k, t0 = 0, time.perf_counter()
        while not self._stop.is_set():
            n_abs = k * fmcw.N + np.arange(fmcw.N)
            t = n_abs / fmcw.FS
            s = 2 / fmcw.C_SOUND * fmcw.FS
            phase = (t % 12.0)
            hand = np.where(phase < 4.0, 0.35 - 0.20 * np.cos(2 * np.pi * phase / 4.0), 0.30)
            breath = np.where(phase >= 4.0, 0.002 * np.sin(2 * np.pi * 0.25 * (phase - 4.0)), 0.0)
            y = 0.2 * self._wave(n_abs, self.latency)
            y += 0.01 * self._wave(n_abs, self.latency + 2.45 * s)
            y += 0.006 * self._wave(n_abs, self.latency + (hand + breath) * s)
            y += 3e-4 * self.rng.standard_normal(fmcw.N)
            self.q.put((time.perf_counter(), y.astype(np.float32)))
            k += 1
            time.sleep(max(0.0, t0 + k * fmcw.T_CHIRP - time.perf_counter()))

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._stop.set()


# ---------------------------------------------------------------- processing
def load_bank_model():
    pkg = bench.model_path(BANK)
    compiled = pkg.with_suffix(".mlmodelc")
    weight = fmcw.dechirp_dft_kernel()
    tag = (ct.models.MLModel(str(pkg), skip_model_load=True).user_defined_metadata.get("kernel_tag")
           if pkg.exists() else None)
    if not compiled.exists() or tag != bench.kernel_tag(weight):
        compiled = bench.compile_once(bench.build_model(BANK, weight))
    plan = bench.compute_plan(compiled, ct.ComputeUnit.CPU_AND_NE)
    device = next(o["preferred"] for o in plan if o["op"].endswith("conv"))
    return ct.models.CompiledMLModel(str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE), device


class Radar:
    def __init__(self):
        self.model, self.device = load_bank_model()
        self.buf = np.zeros(0, np.float32)
        self.offset = None
        self.calib = []
        self.frames = collections.deque(maxlen=BANK)
        self.X = collections.deque(maxlen=RESP_WINDOW)
        self.n_frames = 0
        self.t_calibrated = None
        self.n_backlog = 0
        self.recal = 0
        self.lat_ms, self.ne_ms = [], []
        self.state = {"r": None, "v": None, "status": 0, "bpm": None, "pp_mm": None, "resp_conf": None,
                      "score": None, "wave": None, "track_hist": collections.deque(maxlen=150)}

    def _direct_offset(self, x):
        env = np.abs(hilbert(correlate(x, fmcw.make_chirp(ALPHA), mode="valid", method="fft")))
        folded = np.zeros(fmcw.N)
        for i in range(len(env) // fmcw.N):
            folded = np.maximum(folded, env[i * fmcw.N:(i + 1) * fmcw.N])
        return int(np.argmax(folded))

    def feed(self, t_arrival, block):
        """Add one recorded block; returns True when a new chirp frame was processed."""
        if self.offset is None:
            self.calib.append(block)
            if len(self.calib) >= 25:                     # 1 s
                self.offset = self._direct_offset(np.concatenate(self.calib))
                self.buf = np.concatenate(self.calib)[self.offset:]
                self.calib = []
                self.t_calibrated = time.perf_counter()
                self.n_backlog = len(self.buf) // fmcw.N
            return False
        self.buf = np.concatenate([self.buf, block])
        done = False
        while len(self.buf) >= fmcw.N:
            frame, self.buf = self.buf[:fmcw.N].astype(np.float64), self.buf[fmcw.N:]
            self._process(frame)
            self.lat_ms.append((time.perf_counter() - t_arrival) * 1e3)
            done = True
        if self.n_frames and self.n_frames % 125 == 0 and len(self.frames) == BANK:
            self._recheck_drift()
        return done

    def _recheck_drift(self):
        """Direct-path peak should sit at sample ~0 of every frame; nudge the frame grid if it drifted."""
        env = np.abs(hilbert(correlate(np.concatenate(list(self.frames)), fmcw.make_chirp(ALPHA),
                                       mode="full", method="fft")))
        lag = int(np.argmax(env[fmcw.N - 1 - 8:fmcw.N - 1 + 9])) - 8
        if lag:
            self.buf = self.buf[lag:] if lag > 0 else np.concatenate([np.zeros(-lag, np.float32), self.buf])
            self.recal += 1

    def _process(self, frame):
        self.frames.append(frame)
        if len(self.frames) < BANK:
            return
        f = np.stack(self.frames)
        g = 1.0 / np.abs(f).max(axis=1, keepdims=True)
        x = np.ascontiguousarray((f * g).T[None, :, None, :]).astype(np.float16)
        t0 = time.perf_counter()
        y = self.model.predict({bench.INPUT: x})[bench.OUTPUT].reshape(fmcw.C_OUT, BANK)
        self.ne_ms.append((time.perf_counter() - t0) * 1e3)
        self.X.append(fmcw.to_complex(y[:, -1:])[0] * fmcw.amp_scale() / g[-1, 0])
        self.n_frames += 1
        if len(self.X) >= 20:
            self._track()
        if len(self.X) >= 100 and self.n_frames % 5 == 0:
            self._breathe()

    def _track(self):
        xh = np.array(list(self.X)[-TRACK_WINDOW:])
        yh = fmcw.mti2(xh)
        a, b = fmcw.dual_path_scores(xh, yh)
        score = np.maximum(a, b)
        r_cm = fmcw.range_axis_m() * 100
        r, v, st = fmcw.kalman_track(fmcw.candidates(score, r_cm, GATE_CM, THRESH))
        self.state.update(r=None if st[-1] == 0 else float(r[-1]), v=None if st[-1] == 0 else float(v[-1]),
                          status=int(st[-1]), score=20 * np.log10(score[-1] + 1e-12))
        self.state["track_hist"].append(self.state["r"])

    def _breathe(self):
        xh = np.array(self.X)
        rows = np.arange(len(xh))
        k, _ = fmcw.breathing_bin(xh, rows)
        d, _ = fmcw.displacement_mm(xh, k, reference=True)
        w, est = fmcw.respiration(d)
        moving = est["peak_to_peak_mm"] > RESP_MAX_PP_MM
        self.state.update(bpm=None if moving else est["bpm"], pp_mm=est["peak_to_peak_mm"],
                          resp_conf="motion" if moving else est["confidence"],
                          wave=w, resp_bin_cm=float(fmcw.range_axis_m()[k] * 100))


# ---------------------------------------------------------------- displays
def term_line(st, radar):
    r = st["r"]
    bar = "" if r is None else "█" * int(np.clip(r / 120 * 40, 0, 40))
    rng = "  --.- cm" if r is None else f"{r:6.1f} cm"
    vel = "      " if st["v"] is None else f"{st['v']:+6.1f}"
    br = (f"  -- ({st['resp_conf'] or 'warming up'})" if st["bpm"] is None
          else f"{st['bpm']:4.1f}/min {st['pp_mm']:.2f} mm ({st['resp_conf']})")
    ne = np.median(radar.ne_ms[-25:]) if radar.ne_ms else 0.0
    return f"\r{rng} |{bar:<40}| v {vel} cm/s · breath {br} · NE {ne:.2f} ms ({radar.device})   "


class Window:
    def __init__(self):
        import matplotlib
        import matplotlib.pyplot as plt
        self.plt = plt
        plt.ion()
        self.fig, (self.ax_wf, self.ax_br) = plt.subplots(2, 1, figsize=(9, 7), height_ratios=[2, 1],
                                                          facecolor="#1a1a19")
        self.fig.canvas.manager.set_window_title("SonarANE live radar")
        r_cm = fmcw.range_axis_m() * 100
        self.rm = r_cm <= 120
        self.wf = np.zeros((int(self.rm.sum()), 150))
        for ax in (self.ax_wf, self.ax_br):
            ax.set_facecolor("#1a1a19")
            ax.tick_params(colors="#c3c2b7")
            for s in ax.spines.values():
                s.set_color("#383835")
        self.im = self.ax_wf.imshow(self.wf, origin="lower", aspect="auto", vmin=0, vmax=30, cmap="Blues_r",
                                    extent=[-6, 0, 0, r_cm[self.rm][-1]])
        (self.trk,) = self.ax_wf.plot([], [], color="#d95926", lw=2)
        self.txt = self.ax_wf.text(0.01, 0.97, "", transform=self.ax_wf.transAxes, va="top", color="#f0efec",
                                   fontsize=11, family="monospace")
        self.ax_wf.set_ylabel("Range (cm)", color="#c3c2b7")
        self.ax_wf.set_xlabel("Seconds ago", color="#c3c2b7")
        (self.br,) = self.ax_br.plot([], [], color="#3987e5", lw=2)
        self.ax_br.set_ylabel("Breathing (mm)", color="#c3c2b7")
        self.ax_br.set_xlabel("Seconds", color="#c3c2b7")
        self.fig.tight_layout()
        self.closed = False
        self.fig.canvas.mpl_connect("close_event", lambda e: setattr(self, "closed", True))

    def draw(self, st, radar):
        if st["score"] is not None:
            self.wf = np.roll(self.wf, -1, axis=1)
            self.wf[:, -1] = st["score"][self.rm]
            self.im.set_data(self.wf)
        hist = np.array([np.nan if r is None else r for r in st["track_hist"]], float)
        self.trk.set_data(np.arange(-len(hist), 0) * fmcw.T_CHIRP, hist)
        self.txt.set_text(term_line(st, radar).strip().replace(" · ", "\n"))
        if st["wave"] is not None:
            w = st["wave"]
            self.br.set_data(np.arange(len(w)) * fmcw.T_CHIRP, w)
            self.ax_br.set_xlim(0, len(w) * fmcw.T_CHIRP)
            lim = max(0.5, float(np.abs(w).max()) * 1.2)
            self.ax_br.set_ylim(-lim, lim)
        self.plt.pause(0.001)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sim", action="store_true", help="synthetic scene, no audio")
    ap.add_argument("--term", action="store_true", help="terminal display instead of a window")
    ap.add_argument("--headless", type=float, default=None, metavar="SECONDS",
                    help="no display; run this long and write artifacts/live_radar_stream_stats.json")
    args = ap.parse_args()
    radar = Radar()
    src = SimSource() if args.sim else AudioSource()
    view = None if (args.term or args.headless) else Window()
    t_start = time.perf_counter()
    draws = 0
    try:
        src.start()
        while True:
            try:
                t_arr, block = src.q.get(timeout=1.0)
            except queue.Empty:
                continue
            if radar.feed(t_arr, block) and src.q.qsize() < 2:
                if view is not None and radar.n_frames % 4 == 0:
                    view.draw(radar.state, radar)
                    draws += 1
                    if view.closed:
                        break
                elif args.term:
                    print(term_line(radar.state, radar), end="", flush=True)
            if args.headless and time.perf_counter() - t_start >= args.headless:
                break
    except KeyboardInterrupt:
        pass
    finally:
        src.stop()
    elapsed = time.perf_counter() - t_start
    lat = np.array(radar.lat_ms[5:]) if len(radar.lat_ms) > 5 else np.array(radar.lat_ms)
    stats = {"source": "sim" if args.sim else "audio", "seconds": round(elapsed, 2), "frames": radar.n_frames,
             "fps": (radar.n_frames - radar.n_backlog) / max(1e-9, time.perf_counter() - (radar.t_calibrated or t_start)),
             "dropped_blocks": src.dropped,
             "direct_path_offset_samples": radar.offset, "drift_recalibrations": radar.recal,
             "ne_conv_device": radar.device, "ne_bank": BANK,
             "ne_call_ms_p50": float(np.median(radar.ne_ms)) if radar.ne_ms else None,
             "latency_ms_p50": float(np.median(lat)) if len(lat) else None,
             "latency_ms_p95": float(np.percentile(lat, 95)) if len(lat) else None,
             "latency_ms_max": float(lat.max()) if len(lat) else None,
             "last_range_cm": radar.state["r"], "last_bpm": radar.state["bpm"], "last_resp_state": radar.state["resp_conf"],
             "last_breath_pp_mm": radar.state["pp_mm"], "frame_latency_ms": [round(float(x), 3) for x in lat]}
    if args.headless:
        STATS_JSON.write_text(json.dumps(stats, indent=2))
        print(json.dumps({k: v for k, v in stats.items() if k != "frame_latency_ms"}, indent=1))
    else:
        print(f"\n[radar] {radar.n_frames} frames, NE p50 {stats['ne_call_ms_p50']} ms, "
              f"latency p95 {stats['latency_ms_p95']} ms, dropped {src.dropped}")


if __name__ == "__main__":
    main()
