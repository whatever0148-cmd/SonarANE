"""SonarANE showcase: live ultrasonic MIMO radar on the Apple Neural Engine, 100 % silent.

Pulsed TDM stereo chirps (left 0-16 ms, right 20-36 ms of every 40 ms PRI, 18.8-22.3 kHz) go out of the
MacBook speakers; every recorded PRI is cut into its two chirp frames and sent through ONE CoreML graph
(878 ops, all on the NE: range FFT + 80-iteration unrolled FISTA super-resolution, 2 frames per call,
~0.4 ms). After a 2 s hands-in-lap pre-roll the per-bin noise floors are locked, the static scene is
backprojected, and a joint stereo EKF follows the hand on a 1 cm (X, Y) grid. The torso shows up as a
static reflector with ~1 mm breathing micro-motion; holding still turns the chest phase into a
respiration waveform and rate.

Window:  A  live 2D map (X -40..40, Y 5..55 cm), glowing hand cursor, trail, seated-presence ring
         B  NE FISTA profile vs classical FFT of the latest left chirp, NE latency / pipeline rate
         C  respiration waveform and breaths/min while holding still

Run:  ~/SonarANE/.venv/bin/python ~/SonarANE/src/live_radar_app.py                  # live window (chirps only)
      ~/SonarANE/.venv/bin/python ~/SonarANE/src/live_radar_app.py --sim             # synthetic scene, no audio
      ~/SonarANE/.venv/bin/python ~/SonarANE/src/live_radar_app.py --headless 12     # no window; stats JSON
      ~/SonarANE/.venv/bin/python ~/SonarANE/src/live_radar_app.py --demo-from-wav   # 60 fps video from the
                                                                                     # saved captures
Every cue is on screen. Nothing but the ultrasonic chirps is ever played (library rule).
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
from scipy.signal import correlate, hilbert

ROOT = Path(__file__).resolve().parents[1]
for p in ("src", "ane_kernels", "tests"):
    sys.path.insert(0, str(ROOT / p))
import fmcw  # noqa: E402
import mimo  # noqa: E402
import ane_superres_mimo as sr  # noqa: E402

MIC_NAME, SPK_NAME = "MacBook Air Microphone", "MacBook Air Speakers"
VOLUME, AMP, ALPHA = 75, 0.9, 0.15
BANK = 2                               # frames per NE call: one PRI (L + R) -> a 25 Hz pipeline
PAD = 32                               # samples around each frame for the fractional-delay cut
RESP_WINDOW = 200                      # rows (8 s) kept for breathing and presence
RESP_MIN_ROWS = 150                    # 6 s before a first breathing estimate
RESP_MAX_PP_MM = 15.0                  # larger than this is a hand / body moving, not breathing
QUIET_MARGIN_DB = 6.0
TRACK_YS = np.arange(8.0, mimo.TRACK_Y_MAX + 1e-9, 1.0)
ART = ROOT / "artifacts"
STATS_JSON = ART / "live_demo_stream_stats.json"
DEMO_MP4, DEMO_GIF = ART / "sonar_ane_demo.mp4", ART / "sonar_ane_demo.gif"
DEMO_FPS = 60

BG, SURF, INK, INK2, GRID = "#121212", "#1a1a19", "#f0efec", "#9c9b92", "#2e2e2c"
BLUE, ORANGE, AQUA, YELLOW, GREEN, RED = "#3987e5", "#ff7a3d", "#2ec4b6", "#f5c518", "#2ecc71", "#e5484d"


def osa(script):
    return subprocess.run(["osascript", "-e", script], capture_output=True, text=True).stdout.strip()


def cut_frame(buf, start, n=mimo.NC, pad=PAD):
    """Frame of n samples starting at the FRACTIONAL sample `start` of buf (frequency-domain delay)."""
    i = int(np.floor(start))
    L = n + 2 * pad
    seg = buf[i - pad:i - pad + L]
    ramp = np.exp(2j * np.pi * np.fft.rfftfreq(L) * (start - i))
    return np.fft.irfft(np.fft.rfft(seg) * ramp, L)[pad:pad + n]


# ---------------------------------------------------------------- sources
class TdmSource:
    """Duplex stream: every callback block is one stereo PRI out (L chirp @ 0 ms, R chirp @ 20 ms)."""

    def __init__(self):
        import sounddevice as sd
        self.sd = sd
        self.mic, self.spk = self._find(MIC_NAME, "input"), self._find(SPK_NAME, "output")
        c = mimo.make_chirp(ALPHA) * AMP
        self.pri = np.zeros((mimo.NP, 2), np.float32)
        self.pri[:mimo.NC, 0] = c
        self.pri[mimo.SLOT["R"]:mimo.SLOT["R"] + mimo.NC, 1] = c
        self.q = queue.Queue()
        self.dropped = 0
        self.saved = self.stream = None

    def _find(self, name, kind):
        for d in self.sd.query_devices():
            if name in d["name"] and d[f"max_{kind}_channels"] > 0:
                return d["index"]
        raise RuntimeError(f"{kind} device {name!r} not found")

    def _callback(self, indata, outdata, frames, time_info, status):
        outdata[:] = self.pri
        if status:
            self.dropped += 1
        self.q.put((time.perf_counter(), indata[:, 0].copy()))

    def start(self):
        self.saved = (osa("output volume of (get volume settings)"), osa("output muted of (get volume settings)"))
        osa(f"set volume output volume {VOLUME}")
        osa("set volume without output muted")
        self.stream = self.sd.Stream(samplerate=mimo.FS, blocksize=mimo.NP, device=(self.mic, self.spk),
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


class SimTdmSource:
    """Synthetic TDM scene in real time, no audio: measured geometry (mic 6 cm left, right direct path
    -19 dB), two desk statics, a torso at (-5, 48) cm breathing 4 mm p-p, and after a 2 s pre-roll a hand
    sweeping X -20..+20, Y 20..50 cm. 150 ms I/O latency."""

    def __init__(self, seed=0, x_mic=-6.0):
        self.q = queue.Queue()
        self.dropped = 0
        self.rng = np.random.default_rng(seed)
        self.chirp = mimo.make_chirp(ALPHA).astype(np.float64)
        self.lat = int(0.15 * mimo.FS)
        self.spk = {"L": np.array([-mimo.BASELINE_CM / 2, 0.0]), "R": np.array([mimo.BASELINE_CM / 2, 0.0])}
        self.mic = np.array([x_mic, 0.0])
        self.amp = {"L": 0.2, "R": 0.2 * 10 ** (-19 / 20)}
        self._stop = threading.Event()

    def _tx(self, ch, n):
        """Transmit pattern of channel ch at (fractional) absolute sample n."""
        x = (n - mimo.SLOT[ch]) % mimo.NP
        return np.interp(x, np.arange(mimo.NC), self.chirp, left=0.0, right=0.0) * (x < mimo.NC)

    def _run(self):
        c = fmcw.C_SOUND * 100
        k, t0 = 0, time.perf_counter()
        while not self._stop.is_set():
            n = k * mimo.NP + np.arange(mimo.NP)
            t = n / mimo.FS
            y = np.zeros(mimo.NP)
            tx_, ty_ = np.clip(-20 + 40 * ((t - 2.0) % 8) / 6, -20, 20), 35 - 15 * np.cos(2 * np.pi * (t - 2.0) / 3)
            for ch in ("L", "R"):
                d0 = np.linalg.norm(self.spk[ch] - self.mic)                 # direct path carries the geometry
                y += self.amp[ch] * self._tx(ch, n - self.lat - d0 / c * mimo.FS)
                for rw, a in ((90.0, 0.004), (154.0, 0.003)):
                    y += a * self._tx(ch, n - self.lat - (d0 + 2 * rw) / c * mimo.FS)
                torso = np.stack([np.full_like(t, -5.0), 48 + 0.2 * np.sin(2 * np.pi * 0.27 * t) + 0.1 * np.sin(2 * np.pi * 0.08 * t)], 1)
                path = np.linalg.norm(torso - self.spk[ch], axis=1) + np.linalg.norm(torso - self.mic, axis=1)
                y += 0.2 * 10 ** (-24 / 20) * self._tx(ch, n - self.lat - path / c * mimo.FS)
                hand = np.stack([tx_, ty_], 1)
                path = np.linalg.norm(hand - self.spk[ch], axis=1) + np.linalg.norm(hand - self.mic, axis=1)
                on = ((t - 2.0) % 8 < 6) & (t >= 2.0)
                y += 0.2 * 10 ** (-28 / 20) * on * self._tx(ch, n - self.lat - path / c * mimo.FS)
            y += 3e-4 * self.rng.standard_normal(mimo.NP)
            self.q.put((time.perf_counter(), y.astype(np.float32)))
            k += 1
            time.sleep(max(0.0, t0 + k * mimo.PRI - time.perf_counter()))

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._stop.set()


# ---------------------------------------------------------------- live engine
class Engine:
    """Streaming version of tests/live_phase4_mimo_superres.py + mimo.hand_presence_pipeline."""

    def __init__(self, bank=BANK, preroll_s=mimo.PREROLL_S):
        self.model, plan = sr.load_mimo(bank)
        ops = [o for o in plan if o["op"] != "const"]
        self.ne_ops, self.all_ne = len(ops), all(o["preferred"] == "NeuralEngine" for o in ops)
        self.bank, self.preroll_s = bank, preroll_s
        self.a = sr.dictionary()
        self.chirp = mimo.make_chirp(ALPHA).astype(np.float64)
        self.buf = np.zeros(0, np.float32)
        self.buf0 = 0                                  # absolute sample index of buf[0]
        self.n_in = 0
        self.l_next = self.delta = None                # fractional start of the next L chirp; R = L + delta
        self.x_mic = None
        self.t_grid = None
        self.pri = 0
        self.raw = {"L": [], "R": []}
        self.pre = {"L": [], "R": []}
        self.bg = None
        self.X = {ch: collections.deque(maxlen=RESP_WINDOW) for ch in ("L", "R")}
        self.profile = self.fft_prof = None
        self.cals = self.statics = self.ekf = None
        self.rows = 0
        self.trail = collections.deque(maxlen=90)      # (row, x, y, status) of the joint track
        self.all_xy = collections.deque(maxlen=RESP_WINDOW)
        self.seg = None                                # current track segment
        self.presence = []                             # confirmed torso statics [(x, y)]
        self.evidence = []
        self.resp = {"wave": None, "t": None, "bpm": None, "conf": None, "pp": None, "state": "warming up"}
        self.ne_ms, self.pipe_ms, self.row_times = [], [], []
        self.map = None
        self.map_axes = mimo.bistatic_grid(0.0, TRACK_YS)[:2]
        self.last_cursor = None

    # ---- alignment
    def _find_grid(self):
        env = np.abs(hilbert(correlate(self.buf.astype(np.float64), self.chirp, mode="valid", method="fft")))
        n = len(env) // mimo.NP
        folded = env[:n * mimo.NP].reshape(n, mimo.NP).max(0)
        l_off = int(np.argmax(folded))
        win = [(l_off + mimo.SLOT["R"] + d) % mimo.NP for d in range(-60, 60)]
        r_off = win[int(np.argmax(folded[win]))]
        self.delta = (r_off - l_off) % mimo.NP
        self.x_mic = mimo.mic_offset_cm(self.delta - mimo.SLOT["R"])
        first = l_off if l_off >= PAD else l_off + mimo.NP
        self.l_next = float(self.buf0 + first)
        self.t_grid = time.perf_counter()

    def _refine_grid(self):
        """Sub-sample direct-path residual of the last PRIs (clock drift / sample slips)."""
        i0 = int(self.l_next - self.buf0) - 25 * mimo.NP
        if i0 < 8:
            return
        seg = self.buf[i0 - 8:int(self.l_next - self.buf0) + 8].astype(np.float64)
        env = np.abs(hilbert(correlate(seg, self.chirp, mode="valid", method="fft")))
        n = len(env) // mimo.NP
        if n < 2:
            return
        folded = env[:n * mimo.NP].reshape(n, mimo.NP).max(0)
        k = int(np.argmax(folded))
        a, b, c = np.log(folded[k - 1:k + 2] + 1e-12)
        frac = 0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c < 0 else 0.0
        resid = ((k + frac - 8) + mimo.NP / 2) % mimo.NP - mimo.NP / 2
        if abs(resid) < 6:
            self.l_next += 0.5 * resid

    def feed(self, t_arrival, block):
        """One recorded block in; returns the number of new MTI rows."""
        self.buf = np.concatenate([self.buf, block])
        self.n_in += len(block)
        if self.l_next is None:
            if len(self.buf) >= int(1.0 * mimo.FS):
                self._find_grid()
            return 0
        new = 0
        t0 = time.perf_counter()
        while self.l_next + self.delta + mimo.NC + PAD <= self.buf0 + len(self.buf):
            s = self.l_next - self.buf0
            self.raw["L"].append(cut_frame(self.buf, s))
            self.raw["R"].append(cut_frame(self.buf, s + self.delta))
            self.l_next += mimo.NP
            self.pri += 1
            if self.pri % 25 == 0:
                self._refine_grid()
            if 2 * len(self.raw["L"]) >= self.bank:
                new += self._run_bank()
        keep = int(self.l_next - self.buf0) - 2 * PAD
        if keep > 0:
            self.buf, self.buf0 = self.buf[keep:], self.buf0 + keep
        if new:
            self.pipe_ms.append((time.perf_counter() - t0) * 1e3)
            self.row_times.append(time.perf_counter() - t_arrival)
        return new

    # ---- NE + tracking
    def _run_bank(self):
        rl, rr = np.array(self.raw["L"]), np.array(self.raw["R"])
        self.raw = {"L": [], "R": []}
        for ch, fr in (("L", rl), ("R", rr)):
            if self.bg is None:
                self.pre[ch].extend(fr)
        bg = {ch: (self.bg[ch] if self.bg is not None else (np.mean(self.pre[ch], 0) if self.pre[ch] else 0.0)) for ch in self.pre}
        raw = np.concatenate([rl, rr])
        rem = np.concatenate([rl - bg["L"], rr - bg["R"]])
        gr = 1.0 / np.abs(raw).max(axis=1, keepdims=True)
        xb, _ = sr.normalise(rem, self.a)
        t0 = time.perf_counter()
        out = self.model.predict({"xr": np.ascontiguousarray((raw * gr).T[None, :, None, :]).astype(np.float16),
                                  sr.INPUT: np.ascontiguousarray(xb.T[None, :, None, :]).astype(np.float16)})
        self.ne_ms.append((time.perf_counter() - t0) * 1e3)
        X = mimo.to_complex(out["range_fft"].reshape(mimo.C_OUT, self.bank)) * mimo.amp_scale() / gr
        prof = out[sr.OUTPUT].reshape(sr.NG, self.bank).T
        n = len(rl)
        self.profile, self.fft_prof = prof[n - 1], sr.fft_fine(rem[n - 1:n])[0]
        new = 0
        for i in range(n):
            self.X["L"].append(X[i])
            self.X["R"].append(X[n + i])
            new += self._row()
        return new

    def _row(self):
        m = len(self.X["L"])
        t = m * mimo.PRI
        if self.cals is None:
            if t >= self.preroll_s:
                self._calibrate()
            else:
                self.all_xy.append((np.nan, np.nan))
                return 0
        if m < 3:
            return 0
        row = self.rows
        self.rows += 1
        X3 = {ch: np.array(list(self.X[ch])[-3:]) for ch in self.X}
        scores = {ch: mimo.channel_scores_cal(X3[ch], self.cals[ch]) for ch in X3}
        maps, thr, gx, gy = mimo.backprojection(scores, self.x_mic, ys=TRACK_YS, r_max=62.0)
        self.map = maps[0]
        st, s, events = self.ekf.step(row, mimo.map_candidates(maps[0], thr, gx, gy))
        for ev in events:
            if ev[0] == "fill":
                for tt, xx, yy, vx, vy in ev[1]:
                    self.trail.append((tt, xx, yy, 2))
            else:
                self.trail = collections.deque([p for p in self.trail if not (ev[1] <= p[0] < ev[2])], maxlen=90)
                self.seg = None
        if st and not (events and events[0][0] == "fill"):
            self.trail.append((row, s[0], s[1], st))
        self.all_xy.append((s[0], s[1]) if st else (np.nan, np.nan))
        if st:
            if self.seg is None or self.seg["end"] < row - 1:
                r0 = self.trail[0][0] if events and events[0][0] == "fill" else row
                self.seg = {"start": r0, "x0": s[0], "y0": s[1], "exc": 0.0, "hand": False, "end": row,
                            "xs": [s[0]], "ys": [s[1]]}
            self.seg["end"] = row
            self.seg["xs"].append(s[0])
            self.seg["ys"].append(s[1])
            self.seg["exc"] = max(np.ptp(self.seg["xs"]), np.ptp(self.seg["ys"]))
            dur = (row - self.seg["start"] + 1) * mimo.PRI
            if (not self.seg["hand"] and self.seg["y0"] <= mimo.HAND_INIT_Y[1] and self.seg["exc"] >= mimo.INIT_MIN_DISP
                    and dur >= mimo.MIN_GESTURE_S):
                self.seg["hand"] = True
        else:
            self.seg = None if self.ekf.s is None else self.seg
        if row % 25 == 0:
            self._presence()
        if row % 12 == 0 and m >= RESP_MIN_ROWS:
            self._breathe()
        return 1

    def _calibrate(self):
        t = np.arange(len(self.X["L"])) * mimo.PRI
        rows = np.flatnonzero((t >= 0.2) & (t < self.preroll_s - 0.1))
        self.cals = {ch: mimo.calibrate(np.array(self.X[ch]), rows) for ch in self.X}
        self.statics = mimo.static_reflectors(self.cals, self.x_mic, ys=TRACK_YS, n=6)
        self.bg = {ch: np.mean(np.array(self.pre[ch])[rows], 0) for ch in self.pre}
        self.ekf = mimo.JointEKF(self.x_mic, ys=TRACK_YS, init_y=mimo.HAND_INIT_Y, init_min_speed=mimo.INIT_MIN_SPEED,
                                 init_min_disp=mimo.INIT_MIN_DISP, static_xy=self.statics, hold_y_max=mimo.HAND_HOLD_Y_MAX)

    def _presence(self):
        Xw = {ch: np.array(self.X[ch]) for ch in self.X}
        xy = np.array(self.all_xy)[-len(Xw["L"]):]
        jx, jy = xy[1:-1, 0], xy[1:-1, 1]
        self.evidence = mimo.torso_evidence(Xw, self.statics, self.x_mic, jx, jy)
        self.presence = [tuple(e["xy_cm"]) for e in self.evidence if e["confirmed"]]

    def _breathe(self):
        XL = np.array(self.X["L"])
        r = mimo.range_axis_cm()
        sel = (r >= 15) & (r <= 80)
        e = 20 * np.log10(np.sqrt((np.abs(fmcw.mti2(XL)[:, sel]) ** 2).mean(1)) + 1e-30)
        quiet = bool((e[-50:] < np.median(e) + QUIET_MARGIN_DB).mean() > 0.9)
        moving_hand = self.seg is not None and self.seg["hand"] and self.seg["end"] >= self.rows - 2
        band = (r >= 20) & (r <= 65)
        k = int(np.flatnonzero(band)[np.argmax(np.abs(XL[:, band]).mean(0))])
        d, _ = fmcw.breathing_displacement_mm(XL, k)
        w, est = fmcw.respiration(d, fs=1 / mimo.PRI)
        ok = quiet and not moving_hand and est["peak_to_peak_mm"] <= RESP_MAX_PP_MM and est["confidence"] != "low"
        self.resp = {"wave": w, "t": np.arange(len(w)) * mimo.PRI, "bin_cm": float(r[k]),
                     "bpm": est["bpm"] if ok else None, "conf": est["confidence"], "pp": est["peak_to_peak_mm"],
                     "state": "measuring" if ok else ("hand moving" if moving_hand else "hold still" if not quiet else "settling")}

    # ---- display state
    def state(self):
        t = len(self.X["L"]) * mimo.PRI
        if self.l_next is None:
            phase, banner, color = "warming", "WARMING UP — opening the audio stream", INK2
        elif self.cals is None:
            phase, banner, color = "calibrating", f"HOLD HANDS IN LAP — CALIBRATING ({max(self.preroll_s - t, 0):.0f}s)", YELLOW
        else:
            phase, banner, color = "tracking", "TRACKING — move a hand or a notebook in 12–45 cm · hold still for breathing", GREEN
        cursor = kind = None
        if self.ekf is not None and self.ekf.s is not None and self.trail:
            _, x, y, st = self.trail[-1]
            cursor = (x, y)
            kind = "hand" if (self.seg and self.seg["hand"]) else "candidate"
            if st == 1:
                kind = "coast"
        ne = float(np.median(self.ne_ms[-50:])) if self.ne_ms else None
        rate = (len(self.row_times) / max(1e-9, time.perf_counter() - self.t_grid)) if self.t_grid and self.row_times else None
        gx, gy = self.map_axes
        return {"t": t, "phase": phase, "banner": banner, "banner_color": color,
                "map": self.map, "gx": gx, "gy": gy, "cursor": cursor, "cursor_kind": kind,
                "trail": [(x, y, st) for _, x, y, st in self.trail], "statics": self.statics or [],
                "presence": self.presence, "evidence": self.evidence, "x_mic": self.x_mic,
                "fista": self.profile, "fft": self.fft_prof, "grid_cm": sr.GRID_CM, "profile_label": "latest left chirp",
                "ne_ms": ne, "ne_ops": self.ne_ops, "all_ne": self.all_ne, "frames_per_call": self.bank,
                "rows_per_s": rate, "resp": self.resp, "caption": None}


# ---------------------------------------------------------------- dashboard
class Dashboard:
    """One figure, three panels, driven by a state dict (live window and video renderer share it)."""

    def __init__(self, interactive=False, dpi=100):
        import matplotlib
        if not interactive:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        self.plt = plt
        if interactive:
            plt.ion()
        self.fig = plt.figure(figsize=(12.8, 7.2), dpi=dpi, facecolor=BG)
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 1], left=0.045, right=0.985,
                                   top=0.86, bottom=0.115, wspace=0.16, hspace=0.42)
        self.axA = self.fig.add_subplot(gs[:, 0])
        self.axB = self.fig.add_subplot(gs[0, 1])
        self.axC = self.fig.add_subplot(gs[1, 1])
        for ax in (self.axA, self.axB, self.axC):
            ax.set_facecolor(SURF)
            ax.tick_params(colors=INK2, labelsize=8.5)
            ax.grid(color=GRID, lw=0.6)
            for s in ax.spines.values():
                s.set_color(GRID)
        self.fig.text(0.045, 0.955, "SonarANE", color=INK, fontsize=17, fontweight="bold", va="center")
        self.fig.text(0.178, 0.955, "ultrasonic MIMO sonar on the Apple Neural Engine · M2 MacBook Air, built-in speakers + mic",
                      color=INK2, fontsize=10.5, va="center")
        self.banner = self.fig.text(0.045, 0.905, "", color=INK2, fontsize=12.5, fontweight="bold", va="center")
        self.caption = self.fig.text(0.985, 0.012, "", color=INK2, fontsize=8.5, ha="right", va="bottom")
        # A
        ax = self.axA
        self.cmap = LinearSegmentedColormap.from_list("sonar", [SURF, "#0d2b55", "#1c5fb8", "#5fa3f0", "#cfe6ff"])
        gx, gy, _ = mimo.bistatic_grid(0.0, TRACK_YS)
        self.im = ax.imshow(np.zeros(gx.shape), origin="lower", aspect="auto", cmap=self.cmap, vmin=0, vmax=1,
                            extent=[gx[0, 0] - 0.5, gx[0, -1] + 0.5, gy[0, 0] - 0.5, gy[-1, 0] + 0.5], zorder=1)
        self.trail_sc = ax.scatter([], [], s=14, zorder=3)
        self.glow = [ax.scatter([], [], s=sz, color=ORANGE, alpha=al, lw=0, zorder=4) for sz, al in ((1500, 0.10), (600, 0.25), (160, 0.9))]
        self.cursor_edge = ax.scatter([], [], s=160, facecolors="none", edgecolors=INK, lw=1.2, zorder=5)
        self.statics_sc = ax.scatter([], [], marker="x", s=50, color=INK2, lw=1.5, zorder=3)
        self.pres_ring = ax.scatter([], [], s=400, facecolors="none", edgecolors=AQUA, lw=2.2, zorder=4)
        self.pres_ring2 = ax.scatter([], [], s=900, facecolors="none", edgecolors=AQUA, lw=1.0, alpha=0.5, zorder=4)
        self.pres_txt = ax.text(0, 0, "", color=AQUA, fontsize=9, fontweight="bold", ha="center", va="top", zorder=6)
        ax.axhline(mimo.HAND_INIT_Y[1], color=YELLOW, lw=0.8, ls=":", alpha=0.7)
        ax.text(-39, mimo.HAND_INIT_Y[1] + 0.6, "hand tracks start ≤ 45 cm", color=YELLOW, fontsize=7.5, alpha=0.8)
        ax.plot([-mimo.BASELINE_CM / 2, mimo.BASELINE_CM / 2], [5.4, 5.4], "s", color=BLUE, ms=9, clip_on=False, zorder=6)
        self.mic_pt, = ax.plot([-6], [5.4], "^", color=ORANGE, ms=9, clip_on=False, zorder=6)
        self.hand_txt = ax.text(0.02, 0.975, "", transform=ax.transAxes, color=INK, fontsize=11, family="monospace", va="top", zorder=7)
        self.sub_txt = ax.text(0.02, 0.03, "", transform=ax.transAxes, color=INK2, fontsize=8, va="bottom", zorder=7)
        ax.set_xlim(-40, 40)
        ax.set_ylim(5, 55)
        ax.set_xlabel("X (cm)   ■ speakers   ▲ mic", color=INK2, fontsize=9)
        ax.set_ylabel("Y (cm, distance from the screen)", color=INK2, fontsize=9)
        ax.set_title("A · live 2D map — joint stereo backprojection + Cartesian EKF", color=INK, fontsize=10.5, loc="left")
        # B
        ax = self.axB
        self.fft_line, = ax.plot([], [], color=INK2, lw=1.2, label="classical range FFT (Hann)")
        self.fista_fill = ax.fill_between([], [], color=ORANGE, alpha=0.35)
        self.fista_line, = ax.plot([], [], color=ORANGE, lw=1.6, label="NE FISTA ×80 (0.6 cm grid)")
        self.hand_r = ax.axvline(-10, color=YELLOW, lw=1, ls="--", alpha=0.8)
        self.ne_txt = ax.text(0.99, 0.95, "", transform=ax.transAxes, color=INK, fontsize=9.5, family="monospace", ha="right", va="top")
        ax.set_xlim(5, 90)
        ax.set_ylim(0, 1.08)
        ax.set_xlabel("bistatic range beyond the direct path (cm)", color=INK2, fontsize=9)
        ax.set_ylabel("normalised", color=INK2, fontsize=9)
        ax.set_title("B · Neural Engine super-resolution vs FFT", color=INK, fontsize=10.5, loc="left")
        ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK2)
        # C
        ax = self.axC
        self.wave_line, = ax.plot([], [], color=BLUE, lw=1.8)
        self.bpm_txt = ax.text(0.99, 0.94, "", transform=ax.transAxes, color=INK, fontsize=22, fontweight="bold", ha="right", va="top")
        self.resp_txt = ax.text(0.99, 0.62, "", transform=ax.transAxes, color=INK2, fontsize=9, ha="right", va="top")
        self.resp_state = ax.text(0.5, 0.5, "", transform=ax.transAxes, color=INK2, fontsize=12, ha="center", va="center")
        ax.set_xlabel("time (s)", color=INK2, fontsize=9)
        ax.set_ylabel("chest displacement (mm)", color=INK2, fontsize=9)
        ax.set_ylim(-5, 5)
        ax.set_xlim(0, 20)
        ax.set_title("C · contactless respiration — phase-referenced interferometry", color=INK, fontsize=10.5, loc="left")
        self.closed = False
        self.fig.canvas.mpl_connect("close_event", lambda e: setattr(self, "closed", True))

    def draw(self, st):
        self.banner.set_text(st["banner"])
        self.banner.set_color(st["banner_color"])
        self.caption.set_text(st.get("caption") or "")
        # A
        if st["map"] is not None:
            m = 20 * np.log10(st["map"] + 1e-6)
            lo, hi = np.percentile(m, 5), max(np.percentile(m, 99.5), np.percentile(m, 5) + 6)
            self.im.set_data(np.clip((m - lo) / (hi - lo), 0, 1))
        else:
            self.im.set_data(np.zeros(self.im.get_array().shape))
        if st.get("x_mic") is not None:
            self.mic_pt.set_data([st["x_mic"]], [5.4])
        tr = [p for p in st["trail"] if p[2] > 0]
        if tr:
            n = len(tr)
            cols = np.zeros((n, 4))
            cols[:, :3] = (1.0, 0.48, 0.24)
            cols[:, 3] = np.linspace(0.08, 0.8, n) ** 1.3
            self.trail_sc.set_offsets([(x, y) for x, y, _ in tr])
            self.trail_sc.set_facecolors(cols)
            self.trail_sc.set_sizes(np.linspace(6, 28, n))
        else:
            self.trail_sc.set_offsets(np.zeros((0, 2)))
        cur = st["cursor"]
        if cur is not None:
            kind = st["cursor_kind"]
            col = {"hand": ORANGE, "candidate": INK2, "coast": YELLOW}[kind]
            for g in self.glow:
                g.set_offsets([cur])
                g.set_color(col)
            self.cursor_edge.set_offsets([cur])
            label = {"hand": "HAND", "candidate": "track (not yet a gesture)", "coast": "HAND (coasting)"}[kind]
            self.hand_txt.set_text(f"{label}\nX {cur[0]:+6.1f} cm\nY {cur[1]:6.1f} cm")
        else:
            for g in self.glow:
                g.set_offsets(np.zeros((0, 2)))
            self.cursor_edge.set_offsets(np.zeros((0, 2)))
            self.hand_txt.set_text("no hand" if st["phase"] == "tracking" else "")
        self.statics_sc.set_offsets(np.array(st["statics"]) if st["statics"] else np.zeros((0, 2)))
        if st["presence"]:
            pulse = 1 + 0.25 * np.sin(2 * np.pi * 0.3 * st["t"])
            pts = np.array(st["presence"])
            self.pres_ring.set_offsets(pts)
            self.pres_ring.set_sizes(np.full(len(pts), 400 * pulse))
            self.pres_ring2.set_offsets(pts)
            self.pres_ring2.set_sizes(np.full(len(pts), 1000 * pulse))
            px, py = pts.mean(0)
            self.pres_txt.set_position((px, py - 3.6))
            self.pres_txt.set_text("SEATED PRESENCE · breathing micro-motion")
        else:
            self.pres_ring.set_offsets(np.zeros((0, 2)))
            self.pres_ring2.set_offsets(np.zeros((0, 2)))
            self.pres_txt.set_text("")
        self.sub_txt.set_text(st.get("sub_a") or "× pre-roll static reflectors · map: geometric mean of both channels' CFAR evidence")
        # B
        g = st["grid_cm"]
        if st["fft"] is not None:
            sel = g >= 5
            f = st["fft"][sel] / max(st["fft"][sel].max(), 1e-12)
            self.fft_line.set_data(g[sel], f)
        if st["fista"] is not None:
            sel = g >= 5
            p = st["fista"][sel] / max(st["fista"][sel].max(), 1e-12)
            self.fista_line.set_data(g[sel], p)
            self.fista_fill.remove()
            self.fista_fill = self.axB.fill_between(g[sel], 0, p, color=ORANGE, alpha=0.3, lw=0)
        else:
            self.fista_line.set_data([], [])
            self.fista_fill.remove()
            self.fista_fill = self.axB.fill_between([], [], color=ORANGE, alpha=0.3)
        rl = -10
        if cur is not None and st.get("x_mic") is not None:
            rl = mimo.bistatic(cur[0], cur[1], st["x_mic"])[0]["L"]
        self.hand_r.set_xdata([rl, rl])
        ne = st["ne_ms"]
        lines = [f"NE {ne:.2f} ms / call · {st['frames_per_call']} frames" if ne is not None else "NE warming up",
                 f"{st['ne_ops']} ops, all on NeuralEngine: {st['all_ne']}"]
        if st.get("rows_per_s"):
            lines.append(f"pipeline {st['rows_per_s']:.1f} rows/s (PRF 25 Hz)")
        if st.get("ne_extra"):
            lines.append(st["ne_extra"])
        self.ne_txt.set_text("\n".join(lines))
        # C
        r = st["resp"]
        if r and r.get("wave") is not None and len(r["wave"]) and r["state"] in ("measuring", "settling"):
            w, t = r["wave"], r["t"]
            self.wave_line.set_data(t, w)
            self.axC.set_xlim(max(0, t[-1] - 20), max(20, t[-1]))
            lim = max(2.5, float(np.abs(w).max()) * 1.25)
            self.axC.set_ylim(-lim, lim)
        else:
            self.wave_line.set_data([], [])
        if r and r.get("bpm") is not None:
            self.bpm_txt.set_text(f"{r['bpm']:.1f} /min")
            self.bpm_txt.set_color({"high": GREEN, "ok": INK}.get(str(r["conf"]).split(":")[0], INK2))
            self.resp_txt.set_text(f"{r['conf']} confidence · {r['pp']:.1f} mm p-p · chest bin {r.get('bin_cm', 0):.0f} cm")
            self.resp_state.set_text("")
        else:
            self.bpm_txt.set_text("—")
            self.resp_txt.set_text("")
            self.resp_state.set_text({"hold still": "HOLD STILL to measure breathing", "hand moving": "hand moving — hold still for breathing",
                                      "settling": "settling…", "warming up": "collecting 6 s of chirps…"}.get(r["state"] if r else "warming up", ""))

    def frame(self):
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def flush(self):
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


# ---------------------------------------------------------------- demo video from the saved captures
def demo_sweep_states():
    """Capture 1: artifacts/phase4_capture.wav through the exact offline pipeline (what the telemetry reports)."""
    import live_phase4_mimo_superres as p4
    from scipy.io import wavfile
    _, _, wav, meta_path = p4.paths("phase4")
    meta = json.loads(meta_path.read_text())
    _, rec = wavfile.read(wav)
    rec = rec.astype(np.float64)
    starts, q = p4.align_tdm(rec, meta["tx_start_in"], meta["lead"], meta["n_pri"])
    m = min(len(starts["L"]), len(starts["R"]))
    raw = {ch: p4.frames_at(rec, starts[ch][:m]) for ch in ("L", "R")}
    pre = (np.arange(m) * mimo.PRI >= 0.2) & (np.arange(m) * mimo.PRI < meta["preroll_s"] - 0.1)
    bg = {ch: raw[ch] - raw[ch][pre].mean(0) for ch in raw}
    model, plan = sr.load_mimo()
    ops = [o for o in plan if o["op"] != "const"]
    X_all, P_all, call_ms = sr.run_mimo(model, np.concatenate([raw["L"], raw["R"]]), np.concatenate([bg["L"], bg["R"]]))
    X, P = {"L": X_all[:m], "R": X_all[m:]}, {"L": P_all[:m], "R": P_all[m:]}
    fft = sr.fft_fine(bg["L"])
    x_mic = mimo.mic_offset_cm(q["r_minus_l_direct_samples"])
    hp = mimo.hand_presence_pipeline(X, x_mic, meta["preroll_s"])
    ne_us = np.median(call_ms[1:]) * 1e3 / sr.S
    states = []
    confirmed = [tuple(e["xy_cm"]) for e in hp["torso_evidence"] if e["confirmed"]]
    hand, allt = hp["hand"], hp["all_tracks"]
    trail = []
    for i, t in enumerate(hp["rows_t"]):
        pre_roll = t < meta["preroll_s"]
        st = int(allt["status"][i])
        cur = kind = None
        if st and not pre_roll:
            cur = (float(allt["x"][i]), float(allt["y"][i]))
            kind = "hand" if hand["status"][i] > 0 else "candidate"
            if hand["status"][i] == 1:
                kind = "coast"
            trail.append((cur[0], cur[1], st))
        elif st == 0:
            trail.append((np.nan, np.nan, 0))
        trail = trail[-90:]
        states.append({
            "t": t, "phase": "calibrating" if pre_roll else "tracking",
            "banner": (f"HOLD HANDS IN LAP — CALIBRATING ({max(meta['preroll_s'] - t, 0):.0f}s)" if pre_roll
                       else "SWEEP NOTEBOOK NOW — left ↔ right, in ↔ out (12–45 cm)"),
            "banner_color": YELLOW if pre_roll else GREEN,
            "map": hp["maps"][i] if not pre_roll else None, "gx": hp["gx"], "gy": hp["gy"],
            "cursor": cur, "cursor_kind": kind, "trail": [p for p in trail if p[2] > 0],
            "statics": hp["statics"] if not pre_roll else [], "presence": confirmed if not pre_roll else [],
            "x_mic": x_mic, "fista": P["L"][i + 1], "fft": fft[i + 1], "grid_cm": sr.GRID_CM,
            "ne_ms": float(np.median(call_ms[1:])), "ne_ops": len(ops),
            "all_ne": all(o["preferred"] == "NeuralEngine" for o in ops), "frames_per_call": sr.S,
            "ne_extra": f"= {ne_us:.0f} µs per chirp frame", "rows_per_s": None,
            "resp": {"wave": None, "t": None, "bpm": None, "state": "hand moving" if cur else "warming up"},
            "caption": f"capture 1 · {meta['captured_at'][:16].replace('T', ' ')} · silent library · 2 s pre-roll + 8 s notebook sweep · replayed through the Phase 4d pipeline"})
    summary = hp["summary"]
    return states, summary


def demo_breath_states():
    """Capture 2: artifacts/respiration_capture.wav (30 s, continuous chirps, book on chest)."""
    import live_phase4_mimo_superres as p4
    resp, X = p4.saved_respiration()
    tel = json.loads((ART / "respiration_telemetry.json").read_text())
    r_cm = fmcw.range_axis_m() * 100
    k = int(np.argmin(np.abs(r_cm - resp["stats"]["bin_cm"])))
    t_rows, wave = resp["t"], resp["wave"]
    t0 = t_rows[0]
    states = []
    bpm = conf = pp = None
    g = np.arange(0.0, 90.0 + 1e-9, 0.6)
    for i in range(len(t_rows)):
        t = t_rows[i] - t0
        if i % 12 == 0 and t >= 10.0:
            d = resp["d_ref"][:i + 1]
            _, est = fmcw.respiration(d)
            bpm, conf, pp = est["bpm"], est["confidence"], est["peak_to_peak_mm"]
        row = int(round(t_rows[i] / fmcw.T_CHIRP))
        prof = np.abs(X[min(row, len(X) - 1)])
        prof = np.interp(g, r_cm, prof / prof[(r_cm >= 5) & (r_cm <= 90)].max())
        states.append({
            "t": t, "phase": "tracking", "banner": "HOLD STILL — measuring breathing (book pressed on the chest)",
            "banner_color": AQUA, "map": None, "gx": None, "gy": None, "cursor": None, "cursor_kind": None,
            "trail": [], "statics": [], "presence": [(0.0, resp["stats"]["bin_cm"])], "x_mic": None,
            "sub_a": "capture 2 is a single-speaker continuous-chirp recording: range 22 cm measured, X assumed 0",
            "fista": None, "fft": prof, "grid_cm": g, "ne_ms": tel["ne_range_conv"]["bank_total_ms"] / 64,
            "ne_ops": 1, "all_ne": tel["ne_range_conv"]["bank_conv_device"] == "NeuralEngine", "frames_per_call": 1,
            "ne_extra": "FISTA needs TDM frames: FFT only here", "rows_per_s": None,
            "resp": {"wave": wave[:i + 1], "t": t_rows[:i + 1] - t0, "bpm": bpm, "conf": conf, "pp": pp,
                     "bin_cm": resp["stats"]["bin_cm"], "state": "settling" if t < 10 else "measuring"},
            "caption": f"capture 2 · {tel['timestamp'][:16].replace('T', ' ')} · 30 s seated, breathing normally · readout = causal estimate on the data so far "
                       f"(offline three-way consensus {resp['stats']['bpm']:.1f}/min, {resp['stats']['confidence']})"})
    final = resp["stats"]
    for s in states[-int(1.0 / fmcw.T_CHIRP):]:
        s["resp"].update(bpm=final["bpm"], conf=final["confidence"], pp=final["peak_to_peak_mm"])
    return states, final


def render_demo(fps=DEMO_FPS, gif=True):
    import imageio.v2 as imageio
    from PIL import Image
    t_start = time.perf_counter()
    print("[demo] capture 1: TDM sweep through the NE pipeline…", flush=True)
    s1, sum1 = demo_sweep_states()
    print(f"[demo]   {len(s1)} rows, hand tracks {sum1['hand_tracks']}, coverage {sum1['hand_coverage']:.0%}, "
          f"seated presence {sum1['seated_presence_detected']}", flush=True)
    print("[demo] capture 2: respiration…", flush=True)
    s2, sum2 = demo_breath_states()
    print(f"[demo]   {len(s2)} rows, {sum2['bpm']:.1f}/min ({sum2['confidence']})", flush=True)
    dash = Dashboard(interactive=False)
    segments = [(s1, mimo.PRI), (s2, fmcw.T_CHIRP)]
    writer = imageio.get_writer(str(DEMO_MP4), fps=fps, codec="libx264", quality=8, pixelformat="yuv420p",
                                macro_block_size=1)
    gif_frames, n_frames = [], 0
    try:
        for seg_i, (states, dt) in enumerate(segments):
            dur = len(states) * dt
            n = int(dur * fps)
            for k in range(n):
                tf = k / fps
                i = min(int(tf / dt), len(states) - 1)
                st = dict(states[i])
                st["t"] = tf
                # interpolate the cursor between rows for a smooth 60 fps trace
                if st["cursor"] is not None and i + 1 < len(states) and states[i + 1]["cursor"] is not None:
                    a = (tf - i * dt) / dt
                    c0, c1 = states[i]["cursor"], states[i + 1]["cursor"]
                    st["cursor"] = (c0[0] + a * (c1[0] - c0[0]), c0[1] + a * (c1[1] - c0[1]))
                dash.draw(st)
                fr = dash.frame()
                writer.append_data(fr)
                n_frames += 1
                if gif and seg_i == 0 and 1.0 <= tf <= 10.0 and k % 6 == 0:
                    gif_frames.append(np.asarray(Image.fromarray(fr).resize((640, 360), Image.LANCZOS)))
                if n_frames % 300 == 0:
                    print(f"[demo]   {n_frames} frames ({n_frames / fps:.1f} s of video) in {time.perf_counter() - t_start:.0f} s", flush=True)
    finally:
        writer.close()
    if gif and gif_frames:
        imageio.mimwrite(str(DEMO_GIF), gif_frames, duration=100, loop=0, subrectangles=True)
    info = {"mp4": str(DEMO_MP4), "gif": str(DEMO_GIF) if gif else None, "fps": fps, "frames": n_frames,
            "seconds": n_frames / fps, "render_s": round(time.perf_counter() - t_start, 1),
            "notes": "capture 1 cursor interpolated between 25 Hz rows; capture 2 breaths/min is a causal estimate "
                     "on the data so far (wanders 9-18/min before ~22 s, then settles on the consensus)",
            "capture_1": {k: sum1[k] for k in ("hand_tracks", "hand_rows", "rows_after_preroll", "hand_coverage",
                                                "seated_presence_detected", "seated_presence_static_xy_cm")},
            "capture_2": {k: sum2[k] for k in ("bpm", "confidence", "peak_to_peak_mm", "cycles_observed", "bin_cm")}}
    (ART / "sonar_ane_demo.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=1))
    return info


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sim", action="store_true", help="synthetic TDM scene, no audio")
    ap.add_argument("--headless", type=float, default=None, metavar="SECONDS",
                    help="no window; run this long and write artifacts/live_demo_stream_stats.json")
    ap.add_argument("--demo-from-wav", action="store_true",
                    help="render artifacts/sonar_ane_demo.mp4 (+ .gif) from the saved captures, no audio")
    ap.add_argument("--fps", type=int, default=DEMO_FPS)
    ap.add_argument("--snapshot", type=Path, default=None, help="headless: save the final dashboard frame as PNG")
    args = ap.parse_args()
    if args.demo_from_wav:
        render_demo(args.fps)
        return
    engine = Engine()
    print(f"[live] NE graph: {engine.ne_ops} ops, all on NE: {engine.all_ne}, {engine.bank} frames per call", flush=True)
    src = SimTdmSource() if args.sim else TdmSource()
    view = None if args.headless else Dashboard(interactive=True)
    if view is not None:
        view.fig.canvas.manager.set_window_title("SonarANE — live (silent)")
    t_start = time.perf_counter()
    last_draw, draws = 0.0, 0
    last_banner = None
    try:
        src.start()
        while True:
            try:
                t_arr, block = src.q.get(timeout=1.0)
            except queue.Empty:
                continue
            new = engine.feed(t_arr, block)
            st = engine.state()
            if st["banner"] != last_banner:
                print(f"===== {st['banner']} =====", flush=True)
                last_banner = st["banner"]
            now = time.perf_counter()
            if view is not None and src.q.qsize() < 2 and now - last_draw >= 1 / 25:
                view.draw(st)
                view.flush()
                last_draw, draws = now, draws + 1
                if view.closed:
                    break
            if args.headless and now - t_start >= args.headless:
                break
    except KeyboardInterrupt:
        pass
    finally:
        src.stop()
    elapsed = time.perf_counter() - t_start
    st = engine.state()
    stats = {"source": "sim" if args.sim else "audio", "seconds": round(elapsed, 2), "pri": engine.pri,
             "rows": engine.rows, "rows_per_s": st["rows_per_s"], "display_draws": draws,
             "display_fps": draws / max(1e-9, elapsed), "dropped_blocks": src.dropped,
             "x_mic_cm": engine.x_mic, "ne": {"ops": engine.ne_ops, "all_on_ne": engine.all_ne, "frames_per_call": engine.bank,
                                             "call_ms_p50": float(np.median(engine.ne_ms)) if engine.ne_ms else None,
                                             "call_ms_p95": float(np.percentile(engine.ne_ms, 95)) if engine.ne_ms else None},
             "pipeline_ms_per_block_p50": float(np.median(engine.pipe_ms)) if engine.pipe_ms else None,
             "block_to_row_latency_ms_p50": float(np.median(engine.row_times) * 1e3) if engine.row_times else None,
             "block_to_row_latency_ms_p95": float(np.percentile(engine.row_times, 95) * 1e3) if engine.row_times else None,
             "statics_xy_cm": engine.statics, "presence_xy_cm": engine.presence, "presence_evidence": engine.evidence,
             "last_cursor": st["cursor"], "last_cursor_kind": st["cursor_kind"],
             "last_bpm": engine.resp.get("bpm"), "last_resp_state": engine.resp.get("state")}
    if args.headless:
        STATS_JSON.write_text(json.dumps(stats, indent=2, default=float))
        if args.snapshot:
            from PIL import Image
            d = Dashboard(interactive=False)
            d.draw(st)
            Image.fromarray(d.frame()).save(args.snapshot)
    print(json.dumps(stats, indent=1, default=float))


if __name__ == "__main__":
    main()
