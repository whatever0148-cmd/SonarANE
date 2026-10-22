---
name: ultrasonic-fmcw-radar
description: Math and code conventions for SonarANE's 48 kHz ultrasonic FMCW sonar — chirp design, dechirp + range-DFT as a CoreML FP16 1x1 conv on the Neural Engine, frame alignment, clutter subtraction, range-Doppler coupling and micro-Doppler. Use when writing, changing, benchmarking or debugging anything in src/fmcw.py, ane_kernels/ or tests/.
---

# Ultrasonic FMCW sonar on the Neural Engine (public CoreML only)

## 1. Waveform
| Symbol | Value | Notes |
|---|---|---|
| f_s | 48 000 Hz | built-in speaker + mic native rate |
| f_0 → f_1 | 18 800 → 22 300 Hz | B = 3500 Hz, f_c = 20 550 Hz, λ_c = 1.67 cm (Phase 1: 18.5–22, 1b: 17.2–20.7 kHz) |
| T | 40 ms | N = f_s·T = 1920 samples, PRF 25 Hz |
| slope μ | B/T = 87 500 Hz/s | |
| chirps | 150 → 6.0 s | continuous train, no gaps |
| tx window | Tukey α 0.15, amp 0.9, volume 75 % | retries: clip → amp ×0.5, weak → α−0.05 & amp 1.0, misaligned → α+0.05 |

Phase: `φ(n) = 2π (f0·t + ½ μ t²)`, `t = n/f_s`. Tx: `s[n] = A·w_tukey[n]·cos φ(n)`, tiled 150×,
with a 0.1 s stream-start guard and a 0.6 s silent tail: I/O latency (~0.1 s), 150 ms reverb guard, then
≈ 0.34 s of noise-floor reference. The spoken cue ends right before `playrec()` — no silent wait.

## 2. Dechirp + range-DFT = one linear map
An echo with delay τ is `cos φ(t−τ)`. Mixing with `e^{+jφ(t)}` gives
`½ e^{j(φ(t)−φ(t−τ))} = ½ e^{j2π(μτ t + f0 τ − ½ μ τ²)}` → a tone at the **beat** `f_b = μ τ`,
plus a sum-frequency image at +(37.6…44.6) kHz that aliases to −(10.4…3.4) kHz, far from the bins we keep.
The DFT bin spacing is 1/T = 25 Hz, so:

```
f_b = k_step · k / T,   τ = f_b / μ = k_step·k / B,   R = c τ / 2 = k_step·k · c/(2B)
```
Native resolution `c/(2B) = 4.90 cm`; we zero-pad 2× (`k_step = 0.5`) → 2.45 cm grid, K = 128 bins
→ 0 … 3.11 m. Range is measured **beyond the direct speaker→mic path** (frames are aligned on it).

Because mixing by a fixed reference and a windowed DFT are both linear in x:
```
X_k = Σ_n  h[n] · e^{+jφ(n)} · e^{−j2π k k_step n / N} · x[n]  =  (M x)_k
W = [Re M ; Im M]  ∈ R^{2K × N} = [256 × 1920]      (fmcw.dechirp_dft_kernel)
```
h = Hann. Weights stay un-normalised (|w| ≤ 1) so FP16 partial sums stay in the normal range;
multiply |X| by `fmcw.amp_scale() = 2/Σh` to read echo amplitude in input units.
Only in-band energy (≈17.2–22.3 kHz) can land in bins 0…K, so the kernel is also the band-pass.

## 3. CoreML MIL FP16 conv pipeline
```python
from coremltools.converters.mil import Builder as mb
@mb.program(input_specs=[mb.TensorSpec(shape=(1, N, 1, S), dtype=types.fp16)],
            opset_version=ct.target.macOS15)
def prog(x):
    return mb.conv(x=x, weight=W16[:, :, None, None], pad_type="valid", name="range_fft")
m = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
               compute_precision=ct.precision.FLOAT16)
```
- Layout `[1, C, 1, S]`: channels = the 1920 time samples of one chirp, the spatial axis S = independent
  chirps (a Range-Doppler bank). A 1x1 conv is a dense matmul over channels for every S column.
- Compile once (`ct.models.utils.compile_model`) and load `CompiledMLModel(path, compute_units=...)`.
- **Always check placement** with `MLComputePlan` — `CPU_AND_NE` is a permission, not an order. On the
  M2 the planner sent S ∈ {8, 64, 256…2048} to the NE and S ∈ {1, 32, 96, 128} to the CPU. There is no
  public API to force the NE; pick an S the planner accepts (S=64 for the live bank, S=8 for single-chirp NE latency).
- Input scaling: divide **each frame** by its own max before the fp16 cast (the conv is linear per column),
  undo per column afterwards. Raw and MTI frames go through the NE together in one S=64 bank pass.
- Models carry `user_defined_metadata["kernel_tag"]` = band + weight hash; rebuild when it mismatches.
- NE accuracy: rel. error ≈ 2e-4 vs fp64. CoreML's CPU fp16 path was ≈ 2.7e-2 (much worse).

## 4. Capture & alignment
- `sd.playrec(tx_stereo, fs, channels=1, device=(mic_idx, spk_idx))`, devices found by name.
- Matched filter: `env = |hilbert(correlate(rec, chirp, 'valid'))|`. The first peak > 0.5·max after the
  lead-in is the direct path of chirp 0 → I/O latency. Then refine every chirp's peak within ±48 samples,
  fit `start_m = a + b·m`: `b − N` gives the clock drift (ppm), residual RMS gives jitter.
- Quality gates (else retry with the next Tukey α / amplitude): band SNR ≥ 10 dB, peak-to-median ≥ 15 dB,
  jitter ≤ 4 samples, all frames in bounds.
- Band SNR = `10 log10((P_band,chirp − P_band,silence) / P_band,silence)`, Welch PSD, 18.5–22 kHz.

## 5. Clutter, waterfall, velocity
- Clutter canceller: **second-order MTI `y_m = x_m − 2x_{m−1} + x_{m−2}` on the time-domain frames, before
  the NE range-FFT** (`fmcw.mti2`). Linear, so identical to doing it after the FFT, but it removes the direct
  path before the FP16 cast. Response `4 sin²(π f_d / PRF)`: −12 dB at 2 Hz Doppler, −28 dB at 1 cm/s, so a
  hand is invisible at its turnarounds (tracks come in segments). White-noise amplitude gain √6.
  Compared on the Phase-1 capture: MTI2 0 cm residual −51 dB / 15–80 cm floor −62 dB, adjacent difference
  −50/−62, global or local LS template + per-chirp gain −35…−44 / −59…−61. The real direct path drifts slowly,
  it doesn't jitter from chirp to chirp, so a slow-time high-pass beats a fitted template.
- Static profile: mean_m |X_m|. Dynamic profile and waterfall: |FFT(mti2(frames))|, 0–150 cm, −40 dB floor.
- Tracking: |Y|² smoothed over 5 grid bins (one native mainlobe) so the multi-lobe MTI response becomes one
  hump; argmax of the CFAR ratio in 10–120 cm; parabolic sub-bin interpolation.
- Detection: slow-time OS-CFAR (each bin's floor = its 25th percentile over the capture), 11 dB, then a
  **continuity gate**: keep only runs of ≥ 4 consecutive chirps each within ±10 cm of the previous one.
  Simulated no-hand scenes: 0.7–3.4 % raw hits, 0 confirmed tracks. Do not use far bins as the noise
  reference: real 2–3 m returns decorrelate chirp to chirp.
- **Range-Doppler coupling** (up-chirp only): a target moving at v adds Doppler `2 v f_c / c` to the beat,
  so `R_meas = R + κ·dR/dt`, `κ = f_c/μ = 0.235 s`. Correct with the exact inverse
  `R[n] = R[n−1] + T/(κ+T) · (R_meas[n] − R[n−1])` (sign verified in simulation).
- Range rate = Savitzky–Golay derivative (11 pts) of the corrected track. Micro-Doppler shift
  `f_d = 2 v f_c / c` → 1 cm/s ≙ 1.20 Hz.
- Phase micro-Doppler `v = Δφ c / (4π f_c T)` wraps beyond ±c/(4 f_c T) = ±10.4 cm/s at PRF 25 Hz, so hand
  gestures alias it; it is only meaningful for slow / sub-cm motion.
- Simulated accuracy (palm 18–42 cm at 0.6 Hz, −33 dB re direct path): range RMS 3.6 cm, velocity RMS 13 cm/s.
  The earlier mean-subtraction tracker reached 1.8 cm but cancelled the real direct path 16 dB worse.

## 5b. Phase 2 tracker — dual path + Kalman (`fmcw.dual_path_scores`, `candidates`, `kalman_track`)
- **Path A (motion):** OS-CFAR ratio of range-smoothed |MTI2|. Blind near zero velocity (turnarounds, pauses).
- **Path B (presence):** |X_m − background_m|. Background = per-bin complex *trimmed* mean (drop the 30 % of
  chirps that deviate most, iterate 3×) so a pausing hand doesn't end up in its own background, plus a
  per-chirp **direct-path lock**: least-squares fit, on bins < 10 cm, of the exact direct-path range response
  and its 1st/2nd delay derivatives (`direct_path_basis`), subtracted everywhere. This cancels gain/phase
  *and* sub-sample delay drift, including the sidelobes (sim: −47 → −76 dB at 12–22 cm). Locking with B0 itself
  (scaling the whole scene) mis-scaled real reflectors and made false tracks. Path B is ignored below 15 cm.
- Score = max(A, B); candidates = local maxima > 11 dB in 10–120 cm, sub-bin interpolated.
- **Kalman:** CV state [r cm, v cm/s], measurement `z = r + κ·v` (range-Doppler coupling inside H), σ_a 400 cm/s²,
  σ_z 2.5 cm, gate 3σ and |ν| ≤ 15 cm. Among in-gate candidates take the **nearest** (palm before forearm).
  Coast ≤ 15 chirps on pure CV (velocity decay 0.8 cut Phase 1b coverage 85 → 71 % by losing a real
  re-acquisition), then drop and erase the unconfirmed coast. Initiation is multi-hypothesis: any chain of 4
  candidates with ≤ 10 cm steps; the nearest chain wins.
- Report coverage as updated + coasted separately; coasting is prediction, not detection.
- Validation (`tests/tracker_validation.py`): simulated palm 15→55→15 cm with a 0.8 s pause and a forearm
  12 cm behind: 100 % coverage, 100 % updated, pause fully updated, range RMS 2.0–4.2 cm, no forearm jumps;
  3 no-hand scenes with drifting direct path: 0 false tracks.

## 5d. Phase 3 — range-adaptive CFAR, fractional framing, respiration
- **TGC CFAR** (`tgc_threshold_db`): 11 dB up to 35 cm, raised-cosine to 8 dB at 75 cm. A per-bin gain would
  cancel inside the CFAR ratio, so the threshold is the knob. Lowering it alone gave 4 false tracks in 10
  no-hand scenes; chains with mean range > 35 cm must now persist **6** chirps (near: 4) → 0 / 10 again.
  On the Phase 2 capture this recovered nothing in 0.4–2.4 s: the 7–14 dB returns there never stayed within
  ±10 cm for 6 chirps (noise-only scenes reach ≥ 8 dB in 29–35 % of chirps in 35–85 cm). The lower far
  threshold helps a *locked* track follow a hand outward (Phase 3 sweep to 71 cm), not start one on noise.
- Kalman σ_a 600 cm/s², coast velocity decay 0.9 (pure CV coasting drove a held book 25 → 54 cm onto another
  scatterer in Phase 3; the Phase 1b replay preferred pure CV, 85 vs 71 % at decay 0.8).
- **Fractional framing**: always cut frames with `fractional_frames(rec, a + b*m)`; never round starts.
- **Respiration** (`displacement_mm`, `breathing_bin`, `respiration`): λ = 16.7 mm at 20.55 kHz, 1 mm = 0.753 rad.
  Pick the bin with the most respiratory-band *complex* echo power in 25–55 cm (phase of weak bins is noise).
  Reference every chirp to the 0 cm direct-path phase: removes common frame-timing error. In simulation
  (5.2 ppm drift + integer rounding + jitter) static-wall jitter 1.14 → 0.07 mm; on real data, *after*
  fractional framing, only 0.28–0.39 → 0.18–0.20 mm (most of the earlier "gain" was the rounding artifact).
  Arc-centre (Kasa circle) demodulation only when the samples really lie on an arc (span ≥ 1 rad and radial
  residual < 25 % of radius); a static noisy cluster also "fits" a circle and gives garbage phase.
  Measure the noise floor on the most phase-*coherent* far reflector, not the strongest (the 2.45 m wall return
  decorrelates). Rate needs ≥ 2 cycles: a 4 s hold gives ≤ 1–2 breaths → 'low' confidence; use ≥ 15–20 s.
  Windows < 2/0.12 s cannot carry the 0.12 Hz high-pass → detrend + 0.6 Hz low-pass.
- **Live app**: `sd.Stream` duplex, blocksize = N (one chirp per callback, phase-continuous). 1 s matched-filter
  calibration of the direct-path offset mod N, drift re-check every 125 frames. Per chirp: S=8 NE bank (newest
  column), tracker + breathing re-run on sliding windows. Real audio: NE 0.72 ms/chirp, end-to-end 11 ms p50 /
  14 ms p95 vs 40 ms budget, 0 dropped blocks. A body still for longer than the 4 s presence window becomes
  background (track drops); breathing has its own path.

## 5c. Fused Range-Doppler graph on the NE (`ane_kernels/ane_fused_2d_radar.py`)
- All 13 non-const ops (conv, transpose, conv, 4× slice, sub, add, 2× mul, add, depthwise conv) are placed on
  the NeuralEngine by MLComputePlan. Stage-2 weights = centred 64-pt DFT · Hann · [1,−2,1] Toeplitz (62×64),
  scaled 1/256 so |Y|² stays far below the FP16 max.
- Error vs fp64 is 7–9e-4 on the NE (CoreML CPU FP16: 1.2e-2). A test target whose Doppler aliases into the
  MTI notch (e.g. 20 cm/s at 20.55 kHz → −1 Hz) inflates the relative error to 15 %: test at PRF/4 Doppler.
- Latency per call (18.8–22.3 kHz): B=1 226 µs NE vs 112 µs Accelerate 2×GEMM (NE 0.5×); B=4 390 vs 449 µs
  (1.15×, noisy run to run: 0.9–1.15×); B=16 1103 vs 1840 µs (**1.67×**), vs CoreML CPU 1.13×, vs NumPy
  FFT pipeline 22×. Synchronous `predict()` keeps a core busy for ~85 % of the call, so CPU offload vs GEMM
  is only ~50 % at B=16 and negative at B=1.
- Live use: sliding 64-chirp blocks, hop 8 → 11 maps for a 6 s capture in one B=16 call (~3 ms).

## 5e. Phase 4 — pulsed TDM stereo, super-resolution, 2D, respiration consensus
- **Stereo separation**: TDM (left chirp 0–16 ms, right 20–36 ms, 40 ms PRI). Simulated with a 30 ms room tail,
  right-speaker leakage into the left channel's 10–80 cm bins: max −37.7 / median −53.4 dB re direct path.
  Simultaneous up/down chirps leave a −14 to −15 dB floor (B·T = 56 → only ~17 dB processing gain): unusable.
- **Duty and sag**: 40 % duty per speaker = −4 dB average power and −4 dB SNR (pulsing buys no SNR over a
  quieter continuous chirp at the same average power; it buys the TDM slots). Continuous chirp: +1.4 dB at 2 s,
  then −5.2 dB by ~20 s and flat (limiter-like). TDM 12 s: left −0.5 dB, right −1.4 dB.
- **Warm stream**: open `sd.Stream` before `say`, play silence, switch to the train. Loop latency 174 ms, all
  frames in bounds (cold `playrec` after `say` once took 2874 ms).
- **Super-resolution** (`ane_superres_mimo.py`): 0.6 cm quadrature-atom dictionary of the real Tukey chirp,
  scaled FISTA, separable Re/Im soft threshold = relu(v−λ) − relu(−v−λ), 80 iterations, λ = 0.04 of the
  per-chirp max matched-filter output. 877 ops, all NE; 0.50 ms vs 10.1 ms NumPy per 64 chirps (deep graph of
  small layers: the NE's single dispatch beats 80 Python/BLAS calls). FP16 profile error 8.9 % vs fp64.
  Two equal targets, random phase, 30 dB: FFT never reaches 90 % below 9.8 cm (0 % at ≤ 3.6 cm); NE FISTA
  75–92 % at 2.4–3 cm, ≥ 90 % from 4.8–7.2 cm, ~50 % at 1.2 cm (in-phase pairs merge). On an extended real
  target (book) it reports a 1.2 cm median peak width vs 9 cm FFT, but splits one blob into several spikes —
  sharpening, not proven resolution.
- **2D**: mic x from the right-minus-left direct-path delay (planar model, mic on the speaker line; measured
  −6.1 cm). Ellipse intersection on a 0.5 cm grid, residual-gated. Sim: X median error 3.4 cm, Y 1.4 cm.
  Live: only rows where both channels track the same mover give (X, Y); one capture gave 33 / 149 sweep rows.
- **Joint stereo 2D** (`mimo.backprojection`, `mimo.joint_track`): per-channel max(MTI, presence) CFAR scores are
  backprojected onto a 1 cm (x, y) grid via each cell's bistatic ranges and combined as a geometric mean, so a
  scatterer elevated in one channel only cannot form a peak. A 4-state Cartesian EKF [x, y, vx, vy] observes
  (R_L, R_R) jointly with h = R_s(x, y) + κ ∇R_s·v; χ² gate (2 dof, 99.9 %), strongest gated peak, 4-frame
  initiation within 8 cm, coast ≤ 15 with velocity ×0.9, dropped if it coasts off the grid. Sim sweep: 95.3 %
  coverage, X median 3.3 cm (p90 10.4), Y 1.4 cm (p90 4.6); 0 false tracks in 5 no-hand scenes. Live replay:
  83/149 sweep rows (was 33 with independent trackers + intersection).
- **Library hardening**: silent captures only (`visual_cue`); workspace bubble y ≤ 55 cm and per-channel evidence
  only up to bistatic r ≤ 58 cm (`mimo.WORKSPACE_*`; `ys=GRID_Y_FULL, r_max=None` for the ungated view).
  Joint initiation is multi-hypothesis (sim sweep 95 → 100 % coverage, 0 false tracks in 5 sims).
  Right speaker: far static echoes only 3.4–4.1 dB below the left, direct path 19.8 dB below → its direct
  bleed sits in a ~16 dB null at the mic; per-bin OS-CFAR already normalises each channel (score medians
  L 8.0 / R 7.8 dB in 15–55 cm).
  Real-room limits found: (1) a seated person's body/arms move inside a 55 cm (and even a 40 cm) bubble, so a
  "hands off" baseline still produced 101 / 148 tracked rows (83 at y ≤ 40) — not a valid null test;
  (2) a continuous 8 s notebook sweep raises the per-bin 25th-percentile CFAR floor in every workspace bin
  (median evidence 9–10 dB vs 15 dB when the capture had a quiet hold), so captures should start chirping
  during the countdown while the user holds still, to give each capture its own quiet reference.
- **Gain normalisation** (`mimo.gain_normalise`): divide by a 1 s rolling median of the direct-path amplitude,
  only for channels whose direct path is within 10 dB of the strongest. Here the right direct path is 19 dB
  below the left and swings 3.5 dB std during a hold, uncorrelated with static echoes (0.22): it measures
  interference, not speaker gain, and dividing by it would corrupt every right-channel echo.
- **Respiration consensus** (`fmcw.respiration`): spectrum (parabolic), autocorrelation FIRST peak that rises
  > 0.3 above the preceding trough (irregular breaths make the one-period peak low: 0.13 vs trough −0.51), and
  peaks of the Hilbert-envelope-normalised waveform. Median; 'high' if all within 5 %. 30 s capture:
  15.71 / 15.97 / 16.67 → 15.97/min, spread 4.4 % → high.

## 5f. Phase 4d — pre-roll calibration and the hand / seated-presence split (`mimo.hand_presence_pipeline`)
- **Pre-roll**: the transmit train is `PREROLL_S` (2 s) + task. Chirps start while the user holds their hands
  in their lap. `calibration_rows` takes rows t ∈ [0.2, preroll − 0.1). A capture without a pre-roll uses its
  quietest 2 s window. `calibrate` returns B0 (static background) and per-bin σ_MTI / σ_presence (median),
  each bounded at `FLOOR_MIN_REL` × the workspace median so an unusually quiet bin cannot give infinite
  SNR. `presence_fixed` subtracts B0 and keeps the direct-path drift lock. Floors stay locked for the
  whole capture, so a continuous sweep can no longer raise its own CFAR floor (the Phase 4c failure).
- **Tracker**: a single joint EKF over y 8–58 cm, r ≤ 62 cm. Classify whole segments
  (`track_segments`), not frames:
  hand = start y ≤ `HAND_INIT_Y[1]` (45), excursion ≥ `INIT_MIN_DISP` (6 cm), duration ≥ `MIN_GESTURE_S`
  (0.5 s); rows count as hand only while y ≤ `HAND_HOLD_Y_MAX` (52).
  seated presence = excursion < `PRESENCE_MAX_EXCURSION` (5 cm) with mean y in `PRESENCE_Y` (40–58), or an
  interior pre-roll static reflector in that band (`static_reflectors` drops grid-edge maxima).
  Everything else is `other_motion`.
- **Results**: sim sweep 100 %, X 2.5 / Y 1.0 cm median; 0 / 5 empty; torso presence 3 / 3, with one seed still
  showing 2 ghost hand tracks. Live take 2: 93 % hand coverage, X −29…+21 cm, against 7 % for the
  uncalibrated tracker on the same data. Real hand evidence is only 1.3–6 dB over threshold. `MIN_GESTURE_S`
  was chosen knowing the library baseline's 0.36 s track.
- **Torso micro-Doppler** (`micro_motion`, `torso_evidence`): a pre-roll static reflector in `PRESENCE_Y`
  (25–55 cm) is a seated torso when, over rows with the joint track ≥ 12 cm away, the phase-referenced
  displacement of its range bin has `MICRO_RMS_MM` (0.3–5 mm) rms and ≥ `MICRO_BREATH_FRAC` (50 %) of its
  Welch power in 0.12–0.6 Hz, in either channel. Desk / laptop statics: < 0.2 mm rms.
- **Streaming tracker**: `JointEKF.step(t, cands)` returns (status, state, events); `joint_track` wraps it
  for batch use and applies the "fill" (chain confirmed) and "clear" (coast dropped) events. The live app
  keeps a trail deque and applies the same events. The S=2 MIMO graph keeps all 878 ops on the NE, so the
  live chain runs once per PRI.
- **Cue timing** follows the transmit buffer's playback position (`st["pos"] - lead`), not wall-clock sleeps.
  The countdown is clamped to the pre-roll length (the silent lead-in had shown "3").

## 6. Performance expectations (M2, coremltools 9, Python predict())
The kernel is ~1 MFLOP/chirp with low arithmetic intensity, so per-call overhead dominates:
- `predict()` floor ≈ 50–60 µs on CPU, ≈ 160–210 µs when the NE is used (dispatch).
- NE throughput saturates ≈ 0.9 M chirps/s at S ≥ 1024; Accelerate fp32 GEMM (AMX) reaches ≈ 1.2–1.3 M chirps/s.
- Real time needs 25 chirps/s, so every path is > 1000× real time. The NE's win is offloading the CPU,
  not raw latency for this tiny kernel.
