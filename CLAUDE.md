# SonarANE

Ultrasonic FMCW sonar on an M2 MacBook Air: built-in speakers transmit 18.8–22.3 kHz chirps, the
built-in mic records the echoes, the matched-filter + range-FFT runs as a static FP16 1x1 conv on the
Neural Engine, a fused NE graph builds Range-Doppler maps, a dual-path (MTI motion + leakage-locked
presence) Kalman tracker with a range-adaptive CFAR follows the hand, and direct-path phase-referenced
interferometry measures sub-millimetre chest motion. `src/live_radar_app.py` runs it all live at 25 Hz.
Phase 4 (`src/mimo.py`) adds a pulsed TDM stereo waveform (left chirp 0–16 ms, right 20–36 ms per 40 ms),
NE unrolled-FISTA super-resolution and two-speaker 2D (X, Y) ranging.

## Ground rules
- **Public Apple APIs only**: `coremltools` (MIL Builder → mlprogram → `CompiledMLModel`,
  `MLComputePlan`) and `sounddevice` (PortAudio/CoreAudio). No private ANE frameworks, no
  `_ANEInMemoryModel`, no IOSurface hacks. Those research repos were removed on purpose.
- Python: always `~/SonarANE/.venv/bin/python` (3.12, coremltools 9.0, numpy w/ Accelerate BLAS; `pip` was
  bootstrapped with `ensurepip` for `imageio-ffmpeg`, which bundles the ffmpeg used by the demo renderer).
- Build models with `coremltools.converters.mil.Builder` directly, not PyTorch tracing (the
  venv's torch is untested by coremltools and spams warnings).
- Audio devices are picked **by name** (`MacBook Air Microphone`, `MacBook Air Speakers`), never
  by default index — a Bluetooth headset ("ZENITH", 16 kHz mic) is sometimes the system default.
- **SILENT CAPTURES ONLY — the user works in a university library.** Never call macOS `say`, never mix
  speech into the transmit buffer, never ring the terminal bell. Cues go through `src/visual_cue.py`
  (on-screen countdown window + terminal print). Only the 18.8–22.3 kHz chirps may be played.
- Library scene: people and desk dividers move at 65–90 cm. Interactive hand tracking is gated to the
  personal workspace bubble (y ≤ 55 cm, bistatic r ≤ 58 cm). Every live capture starts with a 2 s
  hands-in-lap pre-roll that locks the background and noise floors (`mimo.calibrate`). Hand tracks must
  start at y ≤ 45 cm and move ≥ 6 cm. A pre-roll static reflector at 25–55 cm whose range bins show
  0.3–5 mm rms micro-motion, mostly in the respiration band, is reported as `seated_presence_detected`.
- Live tests touch the user's volume (test level 75%): always save → set → restore in `try/finally`
  via `osascript`, including the mute state.
- Changing the band in `src/fmcw.py` changes the conv weights: models carry a `kernel_tag` in their
  metadata and the live test rebuilds stale ones; `--from-wav` refuses a capture from another band.

## Layout
| Path | What |
|---|---|
| `src/fmcw.py` | Single source of truth for chirp params, tx waveform, fused dechirp+DFT kernel, range axis |
| `ane_kernels/ane_radar_benchmark.py` | Builds `artifacts/models/sonar_range_fft_S{S}.mlpackage/.mlmodelc`, compute-plan check, CoreML vs NumPy benchmark → `artifacts/ane_benchmark.json` |
| `ane_kernels/ane_fused_2d_radar.py` | Fused range → MTI+Doppler → power → smoothing graph (all ops on NE), benchmark vs CPU → `artifacts/ane_fused_benchmark.json` |
| `src/live_radar_app.py` | Phase 5 showcase: live TDM stereo stream → S=2 MIMO NE graph per PRI → streaming `JointEKF` + presence + breathing, three-panel window; `--sim`, `--headless N [--snapshot PNG]`, `--demo-from-wav` renders `artifacts/sonar_ane_demo.mp4/.gif` from the saved captures |
| `src/live_radar_app_phase3.py` | Earlier continuous-chirp 1-D live app (S=8 range conv bank, dual-path tracker, breathing; `--term` / `--headless` / `--sim`) |
| `tests/live_sonar_ane_test.py` | 8 s Phase 3 capture (sweep, then hold + breathe) → NE → TGC tracker → respiration → `artifacts/phase3_*` |
| `tests/live_respiration_ane_test.py` | 20 s still-breathing capture (or `--from-wav`) → breaths/min + mm → `artifacts/respiration_*` |
| `src/mimo.py` | Phase 4 TDM stereo waveform, 768-sample kernel, per-channel tracking, mic calibration, 2D solve |
| `ane_kernels/ane_superres_mimo.py` | 80-iteration FISTA super-res graph (877 ops, all NE) + combined range+FISTA MIMO model, two-target resolution test → `artifacts/ane_superres_benchmark.json` |
| `tests/live_phase4_mimo_superres.py` | Silent warm-stream TDM capture: 2 s hands-in-lap pre-roll (yellow), then 8 s sweep (green) or 6 s `--library-baseline`; `--from-wav TAG` replays. NE range+FISTA → pre-roll-calibrated hand / seated-presence tracker → `artifacts/phase4_*` |
| `src/visual_cue.py` | Silent on-screen countdown window + terminal cues (replaces every `say`) |
| `tests/joint2d_validation.py` | Silent hand-tracker check with a 2 s pre-roll: sim sweep (coverage, X/Y error), 5 empty scenes (must give 0 hand tracks), 3 seated-torso scenes (presence), replays of `phase4_v1_12s` and `phase4_library_baseline` → `artifacts/phase4_joint2d_validation.json` |
| `tests/tracker_validation.py` | Silent tracker check: simulated hand with pause, 3 no-hand scenes (must be 0 false tracks), WAV replay |
| `artifacts/` | Models, benchmark JSON, capture WAV, dashboard PNG, telemetry |
| `.claude/skills/ultrasonic-fmcw-radar/SKILL.md` | The math and the CoreML pipeline in detail |
| `MEMORY.md` | Measured results and lessons, newest first |
| `README.md` | Public write-up: architecture, measured tables, limits (keep its numbers in sync with the telemetry) |

## Commands
```bash
~/SonarANE/.venv/bin/python ~/SonarANE/ane_kernels/ane_radar_benchmark.py   # range models + benchmark (~2 min)
~/SonarANE/.venv/bin/python ~/SonarANE/ane_kernels/ane_fused_2d_radar.py    # fused RD models + benchmark (~2 min)
~/SonarANE/.venv/bin/python ~/SonarANE/tests/tracker_validation.py phase2   # silent tracker regression
~/SonarANE/.venv/bin/python ~/SonarANE/tests/live_sonar_ane_test.py         # plays sound, records mic
~/SonarANE/.venv/bin/python ~/SonarANE/tests/live_sonar_ane_test.py --from-wav phase3   # re-analyse, silent
~/SonarANE/.venv/bin/python ~/SonarANE/tests/live_respiration_ane_test.py --seconds 20  # breathing, plays sound
~/SonarANE/.venv/bin/python ~/SonarANE/src/live_radar_app.py                # showcase window, chirps only; Ctrl-C stops
~/SonarANE/.venv/bin/python ~/SonarANE/src/live_radar_app.py --sim --headless 12   # silent pipeline check (no audio)
~/SonarANE/.venv/bin/python ~/SonarANE/src/live_radar_app.py --demo-from-wav       # 60 fps demo video from saved WAVs
~/SonarANE/.venv/bin/python ~/SonarANE/ane_kernels/ane_superres_mimo.py    # FISTA graph + resolution test (~5 min)
~/SonarANE/.venv/bin/python ~/SonarANE/tests/live_phase4_mimo_superres.py                     # silent 2 s pre-roll + 8 s 2D sweep
~/SonarANE/.venv/bin/python ~/SonarANE/tests/live_phase4_mimo_superres.py --library-baseline  # silent 2 s pre-roll + 6 s, hands off
~/SonarANE/.venv/bin/python ~/SonarANE/tests/live_phase4_mimo_superres.py --from-wav phase4_v1_12s   # re-analyse
~/SonarANE/.venv/bin/python ~/SonarANE/tests/joint2d_validation.py         # silent joint-2D regression
```
Run both benchmarks first: the live test reads their JSON and loads the compiled `.mlmodelc` files
(stale band models are rebuilt automatically from their `kernel_tag`).

## Standard CoreML pipeline (summary — details in the skill)
1. Kernel `W ∈ R^{2K×N}` (K=128 range bins, N=1920 samples) from `fmcw.dechirp_dft_kernel()`.
2. MIL: `x:[1,N,1,S] fp16 → mb.conv(weight=W[:,:,None,None], pad_type="valid") → [1,2K,1,S]`,
   `opset_version=ct.target.macOS15`, `compute_precision=FLOAT16`.
3. `ct.models.utils.compile_model` once → `.mlmodelc`; load with `CompiledMLModel(..., compute_units=CPU_AND_NE)`.
4. Verify with `MLComputePlan.load_from_path(...).get_compute_device_usage_for_mlprogram_operation(op)`
   — CoreML's planner decides per shape; **S=64 → NE, S=128 → CPU** on this machine (see MEMORY.md).
5. Apply `fmcw.mti2` to the time-domain frames, then normalise **each** frame to [-1, 1] before the FP16 cast;
   rescale each output column with `fmcw.amp_scale()/gain`.
6. Fused 2-D graph (`ane_fused_2d_radar.py`): `x [B,1920,1,64]` → range conv → transpose → MTI+Hann+Doppler
   conv → slice/sub/add complex combine → power → depthwise 1x5 range smoothing. One gain per 64-chirp
   block (Doppler needs slow-time coherence). Only beats Accelerate GEMM when batched (B=16).
7. Frames are cut at **fractional** sample starts (`fractional_frames`, frequency-domain delay) from the clock
   fit; rounding to whole samples made the grid jump one sample every ~3 s at 6.5 ppm drift, a 2.7 rad step
   on every bin that MTI and presence read as full-range motion.
8. Small graphs may never reach the NE: the TDM range conv (768 → 128) stays on the CPU at every bank size
   from 8 to 512. Fusing it into the FISTA graph (two inputs, two outputs) puts all 878 ops on the NE.
9. Live captures open the duplex `sd.Stream` **before** `say` (warm stream) and save WAV + metadata before
   analysing. A cold `playrec()` right after `say` once started the speakers 2.87 s late.
