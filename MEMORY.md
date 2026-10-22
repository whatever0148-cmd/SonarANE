# SonarANE memory

Measured facts and lessons. Newest first. Numbers come from `artifacts/*.json`.

## 2026-10-22 — Phase 5: torso micro-motion, showcase app, demo video, repo packaging
- Seated presence on the real take-2 capture resolved to **true** without a new recording: the torso band is
  now 25–55 cm (the user leans in; torso statics sat at (−3, 35) and (14, 34) cm), and a static reflector only
  counts when its range bins show torso-like micro-motion (`mimo.micro_motion` / `torso_evidence`: 0.3–5 mm
  rms, ≥ 50 % of the power in 0.12–0.6 Hz, over rows with the joint track ≥ 12 cm away). Measured: 0.9–1.4 mm
  rms, 58–92 % in band on both statics; the laptop-edge static at (−13, 15) is 0.04–0.14 mm rms. Take 1 and the
  library baseline have no static in the band → still false (honest: not every posture puts the torso there).
- `joint_track` is now a thin batch wrapper over the streaming `mimo.JointEKF` (per-row `step` with retro
  "fill"/"clear" events); validation numbers are bit-identical to before the refactor. The tracker grid top is
  `TRACK_Y_MAX = 58` (decoupled from `PRESENCE_Y`; tying them cost 13 hand rows on the v1 replay).
- The MIMO range+FISTA graph stays on the NE at S=2 (one stereo PRI per call): 878 ops, 0.42 ms per call on the
  bench, 1.26 ms p50 inside the live stream. The live engine (`src/live_radar_app.py`) runs the full Phase 4d
  chain per PRI: 21.5 rows/s on real audio, block→track latency 9.9 / 13.4 ms (p50 / p95), 0 dropped blocks;
  on the synthetic scene it recovers x_mic −5.7 cm (truth −6), the torso static at (−11, 48) (truth (−5, 48))
  and tracks the sweep as a hand. The live integer-sample direct-path search gave x_mic −2.1 cm on real audio
  vs −6.4 from the offline fits — a sub-sample refinement is the next step there.
- Demo video `artifacts/sonar_ane_demo.mp4` (60 fps, 1280×720): capture 1 replayed through the exact offline
  pipeline (so the video matches the telemetry: 2 hand tracks, 93 %, presence true), then the 30 s respiration
  capture with a causal breaths/min readout converging on the 16.0/min consensus. Cursor positions are
  interpolated between the 25 Hz rows; everything else is per-row data. `.gif` = 9 s of capture 1 at 15 fps.
- Repo: `git init`, `.gitignore` keeps PNG/JSON/MP4 and the two WAVs the demo replays, drops `.venv`, models
  (rebuilt from `kernel_tag`) and the other captures. `README.md` carries the measured tables and the limits.

## 2026-10-21 — Phase 4d: pre-roll calibration + hand / seated-presence split (`artifacts/phase4_*`)
- Every live capture now chirps a 2 s hands-in-lap pre-roll first (yellow window), then the 8 s task (green).
  `mimo.calibrate` locks the static background B0 and per-bin MTI / presence floors (median over rows
  0.2–1.9 s), each floor bounded below at 0.5 × the workspace median. Legacy captures without a pre-roll
  calibrate on their quietest 2 s.
- `mimo.hand_presence_pipeline`: one joint tracker on y 8–58 cm, then whole-segment classification. Hand =
  starts at y ≤ 45, excursion ≥ 6 cm, ≥ 0.5 s, rows count only while y ≤ 52. Seated presence = excursion < 5 cm
  with mean y 40–58, or an interior pre-roll static reflector in the torso band. 4-frame speed gates made
  torso ghost tracks from jitter; whole-segment excursion does not.
- Live 10 s sweeps (8 s analysed): take 1 (23:10, `phase4_4d_take1_*`) 1 hand track, 158 / 199 rows, but
  X only −41…+2 cm. Take 2 (23:26, `phase4_*`) 2 hand tracks, 185 / 199 rows (93 %, 181 updated),
  X −29…+21, Y 8…31 cm, first hand row 0.2 s after GO. The uncalibrated workspace tracker on the same take 2
  data: 7 % coverage. Take 1: 67 %, with Y up to 58 cm.
  Seated presence was false in both takes: the pre-roll statics were at y 15–35 cm (lap/arms), none in 40–58.
- Sims: sweep 100 % coverage, X median 2.5 cm / Y 1.0 cm. Empty: 0 / 5 hand tracks. Torso: presence 3 / 3,
  but seed 3 still gives 2 ghost hand tracks (1.2 s and 2.9 s, about 3.4 dB margin).
  Library baseline replay: 0 hand tracks, 3 other-motion segments. CAVEAT: `MIN_GESTURE_S = 0.5` was picked
  knowing that the baseline had one 0.36 s track, so this is not a blind result. Real hand evidence sits
  only 1.3–6 dB above threshold, so a margin rule would delete real hand rows.
- The v1 12 s replay gives 5 hand tracks / 186 rows in the validation script but 3 / 165 through the live
  test's `--from-wav` path (different frame source); not chased.
- `static_reflectors` now keeps interior map peaks only. Grid-edge maxima at y = 58 were cut-off ridges and
  had faked "presence" on the v1 replay.
- Volume saved at 13 (unmuted) on both takes and restored. Band SNR take 1 60.4 dB, take 2 42.3 dB.

## 2026-10-21 — Phase 4c: library-hardened, silent captures (`artifacts/phase4_*`)
- User works in a university library: every `say` / spoken cue removed; `src/visual_cue.py` shows a silent
  on-screen countdown. Scripts raise volume to 75 % for the chirp, then restore the saved volume and mute state.
- Right speaker: far static echoes −3.4…−4.1 dB vs left, direct path −19.8 dB → direct-bleed null, not low
  output. CFAR score medians L 8.0 / R 7.8 dB (15–55 cm): no right-channel penalty.
- Workspace bubble (y ≤ 55, r ≤ 58 cm) + multi-hypothesis joint initiation, NE path:
  original 12 s capture 81 / 149 rows (ungated 94, independent 33); new silent 8 s notebook sweep 98 / 198
  (was 6 before multi-hypothesis init; 59 updated / 39 coasted), plausible diagonal path from (−37, 8) to
  (+8, 47) cm over 0–5.5 s, then 6.6–7.4 s hugging the 55 cm bubble edge (torso / library, not notebook).
- Library baseline (6 s, hands off): 101 / 148 rows tracked (83 with a 40 cm bubble) → real body/arm motion
  inside the workspace; the baseline is not a clean null and 0 false hand tracks is NOT demonstrated.
- 8 s TDM sweep: SNR 68.1 dB, latency 173 ms, all frames in bounds, level change L −0.03 / R +0.03 dB (no sag).
- Next: chirp during the countdown (quiet CFAR reference), baseline with hands resting in the lap before the
  countdown, a gesture classifier (speed / extent) to separate a deliberate sweep from seated-body motion.

## 2026-10-21 — Phase 4b: joint stereo 2D tracker (replay of `phase4_capture.wav`)
- Right-channel "sag" explained: its direct path is 19 dB below the left at the mic (mic ~6 cm left of
  centre), swings 3.5 dB std during the hold vs 0.04 dB on the left, and correlates only 0.22 with static
  echoes → interference on a weak path, not speaker gain. Gain normalisation applied to the left only.
- Joint backprojection + 4-state EKF: 83 / 149 sweep rows with a 2D position (62 updated, 21 coasted), up
  from 33 with independent trackers. Clean part: a left-to-right arc at Y ≈ 17 cm, X −13 → +20 cm, 4.0–5.6 s.
  0–2 s: both channels consistently moved out to ~65–80 cm bistatic range (the old left tracker's
  nearest-scatterer bias hid that); positions there jump around → low confidence.
- Sim: 95.3 % sweep coverage, X median 3.3 cm / Y 1.4 cm; 0 false 2D tracks in 5 no-hand scenes (simulated
  only — no real empty-room recording exists yet).
- NumPy replay (validation script) gives 88 rows, NE replay 83: CFAR margins differ slightly between paths.

## 2026-10-21 — Phase 4: TDM stereo, NE super-resolution, 2D, respiration consensus (`artifacts/phase4_*`)
- Respiration consensus on the saved 30 s capture: spectrum 15.71, autocorr 15.97, envelope peaks 16.67 →
  **15.97/min, high (4.4 % spread)**. Two estimator fixes came from this capture (ACF first-peak + trough test,
  envelope-normalised peaks); sims (3 seeds) still high/ok/high.
- Sag evidence: continuous 30 s capture +1.4 dB then −5.2 dB by ~20 s, flat after (limiter-like); 6–8 s
  captures never sagged. TDM 40 % duty, 12 s: left −0.5 dB, right −1.4 dB (continuous at 12 s: −2.2 dB).
- Warm stream: loop latency 174 ms (not < 120 ms), 300/300 frames both channels, 0 callback flags.
  SNR 57.7 dB (−4.8 dB vs continuous = the duty cycle).
- Stereo: mic calibrated −6.1 cm (left of centre) from direct-path timing; baseline 27 cm assumed.
  2D only on 33 / 149 sweep rows (both channels on the same mover): an arc X −27 → +20 cm at Y 8–22 cm
  during ~3.5–5.3 s. Right channel locked onto far scatterers (~80 cm) early; after 6 s the held book faded
  into the presence background and both channels tracked 60–90 cm clutter.
- NE: combined range + 80-it FISTA graph, 878 ops all on NE, ~0.6 ms per 64 frames; TDM range conv alone is
  CPU-placed at every bank size. Super-res: see skill §5e (2.4–3 cm at 75–92 %, not 1.2 cm).
- Volume: saved/restored 31 % this run (user had changed it from 69 %); a later read showed 19 % (not set by us).

## 2026-10-21 — 30 s respiration capture, book pressed to chest (`artifacts/respiration_*`)
- Output started **2.87 s late** (first ~2.9 s silent, then a ~1 s fade-in); normal I/O latency is ~0.1 s. The
  last chirps fell past the recording end and crashed framing → frames are now clipped to the recording
  (693 of 750 chirps kept). Direct-path level also sagged ~6 dB over 30 s (speaker protection or the book;
  not confirmed). Alignment jitter 4.1 samples (gate 4) — the fade-in/sag. Telemetry metadata for this run
  was reconstructed (volume read 69 % before and after).
- Book as chest reflector: breathing bin 22 cm (−30 dB re direct, 8 dB stronger than the bare chest at 51 cm).
- New extraction (`fmcw.breathing_displacement_mm`): 3-bin coherent sum (19.6–24.5 cm), per-chirp phase
  increments clamped to ±1.2 rad and amplitude-gated; 1.4 % of steps clamped, 45 gated.
- Result: **15.7 breaths/min** (spectrum, 6.9 cycles, user target ~15), 6.3 mm p-p, ~13 mm slow drift removed.
  Peak spacing said 11.5/min (shallow breaths miss the 30 % prominence bar) → confidence "moderate".
  Rate rule now: spectrum when >= 3 cycles; 'ok' only when peaks agree within 20 %.
- No coherent static reflector in this capture (best coherence 0.29) → noise floor not reported (needs >= 0.8).
- Sim (4 seeds, −43 dB, 15 mm posture shift): new method 14.8–15.4/min vs single-bin 10.3–15.6/min.

## 2026-10-21 — 20 s live respiration capture (`artifacts/respiration_*`)
- Clean capture (500 chirps, drift 3.5 ppm, NE range conv on NeuralEngine, volume 69 % saved and restored).
- First analysis was wrong: 49 cm bin, 26 mm p-p, 8.8/min. Cause: body movement at 1.2–4.4 s and 13.5–14.6 s
  (MTI energy −35 dB vs −53 dB quiet) made the phase jump ~π per chirp, unwrapping slipped whole 8.3 mm
  half-wavelengths, and the breathing bin was chosen on that motion energy.
- Fix: `fmcw.quiet_segment` — analyse only the longest motion-free stretch (MTI energy < median + 6 dB, blips
  ≤ 0.4 s bridged) and report max phase step (unwrap-safe if < π/2).
- Gated result: 4.4–13.5 s (9.1 s), 51.5 cm bin (−37 dB re direct; chest is lower/further from the hinge
  speakers and mic than the face), 8.4 mm p-p, ~10 breaths/min from spectrum (peaks 5.4 s and 10.2 s suggest
  ~12.5/min) but only 1.5 cycles → LOW confidence. No confident breathing rate yet.
- Next: 30–40 s capture sitting back against the chair, torso still, to get >= 3 motion-free cycles.

## 2026-10-21 — Phase 3: TGC CFAR, fractional framing, respiration interferometer, live radar app
- 8 s live capture (18.8–22.3 kHz, vol 75 %, spoken "hold still and breathe" mixed into tx at 4 s, < 8 kHz):
  SNR 62.5 dB, peak −13.9 dBFS, drift 6.5 ppm. Volume restored to 56 % (verified); it read 69 % later (not set
  by the scripts).
- Tracking (after fractional framing + coast decay 0.9): 100 % coverage, 82 % measurement-updated, 1 segment.
  Sweep 0–4 s: 3–71 cm, 92 % updated (the outward leg Phase 2 missed). Hold: 20–33 cm, 73 % updated.
  fp64 NumPy replay of the same WAV gives 87 % — marginal CFAR decisions differ between FP16 NE and fp64.
- TGC on the Phase 2 capture recovered nothing under a 0-false-track rule (see skill §5d); the claimed outward
  arch at 40–85 cm was not distinguishable from noise there.
- Frame-start rounding caused full-range white stripes every ~3.3 s (1 sample per 1/(drift·N) chirps).
  Fixed with fractional framing; this also removed most of the apparent direct-path-referencing gain.
- Respiration: settled hold only 5.9–8.0 s → ~1 deep inhale (~5 mm), 4.2 mm p-p, 0.3 cycles → rate NOT
  measurable from this capture. Noise floor on a coherent 240 cm reflector: 0.18 mm rms referenced vs 0.39 mm
  unreferenced. Use tests/live_respiration_ane_test.py (20 s) for breaths/min.
- Live app (real audio, headless 8 s): NE (S=8 bank) 0.72 ms/chirp, latency 11.2 ms p50 / 13.9 ms p95, 0
  dropped blocks, 1 drift re-alignment. Sim scene: 13.9 vs 15 breaths/min, 4.06 vs 4.0 mm.

## 2026-10-21 — Phase 2: dual-path Kalman tracker + fused NE Range-Doppler, 18.8–22.3 kHz live capture
- Band 18.8–22.3 kHz (vol 75 %, amp 0.9, Tukey 0.15): band SNR **62.4 dB** (+6 dB vs 1b), 53 dB above
  15–18.3 kHz, peak −13.3 dBFS, latency 109 ms, drift 5.2 ppm, align PMR 48 dB. Volume restored to 56 %.
  18.8 kHz is much less audible than 17.2 kHz but not inaudible to everyone.
- Live tracking: coverage 66 % (65 % measurement-updated, 1 % coasted), 2 segments, longest 3.5 s, range
  14.4–39.1 cm (median 18.6), zero-velocity hold 1.28 s (56 % of those updates from presence Path B), Path B
  supplied 52 % of all updates. 0.4–2.4 s had no echo energy in 15–80 cm at all and the track never reached
  the planned 55 cm: the outward sweep either started late or left the speaker/mic beam. Not tuned on this.
- Phase 1b WAV replay (17.2–20.7 kHz, before the band switch): coverage 57.4 % → **85.1 %** (70.3 % updated,
  14.9 % coasted), 6 → 2 segments. ~14 points of that depend on CV coasting (velocity decay 0.8 → 71 %).
- Sims: 0 false tracks in 3 no-hand scenes; hand with 0.8 s pause 100 % updated, RMS 2.0–4.2 cm.
- Bugs found on the way: B0-scaled leakage lock mis-scales real reflectors (false track at 83 cm); gain-only
  lock leaves delay-drift sidelobes at 11–18 cm (false tracks); fixed with the exact direct-path basis.
  Nearest-only track initiation broke on near-range residue; fixed with multi-hypothesis chains.
- Fused NE graph: all ops on NE, err 7e-4. Beats Accelerate GEMM only at B=16 (1.67×, CPU offload ~50 %);
  at B=1 it is 2× slower (call overhead). Synchronous predict() burns CPU for ~85 % of the call.
- `phase1_capture.wav` is now unreadable by --from-wav (band guard). Its Phase 2 replay is in
  `artifacts/phase2_replay_phase1_*`.

## 2026-10-21 — Phase 1b: 17.2–20.7 kHz, vol 75 %, [1,−2,1] MTI, CFAR 11 dB + continuity — **hand tracked**
- Capture (attempt 1, Tukey α 0.15, amp 0.9, vol 75 %): band SNR 56.3 dB, 46.6 dB above 15.0–16.7 kHz,
  peak −17.6 dBFS (no clip), latency 110 ms, drift 0.93 ppm, align peak-to-median 44 dB. Volume restored to the
  user's pre-test 63 % (it was 63, not 50, at test time).
- Hand: confirmed track in 57 % of MTI frames (raw 11 dB CFAR hits 72 %), 6 segments, longest 0.72 s,
  range 18.4–37.0 cm (median 22.5 cm; uncorrected 13.5–42.9 cm), motion SNR 22.9 dB, range rate p95 41 cm/s
  (max approach −60 cm/s), micro-Doppler p95 ≈ 46 Hz. Gaps line up with hand turnarounds where MTI2 is blind.
- MTI2 0 cm residual −48 dB re direct path (noise-gain normalised). Static peaks now at 12 / 22 cm (palm or book
  held near the screen) and the 2.45 m wall.
- NE: 298 frames (150 raw + 148 MTI) in 5 × S=64 NE calls = 4.1 ms (≈ 1470× real time), rel. err 2e-4.
- Band shift: recorded PSD is roughly flat 17.2–20.7 kHz (−67…−62 dB/Hz). In-band power rose 6.5 dB vs Phase 1
  (−71.8 → −65.3 dBFS), but that is mostly the extra drive (vol 55→75, amp 0.6→0.9). Phase 1 showed no roll-off
  inside 18.5–22 kHz, so the claimed ~12 dB transducer gain at the lower band is **not** confirmed.
- Canceller choice is data-driven (see skill §5): on the real Phase-1 capture MTI2 beat global/local LS templates
  by 7–16 dB at 0 cm because the direct path drifts slowly rather than jittering. Cost: simulated range RMS
  3.6 cm vs 1.8 cm for the mean-subtraction tracker. The tracker follows max CFAR ratio, which sometimes sits on
  the 20 cm scatterer while a brighter return is at ~35 cm (forearm or book edge).
- 17.2 kHz is audible to many adults; expect a faint hiss at 75 %.

## 2026-10-21 — Phase 1: live 6 s capture (`artifacts/phase1_telemetry.json`, `phase1_capture.wav`)
- Devices: MacBook Air Speakers → MacBook Air Microphone (selected by name); volume 50 → 55 → restored 50.
- Passed on attempt 1 (Tukey α 0.10, amp 0.6): **band SNR 52.2 dB** (18.5–22 kHz vs silent lead-in),
  in-band 42.4 dB above 15–18 kHz, recording peak −20.5 dBFS. Speaker/mic response falls off a cliff above ~22.8 kHz.
- Alignment: I/O latency 4618 samples (96.2 ms), clock drift 0.21 ppm (speaker and mic share a clock),
  jitter 0.26 samples RMS, matched-filter peak-to-median 47.5 dB. Direct path stable to 1.2 dB p-p / 0.37° RMS.
- NE live processing: 150 chirps in 3 × S=64 NE calls ≈ 2.5–3.7 ms (≈ 1600–2400× real time), rel. err 2e-4 vs fp64.
  Streaming S=1 (CPU-placed) ≈ 78 µs/chirp.
- Room: strongest static reflectors 2.11 / 2.45 / 2.82 m (walls). Those far returns are *incoherent* chirp-to-chirp
  (adjacent-difference energy ≈ static energy, −37…−45 dB re direct path, steady over 6 s, not a moving person).
- **No hand motion was detected in 6–120 cm** in this capture: near-field motion floor −58 dB re direct path,
  energy flat over all 6 s, 3/149 chirps over the 16 dB CFAR (= false-alarm level), no track. Either no hand was
  moving in front of the screen during the window, or the echo is below ≈ −42 dB re direct path. Needs a re-run with
  a confirmed hand gesture (`tests/live_sonar_ane_test.py`; `--from-wav` re-analyses the last capture silently).
- Tracker bugs found and fixed today: (1) far bins ≥ 200 cm as the noise reference put the threshold ~30 dB too high
  in a real room; (2) argmax of |X_m − X_{m−1}| hops between the two lobes of a moving target. Final tracker = OS-CFAR
  on |X − mean X| + coupling correction; on a simulated 20–50 cm, 0.5 Hz hand: 100 % detection, range RMS error
  1.8 cm, velocity RMS error 8 cm/s. Raw track matched r + κ·v (κ = +0.231 s) to 1.0 cm RMS, confirming the sign.

## 2026-10-21 — Phase 1: CoreML NE range-FFT benchmark (synthetic frames)
- Cleanup: removed `research_repos/` and `.claude/skills/ane-private-api`; project is public-API only.
- Model: FP16 1x1 conv `[1,1920,1,S] → [1,256,1,S]`, 0.98 MB weights, 0.98 MFLOP/chirp, built with MIL Builder.
- Synthetic echoes at 30 / 75 cm recovered at 29.4 / 75.95 cm (2.45 cm grid) on every backend.
- **NE placement is decided by CoreML's planner per shape** (MLComputePlan, CPU_AND_NE):
  S=1 → CPU, S=8 → NE, S=32 → CPU, S=64 → NE, S=96/128 → CPU, S=256…2048 → NE. `matmul` at S=128
  and a `[1,1920,2,64]` layout also went to CPU. Live bank therefore uses **S=64**.
- Median predict() latency / throughput (µs, chirps/s):

  | S | CoreML CPU_AND_NE | CoreML CPU_ONLY | NumPy GEMM fp32 | NumPy dechirp+FFT |
  |---|---|---|---|---|
  | 1 | 61 (CPU) / 16 k | 51 / 20 k | 8.7 / 115 k | 95 / 10.5 k |
  | 8 | 160 (NE) / 50 k | 64 / 126 k | 48 / 166 k | 223 / 36 k |
  | 64 | 214 (NE) / 300 k | 111 / 575 k | 65 / 989 k | 1334 / 48 k |
  | 128 | 181 (CPU) / 708 k | 182 / 702 k | 104 / 1.23 M | 2637 / 49 k |
  | 1024 | 1160 (NE) / 883 k | 1203 / 851 k | 789 / 1.30 M | 21 929 / 47 k |

- Lesson: for this small, low-intensity kernel the NE does **not** beat Accelerate/AMX GEMM; NE dispatch
  costs ≈ 100–150 µs per call and NE throughput saturates ≈ 0.9 M chirps/s. Its value here is CPU offload
  and FP16 accuracy (rel. err 2e-4 on NE vs 2.7e-2 on CoreML's CPU fp16 path).
- Lesson: up-chirp FMCW range-Doppler coupling (κ = f_c/μ = 0.231 s) inflates hand range swing and speed by
  ≈ 23 % at 0.5 Hz waving; the live test applies the exact inverse filter (see live-capture entry above).
- Audio: default I/O was a Bluetooth headset "ZENITH" (16 kHz mic) earlier today; always select the
  MacBook Air devices by name. osascript volume only controls the *default* output device.
