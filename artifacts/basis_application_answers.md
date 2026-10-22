# Basis Fellowship application — SonarANE (numbers checked against artifacts/*.json, 2026-10-22)

Portfolio / GitHub: https://github.com/whatever0148-cmd/SonarANE
Roles: Engineer + Scientist
Uploads (from ~/SonarANE/artifacts/): sonar_ane_demo.mp4, phase4_physics_rewritten.png, phase3_ane_sonar_proof.png,
respiration_proof.png, phase4_telemetry.json

## Prompt 1 — What's the best thing you've built, discovered, or pulled off?

Built SonarANE: I turned an unmodified, fanless M2 MacBook Air into a 2D super-resolution ultrasonic radar and a
contactless respiration interferometer, and compiled the whole signal-processing and sparse-deconvolution pipeline
into static FP16 convolution graphs that run on Apple's Neural Engine through public CoreML APIs only.

Laptop acoustics hit three walls:

1. Resolution: a 3.5 kHz band (18.8–22.3 kHz at 48 kHz) is limited to 4.9 cm range resolution (ΔR = c/2B).
2. The speakers: continuous ultrasonic transmission made the built-in speakers sag −5.2 dB within 28 s.
3. Compute: 2D Range-Doppler and iterative sparse deconvolution on every chirp eat the CPU.

How I got past all three on bare consumer hardware:

- 878-op unrolled FISTA on the Neural Engine. The dechirp + range DFT is one fixed-weight 1×1 convolution; the
  [1, −2, 1] MTI, Hann and Doppler stages are a second fused 13-op graph; and an 80-iteration FISTA sparse solver
  (soft threshold written as a pair of ReLUs) unrolls into 877 ops. MLComputePlan confirms every op of the combined
  878-op graph is scheduled on the NE. It processes 64 chirps in 503 µs — 20.0× faster than NumPy with Accelerate
  on the same chip (FP16 relative error 0.089 vs FP64) — and resolves two targets 2.4 cm apart in 90 % of trials
  at 30 dB SNR where a Hann FFT resolves 0 % (the 90 %-threshold resolution is 7.2 cm on the NE, 4.8 cm in fp64).
  On real echoes it sharpens a 12 cm FFT peak to 0.6 cm.
- Pulsed TDM stereo MIMO and elliptical backprojection. Driving the left and right speakers in interleaved 16 ms
  pulses (40 % duty) held the speaker level within 0.25 dB over 10 s instead of sagging 5 dB, with −53.4 dB median
  inter-channel isolation (vs −14 dB for up/down-chirp coding). Both channels' CFAR evidence is backprojected as a
  geometric mean on a 1 cm (X, Y) grid and a 4-state Cartesian EKF [x, y, vx, vy] observes both bistatic ranges
  jointly, with the range-Doppler coupling in the measurement model. On a real 8 s notebook sweep it tracked the
  hand on 93 % of frames across X −29…+21 cm, Y 8…31 cm (7 % without the 2 s calibration pre-roll I added), at
  9.9 ms p50 / 13.4 ms p95 block-to-track latency in the live app, and locked onto the seated torso through its
  ~1 mm breathing micro-motion.
- Sub-millimetre direct-path phase interferometry. Referencing every chirp's phase to the 0 cm speaker-to-mic
  leakage bin cancels CoreAudio clock drift and frame-timing error; a 3-bin coherent neighbourhood with clamped,
  amplitude-gated phase steps turns chest motion into a displacement trace: 6.3 mm peak-to-peak, 16.0 breaths/min
  with a 4.4 % spread across spectral, autocorrelation and envelope-peak estimators over a 30 s seated capture.

Everything — captures, telemetry, dashboards, the validation suite and a 60 fps demo rendered from the real
recordings — is in the repo, including a "known limits" section.

## Prompt 2 — What do you believe that almost nobody else does?

Consumer laptops and phones are already mass-produced scientific instruments; people treat their chips and
transducers as sealed boxes because they respect the OS abstractions too much. Every MacBook ships with a matrix
ASIC that sits idle unless you disguise wave physics and iterative solvers as neural-network convolutions, and
stereo speakers and a mic that become a coherent interferometer the moment you stop trusting the audio stack's
timing and reference the phase yourself. You don't need a $50,000 rig to do frontier physical computing; you need
better math on the silicon already on the desk.

## Prompt 3 — What are you obsessed with right now that has nothing to do with any of the above?

[YOUR 2–3 SENTENCES — personal, specific, unrelated to SonarANE]

## Prompt 4 — What are you doing now, and what's next?

Right now I'm building from Pakistan, turning SonarANE into a zero-dependency open-source macOS spatial/vitals SDK
and pushing non-AI scientific workloads (wave inversion, PDE solvers, fluid dynamics) onto consumer Neural Engine
silicon. Next I'm turning this into a company: the $50k goes to going full-time on hardware-level compute and
physical sensing without institutional bottlenecks.

## Submitted 2026-10-22 — reference BF-2026-SH6S2Q (no confirmation email is sent; keep this reference)
Applicant: Muhammed Ahmed, 20, Gujranwala, Pakistan · UMT Lahore · dreamerworld692@gmail.com ·
instagram.com/cryptic_allure_ · heard via social media.
Prompt 3 as submitted: dairy cattle rations (wanda / khal) and milk-fat tracking at the family feed shop in Gujranwala.
