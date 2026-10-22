"""SonarANE Phase 2: fused multi-stage FP16 Range-Doppler graph that stays on the Neural Engine.

One predict() call turns a block of 64 raw chirp frames into a smoothed 2-D Range-Doppler power map:

    x [B, 1920, 1, 64]  raw frames (samples on channels, chirps on width)
      stage 1  conv 1x1  dechirp + Hann + 128-bin range DFT (fmcw.dechirp_dft_kernel)   -> [B, 256, 1, 64]
               transpose                                                                -> [B, 64, 1, 256]
      stage 2  conv 1x1  [1,-2,1] MTI + slow-time Hann + 64-bin Doppler DFT (one matrix) -> [B, 128, 1, 256]
               slice/sub/add  complex combine (Re|Im of range x Re|Im of Doppler)        -> Yr, Yi [B, 64, 1, 128]
               mul/add  power |Y|^2;  depthwise conv 1x5  range smoothing               -> rd_map [B, 64, 1, 128]

MLComputePlan must report every op on the NeuralEngine. Benchmarked against CoreML CPU_ONLY, the same
pipeline in NumPy with FFTs (dechirp + range FFT + MTI + Doppler FFT + smoothing) and the same pipeline
as two Accelerate GEMMs. CPU offload is measured with getrusage (user + sys CPU time per map).

Run:  ~/SonarANE/.venv/bin/python ~/SonarANE/ane_kernels/ane_fused_2d_radar.py
"""
import json
import resource
import shutil
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from scipy.ndimage import uniform_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ane_kernels"))
import fmcw  # noqa: E402
import ane_radar_benchmark as bench  # noqa: E402

S = 64                      # chirps per Range-Doppler block (2.56 s)
D = 64                      # Doppler bins, centred: -12.5 .. +12.1 Hz at PRF 25 Hz
SMOOTH = 5                  # range smoothing (grid bins)
STAGE2_SCALE = 1.0 / 256    # keeps |Y|^2 far below the FP16 max (65504) for unit-normalised input
BATCHES = (1, 4, 16)
INPUT, OUTPUT = "x", "rd_map"
FUSED_JSON = ROOT / "artifacts" / "ane_fused_benchmark.json"


def doppler_matrix():
    """[D, S] complex: centred D-point DFT of the Hann-windowed [1,-2,1] MTI output (S-2 samples)."""
    mti = np.zeros((S - 2, S))
    for i in range(S - 2):
        mti[i, i:i + 3] = (1.0, -2.0, 1.0)
    f = np.exp(-2j * np.pi * np.outer(np.arange(D) - D // 2, np.arange(S - 2)) / D)
    return (f * np.hanning(S - 2)) @ mti


def doppler_axis_hz():
    return (np.arange(D) - D // 2) / (D * fmcw.T_CHIRP)


def model_path(b):
    return bench.MODEL_DIR / f"sonar_fused_rd_B{b}.mlpackage"


def build_model(b):
    w1 = fmcw.dechirp_dft_kernel().astype(np.float16)[:, :, None, None]
    c = doppler_matrix() * STAGE2_SCALE
    w2 = np.concatenate([c.real, c.imag]).astype(np.float16)[:, :, None, None]
    box = np.full((D, 1, 1, SMOOTH), 1.0 / SMOOTH, np.float16)
    k = fmcw.K_BINS

    @mb.program(input_specs=[mb.TensorSpec(shape=(b, fmcw.N, 1, S), dtype=types.fp16)],
                opset_version=ct.target.macOS15)
    def prog(x):
        r = mb.conv(x=x, weight=w1, pad_type="valid", name="stage1_range")              # [b, 2K, 1, S]
        r = mb.transpose(x=r, perm=[0, 3, 2, 1], name="chirps_to_channels")             # [b, S, 1, 2K]
        d = mb.conv(x=r, weight=w2, pad_type="valid", name="stage2_mti_doppler")       # [b, 2D, 1, 2K]
        dr_rr = mb.slice_by_index(x=d, begin=[0, 0, 0, 0], end=[b, D, 1, k])
        di_rr = mb.slice_by_index(x=d, begin=[0, D, 0, 0], end=[b, 2 * D, 1, k])
        dr_ri = mb.slice_by_index(x=d, begin=[0, 0, 0, k], end=[b, D, 1, 2 * k])
        di_ri = mb.slice_by_index(x=d, begin=[0, D, 0, k], end=[b, 2 * D, 1, 2 * k])
        yr = mb.sub(x=dr_rr, y=di_ri)
        yi = mb.add(x=di_rr, y=dr_ri)
        p = mb.add(x=mb.mul(x=yr, y=yr), y=mb.mul(x=yi, y=yi), name="rd_power")      # [b, D, 1, K]
        return mb.conv(x=p, weight=box, groups=D, pad_type="same", name=OUTPUT)

    m = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                   compute_precision=ct.precision.FLOAT16)
    m.short_description = f"SonarANE fused range + MTI-Doppler + power + smoothing, {b} x {S} chirps per call"
    m.user_defined_metadata["kernel_tag"] = fused_tag()
    path = model_path(b)
    if path.exists():
        shutil.rmtree(path)
    m.save(str(path))
    return path


def fused_tag():
    return bench.kernel_tag(fmcw.dechirp_dft_kernel()) + f"-S{S}D{D}"


def load(b=1, units=ct.ComputeUnit.CPU_AND_NE):
    """Compiled fused model for batch b; rebuilt if missing or built for another band."""
    pkg = model_path(b)
    compiled = pkg.with_suffix(".mlmodelc")
    tag = (ct.models.MLModel(str(pkg), skip_model_load=True).user_defined_metadata.get("kernel_tag")
           if pkg.exists() else None)
    if not compiled.exists() or tag != fused_tag():
        compiled = bench.compile_once(build_model(b))
    return ct.models.CompiledMLModel(str(compiled), compute_units=units), compiled


def to_input(blocks):
    """[b, S, N] frames -> fp16 [b, N, 1, S] with one gain per block (Doppler needs slow-time coherence)."""
    g = 1.0 / np.abs(blocks).max(axis=(1, 2), keepdims=True)
    x = np.ascontiguousarray(np.transpose(blocks * g, (0, 2, 1))[:, :, None, :]).astype(np.float16)
    return x, g[:, 0, 0]


def rd_maps(model, blocks):
    """Run blocks [b, S, N] through the fused model -> power maps [b, D, K] in input units^2."""
    x, g = to_input(blocks)
    p = model.predict({INPUT: x})[OUTPUT][:, :, 0, :].astype(np.float64)
    return p * (fmcw.amp_scale() / (STAGE2_SCALE * g[:, None, None])) ** 2


# ---------------------------------------------------------------- CPU references
def numpy_fft_pipeline(blocks):
    out = []
    for f in blocks:
        X = fmcw.fft_pipeline(f) * fmcw.amp_scale()                        # [S, K]
        y = fmcw.mti2(X) * np.hanning(S - 2)[:, None]
        rd = np.fft.fftshift(np.fft.fft(y, n=D, axis=0), axes=0)
        out.append(uniform_filter1d(np.abs(rd) ** 2, SMOOTH, axis=1, mode="constant"))
    return np.stack(out)


class GemmPipeline:
    """Same math as the fused graph as two Accelerate GEMMs (fp32 range, complex64 Doppler)."""

    def __init__(self):
        self.w = fmcw.dechirp_dft_kernel().astype(np.float32)
        self.c = (doppler_matrix().T * fmcw.amp_scale()).astype(np.complex64)   # [S, D]

    def __call__(self, blocks):
        out = []
        k = fmcw.K_BINS
        for f in blocks:
            r = self.w @ f.T                                                # [2K, S]
            rd = (r[:k] + 1j * r[k:]).astype(np.complex64) @ self.c         # [K, D]
            p = rd.real ** 2 + rd.imag ** 2
            out.append(uniform_filter1d(p.T, SMOOTH, axis=1, mode="constant"))
        return np.stack(out)


def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def bench_fn(fn, iters, warmup=20):
    for _ in range(warmup):
        fn()
    t = np.empty(iters)
    c0 = cpu_seconds()
    w0 = time.perf_counter()
    for i in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        t[i] = (time.perf_counter_ns() - t0) / 1e3
    wall = time.perf_counter() - w0
    cpu = cpu_seconds() - c0
    return {"median_us": float(np.median(t)), "p10_us": float(np.percentile(t, 10)),
            "p90_us": float(np.percentile(t, 90)), "cpu_us_per_call": cpu / iters * 1e6,
            "cpu_utilisation": cpu / wall, "iters": iters}


def synthetic_blocks(b, seed=0):
    """Direct path + static wall + a hand at 30 cm receding at PRF/4 Doppler (mid MTI passband in any band).

    A hand speed whose Doppler aliases near 0 Hz would sit in the MTI notch, and the relative error would
    then measure FP16 cancellation residue instead of the map.
    """
    rng = np.random.default_rng(seed)
    n = np.arange(fmcw.N)
    v = (1 / fmcw.T_CHIRP / 4) * fmcw.C_SOUND / (2 * fmcw.FC)          # m/s, Doppler = PRF/4
    out = []
    for _ in range(b):
        fr = []
        for m in range(S):
            r = 0.30 + v * m * fmcw.T_CHIRP
            x = 0.3 * np.cos(fmcw.chirp_phase(n)) + 0.02 * np.cos(fmcw.chirp_phase(n - 2 * 0.9 / 343 * 48000))
            x = x + 0.005 * np.cos(fmcw.chirp_phase(n - 2 * r / fmcw.C_SOUND * fmcw.FS))
            fr.append(x + 0.001 * rng.standard_normal(fmcw.N))
        out.append(fr)
    return np.asarray(out, np.float32)


def main():
    res = {"S": S, "D": D, "smooth_bins": SMOOTH, "band_hz": [fmcw.F0, fmcw.F1], "kernel_tag": fused_tag(),
           "batches": {}}
    gemm = GemmPipeline()
    for b in BATCHES:
        print(f"[fused] B={b} ...", flush=True)
        blocks = synthetic_blocks(b)
        ref = gemm(blocks.astype(np.float64))
        r = res["batches"][str(b)] = {"chirps_per_call": b * S}
        for name in ("CPU_AND_NE", "CPU_ONLY"):
            model, compiled = load(b, getattr(ct.ComputeUnit, name))
            plan = bench.compute_plan(compiled, getattr(ct.ComputeUnit, name))
            out = rd_maps(model, blocks)
            far = fmcw.range_axis_m() * 100 >= 10
            x, _ = to_input(blocks)
            t = bench_fn(lambda: model.predict({INPUT: x}), {1: 600, 4: 300}.get(b, 100))
            r[f"coreml_{name.lower()}"] = {
                **t, "chirps_per_s": b * S / (t["median_us"] * 1e-6),
                "rel_err_vs_fp64": float(np.linalg.norm(out - ref) / np.linalg.norm(ref)),
                "rel_err_beyond_10cm": float(np.linalg.norm((out - ref)[..., far]) / np.linalg.norm(ref[..., far])),
                "ops": [f"{o['op'].split('.')[-1]}->{o['preferred']}" for o in plan if o["op"] != "const"],
                "all_on_ne": all(o["preferred"] == "NeuralEngine" for o in plan if o["op"] != "const")}
        for name, fn, it in (("numpy_fft_pipeline", lambda: numpy_fft_pipeline(blocks), 60),
                             ("numpy_gemm_pipeline", lambda: gemm(blocks), {1: 600, 4: 300}.get(b, 100))):
            t = bench_fn(fn, it, warmup=5)
            r[name] = {**t, "chirps_per_s": b * S / (t["median_us"] * 1e-6)}
        ne = r["coreml_cpu_and_ne"]
        for base in ("numpy_gemm_pipeline", "numpy_fft_pipeline", "coreml_cpu_only"):
            r[f"speedup_vs_{base}"] = r[base]["median_us"] / ne["median_us"]
            r[f"cpu_offload_vs_{base}_pct"] = 100 * (1 - ne["cpu_us_per_call"] / r[base]["cpu_us_per_call"])
        print(f"  NE fused   {ne['median_us']:8.1f} us  cpu {ne['cpu_us_per_call']:7.1f} us  "
              f"all_on_ne={ne['all_on_ne']}  err={ne['rel_err_vs_fp64']:.1e} (>=10 cm {ne['rel_err_beyond_10cm']:.1e}), "
              f"CoreML CPU err {r['coreml_cpu_only']['rel_err_vs_fp64']:.1e}")
        for name in ("coreml_cpu_only", "numpy_gemm_pipeline", "numpy_fft_pipeline"):
            print(f"  {name:20s} {r[name]['median_us']:8.1f} us  cpu {r[name]['cpu_us_per_call']:7.1f} us  "
                  f"-> NE speedup {r['speedup_vs_' + name]:.2f}x, CPU offload "
                  f"{r['cpu_offload_vs_' + name + '_pct']:.0f}%")
    FUSED_JSON.write_text(json.dumps(res, indent=2))
    print(f"[fused] wrote {FUSED_JSON}")


if __name__ == "__main__":
    main()
