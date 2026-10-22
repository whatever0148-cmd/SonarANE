"""SonarANE Phase 1: FMCW matched filter + range-FFT as a static FP16 1x1 conv on the Neural Engine.

Builds the model directly with the coremltools MIL Builder (no PyTorch), compiles it for
CPU_AND_NE and CPU_ONLY, checks per-op device placement with MLComputePlan, and benchmarks
CoreML predict() latency/throughput against NumPy (Accelerate BLAS) and an FFT pipeline.

Run:  ~/SonarANE/.venv/bin/python ~/SonarANE/ane_kernels/ane_radar_benchmark.py
"""
import hashlib
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.models.compute_plan import MLComputePlan
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import fmcw  # noqa: E402

MODEL_DIR = ROOT / "artifacts" / "models"
BENCH_JSON = ROOT / "artifacts" / "ane_benchmark.json"
# (a) single chirp; S=8 = one chirp in an 8-lane tensor (smallest shape CoreML schedules on the NE);
# (b) 64/128-chirp Range-Doppler banks; 1024 = NE throughput ceiling.
BATCHES = (1, 8, 64, 128, 1024)
UNITS = {"CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE, "CPU_ONLY": ct.ComputeUnit.CPU_ONLY}
INPUT, OUTPUT = "x", "range_fft"


def kernel_tag(weight):
    """Band + FP16 weight hash, stored in model metadata so stale models are never reused."""
    h = hashlib.sha1(weight.astype(np.float16).tobytes()).hexdigest()[:12]
    return f"{fmcw.F0:.0f}-{fmcw.F1:.0f}Hz-{h}"


def model_path(s):
    return MODEL_DIR / f"sonar_range_fft_S{s}.mlpackage"


def build_model(s, weight):
    """x:[1, N, 1, S] fp16 -> conv(W:[2K, N, 1, 1]) -> [1, 2K, 1, S] fp16."""
    w = weight.astype(np.float16)[:, :, None, None]

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, fmcw.N, 1, s), dtype=types.fp16)],
                opset_version=ct.target.macOS15)
    def prog(x):
        return mb.conv(x=x, weight=w, strides=[1, 1], pad_type="valid", name=OUTPUT)

    m = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                   compute_precision=ct.precision.FLOAT16)
    m.short_description = f"SonarANE FMCW dechirp + {fmcw.K_BINS}-bin range DFT, {s} chirp(s) per call"
    m.user_defined_metadata["kernel_tag"] = kernel_tag(weight)
    path = model_path(s)
    if path.exists():
        shutil.rmtree(path)
    m.save(str(path))
    return path


def compile_once(pkg):
    """Compile .mlpackage -> .mlmodelc next to it (reused by benchmark and live test)."""
    dst = pkg.with_suffix(".mlmodelc")
    if dst.exists():
        shutil.rmtree(dst)
    tmp = ct.models.utils.compile_model(str(pkg))
    shutil.move(tmp, dst)
    return dst


def compute_plan(compiled, units):
    plan = MLComputePlan.load_from_path(path=str(compiled), compute_units=units)
    ops = []
    for fn_name, fn in plan.model_structure.program.functions.items():
        for op in fn.block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            cost = plan.get_estimated_cost_for_mlprogram_operation(op)
            dev = lambda d: type(d).__name__.replace("ML", "").replace("ComputeDevice", "")
            ops.append({
                "function": fn_name,
                "op": op.operator_name,
                "preferred": dev(usage.preferred_compute_device) if usage else None,
                "supported": [dev(d) for d in usage.supported_compute_devices] if usage else [],
                "est_cost": round(cost.weight, 4) if cost else None,
            })
    return ops


def timeit(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    t = np.empty(iters)
    for i in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        t[i] = time.perf_counter_ns() - t0
    t /= 1e3
    return {"median_us": float(np.median(t)), "p10_us": float(np.percentile(t, 10)),
            "p90_us": float(np.percentile(t, 90)), "mean_us": float(t.mean()), "iters": iters}


def synthetic_frames(s, ranges_m=(0.30, 0.75), amps=(0.05, 0.02), seed=0):
    """Direct path + two echoes + noise, in the same units as a recording."""
    rng = np.random.default_rng(seed)
    n = np.arange(fmcw.N)
    x = 0.3 * np.cos(fmcw.chirp_phase(n))
    for r, a in zip(ranges_m, amps):
        d = 2 * r / fmcw.C_SOUND * fmcw.FS
        x = x + a * np.cos(fmcw.chirp_phase(n - d))
    frames = x[None, :] + 0.002 * rng.standard_normal((s, fmcw.N))
    return frames.astype(np.float32)


def bench_batch(s, weight, weight64):
    pkg = build_model(s, weight)
    compiled = compile_once(pkg)
    frames = synthetic_frames(s)
    xin = np.ascontiguousarray(frames.T[None, :, None, :]).astype(np.float16)  # [1, N, 1, S]
    ref = (weight64 @ frames.T.astype(np.float64))                               # [2K, S]
    iters = {1: 3000, 8: 2000, 64: 1000, 128: 1000}.get(s, 200)
    res = {"shape_in": [1, fmcw.N, 1, s], "shape_out": [1, fmcw.C_OUT, 1, s],
           "model": str(pkg.relative_to(ROOT)), "coreml": {}}

    for name, units in UNITS.items():
        t0 = time.perf_counter()
        model = ct.models.CompiledMLModel(str(compiled), compute_units=units)
        load_ms = (time.perf_counter() - t0) * 1e3
        out = model.predict({INPUT: xin})[OUTPUT].astype(np.float64).reshape(fmcw.C_OUT, s)
        rel_err = float(np.linalg.norm(out - ref) / np.linalg.norm(ref))
        xc = fmcw.to_complex(out)
        mag = np.abs(xc[0])
        pk, _ = find_peaks(mag[4:])
        pk = pk[np.argsort(mag[4:][pk])[-2:]] + 4
        peaks_cm = sorted(round(float(fmcw.range_axis_m()[k] * 100), 2) for k in pk)
        t = timeit(lambda: model.predict({INPUT: xin}), iters, warmup=50)
        t["chirps_per_s"] = s / (t["median_us"] * 1e-6)
        plan = compute_plan(compiled, units)
        conv_dev = next(o["preferred"] for o in plan if o["op"].endswith("conv"))
        res["coreml"][name] = {"conv_device": conv_dev, "load_ms": load_ms, "rel_err_vs_fp64": rel_err,
                               "synthetic_echo_peaks_cm": peaks_cm, "compute_plan": plan, **t}

    w32 = weight.astype(np.float32)
    x32 = np.ascontiguousarray(frames.T)                       # [N, S]
    t = timeit(lambda: w32 @ x32, iters, warmup=50)
    t["chirps_per_s"] = s / (t["median_us"] * 1e-6)
    res["numpy_gemm_fp32_accelerate"] = t
    t = timeit(lambda: fmcw.fft_pipeline(frames), min(iters, 300), warmup=10)
    t["chirps_per_s"] = s / (t["median_us"] * 1e-6)
    res["numpy_dechirp_fft"] = t
    return res


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    weight = fmcw.dechirp_dft_kernel()
    weight64 = weight.astype(np.float16).astype(np.float64)
    out = {
        "coremltools": ct.__version__, "numpy": np.__version__,
        "dsp": {"fs": fmcw.FS, "f0": fmcw.F0, "f1": fmcw.F1, "T_chirp_s": fmcw.T_CHIRP,
                "C_in": fmcw.N, "C_out": fmcw.C_OUT, "range_bins": fmcw.K_BINS,
                "range_grid_cm": fmcw.RANGE_RES_M * fmcw.BIN_STEP * 100,
                "range_res_cm": fmcw.RANGE_RES_M * 100,
                "max_range_cm": float(fmcw.range_axis_m()[-1] * 100),
                "weight_mb_fp16": weight.size * 2 / 1e6,
                "flops_per_chirp": 2 * weight.size},
        "synthetic_truth_cm": [30.0, 75.0],
        "batches": {},
    }
    for s in BATCHES:
        print(f"[bench] S={s} ...", flush=True)
        out["batches"][str(s)] = r = bench_batch(s, weight, weight64)
        for name, v in r["coreml"].items():
            print(f"  CoreML {name:10s} {v['median_us']:9.1f} us  {v['chirps_per_s']:12.0f} chirps/s  "
                  f"err={v['rel_err_vs_fp64']:.2e}  peaks={v['synthetic_echo_peaks_cm']}  conv->{v['conv_device']}")
        for name in ("numpy_gemm_fp32_accelerate", "numpy_dechirp_fft"):
            v = r[name]
            print(f"  {name:28s} {v['median_us']:9.1f} us  {v['chirps_per_s']:12.0f} chirps/s")
    BENCH_JSON.write_text(json.dumps(out, indent=2))
    print(f"[bench] wrote {BENCH_JSON}")


if __name__ == "__main__":
    main()
