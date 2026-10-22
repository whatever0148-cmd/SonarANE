"""SonarANE Phase 4: sparse super-resolution ranging as an unrolled-FISTA FP16 graph on the Neural Engine.

Model: a 16 ms TDM chirp frame x (768 samples, aligned on its speaker's direct path, static background
removed) is a sparse sum of delayed echoes. Dictionary D has, for every range r_j on a 0.6 cm grid from
0 to 90 cm, the two quadrature atoms of the Tukey-windowed chirp delayed by 2 r_j / c (unit norm), so any
echo phase is a*cos-atom + b*sin-atom. FISTA solves min ||x - D z||^2 / 2 + lam ||z||_1 with a separable
(Re/Im) soft threshold. In scaled form (z~ = z / mu, mu = 1 / ||D||^2):

    c = D^T x                                   (once: the fine-grid matched filter)
    v = (I - mu D^T D) y_k + c                  conv 1x1, add
    z_k+1 = relu(v - lam) - relu(-v - lam)      soft threshold: 2 relu, 2 sub, 1 mul
    y_k+1 = z_k+1 + beta_k (z_k+1 - z_k)        FISTA momentum: sub, mul, add
    profile = sqrt(z_a^2 + z_b^2)               per range cell

Unrolled for ITERS iterations, every op is NE-friendly. Inputs are normalised per chirp so that
max |D^T x| = 1 and lam is a fraction of that.

Sparse recovery is not free resolution: it assumes few point-like reflectors and enough SNR, and how
closely two echoes can be separated depends on their relative phase. The resolution claim here is the
measured two-target success rate versus separation, against a Hann FFT on the same 0.6 cm grid.

Run:  ~/SonarANE/.venv/bin/python ~/SonarANE/ane_kernels/ane_superres_mimo.py
"""
import json
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
from scipy.signal import find_peaks, get_window
from scipy.signal.windows import tukey

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ane_kernels"))
import fmcw  # noqa: E402
import mimo  # noqa: E402
import ane_radar_benchmark as bench  # noqa: E402

GRID_CM = np.arange(0.0, 90.0 + 1e-9, 0.6)
NG = len(GRID_CM)
ITERS = 80
LAM = 0.04
S = 64
INPUT, OUTPUT = "x", "profile"
SR_JSON = ROOT / "artifacts" / "ane_superres_benchmark.json"


def dictionary(alpha=0.15):
    """[NC, 2*NG] real dictionary: quadrature atoms of the delayed Tukey chirp, unit norm."""
    n = np.arange(mimo.NC, dtype=np.float64)
    w = tukey(mimo.NC, alpha)
    cols = []
    for r in GRID_CM:
        d = 2 * r / 100 / fmcw.C_SOUND * mimo.FS
        env = np.interp(n - d, n, w, left=0.0, right=0.0)
        ph = mimo.chirp_phase(n - d)
        cols += [env * np.cos(ph), env * np.sin(ph)]
    a = np.array(cols).T
    return a / np.linalg.norm(a, axis=0)


def fista_mats(a):
    g = a.T @ a
    mu = 1.0 / np.linalg.eigvalsh(g).max()
    return np.eye(len(g)) - mu * g, mu


def momentum(iters=ITERS):
    t, out = 1.0, []
    for _ in range(iters):
        t2 = (1 + np.sqrt(1 + 4 * t * t)) / 2
        out.append((t - 1) / t2)
        t = t2
    return out


def normalise(frames, a):
    c = frames @ a                                                  # [M, 2NG] fine-grid matched filter
    g = 1.0 / np.abs(c).max(axis=1, keepdims=True)
    return frames * g, g


def fista_numpy(frames, a=None, iters=ITERS, lam=LAM, dtype=np.float64):
    """Reference FISTA (scaled form) -> |z| per range cell [M, NG]."""
    a = dictionary() if a is None else a
    w, _ = fista_mats(a)
    x, _ = normalise(np.asarray(frames, np.float64), a)
    w, c = w.astype(dtype), (x @ a).astype(dtype)                  # rows = chirps
    z = np.zeros_like(c)
    y = z
    for beta in momentum(iters):
        v = y @ w.T + c
        z_new = np.maximum(v - lam, 0) - np.maximum(-v - lam, 0)
        y = z_new + beta * (z_new - z)
        z = z_new
    return np.hypot(z[:, 0::2], z[:, 1::2])


def fft_fine(frames):
    """Hann matched filter on the same 0.6 cm grid (16x zero-padded FFT equivalent) -> [M, NG] magnitude."""
    n = np.arange(mimo.NC)
    h = get_window("hann", mimo.NC, fftbins=True)
    k = GRID_CM / 100 * 2 / fmcw.C_SOUND * mimo.SLOPE * mimo.TC     # fractional bin of each range
    m = h * np.exp(1j * mimo.chirp_phase(n)) * np.exp(-2j * np.pi * np.outer(k, n) / mimo.NC)
    return np.abs(frames @ m.T)


# ---------------------------------------------------------------- NE graph
def model_path(s=S):
    return bench.MODEL_DIR / f"sonar_superres_fista{ITERS}_S{s}.mlpackage"


def sr_tag(a):
    import hashlib
    return f"{fmcw.F0:.0f}-{fmcw.F1:.0f}Hz-NC{mimo.NC}-G{NG}-I{ITERS}-L{LAM}-" + \
        hashlib.sha1(a.astype(np.float16).tobytes()).hexdigest()[:10]


def build_model(s=S):
    a = dictionary()
    w, _ = fista_mats(a)
    k2 = 2 * NG
    at = a.T.astype(np.float16)[:, :, None, None]                   # [2NG, NC, 1, 1]
    wt = w.astype(np.float16)[:, :, None, None]                     # [2NG, 2NG, 1, 1]
    betas = momentum()

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, mimo.NC, 1, s), dtype=types.fp16)],
                opset_version=ct.target.macOS15)
    def prog(x):
        c = mb.conv(x=x, weight=at, pad_type="valid", name="matched_filter")
        z = mb.sub(x=mb.relu(x=mb.sub(x=c, y=np.float16(LAM))), y=mb.relu(x=mb.sub(x=mb.mul(x=c, y=np.float16(-1)),
                                                                                  y=np.float16(LAM))))
        y = z
        for i, beta in enumerate(betas[1:]):
            v = mb.add(x=mb.conv(x=y, weight=wt, pad_type="valid"), y=c)
            z_new = mb.sub(x=mb.relu(x=mb.sub(x=v, y=np.float16(LAM))),
                           y=mb.relu(x=mb.sub(x=mb.mul(x=v, y=np.float16(-1)), y=np.float16(LAM))))
            y = mb.add(x=z_new, y=mb.mul(x=mb.sub(x=z_new, y=z), y=np.float16(beta)))
            z = z_new
        zz = mb.reshape(x=z, shape=[1, NG, 2, s])                    # (a, b) pairs per range cell
        p = mb.reduce_sum(x=mb.mul(x=zz, y=zz), axes=[2], keep_dims=True)
        return mb.sqrt(x=p, name=OUTPUT)

    m = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                   compute_precision=ct.precision.FLOAT16)
    m.short_description = f"SonarANE unrolled FISTA ({ITERS} it) sparse ranging, 0.6 cm grid, {s} chirps per call"
    m.user_defined_metadata["kernel_tag"] = sr_tag(a)
    path = model_path(s)
    if path.exists():
        shutil.rmtree(path)
    m.save(str(path))
    return path


def load(s=S, units=ct.ComputeUnit.CPU_AND_NE):
    pkg = model_path(s)
    compiled = pkg.with_suffix(".mlmodelc")
    tag = (ct.models.MLModel(str(pkg), skip_model_load=True).user_defined_metadata.get("kernel_tag")
           if pkg.exists() else None)
    if not compiled.exists() or tag != sr_tag(dictionary()):
        compiled = bench.compile_once(build_model(s))
    return ct.models.CompiledMLModel(str(compiled), compute_units=units), compiled


def fista_ne(model, frames, a=None):
    """[M, NC] frames -> |z| [M, NG] via the NE graph (banks of S)."""
    a = dictionary() if a is None else a
    x, _ = normalise(np.asarray(frames, np.float64), a)
    m = len(x)
    xb = np.concatenate([x, np.zeros((-m % S, mimo.NC))]).astype(np.float16)
    out = []
    for i in range(0, len(xb), S):
        blk = np.ascontiguousarray(xb[i:i + S].T[None, :, None, :])
        out.append(model.predict({INPUT: blk})[OUTPUT].reshape(NG, S).T)
    return np.concatenate(out)[:m].astype(np.float64)


# ---------------------------------------------------------------- resolution test
def two_target_frames(sep_cm, trials, snr_db, r0_cm=40.0, seed=0):
    rng = np.random.default_rng(seed)
    n = np.arange(mimo.NC, dtype=np.float64)
    w = tukey(mimo.NC, 0.15)
    out = []
    for _ in range(trials):
        x = np.zeros(mimo.NC)
        for r, th in ((r0_cm + rng.uniform(-0.3, 0.3), 0.0), (None, rng.uniform(0, 2 * np.pi))):
            r = r0_cm + sep_cm + rng.uniform(-0.3, 0.3) if r is None else r
            d = 2 * r / 100 / fmcw.C_SOUND * mimo.FS
            x += np.interp(n - d, n, w, left=0, right=0) * np.cos(mimo.chirp_phase(n - d) + th)
        sig = np.sqrt(np.mean(x ** 2))
        out.append(x + sig * 10 ** (-snr_db / 20) * rng.standard_normal(mimo.NC))
    return np.array(out)


def resolved(profile, r0, sep, tol=1.2, rel=0.25):
    pk, _ = find_peaks(np.concatenate([[0], profile, [0]]), height=rel * profile.max())
    pos = GRID_CM[pk - 1]
    hit = [np.flatnonzero(np.abs(pos - r) <= tol) for r in (r0, r0 + sep)]
    return bool(len(hit[0]) and len(hit[1]) and set(hit[0]) != set(hit[1]))


def resolution_test(model=None, seps=(1.2, 1.8, 2.4, 3.0, 3.6, 4.8, 6.0, 7.2, 9.8), trials=40):
    a = dictionary()
    res = {}
    for snr in (30.0, 20.0):
        rows = {}
        for sep in seps:
            fr = two_target_frames(sep, trials, snr, seed=int(sep * 10 + snr))
            methods = {"fft": fft_fine(fr), "fista_fp64": fista_numpy(fr, a)}
            if model is not None:
                methods["fista_ne_fp16"] = fista_ne(model, fr, a)
            rows[f"{sep:.1f}"] = {k: float(np.mean([resolved(p, 40.0, sep) for p in v])) for k, v in methods.items()}
        res[f"snr_{snr:.0f}dB"] = rows
    return res


def threshold(rows, key, level=0.9):
    for sep, v in sorted(rows.items(), key=lambda kv: float(kv[0])):
        if v.get(key, 0) >= level and all(rows[s][key] >= level for s in rows if float(s) >= float(sep)):
            return float(sep)
    return None


def main():
    a = dictionary()
    print(f"[sr] dictionary {a.shape}, mu {fista_mats(a)[1]:.4f}, building NE model ...", flush=True)
    model, compiled = load()
    plan = bench.compute_plan(compiled, ct.ComputeUnit.CPU_AND_NE)
    ops = [o for o in plan if o["op"] != "const"]
    off_ne = sorted({o["op"].split(".")[-1] for o in ops if o["preferred"] != "NeuralEngine"})
    print(f"[sr] {len(ops)} ops, off-NE: {off_ne or 'none'}", flush=True)
    res = {"grid_cm": 0.6, "range_cm": [0.0, 90.0], "iters": ITERS, "lambda": LAM, "ops": len(ops),
           "off_ne_op_types": off_ne, "all_on_ne": not off_ne}
    # accuracy vs fp64 on a two-target scene
    fr = two_target_frames(3.0, S, 30.0)
    ne = fista_ne(model, fr, a)
    ref = fista_numpy(fr, a)
    res["ne_rel_err_vs_fp64"] = float(np.linalg.norm(ne - ref) / np.linalg.norm(ref))
    # benchmark one bank
    x, _ = normalise(fr, a)
    blk = np.ascontiguousarray(x.T[None, :, None, :]).astype(np.float16)
    w32 = fista_mats(a)[0].astype(np.float32)
    a32 = a.astype(np.float32)
    x32 = x.astype(np.float32)

    def cpu():
        c = x32 @ a32
        z = np.zeros_like(c)
        y = z
        for beta in momentum():
            v = y @ w32.T + c
            zn = np.maximum(v - LAM, 0) - np.maximum(-v - LAM, 0)
            y = zn + beta * (zn - z)
            z = zn
        return np.hypot(z[:, 0::2], z[:, 1::2])

    for name, fn in (("ne", lambda: model.predict({INPUT: blk})), ("numpy_fp32_accelerate", cpu)):
        for _ in range(10):
            fn()
        t = []
        for _ in range(100):
            t0 = time.perf_counter_ns()
            fn()
            t.append((time.perf_counter_ns() - t0) / 1e3)
        res[f"{name}_us_per_{S}_chirps"] = float(np.median(t))
    res["ne_speedup_vs_numpy"] = res[f"numpy_fp32_accelerate_us_per_{S}_chirps"] / res[f"ne_us_per_{S}_chirps"]
    print(f"[sr] NE {res[f'ne_us_per_{S}_chirps']:.0f} us vs NumPy {res[f'numpy_fp32_accelerate_us_per_{S}_chirps']:.0f} us "
          f"per {S} chirps; NE err vs fp64 {res['ne_rel_err_vs_fp64']:.2e}", flush=True)
    print("[sr] two-target resolution test ...", flush=True)
    res["two_target"] = rt = resolution_test(model)
    for snr, rows in rt.items():
        res[f"resolution_90pct_cm_{snr}"] = {k: threshold(rows, k) for k in next(iter(rows.values()))}
        print(f"  {snr}: " + " | ".join(f"{s} cm: " + "/".join(f"{v * 100:.0f}" for v in r.values()) for s, r in rows.items()))
        print(f"  {snr} 90 % resolution (fft / fista fp64 / fista NE): {res[f'resolution_90pct_cm_{snr}']}")
    SR_JSON.write_text(json.dumps(res, indent=2))
    print(f"[sr] wrote {SR_JSON}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- combined live model: range FFT + FISTA
def mimo_model_path(s=S):
    return bench.MODEL_DIR / f"sonar_mimo_range_fista{ITERS}_S{s}.mlpackage"


def build_mimo_model(s=S):
    """Two inputs, two outputs, one NE call: xr (raw TDM frames) -> range_fft [1, 2K, 1, s];
    x (static-background-removed frames) -> FISTA profile [1, NG, 1, s]. Left and right speaker frames
    share the s axis. On its own the small TDM range conv is always placed on the CPU by CoreML; inside
    this graph it is scheduled with the FISTA stages."""
    a = dictionary()
    w, _ = fista_mats(a)
    at = a.T.astype(np.float16)[:, :, None, None]
    wt = w.astype(np.float16)[:, :, None, None]
    wr = mimo.dechirp_dft_kernel().astype(np.float16)[:, :, None, None]
    betas = momentum()

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, mimo.NC, 1, s), dtype=types.fp16),
                             mb.TensorSpec(shape=(1, mimo.NC, 1, s), dtype=types.fp16)],
                opset_version=ct.target.macOS15)
    def prog(x, xr):
        rng = mb.conv(x=xr, weight=wr, pad_type="valid", name="range_fft")
        c = mb.conv(x=x, weight=at, pad_type="valid", name="matched_filter")
        z = mb.sub(x=mb.relu(x=mb.sub(x=c, y=np.float16(LAM))),
                   y=mb.relu(x=mb.sub(x=mb.mul(x=c, y=np.float16(-1)), y=np.float16(LAM))))
        y = z
        for beta in betas[1:]:
            v = mb.add(x=mb.conv(x=y, weight=wt, pad_type="valid"), y=c)
            z_new = mb.sub(x=mb.relu(x=mb.sub(x=v, y=np.float16(LAM))),
                           y=mb.relu(x=mb.sub(x=mb.mul(x=v, y=np.float16(-1)), y=np.float16(LAM))))
            y = mb.add(x=z_new, y=mb.mul(x=mb.sub(x=z_new, y=z), y=np.float16(beta)))
            z = z_new
        zz = mb.reshape(x=z, shape=[1, NG, 2, s])
        prof = mb.sqrt(x=mb.reduce_sum(x=mb.mul(x=zz, y=zz), axes=[2], keep_dims=True), name=OUTPUT)
        return rng, prof

    m = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                   compute_precision=ct.precision.FLOAT16)
    m.user_defined_metadata["kernel_tag"] = sr_tag(a) + "-mimo"
    path = mimo_model_path(s)
    if path.exists():
        shutil.rmtree(path)
    m.save(str(path))
    return path


def load_mimo(s=S):
    pkg = mimo_model_path(s)
    compiled = pkg.with_suffix(".mlmodelc")
    tag = (ct.models.MLModel(str(pkg), skip_model_load=True).user_defined_metadata.get("kernel_tag")
           if pkg.exists() else None)
    if not compiled.exists() or tag != sr_tag(dictionary()) + "-mimo":
        compiled = bench.compile_once(build_mimo_model(s))
    plan = bench.compute_plan(compiled, ct.ComputeUnit.CPU_AND_NE)
    return ct.models.CompiledMLModel(str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE), plan


def run_mimo(model, raw, bg, a=None):
    """raw, bg: [M, NC] -> (complex range spectra [M, K], FISTA profiles [M, NG], call times ms)."""
    a = dictionary() if a is None else a
    gr = 1.0 / np.abs(raw).max(axis=1, keepdims=True)
    xb, _ = normalise(np.asarray(bg, np.float64), a)
    m = len(raw)
    pad = -m % S
    r16 = np.concatenate([raw * gr, np.zeros((pad, mimo.NC))]).astype(np.float16)
    b16 = np.concatenate([xb, np.zeros((pad, mimo.NC))]).astype(np.float16)
    rng, prof, ms = [], [], []
    for i in range(0, len(r16), S):
        t0 = time.perf_counter()
        out = model.predict({"xr": np.ascontiguousarray(r16[i:i + S].T[None, :, None, :]),
                             INPUT: np.ascontiguousarray(b16[i:i + S].T[None, :, None, :])})
        ms.append((time.perf_counter() - t0) * 1e3)
        rng.append(out["range_fft"].reshape(mimo.C_OUT, S))
        prof.append(out[OUTPUT].reshape(NG, S).T)
    X = mimo.to_complex(np.concatenate(rng, axis=1)[:, :m]) * mimo.amp_scale() / gr
    return X, np.concatenate(prof)[:m].astype(np.float64), ms
