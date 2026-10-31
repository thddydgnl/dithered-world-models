"""E8b (rss_ext/DESIGN_ext.md 1.6; descriptive, no verdict): low-precision GEMM latency and the cost of FP4 activation
quantization on a Blackwell GPU, with the PyTorch of the image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 only
(no new kernels; every path is a PyTorch operator; unsupported paths are recorded with their error).

GEMM shapes: the layers of the 317M dynamics / reward MLPs (width 4096) at M = N (candidates) in {512, 2048, 8192}
  L0 (M, K 1408 = 1380 padded to 128, N 4096), L1 (M, 4096, 4096), L2dyn (M, 4096, 1376), L2rew (M, 4096, 128 = 101 padded)
  the same (padded) shapes for every path
paths: fp32 (TF32 off), tf32, bf16 (torch.mm), int8 (torch._int_mm), fp8 e4m3 with a per-tensor scale and with row /
  column scales (torch._scaled_mm, bf16 out), nvfp4 (float4_e2m1fn_x2, e4m3 scales per 16 along K, swizzled blocked
  layout) and mxfp4 (e8m0 scales per 32) via torch._scaled_mm; the operand values and scales are random / constant
  (latency only)
FP4 activation quantization (M, 4096) fp32 -> fake-quant in PyTorch ops, E9's definitions (DESIGN_ext 2.3):
  nvfp4: blocks of 16, s_b = E4M3(amax_b / (6 g)) g, g = hi / (6 * 448) with hi = 4.0; mxfp4: blocks of 32,
  s_b = 2^(floor(log2 amax_b) - 2); det = round to the nearest E2M1 value (ties to the even mantissa), sh-iter =
  stochastic rounding between the neighbouring E2M1 values with one u per column shared by all rows, fresh every call
  (torch.rand on the default CUDA generator, also inside CUDA graphs); eager and captured as one CUDA graph
timing: CUDA event pair per call followed by a device sync; block = 30 warmup + max(10, ceil(1 s / t)) timed calls;
  3 rounds, conditions shuffled per round (random.Random(0); the image has no numpy); value = median of the 3
  block medians
usage: python e8b_bench.py --out /workspace/results/e8b/e8b.json
"""
import argparse
import json
import math
import os
import subprocess
import time
import traceback

import random
import statistics

import torch

MS = (512, 2048, 8192)
SHAPES = {"L0": (1408, 4096), "L1": (4096, 4096), "L2dyn": (4096, 1376), "L2rew": (4096, 128)}   # (K, N)
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def log(*a):
    print(time.strftime("%H:%M:%S", time.gmtime()), *a, flush=True)


def gpu_state():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw,clocks_event_reasons.active",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as ex:  # noqa: BLE001
        return repr(ex)


def time_block(fn, warm, n):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n)]
    for e0, e1 in ev:
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
    return [e0.elapsed_time(e1) for e0, e1 in ev]


def round_up(x, m):
    return (x + m - 1) // m * m


def to_blocked(s):
    """cuBLAS block-scaling layout of a (rows, cols) scale matrix (torchao to_blocked)."""
    rows, cols = s.shape
    nr, nc = math.ceil(rows / 128), math.ceil(cols / 4)
    p = torch.zeros((nr * 128, nc * 4), device=s.device, dtype=s.dtype)
    p[:rows, :cols] = s
    return p.view(nr, 128, nc, 4).permute(0, 2, 1, 3).reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def make_gemm(path, M, K, N):
    dev = "cuda"
    if path in ("fp32", "tf32"):
        a, b = torch.randn(M, K, device=dev), torch.randn(K, N, device=dev)
        tf = path == "tf32"

        def fn():
            torch.backends.cuda.matmul.allow_tf32 = tf
            return torch.mm(a, b)
        return fn
    if path == "bf16":
        a, b = torch.randn(M, K, device=dev, dtype=torch.bfloat16), torch.randn(K, N, device=dev, dtype=torch.bfloat16)
        return lambda: torch.mm(a, b)
    if path == "int8":
        a = torch.randint(-8, 8, (M, K), device=dev, dtype=torch.int8)
        b = torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8)
        return lambda: torch._int_mm(a, b.t())
    if path.startswith("fp8"):
        a = (torch.randn(M, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        b = (torch.randn(N, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        if path == "fp8_tensor":
            sa, sb = torch.tensor(1.0, device=dev), torch.tensor(1.0, device=dev)
        else:
            sa, sb = torch.ones(M, 1, device=dev), torch.ones(1, N, device=dev)
        return lambda: torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    if path in ("nvfp4", "mxfp4"):
        blk = 16 if path == "nvfp4" else 32
        sdt = torch.float8_e4m3fn if path == "nvfp4" else torch.float8_e8m0fnu
        a = torch.randint(0, 256, (M, K // 2), device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        b = torch.randint(0, 256, (N, K // 2), device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        sa = to_blocked(torch.ones(M, K // blk, device=dev).to(sdt))
        sb = to_blocked(torch.ones(N, K // blk, device=dev).to(sdt))
        return lambda: torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    raise ValueError(path)


# ---------------------------------------------------------------------------------------------------- FP4 quant
_GRID = {}


def e2m1_grid(dev):
    """the E2M1 values as a device tensor, created once outside any CUDA-graph capture (a host-to-device copy inside a
    capture invalidates it; that was the first E8b run's graph-mode error)."""
    if str(dev) not in _GRID:
        _GRID[str(dev)] = torch.tensor(E2M1, device=dev)
    return _GRID[str(dev)]


def fp4_fakequant(x, fmt, dither, g=4.0 / (6 * 448)):
    """x (M, K) fp32 -> fake-quantized fp32 (E9 definitions); dither: one u per column, fresh every call."""
    M, K = x.shape
    blk = 16 if fmt == "nvfp4" else 32
    grid = e2m1_grid(x.device)
    xb = x.view(M, K // blk, blk)
    amax = xb.abs().amax(-1, keepdim=True)
    if fmt == "nvfp4":
        s = (amax / (6 * g)).clamp(max=448.0).to(torch.float8_e4m3fn).float() * g
        s = torch.where(amax == 0, torch.full_like(s, g), s)
    else:
        s = torch.exp2(torch.floor(torch.log2(amax.clamp(min=1e-30))) - 2)
        s = torch.where(amax == 0, torch.ones_like(s), s)
    xs = xb / s
    a = xs.abs().clamp(max=6.0)
    k = (torch.bucketize(a, grid, right=True) - 1).clamp(max=6)
    lo, hi = grid[k], grid[k + 1]
    frac = (a - lo) / (hi - lo)
    if dither:
        u = torch.rand(1, K, device=x.device).view(1, K // blk, blk)
        up = u < frac
    else:
        up = (frac > 0.5) | ((frac == 0.5) & (k % 2 == 1))
    q = torch.where(up, hi, lo)
    return (torch.sign(xs) * q * s).view(M, K)


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g.replay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/results/e8b/e8b.json")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--sections", default="gemm,quant_eager,quant_graph")
    a = ap.parse_args()
    sec = set(a.sections.split(","))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    t_start = time.time()
    env = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
           "capability": list(torch.cuda.get_device_capability(0)), "gpu_state": gpu_state(),
           "nvidia_smi": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,power.limit,"
                                         "memory.total", "--format=csv,noheader"], capture_output=True,
                                        text=True).stdout.strip(),
           "has_float4": hasattr(torch, "float4_e2m1fn_x2"), "has_e8m0": hasattr(torch, "float8_e8m0fnu")}
    log("env", env)
    out = {"label": "E8b (DESIGN_ext 1.6), descriptive", "env": env, "gemm": {}, "quant": {}, "errors": {},
           "settings": {"rounds": a.rounds, "warmup": 30, "block_s": 1.0}}
    torch.backends.cuda.matmul.allow_tf32 = False
    conds = {}
    # GEMMs
    for name, (K, N) in (SHAPES.items() if "gemm" in sec else []):
        for M in MS:
            for path in ("fp32", "tf32", "bf16", "int8", "fp8_tensor", "fp8_row", "nvfp4", "mxfp4"):
                key = f"gemm|{name}|{M}|{path}"
                try:
                    fn = make_gemm(path, M, K, N)
                    y = fn()
                    torch.cuda.synchronize()
                    t = float(statistics.median(time_block(fn, 3, 5)))
                    conds[key] = (fn, t)
                    out["gemm"][key] = {"shape": [M, K, N], "out_dtype": str(y.dtype), "blocks_ms": []}
                except Exception as ex:  # noqa: BLE001
                    out["errors"][key] = traceback.format_exception_only(type(ex), ex)[-1].strip()[:400]
                torch.backends.cuda.matmul.allow_tf32 = False
    log(f"gemm conditions ok {len(conds)}, errors {len(out['errors'])}")
    # FP4 activation quantization (det vs sh-iter). Phase order avoids a PyTorch 2.9 generator-state error seen when an
    # eager RNG call follows the capture of a graph that uses the default CUDA generator: inputs and every eager RNG use
    # (eager conditions and the value check) come first; graph-mode conditions are captured afterwards and only replayed.
    xs = {M: torch.randn(M, 4096, device="cuda") for M in MS}
    e2m1_grid(xs[512].device)
    qconds = {}
    for M in (MS if "quant_eager" in sec else []):
        for fmt in ("nvfp4", "mxfp4"):
            for dither in (False, True):
                key = f"quant|{fmt}|{M}|{'shiter' if dither else 'det'}|eager"
                try:
                    fn = (lambda x=xs[M], fmt=fmt, dither=dither: fp4_fakequant(x, fmt, dither))
                    fn()
                    torch.cuda.synchronize()
                    qconds[key] = (fn, float(statistics.median(time_block(fn, 3, 5))))
                    out["quant"][key] = {"blocks_ms": []}
                except Exception as ex:  # noqa: BLE001
                    out["errors"][key] = traceback.format_exception_only(type(ex), ex)[-1].strip()[:400]
    try:
        if "quant_eager" not in sec:
            raise RuntimeError("skipped (section quant_eager not run)")
        x = xs[512]
        ref = {fmt: fp4_fakequant(x, fmt, False) for fmt in ("nvfp4", "mxfp4")}
        out["quant_check"] = {fmt: {"det_rel_err": float((ref[fmt] - x).norm() / x.norm()),
                                    "shiter_rel_err": float((fp4_fakequant(x, fmt, True) - x).norm() / x.norm()),
                                    "shiter_mean_bias_16draws": float((torch.stack([fp4_fakequant(x, fmt, True)
                                                                                     for _ in range(16)]).mean(0)
                                                                        - x).mean()),
                                    "det_mean_bias": float((ref[fmt] - x).mean())}
                              for fmt in ("nvfp4", "mxfp4")}
    except Exception as ex:  # noqa: BLE001
        out["errors"]["quant_check"] = repr(ex)[:400]
    rng = random.Random(0)

    def run_rounds(cd, tag):
        keys = sorted(cd)
        for rd in range(a.rounds):
            order = list(keys)
            rng.shuffle(order)
            for key in order:
                fn, t = cd[key]
                n = max(10, math.ceil(1000.0 / max(t, 1e-3)))
                g0 = gpu_state()
                ms = time_block(fn, 30, n)
                torch.backends.cuda.matmul.allow_tf32 = False
                rec = out["gemm"].get(key) or out["quant"].get(key)
                rec["blocks_ms"].append(float(statistics.median(ms)))
                rec.setdefault("gpu", []).append([g0, gpu_state()])
                rec["n"] = n
            log(f"{tag} round {rd} done ({time.time() - t_start:.0f}s)")
            json.dump(out, open(a.out, "w"), indent=1)

    run_rounds(conds, "gemm")
    run_rounds(qconds, "quant-eager")
    gconds = {}
    for M in (MS if "quant_graph" in sec else []):
        for fmt in ("nvfp4", "mxfp4"):
            for dither in (False, True):
                key = f"quant|{fmt}|{M}|{'shiter' if dither else 'det'}|graph"
                try:
                    fn = capture(lambda x=xs[M], fmt=fmt, dither=dither: fp4_fakequant(x, fmt, dither))
                    fn()
                    torch.cuda.synchronize()
                    gconds[key] = (fn, float(statistics.median(time_block(fn, 3, 5))))
                    out["quant"][key] = {"blocks_ms": []}
                except Exception as ex:  # noqa: BLE001
                    out["errors"][key] = traceback.format_exception_only(type(ex), ex)[-1].strip()[:400]
    run_rounds(gconds, "quant-graph")
    for rec in list(out["gemm"].values()) + list(out["quant"].values()):
        rec["T_ms"] = float(statistics.median(rec["blocks_ms"]))
    # summaries
    summ = {}
    for name, (K, N) in SHAPES.items():
        for M in MS:
            base = out["gemm"].get(f"gemm|{name}|{M}|bf16", {}).get("T_ms")
            row = {}
            for path in ("fp32", "tf32", "bf16", "int8", "fp8_tensor", "fp8_row", "nvfp4", "mxfp4"):
                r = out["gemm"].get(f"gemm|{name}|{M}|{path}")
                if r:
                    row[path] = {"T_ms": r["T_ms"], "over_bf16": (r["T_ms"] / base) if base else None,
                                 "TFLOPS": 2 * M * K * N / (r["T_ms"] * 1e-3) / 1e12}
            summ[f"{name}|{M}"] = row
    qs = {}
    for M in MS:
        for fmt in ("nvfp4", "mxfp4"):
            for mode in ("eager", "graph"):
                d = out["quant"].get(f"quant|{fmt}|{M}|det|{mode}")
                s = out["quant"].get(f"quant|{fmt}|{M}|shiter|{mode}")
                if d and s:
                    qs[f"{fmt}|{M}|{mode}"] = {"det_ms": d["T_ms"], "shiter_ms": s["T_ms"],
                                               "overhead": (s["T_ms"] - d["T_ms"]) / d["T_ms"]}
    out["summary_gemm"], out["summary_quant"] = summ, qs
    out["secs"] = time.time() - t_start
    json.dump(out, open(a.out, "w"), indent=1)
    log(json.dumps({"errors": out["errors"]}, indent=1))
    for k, v in summ.items():
        log(k, {p: round(x["T_ms"], 4) for p, x in v.items()})
    for k, v in qs.items():
        log(k, {kk: round(vv, 4) for kk, vv in v.items()})
    log("E8B_DONE")


if __name__ == "__main__":
    main()
