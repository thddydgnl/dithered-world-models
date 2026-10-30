"""E7 feasibility probe (v5 draft; a few seconds, 1 process, no episodes): does torch._int_mm run on the server GPU at
the TD-MPC2 hidden-layer shapes, is the int8 path exact against an integer reference, is Triton importable, and how do
the single-GEMM times compare (fp32 / bf16 / int8). Timings on a shared GPU are indicative only (E7 latency needs an
exclusive GPU window).
usage (server): python e7_feasibility.py
"""
import json
import time

import torch


def bench(f, n=200):
    for _ in range(20):
        f()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(5):
        s.record()
        for _ in range(n):
            f()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / n * 1000.0)   # microseconds per call
    ts.sort()
    return ts[len(ts) // 2]


def main():
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    try:
        import triton
        out["triton"] = triton.__version__
    except Exception as ex:  # noqa: BLE001
        out["triton"] = f"not importable: {ex!r}"
    dev = "cuda"
    g = torch.Generator(device=dev)
    g.manual_seed(0)
    res = {}
    for M, K, N in ((512, 520, 512), (512, 512, 512), (24, 512, 512), (1024, 520, 512)):
        a8 = torch.randint(-127, 128, (M, K), generator=g, device=dev, dtype=torch.int8)
        b8 = torch.randint(-127, 128, (K, N), generator=g, device=dev, dtype=torch.int8)
        r = {}
        try:
            c = torch._int_mm(a8, b8)
            ref = (a8.double() @ b8.double()).round().long()   # |sums| < 2^53: exact in double
            r["int_mm_exact"] = bool(torch.equal(c.long(), ref))
            r["us_int8"] = bench(lambda: torch._int_mm(a8, b8))
        except Exception as ex:  # noqa: BLE001
            r["int_mm_error"] = repr(ex)[:200]
        af, bf = a8.float(), b8.float()
        ab, bb = af.bfloat16(), bf.bfloat16()
        r["us_fp32"] = bench(lambda: af @ bf)
        r["us_bf16"] = bench(lambda: ab @ bb)
        u = torch.rand((1, K), generator=g, device=dev)
        x = torch.rand((M, K), generator=g, device=dev) * 15.0
        r["us_dither_quant_eager"] = bench(lambda: torch.floor(x + torch.rand((1, K), generator=g, device=dev))
                                           .clamp_(0, 15).to(torch.int8))
        r["us_det_quant_eager"] = bench(lambda: torch.round(x).clamp_(0, 15).to(torch.int8))
        del u
        res[f"{M}x{K}x{N}"] = r
    out["shapes"] = res
    t0 = time.time()
    print(json.dumps(out, indent=1))
    print(f"E7_PROBE_DONE {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
