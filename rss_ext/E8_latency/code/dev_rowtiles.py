"""E8 development check (before PLAN.md): row-kernel time vs tiles per program (TILES) for det and dither (u read from
the pre-drawn buffer), MODE 0, fp32 input. Used to fix the TILES heuristic by the det time; dither shown alongside."""
import json
import sys

import torch
import triton

import e8lib as E

res = {}
st = torch.zeros(2, dtype=torch.int64, device="cuda")
for M, W in ((512, 512), (2048, 512), (8192, 512), (8192, 1024), (8192, 1792), (512, 4096), (2048, 4096),
             (8192, 4096)):
    y = torch.randn(M, W, device="cuda")
    g, b = torch.ones(W, device="cuda"), torch.zeros(W, device="cuda")
    q = torch.empty(M, W, device="cuda", dtype=torch.int8)
    ub = torch.rand(2, W, device="cuda")
    bw = triton.next_power_of_2(W)
    bm = max(1, 4096 // bw)
    for tiles in (1, 2, 4, 8, 16, 32):
        if bm * tiles > M:
            continue
        E._ROWCFG[(M, W)] = (bw, bm, tiles, (triton.cdiv(M, bm * tiles),))
        for dith in (False, True):
            fn = (lambda dith=dith: E.row_launch(y, None, None, g, b, q, st, M, W, -0.3, 3.0, 0.2, 15.0, 0.0, 0, False,
                                                 dith, u=ub[0]))
            fn()
            res[f"M{M}|W{W}|T{tiles}|{'dither' if dith else 'det'}"] = float(triton.testing.do_bench(fn, warmup=20,
                                                                                                    rep=100))
    E._ROWCFG.pop((M, W), None)
    line = " ".join(f"T{t}:{res.get(f'M{M}|W{W}|T{t}|det', 0) * 1e3:.1f}/{res.get(f'M{M}|W{W}|T{t}|dither', 0) * 1e3:.1f}"
                    for t in (1, 2, 4, 8, 16, 32))
    print(f"M{M} W{W} (det/dither us): {line}", flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
