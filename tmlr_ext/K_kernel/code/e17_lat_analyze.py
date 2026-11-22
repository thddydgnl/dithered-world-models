"""E17 latency analysis (tmlr_ext/DESIGN_tmlr.md 7, K1), written before the main measurement; runs on the pod.

Pre-defined computation (E8 conventions, rss_ext/E8_latency/code/e8_analyze.py):
  T(condition) = median of the condition's block values (block value = median per-replay time of the block; int-det
                 has 20 blocks per cell: its block in each of the two pairs of each round)
  pair overhead = (variant block - det block) / det block of the same pair; per cell and variant 10 pairs
  cell overhead = median of the 10 pair overheads; 95% CI = percentile bootstrap of the median, 10,000 resamples of
                  the pairs, numpy RandomState(0) per cell and variant (E7 post-hoc / E8 convention)
  K1 holds if the det+shift overhead median <= 5% in all 6 cells (5M, 317M x N 512, 2048, 8192); tolerance 1e-9
Descriptive: the sh-iter overheads (E8 L3 analogue: median <= 5% in all cells and CI upper bound <= 10%), CI upper
  bounds, the absolute pair differences (median ms), T ratios to det, E8's int-det / int-sh-iter T of the same cells on
  E8's pod (rss_ext/E8_latency REPORT 2.1; another pod, reported side by side only), measurement state (SM clock,
  temperature, power, clock-event reasons, host load, CPU busy) and the share of blocks more than 15% slower than their
  condition's T.
usage: python e17_lat_analyze.py --main <dir with e17_lat_main.json> --outdir <dir> [--e8 inputs/e8_cells.csv]
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

SIZES = (5, 317)
NS = (512, 2048, 8192)
VARIANTS = ("int-detshift", "int-shiter")
NB = 10000
EPS = 1e-9


def le(x, thr):
    return x <= thr + EPS


def boot_median_ci(x, seed=0):
    x = np.asarray(x, float)
    idx = np.random.RandomState(seed).randint(0, len(x), size=(NB, len(x)))
    m = np.median(x[idx], axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--e8", default=None, help="E8 analysis e8_cells.csv (descriptive cross-pod reference)")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    M = json.load(open(f"{a.main}/e17_lat_main.json"))
    blocks, pairs = M["blocks"], M["pairs"]
    byk = defaultdict(list)
    for b in blocks:
        byk[(b["size"], b["N"], b["prec"])].append(b["median_ms"])
    T = {k: float(np.median(v)) for k, v in byk.items()}
    e8 = {}
    if a.e8 and os.path.exists(a.e8):
        for r in csv.DictReader(open(a.e8)):
            if r.get("mode") == "graph" and int(r["size"]) in SIZES:
                e8[(int(r["size"]), int(r["N"]))] = {kk: r[kk] for kk in r}
    cells = {}
    for size in SIZES:
        for N in NS:
            if (size, N, "int-det") not in T:
                continue
            c = {"size": size, "N": N, "T_ms": {p: T.get((size, N, p)) for p in ("int-det",) + VARIANTS},
                 "n_blocks": {p: len(byk.get((size, N, p), [])) for p in ("int-det",) + VARIANTS}}
            for v in VARIANTS:
                pr = [p for p in pairs if p["size"] == size and p["N"] == N and p["variant"] == v]
                ov = [p["overhead"] for p in pr]
                if not ov:
                    continue
                c[v] = {"pairs": ov, "n": len(ov), "median": float(np.median(ov)),
                        "ci": boot_median_ci(ov) if len(ov) > 1 else [float("nan")] * 2,
                        "abs_diff_ms_median": float(np.median([p["var_ms"] - p["det_ms"] for p in pr])),
                        "ratio_T": T[(size, N, v)] / T[(size, N, "int-det")],
                        "first_det_share": float(np.mean([p["first"] == "int-det" for p in pr]))}
            if (size, N) in e8:
                c["e8_pod_reference"] = e8[(size, N)]
            cells[f"{size}|{N}"] = c
    ds = {k: v["int-detshift"] for k, v in cells.items() if "int-detshift" in v}
    sh = {k: v["int-shiter"] for k, v in cells.items() if "int-shiter" in v}
    K1 = {"holds": len(ds) == 6 and all(le(v["median"], 0.05) for v in ds.values()), "n_cells": len(ds),
          "cells_median_le_5pct": [k for k, v in ds.items() if le(v["median"], 0.05)],
          "max_median": max((v["median"] for v in ds.values()), default=None),
          "max_ci_upper": max((v["ci"][1] for v in ds.values()), default=None),
          "cells_ci_upper_le_10pct": [k for k, v in ds.items() if le(v["ci"][1], 0.10)]}
    SH = {"median_le_5pct_all": len(sh) == 6 and all(le(v["median"], 0.05) for v in sh.values()),
          "cells_ci_upper_le_10pct": [k for k, v in sh.items() if le(v["ci"][1], 0.10)],
          "max_median": max((v["median"] for v in sh.values()), default=None),
          "max_ci_upper": max((v["ci"][1] for v in sh.values()), default=None)}
    # measurement state
    sm, temp, pw, thr, load, busy, slow = [], [], [], [], [], [], 0
    for b in blocks:
        for g in (b.get("gpu_start", {}), b.get("gpu_mid") or {}, b.get("gpu_end", {})):
            if "sm_mhz" in g:
                sm.append(g["sm_mhz"])
                temp.append(g["temp_c"])
                pw.append(g["power_w"])
                thr.append(g.get("throttle", 0))
        load.append(b["load_start"][0])
        busy.append(b["cpu_busy"])
        slow += b["median_ms"] > 1.15 * T[(b["size"], b["N"], b["prec"])]
    q = lambda x: [float(np.min(x)), float(np.median(x)), float(np.max(x))] if x else None
    state = {"sm_mhz_min_med_max": q(sm), "temp_c_min_med_max": q(temp), "power_w_min_med_max": q(pw),
             "power_cap_reason_share": float(np.mean([(t & 0x4) > 0 for t in thr])) if thr else None,
             "thermal_reason_share": float(np.mean([(t & 0x60) > 0 for t in thr])) if thr else None,
             "host_load1_min_med_max": q(load), "cpu_busy_min_med_max": q(busy),
             "slow_blocks_gt15pct_share": slow / max(len(blocks), 1), "n_blocks": len(blocks),
             "rounds_done": M.get("rounds_done"), "secs": M.get("secs"),
             "gpu_compute_procs_start_end": [M["env"].get("gpu_compute_procs_at_start"),
                                             M.get("gpu_compute_procs_at_end")]}
    summary = {"label": "E17 latency (DESIGN_tmlr 7, K1): graph-WM, E8 protocol", "K1": K1, "shiter_descriptive": SH,
               "cells": cells, "state": state, "settings": M["settings"], "env": M["env"]}
    json.dump(summary, open(f"{a.outdir}/e17_lat_summary.json", "w"), indent=1)
    with open(f"{a.outdir}/e17_lat_cells.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["size", "N", "T_det_ms", "T_shiter_ms", "T_detshift_ms", "ov_detshift_median", "ov_detshift_ci_lo",
                    "ov_detshift_ci_hi", "ov_shiter_median", "ov_shiter_ci_lo", "ov_shiter_ci_hi",
                    "absdiff_detshift_ms", "absdiff_shiter_ms"])
        for c in cells.values():
            d, s_ = c.get("int-detshift", {}), c.get("int-shiter", {})
            w.writerow([c["size"], c["N"], c["T_ms"]["int-det"], c["T_ms"]["int-shiter"], c["T_ms"]["int-detshift"],
                        d.get("median"), *(d.get("ci") or [None, None]), s_.get("median"), *(s_.get("ci") or [None, None]),
                        d.get("abs_diff_ms_median"), s_.get("abs_diff_ms_median")])
    lines = ["# E17 latency tables (auto-generated by e17_lat_analyze.py)", "",
             f"- K1 (det+shift overhead median <= 5% in all 6 cells): **{K1['holds']}**; max median "
             f"{K1['max_median']}, max CI upper {K1['max_ci_upper']}",
             f"- sh-iter (descriptive): median <= 5% in all cells {SH['median_le_5pct_all']}, max median "
             f"{SH['max_median']}", "",
             "| size | N | int-det ms | int-sh-iter ms | int-det+shift ms | det+shift overhead [95% CI] | "
             "sh-iter overhead [95% CI] | det+shift abs ms |", "|---|---|---|---|---|---|---|---|"]
    for c in cells.values():
        d, s_ = c.get("int-detshift", {}), c.get("int-shiter", {})
        f = lambda x: f"{100 * x:+.2f}%"
        lines.append(f"| {c['size']}M | {c['N']} | {c['T_ms']['int-det']:.3f} | {c['T_ms']['int-shiter']:.3f} | "
                     f"{c['T_ms']['int-detshift']:.3f} | {f(d['median'])} [{f(d['ci'][0])}, {f(d['ci'][1])}] | "
                     f"{f(s_['median'])} [{f(s_['ci'][0])}, {f(s_['ci'][1])}] | {d['abs_diff_ms_median']:+.3f} |")
    lines += ["", f"- state: {json.dumps(state)}"]
    open(f"{a.outdir}/e17_lat_tables.md", "w").write("\n".join(lines) + "\n")
    print(json.dumps({"K1": K1["holds"], "max_median": K1["max_median"], "n_cells": K1["n_cells"]}))
    print("E17_LAT_ANALYZE_DONE")


if __name__ == "__main__":
    main()
