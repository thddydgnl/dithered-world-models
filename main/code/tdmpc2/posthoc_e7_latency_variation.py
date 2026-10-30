"""POST-HOC, descriptive (decided after the E7 latency summary was read; no verdict uses it): the block-to-block
variation of the E7 latency measurement (coordinator request: was the run affected by the DINO-WM chain on GPU 1 and
by other users' CPU load?).
  - per block: r = block median / median of the condition's 5 block medians
  - the distribution of r (how many blocks are > 15% slower / faster than their condition's typical block)
  - the time order of the blocks, reconstructed from e7_latency.py (numpy default_rng(0).permutation(96) per round),
    and r along that order (runs of slow blocks = periods of a different latency mode)
  - Spearman of r with the recorded host 1-min load, the CPU busy share over the block and the GPU SM clock at the
    block end
  - a robustness view of the H5 overheads: the median over rounds of the per-round overheads, and the share of rounds
    whose overhead exceeds 10% in absolute value
usage (server, CPU only): python posthoc_e7_latency_variation.py --latency <e7_latency.json> --out <json>
"""
import argparse
import json

import numpy as np

import analyze_e1 as A1

TASKS = ["mw-push", "mw-pick-place", "cheetah-run", "walker-walk"]
PRECS = [("fp32", "none"), ("bf16", "none"), ("int8", "det"), ("int8", "shiter"), ("int4", "det"), ("int4", "shiter"),
         ("int4p", "det"), ("int4p", "shiter")]
ITERS = [3, 6, 12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = json.load(open(a.latency))
    conds = [f"{t}|{p}_{v}|it{it}" for t in TASKS for p, v in PRECS for it in ITERS]   # the script's order
    rng = np.random.default_rng(0)
    perms = [rng.permutation(len(conds)) for _ in range(int(L["settings"]["rounds"]))]
    rows = []
    for rd, perm in enumerate(perms):
        for pos, ci in enumerate(perm):
            key = conds[ci]
            blks = sorted(L["blocks"][key], key=lambda b: b["round"])
            med = float(np.median([b["median_ms"] for b in blks]))
            b = blks[rd]
            sm_end = float(b["gpu_end"].split(",")[1])
            rows.append({"order": rd * len(conds) + pos, "round": rd, "cond": key, "r": b["median_ms"] / med,
                         "load": b["loadavg1"], "busy": b["cpu_busy"], "sm_end": sm_end})
    r = np.array([x["r"] for x in rows])
    order = np.array([x["order"] for x in rows])
    srt = np.argsort(order)
    slow = (r > 1.15).astype(int)[srt]
    runs, cur = [], 0
    for s in slow:
        cur = cur + 1 if s else 0
        if s and (len(runs) == 0 or cur == 1):
            runs.append(1)
        elif s:
            runs[-1] = cur
    out = {"label": "post-hoc, descriptive; no verdict uses it",
           "n_blocks": int(len(r)),
           "r_quantiles": {q: float(np.percentile(r, q)) for q in (1, 5, 25, 50, 75, 95, 99)},
           "share_slower_15pct": float(np.mean(r > 1.15)), "share_faster_15pct": float(np.mean(r < 1 / 1.15)),
           "slow_runs_in_time_order": {"n_runs": len(runs), "longest": int(max(runs)) if runs else 0,
                                       "mean_length": float(np.mean(runs)) if runs else 0.0},
           "slow_share_by_round": [float(np.mean([x["r"] > 1.15 for x in rows if x["round"] == rd]))
                                   for rd in range(len(perms))],
           "spearman_r_vs": {"host_load1": A1.spearman([x["load"] for x in rows], r),
                             "cpu_busy": A1.spearman([x["busy"] for x in rows], r),
                             "sm_clock_end": A1.spearman([x["sm_end"] for x in rows], r)},
           "sm_clock_end_below_max_share": float(np.mean([x["sm_end"] < 1890 for x in rows]))}
    # robustness view of the H5 overheads
    rob = {}
    for t in TASKS:
        for it in ITERS:
            for p in ("int8", "int4", "int4p"):
                bd = sorted(L["blocks"][f"{t}|{p}_det|it{it}"], key=lambda b: b["round"])
                bv = sorted(L["blocks"][f"{t}|{p}_shiter|it{it}"], key=lambda b: b["round"])
                pr = [(y["median_ms"] - x["median_ms"]) / x["median_ms"] for x, y in zip(bd, bv)]
                rob[f"{t}|{p}|it{it}"] = {"pooled": L["dither_overhead"][f"{t}|{p}|it{it}"],
                                          "median_of_rounds": float(np.median(pr)),
                                          "share_rounds_abs_gt_10pct": float(np.mean(np.abs(pr) > 0.10))}
    out["overhead_robustness"] = rob
    out["overhead_median_of_rounds_max"] = {p: float(max(v["median_of_rounds"] for k, v in rob.items()
                                                         if k.split("|")[1] == p)) for p in ("int8", "int4", "int4p")}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "overhead_robustness"}, indent=1))


if __name__ == "__main__":
    main()
