"""Analysis of the POST-HOC paired E7 latency re-measurement (DESIGN_main v5.3 10.3; descriptive, the H5 verdict stays
the first measurement's pre-defined computation). Written before the re-measurement ran.
  - per (task, kernel bits, iterations): the pair overheads (dither - det) / det of the 10 rounds (back-to-back blocks),
    their median and a bootstrap 95% CI of the median (10,000 resamples of the rounds, RandomState(0), percentile)
  - the first measurement's pre-defined computation on the same data: (median of all dither steps - median of all det
    steps) / median of all det steps
  - beside them, the first measurement's value (v5/e7_latency.json, pre-defined computation)
  - measurement state: blocks with foreign processes, host load, CPU busy share, SM clock, and the share of blocks more
    than 15% slower than their condition's median block (the two-mode pattern of the first measurement)
usage (server, CPU only): python analyze_e7_posthoc_latency.py --posthoc ../v5/e7_latency_posthoc.json \
                              --first ../v5/e7_latency.json --outdir ../analysis_posthoc_e7lat
"""
import argparse
import json
import os

import numpy as np

NB = 10000


def boot_median_ci(x):
    x = np.asarray(x, float)
    idx = np.random.RandomState(0).randint(0, len(x), size=(NB, len(x)))
    m = np.median(x[idx], axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posthoc", required=True)
    ap.add_argument("--first", default=None)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    P = json.load(open(a.posthoc))
    F = json.load(open(a.first)) if a.first and os.path.exists(a.first) else {}
    out = {"label": "post-hoc (DESIGN_main v5.3 10.3), descriptive", "conditions": {}, "state": {}}
    for key, prs in P["pairs"].items():
        t, kern, it = key.split("|")
        ov = [p["overhead"] for p in prs]
        sd = np.array(P["steps_ms"][f"{t}|{kern}_det|{it}"])
        sv = np.array(P["steps_ms"][f"{t}|{kern}_shiter|{it}"])
        pooled = float((np.median(sv) - np.median(sd)) / np.median(sd))
        out["conditions"][key] = {
            "n_pairs": len(ov), "pair_overheads": ov, "median_pair_overhead": float(np.median(ov)),
            "median_pair_overhead_ci": boot_median_ci(ov) if len(ov) > 1 else [float("nan")] * 2,
            "pooled_median_overhead_same_data": pooled,
            "first_measurement_pooled_overhead": F.get("dither_overhead", {}).get(key),
            "det_first_share": float(np.mean([p["first"] == "det" for p in prs]))}
    by = {}
    for kern in ("int8", "int4", "int4p"):
        c = {k: v for k, v in out["conditions"].items() if k.split("|")[1] == kern}
        if c:
            by[kern] = {"median_pair_overhead_max": max(v["median_pair_overhead"] for v in c.values()),
                        "median_pair_overhead_ci_upper_max": max(v["median_pair_overhead_ci"][1] for v in c.values()),
                        "pooled_same_data_max": max(v["pooled_median_overhead_same_data"] for v in c.values()),
                        "all_median_pair_le_10pct": all(v["median_pair_overhead"] <= 0.10 + 1e-9 for v in c.values()),
                        "all_ci_upper_le_10pct": all(v["median_pair_overhead_ci"][1] <= 0.10 + 1e-9 for v in c.values())}
    out["by_kernel"] = by
    blks = [b for r in P["blocks"].values() for b in r]
    ratio = []
    for key, r in P["blocks"].items():
        med = float(np.median([b["median_ms"] for b in r]))
        ratio += [b["median_ms"] / med for b in r]
    sm = [float(b["gpu_end"].split(",")[1]) for b in blks if len(b["gpu_end"].split(",")) == 5]
    out["state"] = {"n_blocks": len(blks), "blocks_with_foreign_processes": int(sum(b["foreign_procs"] > 0 for b in blks)),
                    "loadavg1": [float(min(b["loadavg1"] for b in blks)), float(max(b["loadavg1"] for b in blks))],
                    "cpu_busy": [float(min(b["cpu_busy"] for b in blks)), float(max(b["cpu_busy"] for b in blks))],
                    "sm_clock_end": [float(min(sm)), float(max(sm))] if sm else None,
                    "share_blocks_slower_15pct": float(np.mean(np.array(ratio) > 1.15)),
                    "secs": P.get("secs"), "settings": P.get("settings")}
    os.makedirs(a.outdir, exist_ok=True)
    json.dump(out, open(f"{a.outdir}/e7_posthoc_latency_summary.json", "w"), indent=1)
    L = ["# E7 지연 재측정(사후, v5.3 10.3, 기술)", "", "| 과제·kernel·반복 | 짝 오버헤드 중앙값 [bootstrap 95% CI] | 같은 자료의 사전 정의 계산 | "
         "첫 측정(사전 정의) |", "|---|---|---|---|"]
    for key, c in out["conditions"].items():
        fm = c["first_measurement_pooled_overhead"]
        L.append(f"| {key} | {c['median_pair_overhead']:+.3f} [{c['median_pair_overhead_ci'][0]:+.3f}, "
                 f"{c['median_pair_overhead_ci'][1]:+.3f}] | {c['pooled_median_overhead_same_data']:+.3f} | "
                 f"{'' if fm is None else format(fm, '+.3f')} |")
    L += ["", f"측정 상태: {json.dumps(out['state'])}"]
    open(f"{a.outdir}/e7_posthoc_latency_tables.md", "w").write("\n".join(L) + "\n")
    print(json.dumps({"by_kernel": by, "state": {k: v for k, v in out["state"].items() if k != "settings"}}))


if __name__ == "__main__":
    main()
