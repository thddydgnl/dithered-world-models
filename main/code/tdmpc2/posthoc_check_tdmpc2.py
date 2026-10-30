"""Check and description of the two POST-HOC TD-MPC2 checks (DESIGN_main v5.3 10.2, 10.3; descriptive, no verdict
changes). Reads the raw files only (nothing is written next to them), recomputes the analysis outputs independently,
and adds descriptive views decided after the results were seen (marked post-hoc in the report).

Latency (v5/e7_latency_posthoc.json, analysis_posthoc_e7lat/e7_posthoc_latency_summary.json):
  - protocol: settings (10 rounds, warmup 50, 200 steps, gpu_check, seed 0); 96 conditions x 2,000 steps and 10 blocks;
    36 pair conditions x 10 pairs; the randomization reconstructed from the script (numpy default_rng(seed):
    one permutation of the 60 units per round, one draw per pair for the order inside the pair) against the recorded
    unit positions and pair orders; the two blocks of every pair adjacent in time; per-block records present (SM clock
    at start / end, host load, CPU busy share, foreign processes)
  - recomputation: block medians from the step lists, pair overheads, per-condition median of the pair overheads and
    its bootstrap 95% CI (RandomState(0), 10,000 resamples of the 10 rounds), the pre-defined pooled-median overhead on
    the same data, the first measurement's pooled overhead; compared with the analysis output
  - descriptive (post-hoc): the two-mode pattern (block / condition-median ratio r > 1.15), its relation to the SM
    clock, host load and CPU busy share (Spearman), runs in time, mode mismatch inside pairs, the pair overheads of the
    pairs whose two blocks are both in the fast mode, and the first measurement's unstable condition
QAT+none (out_posthoc_qatnone, out_e5, analysis_posthoc_qatnone/posthoc_qatnone_summary.json): the 90 pairs per cell
  (the E5 seeds, shared with the original weights' E5 none), D and its paired bootstrap CI recomputed for QAT+none,
  QAT+det, QAT+sh-iter and D(QAT+det) - D(QAT+none); compared with the analysis output
usage (server, CPU only): python posthoc_check_tdmpc2.py --root ~/pmx/A/tdmpc2_main --outdir <dir>
"""
import argparse
import glob
import gzip
import json
import os

import numpy as np
from scipy.stats import spearmanr

NB = 10000
TASKS = ["mw-push", "mw-pick-place", "cheetah-run", "walker-walk"]
ITERS = [3, 6, 12]
PAIRS = [("int8", 8), ("int4", 4), ("int4p", 4)]
SINGLES = [("fp32", 0), ("bf16", 0)]


def boot_median_ci(x):
    x = np.asarray(x, float)
    idx = np.random.RandomState(0).randint(0, len(x), size=(NB, len(x)))
    m = np.median(x[idx], axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def sp(a, b):
    return float(spearmanr(a, b).correlation)


def latency(root, out):
    P = json.load(open(f"{root}/v5/e7_latency_posthoc.json"))
    A = json.load(open(f"{root}/analysis_posthoc_e7lat/e7_posthoc_latency_summary.json"))
    F = json.load(open(f"{root}/v5/e7_latency.json"))
    st = P["settings"]
    chk = {"settings_ok": (st["rounds"], st["warmup"], st["steps"], st["seed"], st["gpu_check"], st["force"],
                           st["quick"]) == (10, 50, 200, 0, True, False, False),
           "conditions_96": len(P["summary"]) == 96 and all(len(v) == 2000 for v in P["steps_ms"].values())
           and all(len(v) == 10 for v in P["blocks"].values()),
           "pair_conditions_36_x10": len(P["pairs"]) == 36 and all(len(v) == 10 for v in P["pairs"].values()),
           "foreign_blocks": int(sum(b["foreign_procs"] > 0 for r in P["blocks"].values() for b in r))}
    # randomization reconstructed from e7_latency_paired.py
    units = [("pair", t, p, it) for t in TASKS for p, _ in PAIRS for it in ITERS] + \
            [("single", t, p, it) for t in TASKS for p, _ in SINGLES for it in ITERS]
    rng = np.random.default_rng(st["seed"])
    bad_pos = bad_first = 0
    for rd in range(st["rounds"]):
        perm = rng.permutation(len(units))
        for upos, ui in enumerate(perm):
            kind, t, p, it = units[ui]
            flip = kind == "pair" and rng.random() < 0.5
            vs = (["shiter", "det"] if flip else ["det", "shiter"]) if kind == "pair" else ["none"]
            for vpos, v in enumerate(vs):
                b = P["blocks"][f"{t}|{p}_{v}|it{it}"][rd]
                bad_pos += int(b["round"] != rd or b["unit_pos"] != upos or b["pos_in_unit"] != vpos)
            if kind == "pair":
                bad_first += int(P["pairs"][f"{t}|{p}|it{it}"][rd]["first"] != vs[0])
    chk["randomization_matches_script"] = bad_pos == 0 and bad_first == 0
    allb = sorted(((b["t_start"], k, b["round"]) for k, r in P["blocks"].items() for b in r))
    order = {(k, rd): i for i, (_, k, rd) in enumerate(allb)}
    adj = all(abs(order[(f"{t}|{p}_det|it{it}", rd)] - order[(f"{t}|{p}_shiter|it{it}", rd)]) == 1
              for t in TASKS for p, _ in PAIRS for it in ITERS for rd in range(st["rounds"]))
    chk["pair_blocks_adjacent_in_time"] = bool(adj)
    chk["det_first_share"] = float(np.mean([pr["first"] == "det" for v in P["pairs"].values() for pr in v]))
    rec = all(all(k_ in b for k_ in ("gpu_start", "gpu_end", "loadavg1", "cpu_busy", "foreign_procs"))
              and len(b["gpu_end"].split(",")) == 5 for r in P["blocks"].values() for b in r)
    chk["block_records_complete"] = bool(rec)
    # recomputation
    bm_bad, pr_bad, med_bad, ci_bad, pool_bad, first_bad = 0, 0, 0, 0, 0, 0
    cond = {}
    for key in P["pairs"]:
        t, p, it = key.split("|")
        md = [float(np.median(P["steps_ms"][f"{t}|{p}_det|{it}"][200 * i:200 * (i + 1)])) for i in range(10)]
        mv = [float(np.median(P["steps_ms"][f"{t}|{p}_shiter|{it}"][200 * i:200 * (i + 1)])) for i in range(10)]
        bm_bad += sum(abs(x - b["median_ms"]) > 1e-9 for x, b in zip(md, P["blocks"][f"{t}|{p}_det|{it}"]))
        bm_bad += sum(abs(x - b["median_ms"]) > 1e-9 for x, b in zip(mv, P["blocks"][f"{t}|{p}_shiter|{it}"]))
        ov = [(y - x) / x for x, y in zip(md, mv)]
        pr_bad += sum(abs(o - pr["overhead"]) > 1e-12 for o, pr in zip(ov, P["pairs"][key]))
        med, ci = float(np.median(ov)), boot_median_ci(ov)
        sd, sv = np.array(P["steps_ms"][f"{t}|{p}_det|{it}"]), np.array(P["steps_ms"][f"{t}|{p}_shiter|{it}"])
        pooled = float((np.median(sv) - np.median(sd)) / np.median(sd))
        a = A["conditions"][key]
        med_bad += abs(med - a["median_pair_overhead"]) > 1e-12
        ci_bad += max(abs(x - y) for x, y in zip(ci, a["median_pair_overhead_ci"])) > 1e-12
        pool_bad += abs(pooled - a["pooled_median_overhead_same_data"]) > 1e-12
        first_bad += abs(F["dither_overhead"][key] - a["first_measurement_pooled_overhead"]) > 1e-12
        cond[key] = {"pair_overheads": ov, "median": med, "ci": ci, "pooled_same_data": pooled,
                     "first_pooled": F["dither_overhead"][key]}
    chk.update({"block_medians_from_steps_mismatch": int(bm_bad), "pair_overheads_mismatch": int(pr_bad),
                "median_mismatch": int(med_bad), "ci_mismatch": int(ci_bad), "pooled_mismatch": int(pool_bad),
                "first_measurement_mismatch": int(first_bad)})
    # descriptive (post-hoc): the two modes
    rows = []
    for key, r in P["blocks"].items():
        mm = float(np.median([b["median_ms"] for b in r]))
        for b in r:
            g0, g1 = b["gpu_start"].split(","), b["gpu_end"].split(",")
            rows.append({"cond": key, "round": b["round"], "unit_pos": b["unit_pos"], "pos_in_unit": b["pos_in_unit"],
                         "t_start": b["t_start"], "median_ms": b["median_ms"], "r": b["median_ms"] / mm,
                         "load": b["loadavg1"], "busy": b["cpu_busy"], "sm_start": float(g0[1]), "sm_end": float(g1[1]),
                         "temp_end": float(g1[3]), "power_end": float(g1[4]), "foreign": b["foreign_procs"]})
    r = np.array([x["r"] for x in rows])
    slow = r > 1.15
    sm_end = np.array([x["sm_end"] for x in rows])
    sm_start = np.array([x["sm_start"] for x in rows])
    smax = float(sm_end.max())
    ts = np.argsort([x["t_start"] for x in rows])
    runs, cur = [], 0
    for s in slow[ts]:
        if s:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    desc = {"n_blocks": len(rows), "share_slower_15pct": float(slow.mean()), "share_faster_15pct": float((r < 1 / 1.15).mean()),
            "slow_ratio_quantiles": {q: float(np.percentile(r[slow], q)) for q in (10, 50, 90)} if slow.any() else None,
            "spearman_r_vs": {"sm_clock_end": sp(r, sm_end), "sm_clock_start": sp(r, sm_start),
                              "host_load1": sp(r, [x["load"] for x in rows]), "cpu_busy": sp(r, [x["busy"] for x in rows])},
            "sm_end_below_max_share": float(np.mean(sm_end < smax)), "sm_end_max_mhz": smax,
            "slow_share_if_sm_end_below_max": float(slow[sm_end < smax].mean()) if (sm_end < smax).any() else None,
            "slow_share_if_sm_end_at_max": float(slow[sm_end >= smax].mean()),
            "share_of_slow_blocks_with_sm_end_below_max": float(np.mean(sm_end[slow] < smax)) if slow.any() else None,
            "slow_runs_in_time": {"n": len(runs), "mean_length": float(np.mean(runs)) if runs else 0.0,
                                  "longest": int(max(runs)) if runs else 0},
            "slow_share_by_round": [float(np.mean([x["r"] > 1.15 for x in rows if x["round"] == rd])) for rd in range(10)],
            "fp32_bf16_slow_share": float(np.mean([x["r"] > 1.15 for x in rows if "_none|" in x["cond"]]))}
    # SM clock below the A5000's usual boost value (1,890 MHz in both measurements; a few posthoc readings show 1,905)
    red = sm_end < 1890
    desc["sm_end_below_1890"] = {"share": float(red.mean()), "slow_share_if_below": float(slow[red].mean()) if red.any() else None,
                                 "slow_share_if_ge_1890": float(slow[~red].mean()),
                                 "share_of_slow_blocks_below": float(np.mean(red[slow])) if slow.any() else None}
    fr = []
    for key, rr_ in F["blocks"].items():
        mm = float(np.median([b["median_ms"] for b in rr_]))
        fr += [(b["median_ms"] / mm, float(b["gpu_end"].split(",")[1])) for b in rr_]
    frr, fsm = np.array([x[0] for x in fr]), np.array([x[1] for x in fr])
    fred, fslow = fsm < 1890, frr > 1.15
    desc["first_measurement_sm_end_below_1890"] = {
        "share": float(fred.mean()), "slow_share_if_below": float(fslow[fred].mean()) if fred.any() else None,
        "slow_share_if_ge_1890": float(fslow[~fred].mean()), "share_of_slow_blocks_below": float(np.mean(fred[fslow])),
        "share_slower_15pct": float(fslow.mean())}
    # pairs: mode mismatch and the fast-fast subset
    rr = {(x["cond"], x["round"]): x["r"] for x in rows}
    mism, ff, allff = 0, {}, []
    for key, prs in P["pairs"].items():
        t, p, it = key.split("|")
        vals = []
        for pr in prs:
            a_, b_ = rr[(f"{t}|{p}_det|{it}", pr["round"])], rr[(f"{t}|{p}_shiter|{it}", pr["round"])]
            mism += int((a_ > 1.15) != (b_ > 1.15))
            if a_ <= 1.15 and b_ <= 1.15:
                vals.append(pr["overhead"])
        allff += vals
        ff[key] = {"n_fast_fast": len(vals), "median": float(np.median(vals)) if vals else None,
                   "min": float(min(vals)) if vals else None, "max": float(max(vals)) if vals else None}
    desc["pairs"] = {"n": 360, "mode_mismatch_share": mism / 360.0, "fast_fast_n": len(allff),
                     "fast_fast_overhead_quantiles": {q: float(np.percentile(allff, q)) for q in (5, 25, 50, 75, 95)},
                     "fast_fast_abs_le_5pct_share": float(np.mean(np.abs(allff) <= 0.05)),
                     "fast_fast_by_condition": ff}
    # the first measurement's unstable condition
    key = "cheetah-run|int4|it6"
    fb = {v: sorted(F["blocks"][f"cheetah-run|int4_{v}|it6"], key=lambda b: b["round"]) for v in ("det", "shiter")}
    desc["cheetah_int4_it6"] = {
        "first_pooled": F["dither_overhead"][key],
        "first_per_round": [(y["median_ms"] - x["median_ms"]) / x["median_ms"] for x, y in zip(fb["det"], fb["shiter"])],
        "posthoc_pair_overheads": cond[key]["pair_overheads"], "posthoc_median": cond[key]["median"],
        "posthoc_ci": cond[key]["ci"], "posthoc_pooled": cond[key]["pooled_same_data"]}
    # condition summaries by kernel
    by = {}
    for p, _ in PAIRS:
        c = {k: v for k, v in cond.items() if k.split("|")[1] == p}
        by[p] = {"median_max": max(v["median"] for v in c.values()), "median_min": min(v["median"] for v in c.values()),
                 "ci_upper_max": max(v["ci"][1] for v in c.values()), "ci_lower_min": min(v["ci"][0] for v in c.values()),
                 "pooled_max": max(v["pooled_same_data"] for v in c.values()),
                 "pooled_min": min(v["pooled_same_data"] for v in c.values()),
                 "n_ci_upper_le_10pct": sum(v["ci"][1] <= 0.10 for v in c.values()),
                 "n_ci_contains_0": sum(v["ci"][0] <= 0 <= v["ci"][1] for v in c.values()),
                 "fast_fast_median_max": max(ff[k]["median"] for k in c if ff[k]["median"] is not None)}
    with open(f"{out}/posthoc_latency_blocks.csv", "w") as f:
        cols = ["cond", "round", "unit_pos", "pos_in_unit", "t_start", "median_ms", "r", "load", "busy", "sm_start",
                "sm_end", "temp_end", "power_end", "foreign"]
        f.write(",".join(cols) + "\n")
        for x in sorted(rows, key=lambda x: x["t_start"]):    # t_start (epoch s) needs more than 6 digits
            f.write(",".join((f"{x[c]:.3f}" if c == "t_start" else f"{x[c]:.6g}") if isinstance(x[c], float)
                             else str(x[c]) for c in cols) + "\n")
    return {"checks": chk, "conditions": cond, "by_kernel": by, "descriptive_posthoc": desc,
            "measurement": {"t_start_utc": P["t_start_utc"], "secs": P["secs"], "gpu": P["gpu"], "torch": P["torch"],
                            "loadavg1": [float(min(x["load"] for x in rows)), float(max(x["load"] for x in rows))],
                            "cpu_busy": [float(min(x["busy"] for x in rows)), float(max(x["busy"] for x in rows))],
                            "sm_end_mhz": [float(sm_end.min()), smax]}}


def load(res, pat):
    E = {}
    for f in glob.glob(f"{res}/jobs/episodes/{pat}"):
        d = json.load(gzip.open(f, "rt"))
        s = d["spec"]
        for e in d["episodes"]:
            E[(s["id"].split("_")[-2], s["task"], int(s["k"]), s["variant"], int(s["ckpt"]), int(e["ep"]))] = \
                (float(e["success"]) if s["task"].startswith("mw-") else float(e["ret"]), int(e["seed"]))
    return E


def qatnone(root):
    A = json.load(open(f"{root}/analysis_posthoc_qatnone/posthoc_qatnone_summary.json"))
    E = load(f"{root}/out_e5", "e5_*_k0_none_base_c*.json.gz")
    E.update(load(f"{root}/out_e5", "e5_*_e5qat_c*.json.gz"))
    E.update(load(f"{root}/out_posthoc_qatnone", "e5_*_e5qatnone_c*.json.gz"))
    out, bad = {}, 0
    for cell, a in A["cells"].items():
        t, k = cell.split("|")
        k, mw = int(k), t.startswith("mw-")
        kv = {"qat_none": ("e5qatnone", t, k, "none"), "qat_det": ("e5qat", t, k, "det"),
              "qat_shiter": ("e5qat", t, k, "shiter")}
        prs = {(x[4], x[5]) for x in E if x[:4] == ("base", t, 0, "none")}
        for v in kv.values():
            prs &= {(x[4], x[5]) for x in E if x[:4] == v}
        prs = sorted(prs)
        seeds_ok = all(E[("base", t, 0, "none") + q][1] == 60000 + q[1] and E[kv["qat_none"] + q][1] == 60000 + q[1]
                       for q in prs)
        idx = np.random.RandomState(0).randint(0, len(prs), size=(NB, len(prs)))
        y0 = np.array([E[("base", t, 0, "none") + q][0] for q in prs])
        r, Db = {"n_pairs": len(prs), "seeds_e5": seeds_ok}, {}
        for n, v in kv.items():
            y = np.array([E[v + q][0] for q in prs])
            D = (y0.mean() - y.mean()) if mw else (y0.mean() - y.mean()) / y0.mean()
            b0, b1 = y0[idx].mean(1), y[idx].mean(1)
            Db[n] = (b0 - b1) if mw else (b0 - b1) / b0
            r[n] = {"D": float(D), "ci": [float(np.percentile(Db[n], 2.5)), float(np.percentile(Db[n], 97.5))]}
            bad += abs(r[n]["D"] - a[n]["D"]) > 1e-12 or max(abs(x - y) for x, y in zip(r[n]["ci"], a[n]["D_ci"])) > 1e-12
        d = Db["qat_det"] - Db["qat_none"]
        r["qat_det_minus_qat_none"] = {"D": r["qat_det"]["D"] - r["qat_none"]["D"],
                                       "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
        bad += max(abs(x - y) for x, y in zip(r["qat_det_minus_qat_none"]["ci"], a["qat_det_minus_qat_none_ci"])) > 1e-12
        d2 = Db["qat_shiter"] - Db["qat_none"]          # descriptive, post-hoc: dither on the QAT weights vs no quantization
        r["qat_shiter_minus_qat_none"] = {"D": r["qat_shiter"]["D"] - r["qat_none"]["D"],
                                          "ci": [float(np.percentile(d2, 2.5)), float(np.percentile(d2, 97.5))]}
        out[cell] = r
    return {"cells": out, "mismatches_with_analysis": int(bad), "missing_jobs": A["n_jobs_missing"],
            "results_not_reading_e5_qat": len(A["results_not_reading_the_e5_qat_file"]),
            "all_cells_90_pairs": all(v["n_pairs"] == 90 for v in out.values()),
            "all_seeds_e5": all(v["seeds_e5"] for v in out.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    root = os.path.expanduser(a.root)
    os.makedirs(a.outdir, exist_ok=True)
    L = latency(root, a.outdir)
    Q = qatnone(root)
    json.dump({"latency": L, "qatnone": Q}, open(f"{a.outdir}/posthoc_check_summary.json", "w"), indent=1)
    print(json.dumps({"latency_checks": L["checks"], "by_kernel": L["by_kernel"],
                      "two_mode": {k: v for k, v in L["descriptive_posthoc"].items() if k not in ("pairs",)},
                      "pairs": {k: v for k, v in L["descriptive_posthoc"]["pairs"].items() if k != "fast_fast_by_condition"},
                      "qatnone": {k: v for k, v in Q.items() if k != "cells"}}, indent=1))


if __name__ == "__main__":
    main()
