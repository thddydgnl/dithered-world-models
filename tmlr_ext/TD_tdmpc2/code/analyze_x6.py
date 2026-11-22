"""X6 analysis (tmlr_ext/DESIGN_tmlr.md 11.1, D7; post hoc, descriptive, no thresholds; frozen in commit 24a59a00).
TD-MPC2 5M planning-free policy baseline on pod td.8 (A40) vs the default planner (none) on the same pod and pairs.
  - per task: policy success rate (MW) / return (DMC), D_policy = none - policy (MW) or 1 - R_policy / R_none (DMC) on
    the 90 pairs (ckpt, seed 100000 + ep), paired bootstrap 10,000 resamples RandomState(0), percentile 95% CI
  - E12's lowest-k usable configuration (td_e12_cells.csv): (a) dither: among (range PT / PC) x (sh-iter, det+shift)
    the smallest k*, ties broken by the smaller D at k*, then the order PT, PC and sh-iter, det+shift; (b) any rounding
    (the E12 'best', same tie rule); its D and CI at k = k* (E12 pod, same A40 and seeds; not paired with td.8)
  - per task the dither configuration is "better" than the policy if its D CI upper < D_policy CI lower, "worse" if
    its D CI lower > D_policy CI upper, otherwise "same" (CIs overlap); counted per unit (MW, DMC)
  - check: td.8's none equals E11's none (td.1-td.6 job files) episode by episode (return and success, bitwise)
Usage: python analyze_x6.py --res <td.8 out> --e11 <dir with E11 none job files> --e12 <td_e12_cells.csv> --outdir <dir>
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_td as A  # noqa: E402
import plan_td as P  # noqa: E402

ORDER = [("PT", "shiter"), ("PT", "detshift"), ("PC", "shiter"), ("PC", "detshift")]
ALL = [(rk, v) for rk in ("PT", "PC") for v in ("det", "detbc", "shiter", "detshift")]


def e12_rows(path):
    rows = {}
    for r in csv.DictReader(open(path)):
        ci = [float(x) for x in r["D_ci"].strip("[]").split(";")]
        rows[(r["task"], r["range"], r["rounding"], int(r["k"]))] = {"D": float(r["D"]), "D_ci": ci, "n": int(r["n"]),
                                                                    "kstar": int(r["kstar"]), "mean": float(r["mean"])}
    return rows


def lowest(rows, task, configs):
    cand = []
    for i, (rk, v) in enumerate(configs):
        ks = [k for (t, r_, vv, k) in rows if t == task and r_ == rk and vv == v]
        if not ks:
            continue
        kstar = rows[(task, rk, v, ks[0])]["kstar"]
        x = rows.get((task, rk, v, kstar))
        if x is None:
            continue
        cand.append((kstar, x["D"], i, {"range": rk, "rounding": v, "k": kstar, **x}))
    if not cand:
        return None
    cand.sort(key=lambda c: (c[0], c[1], c[2]))
    best = cand[0][3]
    best["ties"] = [f"{c[3]['range']}|{c[3]['rounding']}" for c in cand if c[0] == cand[0][0]]
    return best


def compare(dci, pci):
    if dci[1] < pci[0]:
        return "better"
    if dci[0] > pci[1]:
        return "worse"
    return "same"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--e11", required=True)
    ap.add_argument("--e12", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    E, _, meta = A.load([a.res])
    E11, _, _ = A.load([a.e11])
    rows12 = e12_rows(a.e12)
    out, counts = {}, {"mw": {"better": 0, "same": 0, "worse": 0}, "dmc": {"better": 0, "same": 0, "worse": 0}}
    for t in P.MW8 + P.DMC4:
        mw = A.is_mw(t)
        key = "success" if mw else "ret"
        pairs = sorted({(c, ep) for c in P.CKPTS for ep in E.get((t, c, "none", "base"), {})} &
                       {(c, ep) for c in P.CKPTS for ep in E.get((t, c, "none", "policy"), {})})
        if not pairs:
            continue
        y0 = np.array([E[(t, c, "none", "base")][ep][key] for c, ep in pairs], float)
        y1 = np.array([E[(t, c, "none", "policy")][ep][key] for c, ep in pairs], float)
        idx = A.boot_idx(len(pairs))
        D = A.drop(float(y0.mean()), float(y1.mean()), mw)
        Db = A.drop(y0[idx].mean(1), y1[idx].mean(1), mw)
        r = {"unit": "mw" if mw else "dmc", "n": len(pairs), "none_mean": float(y0.mean()), "policy_mean": float(y1.mean()),
             "D_policy": float(D), "D_policy_ci": A.ci95(Db)}
        # bit-identity of td.8 none with E11 none
        tot = eq = 0
        for c in P.CKPTS:
            x, y = E.get((t, c, "none", "base"), {}), E11.get((t, c, "none", "base"), {})
            for ep in set(x) & set(y):
                tot += 1
                eq += int(x[ep]["ret"] == y[ep]["ret"] and x[ep]["success"] == y[ep]["success"])
        r["none_vs_e11"] = {"n": tot, "identical": eq}
        dith = lowest(rows12, t, ORDER)
        anyb = lowest(rows12, t, ALL)
        r["e12_lowest_dither"] = dith
        r["e12_lowest_any"] = anyb
        if dith:
            r["dither_vs_policy"] = compare(dith["D_ci"], r["D_policy_ci"])
            counts[r["unit"]][r["dither_vs_policy"]] += 1
        out[t] = r
    S = {"source": "analyze_x6.py (D7, descriptive)", "meta": {k: meta[k] for k in ("files", "gpus", "seed_mismatch",
                                                                                   "pairing_one_host_per_task_ckpt")},
         "n_episodes": sum(len(d) for d in E.values()), "tasks": out, "counts_dither_vs_policy": counts,
         "rules": "dither config = smallest E12 k* over (PT, PC) x (sh-iter, det+shift), ties by smaller D at k*, then "
                  "PT, PC, sh-iter, det+shift; better / worse = disjoint 95% CIs, same = overlapping CIs; E12 values "
                  "come from other pods (same A40, same seeds): not paired"}
    json.dump(S, open(f"{a.outdir}/x6_summary.json", "w"), indent=1, default=float)
    with open(f"{a.outdir}/x6_cells.csv", "w") as f:
        f.write("task,unit,n,none_mean,policy_mean,D_policy,ci_lo,ci_hi,dither_range,dither_rounding,dither_k,"
                "D_dither,dither_ci_lo,dither_ci_hi,dither_n,dither_vs_policy,any_range,any_rounding,any_k,D_any,"
                "none_identical_e11\n")
        for t, r in out.items():
            d, b = r.get("e12_lowest_dither") or {}, r.get("e12_lowest_any") or {}
            f.write(f"{t},{r['unit']},{r['n']},{r['none_mean']:.6g},{r['policy_mean']:.6g},{r['D_policy']:.6g},"
                    f"{r['D_policy_ci'][0]:.6g},{r['D_policy_ci'][1]:.6g},{d.get('range')},{d.get('rounding')},"
                    f"{d.get('k')},{d.get('D')},{(d.get('D_ci') or [None, None])[0]},{(d.get('D_ci') or [None, None])[1]},"
                    f"{d.get('n')},{r.get('dither_vs_policy')},{b.get('range')},{b.get('rounding')},{b.get('k')},"
                    f"{b.get('D')},{r['none_vs_e11']['identical']}/{r['none_vs_e11']['n']}\n")
    print(json.dumps({"counts": counts, "none_identical": {t: r["none_vs_e11"] for t, r in out.items()},
                      "n_episodes": S["n_episodes"]}, default=float))


if __name__ == "__main__":
    main()
