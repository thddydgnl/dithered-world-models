"""사후 분석 (post-hoc, not part of the pre-registered verdict).

P1  weak vs prediction-matched random at EVERY alpha (not only alpha*): paired per-episode difference of the relative
    drop (same seed = same initial state), paired-bootstrap 90% CI, and the S2-style cell test
    (drop_w >= 2 x max(drop_rp, 0) and drop_w >= 0.05) for every (task, scope, alpha) cell.
P2  error compounding: err3 / err1 of weak vs prediction-matched random (matched at 1 step), WM scope.
Writes <out>/results/posthoc.json and table_posthoc.csv
Usage: python posthoc.py --out /workspace/out_main
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def akey(a):
    return f"{float(a):g}"


def main(out):
    R = f"{out}/results"
    rows = []
    for f in ("episodes.jsonl", "episodes_q.jsonl"):
        if os.path.exists(f"{R}/{f}"):
            rows += [json.loads(l) for l in open(f"{R}/{f}") if l.strip()]
    ep = pd.DataFrame(rows)
    ep["scope"] = ep.get("scope", "wm")
    ep["scope"] = ep["scope"].fillna("wm")
    ep["alpha"] = ep["alpha"].astype(float)
    cells = []
    rng = np.random.default_rng(0)
    for t in ep.task.unique():
        e = ep[ep.task == t]
        metric = "success" if t.startswith("mw-") else "ret"
        base = e[e.cond == "none"].set_index("ep")
        R0 = base[metric].mean()
        for sp in ("wm", "q", "all"):
            for a in sorted(e[(e.scope == sp) & (e.cond == "weak")].alpha.unique()):
                w = e[(e.scope == sp) & (e.cond == "weak") & np.isclose(e.alpha, a)].set_index("ep")
                p = e[(e.scope == sp) & (e.cond == "rand_pred") & np.isclose(e.alpha, a)].set_index("ep")
                common = w.index.intersection(p.index)
                if not len(common):
                    continue
                dw = (R0 - w.loc[common, metric].mean()) / R0
                dp = (R0 - p.loc[common, metric].mean()) / R0
                diff = ((p.loc[common, metric] - w.loc[common, metric]) / R0).values  # > 0: weak hurts more
                b = rng.choice(diff, size=(4000, len(diff)), replace=True).mean(1)
                ret_w = w.loc[common, "ret"].mean()
                ret_p = p.loc[common, "ret"].mean()
                cells.append({"task": t, "scope": sp, "alpha": a, "n": len(common), "metric": metric,
                              "drop_weak": dw, "drop_randpred": dp, "diff": float(diff.mean()),
                              "diff_ci90_lo": float(np.percentile(b, 5)), "diff_ci90_hi": float(np.percentile(b, 95)),
                              "ratio": dw / dp if dp > 0 else float("inf") if dw > 0 else float("nan"),
                              "s2_cell": bool(dw >= 2 * max(dp, 0) and dw >= 0.05),
                              "ret_drop_weak": (base.ret.mean() - ret_w) / base.ret.mean(),
                              "ret_drop_randpred": (base.ret.mean() - ret_p) / base.ret.mean()})
    C = pd.DataFrame(cells)
    C.to_csv(f"{R}/table_posthoc.csv", index=False)
    comp = []
    for t in ep.task.unique():
        f = f"{R}/prep_{t}.json"
        if not os.path.exists(f):
            continue
        p = json.load(open(f))
        for k, v in p["alphas"].items():
            comp.append({"task": t, "alpha": float(k), "err1": v["weak"]["err1"],
                         "err3_over_err1_weak": v["weak"]["err3"] / max(v["weak"]["err1"], 1e-12),
                         "err3_over_err1_randpred": v["rand_pred"]["err3"] / max(v["rand_pred"]["err1"], 1e-12)})
    P2 = pd.DataFrame(comp)
    res = {"P1_cells": cells, "P1_counts": {
        sp: {"cells": int((C.scope == sp).sum()), "weak_worse_ci_excl_0": int(((C.scope == sp) & (C.diff_ci90_lo > 0)).sum()),
             "randpred_worse_ci_excl_0": int(((C.scope == sp) & (C.diff_ci90_hi < 0)).sum()),
             "s2_cells": int(((C.scope == sp) & C.s2_cell).sum())} for sp in C.scope.unique()} if len(C) else {},
        "P2_compounding": comp,
        "P2_median_ratio": {"weak": float(P2.err3_over_err1_weak.median()),
                            "randpred": float(P2.err3_over_err1_randpred.median())} if len(P2) else {}}
    json.dump(res, open(f"{R}/posthoc.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 200)
    print(C.round(3).to_string())
    print(json.dumps(res["P1_counts"], indent=1), res["P2_median_ratio"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    main(ap.parse_args().out)
