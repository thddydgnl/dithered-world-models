"""POST-HOC: H2b-style within-task Spearman with the drift component (-drift) instead of -OEI (not a judgement)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/pmx/A/tdmpc2_main/code"))
from analyze_e1 import spearman
A = os.path.expanduser("~/pmx/A/tdmpc2_main/analysis_e1")
dec = json.load(open(f"{A}/e1_tdmpc2_posthoc_oei_decomposition.json"))
pts = json.load(open(f"{A}/e1_tdmpc2_h2b_points.json"))
res, diffs = {}, []
for t in sorted({p["task"] for p in pts}):
    P = [p for p in pts if p["task"] == t]
    D = [p["D"] for p in P]
    nd = [-dec[f"{p['task']}:{p['k']}:{p['variant']}"]["drift"] for p in P]
    ns = [-dec[f"{p['task']}:{p['k']}:{p['variant']}"]["sel_last"] for p in P]
    r_d, r_s = spearman(nd, D), spearman(ns, D)
    r_1, r_3 = spearman([p["pt_err1h"] for p in P], D), spearman([p["pt_err3"] for p in P], D)
    diff = r_d - max(r_1, r_3)
    diffs.append(diff)
    res[t] = (round(r_d, 3), round(r_s, 3), round(r_1, 3), round(r_3, 3), round(diff, 3))
    print(t, "rho(-drift,D) %.3f rho(-sel_last,D) %.3f rho(err1,D) %.3f rho(err3,D) %.3f diff %.3f" % res[t])
print("mean within-task diff with -drift: %.3f" % np.mean(diffs))
json.dump({"per_task": res, "mean_diff": float(np.mean(diffs))}, open(f"{A}/e1_tdmpc2_posthoc_h2b_drift.json", "w"), indent=1)
