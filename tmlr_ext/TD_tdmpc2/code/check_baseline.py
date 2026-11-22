"""DESIGN_tmlr 0.7 check 3: none (default planner) on the new eval seeds 100000-100029 within 10 points (Meta-World
success rate) / 10% of the return (DMC) of E1's none (inputs/e1_none_baseline.json: e1_tdmpc2_summary.json baseline,
equal to the value implied by every row of e1_tdmpc2_cells.csv). Reads <out>/jobs/episodes/ev_<task>_s*_none_base_*.
Usage: python check_baseline.py --out /workspace/atm/out --tasks mw-push,... [--json <path>]
"""
import argparse
import glob
import gzip
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--tasks", required=True)
ap.add_argument("--json", default="")
a = ap.parse_args()
base = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs", "e1_none_baseline.json")))["tasks"]
res = {}
for t in a.tasks.split(","):
    mw = t.startswith("mw-")
    vals = []
    for f in sorted(glob.glob(f"{a.out}/jobs/episodes/ev_{t}_s*_none_base_c*.json.gz")):
        d = json.load(gzip.open(f, "rt"))
        vals += [e["success"] if mw else e["ret"] for e in d["episodes"]]
    e1 = base[t]["summary_baseline"]
    m = sum(vals) / len(vals) if vals else float("nan")
    lim = 0.10 if mw else 0.10 * e1
    ok = len(vals) == 90 and abs(m - e1) <= lim + 1e-9
    res[t] = {"n": len(vals), "none_new": m, "none_e1": e1, "diff": m - e1, "limit": lim, "pass": ok}
    print(f"{'PASS' if ok else 'FAIL'} {t}: none {m:.4f} (n={len(vals)}) vs E1 {e1:.4f}, diff {m - e1:+.4f}, limit {lim:.4f}")
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)
