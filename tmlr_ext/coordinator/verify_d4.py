"""Coordinator recomputation of D4 (post hoc PushT 3-bit E11 cell on dw.6) from raw job JSON."""
import glob, json, os, re, sys
import numpy as np
ROOT = sys.argv[1]  # DW_dinowm/results/raw
NB, TOL = 10000, 1e-9
PAT = re.compile(r"^eval8_(?:main_(?P<var>[a-z]+)_k3|none)_cem_e(?P<a>\d+)-(?P<b>\d+)\.json$")
cells = {}
for f in glob.glob(os.path.join(ROOT, "dw.6", "jobs", "pusht", "eval8_*_cem_e*.json")):
    m = PAT.match(os.path.basename(f))
    if not m:
        continue
    v = m.group("var") or "none"
    for r in json.load(open(f))["records"]:
        cells.setdefault(v, {})[r["ep"]] = float(bool(r["success"]))
V = ["det", "detbc", "shstep", "detshiftfix", "shiterb", "shiter", "detshift", "detnoise"]
eps = sorted(cells["none"])
assert all(sorted(cells[v]) == eps for v in V), {v: len(cells.get(v, {})) for v in V}
N = len(eps); none = np.array([cells["none"][e] for e in eps])
X = {v: np.array([cells[v][e] for e in eps]) for v in V}
idx = np.random.RandomState(0).randint(0, N, size=(NB, N))
D = {v: float(np.mean(none - X[v])) for v in V}
Db = {v: (none[idx] - X[v][idx]).mean(1) for v in V}
ci = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
p = float(np.mean(Db["det"] <= 0))
red = {v: (D["det"] - D[v]) / D["det"] for v in V}
dp = ci(Db["detshiftfix"] - Db["detshift"]); db = ci(Db["det"] - Db["detbc"])
fix = lambda v: red[v] >= 0.5 - TOL
if fix("detbc"): rd = "bias"
elif dp[0] > TOL: rd = "persistence"
elif fix("detshiftfix") and dp[0] <= TOL <= dp[1] + 2 * TOL: rd = "resolution"
else: rd = "mixed"
print(f"N={N} none={none.mean():.2f} D_det={D['det']:+.3f} {ci(Db['det'])} p={p:.4f} reading={rd}")
print("succ: " + " ".join(f"{v}={X[v].mean():.2f}" for v in V))
print("red:  " + " ".join(f"{v}={red[v]:.2f}" for v in V))
print(f"d_pers={D['detshiftfix']-D['detshift']:+.3f} {dp}  d_bias={D['det']-D['detbc']:+.3f} {db}")
print(f"shiterb-shiter={D['shiterb']-D['shiter']:+.3f} {ci(Db['shiterb']-Db['shiter'])}  detnoise-det={D['detnoise']-D['det']:+.3f} {ci(Db['detnoise']-Db['det'])}  shplan-shiter={D['shstep']-D['shiter']:+.3f} {ci(Db['shstep']-Db['shiter'])}")
