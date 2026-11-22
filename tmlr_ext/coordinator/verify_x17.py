"""Coordinator recomputation of X17 (D5): PushT 4-bit on the E1 split (eval6) on dw.6 (RTX 4090)."""
import glob, json, os, re, sys
import numpy as np
ROOT = sys.argv[1]
NB = 10000
PAT = re.compile(r"^eval6_(?:main_(?P<var>[a-z]+)_k4|none)_cem_e(?P<a>\d+)-(?P<b>\d+)\.json$")
cells = {}
for f in glob.glob(os.path.join(ROOT, "dw.6", "jobs", "pusht", "eval6_*.json")):
    m = PAT.match(os.path.basename(f))
    if not m:
        continue
    v = m.group("var") or "none"
    for r in json.load(open(f))["records"]:
        cells.setdefault(v, {})[r["ep"]] = float(bool(r["success"]))
eps = sorted(cells["none"]); N = len(eps)
assert all(sorted(cells[v]) == eps for v in ("det", "shiter")), {v: len(c) for v, c in cells.items()}
none = np.array([cells["none"][e] for e in eps]); X = {v: np.array([cells[v][e] for e in eps]) for v in ("det", "shiter")}
idx = np.random.RandomState(0).randint(0, N, size=(NB, N))
D = {v: float(np.mean(none - X[v])) for v in X}; Db = {v: (none[idx] - X[v][idx]).mean(1) for v in X}
ci = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
lo, hi = ci(Db["det"]); red = (D["det"] - D["shiter"]) / D["det"] if D["det"] > 0 else float("nan")
E1 = (0.14, 0.34)
if lo > 1e-9 and not (hi < E1[0] or lo > E1[1]): verdict = "(i) split difference (eval8) explains the non-replication"
elif lo <= 1e-9 <= hi and hi < E1[0]: verdict = "(ii) GPU/numerics sensitivity"
else: verdict = "(iii) both open"
print(f"N={N} none={none.mean():.2f} (E1 0.83) det={X['det'].mean():.2f} (E1 0.59) shiter={X['shiter'].mean():.2f} (E1 0.83)")
print(f"D_det={D['det']:+.3f} [{lo:+.3f},{hi:+.3f}] (E1 +0.24 [0.14,0.34]); D_shiter={D['shiter']:+.3f} {ci(Db['shiter'])}; red(sh-iter)={red:.2f}")
print("D5 verdict:", verdict)
