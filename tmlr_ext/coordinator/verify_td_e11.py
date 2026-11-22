"""Coordinator re-check of the TD-MPC2 E11 readings (B1, B3 TD part, B4, DMC reading) from the raw episode table."""
import csv, collections, json, sys
import numpy as np
base = sys.argv[1]
rows = list(csv.DictReader(open(f"{base}/results/analysis/td_episodes.csv")))
PK = {"mw-push": 3, "mw-pick-place": 3, "mw-drawer-open": 3, "mw-button-press-topdown": 3, "mw-sweep-into": 3,
      "quadruped-walk": 3, "walker-walk": 3, "mw-door-open": 4, "mw-peg-insert-side": 4, "mw-window-open": 4,
      "finger-turn-hard": 4, "cheetah-run": 5}
cells = sorted({r["cell"] for r in rows if r["ptag"] == "base" and "none" in r["cell"]})
print("none cell names:", cells)
by = collections.defaultdict(dict)
for r in rows:
    if r["ptag"] != "base":
        continue
    key = (int(r["ckpt"]), int(r["seed"]))
    by[(r["task"], r["cell"])][key] = (float(r["ret"]), float(r["success"]))
VARS = ["det", "detbc", "shep", "detshiftfix", "shiterb", "shiter", "detshift", "detnoise"]
rng_B = 10000
out = {}
for task, k in PK.items():
    mw = task.startswith("mw-")
    none = by.get((task, "none"), {})
    if not none:
        cand = [c for (t, c) in by if t == task and "none" in c]
        none = by[(task, cand[0])] if cand else {}
    cellv = {v: by.get((task, f"PTk{k}_{v}"), {}) for v in VARS}
    keys = sorted(set(none) & set.intersection(*[set(cellv[v]) for v in VARS]))
    n = len(keys)
    idx = np.random.RandomState(0).randint(0, n, size=(rng_B, n))
    def arr(d, j): return np.array([d[kk][j] for kk in keys])
    if mw:
        s0 = arr(none, 1)
        def D(d, ix=None):
            sv = arr(d, 1)
            if ix is None: return s0.mean() - sv.mean()
            return s0[ix].mean(1) - sv[ix].mean(1)
    else:
        r0 = arr(none, 0)
        def D(d, ix=None):
            rv = arr(d, 0)
            if ix is None: return 1 - rv.sum() / r0.sum()
            return 1 - rv[ix].sum(1) / r0[ix].sum(1)
    Dp = {v: D(cellv[v]) for v in VARS}
    Db = {v: D(cellv[v], idx) for v in VARS}
    p_det = (np.sum(Db["det"] <= 0) + 1) / (rng_B + 1)
    red = {v: (Dp["det"] - Dp[v]) / Dp["det"] if Dp["det"] != 0 else float("nan") for v in VARS}
    pers = Db["detshiftfix"] - Db["detshift"]
    lo_pers, hi_pers = np.percentile(pers, [2.5, 97.5])
    harm = Db["shiter"] - Db["det"]
    out[task] = dict(n=n, D=Dp, p_det=p_det, red=red, pers_ci=(lo_pers, hi_pers), harm_lo=np.percentile(harm, 2.5))
# Holm within units
units = {"MW": [t for t in PK if t.startswith("mw-")], "DMC": [t for t in PK if not t.startswith("mw-")]}
for u, ts in units.items():
    ps = sorted((out[t]["p_det"], t) for t in ts)
    m = len(ps); dam = set()
    for i, (p, t) in enumerate(ps):
        if p <= 0.05 / (m - i) + 1e-9 and out[t]["D"]["det"] > 0: dam.add(t)
        else: break
    reads = collections.Counter(); b4 = 0; noise_fix = 0
    for t in ts:
        o = out[t]
        if t not in dam: continue
        rb = o["red"]["detbc"]; lo, hi = o["pers_ci"]
        if rb >= 0.5 - 1e-9: rd = "bias"
        elif lo > 1e-9: rd = "persistence"
        elif o["red"]["detshiftfix"] >= 0.5 - 1e-9 and lo <= 0 <= hi: rd = "resolution"
        else: rd = "mixed"
        reads[rd] += 1
        b4 += o["red"]["shiter"] >= 0.5 - 1e-9
        noise_fix += o["red"]["detnoise"] >= 0.5 - 1e-9
        print(f"{u} {t:26s} n={o['n']} D_det={o['D']['det']:.4f} red: bc={rb:.2f} shep={o['red']['shep']:.2f} dsfix={o['red']['detshiftfix']:.2f} shiterb={o['red']['shiterb']:.2f} shiter={o['red']['shiter']:.2f} dshift={o['red']['detshift']:.2f} dnoise={o['red']['detnoise']:.2f} pers_ci=[{lo:.3f},{hi:.3f}] -> {rd}")
    print(f"== {u}: damaged {len(dam)}/{len(ts)} readings {dict(reads)} sh-iter red>=0.5 {b4} det+noise fixes {noise_fix}")
