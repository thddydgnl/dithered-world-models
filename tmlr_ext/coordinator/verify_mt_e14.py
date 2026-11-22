"""Coordinator recomputation of E14 (317M) from the raw job files (independent of analyze_e14.py)."""
import glob, gzip, json, os, re, sys
import numpy as np

ROOT = sys.argv[1]
NB, TOL = 10000, 1e-9
eps = {}   # (task, variant) -> {seed: (success, rng_sha1)}
ks = {}
for pod in ("mt.1", "mt.2"):
    for f in glob.glob(os.path.join(ROOT, pod, "jobs", "eval", "eval_*.json.gz")):
        d = json.load(gzip.open(f))
        if not d.get("ok", True):
            continue
        s = d["spec"]
        if s["variant"] != "none":
            ks[s["task"]] = s["k"]
        for e in d["episodes"]:
            eps.setdefault((s["task"], s["variant"]), {})[e["seed"]] = (float(e["success"]), e["rng_sha1"], pod)
tasks = sorted({t for t, _ in eps})
V = ["det", "shiter", "shep", "detshift", "detbc"]
seeds = list(range(102000, 102090))
idx = np.random.RandomState(0).randint(0, 90, size=(NB, 90))
rows, pvals = {}, []
rng_bad = 0
for t in tasks:
    none = eps[(t, "none")]
    assert sorted(none) == seeds, (t, len(none))
    S = {"none": np.array([none[s][0] for s in seeds])}
    pods = {none[s][2] for s in seeds}
    for v in V:
        m = eps[(t, v)]
        assert sorted(m) == seeds, (t, v, len(m))
        S[v] = np.array([m[s][0] for s in seeds])
        pods |= {m[s][2] for s in seeds}
        rng_bad += sum(m[s][1] != none[s][1] for s in seeds)
    assert len(pods) == 1, (t, pods)
    D = {v: float(np.mean(S["none"] - S[v])) for v in V}
    Db = {v: (S["none"][idx] - S[v][idx]).mean(1) for v in V}
    p = float(np.mean(Db["det"] <= 0))
    pvals.append(p)
    red = {v: (D["det"] - D[v]) / D["det"] if D["det"] > 0 else None for v in V}
    harm = {v: float(np.percentile(Db[v] - Db["det"], 2.5)) > 0.05 + TOL for v in V}
    dd = Db["shep"] - Db["shiter"]
    rows[t] = dict(k=ks[t], pod=pods.pop(), none=float(S["none"].mean()), D=D, p=p, red=red, harm=harm,
                   shep_m_shiter=(float(D["shep"] - D["shiter"]), float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))),
                   ci_det=(float(np.percentile(Db["det"], 2.5)), float(np.percentile(Db["det"], 97.5))))
# Holm (strict <)
order = np.argsort(pvals)
m = len(tasks)
rej = [False] * m
for i, j in enumerate(order):
    if pvals[j] < 0.05 / (m - i) - TOL:
        rej[j] = True
    else:
        break
dmg = [t for t, r in zip(tasks, rej) if r and rows[t]["D"]["det"] > 0]
print("rng_sha1 mismatches vs none:", rng_bad)
for t in tasks:
    r = rows[t]
    print(f"{t:17s} k{r['k']} {r['pod']} none={r['none']:.3f} D_det={r['D']['det']:+.3f} [{r['ci_det'][0]:+.3f},{r['ci_det'][1]:+.3f}] p={r['p']:.4f} "
          f"red_shiter={r['red']['shiter'] if r['red']['shiter'] is None else round(r['red']['shiter'],3)} "
          f"red_detshift={None if r['red']['detshift'] is None else round(r['red']['detshift'],3)} red_detbc={None if r['red']['detbc'] is None else round(r['red']['detbc'],3)} "
          f"r_shep/det={None if r['D']['det']<=0 else round(r['D']['shep']/r['D']['det'],3)} shep-shiter={r['shep_m_shiter'][0]:+.3f} [{r['shep_m_shiter'][1]:+.3f},{r['shep_m_shiter'][2]:+.3f}] "
          f"harm={[v for v in V if r['harm'][v]]}")
n_fix = sum(rows[t]["red"]["shiter"] >= 0.5 - TOL and not rows[t]["harm"]["shiter"] for t in dmg)
n_fix_red_only = sum(rows[t]["red"]["shiter"] >= 0.5 - TOL for t in dmg)
P1 = len(dmg) >= 6 and n_fix_red_only >= 0.75 * len(dmg) - TOL
n_p2 = sum(rows[t]["shep_m_shiter"][1] > 0 + TOL for t in dmg)
P2 = n_p2 > len(dmg) / 2
print(f"damaged {len(dmg)}/8: {dmg}")
print(f"P1: damaged>=6 {len(dmg)>=6}, sh-iter red>=0.5 in {n_fix_red_only}/{len(dmg)} -> {P1}")
print(f"P2: CI low of D(sh-ep)-D(sh-iter) > 0 in {n_p2}/{len(dmg)} -> {P2}")
