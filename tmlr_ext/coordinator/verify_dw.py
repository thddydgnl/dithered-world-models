"""Coordinator recomputation of the DINO-WM readings (E11 B2/B3 unit, E12 k*/G, E16 2x2 half) from raw job files."""
import glob, gzip, json, os, re, sys
import numpy as np

ROOT = sys.argv[1]
NB, TOL, TAU = 10000, 1e-9, 0.10
PAT = re.compile(r"^(?P<split>eval8|held)_(?:(?P<rng>main|pc|rc_k\d)_(?P<var>[a-z]+)_k(?P<k>\d)|none)_(?P<pl>cem|mppi[\d.]+)_e(?P<a>\d+)-(?P<b>\d+)\.json(\.gz)?$")
cells = {}  # (pod, task, split, rng, var, k, pl) -> {ep: success}
for f in glob.glob(os.path.join(ROOT, "dw.*", "jobs", "*", "*.json*")):
    m = PAT.match(os.path.basename(f))
    if not m:
        continue
    pod = f.split(os.sep)[-4]; task = f.split(os.sep)[-2]
    d = json.load(gzip.open(f) if f.endswith(".gz") else open(f))
    g = m.groupdict()
    key = (pod, task, g["split"], g["rng"] or "-", g["var"] or "none", int(g["k"] or 0), g["pl"])
    for r in d["records"]:
        cells.setdefault(key, {})[r["ep"]] = float(bool(r["success"]))

def none_of(pod, task, pl, split="eval8"):
    return cells.get((pod, task, split, "-", "none", 0, pl), {})

def stats(pod, task, rng, k, pl, variants, idx_seed=0):
    nn = none_of(pod, task, pl)
    out = {}
    eps = sorted(nn)
    for v in variants:
        c = cells.get((pod, task, "eval8", rng, v, k, pl))
        if c is None:
            continue
        e = sorted(set(c) & set(nn))
        out[v] = (np.array([nn[i] for i in e]), np.array([c[i] for i in e]), e)
    return out

def boot(n):
    return np.random.RandomState(0).randint(0, n, size=(NB, n))

def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

E11 = {"pusht": ("dw.1", 4), "pointmaze": ("dw.3", 3), "wall": ("dw.4", 3)}
V11 = ["det", "detbc", "shstep", "detshiftfix", "shiterb", "shiter", "detshift", "detnoise"]
res, pvals = {}, []
for task, (pod, k) in E11.items():
    S = stats(pod, task, "main", k, "cem", V11)
    n = {v: len(S[v][2]) for v in S}
    base_eps = S["det"][2]
    assert all(S[v][2] == base_eps for v in S), (task, n)
    N = len(base_eps)
    none = S["det"][0]
    X = {v: S[v][1] for v in S}
    idx = boot(N)
    D = {v: float(np.mean(none - X[v])) for v in X}
    Db = {v: (none[idx] - X[v][idx]).mean(1) for v in X}
    p = float(np.mean(Db["det"] <= 0)); pvals.append(p)
    red = {v: (D["det"] - D[v]) / D["det"] if D["det"] > 0 else float("nan") for v in X}
    harm = {v: ci(Db[v] - Db["det"])[0] > 0.05 + TOL for v in X}
    dpers = Db["detshiftfix"] - Db["detshift"]; dbias = Db["det"] - Db["detbc"]
    res[task] = dict(N=N, none=float(none.mean()), D=D, p=p, red=red, harm=harm,
                     dpers=(D["detshiftfix"] - D["detshift"],) + ci(dpers), dbias=(D["det"] - D["detbc"],) + ci(dbias),
                     shep_shiter=(D["shstep"] - D["shiter"],) + ci(Db["shstep"] - Db["shiter"]))
order = np.argsort(pvals); m = len(pvals); rej = [False] * m
for i, j in enumerate(order):
    if pvals[j] < 0.05 / (m - i) - TOL: rej[j] = True
    else: break
readings = {}
for (task, r), rj in zip(res.items(), rej):
    r["damaged"] = bool(rj and r["D"]["det"] > 0)
    fix = lambda v: r["red"][v] >= 0.5 - TOL
    if fix("detbc"): rd = "bias"
    elif r["dpers"][1] > 0 + TOL: rd = "persistence"
    elif fix("detshiftfix") and r["dpers"][1] <= 0 + TOL <= r["dpers"][2] + 2 * TOL: rd = "resolution"
    else: rd = "mixed"
    r["reading"] = rd
    print(f"E11 {task:9s} N={r['N']} none={r['none']:.2f} D_det={r['D']['det']:+.3f} p={r['p']:.4f} damaged={r['damaged']} reading={rd}")
    print("    D: " + " ".join(f"{v}={r['D'][v]:+.3f}" for v in V11))
    print("    red: " + " ".join(f"{v}={r['red'][v]:.2f}" for v in V11) + f"  harm={[v for v in V11 if r['harm'][v]]}")
    print(f"    d_bias={r['dbias'][0]:+.3f} [{r['dbias'][1]:+.3f},{r['dbias'][2]:+.3f}]  d_pers={r['dpers'][0]:+.3f} [{r['dpers'][1]:+.3f},{r['dpers'][2]:+.3f}]  shplan-shiter={r['shep_shiter'][0]:+.3f} [{r['shep_shiter'][1]:+.3f},{r['shep_shiter'][2]:+.3f}]")
dmg = [t for t in res if res[t]["damaged"]]
from collections import Counter
cnt = Counter(res[t]["reading"] for t in dmg)
unit = next((rd for rd, c in cnt.items() if c > len(dmg) / 2), "mixed")
print(f"DW unit: damaged {dmg}, readings {dict(cnt)}, unit reading = {unit}; B2 (not persistence; bias or resolution) -> {unit in ('bias','resolution')}")
nfix_noise = sum(res[t]["red"]["detnoise"] >= 0.5 - TOL for t in dmg)
print(f"B3 DW unit: det+noise fixes {nfix_noise}/{len(dmg)} damaged cells -> unit part {'fails' if nfix_noise > len(dmg)/2 else 'holds'}")

# E16 DINO-WM half: MPPI-style planner with the held-selected lambda
for task, pod, pl in (("pointmaze", "dw.3", "mppi2"), ("wall", "dw.4", "mppi1")):
    nm, nc = none_of(pod, task, pl), none_of(pod, task, "cem")
    S = stats(pod, task, "main", 3, pl, ["det", "shiter", "shstep", "detshift"])
    if not S or len(nm) == 0:
        print("E16", task, "missing"); continue
    e = S["det"][2]; N = len(e); idx = boot(N)
    none = S["det"][0]; X = {v: S[v][1] for v in S}
    Db = {v: (none[idx] - X[v][idx]).mean(1) for v in X}
    D = {v: float(np.mean(none - X[v])) for v in X}
    q = np.mean(list(nm.values())) / np.mean(list(nc.values()))
    lo, hi = ci(Db["shstep"] - Db["shiter"])
    print(f"E16 {task} {pl}: none mppi {np.mean(list(nm.values())):.2f} vs cem {np.mean(list(nc.values())):.2f} (ratio {q:.2f}, qualifies {q >= 0.6 - TOL}); "
          f"D det {D['det']:+.3f}, shiter {D['shiter']:+.3f}, shplan {D['shstep']:+.3f}, detshift {D['detshift']:+.3f}; shplan-shiter {D['shstep']-D['shiter']:+.3f} [{lo:+.3f},{hi:+.3f}]")

# E12 DINO-WM: D table and k* per (task, range, rounding)
print("E12 cells (pod, task, rng, var, k): N, D")
grid = {}
for (pod, task, split, rng, var, k, pl), c in sorted(cells.items()):
    if split != "eval8" or pl != "cem" or var == "none" or var not in ("det", "detbc", "shiter", "detshift"):
        continue
    nn = none_of(pod, task, "cem")
    e = sorted(set(c) & set(nn))
    if not e: continue
    D = float(np.mean([nn[i] - c[i] for i in e]))
    grid.setdefault((task, rng, var), {})[k] = (D, len(e), pod)
for (task, rng, var), kd in sorted(grid.items()):
    ksorted = sorted(kd)
    kstar = None
    for kk in ksorted:
        if all(kd[j][0] <= TAU + TOL for j in ksorted if j >= kk):
            kstar = kk; break
    print(f"  {task:9s} {rng:6s} {var:8s} " + " ".join(f"k{kk}:{kd[kk][0]:+.3f}(n{kd[kk][1]},{kd[kk][2]})" for kk in ksorted) + f"  k*={kstar}")
