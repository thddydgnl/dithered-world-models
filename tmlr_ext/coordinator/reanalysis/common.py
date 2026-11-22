"""Shared loaders and helpers for the pod-free re-analyses R1-R5 (JUDGE_write_vs_more_2026-11-22 section 3).

Reads only result files already on this Mac (no pod, no git); numpy only.
Conventions (DESIGN_tmlr v1.0 section 0.3 / 2.3):
  D        drop vs the paired none: MW and DINO-WM success-rate difference, DMC 1 - sum R_v / sum R_none
           (recomputed per resample).
  pairs    TD-MPC2: (ckpt, episode index) inside one pod per task; DINO-WM: eval8 pair index (shared by the pods of a
           task, each cell paired with the none of its own pod).
  boot     10,000 paired resamples, numpy.random.RandomState(0), 95% percentile CI.
  k*       smallest measured k with D <= tau such that every larger measured k also has D <= tau (tolerance 1e-9).
  G        min(k*(det), k*(det+BC)) - min(k*(sh-iter), k*(det+shift)); DINO-WM: dither side = k*(sh-iter) only,
           RC: k*(det) - k*(sh-iter).
Joint resampling (R1): one count-weight matrix W (B x n_pairs) per task, shared by every cell of the task (all ranges,
roundings and k). A cell that has only part of the pairs (two-stage cells) uses the columns of W of its own pairs.
Primary scheme: stratified by sampling stage (DMC: 45 stage-1 pairs = episodes 0-14 of the 3 checkpoints and 45 stage-2
pairs; PushT: eval8 pairs 0-49 and 50-99), so a half cell always gets exactly its own number of draws and shares them
with the full cells. MW, PointMaze and Wall have no two-stage cells (simple bootstrap over all pairs).
Censoring of k* (R1, R2): when the lowest measured k already passes (and k > 1), the true k* may be lower ("upper
bound"); when no measured k passes, the true k* is above the highest measured k ("lower bound"). The point convention
uses the boundary value (lowest measured k, or highest measured k + 1); every G is also given as an interval [G_lo, G_hi]
with the unmeasured side open (k down to 1, or up to 9 = "not reached by k = 8").
"""
import csv
import glob
import gzip
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TX = os.path.abspath(os.path.join(HERE, "..", ".."))          # A/tmlr_ext
A = os.path.dirname(TX)                                        # A_dithered_world_models
TD_EP = os.path.join(TX, "TD_tdmpc2", "results", "analysis", "td_episodes.csv")
TD_E12 = os.path.join(TX, "TD_tdmpc2", "results", "analysis", "td_e12_cells.csv")
TD_CON = os.path.join(TX, "TD_tdmpc2", "results", "analysis", "td_e11_contrasts.csv")
TD_SUM = os.path.join(TX, "TD_tdmpc2", "results", "analysis", "td_summary.json")
DW_RAW = os.path.join(TX, "DW_dinowm", "results", "raw")
DW_AN = os.path.join(TX, "DW_dinowm", "results", "analysis", "analysis_dw.json")
MT_EP = os.path.join(TX, "MT_317m", "results", "analysis", "e14_episodes.csv")
E1_DIR = os.path.join(A, "main", "results", "E1")
E1_CELLS = os.path.join(E1_DIR, "e1_dinowm_cells.csv")

NB, TOL, TAU = 10000, 1e-9, 0.10
KMIN, KMAX = 1, 8
KSENT = KMAX + 1                       # "not reached by k = 8"
MW8 = ["mw-push", "mw-pick-place", "mw-door-open", "mw-drawer-open", "mw-button-press-topdown",
       "mw-peg-insert-side", "mw-window-open", "mw-sweep-into"]
DMC4 = ["cheetah-run", "walker-walk", "quadruped-walk", "finger-turn-hard"]
K1 = {"mw-push": 3, "mw-pick-place": 3, "mw-door-open": 4, "mw-drawer-open": 3, "mw-button-press-topdown": 3,
      "mw-peg-insert-side": 4, "mw-window-open": 4, "mw-sweep-into": 3, "cheetah-run": 5, "walker-walk": 3,
      "quadruped-walk": 3, "finger-turn-hard": 4}
SHORT = {"mw-push": "push", "mw-pick-place": "pick-place", "mw-door-open": "door-open", "mw-drawer-open": "drawer-open",
         "mw-button-press-topdown": "button-press", "mw-peg-insert-side": "peg-insert", "mw-window-open": "window-open",
         "mw-sweep-into": "sweep-into", "cheetah-run": "cheetah", "walker-walk": "walker", "quadruped-walk": "quadruped",
         "finger-turn-hard": "finger", "pusht": "PushT", "pointmaze": "PointMaze", "wall": "Wall"}
TD_ROUND = ["det", "detbc", "shiter", "detshift"]
DW_TASKS = ["pusht", "pointmaze", "wall"]
DW_PODS = ["dw.1", "dw.2", "dw.3", "dw.4", "dw.5"]   # dw.6 = post hoc D4/D5 pod, not part of the published E11/E12/E16
DW_RANGES = {"pusht": ["PT", "PC", "RC"], "pointmaze": ["PT", "PC"], "wall": ["PT", "PC"]}
# Published point values (RESULTS_tmlr.md section 0 / 1.3, td_summary.json, analysis_dw.json)
PUB_TD_G = {"PT": {"mw-push": 2, "mw-pick-place": 0, "mw-door-open": 3, "mw-drawer-open": 2,
                   "mw-button-press-topdown": 2, "mw-peg-insert-side": 2, "mw-window-open": 2, "mw-sweep-into": 2,
                   "cheetah-run": 0, "walker-walk": 0, "quadruped-walk": 0, "finger-turn-hard": 1},
            "PC": {"mw-push": 0, "mw-pick-place": 1, "mw-door-open": 1, "mw-drawer-open": 1,
                   "mw-button-press-topdown": 1, "mw-peg-insert-side": 2, "mw-window-open": 2, "mw-sweep-into": 1,
                   "cheetah-run": 1, "walker-walk": 0, "quadruped-walk": 0, "finger-turn-hard": 1}}
PUB_DW_G = {("pusht", "PT"): 2, ("pusht", "PC"): 2, ("pusht", "RC"): 2, ("wall", "PT"): 2, ("wall", "PC"): 2,
            ("pointmaze", "PT"): 0, ("pointmaze", "PC"): 0}


def is_mw(t):
    return t.startswith("mw-")


def ci95(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def r6(x):
    return None if x is None or not np.isfinite(x) else float(round(float(x), 6))


# ----------------------------------------------------------------------------------------- resampling weights
def counts(idx, n):
    """(B, m) index matrix -> (B, n) count matrix."""
    B = idx.shape[0]
    off = idx + n * np.arange(B)[:, None]
    return np.bincount(off.ravel(), minlength=B * n).reshape(B, n).astype(float)


def w_simple(rs, n, B=NB):
    return counts(rs.randint(0, n, size=(B, n)), n)


def w_strat(rs, strata, n, B=NB):
    """strata: list of index arrays (into 0..n-1); each stratum resampled with replacement to its own size, in order."""
    W = np.zeros((B, n))
    for ids in strata:
        ids = np.asarray(ids)
        loc = rs.randint(0, len(ids), size=(B, len(ids)))
        W += counts(ids[loc], n)
    return W


def w_cluster(rs, comp, B=NB):
    """cluster bootstrap: draw the components with replacement; each pair gets its component's draw count."""
    comp = np.asarray(comp)
    labs = np.unique(comp)
    pos = np.searchsorted(labs, comp)
    C = counts(rs.randint(0, len(labs), size=(B, len(labs))), len(labs))
    return C[:, pos]


def ones(n):
    return np.ones((1, n))


def d_w(W, m, y0, yv, mw):
    """resampled D of one cell: W (B, n) weights, m bool mask of the cell's pairs."""
    Wm = W[:, m]
    with np.errstate(invalid="ignore", divide="ignore"):
        if mw:
            den = Wm.sum(1)
            return np.where(den > 0, (Wm @ (y0[m] - yv[m])) / np.where(den > 0, den, 1), np.nan)
        den = Wm @ y0[m]
        return np.where(den > 0, 1.0 - (Wm @ yv[m]) / np.where(den > 0, den, 1), np.nan)


# ----------------------------------------------------------------------------------------- k* and G
def kstar(ks, Dmat, tau=TAU):
    """ks increasing; Dmat (B, len(ks)). Returns dict of arrays: k (boundary convention), lo, hi, cens_up, cens_dn."""
    ks = np.asarray(ks)
    ok = np.nan_to_num(Dmat, nan=np.inf) <= tau + TOL
    q = np.flip(np.cumprod(np.flip(ok, 1), 1), 1).astype(bool)      # q[:, j] = all(ok[:, j:])
    anyq = q.any(1)
    first = np.argmax(q, 1)
    k = np.where(anyq, ks[first], ks[-1] + 1)
    cens_up = ~anyq                                                   # true k* > highest measured k
    cens_dn = anyq & (first == 0) & (ks[0] > KMIN)                    # true k* may be below the lowest measured k
    k = np.minimum(k, KSENT)
    lo = np.where(cens_dn, KMIN, k)
    hi = np.where(cens_up, KSENT, k)
    return dict(k=k, lo=lo, hi=hi, cens_up=cens_up, cens_dn=cens_dn)


def g_of(det_side, dith_side):
    """det_side, dith_side: lists of kstar() dicts. Returns G (boundary convention) and bounds."""
    dk = np.minimum.reduce([x["k"] for x in det_side])
    dlo = np.minimum.reduce([x["lo"] for x in det_side])
    dhi = np.minimum.reduce([x["hi"] for x in det_side])
    sk = np.minimum.reduce([x["k"] for x in dith_side])
    slo = np.minimum.reduce([x["lo"] for x in dith_side])
    shi = np.minimum.reduce([x["hi"] for x in dith_side])
    return dict(G=dk - sk, lo=dlo - shi, hi=dhi - slo, det=dk, dith=sk)


def dist(x):
    v, c = np.unique(np.asarray(x), return_counts=True)
    return {int(a): float(b / len(x)) for a, b in zip(v, c)}


# ----------------------------------------------------------------------------------------- TD-MPC2 5M
def load_td():
    """E[(task, cell, ptag)] = {(ckpt, ep): (ret, success)}"""
    E = {}
    with open(TD_EP) as f:
        for r in csv.DictReader(f):
            E.setdefault((r["task"], r["cell"], r["ptag"]), {})[(int(r["ckpt"]), int(r["ep"]))] = \
                (float(r["ret"]), float(r["success"]))
    return E


def parse_td_cell(cell):
    head, v = cell.split("_", 1)
    return head[:2], int(head[3:]), v


def td_struct(E, ptag="base", roundings=TD_ROUND):
    """Per task: pairs of the none, y0, stage-1 mask, and cells[(rk, v)][k] = (mask, yv)."""
    S = {}
    for t in MW8 + DMC4:
        mw = is_mw(t)
        j = 1 if mw else 0
        none = E[(t, "none", ptag)]
        pairs = sorted(none)
        pos = {p: i for i, p in enumerate(pairs)}
        y0 = np.array([none[p][j] for p in pairs])
        cells = {}
        for (tt, cell, pt), d in E.items():
            if tt != t or pt != ptag or cell == "none":
                continue
            rk, k, v = parse_td_cell(cell)
            if v not in roundings:
                continue
            m = np.zeros(len(pairs), bool)
            yv = np.zeros(len(pairs))
            for p, val in d.items():
                if p in pos:
                    m[pos[p]] = True
                    yv[pos[p]] = val[j]
            cells.setdefault((rk, v), {})[k] = (m, yv)
        S[t] = dict(mw=mw, pairs=pairs, y0=y0, stage1=np.array([ep < 15 for _, ep in pairs]), cells=cells)
    return S


# ----------------------------------------------------------------------------------------- DINO-WM (eval8 pods)
DW_PAT = re.compile(r"^(?P<split>eval8|held)_(?:(?P<rng>main|pc|rc_k\d)_(?P<var>[a-z]+)_k(?P<k>\d)|none)_"
                    r"(?P<pl>cem|mppi[\d.]+)_e(?P<a>\d+)-(?P<b>\d+)\.json(\.gz)?$")   # = verify_dw.py


def load_dw(pods=DW_PODS):
    """cells[(pod, task, split, rng, var, k, planner)] = {ep: success}; same parsing as coordinator/verify_dw.py."""
    cells, gpus = {}, {}
    for pod in pods:
        for f in sorted(glob.glob(os.path.join(DW_RAW, pod, "jobs", "*", "*.json*"))):
            m = DW_PAT.match(os.path.basename(f))
            if not m:
                continue
            task = f.split(os.sep)[-2]
            d = json.load(gzip.open(f) if f.endswith(".gz") else open(f))
            g = m.groupdict()
            key = (pod, task, g["split"], g["rng"] or "-", g["var"] or "none", int(g["k"] or 0), g["pl"])
            gpus.setdefault(pod, set()).add(d.get("meta", {}).get("gpu"))
            for r in d["records"]:
                cells.setdefault(key, {})[int(r["ep"])] = float(bool(r["success"]))
    return cells, {p: sorted(x for x in v if x) for p, v in gpus.items()}


def dw_none(cells, pod, task, pl="cem", split="eval8"):
    return cells.get((pod, task, split, "-", "none", 0, pl), {})


def dw_components():
    """PushT eval8 overlap components (D1 auxiliary cluster bootstrap); identical in dw.1, dw.2, dw.5."""
    comps = []
    for pod in ("dw.1", "dw.2", "dw.5"):
        d = json.load(open(os.path.join(DW_RAW, pod, "splits", "eval8_pusht.json")))
        comps.append(tuple(d["component_of_pair"]))
    assert len(set(comps)) == 1, "PushT eval8 split differs between pods"
    return list(comps[0])


def dw_struct(cells):
    """Per DINO-WM task: eval8 pairs 0..99, cells[(rk, v)][k] = (mask, y0, yv, pod) for the E12 roundings."""
    S = {}
    for t in DW_TASKS:
        eps = sorted(set().union(*[set(dw_none(cells, p, t)) for p in DW_PODS if dw_none(cells, p, t)]))
        pos = {e: i for i, e in enumerate(eps)}
        cl = {}
        for (pod, task, split, rng, var, k, pl), c in cells.items():
            if task != t or split != "eval8" or pl != "cem" or var == "none":
                continue
            if rng == "main":
                rk = "PT"
            elif rng == "pc":
                rk = "PC"
            elif rng.startswith("rc_k"):
                rk = "RC"
                assert int(rng[4:]) == k, (rng, k)
            else:
                continue
            ok_vars = ("det", "shiter") if rk == "RC" else ("det", "detbc", "shiter")
            if var not in ok_vars:
                continue
            nn = dw_none(cells, pod, t)
            m = np.zeros(len(eps), bool)
            y0 = np.zeros(len(eps))
            yv = np.zeros(len(eps))
            for e, s in c.items():
                if e in nn and e in pos:
                    m[pos[e]] = True
                    y0[pos[e]] = nn[e]
                    yv[pos[e]] = s
            assert k not in cl.get((rk, var), {}), ("duplicate cell", t, rk, var, k)
            cl.setdefault((rk, var), {})[k] = (m, y0, yv, pod)
        S[t] = dict(eps=eps, stage1=np.array([e < 50 for e in eps]), cells=cl)
    return S


# ----------------------------------------------------------------------------------------- curves -> k*, G
def td_kstars(St, W, tau=TAU, ranges=("PT", "PC")):
    """St = td_struct()[task]; returns ks[(rk, v)] = kstar dict, plus D matrices."""
    out, Ds = {}, {}
    for rk in ranges:
        for v in TD_ROUND:
            curve = St["cells"].get((rk, v), {})
            if not curve:
                continue
            ks = sorted(curve)
            D = np.stack([d_w(W, curve[k][0], St["y0"], curve[k][1], St["mw"]) for k in ks], 1)
            out[(rk, v)] = kstar(ks, D, tau)
            out[(rk, v)]["ks"] = ks
            Ds[(rk, v)] = D
    return out, Ds


def td_G(ks, rk, basis="best"):
    det = [ks[(rk, "det")], ks[(rk, "detbc")]]
    if basis == "best":
        dith = [ks[(rk, "shiter")], ks[(rk, "detshift")]]
    elif basis == "shiter":
        dith = [ks[(rk, "shiter")]]
    elif basis == "detshift":
        dith = [ks[(rk, "detshift")]]
    else:
        raise ValueError(basis)
    return g_of(det, dith)


def dw_kstars(St, W, tau=TAU):
    out, Ds = {}, {}
    for (rk, v), curve in St["cells"].items():
        ks = sorted(curve)
        D = np.stack([d_w(W, curve[k][0], curve[k][1], curve[k][2], True) for k in ks], 1)
        out[(rk, v)] = kstar(ks, D, tau)
        out[(rk, v)]["ks"] = ks
        Ds[(rk, v)] = D
    return out, Ds


def dw_G(ks, rk):
    det = [ks[(rk, "det")]] if rk == "RC" else [ks[(rk, "det")], ks[(rk, "detbc")]]
    return g_of(det, [ks[(rk, "shiter")]])


def td_weights(S, rs, scheme="strat"):
    """One weight matrix per task, drawn in the order MW8 + DMC4 from the given RandomState."""
    W = {}
    for t in MW8 + DMC4:
        St = S[t]
        n = len(St["pairs"])
        if St["mw"] or scheme == "simple":
            W[t] = w_simple(rs, n)
        else:
            W[t] = w_strat(rs, [np.where(St["stage1"])[0], np.where(~St["stage1"])[0]], n)
    return W


def dw_weights(S, rs, scheme="strat", comp=None):
    W = {}
    for t in DW_TASKS:
        St = S[t]
        n = len(St["eps"])
        if t == "pusht" and scheme == "strat":
            W[t] = w_strat(rs, [np.where(St["stage1"])[0], np.where(~St["stage1"])[0]], n)
        elif t == "pusht" and scheme == "cluster":
            W[t] = w_cluster(rs, [comp[e] for e in St["eps"]])
        else:
            W[t] = w_simple(rs, n)
    return W


# ----------------------------------------------------------------------------------------- E1 DINO-WM and 317M
def load_e1():
    """R[(task, var, k, ep)] = success; same keys as main/code/dinowm/analyze.py load()."""
    R, gpus = {}, set()
    for sub in ("E1_dinowm_pusht", "E1_dinowm_wp"):
        for f in sorted(glob.glob(os.path.join(E1_DIR, sub, "*", "*.json"))):
            d = json.load(open(f))
            gpus.add(d["meta"].get("gpu"))
            for r in d["records"]:
                R[(r["task"], r["var"], int(r["k"]), int(r["ep"]))] = float(bool(r["success"]))
    return R, sorted(g for g in gpus if g)


def load_mt():
    """eval episodes of E14: M[(task, variant)] = {seed: success}"""
    M = {}
    with open(MT_EP) as f:
        for r in csv.DictReader(f):
            if r["stage"] != "eval":
                continue
            M.setdefault((r["task"], r["variant"]), {})[int(r["seed"])] = float(r["success"])
    return M


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def dump_json(path, obj):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (set, tuple)):
            return list(o)
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=conv)
