"""td group analysis (tmlr_ext/DESIGN_tmlr.md v1.0, TD-MPC2 5M): E11, E12, E13, E15, E16 (TD half).
Written before the main run and dry-run on smoke and synthetic data only (make_fake.py). Runs on a pod.

Data: <res>/jobs/episodes/ev_*.json.gz (closed-loop episodes, eval seeds 100000-100029, checkpoints 1-3),
      <res>/jobs/e13/e13_*.json.gz (held seeds 101000-101029), <res>/jobs/prep/*.json.gz, <res>/jobs/timing/*.json.gz,
      <res>/driver/decisions.jsonl; --res may list several result folders (comma-separated; one per pod).
Statistics (DESIGN_tmlr 0.3 = DESIGN_main v4 6.1): D = none - v success-rate difference (MW) or relative return drop
  1 - mean R_v / mean R_none (DMC, recomputed per resample); pairs (ckpt, episode seed) present for none and every
  variant of the comparison; paired bootstrap 10,000 resamples, numpy RandomState(0) index matrix over the sorted pairs,
  shared by every statistic of a comparison; percentile 95% CI; one-sided p = (#{D* <= 0} + 1) / (B + 1); Holm within
  the unit at alpha 0.05 and D_det > 0 ("damaged"); red = (D_det - D_v) / D_det (point; CI from resamples with
  D_det* > 0); harmful = CI lower bound of D_v - D_det > 0.05; every threshold comparison with tolerance 1e-9.
E11 (1.4): units MW (8 cells), DMC (4 cells). Per damaged cell: fixes(v) = red_v >= 0.5; class bias (det+BC fixes) /
  persistence (det+BC does not fix and the CI lower bound of D_pers = D(det+shift-fixed) - D(det+shift) > 0) /
  resolution (det+BC does not fix, det+shift-fixed fixes, CI of D_pers contains 0) / other; the unit reading is the
  class of more than half of the damaged cells, otherwise "mixed" (no damaged cell: not evaluable).
  B1: MW reading = persistence. B3 (TD units): det+noise fixes at most half of the damaged cells of the unit.
  B4: >= 75% of the MW damaged cells have red_sh-iter >= 0.5, and no harmful sh-iter cell among the 8 MW cells (E1
  H3a convention; the damaged-cells-only reading is reported too).
  Descriptive: D_bias = D(det) - D(det+BC), D_pers, D(sh-ep) - D(sh-iter), D(sh-iter+b) - D(sh-iter),
  D(det+noise) - D(det), D(det+shift) - D(sh-iter) (paired CIs); drift ratio variant / det on mw-push, mw-pick-place,
  cheetah-run (ratio per resample, resamples with drift_det* >= 0 dropped, as E2); the DMC reading.
E12 (2.3): per (task, range kind, rounding) D at every measured k (all pairs of the cell; DMC cells 45 or 90 pairs,
  CIs not corrected for the interim look); k* = smallest measured k with D <= tau = 0.10 such that every larger measured
  k also has D <= tau; no such k -> k* = 9 ("> 8"). G = min(k*(det), k*(det+BC)) - min(k*(sh-iter), k*(det+shift)).
  Fr1: PT, >= 75% of the 8 MW tasks with G >= 1. Fr2: PC, >= 50% of the 8 MW tasks with G >= 1. The application of
  the extension rule is re-checked from the data (boundary D vs tau).
E13 (3.3): cell values = mean over the held episodes of the per-episode means (8 states each), first and last set.
  Dg1: in every E11 damaged cell, Spearman over the 8 variants of D with P1 = mean rho_p x mean (sigma_e / sigma_J)
  (last-iteration sets) and with P2 = latent 1-step error err1h (first-iteration sets, E1 definition); diff =
  rho(P1, D) - rho(P2, D); per unit (MW, DMC) the mean diff over the damaged cells >= 0.3 -> holds.
E15 (5): DMC, iterations 6 (E11 cells), 12 (90 pairs), 24 (60 pairs); D against none of the same iteration count.
  best(n) = the one of sh-iter, det+shift with the larger red at n (red undefined when D_det <= 0). It1: at 12,
  red_best >= 0.5 in >= 3 of 4 tasks and no task where the best one is harmful. It2: red_best(6) < red_best(12) <
  red_best(24) strictly in >= 3 tasks. Descriptive: none return per n, decision time (timing jobs), return-time table.
E16 (6, TD half, descriptive): CEM-like planner cells vs their own none; qualification none_cem >= 0.6 none_default;
  D(sh-ep) - D(sh-iter) with CI for the default (E11) and the CEM-like planner.
Decision D2 (tmlr_ext/LOG.md, DESIGN_tmlr 11; added before any det+BC result with a converged b existed): the bias
  vectors come from the staged rule (alpha = 1, 1/2, 1/4); a (task, ckpt, range kind, k) whose b reached no stage is
  flagged bc_unconverged. A cell (task, range kind, k) is flagged if any checkpoint's b is. B1 and E12's k* / G (Fr1,
  Fr2) are reported (a) on all cells and (b) with the flagged cells' det+BC and sh-iter+b treated as missing (E11: a
  damaged cell without det+BC cannot be classified and is left out of the majority; E12: the det+BC cell at that k is
  not used); when (a) and (b) differ the verdict is "pending (depends on unconverged cells)".
Usage: python analyze_td.py --res <dir[,dir]> --outdir <dir> [--tag <label>]
"""
import argparse
import csv
import glob
import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plan_td as P  # noqa: E402

NB = 10000
ALPHA = 0.05
TOL = 1e-9
TAU = P.TAU
KSENT = P.KMAX + 1
MW8, DMC4, K1 = P.MW8, P.DMC4, P.K1
E11V = P.E11_VARS
E12R = P.E12_ROUND
OEI_KEYS = ("oei", "drift", "sel_last", "oei_indep", "sel_indep", "e_fresh", "f_sd_v0")


def is_mw(t):
    return t.startswith("mw-")


def boot_idx(n, seed=0):
    return np.random.RandomState(seed).randint(0, n, size=(NB, n))


def ci95(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def p_le0(v):
    return float((np.sum(np.asarray(v) <= 0) + 1) / (NB + 1))


def holm(pvals, alpha):
    order = np.argsort(pvals)
    m = len(pvals)
    rej = [False] * m
    for r, i in enumerate(order):
        if pvals[i] <= alpha / (m - r):
            rej[i] = True
        else:
            break
    return rej


def spearman(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).correlation)


def mcnemar(a, b):
    from scipy.stats import binomtest
    n10, n01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    return n10, n01, (1.0 if n10 + n01 == 0 else float(binomtest(n10, n10 + n01, 0.5).pvalue))


def drop(m0, mv, mw):
    return (m0 - mv) if mw else (m0 - mv) / m0


# --------------------------------------------------------------------------------------------------- loading
def load(res_dirs):
    E = defaultdict(dict)       # (task, ckpt, cell, ptag) -> {ep: record}
    H = {}                      # (task, ckpt, ep) -> e13 episode record
    meta = {"gpus": defaultdict(int), "hosts": defaultdict(set), "files": 0, "seed_mismatch": 0, "timing": {},
            "prep": {}, "decisions": [], "e13_files": 0, "res_dirs": res_dirs}
    for res in res_dirs:
        for f in sorted(glob.glob(f"{res}/jobs/episodes/ev_*.json.gz")):
            d = json.load(gzip.open(f, "rt"))
            s = d["spec"]
            meta["files"] += 1
            meta["gpus"][d["env"]["gpu"]] += 1
            meta["hosts"][(s["task"], int(s["ckpt"]))].add(d["env"]["host"])
            cell = P.cell_name(s["variant"], s.get("rk", "PT"), s["k"])
            key = (s["task"], int(s["ckpt"]), cell, s.get("ptag", "base"))
            for e in d["episodes"]:
                if int(e["seed"]) != int(s["seed0"]) + int(e["ep"]) or int(s["seed0"]) != P.SEED0["eval"]:
                    meta["seed_mismatch"] += 1
                om = e.get("oei_mean") or {}
                E[key][int(e["ep"])] = {"ret": float(e["ret"]), "success": float(e["success"]),
                                        "seed": int(e["seed"]), "secs": float(e.get("secs_loop", np.nan)),
                                        "steps": int(e.get("steps", 0)), "gpu": d["env"]["gpu"],
                                        "oei": {kk: om[kk] for kk in OEI_KEYS if om.get(kk) is not None}}
        for f in sorted(glob.glob(f"{res}/jobs/e13/e13_*.json.gz")):
            d = json.load(gzip.open(f, "rt"))
            s = d["spec"]
            meta["e13_files"] += 1
            meta["gpus"][d["env"]["gpu"]] += 1
            meta["hosts"][(s["task"], int(s["ckpt"]))].add(d["env"]["host"])
            for e in d["episodes"]:
                H[(s["task"], int(s["ckpt"]), int(e["ep"]))] = e
        for f in sorted(glob.glob(f"{res}/jobs/timing/timing_*.json.gz")):
            d = json.load(gzip.open(f, "rt"))
            meta["timing"][d["spec"]["task"]] = d["timing"]
        for f in sorted(glob.glob(f"{res}/jobs/prep/prep_*.json.gz")):
            d = json.load(gzip.open(f, "rt"))
            s = d["spec"]
            meta["prep"][f"{s['task']}|s{s['ckpt']}"] = {
                "range_hi_minus_e0_rel_width": {n: v["hi_minus_e0_rel_width"] for n, v in d["ranges"].items()},
                "range_lo_minus_e0": {n: v["lo_minus_e0"] for n, v in d["ranges"].items()},
                "pc_width_mean_rel_pt": {n: v["pc_width_mean_rel_pt"] for n, v in d["ranges"].items()}}
        for f in sorted(glob.glob(f"{res}/jobs/bias/bias_*.json.gz")):
            d = json.load(gzip.open(f, "rt"))
            s = d["spec"]
            meta.setdefault("bias", {})[f"{s['task']}|s{s['ckpt']}"] = {
                "rule": d.get("rule"), "worst_resid": d["worst_resid"],
                "unconverged": d.get("unconverged", []), "cells_by_stage": d.get("cells_by_stage"),
                "max_updates": d.get("max_updates"),
                "resid_by_cell": {f"{rk}k{k}": x["resid_mean_abs_rel_delta_max_layer"]
                                  for rk, byk in d["bias"].items() for k, x in byk.items()},
                "alpha_by_cell": {f"{rk}k{k}": (None if x.get("bc_unconverged") else x.get("alpha"))
                                  for rk, byk in d["bias"].items() for k, x in byk.items()},
                "n_updates_by_cell": {f"{rk}k{k}": x.get("n_updates") for rk, byk in d["bias"].items()
                                      for k, x in byk.items()}}
            for rk, byk in d["bias"].items():
                for k, x in byk.items():
                    if x.get("bc_unconverged"):
                        meta.setdefault("bc_unconverged", []).append([s["task"], int(s["ckpt"]), rk, int(k)])
        dp = f"{res}/driver/decisions.jsonl"
        if os.path.exists(dp):
            meta["decisions"] += [json.loads(l) for l in open(dp) if l.strip()]
    meta["gpus"] = dict(meta["gpus"])
    meta["hosts_per_task_ckpt"] = {f"{t}|s{c}": sorted(h) for (t, c), h in meta["hosts"].items()}
    meta["pairing_one_host_per_task_ckpt"] = all(len(h) == 1 for h in meta["hosts"].values())
    del meta["hosts"]
    return E, H, meta


class Comp:
    """One paired comparison: none(ptag) and the given cells of a task on their common pairs."""

    def __init__(self, E, task, ptag, cells, eps=None):
        self.task, self.mw = task, is_mw(task)
        keys = None
        for cell in ["none"] + list(cells):
            have = set()
            for c in P.CKPTS:
                for ep in E.get((task, c, cell, ptag), {}):
                    if eps is None or ep in eps:
                        have.add((c, ep))
            keys = have if keys is None else keys & have
        self.pairs = sorted(keys or [])
        self.n = len(self.pairs)
        self.ok = self.n > 0
        if not self.ok:
            return
        self.idx = boot_idx(self.n)
        m = "success" if self.mw else "ret"
        self.Y = {cell: np.array([E[(task, c, cell, ptag)][ep][m] for c, ep in self.pairs], float)
                  for cell in ["none"] + list(cells)}
        self.Yb = {cell: y[self.idx].mean(1) for cell, y in self.Y.items()}
        self.raw = {cell: {c_ep: E[(task, c_ep[0], cell, ptag)][c_ep[1]] for c_ep in self.pairs}
                    for cell in ["none"] + list(cells)}

    def D(self, cell):
        return drop(float(self.Y["none"].mean()), float(self.Y[cell].mean()), self.mw), \
            drop(self.Yb["none"], self.Yb[cell], self.mw)

    def field(self, cell, fld):
        x = np.array([self.raw[cell][pr]["oei"].get(fld, np.nan) for pr in self.pairs], float)
        return x if np.isfinite(x).all() else None


def red_of(Dd, Ddb, Dv, Dvb):
    red = float((Dd - Dv) / Dd) if Dd > 0 else None
    with np.errstate(divide="ignore", invalid="ignore"):
        rb = np.where(Ddb > 0, (Ddb - Dvb) / np.where(Ddb > 0, Ddb, 1.0), np.nan)
    return red, ci95(rb)


def ge(x, thr):
    return x is not None and np.isfinite(x) and x >= thr - TOL


# --------------------------------------------------------------------------------------------------- E11
def e11(E, missing=frozenset()):
    """missing: tasks whose det+BC and sh-iter+b are treated as missing (D2 reading (b))."""
    cells = {}
    for t in MW8 + DMC4:
        vt = [v for v in E11V if not (t in missing and v in ("detbc", "shiterb"))]
        names = [P.cell_name(v, "PT", K1[t]) for v in vt]
        C = Comp(E, t, "base", names)
        if not C.ok:
            continue
        r = {"task": t, "unit": "mw" if is_mw(t) else "dmc", "k": K1[t], "n": C.n,
             "none_mean": float(C.Y["none"].mean()), "variants": {}, "missing_variants": [v for v in E11V if v not in vt]}
        Dd, Ddb = C.D(names[0])
        for v, nm in zip(vt, names):
            Dv, Dvb = C.D(nm)
            row = {"mean": float(C.Y[nm].mean()), "D": float(Dv), "D_ci": ci95(Dvb), "p": p_le0(Dvb)}
            if v != "det":
                row["red_all"], row["red_ci_all"] = red_of(Dd, Ddb, Dv, Dvb)
                row["Dv_minus_Ddet"], row["Dv_minus_Ddet_ci"] = float(Dv - Dd), ci95(Dvb - Ddb)
                row["harmful"] = bool(row["Dv_minus_Ddet_ci"][0] > 0.05 + TOL)
            if C.mw:
                S0, Sv = C.Y["none"].astype(bool), C.Y[nm].astype(bool)
                row["mcnemar_vs_none"] = dict(zip(("none_only", "var_only", "p"), mcnemar(S0, Sv)))
            r["variants"][v] = row
        con = {}
        for lab, a, b in (("D_bias=det-detbc", "det", "detbc"), ("D_pers=detshiftfix-detshift", "detshiftfix", "detshift"),
                          ("shep-shiter", "shep", "shiter"), ("shiterb-shiter", "shiterb", "shiter"),
                          ("detnoise-det", "detnoise", "det"), ("detshift-shiter", "detshift", "shiter"),
                          ("det-detshift", "det", "detshift")):     # K2 input (E17, DESIGN_tmlr 7), descriptive
            if a not in vt or b not in vt:
                continue
            Da, Dab = C.D(P.cell_name(a, "PT", K1[t]))
            Db_, Dbb = C.D(P.cell_name(b, "PT", K1[t]))
            con[lab] = {"D": float(Da - Db_), "ci": ci95(Dab - Dbb)}
        r["contrasts"] = con
        # mechanism tasks: drift and the other PlanProbe fields (cell means over the pairs) and drift ratios to det
        if t in P.MECH:
            mech = {}
            xd = C.field(names[0], "drift")
            for v, nm in zip(vt, names):
                row = {}
                for fld in ("oei", "drift", "sel_last", "oei_indep", "sel_indep"):
                    x = C.field(nm, fld)
                    if x is not None:
                        row[fld] = float(x.mean())
                        row[fld + "_ci"] = ci95(x[C.idx].mean(1))
                x = C.field(nm, "drift")
                if x is not None and xd is not None:
                    nb, db = x[C.idx].mean(1), xd[C.idx].mean(1)
                    keep = db < 0
                    row["drift_ratio_to_det"] = float(x.mean() / xd.mean()) if xd.mean() != 0 else None
                    row["drift_ratio_ci"] = ci95(nb[keep] / db[keep]) if keep.any() else [float("nan")] * 2
                    row["drift_ratio_resamples_kept"] = int(keep.sum())
                mech[v] = row
            xn = C.field("none", "f_sd_v0")
            r["mech"] = mech
            r["none_f_sd_v0"] = float(xn.mean()) if xn is not None else None
        cells[t] = r
    # Holm within the unit (alpha 0.05, one-sided), damaged = rejected and D_det > 0
    units = {}
    for u, ts in (("mw", MW8), ("dmc", DMC4)):
        cs = [cells[t] for t in ts if t in cells]
        ps = [c["variants"]["det"]["p"] for c in cs]
        rej = holm(ps, ALPHA) if ps else []
        order = np.argsort(ps) if ps else []
        for rnk, i in enumerate(order):
            cs[i]["holm_rank_alpha"] = [int(rnk + 1), ALPHA / (len(ps) - rnk)]
        for c, rj in zip(cs, rej):
            c["damaged"] = bool(rj and c["variants"]["det"]["D"] > 0)
            for v, row in c["variants"].items():
                if v != "det":
                    row["red"] = row["red_all"] if c["damaged"] else None
                    row["red_ci"] = row["red_ci_all"] if c["damaged"] else None
            c["class"] = classify(c) if c["damaged"] else None
        units[u] = unit_reading(cs)
    J = {}
    mw, dmc = units["mw"], units["dmc"]
    J["B1"] = {"reading_mw": mw["reading"], "holds": (mw["reading"] == "persistence") if mw["evaluable"] else None,
               "evaluable": mw["evaluable"]}
    J["B3_td"] = {}
    for u in ("mw", "dmc"):
        x = units[u]
        J["B3_td"][u] = {"n_damaged": x["n_damaged"], "n_detnoise_fixes": x["n_fix"]["detnoise"],
                         "holds": (x["n_fix"]["detnoise"] <= x["n_damaged"] / 2 + TOL) if x["evaluable"] else None,
                         "evaluable": x["evaluable"]}
    cs = [cells[t] for t in MW8 if t in cells]
    dmg = [c for c in cs if c.get("damaged")]
    ok = [c for c in dmg if ge(c["variants"]["shiter"]["red"], 0.5)]
    harm_all = [c["task"] for c in cs if c["variants"]["shiter"]["harmful"]]
    harm_dmg = [c["task"] for c in dmg if c["variants"]["shiter"]["harmful"]]
    share = len(ok) / len(dmg) if dmg else None
    J["B4"] = {"n_cells": len(cs), "n_damaged": len(dmg), "n_red_ge_half": len(ok), "share": share,
               "cells_red_lt_half": [[c["task"], c["variants"]["shiter"]["red"]] for c in dmg if c not in ok],
               "harmful_all_cells": harm_all, "harmful_damaged_cells": harm_dmg,
               "holds": (bool(share >= 0.75 - TOL and not harm_all) if dmg else None),
               "holds_reading_damaged_only": (bool(share >= 0.75 - TOL and not harm_dmg) if dmg else None),
               "evaluable": bool(dmg)}
    J["DMC_reading"] = {"reading": dmc["reading"], "evaluable": dmc["evaluable"], "counts": dmc["counts"]}
    return cells, units, J


def classify(c):
    V = c["variants"]
    fix = {v: ge(V[v].get("red"), 0.5) for v in E11V if v != "det" and v in V}
    lo, hi = c["contrasts"]["D_pers=detshiftfix-detshift"]["ci"]
    c["fixes"] = fix
    if "detbc" not in fix:             # D2 reading (b): det+BC missing -> the cell cannot be classified
        return "undetermined"
    if fix["detbc"]:
        return "bias"
    if lo > 0 + TOL:
        return "persistence"
    if fix["detshiftfix"] and lo <= 0 + TOL and hi >= 0 - TOL:
        return "resolution"
    return "other"


def unit_reading(cs):
    dmg = [c for c in cs if c.get("damaged")]
    counts = defaultdict(int)
    for c in dmg:
        counts[c["class"]] += 1
    n_fix = {v: sum(1 for c in dmg if c["fixes"].get(v)) for v in E11V if v != "det"}
    cls = [c for c in dmg if c["class"] != "undetermined"]
    reading = None
    if cls:
        reading = "mixed"
        for cl in ("bias", "persistence", "resolution"):
            if counts[cl] > len(cls) / 2 + TOL:
                reading = cl
    return {"n_cells": len(cs), "n_damaged": len(dmg), "n_classified": len(cls), "damaged": [c["task"] for c in dmg],
            "counts": dict(counts), "classes": {c["task"]: c["class"] for c in dmg}, "n_fix": n_fix,
            "reading": reading, "evaluable": bool(cls)}


# --------------------------------------------------------------------------------------------------- E12
def e12(E, decisions, missing=frozenset()):
    """missing: {(task, range kind, k)} whose det+BC cell is treated as missing (D2 reading (b))."""
    out, J = {}, {}
    for t in MW8 + DMC4:
        tk = {"cells": {}, "kstar": {}, "G": {}, "rule_check": []}
        for rk in ("PT", "PC"):
            for v in E12R:
                ks = sorted({int(cell.split("_")[0][3:]) for (tt, c, cell, pt) in E
                             if tt == t and pt == "base" and cell != "none" and cell.startswith(rk)
                             and cell.split("_", 1)[1] == v})
                if v == "detbc":
                    ks = [k for k in ks if (t, rk, k) not in missing]
                curve = {}
                for k in ks:
                    C = Comp(E, t, "base", [P.cell_name(v, rk, k)])
                    if not C.ok:
                        continue
                    D, Db = C.D(P.cell_name(v, rk, k))
                    curve[k] = {"D": float(D), "D_ci": ci95(Db), "n": C.n,
                                "mean": float(C.Y[P.cell_name(v, rk, k)].mean()),
                                "two_stage_interim_uncorrected": (not is_mw(t)) and not (rk == "PT" and k == K1[t])}
                if not curve:
                    continue
                kss = sorted(curve)
                kstar = KSENT
                for k in kss:
                    if all(curve[kk]["D"] <= TAU + TOL for kk in kss if kk >= k):
                        kstar = k
                        break
                lo, hi = kss[0], kss[-1]
                if curve[lo]["D"] <= TAU + TOL and lo > P.KMIN:
                    tk["rule_check"].append(f"{rk} {v}: lowest k {lo} has D <= tau but no lower k measured")
                if curve[hi]["D"] > TAU + TOL and hi < P.KMAX:
                    tk["rule_check"].append(f"{rk} {v}: highest k {hi} has D > tau but no higher k measured")
                full_grid = all(k in curve for k in P.E12_GRID[is_mw(t)][rk])
                tk["cells"][f"{rk}|{v}"] = curve
                tk["kstar"][f"{rk}|{v}"] = {"kstar": kstar, "reached": kstar <= P.KMAX, "ks": kss,
                                            "full_initial_grid": full_grid}
        for rk in ("PT", "PC"):
            ksd = [tk["kstar"].get(f"{rk}|{v}", {}).get("kstar") for v in ("det", "detbc")]
            ksh = [tk["kstar"].get(f"{rk}|{v}", {}).get("kstar") for v in ("shiter", "detshift")]
            if all(x is not None for x in ksd + ksh):
                tk["G"][rk] = int(min(ksd) - min(ksh))
        allk = [(x["kstar"], key) for key, x in tk["kstar"].items()]
        if allk:
            best = min(k for k, _ in allk)
            tk["best"] = {"kstar": best, "configs": [key for k, key in allk if k == best]}
        if tk["cells"]:
            out[t] = tk
    for lab, rk, share_thr in (("Fr1", "PT", 0.75), ("Fr2", "PC", 0.50)):
        gs = {t: out[t]["G"].get(rk) for t in MW8 if t in out}
        n_ok = sum(1 for g in gs.values() if g is not None and g >= 1)
        n_def = sum(1 for g in gs.values() if g is not None)
        J[lab] = {"range": rk, "G": gs, "n_tasks_G_ge_1": n_ok, "n_tasks_defined": n_def,
                  "share": n_ok / 8, "threshold_share": share_thr,
                  "holds": bool(n_def == 8 and n_ok / 8 >= share_thr - TOL) if n_def == 8 else None,
                  "complete": n_def == 8}
    J["DMC_G"] = {t: out[t]["G"] for t in DMC4 if t in out}
    J["n_adaptive_decisions"] = len(decisions)
    J["decisions_by_rule"] = {r: sum(1 for d in decisions if d.get("rule") == r)
                              for r in ("stage2", "extend_down", "extend_up")}
    return out, J


# --------------------------------------------------------------------------------------------------- E15
def e15(E, timing):
    out = {}
    for t in DMC4:
        it = {}
        for n, ptag in ((6, "base"), (12, "it12"), (24, "it24")):
            names = [P.cell_name(v, "PT", K1[t]) for v in ("det", "shiter", "detshift")]
            C = Comp(E, t, ptag, names)
            if not C.ok:
                continue
            Dd, Ddb = C.D(names[0])
            r = {"n": C.n, "none_mean": float(C.Y["none"].mean()), "D_det": float(Dd), "D_det_ci": ci95(Ddb),
                 "p_det": p_le0(Ddb), "variants": {}}
            for v, nm in zip(("shiter", "detshift"), names[1:]):
                Dv, Dvb = C.D(nm)
                red, rci = red_of(Dd, Ddb, Dv, Dvb)
                r["variants"][v] = {"D": float(Dv), "D_ci": ci95(Dvb), "red": red, "red_ci": rci,
                                    "Dv_minus_Ddet": float(Dv - Dd), "Dv_minus_Ddet_ci": ci95(Dvb - Ddb),
                                    "harmful": bool(ci95(Dvb - Ddb)[0] > 0.05 + TOL)}
            reds = {v: r["variants"][v]["red"] for v in r["variants"]}
            cand = [v for v, x in reds.items() if x is not None]
            best = max(cand, key=lambda v: reds[v]) if cand else None
            r["best"] = best
            r["red_best"] = reds[best] if best else None
            r["best_harmful"] = r["variants"][best]["harmful"] if best else None
            r["secs_per_episode_contended"] = float(np.nanmean([C.raw["none"][pr]["secs"] for pr in C.pairs]))
            it[n] = r
        if it:
            out[t] = {"iterations": it, "timing_ms": timing.get(t)}
    J = {}
    rows = {t: out[t]["iterations"].get(12) for t in DMC4 if t in out}
    ok = [t for t, r in rows.items() if r and ge(r["red_best"], 0.5)]
    harm = [t for t, r in rows.items() if r and r["best_harmful"]]
    complete = len([r for r in rows.values() if r]) == 4
    J["It1"] = {"n_tasks_red_best_ge_half": len(ok), "tasks": ok, "harmful_tasks": harm,
                "red_best_12": {t: (r["red_best"] if r else None) for t, r in rows.items()},
                "holds": bool(len(ok) >= 3 and not harm) if complete else None, "complete": complete}
    mono = {}
    for t in DMC4:
        if t not in out:
            continue
        rb = [out[t]["iterations"].get(n, {}).get("red_best") for n in (6, 12, 24)]
        mono[t] = {"red_best_6_12_24": rb,
                   "monotone": bool(all(x is not None for x in rb) and rb[1] > rb[0] + TOL and rb[2] > rb[1] + TOL)}
    c2 = len(mono) == 4 and all(len([n for n in (6, 12, 24) if n in out[t]["iterations"]]) == 3 for t in mono)
    J["It2"] = {"per_task": mono, "n_monotone": sum(m["monotone"] for m in mono.values()),
                "holds": bool(sum(m["monotone"] for m in mono.values()) >= 3) if c2 else None, "complete": c2}
    return out, J


def return_time(e12o, e15o):
    """E15 descriptive: 'det + one more bit' (E12 PT cell k1 + 1, 6 iterations) vs 'dither at k1 + more iterations'
    (E15 cells): mean return and decision time (timing jobs, none at the iteration count as the time proxy)."""
    rows = {}
    for t in DMC4:
        if t not in e15o:
            continue
        tm = e15o[t].get("timing_ms") or {}
        r = {}
        c = e12o.get(t, {}).get("cells", {}).get("PT|det", {}).get(K1[t] + 1)
        if c:
            r["det_k1+1_it6"] = {"mean_return": c["mean"], "D": c["D"], "ms": (tm.get("it6|det") or {}).get("ms_median")}
        for n in (6, 12, 24):
            x = e15o[t]["iterations"].get(n)
            if not x:
                continue
            for v in ("shiter", "detshift"):
                r[f"{v}_k1_it{n}"] = {"D": x["variants"][v]["D"],
                                      "ms": (tm.get(f"it{n}|{v}") or {}).get("ms_median")}
            r[f"none_it{n}"] = {"mean_return": x["none_mean"], "ms": (tm.get(f"it{n}|none") or {}).get("ms_median")}
        rows[t] = r
    return rows


# --------------------------------------------------------------------------------------------------- E16
def e16(E, e11c):
    out = {}
    for t in P.E16_TASKS:
        names = [P.cell_name(v, "PT", K1[t]) for v in ("det", "shiter", "shep", "detshift")]
        C = Comp(E, t, "cem", names)
        if not C.ok:
            continue
        r = {"n": C.n, "none_cem": float(C.Y["none"].mean()), "variants": {}}
        for v, nm in zip(("det", "shiter", "shep", "detshift"), names):
            Dv, Dvb = C.D(nm)
            r["variants"][v] = {"D": float(Dv), "D_ci": ci95(Dvb)}
        Dse, Dseb = C.D(names[2])
        Dsi, Dsib = C.D(names[1])
        r["shep_minus_shiter"] = {"D": float(Dse - Dsi), "ci": ci95(Dseb - Dsib)}
        base = e11c.get(t)
        if base:
            r["none_default"] = base["none_mean"]
            r["qualified"] = bool(r["none_cem"] >= 0.6 * base["none_mean"] - TOL)
            r["default_shep_minus_shiter"] = base["contrasts"]["shep-shiter"]
        out[t] = r
    return out


# --------------------------------------------------------------------------------------------------- E13
METRICS = ("err1", "err3", "err5", "err1h", "errr", "rew_med", "rew_mean", "e_mean", "common_frac", "sigma_e",
           "sigma_J", "ratio", "rho_p", "prod", "spear", "elite_overlap")


def e13(H, e11c, units):
    agg = defaultdict(lambda: defaultdict(list))
    held = defaultdict(list)
    for (t, c, ep), e in H.items():
        held[t].append(e["success"] if is_mw(t) else e["ret"])
        for key, sets in e["cells"].items():
            for sn, m in sets.items():
                for mm in METRICS:
                    if m.get(mm) is not None:
                        agg[(t, key, sn)][mm].append(m[mm])
    cells = {}
    for (t, key, sn), d in agg.items():
        cells.setdefault(t, {}).setdefault(key, {})[sn] = {mm: float(np.mean(x)) for mm, x in d.items()}
        cells[t][key][sn]["n_eps"] = len(next(iter(d.values())))
    # Dg1
    dg = {}
    for u, ts in (("mw", MW8), ("dmc", DMC4)):
        per = {}
        for t in ts:
            c = e11c.get(t)
            if not c or not c.get("damaged") or t not in cells:
                continue
            D, p1, p2 = [], [], []
            for v in E11V:
                key = P.cell_name(v, "PT", K1[t])
                ce = cells[t].get(key)
                if not ce or "last" not in ce or "first" not in ce:
                    continue
                D.append(c["variants"][v]["D"])
                p1.append(ce["last"]["rho_p"] * ce["last"]["ratio"])
                p2.append(ce["first"]["err1h"])
            r1, r2 = spearman(p1, D), spearman(p2, D)
            per[t] = {"n_variants": len(D), "rho_P1_D": r1, "rho_P2_D": r2,
                      "diff": (r1 - r2) if np.isfinite(r1) and np.isfinite(r2) else float("nan"),
                      "P1": p1, "P2": p2, "D": D}
        fin = [x["diff"] for x in per.values() if np.isfinite(x["diff"])]
        dg[u] = {"per_cell": per, "n_damaged_cells": len(per), "n_defined": len(fin),
                 "mean_diff": float(np.mean(fin)) if fin else None,
                 "holds": bool(np.mean(fin) >= 0.3 - TOL) if fin else None}
    # E1 cells on the common held states: det vs sh-iter latent 1-step error (descriptive, E1 H1 analogue)
    h1 = {}
    for t in cells:
        for k in (K1[t] - 1, K1[t], K1[t] + 1):
            d_, s_ = cells[t].get(P.cell_name("det", "PT", k)), cells[t].get(P.cell_name("shiter", "PT", k))
            if d_ and s_ and "first" in d_ and "first" in s_:
                h1[f"{t}|k{k}"] = {"err1h_det": d_["first"]["err1h"], "err1h_shiter": s_["first"]["err1h"],
                                   "det_le_shiter": bool(d_["first"]["err1h"] <= s_["first"]["err1h"] + TOL)}
    return cells, dg, h1, {t: {"n": len(x), "mean": float(np.mean(x))} for t, x in held.items()}


# --------------------------------------------------------------------------------------------------- output
def fmt(x):
    if isinstance(x, (list, tuple)):
        return "[" + ";".join(fmt(y) for y in x) + "]"
    if isinstance(x, float):
        return f"{x:.6g}"
    return "" if x is None else str(x)


def pp(x, mw=True):
    return "" if x is None or not np.isfinite(x) else f"{100 * x:+.1f}"


def cit(ci, scale=100, nd=1):
    if not ci or ci[0] is None or not np.isfinite(ci[0]):
        return ""
    return f"[{scale * ci[0]:.{nd}f}, {scale * ci[1]:.{nd}f}]"


def n2(x, nd=2):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def write_outputs(S, E, outdir):
    os.makedirs(outdir, exist_ok=True)
    json.dump(S, open(f"{outdir}/td_summary.json", "w"), indent=1, default=float)
    with open(f"{outdir}/td_episodes.csv", "w") as f:
        f.write("task,ckpt,ep,seed,cell,ptag,ret,success,secs,gpu\n")
        for (t, c, cell, pt), d in sorted(E.items()):
            for ep, r in sorted(d.items()):
                f.write(f"{t},{c},{ep},{r['seed']},{cell},{pt},{r['ret']:.6f},{r['success']:g},{r['secs']:.2f},{r['gpu']}\n")
    with open(f"{outdir}/td_e11_cells.csv", "w") as f:
        cols = ["task", "unit", "k", "n", "none_mean", "damaged", "class", "variant", "mean", "D", "D_ci", "p", "red",
                "red_ci", "red_all", "Dv_minus_Ddet", "Dv_minus_Ddet_ci", "harmful"]
        f.write(",".join(cols) + "\n")
        for t, c in S["E11"]["cells"].items():
            for v, r in c["variants"].items():
                row = dict(r, task=t, unit=c["unit"], k=c["k"], n=c["n"], none_mean=c["none_mean"],
                           damaged=c.get("damaged"), variant=v, **{"class": c.get("class")})
                f.write(",".join(fmt(row.get(col)) for col in cols) + "\n")
    with open(f"{outdir}/td_e11_contrasts.csv", "w") as f:
        f.write("task,unit,k,n,contrast,D,ci_lo,ci_hi\n")
        for t, c in S["E11"]["cells"].items():
            for lab, x in c["contrasts"].items():
                f.write(f"{t},{c['unit']},{c['k']},{c['n']},{lab},{fmt(x['D'])},{fmt(x['ci'][0])},{fmt(x['ci'][1])}\n")
    with open(f"{outdir}/td_k2_e11_det_minus_detshift.csv", "w") as f:
        f.write("task,unit,k,n,D_det,D_detshift,D_det_minus_D_detshift,ci_lo,ci_hi\n")
        for t, x in S["K2_E11_det_minus_detshift"].items():
            f.write(f"{t},{x['unit']},{x['k']},{x['n']},{fmt(x['D_det'])},{fmt(x['D_detshift'])},{fmt(x['D'])},"
                    f"{fmt(x['ci'][0])},{fmt(x['ci'][1])}\n")
    with open(f"{outdir}/td_e12_cells.csv", "w") as f:
        f.write("task,range,rounding,k,n,mean,D,D_ci,kstar,two_stage_interim_uncorrected\n")
        for t, tk in S["E12"]["tasks"].items():
            for key, curve in tk["cells"].items():
                rk, v = key.split("|")
                for k, c in sorted(curve.items(), key=lambda kv: int(kv[0])):
                    f.write(f"{t},{rk},{v},{k},{c['n']},{fmt(c['mean'])},{fmt(c['D'])},{fmt(c['D_ci'])},"
                            f"{tk['kstar'][key]['kstar']},{c['two_stage_interim_uncorrected']}\n")
    with open(f"{outdir}/td_e13_cells.csv", "w") as f:
        f.write("task,cell,set," + ",".join(METRICS) + ",n_eps\n")
        for t, cs in S["E13"]["cells"].items():
            for key, sets in cs.items():
                for sn, m in sets.items():
                    f.write(f"{t},{key},{sn}," + ",".join(fmt(m.get(mm)) for mm in METRICS) + f",{m.get('n_eps')}\n")
    write_md(S, f"{outdir}/td_tables.md")


def write_md(S, path):
    L = ["# td 분석 표 (analyze_td.py 자동 생성)", ""]
    J = S["judgements"]
    L += ["## 판정", "", "| 가설 | 판정 | 핵심 수치 |", "|---|---|---|"]
    L.append(f"| B1 (MW 지속 지배) | {J['B1']['holds']} | MW 읽기 {J['B1']['reading_mw']} |")
    for u in ("mw", "dmc"):
        b = J["B3_td"][u]
        L.append(f"| B3 ({u}) | {b['holds']} | det+noise가 고친 손상 칸 {b['n_detnoise_fixes']}/{b['n_damaged']} |")
    b = J["B4"]
    L.append(f"| B4 | {b['holds']} | red_sh-iter ≥ 0.5 {b['n_red_ge_half']}/{b['n_damaged']}, 해로운 칸(전체) "
             f"{b['harmful_all_cells']}, (손상 칸만 읽기: {b['holds_reading_damaged_only']}) |")
    L.append(f"| DMC 읽기 | {J['DMC_reading']['reading']} | {J['DMC_reading']['counts']} |")
    for lab in ("Fr1", "Fr2"):
        f_ = J[lab]
        L.append(f"| {lab} ({f_['range']}) | {f_['holds']} | G ≥ 1인 MW 과제 {f_['n_tasks_G_ge_1']}/8, G {f_['G']} |")
    for u in ("mw", "dmc"):
        d = J["Dg1"][u]
        L.append(f"| Dg1 ({u}) | {d['holds']} | 평균 차 {n2(d['mean_diff'], 3)} (손상 칸 {d['n_damaged_cells']}) |")
    d2 = J.get("D2_two_readings", {})
    for lab in ("B1", "Fr1", "Fr2"):
        if lab in d2:
            L.append(f"| {lab} (D2: (가) 모든 칸 / (나) 미수렴 칸 결측) | {d2[lab]['verdict']} | (가) {d2[lab]['a_all_cells']}, "
                     f"(나) {d2[lab]['b_flagged_missing']}; 미수렴 칸 {d2.get('flagged_cells')} |")
    L.append(f"| It1 | {J['It1']['holds']} | red_best(12) {J['It1']['red_best_12']}, 해로운 과제 {J['It1']['harmful_tasks']} |")
    L.append(f"| It2 | {J['It2']['holds']} | 단조 증가 과제 {J['It2']['n_monotone']} |")
    L += ["", "## E11 (PT, 1차 k; MW %p, DMC return %; CI 짝 bootstrap 95%)", "",
          "| 과제 | k | n | none | 손상 | 분류 | det D | det+BC red | sh-ep red | det+shift-fixed red | sh-iter+b red | "
          "sh-iter red | det+shift red | det+noise red | Δ_bias [CI] | Δ_pers [CI] |", "|" + "---|" * 16]
    for t, c in S["E11"]["cells"].items():
        V = c["variants"]
        reds = [n2(V[v].get("red_all")) for v in ("detbc", "shep", "detshiftfix", "shiterb", "shiter", "detshift",
                                                  "detnoise")]
        cb, cp = c["contrasts"]["D_bias=det-detbc"], c["contrasts"]["D_pers=detshiftfix-detshift"]
        L.append(f"| {t} | {c['k']} | {c['n']} | {n2(c['none_mean'], 3)} | {c.get('damaged')} | {c.get('class')} | "
                 f"{pp(V['det']['D'])} {cit(V['det']['D_ci'])} | " + " | ".join(reds) +
                 f" | {pp(cb['D'])} {cit(cb['ci'])} | {pp(cp['D'])} {cit(cp['ci'])} |")
    L += ["", "- red는 손상 칸에서만 판정에 쓴다(표의 값은 D_det > 0일 때의 점 추정 참고값).", "",
          "### E11 보조 대비(D 차, %p 또는 %)", "", "| 과제 | sh-ep − sh-iter | sh-iter+b − sh-iter | det+noise − det | "
          "det+shift − sh-iter |", "|---|---|---|---|---|"]
    for t, c in S["E11"]["cells"].items():
        cc = c["contrasts"]
        L.append(f"| {t} | " + " | ".join(f"{pp(cc[k]['D'])} {cit(cc[k]['ci'])}" for k in
                                          ("shep-shiter", "shiterb-shiter", "detnoise-det", "detshift-shiter")) + " |")
    L += ["", "### 기전 과제 drift (가치 단위) 와 det 대비 비", "", "| 과제 | 변형 | drift [CI] | 비 [CI] | sel_last | "
          "oei_indep |", "|---|---|---|---|---|---|"]
    for t, c in S["E11"]["cells"].items():
        for v, r in (c.get("mech") or {}).items():
            L.append(f"| {t} | {v} | {n2(r.get('drift'))} {cit(r.get('drift_ci'), 1, 2)} | "
                     f"{n2(r.get('drift_ratio_to_det'))} {cit(r.get('drift_ratio_ci'), 1, 2)} | {n2(r.get('sel_last'))} | "
                     f"{n2(r.get('oei_indep'))} |")
    L += ["", "## E12 k\\*와 G", "", "| 과제 | PT det / det+BC / sh-iter / det+shift | G(PT) | PC det / det+BC / sh-iter / "
          "det+shift | G(PC) | 최선 | 규칙 점검 |", "|---|---|---|---|---|---|---|"]
    for t, tk in S["E12"]["tasks"].items():
        def ks(rk):
            return " / ".join(str(tk["kstar"].get(f"{rk}|{v}", {}).get("kstar", "")) for v in E12R)
        L.append(f"| {t} | {ks('PT')} | {tk['G'].get('PT', '')} | {ks('PC')} | {tk['G'].get('PC', '')} | "
                 f"{tk.get('best')} | {'; '.join(tk['rule_check'])} |")
    L += ["", "- k\\* = 9는 k = 8까지 D ≤ τ에 닿지 않았다는 뜻이다. DMC 두 단계 칸의 CI는 중간 확인을 보정하지 않은 값이다.", "",
          "### E12 비트–하락 곡선", "", "| 과제 | 범위 | 반올림 | k: D [CI] (n) |", "|---|---|---|---|"]
    for t, tk in S["E12"]["tasks"].items():
        for key, curve in tk["cells"].items():
            rk, v = key.split("|")
            L.append(f"| {t} | {rk} | {v} | " + "; ".join(f"{k}: {pp(c['D'])} {cit(c['D_ci'])} ({c['n']})" for k, c in
                                                         sorted(curve.items(), key=lambda kv: int(kv[0]))) + " |")
    L += ["", "## E15 (DMC, 반복 수)", "", "| 과제 | 반복 | n | none | D_det [CI] | sh-iter D / red | det+shift D / red | "
          "best | 해로움 |", "|---|---|---|---|---|---|---|---|---|"]
    for t, x in S["E15"]["tasks"].items():
        for n, r in sorted(x["iterations"].items(), key=lambda kv: int(kv[0])):
            V = r["variants"]
            L.append(f"| {t} | {n} | {r['n']} | {n2(r['none_mean'], 1)} | {pp(r['D_det'])} {cit(r['D_det_ci'])} | "
                     f"{pp(V['shiter']['D'])} / {n2(V['shiter']['red'])} | {pp(V['detshift']['D'])} / "
                     f"{n2(V['detshift']['red'])} | {r['best']} | {r['best_harmful']} |")
    L += ["", "## E16 (TD-MPC2 CEM식 계획기, 기술)", "", "| 과제 | none CEM / 기본 | 자격 | det D | sh-iter D | sh-ep D | "
          "det+shift D | sh-ep − sh-iter (CEM) [CI] | sh-ep − sh-iter (기본, E11) [CI] |", "|" + "---|" * 9]
    for t, r in S["E16"].items():
        V = r["variants"]
        dd = r.get("default_shep_minus_shiter") or {}
        L.append(f"| {t} | {n2(r['none_cem'], 3)} / {n2(r.get('none_default'), 3)} | {r.get('qualified')} | "
                 f"{pp(V['det']['D'])} | {pp(V['shiter']['D'])} | {pp(V['shep']['D'])} | {pp(V['detshift']['D'])} | "
                 f"{pp(r['shep_minus_shiter']['D'])} {cit(r['shep_minus_shiter']['ci'])} | {pp(dd.get('D'))} "
                 f"{cit(dd.get('ci'))} |")
    L += ["", "## E13 (held 상태, 첫 / 마지막 반복 후보 집합; E11 칸)", "",
          "| 과제 | 칸 | err1h | err3 | err5 | rew_med | σe/σJ (첫/마지막) | ρ_p (첫/마지막) | Spearman (마지막) | "
          "elite 겹침 (마지막) |", "|" + "---|" * 10]
    for t, cs in S["E13"]["cells"].items():
        for key, sets in cs.items():
            f_, l_ = sets.get("first", {}), sets.get("last", {})
            L.append(f"| {t} | {key} | {n2(f_.get('err1h'), 4)} | {n2(f_.get('err3'), 4)} | {n2(f_.get('err5'), 4)} | "
                     f"{n2(f_.get('rew_med'), 4)} | {n2(f_.get('ratio'), 3)} / {n2(l_.get('ratio'), 3)} | "
                     f"{n2(f_.get('rho_p'), 3)} / {n2(l_.get('rho_p'), 3)} | {n2(l_.get('spear'), 3)} | "
                     f"{n2(l_.get('elite_overlap'), 3)} |")
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    E, H, meta = load([r for r in a.res.split(",") if r])
    e11c, units, J11 = e11(E)
    e12o, J12 = e12(E, meta["decisions"])
    # D2: the same verdicts with the bc_unconverged cells' det+BC / sh-iter+b treated as missing
    flagged = {(t, rk, k) for t, c, rk, k in meta.get("bc_unconverged", [])}
    miss11 = frozenset(t for t, rk, k in flagged if rk == "PT" and k == K1[t])
    _, units_b, J11b = e11(E, miss11)
    e12b, J12b = e12(E, meta["decisions"], frozenset(flagged))
    d2 = {"flagged_cells": sorted([list(x) for x in flagged]), "e11_tasks_missing": sorted(miss11)}
    for lab, ja, jb in (("B1", J11["B1"], J11b["B1"]), ("Fr1", J12["Fr1"], J12b["Fr1"]), ("Fr2", J12["Fr2"], J12b["Fr2"])):
        d2[lab] = {"a_all_cells": ja.get("holds"), "b_flagged_missing": jb.get("holds"),
                   "verdict": (ja.get("holds") if ja.get("holds") == jb.get("holds")
                               else "pending (depends on unconverged cells)")}
    d2["B1"]["reading_mw_b"] = J11b["B1"]["reading_mw"]
    d2["DMC_reading_b"] = J11b["DMC_reading"]["reading"]
    d2["Fr1"]["G_b"], d2["Fr2"]["G_b"] = J12b["Fr1"]["G"], J12b["Fr2"]["G"]
    e15o, J15 = e15(E, meta["timing"])
    e16o = e16(E, e11c)
    e13c, dg1, h1, held = e13(H, e11c, units)
    base = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs", "e1_none_baseline.json")))
    bl = {}
    for t in MW8 + DMC4:
        x = [r["success"] if is_mw(t) else r["ret"] for c in P.CKPTS for r in E.get((t, c, "none", "base"), {}).values()]
        if x:
            e1v = base["tasks"][t]["summary_baseline"]
            lim = 0.10 if is_mw(t) else 0.10 * e1v
            bl[t] = {"n": len(x), "none_new": float(np.mean(x)), "none_e1": e1v, "diff": float(np.mean(x)) - e1v,
                     "within_criterion": bool(abs(float(np.mean(x)) - e1v) <= lim + TOL)}
    J = dict(J11)
    J.update(J12)
    J.update(J15)
    J["Dg1"] = dg1
    J["D2_two_readings"] = d2
    # D2 record for the REPORT: cells converged per stage, unconverged cells, max updates, worst final residual
    bs = meta.get("bias", {})
    agg = defaultdict(int)
    for v in bs.values():
        for kk, n in (v.get("cells_by_stage") or {}).items():
            agg[kk] += n
    d2["bias_summary"] = {"n_bias_jobs": len(bs), "cells_by_stage": dict(agg),
                          "n_unconverged": len(meta.get("bc_unconverged", [])),
                          "unconverged": meta.get("bc_unconverged", []),
                          "max_updates": max([v.get("max_updates") or 0 for v in bs.values()], default=None),
                          "worst_final_resid": max([v.get("worst_resid") or 0 for v in bs.values()], default=None)}
    k2 = {t: dict(e11c[t]["contrasts"]["det-detshift"], n=e11c[t]["n"], k=K1[t], unit=e11c[t]["unit"],
                  D_det=e11c[t]["variants"]["det"]["D"], D_detshift=e11c[t]["variants"]["detshift"]["D"])
          for t in ("mw-push", "mw-pick-place", "cheetah-run", "walker-walk") if t in e11c}
    S = {"tag": a.tag, "n_episodes": sum(len(d) for d in E.values()), "n_e13_episodes": len(H), "meta": meta,
         "K2_E11_det_minus_detshift": k2,
         "baseline_check3": bl, "held_none": held, "judgements": J,
         "E11": {"cells": e11c, "units": units}, "E12": {"tasks": e12o}, "E15": {"tasks": e15o,
                                                                                 "return_time": return_time(e12o, e15o)},
         "E16": e16o, "E13": {"cells": e13c, "e1_h1_common_states": h1}}
    write_outputs(S, E, a.outdir)
    brief = {"D2_" + k: v["verdict"] for k, v in d2.items() if isinstance(v, dict) and "verdict" in v}
    brief.update({k: (v.get("holds") if isinstance(v, dict) and "holds" in v else
                 {u: x.get("holds") for u, x in v.items()} if isinstance(v, dict) and k in ("B3_td", "Dg1") else None)
             for k, v in J.items() if k in ("B1", "B3_td", "B4", "Fr1", "Fr2", "Dg1", "It1", "It2")})
    print(json.dumps({"episodes": S["n_episodes"], "e13_episodes": len(H), "judgements": brief,
                      "DMC_reading": J["DMC_reading"]["reading"], "pairing_one_host": meta["pairing_one_host_per_task_ckpt"],
                      "gpus": meta["gpus"], "seed_mismatch": meta["seed_mismatch"]}, default=str))


if __name__ == "__main__":
    main()
