"""E1 analysis for the TD-MPC2 part of paper A, following DESIGN_main v4 section 6 (E1 judgements computed the same way
for both model families) and the conventions of the DINO-WM worker's code/dinowm/analyze_e1.py / analyze.py (written
before any TD-MPC2 E1 outcome was looked at; checked on smoke-split data and a synthetic dry run only).

usage (server, thread-capped):
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python analyze_e1.py --res <out dir with jobs/> \
      --split eval --outdir <dir>
  --split smoke   runs the same code on the smoke split (code-path check; numbers are not interpreted)

Data: <res>/jobs/episodes/<split>_<task>_s<ckpt>_k<k>_<variant>_base_cXX.json.gz (episodes),
      <res>/jobs/prederr/[smoke_]prederr_<task>_s<ckpt>_k<k>.json.gz, <res>/jobs/decision/[smoke_]decision_...json.gz

Rules (DESIGN_main v4 6.1 / 6.2; same as the DINO-WM family):
- cells = (task, k), confirmatory = DESIGN_main 0.2 table (12 tasks x 3 k); pairs = (checkpoint seed, episode seed),
  90 per cell (30 episodes x checkpoints 1-3); every variant and the k-independent none baseline share the pairs.
- D_v: Meta-World = success-rate difference none - v (absolute, pp); DMC = relative return drop
  (mean R_none - mean R_v) / mean R_none, recomputed as a ratio inside every bootstrap resample.
- paired bootstrap: 10,000 resamples of the pairs, RandomState(0) for every statistic (as analyze.boot), percentile
  95% CI; one-sided p for D_det <= 0 = (#{D* <= 0} + 1) / (B + 1) (as analyze_e1.boot_p_le0).
- det drop real: per task Holm-Bonferroni over its 3 confirmatory k cells at alpha = 0.025 (one-sided) and D_det > 0;
  auxiliary: unadjusted 95% CI lower bound > 0, McNemar exact p (Meta-World success only).
- red_v = (D_det - D_v) / D_det in qualifying cells only; CI from the same resamples as a ratio (NaN resamples with
  D_det* <= 0 dropped, as analyze.boot does).
- H1 (family): among qualifying cells, >= 1 cell with det's planning-time 1-step error <= sh-iter's. 1-step error =
  cell mean over episodes of perr_plan.err1h (the R2-8b 'err1': 1-step relative error at every horizon position of the
  first-iteration candidates, dither-seed 3-draw mean for stochastic variants). Dataset-type (perr_traj) auxiliary.
- H3a (units: Meta-World 24 cells, DMC 12 cells; TD-MPC2 overall also reported): share of qualifying cells with
  red_shiter >= 0.5 is >= 75%, and in no confirmatory cell of the unit is the 95% CI lower bound of D_shiter - D_det
  above +0.05 (5pp; DMC relative return 0.05).
- OEI (v4 6.2): confirmatory `oei` = e(executed elite) - mean e(488 sampled iteration-0 candidates), e = V0 - Vq;
  cell value = mean over pairs of the episode means over measured plans; paired bootstrap CI. Auxiliary oei_w,
  oei_top1, oei_first.
- H2a (TD-MPC2 tasks of experiment 2: mw-push, mw-pick-place, cheetah-run; primary-k cells, default planner): a task
  holds if det OEI 95% CI upper < 0 and sh-iter OEI 95% CI contains 0 (the 4th task, PushT, is the DINO-WM family's).
- H2b (family, confirmatory cells): points = (task, k, variant != none); predictors -OEI, planning-time 1-step (err1h),
  planning-time 3-step (err3); outcome D_v. Primary: per task Spearman(-OEI, D) - max(Spearman(err1, D),
  Spearman(err3, D)) over the task's 18 points, mean over tasks >= 0.15 (tasks with an undefined Spearman are dropped
  from the mean and listed). Auxiliary: pooled Spearman of the raw values (design wording), pooled with OEI divided by
  the task's none-variant first-iteration candidate value std (mean f_sd_v0 of the none episodes), and the same three
  computations with the dataset-type errors.
- Also reported per cell: plan deviation (in-episode reruns: mean dev_mean / mean sn_mean over the cell's measured
  states; fixed-state decision jobs: pooled mean dev / mean seed floor over 3 x 150 states), rho_iter and common_frac
  (decision jobs, first candidate set), held-set errors (prederr jobs, mean over checkpoints) and, per task, the
  near-tie density = median over the decision states of (top1 - top2 unquantized value gap) / candidate value std,
  for the first and the last MPPI candidate set (input for E3's H4b).
"""
import argparse
import glob
import gzip
import json
import os
import re
from collections import defaultdict

import numpy as np

NB = 10000
ALPHA = 0.025
VARS = ["det", "ind", "shiter", "shstep", "shep", "detshift"]
ALLV = ["none"] + VARS
# DESIGN_main v1 section 0.2 (frozen 2026-10-21T18:14Z)
CONF = {"mw-button-press-topdown": [2, 3, 4], "mw-door-open": [3, 4, 5], "mw-drawer-open": [2, 3, 4],
        "mw-peg-insert-side": [3, 4, 5], "mw-pick-place": [2, 3, 4], "mw-push": [2, 3, 4], "mw-sweep-into": [2, 3, 4],
        "mw-window-open": [3, 4, 5], "cheetah-run": [4, 5, 6], "finger-turn-hard": [3, 4, 5],
        "quadruped-walk": [2, 3, 4], "walker-walk": [2, 3, 4]}
PRIMARY = {"mw-button-press-topdown": 3, "mw-door-open": 4, "mw-drawer-open": 3, "mw-peg-insert-side": 4,
           "mw-pick-place": 3, "mw-push": 3, "mw-sweep-into": 3, "mw-window-open": 4, "cheetah-run": 5,
           "finger-turn-hard": 4, "quadruped-walk": 3, "walker-walk": 3}
H2A_TASKS = ["mw-push", "mw-pick-place", "cheetah-run"]
OEI_KEYS = ["oei", "oei_w", "oei_top1", "oei_first", "e_chosen", "f_e_rand", "f_sd_v0", "f_spear", "f_flip", "f_gap",
            "l_spear", "l_flip", "l_gap", "f_sn_spear", "f_sn_flip", "l_e_rand"]
ERR_KEYS = ["err1", "err1h", "err3", "err5", "errr"]
EP_RE = re.compile(r"^(?P<split>[a-z]+)_(?P<task>.+)_s(?P<ckpt>\d+)_k(?P<k>\d+)_(?P<v>[a-z]+)_(?P<ptag>[A-Za-z0-9]+)_c\d+$")


def is_mw(task):
    return task.startswith("mw-")


# ---------------------------------------------------------------- statistics (conventions of code/dinowm/analyze*.py)
def mcnemar(a, b):
    """a, b paired bool arrays -> (#a only, #b only, exact two-sided p) (copied from code/dinowm/analyze.py)."""
    from scipy.stats import binomtest
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    p = 1.0 if n10 + n01 == 0 else float(binomtest(n10, n10 + n01, 0.5).pvalue)
    return n10, n01, p


def holm(pvals, alpha):
    """Holm step-down; returns list of reject flags in the input order (copied from code/dinowm/analyze_e1.py)."""
    order = np.argsort(pvals)
    m = len(pvals)
    rej = [False] * m
    for r, i in enumerate(order):
        if pvals[i] <= alpha / (m - r):
            rej[i] = True
        else:
            break
    return rej


def boot_idx(n, seed=0):
    """The resample index matrix of analyze.boot / analyze_e1.boot_p_le0: RandomState(seed).randint(0, n, (NB, n))."""
    return np.random.RandomState(seed).randint(0, n, size=(NB, n))


def ci95(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def p_le0(v):
    """one-sided bootstrap p for a statistic <= 0 = (#{v* <= 0} + 1) / (B + 1)."""
    v = np.asarray(v, float)
    return float((np.sum(v <= 0) + 1) / (NB + 1))


# ---------------------------------------------------------------- loading
def load(res, split):
    """-> E[(task, ckpt, k, variant, ep)] = compact episode record; P[(task, ckpt, k)] = prederr; Dd[(task, ckpt, k)]."""
    E = {}
    files = sorted(glob.glob(f"{res}/jobs/episodes/{split}_*.json.gz"))
    for f in files:
        m = EP_RE.match(os.path.basename(f)[:-len(".json.gz")])
        if not m or m.group("ptag") != "base" or m.group("split") != split:
            continue
        d = json.load(gzip.open(f, "rt"))
        s = d["spec"]
        if s.get("planner") or s.get("warm_start", True) is False:
            continue
        for e in d["episodes"]:
            r = {"ret": float(e["ret"]), "success": float(e["success"]), "seed": int(e["seed"])}
            om = e.get("oei_mean") or {}
            r["oei"] = {kk: om.get(kk) for kk in OEI_KEYS if om.get(kk) is not None}
            pm = e.get("pd_mean") or {}
            r["pd"] = {kk: pm.get(kk) for kk in ("dev_mean", "dev_act", "sn_mean", "sn_act") if pm.get(kk) is not None}
            for src in ("perr_plan", "perr_traj"):
                pe = e.get(src)
                if pe:
                    r[src] = {kk: pe.get(kk) for kk in ERR_KEYS + ["true1", "true3", "true5"] if pe.get(kk) is not None}
            E[(s["task"], int(s["ckpt"]), int(s["k"]), s["variant"], int(e["ep"]))] = r
    pre = "smoke_" if split == "smoke" else ""
    P, Dd = {}, {}
    for f in sorted(glob.glob(f"{res}/jobs/prederr/{pre}prederr_*.json.gz")):
        if split != "smoke" and os.path.basename(f).startswith("smoke_"):
            continue
        d = json.load(gzip.open(f, "rt"))
        P[(d["task"], int(d["ckpt"]), int(d["k"]))] = {"plan": d["plan"], "traj": d["traj"]}
    for f in sorted(glob.glob(f"{res}/jobs/decision/{pre}decision_*.json.gz")):
        if split != "smoke" and os.path.basename(f).startswith("smoke_"):
            continue
        d = json.load(gzip.open(f, "rt"))
        rows = d["rows"]
        keep = {kk: rows[kk] for kk in ("state", "set", "variant", "gap", "rho_iter", "common_frac", "spear", "flip")
                if kk in rows}
        Dd[(d["task"], int(d["ckpt"]), int(d["k"]))] = {"summary": d["summary"], "rows": keep,
                                                         "plan_rows": d["plan_rows"], "n_states": d["n_states"]}
    return E, P, Dd


# ---------------------------------------------------------------- cell statistics
def pairs_of(E, task, k, variants):
    """(ckpt, ep) pairs present for none and every listed variant at k."""
    base = {(c, e) for (t, c, kk, v, e) in E if t == task and v == "none"}
    for v in variants:
        base &= {(c, e) for (t, c, kk, vv, e) in E if t == task and kk == k and vv == v}
    return sorted(base)


def cell(E, task, k):
    """All statistics of one (task, k) cell. Every bootstrap uses the same resamples (RandomState(0))."""
    vs = [v for v in VARS if any(t == task and kk == k and vv == v for (t, c, kk, vv, e) in E)]
    if "det" not in vs:
        return None
    prs = pairs_of(E, task, k, vs)
    n = len(prs)
    if n == 0:
        return None
    idx = boot_idx(n)
    mw = is_mw(task)
    key = "success" if mw else "ret"
    Y = {"none": np.array([E[(task, c, 0, "none", e)][key] for c, e in prs], float)}
    for v in vs:
        Y[v] = np.array([E[(task, c, k, v, e)][key] for c, e in prs], float)
    yb = {v: Y[v][idx].mean(1) for v in Y}               # resampled means
    m0 = {v: float(Y[v].mean()) for v in Y}

    def drop(mean_none, mean_v):
        return (mean_none - mean_v) if mw else (mean_none - mean_v) / mean_none

    out = {"task": task, "k": k, "family": "mw" if mw else "dmc", "n": n, "metric": key,
           "none_mean": m0["none"], "variants": {}}
    Ddet = drop(m0["none"], m0["det"])
    Ddet_b = drop(yb["none"], yb["det"])
    for v in vs:
        D = drop(m0["none"], m0[v])
        Db = drop(yb["none"], yb[v])
        c = {"mean": m0[v], "D": float(D), "D_ci": ci95(Db)}
        if mw:
            S0, Sv = Y["none"].astype(bool), Y[v].astype(bool)
            n10, n01, p = mcnemar(S0, Sv)
            c.update(none_only=n10, var_only=n01, mcnemar_p=p, s=int(Sv.sum()))
        if v == "det":
            c["p_boot_drop"] = p_le0(Db)
        else:
            dvd = Db - Ddet_b                                 # D_v - D_det per resample
            c["Dv_minus_Ddet"] = float(D - Ddet)
            c["Dv_minus_Ddet_ci"] = ci95(dvd)
            with np.errstate(divide="ignore", invalid="ignore"):
                rb = np.where(Ddet_b > 0, (Ddet_b - Db) / np.where(Ddet_b > 0, Ddet_b, 1.0), np.nan)
            c["red_all"] = float((Ddet - D) / Ddet) if Ddet > 0 else None   # filled into 'red' for qualifying cells
            c["red_ci_all"] = ci95(rb)
            if mw:
                Sd, Sv = Y["det"].astype(bool), Y[v].astype(bool)
                a10, a01, pp = mcnemar(Sv, Sd)
                c["vs_det_mcnemar"] = {"var_only": a10, "det_only": a01, "p": pp}
        # OEI (episode means over measured plans), mean over pairs, paired bootstrap CI
        for kk in ("oei", "oei_w", "oei_top1", "oei_first", "f_e_rand", "f_sd_v0", "f_spear", "l_spear", "f_flip",
                   "f_sn_spear"):
            x = np.array([E[(task, c_, k, v, e)]["oei"].get(kk, np.nan) for c_, e in prs], float)
            if np.isfinite(x).all():
                c[kk] = float(x.mean())
                if kk.startswith("oei"):
                    c[kk + "_ci"] = ci95(x[idx].mean(1))
        # prediction errors (episode means): planning-time (primary) and dataset-type (auxiliary)
        for src, tag in (("perr_plan", "pt"), ("perr_traj", "ds")):
            for kk in ERR_KEYS:
                x = np.array([E[(task, c_, k, v, e)].get(src, {}).get(kk, np.nan) for c_, e in prs], float)
                if np.isfinite(x).all():
                    c[f"{tag}_{kk}"] = float(x.mean())
        # in-episode plan deviation / seed floor (ratio of means over the cell's measured states)
        dv = np.array([E[(task, c_, k, v, e)]["pd"].get("dev_mean", np.nan) for c_, e in prs], float)
        sn = np.array([E[(task, c_, k, v, e)]["pd"].get("sn_mean", np.nan) for c_, e in prs], float)
        if np.isfinite(dv).all() and np.isfinite(sn).all() and sn.mean() > 0:
            c["plandev_ep"] = float(dv.mean() / sn.mean())
        out["variants"][v] = c
    # paired det - sh-iter difference of the planning-time 1-step error (H1 detail)
    if "shiter" in vs:
        for tag, src in (("pt", "perr_plan"), ("ds", "perr_traj")):
            dd = np.array([E[(task, c_, k, "det", e)].get(src, {}).get("err1h", np.nan) -
                           E[(task, c_, k, "shiter", e)].get(src, {}).get("err1h", np.nan) for c_, e in prs], float)
            if np.isfinite(dd).all():
                out[f"{tag}_err1h_det_minus_shiter"] = float(dd.mean())
                out[f"{tag}_err1h_det_minus_shiter_ci"] = ci95(dd[idx].mean(1))
    out["ckpts"] = sorted({c for c, e in prs})
    return out


def add_cell_jobs(C, P, Dd, task, k):
    """Held-set errors (prederr) and fixed-state decision metrics, pooled over the checkpoints that have them."""
    for v, c in C["variants"].items():
        pl = [P[(task, ck, k)]["plan"].get(v) for ck in (1, 2, 3) if (task, ck, k) in P]
        tr = [P[(task, ck, k)]["traj"].get(v) for ck in (1, 2, 3) if (task, ck, k) in P]
        pl, tr = [x for x in pl if x], [x for x in tr if x]
        if pl:
            for kk in ERR_KEYS:
                c[f"held_pt_{kk}"] = float(np.mean([x[kk] for x in pl]))
            c["held_ckpts"] = len(pl)
        if tr:
            for kk in ERR_KEYS:
                c[f"held_ds_{kk}"] = float(np.mean([x[kk] for x in tr]))
        devs, floors, rho, com = [], [], [], []
        for ck in (1, 2, 3):
            d = Dd.get((task, ck, k))
            if not d:
                continue
            pr = d["plan_rows"]
            if f"dev_mean_{v}" in pr:
                devs += [x for x in pr[f"dev_mean_{v}"] if x is not None]
                floors += [x for x in pr["sn_mean"] if x is not None]
            rows = d["rows"]
            for i, (st, vv) in enumerate(zip(rows["set"], rows["variant"])):
                if st == "first" and vv == v:
                    if "rho_iter" in rows and rows["rho_iter"][i] is not None:
                        rho.append(rows["rho_iter"][i])
                    if "common_frac" in rows and rows["common_frac"][i] is not None:
                        com.append(rows["common_frac"][i])
        if devs and floors:
            c["plandev_fixed"] = float(np.mean(devs) / np.mean(floors))
        if rho:
            c["rho_iter"] = float(np.mean(rho))
        if com:
            c["common_frac"] = float(np.mean(com))
        c["decision_ckpts"] = sum(1 for ck in (1, 2, 3) if (task, ck, k) in Dd)


def near_tie(Dd, task):
    """Median (and quartiles) of the unquantized top1-top2 value gap / candidate value std over the decision states of
    the task (the gap does not depend on the variant or on k: same states, candidates, seeds), first and last set."""
    out = {}
    for st in ("first", "last"):
        g = []
        for ck in (1, 2, 3):
            ks = sorted(k for (t, c, k) in Dd if t == task and c == ck)
            if not ks:
                continue
            rows = Dd[(task, ck, ks[0])]["rows"]
            g += [x for x, s_, vv in zip(rows["gap"], rows["set"], rows["variant"])
                  if s_ == st and vv == "det" and x is not None]
        if g:
            out[st] = {"median": float(np.median(g)), "q25": float(np.percentile(g, 25)),
                       "q75": float(np.percentile(g, 75)), "n_states": len(g)}
    return out


# ---------------------------------------------------------------- judgements
def judge(cells, units):
    """H1 / H3a per unit (list of tasks) from the per-task Holm decisions already stored in the cells."""
    out = {}
    for name, tasks in units.items():
        conf = [cells[(t, k)] for t in tasks for k in CONF[t] if (t, k) in cells]
        qual = [c for c in conf if c["det_drop_real_holm"]]
        h1 = [c for c in qual if c["variants"]["det"].get("pt_err1h") is not None and
              c["variants"].get("shiter", {}).get("pt_err1h") is not None and
              c["variants"]["det"]["pt_err1h"] <= c["variants"]["shiter"]["pt_err1h"]]
        h1_ds = [c for c in qual if c["variants"]["det"].get("ds_err1h") is not None and
                 c["variants"].get("shiter", {}).get("ds_err1h") is not None and
                 c["variants"]["det"]["ds_err1h"] <= c["variants"]["shiter"]["ds_err1h"]]
        red_ok = [c for c in qual if c["variants"].get("shiter", {}).get("red") is not None
                  and c["variants"]["shiter"]["red"] >= 0.5]
        harm = [c for c in conf if c["variants"].get("shiter", {}).get("Dv_minus_Ddet_ci", [np.nan])[0] > 0.05]
        share = len(red_ok) / len(qual) if qual else None
        out[name] = {
            "n_confirmatory": len(conf), "n_complete_90": sum(1 for c in conf if c["n"] == 90),
            "qualifying": [[c["task"], c["k"]] for c in qual], "n_qualifying": len(qual),
            "H1": {"holds": len(h1) > 0, "cells_meeting": [[c["task"], c["k"]] for c in h1],
                   "cells_meeting_dataset_type": [[c["task"], c["k"]] for c in h1_ds]},
            "H3a": {"holds": bool(qual) and share >= 0.75 and not harm, "share_red_ge_half": share,
                    "cells_red_ge_half": [[c["task"], c["k"]] for c in red_ok],
                    "harm_cells": [[c["task"], c["k"]] for c in harm]}}
    return out


def spearman(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).correlation)


def h2(cells, none_sd):
    """H2a (3 TD-MPC2 tasks of experiment 2, primary-k cells) and H2b (all confirmatory cells of the family)."""
    h2a = {}
    for t in H2A_TASKS:
        c = cells.get((t, PRIMARY[t]))
        if not c:
            continue
        det, sh = c["variants"]["det"], c["variants"].get("shiter", {})
        ok_det = det.get("oei_ci", [np.nan, np.nan])[1] < 0
        ci_sh = sh.get("oei_ci", [np.nan, np.nan])
        ok_sh = ci_sh[0] <= 0 <= ci_sh[1]
        h2a[t] = {"k": PRIMARY[t], "det_oei": det.get("oei"), "det_oei_ci": det.get("oei_ci"),
                  "shiter_oei": sh.get("oei"), "shiter_oei_ci": ci_sh, "det_ci_upper_lt_0": bool(ok_det),
                  "shiter_ci_contains_0": bool(ok_sh), "holds": bool(ok_det and ok_sh)}
    pts = []
    for (t, k), c in cells.items():
        if k not in CONF.get(t, []):
            continue
        for v in VARS:
            cv = c["variants"].get(v)
            if not cv:
                continue
            pts.append({"task": t, "k": k, "variant": v, "D": cv["D"], "neg_oei": -cv["oei"] if "oei" in cv else np.nan,
                        "neg_oei_norm": (-cv["oei"] / none_sd[t]) if ("oei" in cv and none_sd.get(t)) else np.nan,
                        "pt_err1h": cv.get("pt_err1h", np.nan), "pt_err3": cv.get("pt_err3", np.nan),
                        "ds_err1h": cv.get("ds_err1h", np.nan), "ds_err3": cv.get("ds_err3", np.nan)})

    def sp_diff(r_o, r_1, r_3):
        """rho(-OEI, D) - max(rho(err1, D), rho(err3, D)); NaN if rho(-OEI) or both error rhos are undefined."""
        errs = [r for r in (r_1, r_3) if np.isfinite(r)]
        return float(r_o - max(errs)) if np.isfinite(r_o) and errs else float("nan")

    def per_task(e1, e3, oei_key="neg_oei"):
        res = {}
        for t in sorted({p["task"] for p in pts}):
            P_ = [p for p in pts if p["task"] == t]
            D = [p["D"] for p in P_]
            r_o = spearman([p[oei_key] for p in P_], D)
            r_1 = spearman([p[e1] for p in P_], D)
            r_3 = spearman([p[e3] for p in P_], D)
            res[t] = {"n_points": len(P_), "rho_oei": r_o, "rho_err1": r_1, "rho_err3": r_3,
                      "diff": sp_diff(r_o, r_1, r_3)}
        fin = [v["diff"] for v in res.values() if np.isfinite(v["diff"])]
        return {"per_task": res, "mean_diff": float(np.mean(fin)) if fin else float("nan"),
                "tasks_used": len(fin), "tasks_undefined": [t for t, v in res.items() if not np.isfinite(v["diff"])],
                "holds": bool(fin) and float(np.mean(fin)) >= 0.15}

    def pooled(oei_key, e1, e3):
        D = [p["D"] for p in pts]
        r_o = spearman([p[oei_key] for p in pts], D)
        r_1 = spearman([p[e1] for p in pts], D)
        r_3 = spearman([p[e3] for p in pts], D)
        dd = sp_diff(r_o, r_1, r_3)
        return {"n_points": len(pts), "rho_oei": r_o, "rho_err1": r_1, "rho_err3": r_3, "diff": dd,
                "holds_ge_0.15": bool(np.isfinite(dd) and dd >= 0.15)}

    h2b = {"primary_within_task": per_task("pt_err1h", "pt_err3"),
           "aux_pooled_raw": pooled("neg_oei", "pt_err1h", "pt_err3"),
           "aux_pooled_oei_normalized": pooled("neg_oei_norm", "pt_err1h", "pt_err3"),
           "aux_dataset_type_within_task": per_task("ds_err1h", "ds_err3"),
           "aux_dataset_type_pooled_raw": pooled("neg_oei", "ds_err1h", "ds_err3"),
           "aux_dataset_type_pooled_oei_normalized": pooled("neg_oei_norm", "ds_err1h", "ds_err3"),
           "points": pts}
    return h2a, h2b


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--split", default="eval")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    E, P, Dd = load(a.res, a.split)
    tasks = [t for t in CONF if any(tt == t for (tt, c, k, v, e) in E)]
    cells = {}
    for t in tasks:
        ks = sorted({k for (tt, c, k, v, e) in E if tt == t and v != "none"})
        for k in ks:
            C = cell(E, t, k)
            if C is None:
                continue
            add_cell_jobs(C, P, Dd, t, k)
            cells[(t, k)] = C
        # per-task Holm over the confirmatory k cells (alpha 0.025, one-sided)
        ks_conf = [k for k in CONF[t] if (t, k) in cells]
        ps = [cells[(t, k)]["variants"]["det"]["p_boot_drop"] for k in ks_conf]
        rej = holm(ps, ALPHA) if ps else []
        for k in ks:
            if (t, k) not in cells:
                continue
            C = cells[(t, k)]
            det = C["variants"]["det"]
            C["confirmatory"] = k in ks_conf
            C["det_drop_real_holm"] = bool(k in ks_conf and rej[ks_conf.index(k)] and det["D"] > 0)
            C["det_drop_real_ci95"] = bool(det["D_ci"][0] > 0)
            C["holm_m"] = len(ks_conf)
            for v, c in C["variants"].items():
                if v != "det":
                    c["red"] = c["red_all"] if C["det_drop_real_holm"] else None
                    c["red_ci"] = c["red_ci_all"] if C["det_drop_real_holm"] else None
    none_sd = {}
    for t in tasks:
        x = [E[key]["oei"].get("f_sd_v0") for key in E if key[0] == t and key[3] == "none"]
        x = [v for v in x if v is not None]
        none_sd[t] = float(np.mean(x)) if x else None
    nt = {t: near_tie(Dd, t) for t in tasks}
    units = {"tdmpc2_mw": [t for t in tasks if is_mw(t)], "tdmpc2_dmc": [t for t in tasks if not is_mw(t)],
             "tdmpc2_all": tasks}
    J = judge(cells, units)
    h2a, h2b = h2(cells, none_sd)
    base = {}
    for t in tasks:
        key = "success" if is_mw(t) else "ret"
        x = [E[kk][key] for kk in E if kk[0] == t and kk[3] == "none"]
        base[t] = {"n": len(x), "mean": float(np.mean(x)) if x else None}
    summary = {"split": a.split, "n_episodes": len(E), "n_cells": len(cells), "baseline": base, "none_value_sd": none_sd,
               "near_tie": nt, "judgements": J, "H2a": h2a, "H2b": {k: v for k, v in h2b.items() if k != "points"},
               "cells": {f"{t}:{k}": c for (t, k), c in cells.items()}}
    json.dump(summary, open(f"{a.outdir}/e1_tdmpc2_summary.json", "w"), indent=1, default=float)
    json.dump(h2b["points"], open(f"{a.outdir}/e1_tdmpc2_h2b_points.json", "w"), indent=1, default=float)
    write_csv(cells, f"{a.outdir}/e1_tdmpc2_cells.csv")
    write_md(summary, f"{a.outdir}/e1_tdmpc2_tables.md")
    print_summary(summary)


CSV_COLS = ["task", "k", "family", "confirmatory", "n", "variant", "mean", "D", "D_ci", "p_boot_drop", "holm_real",
            "mcnemar_p", "none_only", "var_only", "red", "red_ci", "Dv_minus_Ddet", "Dv_minus_Ddet_ci", "oei", "oei_ci",
            "oei_w", "oei_top1", "oei_first", "f_e_rand", "f_sd_v0", "f_spear", "l_spear", "pt_err1h", "pt_err1",
            "pt_err3", "pt_err5", "pt_errr", "ds_err1h", "ds_err1", "ds_err3", "ds_err5", "held_pt_err1h",
            "held_pt_err3", "held_pt_err5", "held_ds_err1h", "plandev_ep", "plandev_fixed", "rho_iter", "common_frac"]


def fmt(x):
    if isinstance(x, (list, tuple)):
        return "[" + ";".join(fmt(y) for y in x) + "]"
    if isinstance(x, float):
        return f"{x:.5g}"
    return "" if x is None else str(x)


def write_csv(cells, path):
    with open(path, "w") as f:
        f.write(",".join(CSV_COLS) + "\n")
        for (t, k), C in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            for v, c in C["variants"].items():
                row = dict(c, task=t, k=k, family=C["family"], confirmatory=C["confirmatory"], n=C["n"], variant=v,
                           holm_real=C["det_drop_real_holm"] if v == "det" else None)
                f.write(",".join(fmt(row.get(col)) for col in CSV_COLS) + "\n")


def pct(x, mw):
    """D-type quantity as text: Meta-World in pp, DMC in % of the none return (both x100)."""
    return "" if x is None else f"{100 * x:+.1f}"


def ci_txt(ci, scale=100, nd=1):
    if not ci or ci[0] is None or not np.isfinite(ci[0]):
        return ""
    return f"[{scale * ci[0]:.{nd}f}, {scale * ci[1]:.{nd}f}]"


def num(x, nd=3):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def write_md(S, path):
    L = []
    L.append("## 판정 단위별 요약\n")
    L.append("| 단위 | 확증 칸 | 자격 칸(Holm) | H1 | H1 충족 칸 | red_sh-iter ≥ 0.5 비율 | 해로운 칸 | H3a |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name, J in S["judgements"].items():
        sh = J["H3a"]["share_red_ge_half"]
        L.append(f"| {name} | {J['n_confirmatory']} | {J['n_qualifying']} | {J['H1']['holds']} | "
                 f"{len(J['H1']['cells_meeting'])} | {'' if sh is None else f'{sh:.2f}'} "
                 f"({len(J['H3a']['cells_red_ge_half'])}/{J['n_qualifying']}) | {J['H3a']['harm_cells']} | "
                 f"{J['H3a']['holds']} |")
    L.append("\n## det 하락과 Holm(과제마다 확증 k 3칸, 한쪽 α = 0.025)\n")
    L.append("| 과제 | k | n | none | det | D_det [95% CI] | p_boot | Holm 실재 | CI 하한 > 0 | McNemar p |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for key, C in S["cells"].items():
        d = C["variants"]["det"]
        mw = C["family"] == "mw"
        L.append(f"| {C['task']} | {C['k']} | {C['n']} | {num(C['none_mean'], 3 if mw else 1)} | "
                 f"{num(d['mean'], 3 if mw else 1)} | {pct(d['D'], mw)} {ci_txt(d['D_ci'])} | {d['p_boot_drop']:.4f} | "
                 f"{C['det_drop_real_holm']} | {C['det_drop_real_ci95']} | {num(d.get('mcnemar_p'), 4)} |")
    L.append("\n## 칸별 7변형 표(D·red·D_v − D_det: MW %p, DMC return %; 오차: 계획 시점 err1h/err3/err5, "
             "괄호는 데이터셋형; 계획 편차: episode 안 / 고정 상태, seed 바닥 배수)\n")
    L.append("| 칸 | 변형 | 평균 | D [95% CI] | McNemar p | red [95% CI] | D_v − D_det [95% CI] | OEI [95% CI] | "
             "1/3/5-step 계획 시점 (데이터셋형) | 계획 편차 | rho_iter | common_frac |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, C in S["cells"].items():
        mw = C["family"] == "mw"
        for v, c in C["variants"].items():
            err = (f"{num(c.get('pt_err1h'))}/{num(c.get('pt_err3'))}/{num(c.get('pt_err5'))} "
                   f"({num(c.get('ds_err1h'))}/{num(c.get('ds_err3'))}/{num(c.get('ds_err5'))})")
            red = "" if c.get("red") is None else f"{c['red']:.2f} {ci_txt(c.get('red_ci'), 1, 2)}"
            dvd = "" if v == "det" else f"{pct(c.get('Dv_minus_Ddet'), mw)} {ci_txt(c.get('Dv_minus_Ddet_ci'))}"
            oei = f"{num(c.get('oei'), 2)} {ci_txt(c.get('oei_ci'), 1, 2)}"
            pdv = f"{num(c.get('plandev_ep'), 2)} / {num(c.get('plandev_fixed'), 2)}"
            L.append(f"| {C['task']} k={C['k']} | {v} | {num(c['mean'], 3 if mw else 1)} | {pct(c['D'], mw)} "
                     f"{ci_txt(c['D_ci'])} | {num(c.get('mcnemar_p'), 4)} | {red} | {dvd} | {oei} | {err} | {pdv} | "
                     f"{num(c.get('rho_iter'), 2)} | {num(c.get('common_frac'), 2)} |")
    L.append("\n## H2a(1차 k 칸)\n")
    L.append("| 과제 | k | det OEI [95% CI] | sh-iter OEI [95% CI] | det 상한 < 0 | sh-iter CI가 0 포함 | 성립 |")
    L.append("|---|---|---|---|---|---|---|")
    for t, h in S["H2a"].items():
        L.append(f"| {t} | {h['k']} | {num(h['det_oei'], 2)} {ci_txt(h['det_oei_ci'], 1, 2)} | "
                 f"{num(h['shiter_oei'], 2)} {ci_txt(h['shiter_oei_ci'], 1, 2)} | {h['det_ci_upper_lt_0']} | "
                 f"{h['shiter_ci_contains_0']} | {h['holds']} |")
    L.append("\n## H2b(과제 안 Spearman: ρ(−OEI, D) − max(ρ(1-step, D), ρ(3-step, D)))\n")
    L.append("| 과제 | 점 | ρ(−OEI, D) | ρ(1-step, D) | ρ(3-step, D) | 차 | 데이터셋형 차 |")
    L.append("|---|---|---|---|---|---|---|")
    pr, ds = S["H2b"]["primary_within_task"], S["H2b"]["aux_dataset_type_within_task"]
    for t, r in pr["per_task"].items():
        L.append(f"| {t} | {r['n_points']} | {num(r['rho_oei'])} | {num(r['rho_err1'])} | {num(r['rho_err3'])} | "
                 f"{num(r['diff'])} | {num(ds['per_task'].get(t, {}).get('diff'))} |")
    L.append(f"| **과제 평균(주 계산)** | | | | | **{num(pr['mean_diff'])}** | {num(ds['mean_diff'])} |")
    for lab, key in (("섞은 Spearman(원값)", "aux_pooled_raw"), ("섞은 Spearman(OEI 정규화)", "aux_pooled_oei_normalized"),
                     ("섞은 Spearman(원값, 데이터셋형)", "aux_dataset_type_pooled_raw"),
                     ("섞은 Spearman(OEI 정규화, 데이터셋형)", "aux_dataset_type_pooled_oei_normalized")):
        r = S["H2b"][key]
        L.append(f"| 보조: {lab} | {r['n_points']} | {num(r['rho_oei'])} | {num(r['rho_err1'])} | "
                 f"{num(r['rho_err3'])} | {num(r['diff'])} | |")
    L.append("\n## near-tie 밀도(결정 상태의 비양자화 1·2위 가치 차 / 후보 가치 std, 중앙값 [사분위])\n")
    L.append("| 과제 | 첫 반복 후보 집합 | 마지막 반복 후보 집합 | 상태 수 |")
    L.append("|---|---|---|---|")
    for t, d in S["near_tie"].items():
        f_, l_ = d.get("first", {}), d.get("last", {})
        L.append(f"| {t} | {num(f_.get('median'))} [{num(f_.get('q25'))}, {num(f_.get('q75'))}] | "
                 f"{num(l_.get('median'))} [{num(l_.get('q25'))}, {num(l_.get('q75'))}] | {f_.get('n_states', '')} |")
    open(path, "w").write("\n".join(L) + "\n")


def print_summary(S):
    print(f"split {S['split']}: {S['n_episodes']} episodes, {S['n_cells']} cells")
    for name, J in S["judgements"].items():
        print(f"== {name}: confirmatory {J['n_confirmatory']} (complete {J['n_complete_90']}), qualifying "
              f"{J['n_qualifying']}; H1 {J['H1']['holds']}; H3a {J['H3a']['holds']} share "
              f"{J['H3a']['share_red_ge_half']} harm {J['H3a']['harm_cells']}")
    print("H2a:", {t: v["holds"] for t, v in S["H2a"].items()})
    hb = S["H2b"]["primary_within_task"]
    print("H2b primary: mean within-task diff", hb["mean_diff"], "holds", hb["holds"], "undefined", hb["tasks_undefined"])


if __name__ == "__main__":
    main()
