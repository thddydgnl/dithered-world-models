"""E17 closed-loop analysis (tmlr_ext/DESIGN_tmlr.md 7 and 0.3), written before the main run; runs on the pod.

Data: the episode jobs of split 'eval' (seeds 100000-100029, checkpoints 1-3 -> 90 pairs) in <res>/jobs/episodes/.
Statistics (DESIGN_main v4 6.1 / DESIGN_tmlr 0.3):
  D_v (damage vs the same pod's FP32 none): Meta-World success-rate difference none - v (final-step success, E1),
       DMC relative return drop 1 - sum R_v / sum R_none, recomputed inside every resample
  paired bootstrap over the (checkpoint, seed) pairs common to the none and every variant of the task: 10,000
       resamples, numpy RandomState(0).randint(0, n, (10000, n)), one resample matrix for every statistic of the task,
       95% percentile CI; one-sided p(D <= 0) = (#{D* <= 0} + 1) / (B + 1); threshold comparisons with tolerance 1e-9
  delta_ds = D(det) - D(det+shift) and delta_sh = D(det) - D(sh-iter) on the same resamples (paired)
  red_v = (D_det - D_v) / D_det (point; descriptive here); harmful: CI lower bound of D_v - D_det > 0.05 (descriptive)
Verdicts
  K2 (integer side, this pod): delta_int = D_int(det) - D_int(det+shift) per part-2 task with its CI. K2 holds if in all
     4 tasks sign(delta_int) == sign(delta_fq) and the two 95% CIs overlap, delta_fq = the same quantity of the td
     worker's E11 fake-quant cells (their pod, their none; sign with tolerance 1e-9, i.e. |delta| <= 1e-9 counts as 0).
     The comparison runs when --fq_csv is given (columns task, ckpt, seed, variant, k, success, ret; variant in none,
     det, detshift at the E1 primary k); otherwise only the integer side is reported (the coordinator combines).
  K3 pre-cast BF16: holds if the 95% CI of D(bf16p) contains 0 in all 12 tasks.
  Descriptive (part 4): D of the folded per-channel integer cells (k 2, 3 x det, sh-iter) with CIs; with --fq_csv
     rows of range 'pc' (column range = pc), also the difference D_int_pc - D_fq_pc with a CI from independent resamples
     (RandomState(0) for this pod, RandomState(1) for the other pod: different pods are not pairs, DESIGN_tmlr 0.3).
Checks
  0.7 check 3: |mean_none - E1 none| <= 0.10 (Meta-World success rate) or <= 10% of the E1 none return (DMC), per task.
  integrity: 90 episodes per cell with the expected seeds, the spec of every job, one GPU type and host per pod.
Outputs (<outdir>): e17_summary.json, e17_cells.csv, e17_episodes.csv, e17_tables.md
usage: python analyze_e17.py --res /workspace/atm/out --e1 inputs/e1_tdmpc2_baseline.json --outdir <dir>
       [--fq_csv <the td worker's td_episodes.csv, or a generic per-episode csv>]
"""
import argparse
import csv
import glob
import gzip
import json
import os
from collections import defaultdict

import numpy as np

NB = 10000
EPS = 1e-9
MW8 = ["mw-push", "mw-pick-place", "mw-door-open", "mw-drawer-open", "mw-button-press-topdown",
       "mw-peg-insert-side", "mw-window-open", "mw-sweep-into"]
DMC4 = ["cheetah-run", "walker-walk", "quadruped-walk", "finger-turn-hard"]
TASKS12 = MW8 + DMC4
PART2 = {"mw-push": 3, "mw-pick-place": 3, "cheetah-run": 5, "walker-walk": 3}
PART4_K = (2, 3)
SEEDS = list(range(100000, 100030))
CKPTS = (1, 2, 3)


def is_mw(task):
    return task.startswith("mw-")


def cell_list(task):
    """(name, prec, k, variant) of every non-none cell of a task."""
    out = []
    if task in PART2:
        k = PART2[task]
        out += [(f"intk{k}_{v}", "intk", k, v) for v in ("det", "shiter", "detshift")]
    out.append(("bf16p", "bf16p", 0, "none"))
    if task in PART2:
        out += [(f"intpc{k}_{v}", "intpc", k, v) for k in PART4_K for v in ("det", "shiter")]
    return out


def ci95(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def p_le0(v):
    v = np.asarray(v, float)
    return float((np.sum(v <= 0) + 1) / (len(v) + 1))


def sign(x):
    return 0 if abs(x) <= EPS else (1 if x > 0 else -1)


def overlap(a, b):
    return bool(a[0] <= b[1] + EPS and b[0] <= a[1] + EPS)


def drop(mw, m_none, m_v):
    return (m_none - m_v) if mw else (m_none - m_v) / m_none


# ---------------------------------------------------------------------------------------------------- loading
def load(res):
    """-> E[(task, ckpt, prec, k, variant, seed)] = record; jobs = list of specs; envs = set of (host, gpu)."""
    E, jobs, envs, problems = {}, [], defaultdict(int), []
    for f in sorted(glob.glob(f"{res}/jobs/episodes/eval_*.json.gz")):
        d = json.load(gzip.open(f, "rt"))
        s = d["spec"]
        if s.get("split") != "eval":
            continue
        jobs.append(s)
        envs[(d["env"]["host"], d["env"]["gpu"])] += 1
        prec = s.get("prec", s.get("e7", "fp32"))
        if (s.get("e7") or "fp32") != (prec if prec != "fp32" else "fp32"):
            problems.append(f"{s['id']}: e7 {s.get('e7')} vs prec {prec}")
        if s.get("planner"):
            problems.append(f"{s['id']}: planner override {s['planner']}")
        if "e7" in s and d.get("e7", {}).get("prec") != s["e7"]:
            problems.append(f"{s['id']}: recorded kernel info {d.get('e7')}")
        for e in d["episodes"]:
            key = (s["task"], int(s["ckpt"]), prec, int(s["k"]), s["variant"], int(e["seed"]))
            if key in E:
                problems.append(f"duplicate episode {key}")
            if int(e["seed"]) != int(s["seed0"]) + int(e["ep"]):
                problems.append(f"{s['id']}: seed {e['seed']} != seed0 + ep")
            E[key] = {"success": float(e["success"]), "ret": float(e["ret"]), "steps": int(e["steps"]),
                      "secs": float(e.get("secs_loop", e.get("secs", float("nan")))), "job": s["id"],
                      "host": d["env"]["host"], "gpu": d["env"]["gpu"]}
    return E, jobs, dict(envs), problems


def load_fq_csv(path):
    """the other pod's fake-quant episodes -> F[(range, task, k, variant, ckpt, seed)] (none rows with k 0). Two formats:
      generic  columns task, ckpt, seed, variant, k, success, ret[, range] (range 'pt' if absent)
      td       the td worker's td_episodes.csv: task, ckpt, ep, seed, cell, ptag, ret, success, ... with cell 'none' or
               '<PT|PC>k<k>_<variant>' (tmlr_ext/TD_tdmpc2/code/plan_td.cell_name); only ptag 'base' rows (default
               planner, E11 / E12 cells) are used"""
    F = {}
    for r in csv.DictReader(open(path)):
        if "cell" in r:
            if (r.get("ptag") or "base") != "base":
                continue
            cell = r["cell"].strip()
            if cell == "none":
                rng, k, v = "pt", 0, "none"
            else:
                head, v = cell.split("_", 1)
                rng, k = head[:2].lower(), int(head[3:])
        else:
            rng = (r.get("range") or "pt").strip().lower()
            v = r["variant"].strip()
            k = 0 if v == "none" else int(r["k"])
        F[(rng, r["task"], k, v, int(r["ckpt"]), int(r["seed"]))] = {"success": float(r["success"]),
                                                                    "ret": float(r["ret"])}
    return F


# ---------------------------------------------------------------------------------------------------- statistics
def task_stats(E, task, seed=0):
    """all cell statistics of one task on its common pairs (one resample matrix)."""
    mw = is_mw(task)
    key = "success" if mw else "ret"
    cl = [c for c in cell_list(task)]
    have = [c for c in cl if any(kk[:5] == (task, kk[1], c[1], c[2], c[3]) for kk in E)]
    none_pairs = {(kk[1], kk[5]) for kk in E if kk[0] == task and kk[2] == "fp32" and kk[4] == "none"}
    pairs = set(none_pairs)
    per_cell_pairs = {}
    for name, prec, k, v in have:
        pc = {(kk[1], kk[5]) for kk in E if kk[0] == task and kk[2] == prec and kk[3] == k and kk[4] == v}
        per_cell_pairs[name] = len(pc & none_pairs)
        pairs &= pc
    pairs = sorted(pairs)
    n = len(pairs)
    out = {"task": task, "family": "mw" if mw else "dmc", "metric": key, "n_pairs_common": n,
           "n_none": len(none_pairs), "n_pairs_per_cell": per_cell_pairs, "cells": {}}
    if n == 0:
        return out
    idx = np.random.RandomState(seed).randint(0, n, size=(NB, n))
    Y = {"none": np.array([E[(task, c, "fp32", 0, "none", s)][key] for c, s in pairs], float)}
    for name, prec, k, v in have:
        Y[name] = np.array([E[(task, c, prec, k, v, s)][key] for c, s in pairs], float)
    yb = {nm: Y[nm][idx].mean(1) for nm in Y}
    m0 = {nm: float(Y[nm].mean()) for nm in Y}
    out["none_mean"] = m0["none"]
    Db = {}
    for name, prec, k, v in have:
        D = drop(mw, m0["none"], m0[name])
        Db[name] = drop(mw, yb["none"], yb[name])
        c = {"prec": prec, "k": k, "variant": v, "mean": m0[name], "D": float(D), "D_ci": ci95(Db[name]),
             "p_le0": p_le0(Db[name]), "ci_contains_0": bool(ci95(Db[name])[0] <= EPS and ci95(Db[name])[1] >= -EPS)}
        if mw:
            s0, sv = Y["none"].astype(bool), Y[name].astype(bool)
            c.update(none_only=int(np.sum(s0 & ~sv)), var_only=int(np.sum(~s0 & sv)))
        out["cells"][name] = c
    for grp, k in (("intk", PART2.get(task)), ("intpc", 2), ("intpc", 3)):
        if k is None:
            continue
        det = f"{grp}{k}_det"
        if det not in Db:
            continue
        for v in ("shiter", "detshift"):
            nm = f"{grp}{k}_{v}"
            if nm not in Db:
                continue
            c = out["cells"][nm]
            dd = Db[det] - Db[nm]
            c["delta_det_minus_v"] = float(out["cells"][det]["D"] - c["D"])
            c["delta_det_minus_v_ci"] = ci95(dd)
            Ddet = out["cells"][det]["D"]
            c["red"] = float((Ddet - c["D"]) / Ddet) if Ddet > EPS else None
            c["Dv_minus_Ddet_ci"] = ci95(Db[nm] - Db[det])
            c["harmful"] = bool(c["Dv_minus_Ddet_ci"][0] > 0.05 + EPS)
    return out


def fq_delta(F, task, k, rng="pt", v="detshift", seed=0):
    """delta_fq = D_fq(det) - D_fq(v) of the other pod's fake-quant cells (their none, their pairs)."""
    mw = is_mw(task)
    key = "success" if mw else "ret"
    none = {(kk[4], kk[5]) for kk in F if kk[0] == rng and kk[1] == task and kk[3] == "none"} or \
           {(kk[4], kk[5]) for kk in F if kk[1] == task and kk[3] == "none"}
    nrng = rng if any(kk[0] == rng and kk[1] == task and kk[3] == "none" for kk in F) else "pt"
    pr = set(none)
    for vv in ("det", v):
        pr &= {(kk[4], kk[5]) for kk in F if kk[0] == rng and kk[1] == task and kk[2] == k and kk[3] == vv}
    pr = sorted(pr)
    n = len(pr)
    if n == 0:
        return None
    idx = np.random.RandomState(seed).randint(0, n, size=(NB, n))
    Y = {"none": np.array([F[(nrng, task, 0, "none", c, s)][key] for c, s in pr], float)}
    for vv in ("det", v):
        Y[vv] = np.array([F[(rng, task, k, vv, c, s)][key] for c, s in pr], float)
    yb = {vv: Y[vv][idx].mean(1) for vv in Y}
    m0 = {vv: float(Y[vv].mean()) for vv in Y}
    Dd, Dv = drop(mw, m0["none"], m0["det"]), drop(mw, m0["none"], m0[v])
    Ddb, Dvb = drop(mw, yb["none"], yb["det"]), drop(mw, yb["none"], yb[v])
    return {"n": n, "D_det": float(Dd), "D_v": float(Dv), "delta": float(Dd - Dv), "delta_ci": ci95(Ddb - Dvb),
            "D_v_boot": Dvb, "D_det_boot": Ddb}


# ---------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--e1", required=True, help="E1 TD-MPC2 none baseline (e1_tdmpc2_summary.json 'baseline')")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--fq_csv", default=None)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    E, jobs, envs, problems = load(a.res)
    e1 = json.load(open(a.e1))
    e1 = e1.get("baseline", e1)
    # integrity
    counts = defaultdict(set)
    for (t, c, p, k, v, s) in E:
        counts[(t, p, k, v)].add((c, s))
    expected = {(c, s) for c in CKPTS for s in SEEDS}
    integ = {"n_episodes": len(E), "n_jobs": len(jobs), "envs": {f"{h}|{g}": n for (h, g), n in envs.items()},
             "problems": problems, "cells": {}}
    for t in TASKS12:
        for name, p, k, v in [("none", "fp32", 0, "none")] + cell_list(t):
            got = counts.get((t, p, k, v), set())
            integ["cells"][f"{t}|{name}"] = {"n": len(got), "complete": got == expected,
                                             "unexpected": len(got - expected)}
    integ["all_complete"] = all(c["complete"] for c in integ["cells"].values())
    integ["one_gpu_type"] = len({g for (h, g) in envs}) <= 1
    integ["one_host"] = len({h for (h, g) in envs}) <= 1
    # statistics
    T = {t: task_stats(E, t) for t in TASKS12}
    # check 3 (0.7): FP32 none vs E1 none
    chk3 = {}
    for t in TASKS12:
        if "none_mean" not in T[t]:
            continue
        m_new = T[t]["none_mean"] if T[t]["n_pairs_common"] else None
        none_vals = [E[kk]["success" if is_mw(t) else "ret"] for kk in E if kk[0] == t and kk[2] == "fp32"]
        m_all = float(np.mean(none_vals)) if none_vals else None
        ref = float(e1[t]["mean"])
        dev = (m_all - ref) if is_mw(t) else (m_all - ref) / ref
        chk3[t] = {"none_mean_all_episodes": m_all, "n": len(none_vals), "e1_none": ref, "deviation": dev,
                   "within_10": bool(abs(dev) <= 0.10 + EPS)}
    # K3
    k3 = {t: T[t]["cells"].get("bf16p") for t in TASKS12}
    k3_ok = [t for t, c in k3.items() if c is not None and c["ci_contains_0"]]
    K3 = {"holds": len(k3_ok) == 12, "n_tasks_ci_contains_0": len(k3_ok), "n_tasks_with_data": sum(
        c is not None for c in k3.values()), "tasks_damaged_or_ci_excludes_0": [t for t, c in k3.items()
                                                                              if c is not None and not c["ci_contains_0"]],
          "per_task": {t: (None if c is None else {"D": c["D"], "D_ci": c["D_ci"], "mean": c["mean"]})
                       for t, c in k3.items()}}
    # K2 (integer side) and the comparison when the fake-quant episodes are given
    F = load_fq_csv(a.fq_csv) if a.fq_csv else None
    k2 = {}
    for t, k in PART2.items():
        c = T[t]["cells"].get(f"intk{k}_detshift")
        if c is None or "delta_det_minus_v" not in c:
            k2[t] = None
            continue
        row = {"k": k, "D_int_det": T[t]["cells"][f"intk{k}_det"]["D"], "D_int_detshift": c["D"],
               "delta_int": c["delta_det_minus_v"], "delta_int_ci": c["delta_det_minus_v_ci"],
               "D_int_shiter": T[t]["cells"].get(f"intk{k}_shiter", {}).get("D")}
        if F is not None:
            fq = fq_delta(F, t, k, "pt", "detshift")
            if fq is not None:
                row.update(delta_fq=fq["delta"], delta_fq_ci=fq["delta_ci"], n_fq=fq["n"], D_fq_det=fq["D_det"],
                           D_fq_detshift=fq["D_v"], same_sign=sign(row["delta_int"]) == sign(fq["delta"]),
                           ci_overlap=overlap(row["delta_int_ci"], fq["delta_ci"]))
                row["same_direction"] = bool(row["same_sign"] and row["ci_overlap"])
        k2[t] = row
    K2 = {"integer_side": k2}
    if F is not None:
        ok = [t for t, r in k2.items() if r is not None and r.get("same_direction")]
        K2.update(holds=len(ok) == 4, n_same_direction=len(ok),
                  n_compared=sum(r is not None and "delta_fq" in r for r in k2.values()))
    else:
        K2["holds"] = None
        K2["note"] = "fake-quant side (td worker's E11 cells) not given; the coordinator combines (DESIGN_tmlr 8.5)"
    # descriptive part 4
    pcd = {}
    for t in PART2:
        for k in PART4_K:
            for v in ("det", "shiter"):
                c = T[t]["cells"].get(f"intpc{k}_{v}")
                if c is None:
                    continue
                r = {"D": c["D"], "D_ci": c["D_ci"], "mean": c["mean"]}
                if F is not None:
                    fq = fq_delta(F, t, k, "pc", v, seed=1)
                    if fq is not None:
                        # D_int_pc - D_fq_pc with independent resamples (this pod RandomState(0), the other 1)
                        mw = is_mw(t)
                        st = task_stats(E, t, seed=0)
                        r.update(D_fq=fq["D_v"], diff=float(c["D"] - fq["D_v"]))
                        # the integer cell's resampled D (RandomState(0)) minus the fake-quant's (RandomState(1))
                        # recomputed here on the same matrices as task_stats
                        key = "success" if mw else "ret"
                        prs = sorted({(kk[1], kk[5]) for kk in E if kk[0] == t and kk[2] == "intpc" and kk[3] == k
                                      and kk[4] == v} & {(kk[1], kk[5]) for kk in E if kk[0] == t and kk[2] == "fp32"})
                        idx = np.random.RandomState(0).randint(0, len(prs), size=(NB, len(prs)))
                        yn = np.array([E[(t, cc, "fp32", 0, "none", s)][key] for cc, s in prs])[idx].mean(1)
                        yv = np.array([E[(t, cc, "intpc", k, v, s)][key] for cc, s in prs])[idx].mean(1)
                        r["diff_ci"] = ci95(drop(mw, yn, yv) - fq["D_v_boot"])
                        del st
                pcd[f"{t}|k{k}|{v}"] = r
    summary = {"label": "E17 closed loop (DESIGN_tmlr 7): K2 integer side, K3, part-4 descriptive, check 3",
               "K2": K2, "K3": K3, "pc_descriptive": pcd, "check3_none_vs_e1": chk3,
               "check3_all_within": bool(chk3 and all(v["within_10"] for v in chk3.values())),
               "integrity": integ, "tasks": T}
    json.dump(summary, open(f"{a.outdir}/e17_summary.json", "w"), indent=1, default=float)
    # cells csv
    cols = ["task", "cell", "prec", "k", "variant", "n_pairs", "none_mean", "mean", "D", "D_ci_lo", "D_ci_hi", "p_le0",
            "delta_det_minus_v", "delta_ci_lo", "delta_ci_hi", "red", "harmful"]
    with open(f"{a.outdir}/e17_cells.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for t in TASKS12:
            for name, c in T[t]["cells"].items():
                dci = c.get("delta_det_minus_v_ci", [None, None])
                w.writerow([t, name, c["prec"], c["k"], c["variant"], T[t]["n_pairs_common"], T[t].get("none_mean"),
                            c["mean"], c["D"], c["D_ci"][0], c["D_ci"][1], c["p_le0"], c.get("delta_det_minus_v"),
                            dci[0], dci[1], c.get("red"), c.get("harmful")])
    with open(f"{a.outdir}/e17_episodes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "ckpt", "prec", "k", "variant", "seed", "success", "ret", "steps", "secs", "job", "host",
                    "gpu"])
        for kk in sorted(E):
            r = E[kk]
            w.writerow(list(kk) + [r["success"], r["ret"], r["steps"], r["secs"], r["job"], r["host"], r["gpu"]])
    # tables
    fmt = lambda x, mw: ("-" if x is None else (f"{100 * x:+.1f}" if mw else f"{100 * x:+.1f}%"))
    lines = ["# E17 closed loop tables (auto-generated by analyze_e17.py)", "",
             f"- episodes {len(E)}, jobs {len(jobs)}, envs {integ['envs']}, all cells complete: {integ['all_complete']}",
             f"- problems: {problems[:5]}{' ...' if len(problems) > 5 else ''}", "",
             "## Verdicts", "",
             f"- K2: {K2.get('holds')} ({K2.get('note', '')}{'' if 'note' in K2 else str(K2.get('n_same_direction')) + '/4 tasks same direction, ' + str(K2.get('n_compared')) + ' compared'})",
             f"- K3: {K3['holds']} ({K3['n_tasks_ci_contains_0']}/12 tasks with the CI of D(bf16p) containing 0)",
             f"- check 3 (FP32 none within 10 points / 10% of E1): {summary['check3_all_within']}", "",
             "## Part 2: integer kernel (D vs FP32 none; MW points, DMC %)", "",
             "| task | k | none | det | sh-iter | det+shift | D(det) - D(det+shift) [CI] | D(det) - D(sh-iter) [CI] |",
             "|---|---|---|---|---|---|---|---|"]
    for t, k in PART2.items():
        cs = T[t]["cells"]
        mw = is_mw(t)
        if f"intk{k}_det" not in cs:
            continue
        g = lambda nm: (f"{fmt(cs[nm]['D'], mw)} [{fmt(cs[nm]['D_ci'][0], mw)}, {fmt(cs[nm]['D_ci'][1], mw)}]"
                        if nm in cs else "-")
        dl = lambda nm: (f"{fmt(cs[nm]['delta_det_minus_v'], mw)} [{fmt(cs[nm]['delta_det_minus_v_ci'][0], mw)}, "
                         f"{fmt(cs[nm]['delta_det_minus_v_ci'][1], mw)}]" if nm in cs and
                         "delta_det_minus_v" in cs[nm] else "-")
        lines.append(f"| {t} | {k} | {T[t]['none_mean']:.3f} | {g(f'intk{k}_det')} | {g(f'intk{k}_shiter')} | "
                     f"{g(f'intk{k}_detshift')} | {dl(f'intk{k}_detshift')} | {dl(f'intk{k}_shiter')} |")
    lines += ["", "## Part 3: pre-cast BF16 (K3)", "", "| task | none | bf16p mean | D [CI] | CI contains 0 |",
              "|---|---|---|---|---|"]
    for t in TASKS12:
        c = T[t]["cells"].get("bf16p")
        if c is None:
            continue
        mw = is_mw(t)
        lines.append(f"| {t} | {T[t]['none_mean']:.3f} | {c['mean']:.3f} | {fmt(c['D'], mw)} "
                     f"[{fmt(c['D_ci'][0], mw)}, {fmt(c['D_ci'][1], mw)}] | {c['ci_contains_0']} |")
    lines += ["", "## Part 4: folded per-channel integer (descriptive)", "",
              "| task | k | det D [CI] | sh-iter D [CI] | D(det) - D(sh-iter) [CI] |", "|---|---|---|---|---|"]
    for t in PART2:
        cs = T[t]["cells"]
        mw = is_mw(t)
        for k in PART4_K:
            if f"intpc{k}_det" not in cs:
                continue
            g = lambda nm: (f"{fmt(cs[nm]['D'], mw)} [{fmt(cs[nm]['D_ci'][0], mw)}, {fmt(cs[nm]['D_ci'][1], mw)}]"
                            if nm in cs else "-")
            nm = f"intpc{k}_shiter"
            dl = (f"{fmt(cs[nm]['delta_det_minus_v'], mw)} [{fmt(cs[nm]['delta_det_minus_v_ci'][0], mw)}, "
                  f"{fmt(cs[nm]['delta_det_minus_v_ci'][1], mw)}]" if nm in cs and "delta_det_minus_v" in cs[nm]
                  else "-")
            lines.append(f"| {t} | {k} | {g(f'intpc{k}_det')} | {g(nm)} | {dl} |")
    lines += ["", "## Check 3: FP32 none vs E1 none", "", "| task | n | none (this pod) | E1 none | deviation | within |",
              "|---|---|---|---|---|---|"]
    for t, v in chk3.items():
        lines.append(f"| {t} | {v['n']} | {v['none_mean_all_episodes']:.3f} | {v['e1_none']:.3f} | "
                     f"{v['deviation']:+.4f} | {v['within_10']} |")
    open(f"{a.outdir}/e17_tables.md", "w").write("\n".join(lines) + "\n")
    print(json.dumps({"K2": K2.get("holds"), "K3": K3["holds"], "check3": summary["check3_all_within"],
                      "episodes": len(E), "complete": integ["all_complete"], "problems": len(problems)}))
    print("ANALYZE_E17_DONE")


if __name__ == "__main__":
    main()
