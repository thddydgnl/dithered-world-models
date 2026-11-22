"""E14 analysis (DESIGN_tmlr 0.3 and 4; written before any E14 eval result existed, dry-run on fake data).

Data: one or more result directories (one per pod, e.g. results/raw/mt.1,results/raw/mt.2), each with
jobs/{held,bias,eval}/*.json.gz (mt14_jobs.py) and gate/check3_<task>.json.
Rules (statistics as in E10's analyze_e10.py, DESIGN_tmlr 0.3)
- Cells: the 8 E10 Meta-World tasks at the E10 primary k with the E10 ranges (R4); variants none, det, sh-iter, sh-ep,
  det+shift, det+BC; pairs = eval episode seeds 102000-102089 (one checkpoint, 90 pairs per variant).
- D_v = success rate(none) - success rate(v). Paired bootstrap 10,000 resamples of the 90 pairs,
  numpy.random.RandomState(0) (one index matrix for every statistic and every task, E1/E10 convention), percentile 95%.
- det drop real: one-sided bootstrap p = share of resamples with D_det* <= 0, p < 0.05 with Holm over the 8 tasks
  (strict <), and D_det > 0. The E1 form (# + 1) / (B + 1) is reported too.
- red_v = (D_det - D_v) / D_det (point estimate decides; CI from the resamples with D_det* > 0).
- harmful_v: 95% CI lower bound of D_v - D_det > 0.05.
- P1 (E10 replication): >= 6 of the 8 tasks with a real det drop, and red_sh-iter >= 0.5 in >= 75% of those tasks.
- P2: in a majority (more than half) of the damaged tasks, the 95% CI lower bound of D(sh-ep) - D(sh-iter) > 0.
  No damaged task -> P2 undefined.
- Descriptive: per task r = D(sh-ep) / D(det) with CI (resamples with D_det* > 0), red and harmful of every variant
  (det+BC and det+shift in particular), D differences with CIs, McNemar tests, check 0.7-3 (none vs E10 none), det+BC
  bias sweeps (check 0.7-2), integrity (GPU, steps, nq, sh-ep table refreshes, global RNG state equal across variants).
- Threshold comparisons use a 1e-9 tolerance.
- Deviation D2 (2026-11-21T19:01Z, DESIGN_tmlr 11): b by the staged rule (alpha 1 -> 1/2 -> 1/4 from b = 0, every layer
  below 1% of delta, at most 60 updates per stage); a task whose stages all fail uses the best b and is flagged
  bc_unconverged; its det+BC still runs. Every det+BC descriptive item is reported (a) over all tasks and (b) without
  the flagged tasks. Only D2 bias jobs are read (results of the frozen 3-sweep rule were moved out of jobs/).
usage: python analyze_e14.py --res <dir1>[,<dir2>...] --outdir <dir> [--e10_eval <e10_eval.csv>]
"""
import argparse
import csv
import glob
import gzip
import json
import os

import numpy as np

NB = 10000
ALPHA = 0.05
TOL = 1e-9
TASKS = ["mw-push", "mw-pick-place", "mw-door-open", "mw-drawer-open", "mw-button-press-topdown",
         "mw-peg-insert-side", "mw-window-open", "mw-sweep-into"]
PRIMARY_K = {"mw-push": 5, "mw-pick-place": 3, "mw-door-open": 3, "mw-drawer-open": 4,
             "mw-button-press-topdown": 3, "mw-peg-insert-side": 3, "mw-window-open": 3, "mw-sweep-into": 3}
EVAL_SEEDS = list(range(102000, 102090))
HELD_SEEDS = list(range(103000, 103010))
VARIANTS = ("none", "det", "shiter", "shep", "detshift", "detbc")
QV = VARIANTS[1:]
CMP = ("shiter", "shep", "detshift", "detbc")         # compared with det
CONTRASTS = (("shep", "shiter"), ("detbc", "det"), ("detshift", "shiter"), ("shep", "det"), ("detshift", "det"))
P1_MIN_DAMAGED, P1_SHARE, RED_FIX = 6, 0.75, 0.5
CHECK3_MAX = 0.10
NQ_EXPECTED = 7600                                     # 100 steps x 76 quantized layer calls (E10)


def ge(a, b):
    return a >= b - TOL


def gt(a, b):
    return a > b + TOL


def lt(a, b):
    return a < b - TOL


def le(a, b):
    return a <= b + TOL


def ci95(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def boot_idx(n, seed=0):
    return np.random.RandomState(seed).randint(0, n, size=(NB, n))


def holm_lt(pvals, alpha):
    """Holm step-down with the strict '<' of DESIGN_ext 0.3; reject flags in input order."""
    order = np.argsort(pvals, kind="stable")
    m = len(pvals)
    rej = [False] * m
    for r, i in enumerate(order):
        if lt(pvals[i], alpha / (m - r)):
            rej[i] = True
        else:
            break
    return rej


def mcnemar(a, b):
    from scipy.stats import binomtest
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    p = 1.0 if n10 + n01 == 0 else float(binomtest(n10, n10 + n01, 0.5).pvalue)
    return n10, n01, p


def read(p):
    with gzip.open(p, "rt") as f:
        return json.load(f)


def load(res_dirs):
    """-> E[(stage, task, variant, k)] = {seed: record}, Bias[(task, k)] = job, info. A job present in several
    directories (continuation pod with a copy of the results) must be identical in its episodes."""
    E, Bias, info = {}, {}, {"jobs": 0, "gpus_by_task": {}, "pods_by_task": {}, "dirs": res_dirs, "job_ids": {}}
    for rd in res_dirs:
        for p in sorted(glob.glob(f"{rd}/jobs/*/*.json.gz")):
            d = read(p)
            s = d["spec"]
            jid = s["id"]
            if jid in info["job_ids"]:
                prev = info["job_ids"][jid]
                if s["type"] == "episodes":
                    a = [(e["seed"], e["success"], e["ret"], e.get("act_sha1")) for e in d["episodes"]]
                    if a != prev:
                        raise ValueError(f"job {jid} differs between result directories")
                continue
            info["job_ids"][jid] = ([(e["seed"], e["success"], e["ret"], e.get("act_sha1")) for e in d["episodes"]]
                                    if s["type"] == "episodes" else True)
            info["jobs"] += 1
            t = s["task"]
            info["gpus_by_task"].setdefault(t, set()).add(d.get("env", {}).get("gpu"))
            info["pods_by_task"].setdefault(t, set()).add(os.path.basename(os.path.normpath(rd)))
            if s["type"] == "episodes":
                key = (s["stage"], t, s["variant"], int(s.get("k", 0)))
                for e in d["episodes"]:
                    if int(e["seed"]) in E.setdefault(key, {}):
                        raise ValueError(f"duplicate episode {key} {e['seed']}")
                    E[key][int(e["seed"])] = e
            elif s["type"] == "bias":
                Bias[(t, int(s["k"]))] = d
    info["gpus_by_task"] = {t: sorted(g for g in v if g) for t, v in info["gpus_by_task"].items()}
    info["pods_by_task"] = {t: sorted(v) for t, v in info["pods_by_task"].items()}
    del info["job_ids"]
    return E, Bias, info


def e10_none(path):
    """E10 eval none rates: full precision from inputs/e10_eval_none_exact.json next to the given CSV when present."""
    exact = os.path.join(os.path.dirname(path or "."), "e10_eval_none_exact.json")
    if os.path.exists(exact):
        d = json.load(open(exact))
        return {t: float(v["none"]) for t, v in d.items() if not t.startswith("_")}
    out = {}
    if path and os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                out[r["task"]] = float(r["none"])
    return out


def eval_task(E, t, k, idx):
    Y, recs = {}, {}
    n = {}
    for v in VARIANTS:
        kk = 0 if v == "none" else k
        d = E.get(("eval", t, v, kk), {})
        n[v] = len(d)
        if sorted(d) == EVAL_SEEDS:
            Y[v] = np.array([d[s]["success"] for s in EVAL_SEEDS], float)
            recs[v] = d
    c = {"k": k, "n": n, "complete_variants": sorted(Y), "complete": len(Y) == len(VARIANTS)}
    if "none" not in Y or "det" not in Y:
        return c
    yb = {v: Y[v][idx].mean(1) for v in Y}
    m = {v: float(Y[v].mean()) for v in Y}
    D = {v: m["none"] - m[v] for v in Y if v != "none"}
    Db = {v: yb["none"] - yb[v] for v in Y if v != "none"}
    c["rate"] = m
    c["D"] = D
    c["D_ci"] = {v: ci95(Db[v]) for v in Db}
    c["p_det"] = float(np.mean(Db["det"] <= 0))
    c["p_det_e1form"] = float((np.sum(Db["det"] <= 0) + 1) / (NB + 1))
    Dd, Ddb = D["det"], Db["det"]
    pos = Ddb > 0
    c["red"], c["red_ci"], c["diff_vs_det"], c["diff_vs_det_ci"], c["harmful"] = {}, {}, {}, {}, {}
    for v in CMP:
        if v not in Y:
            continue
        c["red"][v] = float((Dd - D[v]) / Dd) if Dd > 0 else None
        with np.errstate(divide="ignore", invalid="ignore"):
            rb = np.where(pos, (Ddb - Db[v]) / np.where(pos, Ddb, 1.0), np.nan)
        c["red_ci"][v] = ci95(rb)
        c["diff_vs_det"][v] = float(D[v] - Dd)
        c["diff_vs_det_ci"][v] = ci95(Db[v] - Ddb)
        c["harmful"][v] = bool(gt(c["diff_vs_det_ci"][v][0], 0.05))
    if "shep" in Y:
        c["r_shep_over_det"] = float(D["shep"] / Dd) if Dd > 0 else None
        with np.errstate(divide="ignore", invalid="ignore"):
            rr = np.where(pos, Db["shep"] / np.where(pos, Ddb, 1.0), np.nan)
        c["r_shep_over_det_ci"] = ci95(rr)
    c["contrast"], c["contrast_ci"] = {}, {}
    for a_, b_ in CONTRASTS:
        if a_ in Y and b_ in Y:
            key = f"D_{a_}-D_{b_}"
            c["contrast"][key] = float(D[a_] - D[b_])
            c["contrast_ci"][key] = ci95(Db[a_] - Db[b_])
    S = {v: Y[v].astype(bool) for v in Y}
    c["mcnemar"] = {f"none_vs_{v}": mcnemar(S["none"], S[v]) for v in Y if v != "none"}
    if "shep" in S and "shiter" in S:
        c["mcnemar"]["shiter_vs_shep"] = mcnemar(S["shiter"], S["shep"])
    # integrity of the closed loop
    integ = {}
    for v, d in recs.items():
        integ[v] = {"steps": sorted({int(d[s]["steps"]) for s in EVAL_SEEDS}),
                    "nq": sorted({int(d[s]["nq"]) for s in EVAL_SEEDS}),
                    "n_refresh": sorted({int(d[s].get("n_refresh", -1)) for s in EVAL_SEEDS}),
                    "act_ms_median": float(np.median([d[s]["act_ms_median"] for s in EVAL_SEEDS])),
                    "ep_secs_median": float(np.median([d[s]["secs"] for s in EVAL_SEEDS]))}
    rng_same = None
    if all("rng_sha1" in recs[v][s] for v in recs for s in EVAL_SEEDS):
        rng_same = all(len({recs[v][s]["rng_sha1"] for v in recs}) == 1 for s in EVAL_SEEDS)
    c["integrity"] = {"by_variant": integ, "global_rng_end_state_equal_across_variants": rng_same,
                      "nq_ok": all(integ[v]["nq"] == ([0] if v == "none" else [NQ_EXPECTED]) for v in integ),
                      "steps_ok": all(integ[v]["steps"] == [100] for v in integ),
                      "shep_one_table_per_episode": (integ["shep"]["n_refresh"] == [1]) if "shep" in integ else None}
    return c


def judge(C, flagged=()):
    ev = [t for t in TASKS if C.get(t, {}).get("complete")]
    incomplete = [t for t in TASKS if t not in ev]
    ps = [C[t]["p_det"] for t in ev]
    rej = holm_lt(ps, ALPHA) if ps else []
    for t, r in zip(ev, rej):
        C[t]["holm_m"] = len(ev)
        C[t]["drop_real"] = bool(r and C[t]["D"]["det"] > 0)
    dmg = [t for t in ev if C[t]["drop_real"]]
    red_ok = [t for t in dmg if C[t]["red"]["shiter"] is not None and ge(C[t]["red"]["shiter"], RED_FIX)]
    share = len(red_ok) / len(dmg) if dmg else None
    p1 = bool(len(dmg) >= P1_MIN_DAMAGED and share is not None and ge(share, P1_SHARE))
    p2_tasks = [t for t in dmg if gt(C[t]["contrast_ci"]["D_shep-D_shiter"][0], 0.0)]
    p2 = (2 * len(p2_tasks) > len(dmg)) if dmg else None
    harm = {v: [t for t in ev if C[t]["harmful"].get(v)] for v in CMP}
    fix = {v: [t for t in dmg if C[t]["red"].get(v) is not None and ge(C[t]["red"][v], RED_FIX)] for v in CMP}
    return {
        "eval_tasks": ev, "incomplete_tasks": incomplete, "complete": not incomplete, "holm_alpha": ALPHA,
        "holm_m": len(ev), "damaged": dmg,
        "P1": {"holds": p1 if not incomplete else None, "n_damaged": len(dmg), "min_damaged": P1_MIN_DAMAGED,
               "share_red_shiter_ge_half": share, "tasks_red_shiter_ge_half": red_ok, "share_threshold": P1_SHARE,
               "harmful_shiter_tasks": harm["shiter"]},
        "P2": {"holds": (p2 if not incomplete else None), "defined": bool(dmg), "n_damaged": len(dmg),
               "tasks_ci_lower_gt0": p2_tasks, "n": len(p2_tasks),
               "rule": "more than half of the damaged tasks with CI lower bound of D(sh-ep) - D(sh-iter) > 0"},
        "descriptive": {"fixes_red_ge_half_among_damaged": fix, "harmful_eval_tasks": harm,
                        "detbc_two_ways": {
                            "bc_unconverged_tasks": sorted(flagged),
                            "a_all_tasks": {"fixes_among_damaged": fix["detbc"], "n_damaged": len(dmg),
                                            "harmful_eval_tasks": harm["detbc"], "n_eval": len(ev)},
                            "b_without_flagged": {
                                "fixes_among_damaged": [t for t in fix["detbc"] if t not in flagged],
                                "n_damaged": len([t for t in dmg if t not in flagged]),
                                "harmful_eval_tasks": [t for t in harm["detbc"] if t not in flagged],
                                "n_eval": len([t for t in ev if t not in flagged])}}}}


def fmt(x, nd=4):
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        return "[" + ";".join(fmt(y, nd) for y in x) + "]"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (float, np.floating)):
        return f"{x:.{nd}g}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True, help="comma-separated result directories (one per pod)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--e10_eval", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs",
                                                        "e10_eval.csv"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    res_dirs = [d for d in a.res.split(",") if d]
    E, Bias, info = load(res_dirs)
    idx = boot_idx(len(EVAL_SEEDS))
    ref = e10_none(a.e10_eval)
    C = {t: eval_task(E, t, PRIMARY_K[t], idx) for t in TASKS}
    flagged = sorted(t for (t, k), d in Bias.items() if d.get("rule_id") == "D2" and d.get("bc_unconverged"))
    for t in TASKS:
        C[t]["bc_unconverged"] = t in flagged
    J = judge(C, flagged)
    # check 0.7-3: none success on the new seeds vs the E10 eval none (90 episodes each)
    chk3 = {}
    for t in TASKS:
        r = C[t].get("rate", {}).get("none")
        if r is not None and t in ref:
            chk3[t] = {"none_new": r, "none_e10": ref[t], "diff": r - ref[t], "pass": bool(le(abs(r - ref[t]), CHECK3_MAX))}
    # held none (b calibration) and the det+BC bias sweeps (check 0.7-2)
    held = {}
    for t in TASKS:
        h = E.get(("held", t, "none", 0), {})
        held[t] = {"n": len(h), "complete": sorted(h) == HELD_SEEDS,
                   "success": float(np.mean([h[s]["success"] for s in sorted(h)])) if h else None}
    bias = {}
    for (t, k), d in Bias.items():
        if d.get("rule_id") != "D2":
            bias[t] = {"k": k, "rule": "frozen 0.6 (superseded by D2; not used)", "residual_ok": d["residual_ok"]}
            continue
        bias[t] = {"k": k, "rule": "D2", "n_rows": d["n_rows"], "converged": d["converged"],
                   "bc_unconverged": d["bc_unconverged"], "final_eta": d["final_eta"],
                   "final_n_updates": d["final_n_updates"], "residual_ok": d["residual_ok"],
                   "residual_final": d["residual_final"], "residual_final_max": d["residual_final_max"],
                   "b_mean_abs_over_delta": d["b_mean_abs_over_delta"],
                   "mse_over_delta2_b0": d["mse_over_delta2_b0"], "mse_over_delta2_final": d["mse_over_delta2_final"],
                   "latent_err_b0": d["latent_err_b0"], "latent_err_final": d["latent_err_final"],
                   "stages": [{"eta": x["eta"], "converged": x["converged"], "n_updates": x["n_updates"],
                               "worst_r_last": x["worst_r_last"],
                               "passes": [{"pass": q["pass"], "r_mean": q["r_mean"], "r_max": q["r_max"],
                                           "mse_over_delta2": q["mse_over_delta2"]} for q in x["passes"]]}
                              for x in d["stages"]]}
    summary = {"n_jobs": info["jobs"], "gpus_by_task": info["gpus_by_task"], "pods_by_task": info["pods_by_task"],
               "result_dirs": res_dirs, "eval": C, "judgement": J, "check3": chk3, "held": held, "bias": bias,
               "bc_unconverged_tasks": flagged,
               "one_gpu_type_per_task": all(len(v) == 1 for v in info["gpus_by_task"].values())}
    json.dump(summary, open(f"{a.outdir}/e14_summary.json", "w"), indent=1, default=float)
    json.dump(J, open(f"{a.outdir}/e14_judgement.json", "w"), indent=1, default=float)
    # per-task table (one row per task)
    cols = ["k", "complete", "bc_unconverged", "rate_none", "rate_det", "rate_shiter", "rate_shep", "rate_detshift", "rate_detbc",
            "D_det", "D_det_ci", "p_det", "p_det_e1form", "holm_m", "drop_real",
            "D_shiter", "D_shiter_ci", "red_shiter", "red_shiter_ci", "harmful_shiter",
            "D_shep", "D_shep_ci", "red_shep", "red_shep_ci", "harmful_shep", "r_shep_over_det", "r_shep_over_det_ci",
            "D_shep-D_shiter", "D_shep-D_shiter_ci",
            "D_detshift", "D_detshift_ci", "red_detshift", "red_detshift_ci", "harmful_detshift",
            "D_detbc", "D_detbc_ci", "red_detbc", "red_detbc_ci", "harmful_detbc",
            "D_detbc-D_det", "D_detbc-D_det_ci", "D_detshift-D_shiter", "D_detshift-D_shiter_ci"]

    def val(c, col):
        if col in ("k", "complete", "bc_unconverged", "p_det", "p_det_e1form", "holm_m", "drop_real",
                   "r_shep_over_det", "r_shep_over_det_ci"):
            return c.get(col)
        if col.startswith("rate_"):
            return c.get("rate", {}).get(col[5:])
        for pre, key in (("red_", "red"), ("harmful_", "harmful")):
            if col.startswith(pre):
                v = col[len(pre):]
                if v.endswith("_ci"):
                    return c.get("red_ci", {}).get(v[:-3])
                return c.get(key, {}).get(v)
        if col.startswith("D_") and "-" in col:
            if col.endswith("_ci"):
                return c.get("contrast_ci", {}).get(col[:-3])
            return c.get("contrast", {}).get(col)
        if col.startswith("D_"):
            v = col[2:]
            if v.endswith("_ci"):
                return c.get("D_ci", {}).get(v[:-3])
            return c.get("D", {}).get(v)
        return None

    with open(f"{a.outdir}/e14_eval.csv", "w") as f:
        f.write("task," + ",".join(cols) + "\n")
        for t in TASKS:
            f.write(t + "," + ",".join(fmt(val(C[t], col)) for col in cols) + "\n")
    with open(f"{a.outdir}/e14_check3.csv", "w") as f:
        f.write("task,none_new,none_e10,diff,pass\n")
        for t, x in chk3.items():
            f.write(f"{t},{fmt(x['none_new'])},{fmt(x['none_e10'])},{fmt(x['diff'])},{x['pass']}\n")
    with open(f"{a.outdir}/e14_bias.csv", "w") as f:
        f.write("task,k,layer,final_eta,final_n_updates,converged,bc_unconverged,resid_mean_over_delta,"
                "resid_max_over_delta,b_mean_abs_over_delta,mse_over_delta2_b0,mse_over_delta2_final\n")
        for t, x in bias.items():
            if x.get("rule") != "D2":
                continue
            for n in x["residual_final"]:
                f.write(f"{t},{x['k']},{n},{x['final_eta']},{x['final_n_updates']},{x['converged']},"
                        f"{x['bc_unconverged']},{x['residual_final'][n]:.6g},{x['residual_final_max'][n]:.6g},"
                        f"{x['b_mean_abs_over_delta'][n]:.6g},{x['mse_over_delta2_b0'][n]:.6g},"
                        f"{x['mse_over_delta2_final'][n]:.6g}\n")
    with open(f"{a.outdir}/e14_bias_passes.csv", "w") as f:
        f.write("task,k,stage_eta,stage_converged,pass,layer,r_mean,r_max,mse_over_delta2\n")
        for t, x in bias.items():
            for stg in x.get("stages", []):
                for q in stg["passes"]:
                    for n in q["r_mean"]:
                        f.write(f"{t},{x['k']},{stg['eta']},{stg['converged']},{q['pass']},{n},{q['r_mean'][n]:.6g},"
                                f"{q['r_max'][n]:.6g},{q['mse_over_delta2'][n]:.6g}\n")
    # flat episode export (every judged number can be recomputed from it)
    with open(f"{a.outdir}/e14_episodes.csv", "w") as f:
        f.write("stage,task,variant,k,seed,success,success_any,ret,steps,nq,n_refresh,act_ms_median,act_sha1,rng_sha1\n")
        for (stage, t, v, k), d in sorted(E.items()):
            for s in sorted(d):
                e = d[s]
                f.write(f"{stage},{t},{v},{k},{s},{e['success']},{e['success_any']},{e['ret']:.6f},{e['steps']},"
                        f"{e['nq']},{e.get('n_refresh', '')},{e['act_ms_median']:.3f},{e.get('act_sha1', '')},"
                        f"{e.get('rng_sha1', '')}\n")
    print(json.dumps({"judgement": J, "check3": chk3, "bc_unconverged": flagged,
                      "bias_ok": {t: x["residual_ok"] for t, x in bias.items()}}, indent=1, default=float))


if __name__ == "__main__":
    main()
