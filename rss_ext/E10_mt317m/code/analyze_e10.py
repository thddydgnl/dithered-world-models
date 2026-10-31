"""E10 analysis (DESIGN_ext sections 0.3 and 3; written before any held / eval result of E10 existed).

Data: <res>/jobs/{held,ksel,eval,perr,ranges}/*.json.gz (e10_jobs.py), <res>/scales/<task>.json.
Rules
- Qualification (3.3): held none success rate over seeds 91000-91029 >= 0.7. Candidates = the 8 E1 Meta-World tasks;
  the fallback list is used only if fewer than 4 qualify (added in list order until 6 qualify or the list ends).
- Primary k (3.4): det on the held seeds at k = 5, 4, 3, 2 in order; the first k with held none rate - held det rate
  >= 0.20 (largest such k); none down to k = 2 -> 'no damage at >= 2 bits' (no eval, excluded from G2).
- Eval (3.5): none, det, sh-iter at the primary k, seeds 90000-90089; pairs = episode seeds (one checkpoint, 90 pairs).
- D_v = success rate(none) - success rate(v) (Meta-World). Paired bootstrap 10,000 resamples of the 90 pairs,
  numpy.random.RandomState(0) (one index matrix, the same for every statistic and every task, as E1's boot_idx),
  percentile 95% CI.
- Drop real (0.3): one-sided bootstrap p = share of resamples with D* <= 0, p < 0.05, Holm over the eval tasks (tasks
  with a primary k); as in E1 also D_det > 0 (implied when p < 0.05). The E1 form (#{D* <= 0} + 1) / (B + 1) is reported.
- red = (D_det - D_sh) / D_det (point estimate decides; CI from the resamples with D_det* > 0).
- Harmful task: 95% CI lower bound of D_sh - D_det > 0.05.
- G1: >= 3 eval tasks with a real det drop (Holm). G2: among those damaged tasks, share with red >= 0.5 is >= 75% and no
  harmful task among the eval tasks (E1 H3a convention: no harmful confirmatory cell in the unit; the variant 'no
  harmful damaged task' is reported too). G3: >= 1 damaged task with pt_err1h(det) <= pt_err1h(sh-iter).
- pt_err1h = mean over the 30 held none episodes of the per-episode planning-time 1-step relative error (E1 H1
  definition, err1h of qlib.pred_errors on the first-iteration candidates of 8 states per episode; sh-iter = mean of 3
  dither seeds). Paired det - sh-iter difference over the held episodes with a bootstrap CI (RandomState(0)) is descriptive.
- Threshold comparisons use a 1e-9 tolerance (near-equality counts as equality).
usage: python analyze_e10.py --res <dir with jobs/ and scales/> --outdir <dir> [--e1 <E1 summary json>]
"""
import argparse
import glob
import gzip
import json
import os

import numpy as np

NB = 10000
ALPHA = 0.05
TOL = 1e-9
QUAL, DROP = 0.7, 0.20
K_ORDER = (5, 4, 3, 2)
HELD_SEEDS = list(range(91000, 91030))
EVAL_SEEDS = list(range(90000, 90090))
CANDIDATES = ["mw-push", "mw-pick-place", "mw-door-open", "mw-drawer-open", "mw-button-press-topdown",
              "mw-peg-insert-side", "mw-window-open", "mw-sweep-into"]
FALLBACK = ["mw-reach", "mw-button-press", "mw-door-close", "mw-drawer-close", "mw-faucet-open", "mw-faucet-close",
            "mw-plate-slide", "mw-coffee-button", "mw-handle-press", "mw-window-close"]
E1_PRIMARY = {"mw-button-press-topdown": 3, "mw-door-open": 4, "mw-drawer-open": 3, "mw-peg-insert-side": 4,
              "mw-pick-place": 3, "mw-push": 3, "mw-sweep-into": 3, "mw-window-open": 4}


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


def load(res):
    """-> episodes[(stage, task, variant, k)] = {seed: record}, perr[(task, k)], ranges[task]"""
    E, P, R, info = {}, {}, {}, {"gpus": set(), "jobs": 0}
    for p in sorted(glob.glob(f"{res}/jobs/*/*.json.gz")):
        d = read(p)
        s = d["spec"]
        info["jobs"] += 1
        info["gpus"].add(d.get("env", {}).get("gpu"))
        if s["type"] == "episodes":
            key = (s["stage"], s["task"], s["variant"], int(s.get("k", 0)))
            for e in d["episodes"]:
                if int(e["seed"]) in E.setdefault(key, {}):
                    raise ValueError(f"duplicate episode {key} {e['seed']}")
                e["_t_start"] = float(d.get("t_start", 0.0))     # job start (unix time): worker-count era
                E[key][int(e["seed"])] = e
        elif s["type"] == "perr":
            P[(s["task"], int(s["k"]))] = d
        elif s["type"] == "ranges":
            R[s["task"]] = d
    info["gpus"] = sorted(g for g in info["gpus"] if g)
    return E, P, R, info


def rate(recs, seeds):
    return float(np.mean([recs[s]["success"] for s in seeds]))


def qualification(E, tasks):
    out = {}
    for t in tasks:
        h = E.get(("held", t, "none", 0), {})
        if sorted(h) != HELD_SEEDS:
            out[t] = {"complete": False, "n": len(h)}
            continue
        r = rate(h, HELD_SEEDS)
        out[t] = {"complete": True, "n": 30, "held_none_success": r, "qualified": bool(ge(r, QUAL)),
                  "success_any_rate": float(np.mean([h[s]["success_any"] for s in HELD_SEEDS])),
                  "act_ms_median": float(np.median([h[s]["act_ms_median"] for s in HELD_SEEDS]))}
    return out


def task_set(Q):
    """Design 3.3: candidates; if fewer than 4 qualify, fallback tasks in list order until 6 qualify / list ends."""
    qual = [t for t in CANDIDATES if Q.get(t, {}).get("qualified")]
    used_fb, pending = [], []
    if all(Q.get(t, {}).get("complete") for t in CANDIDATES) and len(qual) < 4:
        for t in FALLBACK:
            if len(qual) >= 6:
                break
            if not Q.get(t, {}).get("complete"):
                pending.append(t)
                break
            used_fb.append(t)
            if Q[t]["qualified"]:
                qual.append(t)
    return qual, used_fb, pending


def ksel(E, Q, t):
    rows, primary = [], None
    r0 = Q[t]["held_none_success"]
    for k in K_ORDER:
        d = E.get(("ksel", t, "det", k), {})
        if sorted(d) != HELD_SEEDS:
            rows.append({"k": k, "complete": False, "n": len(d)})
            return rows, None, False
        dr = rate(d, HELD_SEEDS)
        drop = r0 - dr
        rows.append({"k": k, "complete": True, "det_success": dr, "drop": drop, "reaches_20pp": bool(ge(drop, DROP))})
        if ge(drop, DROP):
            primary = k
            break
    return rows, primary, True


def eval_task(E, P, t, k, idx):
    Y = {}
    for v, kk in (("none", 0), ("det", k), ("shiter", k)):
        d = E.get(("eval", t, v, kk), {})
        if sorted(d) != EVAL_SEEDS:
            return {"complete": False, "n": {v: len(E.get(("eval", t, vv, kv), {})) for vv, kv in
                                             (("none", 0), ("det", k), ("shiter", k))}}
        Y[v] = np.array([d[s]["success"] for s in EVAL_SEEDS], float)
    yb = {v: Y[v][idx].mean(1) for v in Y}
    m = {v: float(Y[v].mean()) for v in Y}
    Ddet, Dsh = m["none"] - m["det"], m["none"] - m["shiter"]
    Ddet_b, Dsh_b = yb["none"] - yb["det"], yb["none"] - yb["shiter"]
    with np.errstate(divide="ignore", invalid="ignore"):
        rb = np.where(Ddet_b > 0, (Ddet_b - Dsh_b) / np.where(Ddet_b > 0, Ddet_b, 1.0), np.nan)
    S = {v: Y[v].astype(bool) for v in Y}
    c = {"complete": True, "k": k, "n": len(EVAL_SEEDS), "none": m["none"], "det": m["det"], "shiter": m["shiter"],
         "D_det": Ddet, "D_det_ci": ci95(Ddet_b), "p_det": float(np.mean(Ddet_b <= 0)),
         "p_det_e1form": float((np.sum(Ddet_b <= 0) + 1) / (NB + 1)),
         "D_sh": Dsh, "D_sh_ci": ci95(Dsh_b), "p_sh": float(np.mean(Dsh_b <= 0)),
         "red_point": float((Ddet - Dsh) / Ddet) if Ddet > 0 else None, "red_ci": ci95(rb),
         "Dsh_minus_Ddet": float(Dsh - Ddet), "Dsh_minus_Ddet_ci": ci95(Dsh_b - Ddet_b),
         "mcnemar_none_det": mcnemar(S["none"], S["det"]), "mcnemar_none_sh": mcnemar(S["none"], S["shiter"]),
         "mcnemar_sh_det": mcnemar(S["shiter"], S["det"])}
    c["harmful"] = bool(gt(c["Dsh_minus_Ddet_ci"][0], 0.05))
    for v, kk in (("none", 0), ("det", k), ("shiter", k)):
        d = E[("eval", t, v, kk)]
        c[f"act_ms_median_{v}"] = float(np.median([d[s]["act_ms_median"] for s in EVAL_SEEDS]))
        c[f"nq_{v}"] = sorted({int(d[s]["nq"]) for s in EVAL_SEEDS})
        c[f"steps_{v}"] = sorted({int(d[s]["steps"]) for s in EVAL_SEEDS})
    pe = P.get((t, k))
    if pe:
        rows = pe["episodes"]
        c["pt_err1h_det"] = pe["summary"]["det"]["err1h"]
        c["pt_err1h_sh"] = pe["summary"]["shiter"]["err1h"]
        for m_ in ("err1", "err3", "err5", "errr"):
            c[f"pt_{m_}_det"] = pe["summary"]["det"][m_]
            c[f"pt_{m_}_sh"] = pe["summary"]["shiter"][m_]
        dd = np.array([r["det"]["err1h"] - r["shiter"]["err1h"] for r in rows], float)
        ib = boot_idx(len(dd))
        c["pt_err1h_det_minus_sh"] = float(dd.mean())
        c["pt_err1h_det_minus_sh_ci"] = ci95(dd[ib].mean(1))
        c["pt_n_episodes"] = len(rows)
        c["pt_n_seq"] = int(sum(r["n_seq"] for r in rows))
    return c


def judge(C):
    ev = [t for t, c in C.items() if c.get("complete")]
    ps = [C[t]["p_det"] for t in ev]
    rej = holm_lt(ps, ALPHA) if ps else []
    for t, r in zip(ev, rej):
        C[t]["holm_m"] = len(ev)
        C[t]["drop_real"] = bool(r and C[t]["D_det"] > 0)
    dmg = [t for t in ev if C[t]["drop_real"]]
    red_ok = [t for t in dmg if C[t]["red_point"] is not None and ge(C[t]["red_point"], 0.5)]
    harm_all = [t for t in ev if C[t]["harmful"]]
    harm_dmg = [t for t in dmg if C[t]["harmful"]]
    share = len(red_ok) / len(dmg) if dmg else None
    g3 = [t for t in dmg if C[t].get("pt_err1h_det") is not None and le(C[t]["pt_err1h_det"], C[t]["pt_err1h_sh"])]
    return {
        "eval_tasks": ev, "holm_alpha": ALPHA, "damaged": dmg,
        "G1": {"holds": len(dmg) >= 3, "n_damaged": len(dmg), "threshold": 3},
        "G2": {"holds": (bool(dmg) and ge(share, 0.75) and not harm_all) if dmg else None,
               "defined": bool(dmg), "share_red_ge_half": share, "tasks_red_ge_half": red_ok,
               "harmful_eval_tasks": harm_all, "harmful_damaged_tasks": harm_dmg,
               "holds_if_harm_only_among_damaged": (bool(dmg) and ge(share, 0.75) and not harm_dmg) if dmg else None},
        "G3": {"holds": len(g3) >= 1 if dmg else None, "tasks_det_le_sh": g3, "n": len(g3), "n_damaged": len(dmg)}}


def fmt(x, nd=4):
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        return "[" + ";".join(fmt(y, nd) for y in x) + "]"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--e1", default="")
    ap.add_argument("--w1_since", default="", help="UTC time from which every driver ran one worker (latency summary)")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    E, P, R, info = load(a.res)
    tasks_seen = sorted({k[1] for k in E}, key=lambda t: (CANDIDATES + FALLBACK).index(t)
                        if t in CANDIDATES + FALLBACK else 99)
    Q = qualification(E, CANDIDATES + [t for t in FALLBACK if t in tasks_seen])
    qual, used_fb, pending_fb = task_set(Q)
    K, C = {}, {}
    idx = boot_idx(len(EVAL_SEEDS))
    for t in qual:
        rows, primary, complete = ksel(E, Q, t)
        K[t] = {"rows": rows, "primary_k": primary, "complete": complete, "e1_5M_primary_k": E1_PRIMARY.get(t),
                "no_damage_ge2": bool(complete and primary is None)}
        if primary is not None:
            C[t] = eval_task(E, P, t, primary, idx)
    J = judge(C)
    e1 = {}
    if a.e1 and os.path.exists(a.e1):
        S1 = json.load(open(a.e1))
        for t, kk in E1_PRIMARY.items():
            c1 = S1["cells"].get(f"{t}:{kk}")
            if c1:
                e1[t] = {"k": kk, "none": c1["none_mean"], "D_det": c1["variants"]["det"]["D"],
                         "D_sh": c1["variants"]["shiter"]["D"], "red": c1["variants"]["shiter"].get("red"),
                         "pt_err1h_det": c1["variants"]["det"].get("pt_err1h"),
                         "pt_err1h_sh": c1["variants"]["shiter"].get("pt_err1h")}
    ranges = {t: {"quant": d["quant"], "info": d["info"]} for t, d in R.items()}
    # descriptive: per-decision planning latency in closed-loop episodes run by a single worker process (the drivers
    # ran 2 workers sharing the GPU until --w1_since; those episodes' latency includes the other process' share)
    lat = {}
    if a.w1_since:
        import calendar
        import time as _time
        cut = calendar.timegm(_time.strptime(a.w1_since, "%Y-%m-%dT%H:%M:%SZ"))
        for (stage, t, v, k), d in E.items():
            if stage != "eval":
                continue
            x = [e["act_ms_median"] for e in d.values() if e["_t_start"] >= cut]
            if x:
                lat.setdefault(v, []).extend(x)
        lat = {v: {"median_ms": float(np.median(x)), "q25": float(np.percentile(x, 25)),
                   "q75": float(np.percentile(x, 75)), "n_episodes": len(x)} for v, x in lat.items()}
    summary = {"n_jobs": info["jobs"], "gpus": info["gpus"], "qualification": Q, "qualified": qual,
               "fallback_used": used_fb, "fallback_pending": pending_fb, "ksel": K, "eval": C, "judgement": J,
               "e1_5M_primary_cells": e1, "ranges": ranges, "latency_single_worker_eval": lat,
               "w1_since": a.w1_since}
    json.dump(summary, open(f"{a.outdir}/e10_summary.json", "w"), indent=1, default=float)
    with open(f"{a.outdir}/e10_qualification.csv", "w") as f:
        f.write("task,role,n,held_none_success,qualified,success_any_rate,act_ms_median\n")
        for t, q in Q.items():
            role = "candidate" if t in CANDIDATES else "fallback"
            f.write(f"{t},{role},{q.get('n')},{fmt(q.get('held_none_success'))},{q.get('qualified')},"
                    f"{fmt(q.get('success_any_rate'))},{fmt(q.get('act_ms_median'))}\n")
    with open(f"{a.outdir}/e10_ksel.csv", "w") as f:
        f.write("task,k,det_held_success,drop_vs_none,reaches_20pp,primary_k,e1_5M_primary_k\n")
        for t, kk in K.items():
            for r in kk["rows"]:
                f.write(f"{t},{r['k']},{fmt(r.get('det_success'))},{fmt(r.get('drop'))},{r.get('reaches_20pp')},"
                        f"{kk['primary_k']},{kk['e1_5M_primary_k']}\n")
    cols = ["k", "n", "none", "det", "shiter", "D_det", "D_det_ci", "p_det", "p_det_e1form", "holm_m", "drop_real",
            "D_sh", "D_sh_ci", "red_point", "red_ci", "Dsh_minus_Ddet", "Dsh_minus_Ddet_ci", "harmful",
            "pt_err1h_det", "pt_err1h_sh", "pt_err1h_det_minus_sh", "pt_err1h_det_minus_sh_ci", "pt_err1_det",
            "pt_err1_sh", "pt_err3_det", "pt_err3_sh", "pt_errr_det", "pt_errr_sh", "mcnemar_none_det",
            "mcnemar_none_sh", "mcnemar_sh_det", "act_ms_median_none", "act_ms_median_det", "act_ms_median_shiter"]
    with open(f"{a.outdir}/e10_eval.csv", "w") as f:
        f.write("task," + ",".join(cols) + "\n")
        for t, c in C.items():
            f.write(t + "," + ",".join(fmt(c.get(col)) for col in cols) + "\n")
    # flat exports (every judged number can be recomputed from these two files)
    with open(f"{a.outdir}/e10_episodes.csv", "w") as f:
        f.write("stage,task,variant,k,seed,success,success_any,ret,steps,nq,act_ms_median\n")
        for (stage, t, v, k), d in sorted(E.items()):
            for s in sorted(d):
                e = d[s]
                f.write(f"{stage},{t},{v},{k},{s},{e['success']},{e['success_any']},{e['ret']:.6f},{e['steps']},"
                        f"{e['nq']},{e['act_ms_median']:.3f}\n")
    with open(f"{a.outdir}/e10_perr_episodes.csv", "w") as f:
        f.write("task,k,held_seed,n_seq,variant,err1,err3,err5,err1h,errr,nseeds\n")
        for (t, k), d in sorted(P.items()):
            for r in d["episodes"]:
                for v in ("det", "shiter"):
                    x = r[v]
                    f.write(f"{t},{k},{r['seed']},{r['n_seq']},{v},{x['err1']:.6g},{x['err3']:.6g},{x['err5']:.6g},"
                            f"{x['err1h']:.6g},{x['errr']:.6g},{x.get('nseeds', '')}\n")
    print(json.dumps({"qualified": qual, "fallback_used": used_fb, "ksel": {t: v["primary_k"] for t, v in K.items()},
                      "judgement": J}, indent=1, default=float))


if __name__ == "__main__":
    main()
