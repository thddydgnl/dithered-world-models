"""E9 analysis (rss_ext/DESIGN_ext.md 0.3 and 2.5; statistics copied from E1's analyze_e1.py).

Data: <res>/jobs/episodes/eval_<task>_s<ckpt>_<variant>_cXX.json.gz (closed-loop episodes, seeds 80000-80029) and
      <res>/jobs/heldperr/heldperr_<task>_s<ckpt>_cXX.json.gz (held none episodes 81000-81029 with the per-variant
      planning-time errors).
Rules (written before any eval result was read):
- cell = (task, format), format in {nvfp4, mxfp4}: 24 cells; variants det and shiter of the format; none is the same
  90 episodes for both formats of a task. Pairs = (checkpoint, episode seed), 90 per cell.
- D_v: Meta-World success-rate difference none - v; DMC relative return drop (mean R_none - mean R_v) / mean R_none,
  recomputed inside every resample. Paired bootstrap: 10,000 resamples of the pairs, RandomState(0) index matrix
  shared by every statistic of a cell (E1 analyze.boot), percentile 95% CI.
- drop real (det): one-sided p = (#{D* <= 0} + 1) / (B + 1) (DESIGN_main v4 6.1), Holm over the 12 tasks of the format
  at alpha = 0.05 (DESIGN_ext 0.3), and D_det > 0 (as E1). The literal fraction #{D* <= 0} / B is also reported.
- red = (D_det - D_shiter) / D_det (point estimate; CI from the resamples with D_det* > 0, as E1).
- harmful cell: 95% CI lower bound of D_shiter - D_det > 0.05 (+1e-9 tolerance in every threshold comparison).
- F1 (per format): >= 3 damaged cells (Holm) among the format's 12 tasks.
- F2 (Meta-World and DMC separately, both formats pooled): among the damaged cells, share with red >= 0.5 is >= 0.75,
  and no harmful cell. Harmful cells are checked over every cell of the unit (E1 H3a convention: MW 16 cells, DMC 8
  cells; primary); the reading "no harmful cell among the damaged cells" is reported too. The claim follows MW.
- F3: >= 1 damaged cell with pt_err1h(det) <= pt_err1h(shiter); pt_err1h = mean over the 90 held episodes of
  perr_plan.err1h (E1 definition). The count is reported per unit and format. Dataset-type (perr_traj) auxiliary.
- descriptive: comparison with E1 uniform k = 4 det damage (main/results/E1_tdmpc2/e1_tdmpc2_cells.csv, other seeds,
  unpaired), NVFP4 - MXFP4 paired differences of D_det and D_shiter, McNemar exact p (MW success, auxiliary).
Usage: python analyze_e9.py --res <results dir with jobs/> --e1 <e1_tdmpc2_cells.csv> --outdir <dir>
"""
import argparse
import csv
import glob
import gzip
import json
import os
import re

import numpy as np

NB = 10000
ALPHA = 0.05
TOL = 1e-9
FMTS = ["nvfp4", "mxfp4"]
MW8 = ["mw-push", "mw-pick-place", "mw-door-open", "mw-drawer-open", "mw-button-press-topdown",
       "mw-peg-insert-side", "mw-window-open", "mw-sweep-into"]
DMC4 = ["cheetah-run", "walker-walk", "quadruped-walk", "finger-turn-hard"]
TASKS = MW8 + DMC4
EP_RE = re.compile(r"^eval_(?P<task>.+)_s(?P<ckpt>\d)_(?P<v>none|nvfp4_det|nvfp4_shiter|mxfp4_det|mxfp4_shiter)_c\d+$")
ERR_KEYS = ["err1h", "err1", "err3", "err5", "errr"]


def is_mw(t):
    return t.startswith("mw-")


def holm(pvals, alpha):
    """Holm step-down; reject flags in input order (analyze_e1.holm)."""
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
    return np.random.RandomState(seed).randint(0, n, size=(NB, n))


def ci95(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def p_le0(v):
    v = np.asarray(v, float)
    return float((np.sum(v <= 0) + 1) / (NB + 1))


def mcnemar(a, b):
    from scipy.stats import binomtest
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    p = 1.0 if n10 + n01 == 0 else float(binomtest(n10, n10 + n01, 0.5).pvalue)
    return n10, n01, p


# ---------------------------------------------------------------- loading
def load(res):
    E, meta = {}, {"gpus": {}, "impl": {}, "files": 0}
    for f in sorted(glob.glob(f"{res}/jobs/episodes/eval_*.json.gz")):
        m = EP_RE.match(os.path.basename(f)[:-len(".json.gz")])
        if not m:
            continue
        d = json.load(gzip.open(f, "rt"))
        s = d["spec"]
        meta["files"] += 1
        meta["gpus"][d["env"]["gpu"]] = meta["gpus"].get(d["env"]["gpu"], 0) + 1
        meta["impl"][s.get("fp4_impl", "?")] = meta["impl"].get(s.get("fp4_impl", "?"), 0) + 1
        for e in d["episodes"]:
            assert int(e["seed"]) == int(s["seed0"]) + int(e["ep"])
            E[(s["task"], int(s["ckpt"]), s["variant"], int(e["ep"]))] = {
                "ret": float(e["ret"]), "success": float(e["success"]), "seed": int(e["seed"]), "gpu": d["env"]["gpu"]}
    H = {}
    for f in sorted(glob.glob(f"{res}/jobs/heldperr/heldperr_*.json.gz")):
        d = json.load(gzip.open(f, "rt"))
        s = d["spec"]
        for e in d["episodes"]:
            H[(s["task"], int(s["ckpt"]), int(e["ep"]))] = {
                "seed": int(e["seed"]), "ret": float(e["ret"]), "success": float(e["success"]),
                "plan": {v: {k: e["perr_plan"][v].get(k) for k in ERR_KEYS + ["n_states"]} for v in e["perr_plan"]},
                "traj": {v: {k: e["perr_traj"][v].get(k) for k in ERR_KEYS + ["true1"]} for v in e["perr_traj"]}}
    return E, H, meta


# ---------------------------------------------------------------- cells
def cell(E, H, task, fmt):
    vs = [f"{fmt}_det", f"{fmt}_shiter"]
    keys = {(c, e) for (t, c, v, e) in E if t == task and v == "none"}
    for v in vs:
        keys &= {(c, e) for (t, c, vv, e) in E if t == task and vv == v}
    prs = sorted(keys)
    n = len(prs)
    if n == 0:
        return None
    mw = is_mw(task)
    key = "success" if mw else "ret"
    idx = boot_idx(n)
    Y = {v: np.array([E[(task, c, v, e)][key] for c, e in prs], float) for v in ["none"] + vs}
    yb = {v: Y[v][idx].mean(1) for v in Y}
    m0 = {v: float(Y[v].mean()) for v in Y}

    def drop(mn, mv):
        return (mn - mv) if mw else (mn - mv) / mn

    det, sh = vs
    Dd, Ddb = drop(m0["none"], m0[det]), drop(yb["none"], yb[det])
    Ds, Dsb = drop(m0["none"], m0[sh]), drop(yb["none"], yb[sh])
    with np.errstate(divide="ignore", invalid="ignore"):
        rb = np.where(Ddb > 0, (Ddb - Dsb) / np.where(Ddb > 0, Ddb, 1.0), np.nan)
    dvd = Dsb - Ddb
    C = {"task": task, "fmt": fmt, "family": "mw" if mw else "dmc", "n": n, "metric": key, "none_mean": m0["none"],
         "ckpts": sorted({c for c, e in prs}),
         "det_mean": m0[det], "D_det": float(Dd), "D_det_ci": ci95(Ddb), "p_det": p_le0(Ddb),
         "p_det_frac": float(np.mean(Ddb <= 0)),
         "shiter_mean": m0[sh], "D_shiter": float(Ds), "D_shiter_ci": ci95(Dsb), "p_shiter": p_le0(Dsb),
         "Dsh_minus_Ddet": float(Ds - Dd), "Dsh_minus_Ddet_ci": ci95(dvd),
         "red_all": float((Dd - Ds) / Dd) if Dd > 0 else None, "red_ci_all": ci95(rb),
         "n_red_resamples": int(np.isfinite(rb).sum())}
    C["harmful"] = bool(C["Dsh_minus_Ddet_ci"][0] > 0.05 + TOL)
    if mw:
        S0, Sd, Ss = (Y[v].astype(bool) for v in ("none", det, sh))
        C["mcnemar_det"] = dict(zip(("none_only", "det_only", "p"), mcnemar(S0, Sd)))
        C["mcnemar_shiter"] = dict(zip(("none_only", "shiter_only", "p"), mcnemar(S0, Ss)))
        C["mcnemar_shiter_vs_det"] = dict(zip(("shiter_only", "det_only", "p"), mcnemar(Ss, Sd)))
    # held 1-step errors (pt = planning-time, primary; ds = dataset-type trajectory windows, auxiliary)
    for v, lab in ((det, "det"), (sh, "shiter")):
        for src, tag in (("plan", "pt"), ("traj", "ds")):
            for k in ERR_KEYS:
                x = [h[src][v][k] for (t, c, e), h in H.items() if t == task and v in h[src]
                     and h[src][v].get(k) is not None]
                if x:
                    C[f"{tag}_{k}_{lab}"] = float(np.mean(x))
        C[f"n_held_{lab}"] = sum(1 for (t, c, e), h in H.items() if t == task and v in h["plan"])
    dd = [h["plan"][det]["err1h"] - h["plan"][sh]["err1h"] for (t, c, e), h in sorted(H.items())
          if t == task and det in h["plan"] and sh in h["plan"]]
    if dd:
        dd = np.array(dd)
        C["pt_err1h_det_minus_shiter"] = float(dd.mean())
        C["pt_err1h_det_minus_shiter_ci"] = ci95(dd[boot_idx(len(dd))].mean(1))
    return C, Y, idx


def fmt_pair(task, Ys, idxs):
    """NVFP4 - MXFP4 paired differences of D_det and D_shiter (same pairs: every variant shares the 90 pairs)."""
    if not all(f in Ys for f in FMTS):
        return None
    Yn, Ym = Ys["nvfp4"], Ys["mxfp4"]
    if len(Yn["none"]) != len(Ym["none"]) or not np.array_equal(Yn["none"], Ym["none"]):
        return None
    idx = idxs["nvfp4"]
    mw = is_mw(task)

    def drop(mn, mv):
        return (mn - mv) if mw else (mn - mv) / mn

    out = {}
    for lab in ("det", "shiter"):
        a = drop(Yn["none"][idx].mean(1), Yn[f"nvfp4_{lab}"][idx].mean(1))
        b = drop(Ym["none"][idx].mean(1), Ym[f"mxfp4_{lab}"][idx].mean(1))
        d0 = drop(Yn["none"].mean(), Yn[f"nvfp4_{lab}"].mean()) - drop(Ym["none"].mean(), Ym[f"mxfp4_{lab}"].mean())
        out[f"D_{lab}_nv_minus_mx"] = float(d0)
        out[f"D_{lab}_nv_minus_mx_ci"] = ci95(a - b)
    return out


def load_e1(path):
    rows = {}
    if not path or not os.path.exists(path):
        return rows
    for r in csv.DictReader(open(path)):
        if r["k"] != "4" or r["variant"] not in ("det", "shiter"):
            continue
        ci = [float(x) for x in r["D_ci"].strip("[]").split(";")] if r["D_ci"] else None
        rows[(r["task"], r["variant"])] = {"D": float(r["D"]), "D_ci": ci, "mean": float(r["mean"]),
                                           "holm_real": r["holm_real"] == "True" if r["holm_real"] else None,
                                           "red": float(r["red"]) if r["red"] else None}
    return rows


# ---------------------------------------------------------------- judgements
def judge(cells):
    J = {"F1": {}, "F2": {}, "F3": {}}
    for fmt in FMTS:
        cs = [cells[(t, fmt)] for t in TASKS if (t, fmt) in cells]
        dmg = [c for c in cs if c["damaged"]]
        J["F1"][fmt] = {"n_cells": len(cs), "n_complete_90": sum(c["n"] == 90 for c in cs), "n_damaged": len(dmg),
                        "damaged": [c["task"] for c in dmg], "holds": len(dmg) >= 3}
    for fam, tasks in (("mw", MW8), ("dmc", DMC4)):
        cs = [cells[(t, f)] for t in tasks for f in FMTS if (t, f) in cells]
        dmg = [c for c in cs if c["damaged"]]
        ok = [c for c in dmg if c["red"] is not None and c["red"] >= 0.5 - TOL]
        harm_all = [c for c in cs if c["harmful"]]
        harm_dmg = [c for c in dmg if c["harmful"]]
        share = len(ok) / len(dmg) if dmg else None
        cond = share is not None and share >= 0.75 - TOL
        J["F2"][fam] = {"n_cells": len(cs), "n_damaged": len(dmg), "damaged": [[c["task"], c["fmt"]] for c in dmg],
                        "n_red_ge_half": len(ok), "share_red_ge_half": share,
                        "cells_red_lt_half": [[c["task"], c["fmt"], c["red"]] for c in dmg if c not in ok],
                        "harmful_cells_all": [[c["task"], c["fmt"]] for c in harm_all],
                        "harmful_cells_damaged": [[c["task"], c["fmt"]] for c in harm_dmg],
                        "holds": bool(cond and not harm_all) if dmg else None,
                        "holds_reading_damaged_only": bool(cond and not harm_dmg) if dmg else None,
                        "evaluable": bool(dmg)}
    dmg = [c for c in cells.values() if c["damaged"]]
    meet = [c for c in dmg if c.get("pt_err1h_det") is not None and c.get("pt_err1h_shiter") is not None
            and c["pt_err1h_det"] <= c["pt_err1h_shiter"] + TOL]
    meet_ds = [c for c in dmg if c.get("ds_err1h_det") is not None and c.get("ds_err1h_shiter") is not None
               and c["ds_err1h_det"] <= c["ds_err1h_shiter"] + TOL]
    by = {}
    for fam in ("mw", "dmc"):
        for fmt in FMTS:
            d_ = [c for c in dmg if c["family"] == fam and c["fmt"] == fmt]
            by[f"{fam}/{fmt}"] = {"n_damaged": len(d_), "n_meeting": sum(1 for c in meet if c in d_)}
    J["F3"] = {"n_damaged": len(dmg), "n_meeting": len(meet), "cells_meeting": [[c["task"], c["fmt"]] for c in meet],
               "by_unit": by, "holds": len(meet) >= 1 if dmg else None, "evaluable": bool(dmg),
               "n_meeting_dataset_type": len(meet_ds)}
    return J


# ---------------------------------------------------------------- output
CSV_COLS = ["task", "fmt", "family", "n", "metric", "none_mean", "det_mean", "D_det", "D_det_ci", "p_det", "p_det_frac",
            "holm_rank_alpha", "damaged", "shiter_mean", "D_shiter", "D_shiter_ci", "p_shiter", "red", "red_ci",
            "red_all", "red_ci_all", "Dsh_minus_Ddet", "Dsh_minus_Ddet_ci", "harmful", "pt_err1h_det",
            "pt_err1h_shiter", "pt_err1h_det_minus_shiter", "pt_err1h_det_minus_shiter_ci", "pt_err3_det",
            "pt_err3_shiter", "ds_err1h_det", "ds_err1h_shiter", "n_held_det", "e1_k4_D_det", "e1_k4_D_det_ci",
            "e1_k4_D_shiter", "D_det_nv_minus_mx", "D_det_nv_minus_mx_ci"]


def fmtv(x):
    if isinstance(x, (list, tuple)):
        return "[" + ";".join(fmtv(y) for y in x) + "]"
    if isinstance(x, float):
        return f"{x:.6g}"
    return "" if x is None else str(x)


def num(x, nd=3):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def pp(x):
    return "" if x is None else f"{100 * x:+.1f}"


def cit(ci, scale=100, nd=1):
    if not ci or ci[0] is None or not np.isfinite(ci[0]):
        return ""
    return f"[{scale * ci[0]:.{nd}f}, {scale * ci[1]:.{nd}f}]"


def write_md(S, path):
    L = ["## 판정\n"]
    J = S["judgements"]
    for fmt in FMTS:
        f = J["F1"][fmt]
        L.append(f"- F1 {fmt}: 손상 칸 {f['n_damaged']}/{f['n_cells']} ({', '.join(f['damaged'])}) → "
                 f"{'성립' if f['holds'] else '불성립'}")
    for fam in ("mw", "dmc"):
        f = J["F2"][fam]
        if not f["evaluable"]:
            L.append(f"- F2 {fam}: 손상 칸 없음 → 판정 불가")
            continue
        L.append(f"- F2 {fam}: red ≥ 0.5 {f['n_red_ge_half']}/{f['n_damaged']} ({f['share_red_ge_half']:.2f}), "
                 f"해로운 칸(단위 전체) {f['harmful_cells_all']}, (손상 칸만) {f['harmful_cells_damaged']} → "
                 f"{'성립' if f['holds'] else '불성립'} (손상 칸만 읽기: {'성립' if f['holds_reading_damaged_only'] else '불성립'})")
    f = J["F3"]
    L.append(f"- F3: 손상 칸 {f['n_damaged']}개 중 det pt_err1h ≤ sh-iter인 칸 {f['n_meeting']}개 "
             f"{f['by_unit']} → {'성립' if f['holds'] else ('불성립' if f['evaluable'] else '판정 불가')}")
    L.append("\n## 24칸 (MW: %p, DMC: none 대비 return %; CI는 짝 bootstrap 95%)\n")
    L.append("| 과제 | 형식 | none | D_det [CI] | p (Holm) | D_sh-iter [CI] | red [CI] | D_sh − D_det [CI] | "
             "pt_err1h det / sh-iter | E1 k=4 D_det |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for key, c in S["cells"].items():
        mw = c["family"] == "mw"
        red = "" if c.get("red_all") is None else f"{c['red_all']:.2f} {cit(c['red_ci_all'], 1, 2)}"
        if not c["damaged"] and red:
            red = f"({red})"
        L.append(f"| {c['task']} | {c['fmt']} | {num(c['none_mean'], 3 if mw else 1)} | {pp(c['D_det'])} "
                 f"{cit(c['D_det_ci'])} | {c['p_det']:.4f} ({'실재' if c['damaged'] else '-'}) | {pp(c['D_shiter'])} "
                 f"{cit(c['D_shiter_ci'])} | {red} | {pp(c['Dsh_minus_Ddet'])} {cit(c['Dsh_minus_Ddet_ci'])}"
                 f"{' 해로움' if c['harmful'] else ''} | {num(c.get('pt_err1h_det'))} / {num(c.get('pt_err1h_shiter'))} | "
                 f"{pp(c.get('e1_k4_D_det'))} |")
    L.append("\n- red는 손상 칸에서만 판정에 쓴다. 괄호 안 red는 손상 칸이 아닌 칸의 참고값이다(D_det > 0일 때).")
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--e1", default="")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    E, H, meta = load(a.res)
    e1 = load_e1(a.e1)
    cells, Ys, idxs = {}, {}, {}
    for t in TASKS:
        for fmt in FMTS:
            r = cell(E, H, t, fmt)
            if r is None:
                continue
            C, Y, idx = r
            cells[(t, fmt)] = C
            Ys.setdefault(t, {})[fmt] = Y
            idxs.setdefault(t, {})[fmt] = idx
            if (t, "det") in e1:
                C["e1_k4_D_det"] = e1[(t, "det")]["D"]
                C["e1_k4_D_det_ci"] = e1[(t, "det")]["D_ci"]
                C["e1_k4_D_det_holm_real"] = e1[(t, "det")]["holm_real"]
                C["e1_k4_D_shiter"] = e1[(t, "shiter")]["D"] if (t, "shiter") in e1 else None
        if t in Ys:
            fp_ = fmt_pair(t, Ys[t], idxs[t])
            if fp_:
                for fmt in FMTS:
                    cells[(t, fmt)].update({k: v for k, v in fp_.items()})
    # Holm per format over its 12 tasks (alpha 0.05, one-sided), plus D_det > 0
    for fmt in FMTS:
        ts = [t for t in TASKS if (t, fmt) in cells]
        ps = [cells[(t, fmt)]["p_det"] for t in ts]
        rej = holm(ps, ALPHA) if ps else []
        order = np.argsort(ps)
        for r_, i in enumerate(order):
            cells[(ts[i], fmt)]["holm_rank_alpha"] = [int(r_ + 1), ALPHA / (len(ps) - r_)]
        rej_frac = holm([cells[(t, fmt)]["p_det_frac"] for t in ts], ALPHA) if ps else []
        for t, rj, rjf in zip(ts, rej, rej_frac):
            c = cells[(t, fmt)]
            c["damaged"] = bool(rj and c["D_det"] > 0)
            c["damaged_literal_fraction_p"] = bool(rjf and c["D_det"] > 0)
            c["red"] = c["red_all"] if c["damaged"] else None
            c["red_ci"] = c["red_ci_all"] if c["damaged"] else None
    J = judge(cells)
    base = {t: {"n": sum(1 for k in E if k[0] == t and k[2] == "none"),
                "none_mean": float(np.mean([E[k]["success" if is_mw(t) else "ret"] for k in E
                                             if k[0] == t and k[2] == "none"]))} for t in TASKS
            if any(k[0] == t and k[2] == "none" for k in E)}
    held = {t: {"n": sum(1 for k in H if k[0] == t),
                "none_mean": float(np.mean([H[k]["success" if is_mw(t) else "ret"] for k in H if k[0] == t]))}
            for t in TASKS if any(k[0] == t for k in H)}
    diff_lit = [[t, f] for (t, f), c in cells.items() if c["damaged"] != c["damaged_literal_fraction_p"]]
    S = {"n_episodes": len(E), "n_held_episodes": len(H), "meta": meta, "baseline_none": base, "held_none": held,
         "judgements": J, "damaged_differs_with_literal_p": diff_lit,
         "cells": {f"{t}:{f}": c for (t, f), c in sorted(cells.items(), key=lambda kv: (TASKS.index(kv[0][0]),
                                                                                       FMTS.index(kv[0][1])))}}
    json.dump(S, open(f"{a.outdir}/e9_summary.json", "w"), indent=1, default=float)
    with open(f"{a.outdir}/e9_cells.csv", "w") as f:
        f.write(",".join(CSV_COLS) + "\n")
        for key, c in S["cells"].items():
            f.write(",".join(fmtv(c.get(col)) for col in CSV_COLS) + "\n")
    with open(f"{a.outdir}/e9_episodes.csv", "w") as f:
        f.write("task,ckpt,ep,seed,variant,ret,success,gpu\n")
        for (t, c, v, e), r in sorted(E.items()):
            f.write(f"{t},{c},{e},{r['seed']},{v},{r['ret']:.6f},{r['success']:g},{r['gpu']}\n")
    with open(f"{a.outdir}/e9_held_errors.csv", "w") as f:
        f.write("task,ckpt,ep,seed,variant,source," + ",".join(ERR_KEYS) + "\n")
        for (t, c, e), h in sorted(H.items()):
            for src in ("plan", "traj"):
                for v, d in sorted(h[src].items()):
                    f.write(f"{t},{c},{e},{h['seed']},{v},{src}," + ",".join(fmtv(d.get(k)) for k in ERR_KEYS) + "\n")
    write_md(S, f"{a.outdir}/e9_tables.md")
    print(json.dumps(J, indent=1, default=float))
    print(f"episodes {len(E)} held {len(H)} cells {len(cells)} literal-p differences {diff_lit}")


if __name__ == "__main__":
    main()
