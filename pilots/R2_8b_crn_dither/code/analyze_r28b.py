"""Phase E (R2-8b): planning drops with paired-bootstrap 90% CIs, S1-S3 and the verdict (PLAN.md section 5),
tables and the two figures.

Inputs (<out>/results): episodes.jsonl (none + Q{v}{k} per cell), prep_<task>.json (held-out prediction errors),
decision_<task>.csv (value-level metrics and final-plan deviation; optional).
Usage: python analyze_r28b.py --cells mw-push:3,mw-pick-place:3,cheetah-run:4,walker-walk:4 --out /workspace/out_main
Writes summary.json, table_cells.csv, table_pairs.csv, fig_drop_by_scope.png, fig_err1_vs_drop.png in <out>/results.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

VARS = ("det", "ind", "shfwd", "shstep", "shep")
NAME = {"det": "det", "ind": "ind", "shfwd": "sh-fwd", "shstep": "sh-step", "shep": "sh-ep"}
NBOOT, BOOT_SEED = 4000, 3
RED = 0.5
# categorical slots 1-5 of the reference palette, fixed order (validated: adjacent CVD dE >= 9.1, normal >= 19.6)
COL = {"det": "#2a78d6", "ind": "#eb6834", "shfwd": "#1baf7a", "shstep": "#eda100", "shep": "#e87ba4"}
MARK = {"det": "o", "ind": "s", "shfwd": "D", "shstep": "^", "shep": "v"}
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
PAIRS = (("shfwd", "ind"), ("shep", "det"), ("shfwd", "det"), ("shstep", "det"), ("ind", "det"),
         ("shstep", "shfwd"), ("shep", "shfwd"), ("shep", "shstep"))


def metric_of(task):
    return "success" if task.startswith("mw-") else "ret"


def pct(b):
    b = b[np.isfinite(b)]
    return [float(np.percentile(b, 5)), float(np.percentile(b, 95))] if len(b) else [float("nan")] * 2


def cell_stats(ep, task, k, metric, seed=BOOT_SEED):
    """Per-episode paired drops d_v,i = (base_i - x_v,i) / R0 on the episodes every condition has."""
    sub = ep[ep.task == task]
    piv = sub.pivot_table(index="ep", columns="cond", values=metric, aggfunc="first")
    conds = ["none"] + [f"Q{v}{k}" for v in VARS]
    have = [c for c in conds if c in piv.columns]
    piv = piv[have].dropna()
    n = len(piv)
    base = piv["none"].values
    R0 = float(base.mean())
    d = {v: (base - piv[f"Q{v}{k}"].values) / R0 for v in VARS if f"Q{v}{k}" in piv.columns}
    idx = np.random.default_rng(seed).integers(0, n, size=(NBOOT, n))
    out = {"n": n, "R0": R0, "metric": metric, "drop": {}, "drop_ci": {}, "pair": {}, "pair_ci": {}, "red": {},
           "red_ci": {}, "red_nan_frac": {}, "mean": {v: float(piv[f"Q{v}{k}"].mean()) for v in d}}
    for v, x in d.items():
        out["drop"][v] = float(x.mean())
        out["drop_ci"][v] = pct(x[idx].mean(1))
    for a, b in PAIRS:
        if a in d and b in d:
            x = d[a] - d[b]   # drop_a - drop_b per episode (negative: a is less harmful)
            out["pair"][f"{a}-{b}"] = float(x.mean())
            out["pair_ci"][f"{a}-{b}"] = pct(x[idx].mean(1))
    if "det" in d:
        bd = d["det"][idx].mean(1)
        for v in ("ind", "shfwd", "shstep", "shep"):
            if v in d:
                Dd, Dv = out["drop"]["det"], out["drop"][v]
                out["red"][v] = float((Dd - Dv) / Dd) if Dd > 0 else float("nan")
                bv = d[v][idx].mean(1)
                r = np.where(bd > 0, (bd - bv) / np.where(bd > 0, bd, 1.0), np.nan)
                out["red_ci"][v] = pct(r)
                out["red_nan_frac"][v] = float(np.mean(~np.isfinite(r)))
    return out


def criteria(cells):
    per, n1, n1s, n2, n3, n3lit, aux = {}, 0, 0, 0, 0, 0, {"shep>shfwd": 0, "shstep>shfwd": 0, "shep>shstep": 0}
    for t, c in cells.items():
        p = c["primary"]
        det_real = bool(p["drop_ci"]["det"][0] > 0)
        reds = {v: p["red"].get(v, float("nan")) for v in ("shfwd", "shstep")}
        best = max(reds, key=lambda v: -np.inf if not np.isfinite(reds[v]) else reds[v])
        s1 = bool(det_real and np.isfinite(reds[best]) and reds[best] >= RED)
        s1_strict = bool(det_real and any(np.isfinite(reds[v]) and reds[v] >= RED and p["pair_ci"][f"{v}-det"][1] < 0
                                          for v in reds))
        s2 = bool(p["pair_ci"]["shfwd-ind"][1] < 0)
        ep_det_hi = p["pair_ci"]["shep-det"][1]
        s3_lit = bool(ep_det_hi >= 0)
        s3 = bool(det_real and s3_lit)
        a = {"shep>shfwd": p["pair_ci"]["shep-shfwd"][0] > 0, "shstep>shfwd": p["pair_ci"]["shstep-shfwd"][0] > 0,
             "shep>shstep": p["pair_ci"]["shep-shstep"][0] > 0}
        for kk, vv in a.items():
            aux[kk] += int(vv)
        per[t] = {"k": c["k"], "det_drop_real": det_real, "det_drop": p["drop"]["det"], "det_drop_ci": p["drop_ci"]["det"],
                  "red_shfwd": reds["shfwd"], "red_shfwd_ci": p["red_ci"].get("shfwd"),
                  "red_shstep": reds["shstep"], "red_shstep_ci": p["red_ci"].get("shstep"), "S1_best_variant": best,
                  "S1_task": s1, "S1_task_strict": s1_strict,
                  "shfwd_minus_ind": p["pair"]["shfwd-ind"], "shfwd_minus_ind_ci": p["pair_ci"]["shfwd-ind"], "S2_task": s2,
                  "shep_minus_det": p["pair"]["shep-det"], "shep_minus_det_ci": p["pair_ci"]["shep-det"],
                  "S3_task": s3, "S3_task_literal": s3_lit, "aux": {kk: bool(vv) for kk, vv in a.items()}}
        n1 += s1
        n1s += s1_strict
        n2 += s2
        n3 += s3
        n3lit += s3_lit
    S1, S2, S3 = n1 >= 2, n2 >= 3, n3 >= 2
    if S1 and S2 and S3:
        verdict = "강한 있음"
    elif S1 and S2:
        verdict = "있음"
    elif S1 or S2:
        verdict = "조건부 있음"
    else:
        verdict = "없음"
    return {"per_task": per, "S1": S1, "S1_tasks": n1, "S1_strict_tasks": n1s, "S2": S2, "S2_tasks": n2, "S3": S3,
            "S3_tasks": n3, "S3_literal_tasks": n3lit, "aux_counts": aux, "verdict": verdict}


def decision_table(R, task, k):
    f = f"{R}/decision_{task}.csv"
    if not os.path.exists(f):
        return {}
    d = pd.read_csv(f)
    d["alpha"] = d["alpha"].astype(float)
    first = d[d.set == "first"]
    plan = d[d.set == "plan"]
    sd_mu = float(first[first.cond == "seednoise"].dmu.mean())
    sd_dev = float(plan[plan.cond == "seednoise"].dev_mean.mean())
    out = {"seed_dmu": sd_mu, "seed_dev_mean": sd_dev, "n_states": int(first.state.nunique()),
           "n_plan_states": int(plan.state.nunique())}
    for v in VARS:
        c = f"Q{v}{k}"
        f1, pl = first[first.cond == c], plan[plan.cond == c]
        if len(f1):
            out[v] = {"cvb_norm": float(f1.dmu.mean() / sd_mu), "spear": float(f1.spear.mean()),
                      "flip": float(f1.flip.mean()), "nerr_c": float(f1.nerr_c.mean())}
        if len(pl):
            out.setdefault(v, {}).update(dev_norm=float(pl.dev_mean.mean() / sd_dev), dev_mean=float(pl.dev_mean.mean()),
                                         dev_act=float(pl.dev_act.mean()))
    out["seed_spear"] = float(first[first.cond == "seednoise"].spear.mean())
    return out


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, labelcolor=INK2)


def figures(R, cells, order):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "axes.titlesize": 10,
                         "figure.facecolor": SURF, "font.family": "sans-serif"})
    n = len(order)
    handles = [Line2D([0], [0], marker=MARK[v], color=COL[v], lw=0, ms=8, markeredgecolor=SURF, markeredgewidth=1.5,
                      label=NAME[v]) for v in VARS]
    # 1) drop by sharing scope, per task
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), squeeze=False)
    for ax, t in zip(axes[0], order):
        c = cells[t]["primary"]
        style(ax)
        ax.axhline(0, color=AXIS, lw=1)
        for i, v in enumerate(VARS):
            if v not in c["drop"]:
                continue
            y, lo, hi = 100 * c["drop"][v], 100 * c["drop_ci"][v][0], 100 * c["drop_ci"][v][1]
            ax.plot([i, i], [lo, hi], color=COL[v], lw=2, solid_capstyle="round")
            ax.plot(i, y, marker=MARK[v], color=COL[v], ms=9, markeredgecolor=SURF, markeredgewidth=1.5, lw=0)
            ax.annotate(f"{y:.0f}", (i, hi), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7.5,
                        color=INK2)
        ax.set_xticks(range(len(VARS)))
        ax.set_xticklabels([NAME[v] for v in VARS], rotation=0, fontsize=7.5)
        ax.set_xlim(-0.6, len(VARS) - 0.4)
        unit = "success-rate drop (%p)" if c["metric"] == "success" else "return drop (%)"
        ax.set_ylabel(unit)
        ax.set_title(f"{t}  (k={cells[t]['k']})", color=INK)
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8)
    fig.suptitle(f"Planning drop by dither-sharing scope (vs. unquantized, {cells[order[0]]['primary']['n']} paired "
                 "episodes, 90% CI)", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(f"{R}/fig_drop_by_scope.png", dpi=130)
    plt.close(fig)
    # 2) 1-step error vs drop
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), squeeze=False)
    for ax, t in zip(axes[0], order):
        c = cells[t]
        p = c["primary"]
        style(ax)
        ax.grid(True, axis="x", color=GRID, lw=0.8)
        ax.axhline(0, color=AXIS, lw=1)
        for v in VARS:
            e = c["err"].get(v)
            if v not in p["drop"] or not e:
                continue
            x, y = e["err1"], 100 * p["drop"][v]
            lo, hi = 100 * p["drop_ci"][v][0], 100 * p["drop_ci"][v][1]
            ax.plot([x, x], [lo, hi], color=COL[v], lw=2, alpha=0.55, solid_capstyle="round")
            ax.plot(x, y, marker=MARK[v], color=COL[v], ms=9, markeredgecolor=SURF, markeredgewidth=1.5, lw=0)
        ax.set_xlabel("normalized 1-step latent error (held-out)")
        ax.set_ylabel("success-rate drop (%p)" if p["metric"] == "success" else "return drop (%)")
        ax.set_title(f"{t}  (k={c['k']})", color=INK)
        xs = [c["err"][v]["err1"] for v in VARS if c["err"].get(v)]
        if xs:
            pad = 0.12 * (max(xs) - min(xs) + 1e-9)
            ax.set_xlim(min(xs) - pad, max(xs) + pad)
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8)
    fig.suptitle("Held-out 1-step prediction error vs. planning drop (90% CI)", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(f"{R}/fig_err1_vs_drop.png", dpi=130)
    plt.close(fig)


def main(out, cells_arg):
    R = f"{out}/results"
    order = [c.split(":")[0] for c in cells_arg.split(",")]
    kmap = {c.split(":")[0]: int(c.split(":")[1]) for c in cells_arg.split(",")}
    ep = pd.DataFrame([json.loads(l) for l in open(f"{R}/episodes.jsonl") if l.strip()])
    cells, rows, prow = {}, [], []
    for t in order:
        k = kmap[t]
        prim = cell_stats(ep, t, k, metric_of(t))
        sec = cell_stats(ep, t, k, "ret") if t.startswith("mw-") else None
        pj = f"{R}/prep_{t}.json"
        pred = json.load(open(pj))["pred"] if os.path.exists(pj) else {}
        err = {v: pred.get(f"Q{v}{k}@{k}") for v in VARS}
        dec = decision_table(R, t, k)
        nref = {v: sorted(ep[(ep.task == t) & (ep.cond == f"Q{v}{k}")].n_refresh.unique().tolist())
                for v in ("shstep", "shep")} if "n_refresh" in ep.columns else {}
        cells[t] = {"k": k, "primary": prim, "secondary_return": sec, "err": err, "decision": dec,
                    "n_refresh_values": nref,
                    "baseline": {"ret": float(ep[(ep.task == t) & (ep.cond == "none")].ret.mean()),
                                 "success": float(ep[(ep.task == t) & (ep.cond == "none")].success.mean())}}
        for v in VARS:
            if v not in prim["drop"]:
                continue
            r = {"task": t, "k": k, "variant": NAME[v], "n": prim["n"], "metric": prim["metric"], "R0": prim["R0"],
                 "mean": prim["mean"][v], "drop": prim["drop"][v], "drop_lo": prim["drop_ci"][v][0],
                 "drop_hi": prim["drop_ci"][v][1], "red_vs_det": prim["red"].get(v),
                 "red_lo": (prim["red_ci"].get(v) or [None, None])[0], "red_hi": (prim["red_ci"].get(v) or [None, None])[1]}
            if sec:
                r.update(ret_drop=sec["drop"][v], ret_drop_lo=sec["drop_ci"][v][0], ret_drop_hi=sec["drop_ci"][v][1])
            if err.get(v):
                r.update(err1=err[v]["err1"], err3=err[v]["err3"], errr=err[v]["errr"])
            if dec.get(v):
                r.update({kk: vv for kk, vv in dec[v].items()})
            rows.append(r)
        for key, val in prim["pair"].items():
            prow.append({"task": t, "k": k, "pair": key.replace("shfwd", "sh-fwd").replace("shstep", "sh-step")
                        .replace("shep", "sh-ep"), "drop_diff": val, "lo": prim["pair_ci"][key][0],
                         "hi": prim["pair_ci"][key][1]})
    crit = criteria(cells)
    pd.DataFrame(rows).to_csv(f"{R}/table_cells.csv", index=False)
    pd.DataFrame(prow).to_csv(f"{R}/table_pairs.csv", index=False)
    summary = {"verdict": crit["verdict"], "criteria": crit, "cells": cells}
    json.dump(summary, open(f"{R}/summary.json", "w"), indent=1, default=float)
    print(json.dumps({kk: crit[kk] for kk in crit if kk != "per_task"}, indent=1, ensure_ascii=False))
    for t in order:
        p = cells[t]["primary"]
        print(f"{t} k={cells[t]['k']} n={p['n']} R0={p['R0']:.3f}: " + "  ".join(
            f"{NAME[v]} {100 * p['drop'][v]:.1f} [{100 * p['drop_ci'][v][0]:.1f},{100 * p['drop_ci'][v][1]:.1f}]"
            for v in VARS if v in p["drop"]))
        print("   pairs: " + "  ".join(f"{kk} {100 * vv:.1f} [{100 * p['pair_ci'][kk][0]:.1f},{100 * p['pair_ci'][kk][1]:.1f}]"
                                       for kk, vv in p["pair"].items()))
        print("   " + json.dumps(crit["per_task"][t], default=float, ensure_ascii=False))
    figures(R, cells, order)
    print("ANALYZE_DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.out, a.cells)
