"""Phase E (R2-3b): consistency vs weak-specific verdict, determinism check, tables and figures.

Inputs (<out>/results): episodes_3b.jsonl (this run: none, A, B, D1z, D3z, Dfix1_s*, Dfix3_s*, Ddir1_s*),
r23_episodes.jsonl (R2-3, A5000 pod: only for the cross-GPU comparison), prep_<task>.json, decision_<task>.csv
Rules (PLAN_3b.md section 4):
  cells: 4 tasks x alpha in {0.1, 0.2}. Drop: DMC relative return drop, MW success-rate drop (secondary: return).
  D-fix drop per episode k = mean over the 3 fixed subsets s of the per-episode metric; paired bootstrap over the 10
  shared episode seeds (4000 resamples), 90% CI.
  Consistency: (i) CI_lo(drop_A - drop_Dfix1) <= 0 in >= 6/8 cells AND (ii) CI_lo(drop_Dfix1 - drop_D1z) > 0 in >= 6/8.
  Weak-specific: CI_lo > 0 for A-Dfix1, A-Dfix3, B-Dfix1 and B-Dfix3, all four, in >= 6/8 cells.
  Otherwise mixed: share of A's drop reproduced by D-fix (eligible cells: drop_A >= 0.10 and its CI excludes 0).
Usage: python analyze.py --tasks mw-push,mw-pick-place,cheetah-run,walker-walk --alphas 0.1,0.2 --out /workspace/out_main
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

SEEDS = (0, 1, 2)
FAM = ("Dfix1", "Dfix3", "Ddir1")
NBOOT = 4000
COL = {"A": "#2a78d6", "B": "#1baf7a", "D1z": "#898781", "Dfix1": "#eb6834", "Dfix3": "#e87ba4", "Ddir1": "#6250d6",
       "seed": "#898781"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
LABEL = {"A": "A: zero weak", "B": "B: weak → cond. mean", "D1z": "D1z: independent random (1-step matched)",
         "Dfix1": "D-fix: fixed random units, 1-step matched", "Dfix3": "D-fix: fixed random units, 3-step matched",
         "Ddir1": "D-dir: fixed random direction, 1-step matched"}


def akey(a):
    return f"{float(a):g}"


def boot_ci(x, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [float("nan"), float("nan")]
    b = np.random.default_rng(seed).choice(x, size=(NBOOT, len(x)), replace=True).mean(1)
    return [float(np.percentile(b, 5)), float(np.percentile(b, 95))]


def load(R):
    r23 = pd.DataFrame([json.loads(l) for l in open(f"{R}/r23_episodes.jsonl") if l.strip()])
    new = pd.DataFrame([json.loads(l) for l in open(f"{R}/episodes_3b.jsonl") if l.strip()])
    for d in (r23, new):
        d["alpha"] = d["alpha"].astype(float)
    return r23, new


def determinism(r23, new):
    """Cross-GPU check: the same (task, cond, alpha, episode seed) on the R2-3 A5000 pod and on this pod."""
    out = []
    for r in new[new.cond.isin(["none", "A", "B", "D1z", "D3z"])].itertuples():
        m = r23[(r23.task == r.task) & (r23.cond == r.cond) & np.isclose(r23.alpha, r.alpha) & (r23.ep == r.ep)]
        if len(m):
            out.append({"task": r.task, "cond": r.cond, "alpha": r.alpha, "ep": r.ep, "ret_3b": r.ret,
                        "ret_r23": float(m.ret.iloc[0]), "identical": bool(float(m.ret.iloc[0]) == float(r.ret)),
                        "absdiff": abs(float(m.ret.iloc[0]) - float(r.ret))})
    return out


def per_episode(df, task, cond, alpha, metric):
    d = df[(df.task == task) & (df.cond == cond) & np.isclose(df.alpha, alpha)]
    return d.set_index("ep")[metric]


def cells(r23, new, tasks, alphas, mode):
    rows = []
    for t in tasks:
        metric = "success" if (t.startswith("mw-") and mode == "primary") else "ret"
        base = new[(new.task == t) & (new.cond == "none")].set_index("ep")[metric]
        R0 = float(base.mean())
        if R0 == 0:
            continue
        for a in alphas:
            v = {c: per_episode(new, t, c, a, metric) for c in ("A", "B", "D1z", "D3z")}
            seeds = {}
            for fam in FAM:
                per = [per_episode(new, t, f"{fam}_s{s}", a, metric) for s in SEEDS]
                per = [p for p in per if len(p)]
                if per:
                    common = per[0].index
                    for p in per[1:]:
                        common = common.intersection(p.index)
                    v[fam] = pd.concat([p.loc[common] for p in per], axis=1).mean(1)
                    seeds[fam] = [float((R0 - p.mean()) / R0) for p in per]
            row = {"task": t, "alpha": a, "metric": metric, "R0": R0}
            for c, s in v.items():
                if len(s):
                    row[f"drop_{c}"] = float((R0 - s.mean()) / R0)
                    row[f"n_{c}"] = int(len(s))
            for fam, ds in seeds.items():
                row[f"drop_{fam}_per_seed"] = ds
            if len(v["A"]):
                c_ = base.index.intersection(v["A"].index)
                row["dropA_ci"] = boot_ci(((base.loc[c_] - v["A"].loc[c_]) / R0).values, seed=1)
            for x, y in (("A", "Dfix1"), ("A", "Dfix3"), ("B", "Dfix1"), ("B", "Dfix3"), ("Dfix1", "D1z"),
                         ("Dfix3", "D3z"), ("Dfix3", "D1z"), ("Ddir1", "D1z"), ("A", "Ddir1"), ("A", "D1z")):
                if x in v and y in v and len(v[x]) and len(v[y]):
                    c_ = v[x].index.intersection(v[y].index)
                    d = ((v[y].loc[c_] - v[x].loc[c_]) / R0).values  # drop_x - drop_y per episode
                    row[f"{x}-{y}"] = float(d.mean())
                    row[f"{x}-{y}_ci"] = boot_ci(d, seed=2)
            rows.append(row)
    return rows


def verdict(rows, fam="Dfix1", rand="D1z"):
    ok = lambda r, k: f"{k}_ci" in r
    c1 = [r for r in rows if ok(r, f"A-{fam}") and r[f"A-{fam}_ci"][0] <= 0]       # D-fix not significantly smaller than A
    c2 = [r for r in rows if ok(r, f"{fam}-{rand}") and r[f"{fam}-{rand}_ci"][0] > 0]  # D-fix larger than independent random
    ws = [r for r in rows if all(ok(r, k) and r[f"{k}_ci"][0] > 0 for k in ("A-Dfix1", "A-Dfix3", "B-Dfix1", "B-Dfix3"))]
    n = len([r for r in rows if ok(r, f"A-{fam}")])
    elig = [r for r in rows if r.get("drop_A", 0) >= 0.10 and r.get("dropA_ci", [0])[0] > 0 and f"drop_{fam}" in r]
    sumA = sum(r["drop_A"] for r in elig)
    cid = lambda r: f"{r['task']}@{r['alpha']:g}"
    cons = len(c1) >= 6 and len(c2) >= 6
    weak = len(ws) >= 6
    return {"family": fam, "random": rand, "n_cells": n,
            "consistency_i_not_smaller_than_A": [cid(r) for r in c1], "consistency_ii_larger_than_random": [cid(r) for r in c2],
            "consistency": cons, "weak_specific_cells": [cid(r) for r in ws], "weak_specific": weak,
            "verdict": "consistency" if cons and not weak else ("weak-specific" if weak and not cons else
                                                                  ("both?" if cons and weak else "mixed")),
            "eligible": [cid(r) for r in elig],
            "share_of_A_per_cell": {cid(r): r[f"drop_{fam}"] / r["drop_A"] for r in elig},
            "share_of_A_pooled": sum(r[f"drop_{fam}"] for r in elig) / sumA if sumA > 0 else float("nan")}


def verdict_eligible(rows, fam="Dfix1", rand="D1z"):
    """PLAN_3b 4.2-5 (secondary): the same rules on the eligible cells only (drop_A >= 0.10 and its CI excludes 0),
    with the same proportional threshold (>= 75% of the eligible cells)."""
    elig = [r for r in rows if r.get("drop_A", 0) >= 0.10 and r.get("dropA_ci", [0])[0] > 0 and f"A-{fam}_ci" in r]
    need = int(np.ceil(0.75 * len(elig))) if elig else 0
    ok = lambda r, k: f"{k}_ci" in r
    c1 = [r for r in elig if r[f"A-{fam}_ci"][0] <= 0]
    c2 = [r for r in elig if ok(r, f"{fam}-{rand}") and r[f"{fam}-{rand}_ci"][0] > 0]
    ws = [r for r in elig if all(ok(r, k) and r[f"{k}_ci"][0] > 0 for k in ("A-Dfix1", "A-Dfix3", "B-Dfix1", "B-Dfix3"))]
    cid = lambda r: f"{r['task']}@{r['alpha']:g}"
    cons = bool(elig) and len(c1) >= need and len(c2) >= need
    weak = bool(elig) and len(ws) >= need
    return {"family": fam, "eligible": [cid(r) for r in elig], "threshold": need,
            "not_smaller_than_A": [cid(r) for r in c1], "larger_than_random": [cid(r) for r in c2],
            "weak_specific_cells": [cid(r) for r in ws], "consistency": cons, "weak_specific": weak,
            "verdict": "consistency" if cons and not weak else ("weak-specific" if weak and not cons else "mixed")}


def decision_table(R, tasks):
    dec = [pd.read_csv(f"{R}/decision_{t}.csv") for t in tasks if os.path.exists(f"{R}/decision_{t}.csv")]
    if not dec:
        return None
    dec = pd.concat(dec, ignore_index=True)
    dec["alpha"] = dec["alpha"].astype(float)
    dec["fam"] = dec["cond"].str.split("_s").str[0]
    rows = []
    for (t, s, f, a), dd in dec.groupby(["task", "set", "fam", "alpha"]):
        row = {"task": t, "set": s, "cond": f, "alpha": a, "n": len(dd)}
        if s == "plan":
            row.update(dev_mean=dd.dev_mean.mean(), dev_act=dd.dev_act.mean())
        else:
            row.update(spearman=dd.spear.mean(), elite_jaccard=dd.jac.mean(), regret_n_mean=dd.regret_n.mean(),
                       value_nerr_median=dd.nerr.median(), value_nerr_centered_median=dd.nerr_c.median())
        rows.append(row)
    D = pd.DataFrame(rows)
    D.to_csv(f"{R}/table_decision_3b.csv", index=False)
    return D


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=8)


def figures(R, rows, D, tasks, alphas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "axes.titlesize": 10})
    T = [t for t in tasks if any(r["task"] == t for r in rows)]
    fig, axes = plt.subplots(1, len(T), figsize=(3.3 * len(T), 3.3), squeeze=False)
    conds = ("A", "B", "D1z", "Dfix1", "Dfix3", "Ddir1")
    w = 0.13
    for ax, t in zip(axes[0], T):
        style(ax)
        for i, a in enumerate(alphas):
            r = [x for x in rows if x["task"] == t and np.isclose(x["alpha"], a)]
            if not r:
                continue
            r = r[0]
            for j, c in enumerate(conds):
                if f"drop_{c}" not in r:
                    continue
                x = i + (j - (len(conds) - 1) / 2) * w
                ax.bar(x, r[f"drop_{c}"], width=w * 0.9, color=COL[c], label=LABEL[c] if i == 0 else None)
                if f"drop_{c}_per_seed" in r:
                    ps = r[f"drop_{c}_per_seed"]
                    ax.plot([x, x], [min(ps), max(ps)], color=INK2, lw=1)
        ax.axhline(0, color="#c3c2b7", lw=0.8)
        ax.set_xticks(range(len(alphas)), [f"α={a:g}" for a in alphas])
        ax.set_ylabel("relative drop in " + ("success rate" if t.startswith("mw-") else "return"))
        ax.set_title(t, color=INK)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, loc="lower center", ncol=3)
    fig.suptitle("Candidate-consistent random controls (fixed units / fixed direction) vs weak ablation; "
                 "bars over seeds show min-max of the 3 fixed subsets", fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(f"{R}/fig_3b_drops.png", dpi=130)
    plt.close(fig)
    if D is not None and len(D):
        P = D[D.set == "plan"]
        fig, axes = plt.subplots(1, len(T), figsize=(3.3 * len(T), 3.2), squeeze=False, sharey=True)
        for ax, t in zip(axes[0], T):
            style(ax)
            pt = P[P.task == t]
            for c, mk in (("A", "o"), ("Dfix1", "s"), ("Dfix3", "D"), ("Ddir1", "^")):
                q = pt[pt.cond == c].sort_values("alpha")
                if len(q):
                    ax.plot(q.alpha, q.dev_mean, marker=mk, ms=5, lw=1.6, color=COL[c], label=LABEL[c])
            sn = pt[pt.cond == "seednoise"]
            if len(sn):
                ax.axhline(float(sn.dev_mean.iloc[0]), color=COL["seed"], lw=1.2, label="seed-noise floor")
            ax.set_xticks(alphas)
            ax.set_xlabel("α")
            ax.set_title(t, color=INK)
        axes[0][0].set_ylabel("RMS deviation of the final MPPI mean (step 0)")
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, fontsize=8, frameon=False, loc="lower center", ncol=3)
        fig.suptitle("Final-plan deviation from the unablated planner (same seed)", fontsize=10, color=INK)
        fig.tight_layout(rect=(0, 0.14, 1, 1))
        fig.savefig(f"{R}/fig_3b_plan_deviation.png", dpi=130)
        plt.close(fig)


def main(out, tasks, alphas):
    R = f"{out}/results"
    r23, new = load(R)
    det = determinism(r23, new)
    rows = cells(r23, new, tasks, alphas, "primary")
    rows_ret = cells(r23, new, tasks, alphas, "return")
    summary = {"cross_gpu_check": det,
               "cross_gpu_identical_fraction": (sum(d["identical"] for d in det) / len(det)) if det else None,
               "determinism_all_identical": bool(det) and all(d["identical"] for d in det),
               "primary": verdict(rows, "Dfix1", "D1z"), "secondary_Dfix3": verdict(rows, "Dfix3", "D3z"),
               "secondary_return_metric": verdict(rows_ret, "Dfix1", "D1z"),
               "secondary_return_Dfix3": verdict(rows_ret, "Dfix3", "D3z"),
               "secondary_eligible_Dfix1": verdict_eligible(rows, "Dfix1", "D1z"),
               "secondary_eligible_Dfix3": verdict_eligible(rows, "Dfix3", "D3z"),
               "secondary_eligible_return_Dfix1": verdict_eligible(rows_ret, "Dfix1", "D1z"),
               "dir_vs_random": {"Ddir1>D1z": [f"{r['task']}@{r['alpha']:g}" for r in rows if "Ddir1-D1z_ci" in r and r["Ddir1-D1z_ci"][0] > 0],
                                 "A>Ddir1": [f"{r['task']}@{r['alpha']:g}" for r in rows if "A-Ddir1_ci" in r and r["A-Ddir1_ci"][0] > 0]},
               "cells": rows}
    D = decision_table(R, tasks)
    flat = lambda rs: pd.DataFrame([{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()} for r in rs])
    flat(rows).to_csv(f"{R}/table_cells_3b.csv", index=False)
    flat(rows_ret).to_csv(f"{R}/table_cells_3b_return.csv", index=False)
    json.dump(summary, open(f"{R}/summary_3b.json", "w"), indent=1, default=float)
    print(json.dumps({k: summary[k] for k in ("determinism_all_identical", "primary", "secondary_Dfix3",
                                              "secondary_return_metric", "secondary_eligible_Dfix1",
                                              "secondary_eligible_Dfix3", "dir_vs_random")}, indent=1, default=float))
    figures(R, rows, D, tasks, alphas)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--alphas", default="0.1,0.2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.out, a.tasks.split(","), [float(x) for x in a.alphas.split(",")])
