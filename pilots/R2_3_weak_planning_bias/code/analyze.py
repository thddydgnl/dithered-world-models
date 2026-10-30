"""Phase E (R2-3): pre-registered verdict (A = bias, I = information, else mixed), tables and figures.

Rules (PLAN.md section 5.1):
  cells = (task, alpha), 12. Drop = relative drop vs the no-ablation baseline (DMC: return, MW: final success rate).
  Paired-bootstrap 90% CI (4000 resamples over the shared episode seeds) of drop_B - drop_D1m.
  Verdict A: (1) CI lower bound <= 0 in >= 10 of 12 cells AND (2) drop_C >= 0.7 drop_A in more than half of the
             eligible cells (drop_A >= 0.10 and the CI of drop_A excludes 0).
  Verdict I: (1) CI lower bound > 0 in >= 6 of 12 cells AND (2) pooled sum(drop_C) / sum(drop_A) < 0.5 over eligible cells.
  Otherwise: mixed, with the per-cell shares C/A and B/A.
Secondary (not the verdict): the same with return-based drops for all tasks; A vs D1z (R1-1 replication), A vs D3z,
B vs D3m; prediction errors; decision-level rank preservation and final-plan deviation vs the seed-noise floor.

Usage: python analyze.py --tasks mw-push,mw-pick-place,cheetah-run,walker-walk --alphas 0.1,0.2,0.3 --out /workspace/out_main
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

CONDS = ("A", "B", "C", "D1z", "D3z", "D1m", "D3m")
A1_CELLS, I1_CELLS, C_REPRO, I_RATIO, ELIG_DROP = 10, 6, 0.7, 0.5, 0.10
NBOOT = 4000
# reference categorical palette (dataviz skill, light mode) in fixed order; D controls share the orange family
COL = {"A": "#2a78d6", "B": "#1baf7a", "C": "#eda100", "D1m": "#eb6834", "D3m": "#e87ba4", "D1z": "#6250d6",
       "D3z": "#898781", "seed": "#898781"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
LABEL = {"A": "A: zero weak", "B": "B: weak → conditional mean", "C": "C: bias only (A's mean shift)",
         "D1z": "D1z: random zero, 1-step matched to A", "D3z": "D3z: random zero, 3-step matched to A",
         "D1m": "D1m: random mean-fill, 1-step matched to B", "D3m": "D3m: random mean-fill, 3-step matched to B"}


def akey(a):
    return f"{float(a):g}"


def boot_ci(x, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [float("nan"), float("nan")]
    b = np.random.default_rng(seed).choice(x, size=(NBOOT, len(x)), replace=True).mean(1)
    return [float(np.percentile(b, 5)), float(np.percentile(b, 95))]


def cell_table(ep, prep, tasks, alphas, metric_mode):
    """metric_mode 'primary' (DMC return, MW success) or 'return' (return for every task)."""
    rows = []
    for t in tasks:
        e = ep[ep.task == t]
        base = e[e.cond == "none"].set_index("ep")
        if not len(base):
            continue
        metric = "success" if (t.startswith("mw-") and metric_mode == "primary") else "ret"
        R0 = float(base[metric].mean())
        if R0 == 0:
            continue
        for a in alphas:
            v = {}
            for c in CONDS:
                ec = e[(e.cond == c) & np.isclose(e.alpha, a)].set_index("ep")
                if len(ec):
                    v[c] = ec[metric]
            if not v:
                continue
            row = {"task": t, "alpha": a, "metric": metric, "R0": R0, "n_base": int(len(base))}
            for c, s in v.items():
                row[f"n_{c}"] = int(len(s))
                row[f"drop_{c}"] = float((R0 - s.mean()) / R0)
                row[f"mean_{c}"] = float(s.mean())
            if "A" in v:
                common = base.index.intersection(v["A"].index)
                row["dropA_ci"] = boot_ci(((base.loc[common, metric] - v["A"].loc[common]) / R0).values, seed=1)
            for x, y, name in (("B", "D1m", "B_minus_D1m"), ("A", "D1z", "A_minus_D1z"), ("A", "D3z", "A_minus_D3z"),
                               ("B", "D3m", "B_minus_D3m"), ("C", "A", "C_minus_A"), ("B", "A", "B_minus_A")):
                if x in v and y in v:
                    common = v[x].index.intersection(v[y].index)
                    d = ((v[y].loc[common] - v[x].loc[common]) / R0).values  # drop_x - drop_y per episode
                    row[f"{name}"] = float(d.mean())
                    row[f"{name}_ci"] = boot_ci(d, seed=2)
            for c in CONDS:
                if f"drop_{c}" in row:
                    pa = prep[t]["alphas"].get(akey(a), {}).get(c, {})
                    for k in ("err1", "err3", "errr"):
                        if k in pa:
                            row[f"{k}_{c}"] = pa[k]
                    if pa.get("frac") is not None:
                        row[f"frac_{c}"] = pa["frac"]["all"]
            rows.append(row)
    return rows


def verdict(rows):
    cells = [r for r in rows if "B_minus_D1m_ci" in r]
    notsig = [r for r in cells if not (r["B_minus_D1m_ci"][0] > 0)]
    sig = [r for r in cells if r["B_minus_D1m_ci"][0] > 0]
    elig = [r for r in rows if "drop_A" in r and "drop_C" in r and r["drop_A"] >= ELIG_DROP
            and r.get("dropA_ci", [0])[0] > 0]
    repro = [r for r in elig if r["drop_C"] >= C_REPRO * r["drop_A"]]
    sumA = sum(r["drop_A"] for r in elig)
    pooled = sum(r["drop_C"] for r in elig) / sumA if sumA > 0 else float("nan")
    pooled_B = sum(r["drop_B"] for r in elig) / sumA if sumA > 0 else float("nan")
    a1 = len(notsig) >= A1_CELLS
    a2 = len(elig) > 0 and len(repro) > len(elig) / 2
    i1 = len(sig) >= I1_CELLS
    i2 = len(elig) > 0 and pooled < I_RATIO
    v = "A(편향)" if (a1 and a2) else ("I(정보)" if (i1 and i2) else "혼합")
    cellid = lambda r: f"{r['task']}@{r['alpha']:g}"
    return {"verdict": v, "n_cells": len(cells),
            "A1_notsig_cells": len(notsig), "A1": a1, "A2_eligible": [cellid(r) for r in elig],
            "A2_reproduced": [cellid(r) for r in repro], "A2": a2,
            "I1_sig_cells": [cellid(r) for r in sig], "I1": i1, "I2_pooled_C_over_A": pooled, "I2": i2,
            "share_C_over_A": {cellid(r): r["drop_C"] / r["drop_A"] for r in elig},
            "share_B_over_A": {cellid(r): r["drop_B"] / r["drop_A"] for r in elig},
            "pooled_B_over_A": pooled_B,
            "median_C_over_A": float(np.median([r["drop_C"] / r["drop_A"] for r in elig])) if elig else float("nan")}


def secondary_counts(rows):
    out = {}
    for name in ("A_minus_D1z", "A_minus_D3z", "B_minus_D3m", "B_minus_D1m", "C_minus_A", "B_minus_A"):
        cs = [r for r in rows if f"{name}_ci" in r]
        out[name] = {"cells": len(cs), "pos_sig": sum(r[f"{name}_ci"][0] > 0 for r in cs),
                     "neg_sig": sum(r[f"{name}_ci"][1] < 0 for r in cs),
                     "mean": float(np.mean([r[name] for r in cs])) if cs else float("nan")}
    return out


def decision_tables(R, tasks):
    dec = [pd.read_csv(f"{R}/decision_{t}.csv") for t in tasks if os.path.exists(f"{R}/decision_{t}.csv")]
    if not dec:
        return None
    dec = pd.concat(dec, ignore_index=True)
    dec["alpha"] = dec["alpha"].astype(float)
    rows = []
    for (t, s, c, a), dd in dec.groupby(["task", "set", "cond", "alpha"]):
        if s == "plan":
            rows.append({"task": t, "set": s, "cond": c, "alpha": a, "n": len(dd), "dev_mean": dd.dev_mean.mean(),
                         "dev_act": dd.dev_act.mean()})
        else:
            rows.append({"task": t, "set": s, "cond": c, "alpha": a, "n": len(dd), "spearman": dd.spear.mean(),
                         "elite_jaccard": dd.jac.mean(), "argmax_agree": 1 - dd.flip.mean(),
                         "regret_n_mean": dd.regret_n.mean(), "w_overlap": dd.w_overlap.mean(),
                         "value_nerr_median": dd.nerr.median(), "value_nerr_centered_median": dd.nerr_c.median()})
    D = pd.DataFrame(rows)
    D.to_csv(f"{R}/table_decision.csv", index=False)
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
    get = lambda t, c: [(r["alpha"], r[f"drop_{c}"]) for r in rows if r["task"] == t and f"drop_{c}" in r]
    ylab = lambda t: "relative drop in " + ("success rate" if t.startswith("mw-") else "return")
    # 1) decomposition: A, B, C, D1m
    fig, axes = plt.subplots(1, len(T), figsize=(3.3 * len(T), 3.2), squeeze=False)
    for ax, t in zip(axes[0], T):
        style(ax)
        for c, mk in (("A", "o"), ("B", "s"), ("C", "D"), ("D1m", "^")):
            p = get(t, c)
            if p:
                ax.plot(*zip(*p), marker=mk, ms=5, lw=1.6, color=COL[c], label=LABEL[c])
        ax.axhline(0, color="#c3c2b7", lw=0.8)
        ax.set_xticks(alphas)
        ax.set_xlabel("α")
        ax.set_ylabel(ylab(t))
        ax.set_title(t, color=INK)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, loc="lower center", ncol=4)
    fig.suptitle("Bias or information? zero (A) vs mean-preserving (B) vs bias-only (C) vs matched random (D1m)",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(f"{R}/fig_decomposition.png", dpi=130)
    plt.close(fig)
    # 2) random controls: zero family and mean family
    fig, axes = plt.subplots(2, len(T), figsize=(3.3 * len(T), 5.4), squeeze=False, sharex=True)
    for j, t in enumerate(T):
        for i, fam in enumerate((("A", "D1z", "D3z"), ("B", "D1m", "D3m"))):
            ax = axes[i][j]
            style(ax)
            for c, mk in zip(fam, ("o", "s", "^")):
                p = get(t, c)
                if p:
                    ax.plot(*zip(*p), marker=mk, ms=5, lw=1.6, color=COL[c], label=LABEL[c])
            ax.axhline(0, color="#c3c2b7", lw=0.8)
            ax.set_xticks(alphas)
            if i == 0:
                ax.set_title(t, color=INK)
            if i == 1:
                ax.set_xlabel("α")
            if j == 0:
                ax.set_ylabel(("zero family" if i == 0 else "mean-preserving family") + "\nrelative drop")
        for i in range(2):
            axes[i][-1].legend(fontsize=7, frameon=False, loc="upper left")  # walker panels: upper left is empty
    fig.suptitle("Weak ablation vs random ablation matched on 1-step or 3-step error", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(f"{R}/fig_random_controls.png", dpi=130)
    plt.close(fig)
    # 3) final-plan deviation vs seed-noise floor
    if D is not None and len(D):
        P = D[D.set == "plan"]
        fig, axes = plt.subplots(1, len(T), figsize=(3.3 * len(T), 3.2), squeeze=False, sharey=True)
        for ax, t in zip(axes[0], T):
            style(ax)
            pt = P[P.task == t]
            for c, mk in (("A", "o"), ("B", "s"), ("C", "D"), ("D1m", "^"), ("D1z", "v")):
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
        fig.suptitle("Final-plan action deviation from the unablated planner (same seed)", fontsize=10, color=INK)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        fig.savefig(f"{R}/fig_plan_deviation.png", dpi=130)
        plt.close(fig)


def main(out, tasks, alphas):
    R = f"{out}/results"
    prep = {t: json.load(open(f"{R}/prep_{t}.json")) for t in tasks if os.path.exists(f"{R}/prep_{t}.json")}
    ep = pd.DataFrame([json.loads(l) for l in open(f"{R}/episodes.jsonl") if l.strip()])
    ep["alpha"] = ep["alpha"].astype(float)
    rows = cell_table(ep, prep, tasks, alphas, "primary")
    rows_ret = cell_table(ep, prep, tasks, alphas, "return")
    summary = {"rules": {"A1": f">= {A1_CELLS} of 12 cells with CI_lo(drop_B - drop_D1m) <= 0",
                         "A2": f"drop_C >= {C_REPRO} drop_A in > half of eligible cells (drop_A >= {ELIG_DROP}, CI excl 0)",
                         "I1": f">= {I1_CELLS} of 12 cells with CI_lo(drop_B - drop_D1m) > 0",
                         "I2": f"pooled sum drop_C / sum drop_A < {I_RATIO} over eligible cells"},
               "primary": verdict(rows), "secondary_return_metric": verdict(rows_ret),
               "secondary_counts": secondary_counts(rows), "secondary_counts_return": secondary_counts(rows_ret),
               "baselines": {t: {"ret": float(ep[(ep.task == t) & (ep.cond == "none")].ret.mean()),
                                 "ret_sd": float(ep[(ep.task == t) & (ep.cond == "none")].ret.std()),
                                 "succ": float(ep[(ep.task == t) & (ep.cond == "none")].success.mean()),
                                 "n": int(((ep.task == t) & (ep.cond == "none")).sum())} for t in tasks},
               "episodes_total": int(len(ep)), "cells": rows}
    D = decision_tables(R, tasks)
    pd.DataFrame([{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()} for r in rows]).to_csv(
        f"{R}/table_cells.csv", index=False)
    pd.DataFrame([{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()} for r in rows_ret]).to_csv(
        f"{R}/table_cells_return.csv", index=False)
    json.dump(summary, open(f"{R}/summary.json", "w"), indent=1, default=float)
    print(json.dumps({k: summary[k] for k in ("primary", "secondary_return_metric", "secondary_counts")}, indent=1,
                     default=float))
    figures(R, rows, D, tasks, alphas)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--alphas", default="0.1,0.2,0.3")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.out, a.tasks.split(","), [float(x) for x in a.alphas.split(",")])
