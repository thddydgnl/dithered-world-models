"""Phase E (R2-8): S1 (metric), S2 (prescription), S3 (consistency control) and the verdict (PLAN.md section 5).

Inputs
  <out>/results: episodes.jsonl (R2-8: none + Q conditions), prep_<task>.json (R2-8 prep), decision_<task>.csv (this pod)
  <inputs>/r23: table_cells.csv, table_decision.csv (R2-3, A5000 pod)
  <inputs>/r3b: table_cells_3b.csv, table_decision_3b.csv, prep_<task>.json (R2-3b, A40 pod)
Usage: python analyze_r28.py --tasks mw-push,mw-pick-place,cheetah-run,walker-walk --kbits 3,4,5 --out /workspace/out_main \
         --inputs /workspace/inputs
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

QVARS = ("det", "ind", "sh", "sub")
NBOOT = 4000
THR = {"ret": 0.20, "success": 0.30}   # S2: deterministic drop that counts as real
RED, S1_RHO, S1_MARGIN = 0.5, 0.6, 0.2
COL = {"det": "#2a78d6", "ind": "#eb6834", "sh": "#1baf7a", "sub": "#eda100", "seed": "#898781",
       "R2-3": "#2a78d6", "R2-3b": "#eb6834", "R2-8": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
LABEL = {"det": "(a) deterministic rounding", "ind": "(b) dither, independent per candidate",
         "sh": "(c) dither shared across candidates", "sub": "(b2) subtractive dither, independent"}


def boot_ci(x, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [float("nan"), float("nan")]
    b = np.random.default_rng(seed).choice(x, size=(NBOOT, len(x)), replace=True).mean(1)
    return [float(np.percentile(b, 5)), float(np.percentile(b, 95))]


def metric_of(task):
    return "success" if task.startswith("mw-") else "ret"


def r28_cells(ep, prep, tasks, kbits):
    rows = []
    for t in tasks:
        m = metric_of(t)
        base = ep[(ep.task == t) & (ep.cond == "none")].set_index("ep")[m]
        R0 = float(base.mean())
        for k in kbits:
            row = {"task": t, "k": k, "metric": m, "R0": R0}
            vals = {}
            for v in QVARS:
                c = f"Q{v}{k}"
                e = ep[(ep.task == t) & (ep.cond == c)].set_index("ep")[m]
                if len(e) and R0:
                    vals[v] = e
                    row[f"drop_{v}"] = float((R0 - e.mean()) / R0)
                    row[f"n_{v}"] = int(len(e))
                    ret = ep[(ep.task == t) & (ep.cond == c)].ret.mean()
                    row[f"retdrop_{v}"] = float((ep[(ep.task == t) & (ep.cond == "none")].ret.mean() - ret) /
                                                ep[(ep.task == t) & (ep.cond == "none")].ret.mean())
                pr = prep[t]["pred"].get(f"{c}@{k}")
                if pr:
                    row.update({f"err1_{v}": pr["err1"], f"err3_{v}": pr["err3"], f"errr_{v}": pr["errr"]})
            for x, y in (("det", "ind"), ("sh", "ind"), ("det", "sub"), ("det", "sh")):
                if x in vals and y in vals:
                    c_ = vals[x].index.intersection(vals[y].index)
                    d = ((vals[y].loc[c_] - vals[x].loc[c_]) / R0).values  # drop_x - drop_y per episode
                    row[f"{x}-{y}"] = float(d.mean())
                    row[f"{x}-{y}_ci"] = boot_ci(d, seed=3)
            rows.append(row)
    return rows


def s2_s3(rows, tasks, kbits, dither="ind"):
    out = {"per_task": {}}
    n2 = n3 = 0
    for t in tasks:
        rt = sorted([r for r in rows if r["task"] == t], key=lambda r: r["k"])
        thr = THR[metric_of(t)]
        qual = [r for r in rt if r.get("drop_det", -1) >= thr]
        kstar = max(r["k"] for r in qual) if qual else None
        rs = [r for r in rt if r["k"] == (kstar if kstar is not None else min(kbits))][0]
        info = {"qualifying_k": [r["k"] for r in qual], "k_star": kstar}
        ok2 = False
        if kstar is not None:
            dd, di = rs["drop_det"], rs.get(f"drop_{dither}", float("nan"))
            red = (dd - di) / dd if dd > 0 else float("nan")
            e_ok = rs.get(f"err1_{dither}", 0) >= rs.get("err1_det", float("inf"))
            ok2 = bool(red >= RED and e_ok)
            info.update(drop_det=dd, drop_dither=di, reduction=red, err1_det=rs.get("err1_det"),
                        err1_dither=rs.get(f"err1_{dither}"), err1_not_smaller=bool(e_ok))
        c3 = rs.get("sh-ind_ci", [float("nan")] * 2)
        ok3 = bool(c3[0] > 0)
        info.update(S2_task=ok2, S3_k=rs["k"], sh_minus_ind=rs.get("sh-ind"), sh_minus_ind_ci=c3, S3_task=ok3)
        out["per_task"][t] = info
        n2 += ok2
        n3 += ok3
    out["S2"], out["S3"], out["S2_tasks"], out["S3_tasks"] = n2 >= 2, n3 >= 2, n2, n3
    return out


def s1_table(R, inputs, tasks, kbits, rows28, ep):
    dec = pd.concat([pd.read_csv(f"{R}/decision_{t}.csv") for t in tasks if os.path.exists(f"{R}/decision_{t}.csv")],
                    ignore_index=True)
    dec["alpha"] = dec["alpha"].astype(float)
    first = dec[dec.set == "first"]
    seed_dmu = first[first.cond == "seednoise"].groupby("task").dmu.mean().to_dict()
    cvb = first.groupby(["task", "cond", "alpha"]).dmu.mean().to_dict()
    plan = dec[dec.set == "plan"].groupby(["task", "cond", "alpha"]).dev_mean.mean().to_dict()
    seed_dev_new = dec[(dec.set == "plan") & (dec.cond == "seednoise")].groupby("task").dev_mean.mean().to_dict()
    cells = []
    # R2-3 (A5000 pod): drops and errors from its tables, CVB recomputed here
    c23 = pd.read_csv(f"{inputs}/r23/table_cells.csv")
    d23 = pd.read_csv(f"{inputs}/r23/table_decision.csv")
    seed23 = d23[(d23.set == "plan") & (d23.cond == "seednoise")].groupby("task").dev_mean.mean().to_dict()
    for r in c23.itertuples():
        for c in ("A", "B", "C", "D1z", "D3z", "D1m", "D3m"):
            if pd.isna(getattr(r, f"drop_{c}", np.nan)):
                continue
            dv = d23[(d23.task == r.task) & (d23.set == "plan") & (d23.cond == c) & np.isclose(d23.alpha, r.alpha)].dev_mean
            cells.append({"source": "R2-3", "task": r.task, "cond": c, "alpha": r.alpha, "drop": getattr(r, f"drop_{c}"),
                          "err1": getattr(r, f"err1_{c}"), "err3": getattr(r, f"err3_{c}"),
                          "dev": float(dv.iloc[0]) if len(dv) else np.nan, "dev_seed": seed23.get(r.task, np.nan),
                          "cvb": cvb.get((r.task, c, float(r.alpha)), np.nan)})
    # R2-3b (A40 pod): new deterministic controls, seed-averaged
    c3b = pd.read_csv(f"{inputs}/r3b/table_cells_3b.csv")
    d3b = pd.read_csv(f"{inputs}/r3b/table_decision_3b.csv")
    seed3b = d3b[(d3b.set == "plan") & (d3b.cond == "seednoise")].groupby("task").dev_mean.mean().to_dict()
    for r in c3b.itertuples():
        p3 = json.load(open(f"{inputs}/r3b/prep_{r.task}.json"))["alphas"][f"{float(r.alpha):g}"]
        for fam in ("Dfix1", "Dfix3", "Ddir1"):
            if pd.isna(getattr(r, f"drop_{fam}", np.nan)):
                continue
            dv = d3b[(d3b.task == r.task) & (d3b.set == "plan") & (d3b.cond == fam) & np.isclose(d3b.alpha, r.alpha)].dev_mean
            cv = [cvb.get((r.task, f"{fam}_s{s}", float(r.alpha)), np.nan) for s in (0, 1, 2)]
            cells.append({"source": "R2-3b", "task": r.task, "cond": fam, "alpha": r.alpha, "drop": getattr(r, f"drop_{fam}"),
                          "err1": float(np.mean([p3[f"{fam}_s{s}"]["err1"] for s in (0, 1, 2)])),
                          "err3": float(np.mean([p3[f"{fam}_s{s}"]["err3"] for s in (0, 1, 2)])),
                          "dev": float(dv.iloc[0]) if len(dv) else np.nan, "dev_seed": seed3b.get(r.task, np.nan),
                          "cvb": float(np.nanmean(cv))})
    # R2-8 (this pod)
    for r in rows28:
        for v in QVARS:
            if f"drop_{v}" not in r:
                continue
            c = f"Q{v}{r['k']}"
            cells.append({"source": "R2-8", "task": r["task"], "cond": c, "alpha": float(r["k"]), "drop": r[f"drop_{v}"],
                          "err1": r.get(f"err1_{v}"), "err3": r.get(f"err3_{v}"),
                          "dev": plan.get((r["task"], c, float(r["k"])), np.nan), "dev_seed": seed_dev_new.get(r["task"], np.nan),
                          "cvb": cvb.get((r["task"], c, float(r["k"])), np.nan)})
    C = pd.DataFrame(cells)
    C["cvb_norm"] = C.cvb / C.task.map(seed_dmu)
    C["dev_norm"] = C.dev / C.dev_seed
    C.to_csv(f"{R}/table_s1_cells.csv", index=False)
    rho = lambda d, col: float(d[["drop", col]].dropna().corr(method="spearman").iloc[0, 1]) if d[col].notna().sum() > 3 else np.nan
    res = {"n_cells": int(len(C)), "pooled": {col: rho(C, col) for col in ("cvb_norm", "cvb", "err1", "err3", "dev_norm", "dev")},
           "per_task": {t: {col: rho(C[C.task == t], col) for col in ("cvb_norm", "err1", "err3", "dev_norm")} for t in tasks},
           "per_source": {s: {col: rho(C[C.source == s], col) for col in ("cvb_norm", "err1", "err3", "dev_norm")}
                          for s in C.source.unique()}}
    p = res["pooled"]
    res["S1"] = bool(p["cvb_norm"] >= S1_RHO and p["cvb_norm"] >= p["err1"] + S1_MARGIN)
    res["S1_dev_secondary"] = bool(p["dev_norm"] >= S1_RHO and p["dev_norm"] >= p["err1"] + S1_MARGIN)
    res["seed_dmu"] = seed_dmu
    return res, C, dec


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=8)


def figures(R, rows28, C, dec, tasks, kbits, s1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "axes.titlesize": 10})
    n = len(tasks)
    # 1) the striking claim: 1-step error vs planning drop, per variant across k
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.3), squeeze=False)
    for ax, t in zip(axes[0], tasks):
        style(ax)
        rt = sorted([r for r in rows28 if r["task"] == t], key=lambda r: r["k"])
        for v, mk in (("det", "o"), ("ind", "s"), ("sh", "D"), ("sub", "^")):
            pts = [(r[f"err1_{v}"], r[f"drop_{v}"], r["k"]) for r in rt if f"drop_{v}" in r and r.get(f"err1_{v}")]
            if pts:
                xs, ys, ks = zip(*pts)
                ax.plot(xs, ys, marker=mk, ms=5, lw=1.5, color=COL[v], label=LABEL[v])
                if v == "det":
                    for x, y, k in zip(xs, ys, ks):
                        ax.annotate(f"k={k}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7, color=INK2)
        ax.axhline(0, color="#c3c2b7", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("normalized 1-step latent error")
        ax.set_ylabel("relative drop in " + ("success rate" if t.startswith("mw-") else "return"))
        ax.set_title(t, color=INK)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, loc="lower center", ncol=4)
    fig.suptitle("k-bit activation quantization: prediction error vs planning drop (points: k)", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(f"{R}/fig_quant_err_vs_drop.png", dpi=130)
    plt.close(fig)
    # 2) S1: drop vs metrics over all cells
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, col, name in zip(axes, ("cvb_norm", "err1", "dev_norm"),
                             ("candidate-consistent value bias\n(elite-mean action shift / seed floor)",
                              "normalized 1-step latent error", "final-plan deviation / seed floor")):
        style(ax)
        for s in ("R2-3", "R2-3b", "R2-8"):
            d = C[C.source == s]
            ax.scatter(d[col], d["drop"], s=14, color=COL[s], alpha=0.8, label=s, edgecolors="none")
        ax.set_xscale("log")
        ax.set_xlabel(name)
        ax.set_title(f"Spearman ρ = {s1['pooled'][col]:.2f}", color=INK)
    axes[0].set_ylabel("relative planning drop")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle(f"Which metric predicts planning damage? ({s1['n_cells']} cells: R2-3, R2-3b, R2-8)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(f"{R}/fig_s1_metrics.png", dpi=130)
    plt.close(fig)
    # 3) final-plan deviation per variant vs k, with the seed floor
    P = dec[dec.set == "plan"].groupby(["task", "cond", "alpha"]).dev_mean.mean().reset_index()
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.1), squeeze=False, sharey=True)
    for ax, t in zip(axes[0], tasks):
        style(ax)
        for v, mk in (("det", "o"), ("ind", "s"), ("sh", "D"), ("sub", "^")):
            q = P[(P.task == t) & (P.cond.isin([f"Q{v}{k}" for k in kbits]))].sort_values("alpha")
            if len(q):
                ax.plot(q.alpha, q.dev_mean, marker=mk, ms=5, lw=1.5, color=COL[v], label=LABEL[v])
        sn = P[(P.task == t) & (P.cond == "seednoise")]
        if len(sn):
            ax.axhline(float(sn.dev_mean.iloc[0]), color=COL["seed"], lw=1.2, label="seed-noise floor")
        ax.set_xticks(kbits)
        ax.set_xlabel("bits k")
        ax.set_title(t, color=INK)
    axes[0][0].set_ylabel("RMS deviation of the final MPPI mean")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, loc="lower center", ncol=3)
    fig.suptitle("Final-plan deviation under quantization (same seed)", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(f"{R}/fig_quant_plan_deviation.png", dpi=130)
    plt.close(fig)


def main(out, inputs, tasks, kbits):
    R = f"{out}/results"
    ep = pd.DataFrame([json.loads(l) for l in open(f"{R}/episodes.jsonl") if l.strip()])
    prep = {t: json.load(open(f"{R}/prep_{t}.json")) for t in tasks}
    rows28 = r28_cells(ep, prep, tasks, kbits)
    s23 = s2_s3(rows28, tasks, kbits, "ind")
    s23_sub = s2_s3(rows28, tasks, kbits, "sub")
    s1, C, dec = s1_table(R, inputs, tasks, kbits, rows28, ep)
    S2, S3, S1 = s23["S2"], s23["S3"], s1["S1"]
    if S2 and S3:
        verdict = "강한 있음" if S1 else "있음(처방)"
    elif S1 and not S2:
        verdict = "조건부(지표 논문)"
    elif S2 and not S3:
        verdict = "조건부 있음(처방 효과, 일관성 기전 미확인)"
    elif S3 and not S2:
        verdict = "조건부(일관성 대조만)"
    else:
        verdict = "없음"
    summary = {"verdict": verdict, "S1": s1, "S2_S3": s23, "S2_S3_subtractive_secondary": s23_sub, "cells_r28": rows28,
               "baselines": {t: {"ret": float(ep[(ep.task == t) & (ep.cond == "none")].ret.mean()),
                                 "succ": float(ep[(ep.task == t) & (ep.cond == "none")].success.mean()),
                                 "n": int(((ep.task == t) & (ep.cond == "none")).sum())} for t in tasks}}
    pd.DataFrame([{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()} for r in rows28]).to_csv(
        f"{R}/table_quant_cells.csv", index=False)
    json.dump(summary, open(f"{R}/summary.json", "w"), indent=1, default=float)
    print(json.dumps({"verdict": verdict, "S1": {k: s1[k] for k in ("S1", "S1_dev_secondary", "pooled", "n_cells")},
                      "S2_S3": s23, "S2_S3_sub": {k: s23_sub[k] for k in ("S2", "S3", "S2_tasks")}}, indent=1, default=float))
    figures(R, rows28, C, dec, tasks, kbits, s1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--kbits", default="3,4,5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", default="/workspace/inputs")
    a = ap.parse_args()
    main(a.out, a.inputs, a.tasks.split(","), [int(x) for x in a.kbits.split(",")])
