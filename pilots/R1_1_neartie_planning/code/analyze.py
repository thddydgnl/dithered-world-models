"""Phase E: pre-registered criteria S1-S3 (WM scope = primary verdict), amendment-1 scopes (Q, ALL), tables, figures.

Inputs (in <out>/results): prep_<task>.json, prep_q_<task>.json, episodes.jsonl (WM scope), episodes_q.jsonl (Q / ALL
scopes, amendment 1), decision_<task>.csv
Outputs: summary.json, table_main.csv, table_decision.csv, table_flip_quartiles.csv,
         fig_pred_vs_drop.png, fig_flip_by_gap.png, fig_frac_err_vs_alpha.png, fig_scopes.png

Usage: python analyze.py --tasks walker-walk,cheetah-run,... --alphas 0.05,0.1,0.2,0.3,0.5 --out /workspace/out_main
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

CONDS = ("weak", "rand_count", "rand_pred", "strong")
S1_FRAC, S1_ERR = 0.5, 0.10          # S1: >= 50% removed with normalized 1-step error <= 0.10, at some alpha >= 0.1
S2_RATIO, S2_FLOOR, S2B_DROP = 2.0, 0.05, 0.20
S3_RATIO, S3_MIN_FLIPS = 2.0, 10
S3_SET = "first"                      # primary candidate set for S3 (PLAN.md 5.1-3 / 6.1)
FINE_DROP = 0.10                      # "planning fine" for the other-direction verdict
Q_ALPHAS = (0.1, 0.2, 0.3)            # amendment 1: alphas run for the Q and ALL scopes
W_WM, W_Q = 12, 10                    # entries per MPPI iteration: WM 4 layers x 3 steps, Q 5 members x 2 layers

# reference categorical palette (dataviz skill, light mode); strong / seed-noise are gray reference series
COL = {"weak": "#2a78d6", "rand_pred": "#eb6834", "rand_count": "#1baf7a", "strong": "#898781",
       "seednoise": "#898781"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
LABEL = {"weak": "weak (|a|<α·m)", "rand_pred": "random, prediction-matched", "rand_count": "random, count-matched",
         "strong": "strong top-k (sanity)", "seednoise": "estimator seed noise"}


def akey(a):
    return f"{float(a):g}"


def is_mw(task):
    return task.startswith("mw-")


def boot_ci(x, n=4000, q=(5, 95), seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [float("nan"), float("nan")]
    b = np.random.default_rng(seed).choice(x, size=(n, len(x)), replace=True).mean(1)
    return [float(np.percentile(b, q[0])), float(np.percentile(b, q[1]))]


def load_episodes(R):
    rows = []
    for f in ("episodes.jsonl", "episodes_q.jsonl"):
        if os.path.exists(f"{R}/{f}"):
            rows += [json.loads(l) for l in open(f"{R}/{f}") if l.strip()]
    ep = pd.DataFrame(rows)
    if "scope" not in ep:
        ep["scope"] = "wm"
    ep["scope"] = ep["scope"].fillna("wm")
    ep["alpha"] = ep["alpha"].astype(float)
    return ep


def pred_metrics(scope, cond, a, prep, prep_q):
    """Prediction-side numbers of one condition: removed fraction and the error that S1 uses for this scope."""
    pa = prep["alphas"][akey(a)][cond]
    d = {"err1": pa["err1"], "err3": pa["err3"], "errr": pa["errr"], "frac_wm": pa["frac"]["all"],
         "frac_dyn": pa["frac"]["dyn"], "frac_rew": pa["frac"]["rew"]}
    if prep_q is not None and akey(a) in prep_q["alphas"] and cond in prep_q["alphas"][akey(a)]:
        pq = prep_q["alphas"][akey(a)][cond]
        d.update(errq=pq["errq"], frac_q=pq["frac_q"])
    if scope == "wm":
        d.update(frac=d["frac_wm"], err_s1=d["err1"])
    elif scope == "q":
        d.update(frac=d.get("frac_q"), err_s1=d.get("errq"))
    elif "frac_q" in d:  # all: both errors must be small; fraction over all six layer families weighted by entries
        d.update(frac=(W_WM * d["frac_wm"] + W_Q * d["frac_q"]) / (W_WM + W_Q), err_s1=max(d["err1"], d["errq"]))
    return d


def task_scope_summary(t, scope, e, prep, prep_q, alphas):
    base = e[(e.cond == "none")].sort_values("ep")
    metric = "success" if is_mw(t) else "ret"
    R0 = float(base[metric].mean())
    b0 = base.set_index("ep")[metric]
    ts = {"metric": metric, "R0": R0, "ret0": float(base.ret.mean()), "ret0_sd": float(base.ret.std()),
          "succ0": float(base.success.mean()), "n_eps": int(len(base)), "alpha": {}}
    es = e[e.scope == scope]
    conds = CONDS if scope == "wm" else ("weak", "rand_pred")
    rows = []
    for a in alphas:
        if akey(a) not in prep["alphas"]:
            continue
        ad = {}
        for c in conds:
            d = pred_metrics(scope, c, a, prep, prep_q)
            if c == "rand_pred":
                d.update(r_dyn=prep["alphas"][akey(a)][c]["r_dyn"], r_rew=prep["alphas"][akey(a)][c]["r_rew"])
                if prep_q is not None and akey(a) in prep_q["alphas"]:
                    d.update(r_q=prep_q["alphas"][akey(a)][c]["r_q"])
            ec = es[(es.cond == c) & (np.isclose(es.alpha, a))].sort_values("ep")
            if len(ec):
                m = float(ec[metric].mean())
                paired = (b0 - ec.set_index("ep")[metric]).dropna()
                fk = "all" if scope == "wm" else ("q" if scope == "q" else "all6")
                d.update(n=int(len(ec)), mean=m, ret=float(ec.ret.mean()), succ=float(ec.success.mean()),
                         ret_drop=(ts["ret0"] - float(ec.ret.mean())) / ts["ret0"],
                         plan_frac=float(np.nanmean([f.get(fk) if f.get(fk) is not None else np.nan for f in ec.frac])),
                         drop=(R0 - m) / R0 if R0 else float("nan"),
                         drop_ci90=[x / R0 for x in boot_ci(paired.values)] if R0 else None)
            ad[c] = d
            rows.append({"task": t, "scope": scope, "alpha": a, "cond": c,
                         **{k: v for k, v in d.items() if k != "drop_ci90"}})
        ts["alpha"][akey(a)] = ad
    A = [a for a in alphas if akey(a) in ts["alpha"]]
    ok = lambda a: (ts["alpha"][akey(a)]["weak"].get("frac") is not None
                    and ts["alpha"][akey(a)]["weak"].get("err_s1") is not None)
    s1 = [a for a in A if a >= 0.1 - 1e-9 and ok(a) and ts["alpha"][akey(a)]["weak"]["frac"] >= S1_FRAC
          and ts["alpha"][akey(a)]["weak"]["err_s1"] <= S1_ERR]
    ts["S1"] = bool(s1)
    ts["S1_alphas"] = s1
    ts["alpha_star"] = max(s1) if s1 else None

    def s2a_at(a):
        w, p = ts["alpha"][akey(a)]["weak"], ts["alpha"][akey(a)]["rand_pred"]
        if "drop" not in w or "drop" not in p or not np.isfinite(w["drop"]):
            return False, False
        return (bool(w["drop"] >= S2_RATIO * max(p["drop"], 0.0) and w["drop"] >= S2_FLOOR),
                bool(w["drop"] >= S2_RATIO * p["drop"] and w["drop"] > 0))

    ts["S2a"], ts["S2a_nofloor"] = s2a_at(ts["alpha_star"]) if ts["alpha_star"] is not None else (False, False)
    ts["S2b_alphas"] = [a for a in A if ok(a) and ts["alpha"][akey(a)]["weak"]["err_s1"] <= S1_ERR
                        and ts["alpha"][akey(a)]["weak"].get("drop", -1) >= S2B_DROP]
    ts["S2b"] = bool(ts["S2b_alphas"])
    okerr = [a for a in A if ok(a) and ts["alpha"][akey(a)]["weak"]["err_s1"] <= S1_ERR
             and "drop" in ts["alpha"][akey(a)]["weak"]]
    ts["alpha_err_ok"] = max(okerr) if okerr else None
    ts["S2prime"] = bool(s2a_at(ts["alpha_err_ok"])[0]) if okerr else False
    ts["planning_fine"] = (ts["alpha_star"] is not None and
                           ts["alpha"][akey(ts["alpha_star"])]["weak"].get("drop", 1) <= FINE_DROP)
    return ts, rows


def verdict_of(T, c):
    c["S1_tasks"] = [t for t in T if T[t]["S1"]]
    c["S1"] = len(c["S1_tasks"]) >= 2
    c["S2a_tasks"] = [t for t in T if T[t]["S2a"]]
    c["S2a_nofloor_tasks"] = [t for t in T if T[t]["S2a_nofloor"]]
    c["S2b_tasks"] = [t for t in T if T[t]["S2b"]]
    c["S2"] = len(c["S2a_tasks"]) >= 2 or len(c["S2b_tasks"]) >= 2
    c["S2prime_tasks"] = [t for t in T if T[t]["S2prime"]]
    c.setdefault("S3", False)
    fine = [T[t]["planning_fine"] for t in c["S1_tasks"]]
    if c["S1"] and (c["S2"] or c["S3"]):
        v = "있음"
    elif c["S1"] and fine and all(fine):
        v = "있음(다른 방향)"
    elif c["S1"]:
        v = "없음"
    else:
        v = "조건부 있음" if len(c["S2prime_tasks"]) >= 2 or len(c["S2b_tasks"]) >= 2 else "없음"
    c["verdict"] = v
    return c


def s3_for(Q, T, set_name, scope):
    """Flip rate by gap quartile for weak at alpha* (or 0.1), pooled over tasks; controls rand_pred and seednoise."""
    pooled = {k: [0, 0] for k in (1, 2, 3, 4)}
    ctrl = {c: {k: [0, 0] for k in (1, 2, 3, 4)} for c in ("rand_pred", "seednoise")}
    per_task = {}
    for t, ts in T.items():
        a3 = ts["alpha_star"] if ts["alpha_star"] is not None else 0.1
        per = {}
        for c, aa, sp in (("weak", a3, scope), ("rand_pred", a3, scope), ("seednoise", 0.0, "wm")):
            qq = Q[(Q.task == t) & (Q.set == set_name) & (Q.scope == sp) & (Q.cond == c) & np.isclose(Q.alpha, aa)]
            per[c] = {int(r.quartile): [int(r.flips), int(r.n)] for r in qq.itertuples()}
            for k, (fl, nn) in per[c].items():
                tgt = pooled if c == "weak" else ctrl[c]
                tgt[k][0] += fl
                tgt[k][1] += nn
        w = per["weak"]
        if w and 1 in w and 4 in w:
            r1, r4 = w[1][0] / w[1][1], w[4][0] / w[4][1]
            per_task[t] = {"alpha": a3, "flip_by_q": {k: w[k][0] / w[k][1] for k in w}, "flips_q1": w[1][0],
                           "holds": bool(r1 >= S3_RATIO * r4 and w[1][0] >= S3_MIN_FLIPS),
                           "controls_by_q": {c: {k: per[c][k][0] / per[c][k][1] for k in per[c]} for c in ctrl}}
    rate = lambda d, k: d[k][0] / d[k][1] if d[k][1] else float("nan")
    p1, p4 = rate(pooled, 1), rate(pooled, 4)
    return {"pooled": {"flip_by_quartile": {k: rate(pooled, k) for k in pooled}, "flips_q1": pooled[1][0],
                       "n_q1": pooled[1][1], "controls": {c: {k: rate(v, k) for k in v} for c, v in ctrl.items()}},
            "holds": bool(pooled[1][1] and p1 >= S3_RATIO * p4 and pooled[1][0] >= S3_MIN_FLIPS),
            "specific": bool(p1 > rate(ctrl["rand_pred"], 1) and p1 > rate(ctrl["seednoise"], 1)),
            "per_task": per_task}


def decision_tables(R, tasks):
    dec = [pd.read_csv(f"{R}/decision_{t}.csv") for t in tasks if os.path.exists(f"{R}/decision_{t}.csv")]
    if not dec:
        return None, None
    dec = pd.concat(dec, ignore_index=True)
    dec["alpha"] = dec["alpha"].astype(float)
    if "scope" not in dec:
        dec["scope"] = "wm"
    dec["scope"] = dec["scope"].fillna("wm")
    qrows, drows = [], []
    for t in dec.task.unique():
        dt = dec[dec.task == t]
        for set_name in ("first", "last"):
            ds = dt[dt.set == set_name]
            g = ds[ds.cond == "seednoise"].set_index("state")["gap"]
            qs = pd.qcut(g.rank(method="first"), 4, labels=[1, 2, 3, 4])  # Q1 = smallest gap
            for (sp, c, a), dd in ds.groupby(["scope", "cond", "alpha"]):
                dd = dd.set_index("state")
                q = qs.reindex(dd.index)
                fl = dd.groupby(q, observed=False)["flip"].agg(["mean", "sum", "count"])
                row = {"task": t, "set": set_name, "scope": sp, "cond": c, "alpha": a, "n_states": len(dd),
                       "spearman": dd.spear.mean(), "argmax_agree": 1 - dd.flip.mean(), "elite_jaccard": dd.jac.mean(),
                       "value_nerr_median": dd.nerr.median(), "value_nerr_centered_median": dd.nerr_c.median(),
                       **{f"flip_q{k}": fl.loc[k, "mean"] for k in (1, 2, 3, 4)}}
                if "regret" in dd:
                    row.update(regret_mean=dd.regret.mean(), regret_n_median=dd.regret_n.median(),
                               regret_n_mean=dd.regret_n.mean(), ent0_mean=dd.ent0.mean(), ent1_mean=dd.ent1.mean(),
                               dent_mean=dd.dent.mean(), w_overlap_mean=dd.w_overlap.mean())
                drows.append(row)
                for k in (1, 2, 3, 4):
                    qrows.append({"task": t, "set": set_name, "scope": sp, "cond": c, "alpha": a, "quartile": k,
                                  "flip_rate": fl.loc[k, "mean"], "flips": int(fl.loc[k, "sum"]),
                                  "n": int(fl.loc[k, "count"]), "gap_median": float(g[qs == k].median())})
        dp = dt[dt.set == "plan"]
        for (sp, c, a), dd in dp.groupby(["scope", "cond", "alpha"]):
            drows.append({"task": t, "set": "plan", "scope": sp, "cond": c, "alpha": a, "dev_act": dd.dev_act.mean(),
                          "dev_mean": dd.dev_mean.mean(), "n_states": len(dd)})
    D, Q = pd.DataFrame(drows), pd.DataFrame(qrows)
    D.to_csv(f"{R}/table_decision.csv", index=False)
    Q.to_csv(f"{R}/table_flip_quartiles.csv", index=False)
    return D, Q


def main(out, tasks, alphas):
    R = f"{out}/results"
    prep = {t: json.load(open(f"{R}/prep_{t}.json")) for t in tasks if os.path.exists(f"{R}/prep_{t}.json")}
    prep_q = {t: (json.load(open(f"{R}/prep_q_{t}.json")) if os.path.exists(f"{R}/prep_q_{t}.json") else None)
              for t in tasks}
    ep = load_episodes(R)
    D, Q = decision_tables(R, tasks)
    summary = {"scopes": {}, "thresholds": {
        "S1": f"alpha>=0.1, weak removed fraction (scope layers) >= {S1_FRAC}, scope prediction error <= {S1_ERR}",
        "S2a": f"at alpha*: drop_weak >= {S2_RATIO} x max(drop_randpred, 0) and drop_weak >= {S2_FLOOR}; >= 2 tasks",
        "S2b": f"some alpha: error <= {S1_ERR} and drop_weak >= {S2B_DROP}; >= 2 tasks",
        "S3": f"pooled over tasks at alpha* (or 0.1), {S3_SET}-iteration set: flip(Q1) >= {S3_RATIO} x flip(Q4), "
              f"Q1 flips >= {S3_MIN_FLIPS}"}}
    all_rows = []
    for scope in ("wm", "q", "all"):
        sa = list(alphas) if scope == "wm" else [a for a in alphas if np.isclose(a, Q_ALPHAS).any()]
        if scope != "wm" and not (ep.scope == scope).any():
            continue
        T = {}
        for t in tasks:
            if t not in prep or not ((ep.task == t) & (ep.cond == "none")).any():
                continue
            ts, rows = task_scope_summary(t, scope, ep[ep.task == t], prep[t], prep_q.get(t), sa)
            T[t] = ts
            all_rows += rows
        crit = {}
        if Q is not None and len(Q):
            for set_name, suffix in ((S3_SET, ""), ("last" if S3_SET == "first" else "first", "_secondary")):
                s3 = s3_for(Q, T, set_name, scope)
                crit["S3" + suffix + "_set"] = set_name
                crit["S3_pooled" + suffix] = s3["pooled"]
                crit["S3" + suffix] = s3["holds"]
                crit["S3_specific" + suffix] = s3["specific"]
                for t, v in s3["per_task"].items():
                    T[t]["S3_task" + suffix] = v
                    T[t]["S3_alpha"] = v["alpha"]
        summary["scopes"][scope] = {"tasks": T, "criteria": verdict_of(T, crit)}
    summary["verdict_primary_wm"] = summary["scopes"]["wm"]["criteria"]["verdict"]
    pd.DataFrame(all_rows).to_csv(f"{R}/table_main.csv", index=False)
    json.dump(summary, open(f"{R}/summary.json", "w"), indent=1, default=float)
    for sp, v in summary["scopes"].items():
        c = v["criteria"]
        print(sp, {k: c[k] for k in ("S1_tasks", "S2a_tasks", "S2a_nofloor_tasks", "S2b_tasks", "S2prime_tasks", "S3",
                                     "S3_specific", "S3_secondary", "verdict") if k in c})
    figures(R, summary, Q, alphas)


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=8)


def figures(R, summary, Q, alphas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "axes.titlesize": 10})
    T = summary["scopes"]["wm"]["tasks"]
    tasks = list(T)
    n = len(tasks)
    # 1) prediction error vs planning drop (WM scope)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.3), squeeze=False)
    for ax, t in zip(axes[0], tasks):
        style(ax)
        for c, mk in (("weak", "o"), ("rand_pred", "s"), ("rand_count", "^"), ("strong", "x")):
            pts = [(a, T[t]["alpha"][akey(a)][c]["err1"], T[t]["alpha"][akey(a)][c]["drop"]) for a in alphas
                   if akey(a) in T[t]["alpha"] and "drop" in T[t]["alpha"][akey(a)][c]]
            if not pts:
                continue
            aa, xs, ys = zip(*pts)
            xs = [max(x, 1e-5) for x in xs]
            ax.plot(xs, ys, marker=mk, ms=6, lw=1.5, color=COL[c], label=LABEL[c],
                    mfc="none" if c == "rand_count" else COL[c])
            if c == "weak":
                for a, x, y in zip(aa, xs, ys):
                    ax.annotate(f"α={a:g}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7, color=INK2)
        ax.axhline(0, color="#c3c2b7", lw=0.8)
        ax.axvline(S1_ERR, color="#c3c2b7", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("normalized 1-step latent error")
        ax.set_title(t, color=INK)
        ax.set_ylabel("relative drop in " + ("success rate" if T[t]["metric"] == "success" else "return"))
    axes[0][0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Planning loss vs prediction loss, dynamics+reward ablation (weak vs random at matched error)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(f"{R}/fig_pred_vs_drop.png", dpi=130)
    plt.close(fig)
    # 2) flip rate by gap quartile (WM scope, primary set)
    if Q is not None and len(Q):
        fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.0), squeeze=False)
        for ax, t in zip(axes[0], tasks):
            style(ax)
            a3 = T[t].get("S3_alpha", 0.1)
            for j, (c, aa) in enumerate((("weak", a3), ("rand_pred", a3), ("seednoise", 0.0))):
                qq = Q[(Q.task == t) & (Q.set == S3_SET) & (Q.scope == "wm") & (Q.cond == c)
                       & np.isclose(Q.alpha, aa)].sort_values("quartile")
                ax.bar(qq.quartile + (j - 1) * 0.26, qq.flip_rate, width=0.24, color=COL[c],
                       alpha=0.55 if c == "seednoise" else 1.0,
                       label=LABEL[c] + ("" if c == "seednoise" else f" α={aa:g}"))
            ax.set_xticks([1, 2, 3, 4], ["Q1\nsmallest gap", "Q2", "Q3", "Q4\nlargest"])
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("argmax flip rate")
            ax.set_title(t, color=INK)
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, fontsize=8, frameon=False, loc="lower center", ncol=3)
        fig.suptitle(f"Argmax flips by top1-top2 value-gap quartile ({S3_SET} MPPI iteration, 512 fixed candidates)",
                     fontsize=10, color=INK)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(f"{R}/fig_flip_by_gap.png", dpi=130)
        plt.close(fig)
    # 3) removed fraction and error vs alpha (WM scope)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    for i, t in enumerate(tasks):
        aa = [a for a in alphas if akey(a) in T[t]["alpha"]]
        axes[0].plot(aa, [T[t]["alpha"][akey(a)]["weak"]["frac"] for a in aa], marker="o", ms=5, lw=1.5,
                     color=cols[i % 4], label=t)
        axes[1].plot(aa, [T[t]["alpha"][akey(a)]["weak"]["err1"] for a in aa], marker="o", ms=5, lw=1.5,
                     color=cols[i % 4], label=t)
    for ax in axes:
        style(ax)
        ax.set_xlabel("α (threshold as fraction of per-unit 99.9th pct |a|)")
    axes[0].axhline(S1_FRAC, color="#c3c2b7", lw=0.8)
    axes[0].set_ylabel("fraction zeroed (weak, dyn+rew)")
    axes[1].axhline(S1_ERR, color="#c3c2b7", lw=0.8)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("normalized 1-step latent error (weak)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{R}/fig_frac_err_vs_alpha.png", dpi=130)
    plt.close(fig)
    # 4) scopes: drop vs alpha for weak and prediction-matched random, rows = scope
    scopes = [s for s in ("wm", "q", "all") if s in summary["scopes"]]
    if len(scopes) > 1:
        fig, axes = plt.subplots(len(scopes), n, figsize=(3.0 * n, 2.4 * len(scopes)), squeeze=False, sharey=True)
        names = {"wm": "dyn+rew", "q": "Q only", "all": "dyn+rew+Q"}
        for i, sp in enumerate(scopes):
            TS = summary["scopes"][sp]["tasks"]
            for j, t in enumerate(tasks):
                ax = axes[i][j]
                style(ax)
                if t not in TS:
                    continue
                for c in ("weak", "rand_pred"):
                    pts = [(a, TS[t]["alpha"][akey(a)][c]["drop"]) for a in Q_ALPHAS
                           if akey(a) in TS[t]["alpha"] and "drop" in TS[t]["alpha"][akey(a)][c]]
                    if pts:
                        ax.plot(*zip(*pts), marker="o" if c == "weak" else "s", ms=5, lw=1.5, color=COL[c],
                                label=LABEL[c])
                ax.axhline(0, color="#c3c2b7", lw=0.8)
                ax.set_xticks(Q_ALPHAS)
                if i == 0:
                    ax.set_title(t, color=INK)
                if j == 0:
                    ax.set_ylabel(f"{names[sp]}\nrelative drop")
                if i == len(scopes) - 1:
                    ax.set_xlabel("α")
        axes[0][0].legend(fontsize=7, frameon=False, loc="upper left")
        fig.suptitle("Planning drop by ablation scope (prediction-matched random within each scope)", fontsize=10,
                     color=INK)
        fig.tight_layout()
        fig.savefig(f"{R}/fig_scopes.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--alphas", default="0.05,0.1,0.2,0.3,0.5")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.out, a.tasks.split(","), [float(x) for x in a.alphas.split(",")])
