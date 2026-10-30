"""Phase E (R2-14): iteration dose-response of the planning drop, S1-S3 and the verdict (PLAN.md section 5),
plan deviation and wall-clock tables, two figures.

Inputs (<out>/results): episodes.jsonl (none + Q{v}{k} at every iteration count), plandev_<task>.csv (optional),
timing.json (optional).
Drop of variant v at iteration count n: per episode d_i = (base_i(n) - x_v,i(n)) / R0(n), where base is the
unquantized planner with the same n and the same seed and R0(n) its mean. One bootstrap index matrix per task
(episodes resampled jointly across n and variants, so cross-n differences are paired by seed).
Usage: python analyze_r214.py --cells mw-push:3,cheetah-run:4 --iters 1,3,6,12 --out /workspace/out_main
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

VARS = ("det", "shstep", "shfwd")
NAME = {"det": "det", "shstep": "sh-step", "shfwd": "sh-fwd"}
NBOOT, BOOT_SEED = 4000, 3
# fixed colors from R2-8b (reference palette slots 1, 4, 3)
COL = {"det": "#2a78d6", "shstep": "#eda100", "shfwd": "#1baf7a", "seed": "#898781"}
MARK = {"det": "o", "shstep": "^", "shfwd": "D"}
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def metric_of(task):
    return "success" if task.startswith("mw-") else "ret"


def pct(b):
    b = b[np.isfinite(b)]
    return [float(np.percentile(b, 5)), float(np.percentile(b, 95))] if len(b) else [float("nan")] * 2


def task_stats(ep, task, k, iters, metric):
    sub = ep[ep.task == task]
    conds = ["none"] + [f"Q{v}{k}" for v in VARS]
    piv = sub.pivot_table(index="ep", columns=["iters", "cond"], values=metric, aggfunc="first")
    cols = [(n, c) for n in iters for c in conds if (n, c) in piv.columns]
    piv = piv[cols].dropna()
    m = len(piv)
    idx = np.random.default_rng(BOOT_SEED).integers(0, m, size=(NBOOT, m))
    out = {"n": m, "metric": metric, "R0": {}, "R0_ci": {}, "drop": {}, "drop_ci": {}, "d": {}}
    for n in iters:
        base = piv[(n, "none")].values
        R0 = float(base.mean())
        out["R0"][n] = R0
        out["R0_ci"][n] = pct(base[idx].mean(1))
        for v in VARS:
            if (n, f"Q{v}{k}") not in piv.columns:
                continue
            d = (base - piv[(n, f"Q{v}{k}")].values) / R0
            out["d"][(n, v)] = d
            out["drop"].setdefault(v, {})[n] = float(d.mean())
            out["drop_ci"].setdefault(v, {})[n] = pct(d[idx].mean(1))
    lo, hi = min(iters), max(iters)
    out["inc"], out["inc_ci"] = {}, {}
    for v in VARS:
        if (lo, v) in out["d"] and (hi, v) in out["d"]:
            x = out["d"][(hi, v)] - out["d"][(lo, v)]
            out["inc"][v] = float(x.mean())
            out["inc_ci"][v] = pct(x[idx].mean(1))
    out["step_minus_fwd"], out["step_minus_fwd_ci"] = {}, {}
    for n in iters:
        if (n, "shstep") in out["d"] and (n, "shfwd") in out["d"]:
            x = out["d"][(n, "shstep")] - out["d"][(n, "shfwd")]
            out["step_minus_fwd"][n] = float(x.mean())
            out["step_minus_fwd_ci"][n] = pct(x[idx].mean(1))
    if (lo, "shstep") in out["d"] and (hi, "shstep") in out["d"]:
        x = (out["d"][(hi, "shstep")] - out["d"][(hi, "shfwd")]) - (out["d"][(lo, "shstep")] - out["d"][(lo, "shfwd")])
        out["step_minus_fwd_inc"] = float(x.mean())
        out["step_minus_fwd_inc_ci"] = pct(x[idx].mean(1))
    # baseline change with iterations (H3): per-episode base(hi) - base(lo), in units of R0(lo)
    b = (piv[(hi, "none")].values - piv[(lo, "none")].values) / out["R0"][lo]
    out["base_change_rel"], out["base_change_rel_ci"] = float(b.mean()), pct(b[idx].mean(1))
    out.pop("d")
    return out


def criteria(stats, iters):
    lo, hi = min(iters), max(iters)
    per, n1, n1lit, n2, n3 = {}, 0, 0, 0, 0
    for t, s in stats.items():
        D = s["drop"]["det"]
        seq = [D[n] for n in iters]
        mono = bool(all(b >= a for a, b in zip(seq, seq[1:])) and seq[-1] > seq[0])
        twice = bool(D[hi] > 0 and D[hi] >= 2 * max(D[lo], 0.0))
        real_hi = bool(s["drop_ci"]["det"][hi][0] > 0)
        s1_lit = mono or twice
        s1 = bool(s1_lit and real_hi)
        inc_det, inc_fwd, inc_step = s["inc"]["det"], s["inc"]["shfwd"], s["inc"]["shstep"]
        s2 = bool(inc_det > 0 and inc_fwd <= inc_det / 3.0)
        s3 = bool(inc_det > 0 and inc_step > 0)
        per[t] = {"det_drops": {n: D[n] for n in iters}, "monotone": mono, "twice": twice, "det_drop_hi_real": real_hi,
                  "S1_task_literal": s1_lit, "S1_task": s1, "inc_det": inc_det, "inc_det_ci": s["inc_ci"]["det"],
                  "inc_shfwd": inc_fwd, "inc_shfwd_ci": s["inc_ci"]["shfwd"], "S2_ratio": inc_fwd / inc_det if inc_det else None,
                  "S2_task": s2, "inc_shstep": inc_step, "inc_shstep_ci": s["inc_ci"]["shstep"], "S3_task": s3}
        n1 += s1
        n1lit += s1_lit
        n2 += s2
        n3 += s3
    k = len(stats)
    S1, S2, S3 = n1 == k, n2 == k, n3 == k
    verdict = "있음" if (S1 and S2) else ("조건부 있음" if S1 else "없음")
    return {"per_task": per, "S1": S1, "S1_tasks": n1, "S1_literal_tasks": n1lit, "S2": S2, "S2_tasks": n2,
            "S3": S3, "S3_tasks": n3, "verdict": verdict}


def plandev_table(R, task, k, iters):
    f = f"{R}/plandev_{task}.csv"
    if not os.path.exists(f):
        return {}
    d = pd.read_csv(f)
    out = {}
    for n in iters:
        dn = d[d.iters == n]
        sd = float(dn[dn.cond == "seednoise"].dev_mean.mean())
        out[n] = {"seed_dev_mean": sd}
        for v in VARS:
            x = dn[dn.cond == f"Q{v}{k}"]
            if len(x):
                out[n][v] = {"dev_mean": float(x.dev_mean.mean()), "dev_norm": float(x.dev_mean.mean() / sd),
                             "dev_act": float(x.dev_act.mean())}
    return out


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, labelcolor=INK2)


def figures(R, stats, pdev, order, kmap, iters):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "axes.titlesize": 10,
                         "figure.facecolor": SURF})
    handles = [Line2D([0], [0], marker=MARK[v], color=COL[v], lw=2, ms=8, markeredgecolor=SURF, markeredgewidth=1.5,
                      label=NAME[v]) for v in VARS]
    dodge = {"det": -0.045, "shstep": 0.0, "shfwd": 0.045}
    fig, axes = plt.subplots(1, len(order), figsize=(4.4 * len(order), 3.5), squeeze=False)
    for ax, t in zip(axes[0], order):
        s = stats[t]
        style(ax)
        ax.axhline(0, color=AXIS, lw=1)
        for v in VARS:
            xs = np.array(iters, float) * (2 ** dodge[v])
            ys = np.array([100 * s["drop"][v][n] for n in iters])
            lo = np.array([100 * s["drop_ci"][v][n][0] for n in iters])
            hi = np.array([100 * s["drop_ci"][v][n][1] for n in iters])
            ax.vlines(xs, lo, hi, color=COL[v], lw=2, alpha=0.5)
            ax.plot(xs, ys, marker=MARK[v], color=COL[v], lw=2, ms=8, markeredgecolor=SURF, markeredgewidth=1.5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(iters)
        ax.set_xticklabels([str(n) for n in iters])
        ax.set_xlabel("MPPI iterations per control step")
        ax.set_ylabel("success-rate drop (%p)" if s["metric"] == "success" else "return drop (%)")
        ax.set_title(f"{t}  (k={kmap[t]})", color=INK)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"Drop vs. unquantized planner with the same iterations ({stats[order[0]]['n']} paired episodes, 90% CI)",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(f"{R}/fig_dose_response.png", dpi=130)
    plt.close(fig)
    if not all(pdev.get(t) for t in order):
        return
    fig, axes = plt.subplots(1, len(order), figsize=(4.4 * len(order), 3.5), squeeze=False)
    for ax, t in zip(axes[0], order):
        style(ax)
        for v in VARS:
            ys = [pdev[t][n][v]["dev_norm"] for n in iters]
            ax.plot(iters, ys, marker=MARK[v], color=COL[v], lw=2, ms=8, markeredgecolor=SURF, markeredgewidth=1.5)
        ax.axhline(1, color=COL["seed"], lw=1.2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(iters)
        ax.set_xticklabels([str(n) for n in iters])
        ax.set_xlabel("MPPI iterations per control step")
        ax.set_ylabel("final-plan deviation / seed-noise floor")
        ax.set_title(f"{t}  (k={kmap[t]})", color=INK)
    fig.legend(handles=handles + [Line2D([0], [0], color=COL["seed"], lw=1.2, label="seed-noise floor (=1)")],
               loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("One-step plan reruns from 150 stored states: deviation of the final MPPI mean", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(f"{R}/fig_plandev_vs_iters.png", dpi=130)
    plt.close(fig)


def main(out, cells_arg, iters):
    R = f"{out}/results"
    order = [c.split(":")[0] for c in cells_arg.split(",")]
    kmap = {c.split(":")[0]: int(c.split(":")[1]) for c in cells_arg.split(",")}
    ep = pd.DataFrame([json.loads(l) for l in open(f"{R}/episodes.jsonl") if l.strip()])
    stats = {t: task_stats(ep, t, kmap[t], iters, metric_of(t)) for t in order}
    sec = {t: task_stats(ep, t, kmap[t], iters, "ret") for t in order if t.startswith("mw-")}
    crit = criteria(stats, iters)
    pdev = {t: plandev_table(R, t, kmap[t], iters) for t in order}
    timing = json.load(open(f"{R}/timing.json")) if os.path.exists(f"{R}/timing.json") else []
    grid_ms = ep.groupby(["task", "iters", "cond"]).plan_ms_per_step.mean().reset_index().to_dict("records")
    checks = ep.groupby(["task", "iters", "cond"]).agg(cfg_it=("cfg_iterations", "unique"),
                                                         refresh=("n_refresh", "unique")).reset_index()
    checks = [{**r, "cfg_it": [int(x) for x in r["cfg_it"]], "refresh": [int(x) for x in r["refresh"]]}
              for r in checks.to_dict("records")]
    summary = {"verdict": crit["verdict"], "criteria": crit, "stats": stats, "secondary_return_mw": sec,
               "plandev": pdev, "timing_single_process": timing, "grid_plan_ms_per_step": grid_ms, "checks": checks}
    json.dump(summary, open(f"{R}/summary.json", "w"), indent=1, default=float)
    rows = []
    for t in order:
        s = stats[t]
        for n in iters:
            r = {"task": t, "k": kmap[t], "iters": n, "R0": s["R0"][n], "R0_lo": s["R0_ci"][n][0], "R0_hi": s["R0_ci"][n][1]}
            for v in VARS:
                r[f"drop_{v}"] = s["drop"][v][n]
                r[f"drop_{v}_lo"], r[f"drop_{v}_hi"] = s["drop_ci"][v][n]
                if pdev.get(t):
                    r[f"devnorm_{v}"] = pdev[t][n][v]["dev_norm"]
            r["step_minus_fwd"] = s["step_minus_fwd"].get(n)
            r["step_minus_fwd_lo"], r["step_minus_fwd_hi"] = s["step_minus_fwd_ci"].get(n, [None, None])
            rows.append(r)
    pd.DataFrame(rows).to_csv(f"{R}/table_dose.csv", index=False)
    print(json.dumps({kk: crit[kk] for kk in crit if kk != "per_task"}, indent=1, ensure_ascii=False))
    for t in order:
        s = stats[t]
        print(f"{t}: n={s['n']} baseline " + ", ".join(f"it{n} {s['R0'][n]:.3f}" for n in iters))
        for v in VARS:
            print(f"   {NAME[v]:<8} " + "  ".join(f"it{n} {100 * s['drop'][v][n]:.1f} [{100 * s['drop_ci'][v][n][0]:.1f},"
                                                   f"{100 * s['drop_ci'][v][n][1]:.1f}]" for n in iters)
                  + f" | inc {100 * s['inc'][v]:.1f} [{100 * s['inc_ci'][v][0]:.1f},{100 * s['inc_ci'][v][1]:.1f}]")
        print("   sh-step - sh-fwd: " + "  ".join(f"it{n} {100 * s['step_minus_fwd'][n]:.1f} [{100 * s['step_minus_fwd_ci'][n][0]:.1f},"
                                                  f"{100 * s['step_minus_fwd_ci'][n][1]:.1f}]" for n in iters))
        print("   " + json.dumps(crit["per_task"][t], default=float))
        if pdev.get(t):
            print("   plandev/seed: " + "  ".join(f"it{n} " + "/".join(f"{pdev[t][n][v]['dev_norm']:.2f}" for v in VARS)
                                                for n in iters))
    figures(R, stats, pdev, order, kmap, iters)
    print("ANALYZE_DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--iters", default="1,3,6,12")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.out, a.cells, [int(x) for x in a.iters.split(",")])
