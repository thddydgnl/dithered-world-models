"""Metrics, pre-registered criteria (PLAN.md section 5) and figures for R2-5.
Usage: python analyze.py <prep_dir> <run_dir> [--nofig]
Writes <run_dir>/summary.json, table_conditions.csv, table_flip_quartiles.csv, table_criteria.csv and figures.
"""
import csv
import json
import sys

import numpy as np
from scipy.stats import kendalltau

PREP, RUN = sys.argv[1], sys.argv[2]
NOFIG = "--nofig" in sys.argv
ALPHAS = [0.05, 0.1, 0.2]
NB_BOOT = 2000
rng = np.random.default_rng(0)

R = np.load(f"{RUN}/eval_raw.npz", allow_pickle=True)
E, DN, MF, DEN, SPR = R["E"], R["DN"], R["MF"], R["DEN"], R["SPR"]
names = [str(x) for x in R["cond_names"]]
nW = E.shape[0]
ci = {n: i for i, n in enumerate(names)}
match = json.load(open(f"{RUN}/match.json"))
stats = json.load(open(f"{RUN}/stats.json"))
prep = json.load(open(f"{PREP}/prep.json"))
NEL = 256 * 1408


# ----------------------------------------------------------------------------------------------- per-instance metrics
def inst_metrics(c):
    """Per (window, k) arrays for condition index c vs base (index 0)."""
    tau = np.full((nW, 2), np.nan)
    flip = np.full((nW, 2), np.nan)
    rank = np.full((nW, 2), np.nan)
    for w in range(nW):
        for k in range(2):
            e0, e1 = E[w, 0, k], E[w, c, k]
            if np.isnan(e1).any():
                continue
            tau[w, k] = kendalltau(e0, e1)[0]
            flip[w, k] = float(np.argmin(e0) != np.argmin(e1))
            rank[w, k] = 1 + (e1 < e1[0]).sum()
    return tau, flip, rank


base_rank = np.array([[1 + (E[w, 0, k] < E[w, 0, k, 0]).sum() for k in range(2)] for w in range(nW)])
srt = np.sort(E[:, 0], axis=-1)
gap = srt[..., 1] - srt[..., 0]                                     # [nW, 2] raw top1-top2 energy gap
gap_n = gap / E[:, 0].std(-1)                                        # gap in units of the candidate energy std
quart = np.zeros((nW, 2), int)                                       # gap quartile within each k (0 = smallest gaps)
for k in range(2):
    qs = np.quantile(gap[:, k], [0.25, 0.5, 0.75])
    quart[:, k] = np.searchsorted(qs, gap[:, k], side="right")

M = {}
for n, c in ci.items():
    tau, flip, rank = inst_metrics(c)
    ok = ~np.isnan(tau)
    wmask = ok.all(1)
    d_k = [float(np.nansum(DN[wmask, c, k]) / DEN[wmask, k].sum()) for k in range(2)]
    dE_k = [float(np.nansum(DN[wmask, c, k]) / (E[wmask, 0, k].sum() * NEL)) for k in range(2)]
    dS_k = [float(np.nansum(DN[wmask, c, k]) / SPR[wmask, k].sum()) for k in range(2)]
    fq = []
    for q in range(4):
        sel = (quart == q) & ok
        fq.append([float(np.nansum(flip[sel])), int(sel.sum())])
    M[n] = dict(tau=tau, flip=flip, rank=rank, n_win=int(wmask.sum()),
                rd=float(np.nanmean(1 - tau)), fr=float(np.nanmean(flip)),
                rd_k=[float(np.nanmean(1 - tau[:, k])) for k in range(2)], fr_k=[float(np.nanmean(flip[:, k])) for k in range(2)],
                exec_top1=float(np.nanmean(rank == 1)), exec_rank=float(np.nanmean(rank)),
                delta=0.5 * (d_k[0] + d_k[1]), delta_k=d_k, delta_E=0.5 * (dE_k[0] + dE_k[1]), delta_spread=0.5 * (dS_k[0] + dS_k[1]),
                modfrac=float(np.nanmean(MF[wmask, c])), flipq=fq)


def group(prefix, a, seeds=(0, 1, 2)):
    """Condition names of a family at alpha a (F families pool their fixed masks)."""
    if prefix in ("F1", "F2"):
        return [f"{prefix}s{s}@{a}" for s in seeds if f"{prefix}s{s}@{a}" in ci]
    return [f"{prefix}@{a}"] if f"{prefix}@{a}" in ci else []


def per_window(nms, key):
    """Window-level metric (mean over k, then over pooled conditions): rd = 1 - tau, fr = flip."""
    arrs = [(1 - M[n]["tau"]) if key == "rd" else M[n]["flip"] for n in nms]
    return np.nanmean(np.stack(arrs), axis=(0, 2))  # [nW]


def agg(nms):
    out = {}
    for key in ["rd", "fr", "delta", "delta_E", "delta_spread", "modfrac", "exec_top1", "exec_rank"]:
        out[key] = float(np.mean([M[n][key] for n in nms]))
    out["rd_k"] = np.mean([M[n]["rd_k"] for n in nms], 0).tolist()
    out["fr_k"] = np.mean([M[n]["fr_k"] for n in nms], 0).tolist()
    out["delta_k"] = np.mean([M[n]["delta_k"] for n in nms], 0).tolist()
    fq = np.sum([M[n]["flipq"] for n in nms], 0)  # [4, 2] (flips, n)
    out["flipq_rate"] = (fq[:, 0] / np.maximum(fq[:, 1], 1)).tolist()
    out["flipq_counts"] = fq.tolist()
    out["q1q4_ratio"] = float(((fq[0, 0] + 0.5) / (fq[0, 1] + 1)) / ((fq[3, 0] + 0.5) / (fq[3, 1] + 1)))
    out["members"] = nms
    return out


def boot_ratio(num_nms, den_nms, key):
    a, b = per_window(num_nms, key), per_window(den_nms, key)
    idx = rng.integers(0, nW, (NB_BOOT, nW))
    r = a[idx].mean(1) / np.maximum(b[idx].mean(1), 1e-12)
    return float(a.mean() / max(b.mean(), 1e-12)), [float(np.quantile(r, 0.05)), float(np.quantile(r, 0.95))]


G = {}
for a in ALPHAS:
    for fam in ["A", "B", "C", "F1", "F2", "D1", "D2"]:
        nms = group(fam, a)
        if nms:
            G[f"{fam}@{a}"] = agg(nms)
for n in names:
    if n.startswith("NUM_"):
        G[n] = agg([n])

# ------------------------------------------------------------------------------------------------- criteria (PLAN 5)
crit = {}
s1_rows = []
for a in ALPHAS:
    for fam in ["A", "B"]:
        g = G[f"{fam}@{a}"]
        s1_rows.append(dict(alpha=a, cond=fam, removed=g["modfrac"], delta=g["delta"],
                            ok=bool(g["modfrac"] >= 0.5 and g["delta"] <= 0.10)))
crit["S1"] = dict(pass_=any(r["ok"] for r in s1_rows), rows=s1_rows)


def s2(ctrl):
    rows = []
    for a in ALPHAS:
        rd, rd_ci = boot_ratio(group("B", a), group(ctrl, a), "rd")
        fr, fr_ci = boot_ratio(group("B", a), group(ctrl, a), "fr")
        rows.append(dict(alpha=a, rd_B=G[f"B@{a}"]["rd"], rd_ctrl=G[f"{ctrl}@{a}"]["rd"], ratio_rd=rd, ratio_rd_ci90=rd_ci,
                         fr_B=G[f"B@{a}"]["fr"], fr_ctrl=G[f"{ctrl}@{a}"]["fr"], ratio_fr=fr, ratio_fr_ci90=fr_ci,
                         delta_B=G[f"B@{a}"]["delta"], delta_ctrl=G[f"{ctrl}@{a}"]["delta"],
                         ok_rd=bool(rd >= 2), ok_fr=bool(fr >= 2)))
    n_rd, n_fr = sum(r["ok_rd"] for r in rows), sum(r["ok_fr"] for r in rows)
    return dict(pass_=n_rd >= 2, pass_flip_only=bool(n_rd < 2 and n_fr >= 2), n_alpha_rd=n_rd, n_alpha_fr=n_fr, rows=rows)


def s3(ctrl):
    rows = []
    for a in ALPHAS:
        gb, gc = G[f"B@{a}"], G[f"{ctrl}@{a}"]
        rows.append(dict(alpha=a, B_flipq=gb["flipq_rate"], ctrl_flipq=gc["flipq_rate"], ratio_B=gb["q1q4_ratio"],
                         ratio_ctrl=gc["q1q4_ratio"], B_flips_total=int(np.sum(gb["flipq_counts"], 0)[0]),
                         ok=bool(np.sum(gb["flipq_counts"], 0)[0] > 0 and gb["q1q4_ratio"] >= 2 and gb["q1q4_ratio"] > gc["q1q4_ratio"])))
    return dict(pass_=sum(r["ok"] for r in rows) >= 2, n_alpha=sum(r["ok"] for r in rows), rows=rows)


def bias_clause(zero_ctrl, mean_ctrl):
    rows = []
    for a in ALPHAS:
        ra, _ = boot_ratio(group("A", a), group(zero_ctrl, a), "rd")
        rb, _ = boot_ratio(group("B", a), group(mean_ctrl, a), "rd")
        rows.append(dict(alpha=a, ratio_A=ra, ratio_B=rb, rd_A=G[f"A@{a}"]["rd"], rd_C=G[f"C@{a}"]["rd"],
                         C_over_A=G[f"C@{a}"]["rd"] / max(G[f"A@{a}"]["rd"], 1e-12),
                         A_large=bool(ra >= 2), B_large=bool(rb >= 2)))
    large = [r for r in rows if r["A_large"] and not r["B_large"]]
    c_rep = [r for r in large if r["C_over_A"] >= 0.7]
    return dict(A_only_large_alphas=[r["alpha"] for r in large], C_reproduces=[r["alpha"] for r in c_rep],
                bias_verdict=bool(len(large) >= 2 and len(c_rep) > len(large) / 2), rows=rows)


def verdict(s1p, s2p, s3p):
    if s2p and s3p:
        return "있음"
    if s2p:
        return "조건부 있음"
    if s1p:
        return "있음(다른 방향)"
    return "없음"


for tag, mctrl, zctrl in [("primary_Dfix", "F2", "F1"), ("secondary_Dindep", "D2", "D1")]:
    c2, c3 = s2(mctrl), s3(mctrl)
    crit[tag] = dict(control_mean=mctrl, control_zero=zctrl, S2=c2, S3=c3, bias=bias_clause(zctrl, mctrl),
                     verdict=verdict(crit["S1"]["pass_"], c2["pass_"], c3["pass_"]))

# --------------------------------------------------------------------------------------------------------- outputs
base = M["base"]
summary = dict(n_windows=nW, n_instances=2 * nW, cond_names=names,
               baseline=dict(exec_top1=float(np.mean(base_rank == 1)), exec_top1_k=np.mean(base_rank == 1, 0).tolist(),
                             exec_rank_mean=float(base_rank.mean()), exec_rank_median=float(np.median(base_rank)),
                             exec_rank_k=base_rank.mean(0).tolist(), gap_quartile_edges=[np.quantile(gap[:, k], [0.25, 0.5, 0.75]).tolist() for k in range(2)],
                             gap_norm_median=float(np.median(gap_n)), energy_exec_mean=E[:, 0, :, 0].mean(0).tolist(),
                             energy_cand_std_mean=E[:, 0].std(-1).mean(0).tolist()),
               groups=G, criteria=crit, match_p=match["p"], match_achieved=match["achieved"],
               stats={k: v for k, v in stats.items() if k != "weak_frac_per_block"})
json.dump(summary, open(f"{RUN}/summary.json", "w"), indent=1)

with open(f"{RUN}/table_conditions.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["condition", "alpha", "members", "removed_frac", "delta", "delta_k1", "delta_k2", "delta_rel_energy", "delta_rel_spread",
                 "rank_damage_1-tau", "rd_k1", "rd_k2", "top1_flip", "flip_k1", "flip_k2", "exec_top1", "exec_mean_rank"])
    for key, g in G.items():
        fam, a = (key.split("@") + [""])[:2]
        wr.writerow([fam, a, len(g["members"]), f"{g['modfrac']:.4f}", f"{g['delta']:.5f}", f"{g['delta_k'][0]:.5f}", f"{g['delta_k'][1]:.5f}",
                     f"{g['delta_E']:.5f}", f"{g['delta_spread']:.5f}", f"{g['rd']:.5f}", f"{g['rd_k'][0]:.5f}", f"{g['rd_k'][1]:.5f}",
                     f"{g['fr']:.4f}", f"{g['fr_k'][0]:.4f}", f"{g['fr_k'][1]:.4f}", f"{g['exec_top1']:.4f}", f"{g['exec_rank']:.3f}"])
with open(f"{RUN}/table_flip_quartiles.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["condition", "alpha", "Q1_smallest_gap", "Q2", "Q3", "Q4_largest_gap", "Q1/Q4_ratio_(+0.5)", "flips_Q1..Q4", "n_Q1..Q4"])
    for key, g in G.items():
        fam, a = (key.split("@") + [""])[:2]
        fq = np.array(g["flipq_counts"])
        wr.writerow([fam, a] + [f"{x:.4f}" for x in g["flipq_rate"]] + [f"{g['q1q4_ratio']:.3f}", fq[:, 0].astype(int).tolist(),
                                                                          fq[:, 1].astype(int).tolist()])
with open(f"{RUN}/table_criteria.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["comparison", "criterion", "alpha", "value", "ok"])
    for r in crit["S1"]["rows"]:
        wr.writerow(["-", f"S1 {r['cond']}", r["alpha"], f"removed {r['removed']:.3f} delta {r['delta']:.4f}", r["ok"]])
    for tag in ["primary_Dfix", "secondary_Dindep"]:
        c = crit[tag]
        for r in c["S2"]["rows"]:
            wr.writerow([tag, "S2 rd ratio B/ctrl", r["alpha"], f"{r['ratio_rd']:.3f} CI90 {np.round(r['ratio_rd_ci90'], 3).tolist()}", r["ok_rd"]])
            wr.writerow([tag, "S2 flip ratio B/ctrl", r["alpha"], f"{r['ratio_fr']:.3f} CI90 {np.round(r['ratio_fr_ci90'], 3).tolist()}", r["ok_fr"]])
        for r in c["S3"]["rows"]:
            wr.writerow([tag, "S3 Q1/Q4 B vs ctrl", r["alpha"], f"{r['ratio_B']:.3f} vs {r['ratio_ctrl']:.3f}", r["ok"]])
        wr.writerow([tag, "verdict", "-", c["verdict"], "-"])

print(json.dumps(dict(baseline=summary["baseline"],
                      S1=crit["S1"],
                      primary={k: (v if k not in ("S2", "S3") else dict(pass_=v["pass_"], rows=v["rows"])) for k, v in crit["primary_Dfix"].items()},
                      secondary_verdict=crit["secondary_Dindep"]["verdict"]), indent=1, default=float)[:20000])
for key, g in G.items():
    print(f"{key:12s} rm {g['modfrac']:.3f} delta {g['delta']:.4f} rd {g['rd']:.4f} fr {g['fr']:.3f} exec1 {g['exec_top1']:.3f} "
          f"flipQ {np.round(g['flipq_rate'], 3).tolist()} q1/q4 {g['q1q4_ratio']:.2f}")

# --------------------------------------------------------------------------------------------------------- figures
if not NOFIG:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    COL = {"weak": "#2a78d6", "fix": "#eb6834", "ind": "#1baf7a", "bias": "#6250d6"}
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})

    # fig 1: rank damage vs normalized prediction change
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    panels = [("mean-filled: B vs D-fix(F2) vs D-indep(D2)", [("B", "weak", "o", "B weak->mu"), ("F2", "fix", "s", "D-fix F2 (3 masks)"),
                                                               ("D2", "ind", "^", "D-indep D2")]),
              ("zero-filled: A vs D-fix(F1) vs D-indep(D1); C bias only", [("A", "weak", "o", "A weak->0"), ("F1", "fix", "s", "D-fix F1 (3 masks)"),
                                                                           ("D1", "ind", "^", "D-indep D1"), ("C", "bias", "D", "C bias only")])]
    for ax, (title, fams) in zip(axs, panels):
        offs = [(-34, 6), (6, -13), (6, 5), (-34, -13)]
        for j, (fam, col, mk, lab) in enumerate(fams):
            xs = [G[f"{fam}@{a}"]["delta"] for a in ALPHAS]
            ys = [max(G[f"{fam}@{a}"]["rd"], 1e-5) for a in ALPHAS]
            ax.plot(xs, ys, color=COL[col], lw=2, marker=mk, ms=8, mec=SURF, mew=1.5, label=lab)
            for a, x, y in zip(ALPHAS, xs, ys):
                ax.annotate(f"α={a}", (x, y), textcoords="offset points", xytext=offs[j], fontsize=7, color=INK2)
        for n, ls in [("NUM_bf16", "--"), ("NUM_fp32", ":")]:
            if n in G:
                ax.axhline(max(G[n]["rd"], 1e-5), color=INK2, lw=1, ls=ls, label=f"numerical floor {n[4:]} vs main")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("normalized prediction change Δ = Σ|ẑ'−ẑ| / Σ|ẑ| (mean of k=1,2)")
        ax.set_title(title, fontsize=9, color=INK)
        ax.legend(fontsize=7, frameon=False)
    axs[0].set_ylabel("rank damage 1 − Kendall τ (64 candidates)")
    fig.tight_layout()
    fig.savefig(f"{RUN}/fig_rank_damage_vs_change.png", dpi=130)

    # fig 2: top-1 flip rate by gap quartile
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    fams = [("B", "weak", "B weak->mu"), ("F2", "fix", "D-fix F2"), ("D2", "ind", "D-indep D2")]
    for ax, a in zip(axs, ALPHAS):
        for j, (fam, col, lab) in enumerate(fams):
            r = G[f"{fam}@{a}"]["flipq_rate"]
            ax.bar(np.arange(4) + (j - 1) * 0.26, r, width=0.24, color=COL[col], label=lab, edgecolor=SURF, linewidth=1)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["Q1\nsmallest gap", "Q2", "Q3", "Q4\nlargest gap"])
        ax.set_title(f"α = {a}", fontsize=9, color=INK)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("top-1 flip rate")
    axs[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{RUN}/fig_flip_by_gap.png", dpi=130)

    # fig 3: S1 view, removed fraction vs prediction change
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for fam, col, mk, ls, mfc, lab in [("A", "weak", "o", "--", SURF, "A weak->0 (hollow)"), ("B", "weak", "o", "-", COL["weak"], "B weak->mu"),
                                       ("F1", "fix", "s", "--", SURF, "D-fix F1, matched to A (hollow)"),
                                       ("F2", "fix", "s", "-", COL["fix"], "D-fix F2, matched to B")]:
        xs = [G[f"{fam}@{a}"]["modfrac"] for a in ALPHAS]
        ys = [G[f"{fam}@{a}"]["delta"] for a in ALPHAS]
        ax.plot(xs, ys, color=COL[col], lw=2, ls=ls, marker=mk, ms=8, mfc=mfc, mec=COL[col], mew=1.5, label=lab)
        for a, x, y in zip(ALPHAS, xs, ys):
            ax.annotate(f"α={a}", (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7, color=INK2)
    ax.axvline(0.5, color=INK2, lw=1, ls="--")
    ax.axhline(0.10, color=INK2, lw=1, ls="--")
    ax.set_xlabel("fraction of predictor MLP hidden entries replaced")
    ax.set_ylabel("normalized prediction change Δ")
    ax.set_title("S1 region: right of x=0.5 and below Δ=0.10 (dashed)", fontsize=9, color=INK)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{RUN}/fig_s1_removed_vs_change.png", dpi=130)
print("ANALYZE_DONE")
