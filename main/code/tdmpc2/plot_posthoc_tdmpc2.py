"""Figures of the post-hoc TD-MPC2 checks (DESIGN_main v5.3 10.2, 10.3; descriptive). Reads the analysis outputs only.
  posthoc_latency_pairs.png  per condition: median pair overhead with its bootstrap 95% CI (paired re-measurement),
                             the pre-defined pooled computation on the same data, and the first measurement's value
  posthoc_latency_modes.png  block / condition-median ratio of the 960 re-measurement blocks in time order (slow mode
                             above 1.15), marker = SM clock at the block end below 1,890 MHz or not
  posthoc_qatnone.png        per QAT cell: D of QAT+none, QAT+det, QAT+sh-iter (95% CI) and B+det, Meta-World and DMC
                             in separate panels (different measures)
usage (server, CPU only): MPLBACKEND=Agg python plot_posthoc_tdmpc2.py --root ~/pmx/A/tdmpc2_main --outdir <dir>
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def fig_pairs(A, path):
    kerns = ["int8", "int4", "int4p"]
    names = {"int8": "int8", "int4": "int4 (container)", "int4p": "int4 (packed)"}
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.6), facecolor=SURF, sharey=True)
    for ax, kern in zip(axs, kerns):
        rows = [(k, v) for k, v in A["conditions"].items() if k.split("|")[1] == kern]
        labels = []
        for i, (k, v) in enumerate(rows):
            y = len(rows) - 1 - i
            lo, hi = v["median_pair_overhead_ci"]
            ax.plot([100 * lo, 100 * hi], [y, y], color=C1, lw=2)
            ax.plot(100 * v["median_pair_overhead"], y, "o", color=C1, ms=6,
                    label="median pair overhead [95% CI]" if i == 0 else None)
            ax.plot(100 * v["pooled_median_overhead_same_data"], y, "s", color=C2, ms=5, mfc=SURF,
                    label="pooled medians, same data" if i == 0 else None)
            if v.get("first_measurement_pooled_overhead") is not None:
                ax.plot(100 * v["first_measurement_pooled_overhead"], y, "^", color=C3, ms=6, mfc=SURF,
                        label="first measurement (pre-defined)" if i == 0 else None)
            t, _, it = k.split("|")
            labels.append(f"{t.replace('mw-', '')} {it}")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(labels[::-1], fontsize=8, color=INK)
        ax.axvline(0, color=INK2, lw=1)
        ax.axvline(10, color=INK2, lw=1, ls="--")
        style(ax)
        ax.set_title(names[kern], color=INK, fontsize=10)
        ax.set_xlabel("dither overhead vs det (%)", color=INK2, fontsize=9)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=8, frameon=False, labelcolor=INK)
    fig.suptitle("Post-hoc paired latency re-measurement (v5.3 10.3; 10 rounds, back-to-back det/dither blocks; "
                 "dashed = 10%)", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path, dpi=110, facecolor=SURF)
    plt.close(fig)


def fig_modes(csv_path, path):
    rows = list(csv.DictReader(open(csv_path)))
    t0 = min(float(r["t_start"]) for r in rows)
    fig, ax = plt.subplots(figsize=(12, 3.6), facecolor=SURF)
    for red, col, mk, lab in ((False, C1, "o", "SM clock at block end >= 1,890 MHz"),
                              (True, C2, "x", "SM clock at block end < 1,890 MHz")):
        xs = [(float(r["t_start"]) - t0) / 60 for r in rows if (float(r["sm_end"]) < 1890) == red]
        ys = [float(r["r"]) for r in rows if (float(r["sm_end"]) < 1890) == red]
        ax.plot(xs, ys, mk, color=col, ms=3.5, ls="none", label=lab)
    ax.axhline(1.15, color=INK2, lw=1, ls="--")
    style(ax)
    ax.set_xlabel("minutes since the start of the re-measurement", color=INK2, fontsize=9)
    ax.set_ylabel("block / condition median", color=INK2, fontsize=9)
    ax.set_title("Two latency modes in the re-measurement (960 blocks; dashed = 1.15)", color=INK, fontsize=10)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURF)
    plt.close(fig)


def fig_qat(Q, path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURF, gridspec_kw={"width_ratios": [8, 3]})
    for ax, fam in zip(axs, ("mw", "dmc")):
        rows = [(c, r) for c, r in Q["cells"].items() if r["family"] == fam]
        for i, (c, r) in enumerate(rows):
            y = len(rows) - 1 - i
            for off, n, col, mk, lab in ((0.22, "qat_none", C2, "o", "QAT+none"), (0.0, "qat_det", C1, "s", "QAT+det"),
                                         (-0.22, "qat_shiter", C3, "^", "QAT+sh-iter")):
                lo, hi = r[n]["D_ci"]
                ax.plot([lo, hi], [y + off, y + off], color=col, lw=2)
                ax.plot(r[n]["D"], y + off, mk, color=col, ms=6, label=lab if i == 0 else None)
            if "B_det" in r:
                ax.plot(r["B_det"]["D"], y, "|", color=INK, ms=14, mew=2, label="B+det (best PTQ)" if i == 0 else None)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([c.replace("mw-", "").replace("|", " k=") for c, _ in rows][::-1], fontsize=8, color=INK)
        ax.axvline(0, color=INK2, lw=1)
        style(ax)
        ax.set_xlabel("D vs the original weights' none (" + ("success rate" if fam == "mw" else "rel. return") + ")",
                      color=INK2, fontsize=9)
        ax.set_title("Meta-World" if fam == "mw" else "DMC", color=INK, fontsize=10)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=8, frameon=False, labelcolor=INK)
    fig.suptitle("Post-hoc QAT+none (v5.3 10.2): the QAT weights without activation quantization", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(path, dpi=110, facecolor=SURF)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    root = os.path.expanduser(a.root)
    os.makedirs(a.outdir, exist_ok=True)
    fig_pairs(json.load(open(f"{root}/analysis_posthoc_e7lat/e7_posthoc_latency_summary.json")),
              f"{a.outdir}/posthoc_latency_pairs.png")
    fig_modes(f"{root}/analysis_posthoc_check/posthoc_latency_blocks.csv", f"{a.outdir}/posthoc_latency_modes.png")
    fig_qat(json.load(open(f"{root}/analysis_posthoc_qatnone/posthoc_qatnone_summary.json")),
            f"{a.outdir}/posthoc_qatnone.png")
    print("figures written")


if __name__ == "__main__":
    main()
