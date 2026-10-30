# Figure 1 of the JKAIA 2026 poster summary, drawn straight from the A result files.
#  (a) sh-iter damage vs det damage for every eligible E1 cell (Holm-real det drop).
#  (b) TD-MPC2 E2 iteration curves (warm start on): damage vs MPPI iterations for det, sh-step, sh-iter.
# Usage: python3 make_fig.py   (writes ../figures/fig_main.pdf and .png)
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "..", "main", "results")
OUT = os.path.join(HERE, "..", "figures")

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
C_SHITER, C_DET, C_SHSTEP = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3, validated all-pairs

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.labelcolor": INK, "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def ci(s):
    lo, hi = s.strip("[]").split(";")
    return float(lo), float(hi)


def e1_cells():
    """Eligible confirmatory cells: (family, task, k, D_det %, D_sh-iter %)."""
    out = []
    rows = list(csv.DictReader(open(os.path.join(RES, "E1_tdmpc2", "e1_tdmpc2_cells.csv"))))
    by = {}
    for r in rows:
        by.setdefault((r["task"], r["k"], r["family"]), {})[r["variant"]] = r
    for (task, k, fam), v in by.items():
        if v["det"]["holm_real"] == "True":
            out.append(("Meta-World" if fam == "mw" else "DMC", task, int(k),
                        100 * float(v["det"]["D"]), 100 * float(v["shiter"]["D"])))
    rows = list(csv.DictReader(open(os.path.join(RES, "E1", "e1_dinowm_cells.csv"))))
    by = {}
    for r in rows:
        if r["role"] == "confirmatory":
            by.setdefault((r["task"], r["k"]), {})[r["var"]] = r
    for (task, k), v in by.items():
        if v["det"]["drop_real_holm"] == "True":
            out.append(("DINO-WM", task, int(k), 100 * float(v["det"]["D"]), 100 * float(v["shiter"]["D"])))
    return out


def e2_curves():
    rows = list(csv.DictReader(open(os.path.join(RES, "E2_tdmpc2", "e2_tdmpc2_cells.csv"))))
    e2 = {(r["task"], r["setting"], r["variant"]): r for r in rows}
    its, settings = [1, 3, 6, 12], ["it1", "it3", "base", "it12"]
    curves = {}
    for task in ("mw-push", "mw-pick-place", "cheetah-run"):
        for var in ("det", "shstep", "shiter"):
            d = [100 * float(e2[(task, s, var)]["D"]) for s in settings]
            c = [tuple(100 * x for x in ci(e2[(task, s, var)]["D_ci"])) for s in settings]
            curves[(task, var)] = (its, d, c)
    return curves


def main():
    cells = e1_cells()
    curves = e2_curves()
    fig = plt.figure(figsize=(6.69, 1.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1, 1, 1], wspace=0.42,
                          left=0.065, right=0.995, top=0.775, bottom=0.235)

    # (a) scatter
    ax = fig.add_subplot(gs[0, 0])
    ax.plot([0, 100], [0, 100], color=MUTED, lw=0.7, ls=(0, (1, 1.5)), zorder=1)
    ax.plot([0, 100], [0, 50], color=INK2, lw=0.7, ls=(0, (4, 2)), zorder=1)
    shapes = {"Meta-World": ("o", 16), "DINO-WM": ("s", 14), "DMC": ("^", 17)}
    fam_handles = []
    for fam, (m, s) in shapes.items():
        pts = [(x, y) for f, _, _, x, y in cells if f == fam]
        fam_handles.append(ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker=m, s=s, facecolor=C_SHITER,
                                      edgecolor="white", linewidth=0.5, alpha=0.9, zorder=3,
                                      label=f"{fam} ({len(pts)})"))
    wk = [(x, y) for f, t, k, x, y in cells if t == "walker-walk" and k == 3][0]
    ax.annotate("walker k=3", xy=wk, xytext=(4, 86), fontsize=6, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkB=3))
    ax.set_xlim(-3, 103)
    ax.set_ylim(-6, 103)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    for slope, x0, label, col in ((1.0, 21, "no effect", MUTED), (0.5, 33, "50% removed", INK2)):
        (ax0, ay0), (ax1, ay1) = ax.transData.transform([(0, 0), (100, 100 * slope)])
        ang = math.degrees(math.atan2(ay1 - ay0, ax1 - ax0))  # line angle on screen (axes are not square)
        ax.text(x0, slope * x0 + 2.5, label, color=col, fontsize=6, rotation=ang, rotation_mode="anchor",
                ha="left", va="bottom")
    ax.set_xlabel("damage under det (%)")
    ax.set_ylabel("damage under sh-iter (%)")
    ax.set_title("(a) every affected cell", loc="left", color=INK, pad=3)
    ax.grid(color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # (b) iteration curves
    style = {"det": (C_DET, "s", (0, (4, 1.5)), "det"),
             "shstep": (C_SHSTEP, "^", (0, (1, 1)), "sh-step (fixed within a step)"),
             "shiter": (C_SHITER, "o", "-", "sh-iter (fresh every iteration)")}
    names = {"mw-push": "mw-push (3 bits)", "mw-pick-place": "mw-pick-place (3 bits)", "cheetah-run": "cheetah-run (5 bits)"}
    handles = []
    for j, task in enumerate(("mw-push", "mw-pick-place", "cheetah-run")):
        ax = fig.add_subplot(gs[0, j + 1])
        for var, dodge in (("det", 0.92), ("shstep", 1.0), ("shiter", 1.08)):
            col, mk, ls, lab = style[var]
            its, d, c = curves[(task, var)]
            xs = [x * dodge for x in its]  # small offset on the log axis so the CI bars do not overlap
            ax.vlines(xs, [lo for lo, _ in c], [hi for _, hi in c], color=col, lw=0.6, alpha=0.6, zorder=2)
            h, = ax.plot(xs, d, color=col, ls=ls, lw=1.3, marker=mk, ms=3.6, mec="white", mew=0.4,
                         zorder=3, label=lab)
            if j == 0:
                handles.append(h)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 3, 6, 12])
        ax.set_xticklabels(["1", "3", "6", "12"])
        ax.minorticks_off()
        ax.set_xlim(0.85, 14)
        ax.set_ylim(-4, 72)
        ax.set_yticks([0, 20, 40, 60])
        ax.set_xlabel("MPPI iterations")
        if j == 0:
            ax.set_ylabel("damage (%)")
        ax.set_title(("(b) " if j == 0 else "") + names[task], loc="left", color=INK, pad=3)
        ax.grid(color=GRID, lw=0.4)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    # family legend for (a) sits in the free top-left band so it cannot cover points or labels
    fig.legend(handles=fam_handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False,
               fontsize=6, handletextpad=0.1, columnspacing=0.7, borderaxespad=0.2)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.36, 1.0), ncol=3, frameon=False,
               handlelength=2.6, columnspacing=1.4, handletextpad=0.4)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "fig_main.pdf"))
    fig.savefig(os.path.join(OUT, "fig_main.png"), dpi=300)
    print("cells:", {f: sum(1 for c in cells if c[0] == f) for f in shapes})


if __name__ == "__main__":
    main()
