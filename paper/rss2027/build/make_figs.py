# Figures of the RSS 2027 Stage 1 extended abstract of paper A, drawn straight from the result files.
#  fig_damage:    (a) sh-iter vs det damage in every affected cell; (b) latent one-step error ratio vs damage gap.
#  fig_mechanism: (a) drift kept by each variant relative to det; (b) damage vs MPPI iterations (TD-MPC2).
#  fig_latency:   E8 graph-WM speed-ups of the W8A4 integer world model (E8c full-step panel added when present).
# Text is drawn as paths (SVG, converted to PDF with PyMuPDF), so the figure PDFs embed no fonts.
# Usage: python3 make_figs.py   (writes ../figures/*.pdf and *.png previews)
import csv
import json
import math
import os
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RES = os.path.join(A, "main", "results")
EXT = os.path.join(A, "rss_ext")
OUT = os.path.join(HERE, "..", "figures")

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
# Categorical slots, fixed per variant across figures (validated all-pairs within each figure).
C_SHITER, C_DET, C_SHSTEP, C_SHEP = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
# Ordinal blue ramp for candidate count N (light -> dark), with markers and direct labels as the second channel.
C_N = {512: "#6da7ec", 2048: "#2a78d6", 8192: "#104281"}
COL_W, TEXT_W = 3.5, 7.16  # IEEEtran conference column and text widths (in)

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.labelcolor": INK, "text.color": INK, "svg.fonttype": "path",
})

FAMILIES = {  # label -> (marker, size)
    "5M Meta-World": ("o", 15),
    "5M DMC": ("^", 16),
    "DINO-WM": ("s", 13),
    "317M Meta-World": ("D", 11),
}


def ci(s):
    lo, hi = s.strip("[]").replace(";", ",").split(",")
    return float(lo), float(hi)


def tidy(ax):
    ax.grid(color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        svg = os.path.join(tmp, name + ".svg")
        fig.savefig(svg, format="svg")
        with pymupdf.open(svg) as doc:
            pdf = doc.convert_to_pdf()
    with open(os.path.join(OUT, name + ".pdf"), "wb") as f:
        f.write(pdf)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=220)
    plt.close(fig)


def affected_cells():
    """Every affected cell: (family, task, k, D_det %, D_sh-iter %, latent one-step error det, sh-iter)."""
    out = []
    by = {}
    for r in csv.DictReader(open(os.path.join(RES, "E1_tdmpc2", "e1_tdmpc2_cells.csv"))):
        by.setdefault((r["task"], r["k"], r["family"]), {})[r["variant"]] = r
    for (task, k, fam), v in by.items():
        if v["det"]["confirmatory"] == "True" and v["det"]["holm_real"] == "True":
            out.append(("5M Meta-World" if fam == "mw" else "5M DMC", task, int(k),
                        100 * float(v["det"]["D"]), 100 * float(v["shiter"]["D"]),
                        float(v["det"]["pt_err1h"]), float(v["shiter"]["pt_err1h"])))
    by = {}
    for r in csv.DictReader(open(os.path.join(RES, "E1", "e1_dinowm_cells.csv"))):
        if r["role"] == "confirmatory":
            by.setdefault((r["task"], r["k"]), {})[r["var"]] = r
    for (task, k), v in by.items():
        if v["det"]["drop_real_holm"] == "True":
            out.append(("DINO-WM", task, int(k), 100 * float(v["det"]["D"]), 100 * float(v["shiter"]["D"]),
                        float(v["det"]["rel_err1"]), float(v["shiter"]["rel_err1"])))
    for r in csv.DictReader(open(os.path.join(EXT, "E10_mt317m", "results", "analysis", "e10_eval.csv"))):
        if r["drop_real"] == "True":
            out.append(("317M Meta-World", r["task"], int(r["k"]), 100 * float(r["D_det"]), 100 * float(r["D_sh"]),
                        float(r["pt_err1h_det"]), float(r["pt_err1h_sh"])))
    return out


def fig_damage(cells):
    fig = plt.figure(figsize=(COL_W, 1.72))
    gs = fig.add_gridspec(1, 2, wspace=0.42, left=0.115, right=0.985, top=0.80, bottom=0.215)
    handles = []
    # (a) damage under sh-iter vs damage under det
    ax = fig.add_subplot(gs[0, 0])
    ax.plot([0, 100], [0, 100], color=MUTED, lw=0.7, ls=(0, (1, 1.5)), zorder=1)
    ax.plot([0, 100], [0, 50], color=INK2, lw=0.7, ls=(0, (4, 2)), zorder=1)
    for fam, (m, s) in FAMILIES.items():
        pts = [(c[3], c[4]) for c in cells if c[0] == fam]
        handles.append(ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker=m, s=s, facecolor=C_SHITER,
                                  edgecolor="white", linewidth=0.5, alpha=0.9, zorder=3, label=f"{fam} ({len(pts)})"))
    wk = [(c[3], c[4]) for c in cells if c[1] == "walker-walk" and c[2] == 3][0]
    ax.annotate("walker k=3", xy=wk, xytext=(3, 88), fontsize=6, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkB=3))
    ax.set_xlim(-3, 103)
    ax.set_ylim(-6, 103)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 50, 100])
    for slope, x0, label, col in ((1.0, 30, "no effect", MUTED), (0.5, 46, "50% removed", INK2)):
        (ax0, ay0), (ax1, ay1) = ax.transData.transform([(0, 0), (100, 100 * slope)])
        ang = math.degrees(math.atan2(ay1 - ay0, ax1 - ax0))  # line angle on screen (axes are not square)
        ax.text(x0, slope * x0 + 3, label, color=col, fontsize=5.8, rotation=ang, rotation_mode="anchor",
                ha="left", va="bottom")
    ax.set_xlabel("damage under det (%)")
    ax.set_ylabel("damage under sh-iter (%)")
    ax.set_title("(a) prescription", loc="left", pad=3)
    tidy(ax)
    # (b) det/sh-iter latent one-step error ratio vs extra damage of det
    ax = fig.add_subplot(gs[0, 1])
    ax.axvline(1.0, color=MUTED, lw=0.7, ls=(0, (1, 1.5)), zorder=1)
    for fam, (m, s) in FAMILIES.items():
        pts = [(c[5] / c[6], c[3] - c[4]) for c in cells if c[0] == fam]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker=m, s=s, facecolor=C_SHITER,
                   edgecolor="white", linewidth=0.5, alpha=0.9, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.25, 0.5, 1, 2])
    ax.set_xticklabels(["1/4", "1/2", "1", "2"])
    ax.minorticks_off()
    ax.set_xlim(0.25, 3.2)
    ax.set_ylim(-38, 106)
    ax.set_yticks([0, 50, 100])
    ax.axvspan(0.25, 1.0, color=GRID, alpha=0.3, lw=0, zorder=0)
    ax.text(0.27, -34, "det error \u2264 sh-iter", fontsize=5.6, color=INK2, va="bottom")
    ax.set_xlabel("latent 1-step error, det / sh-iter")
    ax.set_ylabel("extra damage of det (%)")
    ax.set_title("(b) error size hides it", loc="left", pad=3)
    tidy(ax)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=4, frameon=False, fontsize=5.8,
               handletextpad=0.05, columnspacing=0.6, borderaxespad=0.15)
    save(fig, "fig_damage")


def drift_ratios():
    """(task, {variant: (r, lo, hi)}) for the four mechanism tasks; PushT's fixed table is sh-plan (= sh-step = sh-ep)."""
    e2 = json.load(open(os.path.join(RES, "E2_tdmpc2", "e2_tdmpc2_summary.json")))["layer_b"]
    out = []
    for task in ("mw-push", "mw-pick-place", "cheetah-run"):
        b = e2[task]["b2_H2a_prime"]
        out.append((task, {v: (b["r_" + v]["r"], *b["r_" + v]["ci"]) for v in ("shiter", "shstep", "shep")}))
    b = json.load(open(os.path.join(RES, "E1", "e1_dinowm_v52.json")))["b2_H2a_prime_pusht_k4"]
    out.append(("PushT", {"shiter": (b["r_shiter"], *b["r_shiter_ci"]),
                          "shstep": (b["positive_control_r_shplan"], *b["positive_control_r_shplan_ci"])}))
    return out


def e2_curves():
    e2 = {(r["task"], r["setting"], r["variant"]): r
          for r in csv.DictReader(open(os.path.join(RES, "E2_tdmpc2", "e2_tdmpc2_cells.csv")))}
    its, settings = [1, 3, 6, 12], ["it1", "it3", "base", "it12"]
    curves = {}
    for task in ("mw-push", "mw-pick-place", "cheetah-run"):
        for var in ("det", "shstep", "shiter"):
            d = [100 * float(e2[(task, s, var)]["D"]) for s in settings]
            c = [tuple(100 * x for x in ci(e2[(task, s, var)]["D_ci"])) for s in settings]
            curves[(task, var)] = (its, d, c)
    return curves


def fig_mechanism(ratios, curves):
    fig = plt.figure(figsize=(TEXT_W, 1.52))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.45, 1, 1, 1], wspace=0.36,
                          left=0.052, right=0.995, top=0.80, bottom=0.215)
    # (a) drift kept relative to det
    ax = fig.add_subplot(gs[0, 0])
    ax.axhline(1.0, color=C_DET, lw=0.9, ls=(0, (5, 2)), zorder=1)
    ax.axhline(0.0, color=MUTED, lw=0.6, zorder=1)
    ax.text(3.62, 1.03, "det", color=INK2, fontsize=6, ha="right", va="bottom")
    style = {"shiter": (C_SHITER, "sh-iter (fresh every call)"), "shstep": (C_SHSTEP, "sh-step (fixed in a step)"),
             "shep": (C_SHEP, "sh-ep (fixed in an episode)")}
    w = 0.25
    bar_handles = {}
    for i, (task, rs) in enumerate(ratios):
        vs = [v for v in ("shiter", "shstep", "shep") if v in rs]
        for j, v in enumerate(vs):
            x = i + (j - (len(vs) - 1) / 2) * (w + 0.03)
            r, lo, hi = rs[v]
            h = ax.bar(x, r, width=w, color=style[v][0], edgecolor="white", linewidth=0.4, zorder=2,
                       label=style[v][1])
            ax.vlines(x, lo, hi, color=INK2, lw=0.6, zorder=3)
            bar_handles.setdefault(v, h)
    ax.text(3 + (w + 0.03) / 2, 0.12, "sh-plan", fontsize=5.5, color=INK2, ha="center", va="bottom", rotation=90)
    ax.set_xticks(range(len(ratios)))
    ax.set_xticklabels(["push", "pick-place", "cheetah", "PushT"])
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(-0.12, 1.5)
    ax.set_yticks([0, 0.5, 1.0, 1.5])
    ax.set_ylabel("drift / drift of det")
    ax.set_title("(a) persistence brings the drift back", loc="left", pad=3)
    tidy(ax)
    ax.grid(axis="x", visible=False)
    # (b) damage vs MPPI iterations
    lstyle = {"det": (C_DET, "s", (0, (4, 1.5)), "det"),
              "shstep": (C_SHSTEP, "^", (0, (1, 1)), "sh-step"),
              "shiter": (C_SHITER, "o", "-", "sh-iter")}
    names = {"mw-push": "mw-push (3 bits)", "mw-pick-place": "mw-pick-place (3 bits)",
             "cheetah-run": "cheetah-run (5 bits)"}
    line_handles = []
    for j, task in enumerate(("mw-push", "mw-pick-place", "cheetah-run")):
        ax = fig.add_subplot(gs[0, j + 1])
        for var, dodge in (("det", 0.92), ("shstep", 1.0), ("shiter", 1.08)):
            col, mk, ls, lab = lstyle[var]
            its, d, c = curves[(task, var)]
            xs = [x * dodge for x in its]  # small offset on the log axis so the CI bars do not overlap
            ax.vlines(xs, [lo for lo, _ in c], [hi for _, hi in c], color=col, lw=0.6, alpha=0.6, zorder=2)
            h, = ax.plot(xs, d, color=col, ls=ls, lw=1.3, marker=mk, ms=3.4, mec="white", mew=0.4, zorder=3,
                         label=lab)
            if j == 0:
                line_handles.append(h)
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
        ax.set_title(("(b) " if j == 0 else "") + names[task], loc="left", pad=3)
        tidy(ax)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    proxies = [Line2D([], [], color=C_DET, ls=(0, (4, 1.5)), lw=1.3, marker="s", ms=3.4, mec="white", mew=0.4,
                      label="det (round-to-nearest)"),
               Line2D([], [], color=C_SHITER, ls="-", lw=1.3, marker="o", ms=3.4, mec="white", mew=0.4,
                      label="sh-iter (shared dither, fresh every call)"),
               Line2D([], [], color=C_SHSTEP, ls=(0, (1, 1)), lw=1.3, marker="^", ms=3.4, mec="white", mew=0.4,
                      label="sh-step (same dither, fixed within a step)"),
               Patch(facecolor=C_SHEP, edgecolor="white", label="sh-ep (fixed within an episode)")]
    fig.legend(handles=proxies, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=4, frameon=False, fontsize=6,
               handlelength=2.6, handletextpad=0.4, columnspacing=1.6, borderaxespad=0.2)
    save(fig, "fig_mechanism")


def e8_graph():
    """{(size, N): (fp32 / int-det, min(tf32, bf16) / int-det)} for the E8 graph-WM cells."""
    out = {}
    for r in csv.DictReader(open(os.path.join(EXT, "E8_latency", "results", "analysis", "e8_cells.csv"))):
        if r["mode"] == "graph":
            det = float(r["T_int_det_ms"])
            fast = min(float(r["T_tf32_ms"]), float(r["T_bf16_ms"]))
            out[(int(r["size"]), int(r["N"]))] = (float(r["T_fp32_ms"]) / det, fast / det)
    return out


def fig_latency(e8):
    sizes = [5, 19, 48, 317]
    fig = plt.figure(figsize=(COL_W, 1.55))
    gs = fig.add_gridspec(1, 2, wspace=0.38, left=0.105, right=0.90, top=0.84, bottom=0.215)
    panels = [("(a) vs FP32", 0, (0, 3.6), [0, 1, 2, 3]), ("(b) vs fastest of TF32, BF16", 1, (0, 1.65), [0, 0.5, 1, 1.5])]
    for p, (title, idx, ylim, yt) in enumerate(panels):
        ax = fig.add_subplot(gs[0, p])
        ax.axhline(1.0, color=MUTED, lw=0.7, ls=(0, (1, 1.5)), zorder=1)
        ends = []
        for N, mk in ((512, "o"), (2048, "s"), (8192, "D")):
            ys = [e8[(s, N)][idx] for s in sizes]
            ax.plot(range(4), ys, color=C_N[N], lw=1.3, marker=mk, ms=3.2, mec="white", mew=0.4, zorder=3)
            ends.append([ys[-1], N])
        if p == 1:
            ends.sort()
            for i in range(1, len(ends)):  # spread direct labels that would touch
                ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.11)
            for y, N in ends:
                ax.text(3.18, y, f"N={N}", color=INK2, fontsize=5.8, va="center", ha="left")
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"{s}M" for s in sizes])
        ax.set_xlim(-0.3, 3.3)
        ax.set_ylim(*ylim)
        ax.set_yticks(yt)
        ax.set_xlabel("world-model size")
        if p == 0:
            ax.set_ylabel("speed-up of W8A4 (×)")
        ax.set_title(title, loc="left", pad=3)
        tidy(ax)
    save(fig, "fig_latency")


def main():
    cells = affected_cells()
    fig_damage(cells)
    fig_mechanism(drift_ratios(), e2_curves())
    fig_latency(e8_graph())
    fams = {f: sum(1 for c in cells if c[0] == f) for f in FAMILIES}
    below = {f: sum(1 for c in cells if c[0] == f and c[5] <= c[6]) for f in FAMILIES}
    print("affected cells:", fams, "| det latent error <= sh-iter:", below)


if __name__ == "__main__":
    main()
