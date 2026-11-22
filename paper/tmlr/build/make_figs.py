# Figures of the TMLR manuscript of paper A, drawn straight from the result files (no hand-typed data values).
# Figure numbers follow PLAN_tmlr.md 4.3 and main.tex. Every figure is 6.5 in wide, the TMLR text width: include each
# with \includegraphics[width=\linewidth]{...} so that the 7 pt text prints at 7 pt. (In a one-column layout a figure
# takes the full width of the page whatever its own width, so full width at a small height costs the least space.)
#  Figure 1  fig_schematic:  (a) rounding variants on two axes (systematic error kept/removed x draw kept/redrawn);
#                            (b) push, per-tensor ranges: success vs bit width under det and sh-iter (E12)
#  Figure 2  fig_damage:     sh-iter vs det damage in (a) every affected cell of E1/E10 and (b) every damaged cell of the
#                            replications (E11, E14, DINO-WM new split); (c) latent one-step error ratio vs extra damage
#                            (E1 errors on each variant's own planning states, E10 errors on held-out states: own style)
#  Figure 3  fig_mechanism:  (a) drift kept by each variant relative to det (E2); (b) damage vs MPPI iterations (E2)
#  Figure 4  fig_components: fraction of the damage removed (red in the text) by each control in every damaged cell,
#                            per model family (E11, E14, DINO-WM Wall and the post hoc PushT 3-bit cell), under a header
#                            line per control with the cause(s) of damage it was predicted to repair
#  Figure 5  fig_bitlimit:   damage vs bit width with k*, Meta-World per tensor and DINO-WM (E12; det+noise from X5,
#                            Meta-World only); the k* marks sit in a shaded strip below each panel, outside the data
#            fig_bitlimit_pc (appendix): the same for Meta-World with per-channel ranges
#  Figure 6  fig_iterations: (a) drift share S of det's OEI vs the fraction of damage removed by sh-iter (E3);
#                            (b-e) DMC: fraction of damage removed by sh-iter and det+shift vs MPPI iterations (E15)
#  Figure 7  fig_latency:    speed-ups over FP32 of W8A4 and of BF16 with pre-converted weights (second run, E8c):
#                            (a) world-model rollouts, (b) whole planning step; (c) E17 latency overhead of sh-iter and
#                            det+shift in the integer kernel
# Conventions
#  * One style per rounding variant in every figure (VS below). Hue = family, marker fill/shape = member:
#    det family orange (det filled square, det+BC open square), redrawn dither blue (sh-iter filled circle,
#    det+shift open diamond), fixed dither yellow (sh-ep / sh-plan filled down-triangle, det+shift-fixed open
#    diamond), sh-step aqua, det+noise neutral gray. Line style: dashed = deterministic grid, solid = redrawn dither,
#    dotted = sh-step / det+noise. So the five lines of Figure 5 need two hues plus neutral gray, which pass the
#    dataviz all-pairs CVD and normal-vision checks (validate_palette.js --pairs all: worst pair 18.3 / 21.8; the
#    dataviz palette passes all pairs only up to three hues), and every figure stays readable in grayscale through
#    marker shape, fill and line style.
#  * Text is drawn as paths (matplotlib PDF backend, text through a path effect), so the figure PDFs embed no fonts;
#    line dashes and transparency are kept in the PDF.
#  * No text is smaller than 6 pt, task names are the short ones (NAME), and the quantity red of the text is written
#    "fraction of damage removed" (the bare word "red" reads as a colour); save() asserts all three.
#  * Counts drawn in Figures 4 and 5 are asserted against the analyses' own results (tab_components.tex, E12/X5 G).
#  * `--captions` lists the values that the captions quote, with their result file and key (for \nv{} macros).
# Usage: python3 make_figs.py [--captions]   (writes ../figures/*.pdf and *.png previews)
import csv
import json
import math
import os
import re
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms
import pymupdf
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RES = os.path.join(A, "main", "results")
EXT = os.path.join(A, "rss_ext")
TX = os.path.join(A, "tmlr_ext")
OUT = os.path.join(HERE, "..", "figures")

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
# Categorical slots of the dataviz reference palette (light mode), fixed per variant across figures.
C_SHITER, C_DET, C_SHSTEP, C_SHEP = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_NOISE = INK2  # det+noise: neutral, so that Figure 5 needs no fourth hue next to orange and blue
# Latency panels: our integer path in blue, the BF16 baseline in neutral gray (marker shape and legend carry identity too).
C_INT, C_BF16 = "#2a78d6", INK2
C_E10 = INK2      # Figure 2(c): E10 errors were measured on held-out states, not on each variant's own planning states
C_STRIP = "#efeee9"  # Figure 5: fill of the k* strip below each panel
TEXT_W = 6.5  # TMLR text width (in)
TAU = 10.0    # damage threshold of k* (points of success; % of return on DMC)
MIN_PT = 6.0  # smallest text size in any figure (asserted in save())

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.pad": 2.5, "ytick.major.pad": 2.5, "axes.labelpad": 2.5,  # tighter than the defaults (page budget)
    "axes.labelcolor": INK, "text.color": INK,
})

# variant -> label, color, marker, line style, filled marker
VS = {
    "det": ("det", C_DET, "s", (0, (4, 1.5)), True),
    "detbc": ("det+BC", C_DET, "s", (0, (4, 1.5)), False),
    "shiter": ("sh-iter", C_SHITER, "o", "-", True),
    "detshift": ("det+shift", C_SHITER, "D", "-", False),
    "detnoise": ("det+noise", C_NOISE, "X", (0, (1, 1.2)), True),
    "shstep": ("sh-step", C_SHSTEP, "^", (0, (1, 1)), True),
    "shep": ("sh-ep", C_SHEP, "v", (0, (2.5, 1)), True),
    "detshiftfix": ("det+shift-fixed", C_SHEP, "D", (0, (2.5, 1)), False),
}

FAMILIES = {  # Figure 2 marker per model family: label -> (marker, size)
    "5M Meta-World": ("o", 15),
    "5M DMC": ("^", 16),
    "DINO-WM": ("s", 13),
    "317M Meta-World": ("D", 11),
}
MW = ["mw-push", "mw-pick-place", "mw-door-open", "mw-drawer-open", "mw-button-press-topdown",
      "mw-peg-insert-side", "mw-window-open", "mw-sweep-into"]
DMC = ["cheetah-run", "walker-walk", "quadruped-walk", "finger-turn-hard"]
NAME = {"mw-push": "push", "mw-pick-place": "pick-place", "mw-door-open": "door-open",
        "mw-drawer-open": "drawer-open", "mw-button-press-topdown": "button-press",
        "mw-peg-insert-side": "peg-insert", "mw-window-open": "window-open", "mw-sweep-into": "sweep-into",
        "cheetah-run": "cheetah", "walker-walk": "walker", "quadruped-walk": "quadruped",
        "finger-turn-hard": "finger"}
EPS = 1e-9  # tolerance of threshold comparisons (as in the analyses)
CAPTION = []  # (figure, quantity, value, source), listed by main() with --captions


def note(fig, what, value, source):
    CAPTION.append((fig, what, value, source))


def load(*parts):
    with open(os.path.join(*parts)) as fh:
        return json.load(fh)


def rows(*parts):
    with open(os.path.join(*parts)) as fh:
        return list(csv.DictReader(fh))


def ci(s):
    lo, hi = s.strip("[]").replace(";", ",").split(",")
    return float(lo), float(hi)


def tidy(ax, grid=True):
    if grid:
        ax.grid(color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def mk(var, ms=3.4, ring=0.4):
    """Marker keywords of a variant: filled with a white ring, or open (white face, colored edge)."""
    _, c, m, _, filled = VS[var]
    if filled:
        return dict(marker=m, ms=ms, mfc=c, mec="white", mew=ring)
    return dict(marker=m, ms=ms * 0.92, mfc="white", mec=c, mew=0.8)


def lk(var, lw=1.2):
    """Line keywords of a variant (open-marker members draw a thinner line)."""
    _, c, _, ls, filled = VS[var]
    return dict(color=c, ls=ls, lw=lw if filled else lw * 0.7)


def handle(var, line=True, ms=3.4, lw=1.2, label=None):
    kw = lk(var, lw) if line else dict(ls="none")
    if not line:
        kw["color"] = VS[var][1]
    return Line2D([], [], label=label or VS[var][0], **kw, **mk(var, ms))


def save(fig, name):
    """PDF with every text drawn as outlines (no embedded fonts) plus a PNG preview.
    Text artists get a Normal path effect, which makes matplotlib's PDF backend draw them as paths. (The earlier route,
    SVG converted by PyMuPDF, lost line dash patterns and transparency, so dashed lines came out solid in the PDF.)
    Creator, producer and date are left out so that unchanged data give a byte-identical PDF.
    Checks every visible text: at least MIN_PT, short task names only (NAME), and no bare word "red" (the paper's red
    is written "fraction of damage removed")."""
    os.makedirs(OUT, exist_ok=True)
    fig.canvas.draw()  # creates the tick labels, so that they get the path effect too
    shown = [t.get_text() for t in fig.findobj(mtext.Text) if t.get_visible() and t.get_text().strip()]
    small = sorted({(t.get_text(), t.get_fontsize()) for t in fig.findobj(mtext.Text)
                    if t.get_visible() and t.get_text().strip() and t.get_fontsize() < MIN_PT - 1e-9})
    assert not small, f"{name}: text below {MIN_PT} pt: {small}"
    assert not [s for s in shown for long in NAME if long != NAME[long] and long in s], f"{name}: long task name"
    assert not [s for s in shown if re.search(r"\bred\b", s)], f"{name}: bare word 'red'"
    for t in fig.findobj(mtext.Text):
        t.set_path_effects([patheffects.Normal()])
    fig.savefig(os.path.join(OUT, name + ".pdf"), format="pdf",
                metadata={"Creator": None, "Producer": None, "CreationDate": None})
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=220)
    with pymupdf.open(os.path.join(OUT, name + ".pdf")) as doc:  # check: no embedded font, no extractable text
        assert not doc[0].get_fonts() and not doc[0].get_text().strip(), f"{name}.pdf still contains text objects"
    w, h = fig.get_size_inches()
    plt.close(fig)
    print(f"wrote figures/{name}.pdf  ({w:.2f} x {h:.2f} in)")


# ================================================================================================ data
def affected_cells():
    """Every affected cell of the main stages: (family, task, k, D_det %, D_sh-iter %, latent 1-step error det, sh-iter)."""
    out = []
    by = {}
    for r in rows(RES, "E1_tdmpc2", "e1_tdmpc2_cells.csv"):
        by.setdefault((r["task"], r["k"], r["family"]), {})[r["variant"]] = r
    for (task, k, fam), v in by.items():
        if v["det"]["confirmatory"] == "True" and v["det"]["holm_real"] == "True":
            out.append(("5M Meta-World" if fam == "mw" else "5M DMC", task, int(k),
                        100 * float(v["det"]["D"]), 100 * float(v["shiter"]["D"]),
                        float(v["det"]["pt_err1h"]), float(v["shiter"]["pt_err1h"])))
    by = {}
    for r in rows(RES, "E1", "e1_dinowm_cells.csv"):
        if r["role"] == "confirmatory":
            by.setdefault((r["task"], r["k"]), {})[r["var"]] = r
    for (task, k), v in by.items():
        if v["det"]["drop_real_holm"] == "True":
            out.append(("DINO-WM", task, int(k), 100 * float(v["det"]["D"]), 100 * float(v["shiter"]["D"]),
                        float(v["det"]["rel_err1"]), float(v["shiter"]["rel_err1"])))
    for r in rows(EXT, "E10_mt317m", "results", "analysis", "e10_eval.csv"):
        if r["drop_real"] == "True":
            out.append(("317M Meta-World", r["task"], int(r["k"]), 100 * float(r["D_det"]), 100 * float(r["D_sh"]),
                        float(r["pt_err1h_det"]), float(r["pt_err1h_sh"])))
    return out


def td_summary():
    return load(TX, "TD_tdmpc2", "results", "analysis", "td_summary.json")


def dw_analysis():
    return load(TX, "DW_dinowm", "results", "analysis", "analysis_dw.json")


def dw_posthoc():
    return load(TX, "DW_dinowm", "results", "analysis_posthoc", "posthoc_pusht3.json")["cell"]


def e14_rows():
    return rows(TX, "MT_317m", "results", "analysis", "e14_eval.csv")


def replication_cells(td, dw, pu3, e14):
    """Every damaged cell of the replications: (family, task, k, D_det %, D_sh-iter %, post hoc)."""
    out = []
    c = td["E11"]["cells"]
    for t in MW + DMC:
        x = c[t]
        if x["damaged"]:
            out.append(("5M Meta-World" if x["unit"] == "mw" else "5M DMC", t, x["k"],
                        100 * x["variants"]["det"]["D"], 100 * x["variants"]["shiter"]["D"], False))
    for r in e14:
        if r["drop_real"] == "True":
            out.append(("317M Meta-World", r["task"], int(r["k"]), 100 * float(r["D_det"]), 100 * float(r["D_shiter"]),
                        False))
    for x in dw["E11"]["cells"]:
        if x["damaged"]:
            out.append(("DINO-WM", x["task"], x["k"], 100 * x["variants"]["det"]["D"],
                        100 * x["variants"]["shiter"]["D"], False))
    assert pu3["damaged"]
    out.append(("DINO-WM", pu3["task"], pu3["k"], 100 * pu3["variants"]["det"]["D"], 100 * pu3["variants"]["shiter"]["D"],
                True))
    return out


# ================================================================================================ Figure 1
def fig_schematic(td):
    e12 = td["E12"]["tasks"]["mw-push"]
    H = 1.55
    fig = plt.figure(figsize=(TEXT_W, H))
    # ---- (a) two-axis diagram, drawn in inch coordinates
    Wa = 4.3
    ax = fig.add_axes([0, 0, Wa / TEXT_W, 1])
    ax.set_xlim(0, Wa)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.text(0.03, H - 0.03, "(a) rounding variants", ha="left", va="top", fontsize=7)
    x0, x2 = 0.86, Wa - 0.02           # left and right edge of the grid
    x1 = (x0 + x2) / 2                 # column split
    e, pad = 0.19, 0.045               # height of one entry (name and description), padding inside a cell
    y3 = H - 0.29                      # top of the grid
    y2 = y3 - (2 * e + 2 * pad)        # row split (top row: 2 entries)
    y1 = y2 - (3 * e + 2 * pad)        # bottom (bottom row: 3 entries)
    assert y1 > 0.05
    for x in (x0, x1, x2):
        ax.plot([x, x], [y1, y3], color=GRID, lw=0.8, zorder=1)
    for y in (y1, y2, y3):
        ax.plot([x0, x2], [y, y], color=GRID, lw=0.8, zorder=1)
    ax.text((x0 + x1) / 2, y3 + 0.04, "same error in every planner iteration", ha="center", va="bottom", fontsize=6.6)
    ax.text((x1 + x2) / 2, y3 + 0.04, "random part redrawn at every call", ha="center", va="bottom", fontsize=6.6)
    ax.text(x0 - 0.07, (y2 + y3) / 2, "systematic\nerror kept:\ndeterministic\nor biased", ha="right", va="center",
            fontsize=6.3, linespacing=1.1)
    ax.text(x0 - 0.07, (y1 + y2) / 2, "systematic\nerror removed:\nunbiased\nrandom\nrounding", ha="right",
            va="center", fontsize=6.3, linespacing=1.1)
    cells = {  # (column, row) -> [(variant or None, name, description)]; row 1 = top
        (0, 1): [("det", "det", "round to nearest"),
                 ("detbc", "det+BC", "det minus its mean error")],
        (1, 1): [("detnoise", "det+noise", "det plus fresh shared noise"),
                 (None, "sh-iter+b", "sh-iter plus det's mean error")],
        (0, 0): [("shstep", "sh-step", "shared dither, kept for a step"),
                 ("shep", "sh-ep / sh-plan", "kept for an episode (or plan)"),
                 ("detshiftfix", "det+shift-fixed", "shifted grid, kept per episode")],
        (1, 0): [("shiter", "sh-iter", "shared dither, new every call"),
                 ("detshift", "det+shift", "shifted grid, new every call"),
                 (None, "ind", "unshared dither, new every call")],
    }
    for (col, row), items in cells.items():
        cx = (x0 if col == 0 else x1) + 0.08
        top = y3 if row == 1 else y2
        for i, (var, name, desc) in enumerate(items):
            y = top - pad - 0.055 - i * e
            if var is not None:
                ax.plot([cx, cx + 0.27], [y, y], **lk(var, 1.1), solid_capstyle="butt", zorder=3)
                ax.plot([cx + 0.135], [y], ls="none", **mk(var, 3.6), zorder=4)
            ax.text(cx + 0.34, y, name, ha="left", va="center", fontsize=6.6, color=INK if var else INK2)
            ax.text(cx + 0.34, y - 0.095, desc, ha="left", va="center", fontsize=MIN_PT, color=INK2)
    # ---- (b) representative result: push, per-tensor ranges, success vs bit width (E12)
    bl, bb, bt = 4.82, 0.32, H - 0.24
    bx = fig.add_axes([bl / TEXT_W, bb / H, (TEXT_W - 0.06 - bl) / TEXT_W, (bt - bb) / H])
    cells_ = e12["cells"]
    nones = {round(cells_[f"PT|{v}"][k]["mean"] + cells_[f"PT|{v}"][k]["D"], 9)
             for v in ("det", "shiter") for k in cells_[f"PT|{v}"]}
    assert len(nones) == 1
    none = 100 * nones.pop()
    bx.axhspan(none - TAU, none, color=GRID, alpha=0.5, lw=0, zorder=0)
    bx.axhline(none, color=MUTED, lw=0.7, ls=(0, (1, 1.5)), zorder=1)
    ks = {}
    for var, dx in (("det", -0.03), ("shiter", 0.03)):
        cc = cells_[f"PT|{var}"]
        kk = sorted(int(k) for k in cc)
        bx.plot([k + dx for k in kk], [100 * cc[str(k)]["mean"] for k in kk], **lk(var, 1.3), **mk(var, 3.6),
                zorder=3)
        ks[var] = e12["kstar"][f"PT|{var}"]["kstar"]
        kstar_check({int(k): 100 * x["D"] for k, x in cc.items()}, ks[var])
    bx.text(2.1, 100 * cells_["PT|det"]["2"]["mean"] + 3, "det", fontsize=6.3, ha="left", va="bottom")
    bx.text(2.1, 100 * cells_["PT|shiter"]["2"]["mean"] - 4, "sh-iter", fontsize=6.3, ha="left", va="top")
    bx.text(5.15, 4, f"within {TAU:.0f} points of\nunquantized from\nk = {ks['det']} (det), "
            f"k = {ks['shiter']} (sh-iter)", fontsize=MIN_PT, color=INK2, ha="right", va="bottom", linespacing=1.1)
    bx.set_xlim(1.8, 5.2)
    bx.set_xticks([2, 3, 4, 5])
    bx.set_ylim(0, 104)
    bx.set_yticks([0, 50, 100])
    bx.set_xlabel("bit width k")
    bx.set_ylabel("success (%)")
    bx.set_title(f"(b) {NAME['mw-push']}, per-tensor ranges", loc="left", pad=3)
    tidy(bx)
    src = "tmlr_ext/TD_tdmpc2/results/analysis/td_summary.json E12.tasks.mw-push (same values in td_e12_cells.csv)"
    note(1, "k* det (per tensor)", ks["det"], src + " kstar['PT|det'].kstar")
    note(1, "k* sh-iter (per tensor)", ks["shiter"], src + " kstar['PT|shiter'].kstar")
    for var in ("det", "shiter"):
        note(1, f"success {var} at k=2..5", " / ".join(f"{100 * cells_[f'PT|{var}'][k]['mean']:.1f}" for k in "2345"),
             src + f" cells['PT|{var}'][k].mean")
    note(1, "unquantized success", f"{none:.0f}", src + " (mean + D, identical in all 8 cells)")
    save(fig, "fig_schematic")


# ================================================================================================ Figure 2
DMG_LIM = (-12, 103)


def damage_scatter(ax, pts, title):
    ax.plot([0, 100], [0, 100], color=MUTED, lw=0.7, ls=(0, (1, 1.5)), zorder=1)
    ax.plot([0, 100], [0, 50], color=INK2, lw=0.7, ls=(0, (4, 2)), zorder=1)
    assert all(DMG_LIM[0] < y < DMG_LIM[1] and -3 < x < 103 for _, x, y, _ in pts)
    for fam, (m, s) in FAMILIES.items():
        p = [(x, y) for f, x, y, ph in pts if f == fam and not ph]
        if p:
            ax.scatter([q[0] for q in p], [q[1] for q in p], marker=m, s=s, facecolor=C_SHITER, edgecolor="white",
                       linewidth=0.5, alpha=0.9, zorder=3)
        p = [(x, y) for f, x, y, ph in pts if f == fam and ph]
        if p:  # post hoc cell: open marker
            ax.scatter([q[0] for q in p], [q[1] for q in p], marker=m, s=s + 3, facecolor="white", edgecolor=C_SHITER,
                       linewidth=0.9, zorder=4)
    ax.set_xlim(-3, 103)
    ax.set_ylim(*DMG_LIM)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 50, 100])
    for slope, x0, lab, col in ((1.0, 30, "no effect", MUTED), (0.5, 50, "50% removed", INK2)):
        (ax0, ay0), (ax1, ay1) = ax.transData.transform([(0, 0), (100, 100 * slope)])
        ang = math.degrees(math.atan2(ay1 - ay0, ax1 - ax0))
        ax.text(x0, slope * x0 + 3, lab, color=col, fontsize=MIN_PT, rotation=ang, rotation_mode="anchor",
                ha="left", va="bottom")
    ax.set_xlabel("damage under det (points or %)")  # success points (Meta-World, DINO-WM), % of return (DMC)
    ax.set_title(title, loc="left", pad=3)
    tidy(ax)


def fig_damage(cells, rep):
    H = 1.8
    fig = plt.figure(figsize=(TEXT_W, H))
    # axes placed in inches: y labels in two lines, and (c) wider than (a) and (b) so that its title fits the page
    bot, top = 0.35, H - 0.32
    box = lambda left, width: fig.add_axes([left / TEXT_W, bot / H, width / TEXT_W, (top - bot) / H])
    # (a) main stages
    ax = box(0.5, 1.62)
    damage_scatter(ax, [(c[0], c[3], c[4], False) for c in cells], "(a) main benchmark and 317M")
    ax.set_ylabel("damage under sh-iter\n(points or %)")
    wk = [(c[3], c[4]) for c in cells if c[1] == "walker-walk" and c[2] == 3][0]
    ax.annotate("walker k=3", xy=wk, xytext=(3, 88), fontsize=MIN_PT, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkB=3))
    # (b) replications on new seeds and splits
    bx = box(2.38, 1.62)
    damage_scatter(bx, [(r[0], r[3], r[4], r[5]) for r in rep], "(b) replications on new seeds and split")
    wk = [(r[3], r[4]) for r in rep if r[1] == "walker-walk"][0]
    bx.annotate("walker", xy=wk, xytext=(3, 88), fontsize=MIN_PT, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkB=3))
    # (c) det/sh-iter latent one-step error ratio vs extra damage of det (main stages). E1 measured the errors on each
    # variant's own planning states, E10 (317M) on held-out states of the unquantized planner: E10 in its own color.
    cx = box(4.46, 1.97)
    cx.axvline(1.0, color=MUTED, lw=0.7, ls=(0, (1, 1.5)), zorder=1)
    for fam, (m, s) in FAMILIES.items():
        p = [(c[5] / c[6], c[3] - c[4]) for c in cells if c[0] == fam]
        cx.scatter([q[0] for q in p], [q[1] for q in p], marker=m, s=s,
                   facecolor=C_E10 if fam == "317M Meta-World" else C_SHITER,
                   edgecolor="white", linewidth=0.5, alpha=0.9, zorder=3)
    assert all(0.25 < c[5] / c[6] < 3.2 and -38 < c[3] - c[4] < 106 for c in cells)
    cx.set_xscale("log", base=2)
    cx.set_xticks([0.25, 0.5, 1, 2])
    cx.set_xticklabels(["1/4", "1/2", "1", "2"])
    cx.minorticks_off()
    cx.set_xlim(0.25, 3.2)
    cx.set_ylim(-38, 106)
    cx.set_yticks([0, 50, 100])
    cx.axvspan(0.25, 1.0, color=GRID, alpha=0.3, lw=0, zorder=0)
    cx.text(0.27, -34, "det error ≤ sh-iter", fontsize=MIN_PT, color=INK2, va="bottom")
    # direct labels of the two kinds of error measurement, in empty corners (upper left, lower right)
    assert not [c for c in cells if c[5] / c[6] < 0.55 and c[3] - c[4] > 75]
    cx.text(0.265, 104, "317M:\nheld-out states", fontsize=MIN_PT, color=C_E10, ha="left", va="top",
            linespacing=1.05)
    assert not [c for c in cells if c[5] / c[6] > 1.1 and c[3] - c[4] < 0]
    cx.text(3.1, -34, "5M and DINO-WM:\nown planning states", fontsize=MIN_PT, color=C_SHITER, ha="right",
            va="bottom", linespacing=1.05)
    cx.set_xlabel("latent 1-step error, det / sh-iter")
    cx.set_ylabel("extra damage of det\n(points or %)")
    cx.set_title("(c) extra damage against error ratio", loc="left", pad=3)
    tidy(cx)
    handles = [Line2D([], [], ls="none", marker=m, ms=math.sqrt(s) * 1.05, mfc=C_SHITER, mec="white", mew=0.5,
                      label=fam) for fam, (m, s) in FAMILIES.items()]
    handles.append(Line2D([], [], ls="none", marker="s", ms=math.sqrt(16) * 1.05, mfc="white", mec=C_SHITER, mew=0.9,
                          label="post hoc cell (PushT, 3 bits)"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=5, frameon=False, fontsize=6.2,
               handletextpad=0.1, columnspacing=1.3, borderaxespad=0.15)
    src_a = "E1: main/results/E1_tdmpc2/e1_tdmpc2_cells.csv (det confirmatory & holm_real), " \
            "main/results/E1/e1_dinowm_cells.csv (confirmatory, det drop_real_holm); " \
            "E10: rss_ext/E10_mt317m/results/analysis/e10_eval.csv (drop_real)"
    src_b = "E11: td_summary.json E11.cells (damaged); E14: MT_317m/results/analysis/e14_eval.csv (drop_real); " \
            "DINO-WM: DW_dinowm/results/analysis/analysis_dw.json E11.cells (damaged) + analysis_posthoc/posthoc_pusht3.json"
    keys_a = {"5M Meta-World": "\\nv{e1.mw.naff}", "5M DMC": "\\nv{e1.dmc.naff}", "DINO-WM": "\\nv{e1.dw.naff}",
              "317M Meta-World": "\\nv{e10.naff}"}
    keys_b = {"5M Meta-World": "\\nv{e11.mw.ndam}", "5M DMC": "\\nv{e11.dmc.ndam}",
              "DINO-WM": "\\nv{d4.comb.ndam} (Wall + post hoc PushT)", "317M Meta-World": "\\nv{e14.ndam}"}
    for fam in FAMILIES:
        note(2, f"(a) cells {fam}", sum(c[0] == fam for c in cells), keys_a[fam] + "; " + src_a)
    for fam in FAMILIES:
        note(2, f"(b) cells {fam}", sum(r[0] == fam for r in rep), keys_b[fam] + "; " + src_b)
    e1 = [c for c in cells if c[0] != "317M Meta-World"]
    note(2, "(c) E1 cells with det error <= sh-iter", f"{sum(c[5] <= c[6] for c in e1)} of {len(e1)}",
         "\\nv{e1.h1.n} of \\nv{e1.h1.of}")
    e10 = [c for c in cells if c[0] == "317M Meta-World"]
    note(2, "(c) E10 tasks with det error <= sh-iter", f"{sum(c[5] <= c[6] for c in e10)} of {len(e10)}",
         "\\nv{e10.h1.n} of \\nv{e10.naff}")
    note(2, "(c) E10 tasks drawn in gray (errors on held-out states; E1 in blue: own planning states)", len(e10),
         "\\nv{e10.naff}; e10_eval.csv pt_err1h_det / pt_err1h_sh")
    mw_a = [c for c in cells if c[0] == "5M Meta-World"]
    note(2, "(a) E1 Meta-World cells at or below the 50% line", sum(c[4] <= c[3] / 2 + EPS for c in mw_a),
         "\\nv{e1.mw.nfix} of \\nv{e1.mw.naff}")
    dw_a = [c for c in cells if c[0] == "DINO-WM"]
    note(2, "(a) E1 DINO-WM cells at or below the 50% line", sum(c[4] <= c[3] / 2 + EPS for c in dw_a),
         "\\nv{e1.dw.nfix} of \\nv{e1.dw.naff}")
    rr = [r for r in rep if r[0] == "5M Meta-World"]
    note(2, "(b) E11 Meta-World cells at or below the 50% line", sum(r[4] <= r[3] / 2 + EPS for r in rr),
         "\\nv{e11.b4.n} of \\nv{e11.b4.of}")
    r317 = [r for r in rep if r[0] == "317M Meta-World"]
    note(2, "(b) E14 tasks at or below the 50% line", sum(r[4] <= r[3] / 2 + EPS for r in r317),
         "\\nv{e14.p1.n} of \\nv{e14.ndam}")
    save(fig, "fig_damage")


# ================================================================================================ Figure 3
def drift_ratios():
    """(task, {variant: (r, lo, hi)}) for the four mechanism tasks; PushT's fixed table is sh-plan (= sh-step = sh-ep)."""
    e2 = load(RES, "E2_tdmpc2", "e2_tdmpc2_summary.json")["layer_b"]
    out = []
    for task in ("mw-push", "mw-pick-place", "cheetah-run"):
        b = e2[task]["b2_H2a_prime"]
        out.append((task, {v: (b["r_" + v]["r"], *b["r_" + v]["ci"]) for v in ("shiter", "shstep", "shep")}))
    b = load(RES, "E1", "e1_dinowm_v52.json")["b2_H2a_prime_pusht_k4"]
    out.append(("PushT", {"shiter": (b["r_shiter"], *b["r_shiter_ci"]),
                          "shep": (b["positive_control_r_shplan"], *b["positive_control_r_shplan_ci"])}))
    return out


def e2_curves():
    e2 = {(r["task"], r["setting"], r["variant"]): r for r in rows(RES, "E2_tdmpc2", "e2_tdmpc2_cells.csv")}
    its, settings = [1, 3, 6, 12], ["it1", "it3", "base", "it12"]
    curves = {}
    for task in ("mw-push", "mw-pick-place", "cheetah-run"):
        for var in ("det", "shstep", "shiter"):
            d = [100 * float(e2[(task, s, var)]["D"]) for s in settings]
            c = [tuple(100 * x for x in ci(e2[(task, s, var)]["D_ci"])) for s in settings]
            curves[(task, var)] = (its, d, c)
    return curves


def e2_bits():
    """{task: primary bit width} of the E2 mechanism tasks."""
    e2 = load(RES, "E2_tdmpc2", "e2_tdmpc2_summary.json")["H2a_replication"]
    return {task: e2[task]["k"] for task in ("mw-push", "mw-pick-place", "cheetah-run")}


def fig_mechanism(ratios, curves, bits):
    H = 1.45
    fig = plt.figure(figsize=(TEXT_W, H))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.62, 1, 1, 1], wspace=0.36,
                          left=0.06, right=0.995, top=1 - 0.31 / H, bottom=0.315 / H)
    # (a) drift kept relative to det
    ax = fig.add_subplot(gs[0, 0])
    # det's own drift (= 1): drawn in det's color and dash, as det's entry in the legend
    ax.axhline(1.0, color=C_DET, lw=1.0, ls=VS["det"][3], zorder=1)
    ax.axhline(0.0, color=MUTED, lw=0.6, zorder=1)
    ax.text(3.62, 1.03, "det", color=C_DET, fontsize=6.2, ha="right", va="bottom")
    w = 0.25
    for i, (task, rs) in enumerate(ratios):
        vs = [v for v in ("shiter", "shstep", "shep") if v in rs]
        for j, v in enumerate(vs):
            x = i + (j - (len(vs) - 1) / 2) * (w + 0.03)
            r, lo, hi = rs[v]
            ax.bar(x, r, width=w, color=VS[v][1], edgecolor="white", linewidth=0.4, zorder=2)
            ax.vlines(x, lo, hi, color=INK2, lw=0.6, zorder=3)
    ax.text(3 + (w + 0.03) / 2, 0.12, "sh-plan", fontsize=MIN_PT, color=INK2, ha="center", va="bottom", rotation=90)
    ax.set_xticks(range(len(ratios)))
    ax.set_xticklabels([NAME.get(task, task) for task, _ in ratios], fontsize=6.2)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(-0.12, 1.5)
    ax.set_yticks([0, 0.5, 1.0, 1.5])
    ax.set_ylabel("drift / drift of det")
    ax.set_title("(a) drift kept by each variant", loc="left", pad=3)
    tidy(ax)
    ax.grid(axis="x", visible=False)
    # (b) damage vs MPPI iterations, at the primary bit width of each task
    for j, task in enumerate(("mw-push", "mw-pick-place", "cheetah-run")):
        ax = fig.add_subplot(gs[0, j + 1])
        for var, dodge in (("det", 0.92), ("shstep", 1.0), ("shiter", 1.08)):
            its, d, c = curves[(task, var)]
            xs = [x * dodge for x in its]  # small offset on the log axis so the CI bars do not overlap
            ax.vlines(xs, [lo for lo, _ in c], [hi for _, hi in c], color=VS[var][1], lw=0.6, alpha=0.6, zorder=2)
            ax.plot(xs, d, **lk(var, 1.3), **mk(var, 3.4), zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 3, 6, 12])
        ax.set_xticklabels(["1", "3", "6", "12"])
        ax.minorticks_off()
        ax.set_xlim(0.85, 14)
        ax.set_ylim(-4, 72)
        ax.set_yticks([0, 20, 40, 60])
        ax.set_xlabel("MPPI iterations")
        if j == 0:
            ax.set_ylabel("damage\n(points or %)")  # success points (Meta-World), % of return (cheetah)
        ax.set_title(("(b) " if j == 0 else "") + f"{NAME[task]}, {bits[task]} bits",
                     loc="left", pad=3)
        tidy(ax)
    proxies = [handle("det", label="det (round to nearest)", lw=1.3),
               handle("shiter", label="sh-iter (shared dither, new every call)", lw=1.3),
               handle("shstep", label="sh-step (kept for a step)", lw=1.3),
               Patch(facecolor=C_SHEP, edgecolor="white", label="sh-ep / sh-plan (kept for an episode)")]
    fig.legend(handles=proxies, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=4, frameon=False, fontsize=6.2,
               handlelength=2.5, handletextpad=0.4, columnspacing=1.5, borderaxespad=0.2)
    save(fig, "fig_mechanism")


# ================================================================================================ Figure 4
COMP_COLS = [("shep", "sh-ep / sh-plan"), ("detshiftfix", "det+shift-fixed"), ("detnoise", "det+noise"),
             ("detbc", "det+BC"), ("shiter", "sh-iter"), ("detshift", "det+shift")]
# (group, first column, last column, the design's prediction). A mean bias is repaired by det+BC and by every unbiased
# variant; a resolution loss by every randomized threshold, fixed or redrawn, but not by det+BC; a persistent error
# pattern only by redrawing. det+noise keeps det's rounding and adds fresh shared noise: the design predicted that it
# repairs at most half of the damaged cells of every benchmark.
COMP_GROUPS = [("fixed unbiased error", 0, 1, "repairs if: bias or resolution"),
               ("fresh noise", 2, 2, "predicted: at most half"),
               ("bias removed", 3, 3, "repairs if: bias"),
               ("redrawn dither", 4, 5, "repairs if: bias, resolution or persistence")]
RED_LO, RED_HI = -1.0, 1.5
UNIT_KEY = {"TD-MPC2 5M\nMeta-World": "mw", "TD-MPC2 5M\nDMC": "dmc",
            "TD-MPC2 317M\nMeta-World": "317m", "DINO-WM\n(new split)": "dw"}


def component_units(td, dw, pu3, e14):
    """[(unit label, [(row label, post hoc, {variant: (red, lo, hi)})])] for the damaged cells of each family unit."""
    c = td["E11"]["cells"]
    units = []
    for unit, tasks, lab in (("mw", MW, "TD-MPC2 5M\nMeta-World"), ("dmc", DMC, "TD-MPC2 5M\nDMC")):
        rws = []
        for t in tasks:
            x = c[t]
            assert x["damaged"]
            rws.append((NAME[t], False, {v: (x["variants"][v]["red"], *x["variants"][v]["red_ci"])
                                         for v, _ in COMP_COLS}))
        units.append((lab, rws))
    rws = []
    for r in e14:
        assert r["drop_real"] == "True"
        vals = {}
        for v in ("shep", "detbc", "shiter", "detshift"):
            vals[v] = (float(r[f"red_{v}"]), *ci(r[f"red_{v}_ci"]))
        rws.append((NAME[r["task"]], False, vals))
    units.append(("TD-MPC2 317M\nMeta-World", rws))
    wall = [x for x in dw["E11"]["cells"] if x["damaged"]]
    assert [(x["task"], x["k"]) for x in wall] == [("wall", 3)]
    rws = []
    for cell, lab, ph in ((wall[0], "Wall, 3 bits", False), (pu3, "PushT, 3 bits†", True)):
        vals = {}
        for v, _ in COMP_COLS:
            key = "shstep" if v == "shep" else v  # open-loop CEM: sh-step = sh-ep = sh-plan
            x = cell["variants"][key]
            vals[v] = (x["red"], *x["red_ci"])
        rws.append((lab, ph, vals))
    units.append(("DINO-WM\n(new split)", rws))
    return units


def check_component_counts(counts, td, dw, pu3):
    """The repaired-cell counts drawn in Figure 4 must equal the analyses' own (tab_components.tex uses the same)."""
    u = td["E11"]["units"]
    j14 = load(TX, "MT_317m", "results", "analysis", "e14_judgement.json")
    fix14 = j14["descriptive"]["fixes_red_ge_half_among_damaged"]
    wall = [x for x in dw["E11"]["cells"] if x["damaged"]][0]
    for var, _ in COMP_COLS:
        assert counts[("mw", var)] == (u["mw"]["n_fix"][var], u["mw"]["n_damaged"]), var
        assert counts[("dmc", var)] == (u["dmc"]["n_fix"][var], u["dmc"]["n_damaged"]), var
        if var in fix14:
            assert counts[("317m", var)] == (len(fix14[var]), j14["P1"]["n_damaged"]), var
        else:
            assert ("317m", var) not in counts
        key = "shstep" if var == "shep" else var
        n = int(wall["variants"][key]["red"] >= 0.5 - EPS) + int(pu3["variants"][key]["red"] >= 0.5 - EPS)
        assert counts[("dw", var)] == (n, 2), var
    for k, v in sorted(counts.items(), key=lambda kv: (kv[0][0], [c for c, _ in COMP_COLS].index(kv[0][1]))):
        note(4, f"repaired {k[0]} {k[1]}", f"{v[0]}/{v[1]}", "td_summary.json E11.units.<unit>.n_fix / "
             "e14_judgement.json descriptive.fixes_red_ge_half_among_damaged / analysis_dw.json + posthoc_pusht3.json red")


def fig_components(units):
    # row pitch, extra space between family units, margin above the first and below the last row (row units)
    pitch, gap, pad_top, pad_bot = 1.0, 0.4, 0.5, 0.7
    ys, y = [], 0.0
    for _, rws in units:
        block = []
        for _ in rws:
            block.append(y)
            y += pitch
        ys.append(block)
        y += gap
    ymax = y - gap - pitch
    row_in = 0.0775  # inches per row: 6 pt row labels (ascender to descender 5.6 pt) do not touch
    top_in, bot_in = 0.375, 0.16  # header stack above the columns; tick labels below (axis name left of them)
    fig_h = top_in + bot_in + row_in * (ymax + pad_top + pad_bot)
    fig = plt.figure(figsize=(TEXT_W, fig_h))
    left, right = 0.215, 0.995
    gs = fig.add_gridspec(1, len(COMP_COLS), wspace=0.12, left=left, right=right, top=1 - top_in / fig_h,
                          bottom=bot_in / fig_h)
    x_cnt = RED_HI + 0.42  # where the count of repaired cells is printed
    axes = []
    counts = {}
    for j, (var, title) in enumerate(COMP_COLS):
        ax = fig.add_subplot(gs[0, j])
        axes.append(ax)
        ax.set_xlim(RED_LO - 0.1, x_cnt + 0.28)
        ax.set_ylim(ymax + pad_bot, -pad_top)
        ax.axvline(0.5, color=INK2, lw=0.7, ls=(0, (3, 1.5)), zorder=1)
        for xv in (0.0, 1.0):
            ax.axvline(xv, color=GRID, lw=0.5, zorder=0)
        ax.set_xticks([-1, 0, 0.5, 1])
        ax.set_xticklabels(["−1", "0", ".5", "1"])
        ax.tick_params(axis="x", labelsize=6, pad=1.5)
        ax.spines["bottom"].set_bounds(RED_LO, RED_HI)
        ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.set_title(title, pad=2.5, fontsize=6.6)
        _, col, m, _, filled = VS[var]
        for (ulab, rws), block in zip(units, ys):
            n_fix = n_have = 0
            for (rlab, ph, vals), yy in zip(rws, block):
                if var not in vals:
                    continue
                r, lo, hi = vals[var]
                n_have += 1
                n_fix += r >= 0.5 - EPS
                ax.plot([max(lo, RED_LO), min(hi, RED_HI)], [yy, yy], color=col, lw=0.8, solid_capstyle="butt",
                        zorder=2)
                for edge, beyond in ((RED_LO, lo < RED_LO), (RED_HI, hi > RED_HI)):
                    if beyond:  # interval runs past the axis: arrowhead at the edge
                        ax.plot([edge], [yy], ls="none", marker="<" if edge < 0 else ">", ms=2.6, mfc=col, mec=col,
                                mew=0, zorder=3)
                kw = mk(var, {"X": 3.9, "v": 4.0, "^": 4.0}.get(m, 3.3), ring=0.4)
                if ph:  # post hoc cell: open marker with a dark edge
                    kw = dict(marker=m, ms=3.6, mfc="white", mec=INK, mew=0.75)
                if RED_LO - EPS <= r <= RED_HI + EPS:
                    ax.plot([r], [yy], ls="none", zorder=4, **kw)
                else:  # point estimate off the axis: print it at the edge
                    ax.text(RED_LO + 0.08, yy, f"{r:.1f}".replace("-", "\u2212"), fontsize=MIN_PT, color=INK2,
                            ha="left", va="center", zorder=5,
                            bbox=dict(boxstyle="square,pad=0.06", fc="white", ec="none"))
            mid = (block[0] + block[-1]) / 2
            if n_have:
                ax.text(x_cnt, mid, f"{n_fix}/{n_have}", fontsize=6.2, color=INK, ha="center", va="center")
                counts[(UNIT_KEY[ulab], var)] = (n_fix, n_have)
            else:
                ax.text((RED_LO + RED_HI) / 2, mid, "not run", fontsize=MIN_PT, color=MUTED, ha="center",
                        va="center", zorder=5, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    # row labels (task) and family-unit labels, in figure coordinates
    b = axes[0].get_position()
    to_fig = lambda yy: b.y0 + (b.y1 - b.y0) * (ymax + pad_bot - yy) / (ymax + pad_top + pad_bot)
    for (ulab, rws), block in zip(units, ys):
        for (rlab, ph, vals), yy in zip(rws, block):
            fig.text(b.x0 - 0.004, to_fig(yy), rlab, fontsize=MIN_PT, ha="right", va="center")
        fig.text(0.004, to_fig((block[0] + block[-1]) / 2), ulab, fontsize=6.2, ha="left", va="center",
                 linespacing=1.1)
    # group headers over the columns: group name, the design's prediction (one line), rule, column titles
    for gname, j0, j1, pred in COMP_GROUPS:
        b0, b1 = axes[j0].get_position(), axes[j1].get_position()
        y_rule = b0.y1 + 0.15 / fig_h
        xc = (b0.x0 + b1.x1) / 2
        fig.text(xc, y_rule + 0.115 / fig_h, gname, ha="center", va="bottom", fontsize=6.6)
        fig.text(xc, y_rule + 0.02 / fig_h, pred, ha="center", va="bottom", fontsize=MIN_PT, color=INK2)
        fig.lines.append(Line2D([b0.x0 + 0.004, b1.x1 - 0.004], [y_rule] * 2, transform=fig.transFigure,
                                color=MUTED, lw=0.6))
    # x axis name level with the tick labels (left margin, as in Figure 5); key of the dashed line and counts top left
    fig.text(b.x0 - 0.07 / TEXT_W, (bot_in - 4 / 72) / fig_h, "fraction of damage removed", ha="right", va="top",
             fontsize=6.2, color=INK2)
    fig.text(0.004, 1 - 0.02 / fig_h, "dashed line: half removed\nn/N: cells repaired\n(at least half removed)",
             ha="left", va="top", fontsize=MIN_PT, color=INK2, linespacing=1.1)
    save(fig, "fig_components")
    return counts


# ================================================================================================ Figure 5
def bit_curves(td, dw):
    """Panels of Figure 5: [(row, title, {var: {k: D %}}, {var: k*})]."""
    e12 = td["E12"]["tasks"]
    x5 = load(TX, "TD_tdmpc2", "results", "analysis_x5", "x5_summary.json")
    out = []
    for rg in ("PT", "PC"):
        for t in MW:
            curves, ks = {}, {}
            for var in ("det", "detbc", "shiter", "detshift"):
                cc = e12[t]["cells"][f"{rg}|{var}"]
                curves[var] = {int(k): 100 * x["D"] for k, x in cc.items()}
                ks[var] = e12[t]["kstar"][f"{rg}|{var}"]["kstar"]
            cv = x5["curves"][f"{t}|{rg}"]
            curves["detnoise"] = {int(k): 100 * x["D"] for k, x in cv.items()}
            ks["detnoise"] = x5["kstar"][f"{t}|{rg}"]["kstar_detnoise"]
            # x5's copies of the E12 k* must equal E12's
            for var in ("det", "detbc", "shiter", "detshift"):
                assert x5["kstar"][f"{t}|{rg}"][f"kstar_{var}"] == ks[var], (t, rg, var)
            out.append((rg, NAME[t], curves, ks))
    for task, rgs in (("pusht", ("main", "pc", "rc")), ("wall", ("main", "pc")), ("pointmaze", ("main", "pc"))):
        e = dw["E12"][task]
        for rg in rgs:
            curves, ks = {}, {}
            for var in ("det", "detbc", "shiter"):
                key = f"{rg}|{var}"
                if key in e["kstar"]:
                    curves[var] = {int(k): 100 * x["D"] for k, x in e["cells"][rg][var].items()}
                    ks[var] = e["kstar"][key]["kstar"]
            out.append(("DW", (task, rg), curves, ks))
    return out


def kstar_check(curve, k_star):
    """k* = smallest tested k with D <= tau at that and every larger tested k (recomputed here as a check)."""
    kk = sorted(curve)
    for k in kk:
        if all(curve[j] <= TAU + 1e-7 for j in kk if j >= k):
            assert k == k_star, (curve, k_star)
            return
    raise AssertionError((curve, k_star))


BIT_XLIM = (0.55, 6.45)
BIT_YLIM = {"MW": (-9, 104), "DW": (-28, 104)}  # damage (points); the k* marks are drawn outside, in the strip
BIT_X0, BIT_X1, BIT_COLGAP = 0.86, TEXT_W - 0.03, 0.09  # panel area (in) and the gap between panels
BIT_PW = (BIT_X1 - BIT_X0 - 7 * BIT_COLGAP) / 8  # panel width (in): eight panels per row
BIT_PH, BIT_GAP = 0.43, 0.03  # height of the data axes, gap between data axes and k* strip (in)
BIT_STRIP = {2: 0.125, 1: 0.075}  # height of the k* strip (in) with two lanes (det+noise on Meta-World) or one
BIT_TICK, BIT_BOT = 0.105, 0.035  # tick labels under a strip; margin below the last row (in)
SUB_DW = {("pusht", "main"): "per tensor", ("pusht", "pc"): "per channel", ("pusht", "rc"): "rollout-chosen",
          ("wall", "main"): "per tensor", ("wall", "pc"): "per channel",
          ("pointmaze", "main"): "per tensor", ("pointmaze", "pc"): "per channel"}
KSTAR_KEYS = (("s", C_DET, 3.8, "det or det+BC"), ("o", C_SHITER, 3.5, "sh-iter or det+shift"),
              ("X", C_NOISE, 3.3, "det+noise"))  # k* marks: better det-family k*, better dither k*, det+noise


def bit_row(fig, H, plist, top, G, ph=BIT_PH):
    """One row of Figure 5. Each panel is a data axes (height ph, top edge at `top`, in inches from the bottom of the
    figure) and, directly below it, a shaded strip that holds the k* marks, so that no mark can be read as a data
    point: in the upper lane the smaller k* of det and det+BC (square) and of sh-iter and det+shift (circle), joined by
    a line whose length is the point estimate of G; on Meta-World a lower lane with k* of det+noise (cross). The
    bit-width tick labels sit under the strip. Fills G[(range, title)] = (k* det family, k* dither, k* det+noise) and
    returns the data axes and the bottom of the strips (in)."""
    n_lanes = 2 if any("detnoise" in ks for _, _, _, ks in plist) else 1
    sh = BIT_STRIP[n_lanes]
    lanes = (0.67, 0.27) if n_lanes == 2 else (0.5,)
    y_ax = top - ph
    y_st = y_ax - BIT_GAP - sh
    order = ["det", "detbc", "detnoise", "detshift", "shiter"]  # draw order: sh-iter on top
    axes = []
    for j, (rg, title, curves, ks) in enumerate(plist):
        x0 = BIT_X0 + j * (BIT_PW + BIT_COLGAP)
        ax = fig.add_axes([x0 / TEXT_W, y_ax / H, BIT_PW / TEXT_W, ph / H])
        axes.append(ax)
        yl = BIT_YLIM["DW" if rg == "DW" else "MW"]
        ax.set_ylim(*yl)
        ax.set_xlim(*BIT_XLIM)
        ax.axhline(TAU, color=INK2, lw=0.6, ls=(0, (1, 1.2)), zorder=1)
        for var in order:
            if var not in curves:
                continue
            kk = sorted(curves[var])
            assert all(yl[0] < curves[var][k] < yl[1] for k in kk), (title, var)
            kstar_check(curves[var], ks[var])
            ax.plot(kk, [curves[var][k] for k in kk], **lk(var, 1.0), **mk(var, 2.6, ring=0.3), zorder=3,
                    clip_on=False)
        ax.set_xticks(range(1, 7))
        ax.set_yticks([0, 50, 100])
        ax.tick_params(axis="x", length=0, labelbottom=False)
        ax.tick_params(axis="y", labelsize=MIN_PT, length=2, pad=1.5)
        if j > 0:
            ax.set_yticklabels([])
        tidy(ax)
        # the k* strip
        sx = fig.add_axes([x0 / TEXT_W, y_st / H, BIT_PW / TEXT_W, sh / H])
        sx.set_xlim(*BIT_XLIM)
        sx.set_ylim(0, 1)
        sx.set_facecolor(C_STRIP)
        for sp in sx.spines.values():
            sp.set_visible(False)
        sx.set_yticks([])
        sx.set_xticks(range(1, 7))
        sx.tick_params(axis="x", labelsize=MIN_PT, length=2, pad=1.5, color=MUTED)
        sx.grid(axis="x", color="white", lw=0.8)
        sx.set_axisbelow(True)
        kd = min(ks[v] for v in ("det", "detbc") if v in ks)
        kdi = min(ks[v] for v in ("shiter", "detshift") if v in ks)
        G[(rg, title)] = (kd, kdi, ks.get("detnoise"))
        (m_d, c_d, ms_d, _), (m_i, c_i, ms_i, _), (m_n, c_n, ms_n, _) = KSTAR_KEYS
        if kd != kdi:
            sx.plot([kdi, kd], [lanes[0]] * 2, color=INK2, lw=0.8, zorder=4, solid_capstyle="butt")
        sx.plot([kd], [lanes[0]], ls="none", marker=m_d, ms=ms_d, mfc=c_d, mec="white", mew=0.4, zorder=5)
        sx.plot([kdi], [lanes[0]], ls="none", marker=m_i, ms=2.6 if kd == kdi else ms_i, mfc=c_i, mec="white",
                mew=0.4, zorder=6)
        if "detnoise" in ks:
            sx.plot([ks["detnoise"]], [lanes[1]], ls="none", marker=m_n, ms=ms_n, mfc=c_n, mec="white", mew=0.35,
                    zorder=7)
    # left margin, level with what it names: y label of the data axes, the strip, the x axis (tick labels)
    fig.text((BIT_X0 - 0.25) / TEXT_W, (y_ax + ph / 2) / H, "damage", rotation=90, ha="center", va="center",
             fontsize=MIN_PT, color=INK2)
    fig.text((BIT_X0 - 0.04) / TEXT_W, (y_st + sh / 2) / H, "k*", ha="right", va="center", fontsize=6.2, color=INK2)
    fig.text((BIT_X0 - 0.04) / TEXT_W, (y_st - 3.5 / 72) / H, "bit width k", ha="right", va="top", fontsize=6.2,
             color=INK2)
    return axes, y_st


def bit_legend(fig, noise_label):
    lines = [handle(v, ms=2.8, lw=1.0) for v in ("det", "detbc", "shiter", "detshift")]
    lines.append(handle("detnoise", ms=2.8, lw=1.0, label=noise_label))
    lines.append(Line2D([], [], color=INK2, lw=0.6, ls=(0, (1, 1.2)), label=f"τ = {TAU:.0f} points"))
    fig.legend(handles=lines, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=6, frameon=False, fontsize=6.2,
               handlelength=2.4, handletextpad=0.35, columnspacing=1.1, borderaxespad=0.15)


def kstar_proxies(wrap=False):
    return [Line2D([], [], ls="none", marker=m, ms=ms_, mfc=c, mec="white", mew=0.4,
                   label=lab.replace(" or ", " or\n") if wrap and len(lab) > 14 else lab)
            for m, c, ms_, lab in KSTAR_KEYS]


def fig_bitlimit(panels):
    """Main figure: Meta-World with per-tensor ranges (with the post hoc det+noise curves) and DINO-WM. The Meta-World
    row with per-channel ranges is fig_bitlimit_pc (appendix)."""
    mw_pt = [p for p in panels if p[0] == "PT"]
    dwp = [p for p in panels if p[0] == "DW"]
    assert len(mw_pt) == 8 and len(dwp) == 7
    # data axes a little lower than in fig_bitlimit_pc (page budget of the main text)
    ph, head_in, sub_in, title_in, legend_in, row_gap = 0.39, 0.105, 0.125, 0.125, 0.16, 0.05
    h_mw = ph + BIT_GAP + BIT_STRIP[2]
    h_dw = ph + BIT_GAP + BIT_STRIP[1]
    top_dw = BIT_BOT + BIT_TICK + h_dw
    top_mw = top_dw + sub_in + head_in + row_gap + BIT_TICK + h_mw
    H = top_mw + title_in + legend_in
    G = {}
    fig = plt.figure(figsize=(TEXT_W, H))
    axes, _ = bit_row(fig, H, mw_pt, top_mw, G, ph)
    for ax, (_, title, _, _) in zip(axes, mw_pt):
        ax.set_title(title, pad=2.5, fontsize=6.4)
    axes, st_dw = bit_row(fig, H, dwp, top_dw, G, ph)
    for ax, (_, key, _, _) in zip(axes, dwp):
        ax.set_title(SUB_DW[key], pad=2.0, fontsize=MIN_PT, color=INK2)
    # row labels and DINO-WM task headers
    for top, lab in ((top_mw, "Meta-World,\nper tensor"), (top_dw, "DINO-WM")):
        fig.text(0.006, (top - ph / 2) / H, lab, ha="left", va="center", fontsize=6.6, linespacing=1.1)
    y_rule = top_dw + sub_in
    for name, j0, j1 in (("PushT", 0, 2), ("Wall", 3, 4), ("PointMaze", 5, 6)):
        xa = BIT_X0 + j0 * (BIT_PW + BIT_COLGAP)
        xb = BIT_X0 + j1 * (BIT_PW + BIT_COLGAP) + BIT_PW
        fig.text((xa + xb) / 2 / TEXT_W, (y_rule + 0.015) / H, name, ha="center", va="bottom", fontsize=6.6)
        fig.lines.append(Line2D([(xa + 0.02) / TEXT_W, (xb - 0.02) / TEXT_W], [y_rule / H] * 2,
                                transform=fig.transFigure, color=MUTED, lw=0.6))
    bit_legend(fig, "det+noise (Meta-World only)")
    # key of the k* marks in the free eighth slot of the DINO-WM row
    xa = BIT_X0 + 7 * (BIT_PW + BIT_COLGAP) - 0.04
    y0 = st_dw - BIT_TICK + 0.02
    lx = fig.add_axes([xa / TEXT_W, y0 / H, (TEXT_W - 0.01 - xa) / TEXT_W, (y_rule + 0.14 - y0) / H])
    lx.axis("off")
    lx.legend(handles=kstar_proxies(wrap=True), loc="lower left", bbox_to_anchor=(-0.04, 0.0), frameon=False,
              fontsize=MIN_PT,
              title="k* (in the strip):\nsmallest k with\nD ≤ τ from\nthere on", title_fontsize=MIN_PT,
              alignment="left", handlelength=0.9, handletextpad=0.3, labelspacing=0.35, borderaxespad=0.0,
              borderpad=0.0)
    save(fig, "fig_bitlimit")
    return G


def fig_bitlimit_pc(panels):
    """Appendix figure: Meta-World with per-channel ranges, drawn as the rows of fig_bitlimit."""
    mw_pc = [p for p in panels if p[0] == "PC"]
    assert len(mw_pc) == 8
    title_in, legend_in = 0.125, 0.3
    top = BIT_BOT + BIT_TICK + BIT_PH + BIT_GAP + BIT_STRIP[2]
    H = top + title_in + legend_in
    G = {}
    fig = plt.figure(figsize=(TEXT_W, H))
    axes, _ = bit_row(fig, H, mw_pc, top, G)
    for ax, (_, title, _, _) in zip(axes, mw_pc):
        ax.set_title(title, pad=2.5, fontsize=6.4)
    fig.text(0.006, (top - BIT_PH / 2) / H, "Meta-World,\nper channel", ha="left", va="center", fontsize=6.6,
             linespacing=1.1)
    bit_legend(fig, "det+noise (post hoc)")
    fig.legend(handles=[Line2D([], [], ls="none", label="k* (in the strip), smallest k with D ≤ τ from there on:")]
               + kstar_proxies(), loc="upper left", bbox_to_anchor=(0.0, 1.0 - 0.15 / H), ncol=4, frameon=False,
               fontsize=6.2, handlelength=0.9, handletextpad=0.3, columnspacing=1.1, borderaxespad=0.15)
    save(fig, "fig_bitlimit_pc")
    return G


def check_bitlimit(G, td, dw):
    """Bit gains drawn in Figure 5 (per tensor and DINO-WM) and in fig_bitlimit_pc (Meta-World per channel) against
    the analyses' own G (td_summary E12, analysis_dw E12, x5_kstar)."""
    e12 = td["E12"]["tasks"]
    x5 = load(TX, "TD_tdmpc2", "results", "analysis_x5", "x5_summary.json")["kstar"]
    n_ge1 = {}
    for rg in ("PT", "PC"):
        where = "fig_bitlimit" if rg == "PT" else "fig_bitlimit_pc"
        for t in MW:
            kd, kdi, kn = G[(rg, NAME[t])]
            assert kd - kdi == e12[t]["G"][rg], (t, rg)
            assert min(x5[f"{t}|{rg}"]["kstar_det"], x5[f"{t}|{rg}"]["kstar_detbc"]) - kn == x5[f"{t}|{rg}"]["G_noise"]
            n_ge1[rg] = n_ge1.get(rg, 0) + (kd - kdi >= 1)
        note(5, f"Meta-World tasks with G >= 1 ({rg}, {where})", f"{n_ge1[rg]} of {len(MW)}",
             f"\\nv{{e12.{rg.lower()}.nG}} of \\nv{{e12.mw.n}} (td_summary.json E12.tasks.<task>.G.{rg})")
        nn = sum(min(x5[f"{t}|{rg}"]["kstar_det"], x5[f"{t}|{rg}"]["kstar_detbc"]) - G[(rg, NAME[t])][2] >= 1 for t in MW)
        note(5, f"Meta-World tasks with G_noise >= 1 ({rg}, {where})", f"{nn} of {len(MW)}",
             f"\\nv{{x5.{rg.lower()}.nG}} (x5_summary.json counts.{rg}.G_noise_ge_1)")
    for task, rgs in (("pusht", ("main", "pc", "rc")), ("wall", ("main", "pc")), ("pointmaze", ("main", "pc"))):
        for rg in rgs:
            kd, kdi, _ = G[("DW", (task, rg))]
            assert kd - kdi == dw["E12"][task]["G"][rg]["G"], (task, rg)
            note(5, f"{task} {rg}: G (point estimate)", kd - kdi, f"analysis_dw.json E12.{task}.G.{rg}.G")


# ================================================================================================ Figure 6
def fig_iterations(td):
    e3 = load(RES, "E3_tdmpc2", "e3_tdmpc2_final.json")["DMC_explanation"]
    e15 = td["E15"]["tasks"]
    H = 1.45
    fig = plt.figure(figsize=(TEXT_W, H))
    # axes placed in inches: (a) and (b) carry y labels (two lines), (c-e) share (b)'s scale
    bot, top = 0.3, H - 0.32
    box = lambda left, width: fig.add_axes([left / TEXT_W, bot / H, width / TEXT_W, (top - bot) / H])
    # (a) drift share S of det's OEI vs the fraction of damage removed by sh-iter at the default iterations (E3)
    ax = box(0.5, 1.38)
    ax.axhline(0.5, color=INK2, lw=0.6, ls=(0, (3, 1.5)), zorder=1)
    ax.axhline(0.0, color=GRID, lw=0.5, zorder=0)
    for r in e3["rows"]:
        mw = r["task"].startswith("mw-")
        lo, hi = r["red_shiter_ci"]
        ax.vlines(r["S_det"], lo, min(hi, 1.45), color=C_SHITER, lw=0.6, alpha=0.6, zorder=2)
        ax.plot([r["S_det"]], [r["red_shiter"]], ls="none", marker="o" if mw else "^", ms=3.4 if mw else 3.8,
                mfc=C_SHITER, mec="white", mew=0.4, zorder=3)
        if not mw:  # label the four DMC tasks (finger sits next to Meta-World points: label on its left)
            right = r["task"] != "finger-turn-hard"
            ax.text(r["S_det"] + (0.014 if right else -0.014), r["red_shiter"], NAME[r["task"]], fontsize=MIN_PT,
                    color=INK2, ha="left" if right else "right", va="center")
    ax.text(0.655, 0.36, "Meta-World", fontsize=MIN_PT, color=INK2, ha="center", va="center")
    ax.set_xlim(0.28, 0.78)
    ax.set_ylim(-0.55, 1.45)
    ax.set_xticks([0.3, 0.5, 0.7])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("drift share S = drift / OEI of det")
    ax.set_ylabel("fraction of damage\nremoved by sh-iter")
    ax.set_title("(a) sh-iter, 12 tasks", loc="left", pad=3)
    ax.text(0.77, -0.5, f"Spearman {e3['spearman_S_red']:.2f}", fontsize=MIN_PT, color=INK2, ha="right",
            va="bottom")
    tidy(ax)
    note(6, "(a) Spearman S vs red", f"{e3['spearman_S_red']:.2f}", "\\nv{e3.spearman} (main/results/E3_tdmpc2/e3_tdmpc2_final.json DMC_explanation.spearman_S_red)")
    s_dmc = [r["S_det"] for r in e3["rows"] if not r["task"].startswith("mw-")]
    s_mw = [r["S_det"] for r in e3["rows"] if r["task"].startswith("mw-")]
    note(6, "(a) S range DMC / Meta-World", f"{min(s_dmc):.2f}-{max(s_dmc):.2f} / {min(s_mw):.2f}-{max(s_mw):.2f}",
         "\\nv{e3.share.dmc}, \\nv{e3.share.mw} (e3_tdmpc2_final.json DMC_explanation.rows[].S_det)")
    # (b-e) DMC: fraction of damage removed vs iterations
    its = [6, 12, 24]
    for j, t in enumerate(DMC):
        bx = box(2.38 + j * (0.91 + 0.15), 0.91)
        bx.axhline(0.5, color=INK2, lw=0.6, ls=(0, (3, 1.5)), zorder=1)
        bx.axhline(0.0, color=GRID, lw=0.5, zorder=0)
        for var, dodge in (("detshift", 1.06), ("shiter", 0.94)):
            xs = [i * dodge for i in its]
            vv = [e15[t]["iterations"][str(i)]["variants"][var] for i in its]
            bx.vlines(xs, [v["red_ci"][0] for v in vv], [min(v["red_ci"][1], 1.45) for v in vv], color=VS[var][1],
                      lw=0.6, alpha=0.6, zorder=2)
            bx.plot(xs, [v["red"] for v in vv], **lk(var, 1.2), **mk(var, 3.4), zorder=3)
        bx.set_xscale("log", base=2)
        bx.set_xticks(its)
        bx.set_xticklabels([str(i) for i in its])
        bx.minorticks_off()
        bx.set_xlim(4.9, 29.5)
        bx.set_ylim(-0.55, 1.45)
        bx.set_yticks([0, 0.5, 1.0])
        if j > 0:
            bx.set_yticklabels([])
        else:
            bx.set_ylabel("fraction of\ndamage removed")
        bx.set_xlabel("MPPI iterations")
        k = td["E11"]["cells"][t]["k"]
        bx.set_title(("(b) " if j == 0 else "") + f"{NAME[t]} ({k} bits)", loc="left", pad=3)
        tidy(bx)
        rb = [e15[t]["iterations"][str(i)]["red_best"] for i in its]
        note(6, f"{t} red of the better dither at 6/12/24", " / ".join(f"{x:.2f}" for x in rb),
             "td_summary.json judgements.It2.per_task.<task>.red_best_6_12_24 (12: \\nv{e15.<task>.red12})")
        for i in its:
            note(6, f"{t} n at {i}", e15[t]["iterations"][str(i)]["n"], f"td_summary.json E15.tasks.{t}.iterations.{i}.n")
    proxies = [handle("shiter", lw=1.2), handle("detshift", lw=1.2),
               Line2D([], [], color=INK2, lw=0.6, ls=(0, (3, 1.5)), label="half removed (repaired)")]
    fig.legend(handles=proxies, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False, fontsize=6.2,
               handlelength=2.6, handletextpad=0.4, columnspacing=1.6, borderaxespad=0.2)
    save(fig, "fig_iterations")


# ================================================================================================ Figure 7
def e8c_speedups(mode):
    """{size: {"int": [fp32 / int-det per N], "bf16c": [fp32 / pre-converted bf16 per N]}} from the second latency run
    (E8c), as CUDA graphs: mode "graph" = world-model rollouts of one MPPI step, "gstep" = whole planning step (the
    rest of the step in FP32)."""
    out = {}
    for r in rows(EXT, "E8c_latency", "results", "analysis", "e8c_cells.csv"):
        if r["mode"] == mode and (mode == "graph" or r["rest"] == "fp32"):
            fp32 = float(r["T_fp32_ms"])
            d = out.setdefault(int(r["size"]), {"int": [], "bf16c": []})
            d["int"].append(fp32 / float(r["T_int-det_ms"]))
            d["bf16c"].append(fp32 / float(r["T_bf16cast_ms"]))
    assert sorted(out) == [5, 19, 48, 317] and all(len(v) == 3 for d in out.values() for v in d.values()), mode
    return out


def e17_overheads():
    kl = load(TX, "K_kernel", "results", "raw", "k.1", "analysis_lat", "e17_lat_summary.json")
    return kl


def fig_latency(roll, step, kl):
    sizes = [5, 19, 48, 317]
    H = 1.45
    fig = plt.figure(figsize=(TEXT_W, H))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.3, left=0.065, right=0.995, top=1 - 0.2 / H,
                          bottom=0.3 / H)
    # (a) world-model rollouts of one MPPI step and (b) the whole planning step, both from the second run (E8c):
    # W8A4 integer and BF16 with the weights converted once in advance, median over N, bar = range over N
    ax = fig.add_subplot(gs[0, 0])
    bx = fig.add_subplot(gs[0, 1], sharey=ax)
    ymax = 5.0
    for a, data in ((ax, roll), (bx, step)):
        assert all(1.0 < v < ymax for d in data.values() for vs in d.values() for v in vs)
        for key, col, m, dx, lab in (("int", C_INT, "o", -0.11, "W8A4 integer"),
                                     ("bf16c", C_BF16, "s", 0.11, "BF16, pre-converted weights")):
            xs = [i + dx for i in range(4)]
            a.vlines(xs, [min(data[s][key]) for s in sizes], [max(data[s][key]) for s in sizes], color=col, lw=0.8,
                     zorder=2)
            a.plot(xs, [statistics.median(data[s][key]) for s in sizes], ls="none", color=col, marker=m, ms=3.6,
                   mec="white", mew=0.4, zorder=3, label=lab)
    bx.legend(loc="upper left", frameon=False, fontsize=MIN_PT, handletextpad=0.3, labelspacing=0.25,
              borderaxespad=0.2)  # one legend for (a) and (b), where (b) leaves room above the points
    ax.set_ylabel("speed-up over\nFP32 (×)")
    ax.set_title("(a) world-model rollouts", loc="left", pad=3)
    bx.tick_params(labelleft=False)
    bx.set_title("(b) whole planning step", loc="left", pad=3)
    for a in (ax, bx):
        a.axhline(1.0, color=MUTED, lw=0.7, ls=(0, (1, 1.5)), zorder=1)
        a.set_xticks(range(4))
        a.set_xticklabels([f"{s}M" for s in sizes])
        a.set_xlim(-0.35, 3.35)
        a.set_ylim(0, ymax)
        a.set_yticks([0, 1, 2, 3, 4])
        a.set_xlabel("model size")
        tidy(a)
    src = "rss_ext/E8c_latency/results/analysis/e8c_cells.csv"
    for p, d, mode, keys in (("(a)", roll, "graph", ("\\nv{e8c.speedup}", "no macro")),
                             ("(b)", step, "gstep, rest fp32", ("\\nv{e8c.step.speedup}", "no macro"))):
        for (key, lab), mac in zip((("int", "W8A4"), ("bf16c", "BF16 pre-converted")), keys):
            v = [x for s in sizes for x in d[s][key]]
            note(7, f"{p} speed-up over FP32, {lab} (E8c, {len(v)} conditions)", f"{min(v):.2f}-{max(v):.2f}x",
                 f"{mac} ({src} mode={mode}, T_fp32_ms / T_{'int-det' if key == 'int' else 'bf16cast'}_ms)")
            note(7, f"{p} {lab}: median over N per size {'/'.join(f'{s}M' for s in sizes)}",
                 " / ".join(f"{statistics.median(d[s][key]):.2f}" for s in sizes), src)
        n_bf = sum(b > i for s in sizes for i, b in zip(d[s]["int"], d[s]["bf16c"]))
        note(7, f"{p} conditions where pre-converted BF16 is faster than W8A4", f"{n_bf} of {3 * len(sizes)}",
             ("\\nv{e8c.int.over.bf16}" if p == "(a)" else "\\nv{e8c.step.over.bf16}") + " = W8A4 time over BF16")
    # (c) E17: latency overhead of the dither variants over int det, fused integer kernel (median, 95% interval)
    cx = fig.add_subplot(gs[0, 2])
    conds = [("5", "512"), ("5", "2048"), ("5", "8192"), ("317", "512"), ("317", "2048"), ("317", "8192")]
    xpos = [0, 1, 2, 3.4, 4.4, 5.4]
    cx.axhline(0.0, color=MUTED, lw=0.7, zorder=1)
    for var, dx in (("shiter", -0.13), ("detshift", 0.13)):
        key = "int-" + var
        med = [100 * kl["cells"][f"{s}|{n}"][key]["median"] for s, n in conds]
        lo = [100 * kl["cells"][f"{s}|{n}"][key]["ci"][0] for s, n in conds]
        hi = [100 * kl["cells"][f"{s}|{n}"][key]["ci"][1] for s, n in conds]
        xs = [x + dx for x in xpos]
        cx.vlines(xs, lo, hi, color=VS[var][1], lw=0.8, zorder=2)
        cx.plot(xs, med, ls="none", **mk(var, 3.6), zorder=3)
    cx.set_xticks(xpos)
    cx.set_xticklabels(["512", "2048", "8192"] * 2, fontsize=6)
    cx.set_xlim(-0.5, 5.9)
    cx.set_ylim(-0.6, 8.2)
    cx.set_yticks([0, 2, 4, 6, 8])
    cx.set_ylabel("latency\noverhead (%)")
    cx.set_xlabel("candidates N")
    trans = mtransforms.blended_transform_factory(cx.transData, cx.transAxes)
    cx.text(1, 0.97, "5M", ha="center", va="top", fontsize=6.2, color=INK2, transform=trans)
    cx.text(4.4, 0.97, "317M", ha="center", va="top", fontsize=6.2, color=INK2, transform=trans)
    cx.axvline(2.7, color=GRID, lw=0.6, zorder=0)
    cx.set_title("(c) dither overhead, integer kernel", loc="left", pad=3)
    tidy(cx)
    cx.grid(axis="x", visible=False)
    cx.legend(handles=[handle("shiter", line=False, ms=3.6), handle("detshift", line=False, ms=3.6)],
              loc="upper right", bbox_to_anchor=(1.0, 0.88), frameon=False, fontsize=MIN_PT, handletextpad=0.2,
              labelspacing=0.25, borderaxespad=0.2)
    note(7, "(c) sh-iter max median overhead", f"{100 * kl['shiter_descriptive']['max_median']:.2f}%",
         "\\nv{e17.shiter.overhead.max} (K_kernel/results/raw/k.1/analysis_lat/e17_lat_summary.json shiter_descriptive.max_median)")
    note(7, "(c) det+shift at N=512", f"{100 * kl['cells']['5|512']['int-detshift']['median']:.2f}% (5M), "
         f"{100 * kl['cells']['317|512']['int-detshift']['median']:.2f}% (317M)",
         "\\nv{e17.detshift.5m512}, \\nv{e17.detshift.317m512}")
    note(7, "(c) det+shift at N>=2048", "", "\\nv{e17.detshift.big}")
    save(fig, "fig_latency")


# ================================================================================================ main
def main():
    td = td_summary()
    dw = dw_analysis()
    pu3 = dw_posthoc()
    e14 = e14_rows()
    fig_schematic(td)
    cells = affected_cells()
    rep = replication_cells(td, dw, pu3, e14)
    fig_damage(cells, rep)
    fig_mechanism(drift_ratios(), e2_curves(), e2_bits())
    check_component_counts(fig_components(component_units(td, dw, pu3, e14)), td, dw, pu3)
    panels = bit_curves(td, dw)
    G = fig_bitlimit(panels)
    G.update(fig_bitlimit_pc(panels))
    check_bitlimit(G, td, dw)
    fig_iterations(td)
    fig_latency(e8c_speedups("graph"), e8c_speedups("gstep"), e17_overheads())
    if "--captions" in sys.argv:
        print("\ncaption values (figure | quantity | value | source):")
        for f, what, value, src in CAPTION:
            print(f"  Fig {f} | {what} | {value} | {src}")
    else:
        print(f"{len(CAPTION)} caption values recorded with their sources (list them: make_figs.py --captions)")


if __name__ == "__main__":
    main()
