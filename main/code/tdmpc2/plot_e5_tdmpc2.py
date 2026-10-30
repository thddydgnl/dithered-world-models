"""Descriptive figure of the TD-MPC2 E5 H3b cells (reads e5_tdmpc2_summary.json only; nothing is computed here):
D(B+det) - D(B+sh-iter) with its 95% paired-bootstrap CI per cell, Meta-World and DMC in separate panels (different
measures: success-rate drop vs relative return drop). Filled marker = the cell passes (CI lower > 0), hollow = not.
usage (server, CPU only): MPLBACKEND=Agg python plot_e5_tdmpc2.py --e5 <summary> --out <png>
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"mw": "#2a78d6", "dmc": "#eb6834"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e5", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = json.load(open(a.e5))
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2), facecolor=SURF, gridspec_kw={"width_ratios": [12, 7]})
    for ax, fam, xl in ((axs[0], "mw", "D(B+det) - D(B+sh-iter), success rate"),
                        (axs[1], "dmc", "D(B+det) - D(B+sh-iter), rel. return")):
        rows = [(c, r) for c, r in S["H3b_cells"].items() if r["family"] == fam]
        for i, (c, r) in enumerate(rows):
            lo, hi = r["Da_minus_Db_ci"]
            y = len(rows) - 1 - i
            ax.plot([lo, hi], [y, y], color=COL[fam], lw=2)
            ax.plot(r["Da_minus_Db"], y, marker="o", ms=7, color=COL[fam], mec=COL[fam],
                    mfc=COL[fam] if r["passes"] else SURF, ls="none")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{c.replace('mw-', '').replace('|', ' k=')} ({r['B']})" for c, r in rows][::-1],
                           fontsize=8, color=INK)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_facecolor(SURF)
        ax.grid(True, axis="x", color=GRID, linewidth=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
        n = sum(r["passes"] for _, r in rows)
        ax.set_title(f"{'Meta-World' if fam == 'mw' else 'DMC'}: {n}/{len(rows)} cells with CI lower > 0", color=INK,
                     fontsize=10)
        ax.set_xlabel(xl, color=INK2, fontsize=9)
    h = S["H3b"]
    fig.suptitle(f"E5 TD-MPC2 H3b: best PTQ baseline B + sh-iter vs B + det ({h['passing']}/{h['cells']} cells pass; "
                 f"filled = passes)", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(a.out, dpi=110, facecolor=SURF)
    print("figure written")


if __name__ == "__main__":
    main()
