"""Latency-success figure of the TD-MPC2 E7 grid (reads e7_tdmpc2_summary.json only): one panel per task, x = median
control-step latency (ms; int4 = the container kernel of the success runs), y = mean success (Meta-World) or return
(DMC) with the 95% paired-bootstrap CI; one curve per precision across MPPI iterations 3 -> 6 -> 12. Hue = precision
family (float grey, int8 blue, int4 orange), hollow = det, filled = sh-iter dither, dashed = bf16. Descriptive: the
overall latency-success Pareto front of the 18 configurations per task is written to --front (configurations not
dominated by any other: latency <= and success >= with one strict; point estimates).
usage (server, CPU only): MPLBACKEND=Agg python plot_e7_tdmpc2.py --e7 <summary> --out <png> --front <json>
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
STY = {("fp32", "none"): ("#52514e", "-", "o", True), ("bf16", "none"): ("#52514e", "--", "s", True),
       ("int8", "det"): ("#2a78d6", "-", "o", False), ("int8", "shiter"): ("#2a78d6", "-", "o", True),
       ("int4", "det"): ("#eb6834", "-", "^", False), ("int4", "shiter"): ("#eb6834", "-", "^", True)}
NAME = {("fp32", "none"): "fp32", ("bf16", "none"): "bf16", ("int8", "det"): "int8 det", ("int8", "shiter"): "int8 dither",
        ("int4", "det"): "int4 det", ("int4", "shiter"): "int4 dither"}
ITERS = [3, 6, 12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e7", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--front", required=True)
    a = ap.parse_args()
    S = json.load(open(a.e7))["success"]
    tasks = ["mw-push", "mw-pick-place", "cheetah-run", "walker-walk"]
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.2), facecolor=SURF)
    front = {}
    for ax, t in zip(axs, tasks):
        pts = []
        for (p, v), (col, ls, mk, filled) in STY.items():
            xs, ys, lo, hi = [], [], [], []
            for it in ITERS:
                r = S[f"{t}|{p}_{v}|it{it}"]
                x = r[f"latency_ms_{p}"]
                xs.append(x)
                ys.append(r["mean"])
                lo.append(r["mean"] - r["ci"][0])
                hi.append(r["ci"][1] - r["mean"])
                pts.append({"config": f"{p}_{v}|it{it}", "latency_ms": x, "success": r["mean"]})
            ax.errorbar(xs, ys, yerr=[lo, hi], color=col, ls=ls, lw=2, marker=mk, ms=7, capsize=2,
                        mfc=col if filled else SURF, mec=col, label=NAME[(p, v)])
            for x, y, it in zip(xs, ys, ITERS):
                if (p, v) == ("fp32", "none"):
                    ax.annotate(str(it), (x, y), textcoords="offset points", xytext=(0, 8), fontsize=7, color=INK2,
                                ha="center")
        nd = [q for q in pts if not any(o["latency_ms"] <= q["latency_ms"] and o["success"] >= q["success"] and
                                        (o["latency_ms"] < q["latency_ms"] or o["success"] > q["success"]) for o in pts)]
        front[t] = sorted(nd, key=lambda q: q["latency_ms"])
        ax.set_facecolor(SURF)
        ax.set_title(t, color=INK, fontsize=10)
        ax.set_xlabel("median control-step latency (ms)", color=INK2, fontsize=9)
        ax.set_ylabel("success rate" if t.startswith("mw-") else "return", color=INK2, fontsize=9)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.grid(True, color=GRID, linewidth=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=6, fontsize=8, frameon=False, labelcolor=INK)
    fig.suptitle("E7 TD-MPC2 real kernels: latency vs success (MPPI iterations 3, 6, 12 along each curve; 95% CI)",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(a.out, dpi=110, facecolor=SURF)
    json.dump({"label": "descriptive: overall Pareto front of the 18 configurations per task (point estimates, int4 "
                        "latency = container kernel)", "front": front}, open(a.front, "w"), indent=1)
    print(json.dumps({t: [q["config"] for q in f] for t, f in front.items()}))


if __name__ == "__main__":
    main()
