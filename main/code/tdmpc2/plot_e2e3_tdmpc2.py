"""Small descriptive figures of the TD-MPC2 E2 / E3 analyses (read the summary JSONs only; no statistic is computed here).
  e2_iteration_curves.png  D vs MPPI iterations, warm start on (solid) and off (dashed), det / sh-iter / sh-step
  e2_h2c_nsweep.png        H2c confirmatory fixed-count sweep: |OEI| and D vs N (two rows, one measure per axis)
  e3_table_sweep.png       H4a table-size sweep: D of the shared plan-fixed tables vs N_eff, fresh controls beside
Colors: the dataviz reference palette's first three categorical slots (validated all-pairs, light surface); the line
style / marker shape carries the second encoding, the numbers are in the reports' tables.
usage (server, CPU only): MPLBACKEND=Agg python plot_e2e3_tdmpc2.py --e2 <e2 summary> --e3 <e3 summary> --out2 <dir> --out3 <dir>
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"det": "#2a78d6", "shiter": "#eb6834", "shstep": "#1baf7a"}
MARK = {"det": "o", "shiter": "s", "shstep": "^"}
NAME = {"det": "det", "shiter": "sh-iter", "shstep": "sh-step"}


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURF)
    ax.set_title(title, color=INK, fontsize=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def err(pts, key="D", ci="D_ci"):
    y = [p[key] for p in pts]
    lo = [p[key] - p[ci][0] for p in pts]
    hi = [p[ci][1] - p[key] for p in pts]
    return y, [lo, hi]


def ylab(task):
    return "D (success-rate drop)" if task.startswith("mw-") else "D (relative return drop)"


def fig_iterations(S, path):
    tasks = list(S["iteration_curves"])
    fig, axs = plt.subplots(1, len(tasks), figsize=(4.0 * len(tasks), 3.3), facecolor=SURF)
    for ax, t in zip(axs, tasks):
        cur = S["iteration_curves"][t]
        for v in ("det", "shiter", "shstep"):
            for warm, ls in (("warm", "-"), ("no_warm", "--")):
                pts = cur.get(f"{warm}|{v}", [])
                if not pts:
                    continue
                x = [p["iterations"] for p in pts]
                y, e = err(pts)
                ax.errorbar(x, y, yerr=e, color=COL[v], ls=ls, lw=2, marker=MARK[v], ms=6, capsize=2,
                            mfc=COL[v] if warm == "warm" else SURF, mec=COL[v],
                            label=f"{NAME[v]}, warm start {'on' if warm == 'warm' else 'off'}")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 3, 6, 12])
        ax.set_xticklabels(["1", "3", "6", "12"])
        style(ax, t, "MPPI iterations", ylab(t))
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=8, frameon=False, labelcolor=INK)
    fig.suptitle("E2 TD-MPC2: damage vs iterations (95% paired-bootstrap CI)", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
    fig.savefig(path, dpi=110, facecolor=SURF)
    plt.close(fig)


def fig_nsweep(S, path):
    tasks = list(S["H2c"])
    fig, axs = plt.subplots(2, len(tasks), figsize=(4.0 * len(tasks), 5.6), facecolor=SURF)
    for j, t in enumerate(tasks):
        h = S["H2c"][t]
        for v in ("det", "shiter", "shstep"):
            pts = h["decomposition"][v]
            x = [p["N"] for p in pts]
            yo = [abs(p["oei"]) for p in pts]
            axs[0, j].plot(x, yo, color=COL[v], lw=2, marker=MARK[v], ms=6, label=NAME[v])
            y, e = err(pts)
            axs[1, j].errorbar(x, y, yerr=e, color=COL[v], lw=2, marker=MARK[v], ms=6, capsize=2, label=NAME[v])
        for i in (0, 1):
            axs[i, j].set_xscale("log", base=2)
            axs[i, j].set_xticks([64, 128, 256, 512, 1024])
            axs[i, j].set_xticklabels(["64", "128", "256", "512", "1024"])
        style(axs[0, j], f"{t}: rho(N,|OEI_det|) = {h['rho_N_absOEI_det']:.2f}", "candidates N (elite 64 fixed)",
              "|OEI| (value units)")
        style(axs[1, j], f"rho(N, D_det) = {h['rho_N_D_det']:.2f}", "candidates N (elite 64 fixed)", ylab(t))
    hd, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(hd, lab, loc="lower center", ncol=3, fontsize=8, frameon=False, labelcolor=INK)
    fig.suptitle("E2 TD-MPC2 H2c confirmatory sweep (fixed elite count)", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path, dpi=110, facecolor=SURF)
    plt.close(fig)


def fig_table(S3, path):
    tasks = list(S3["sweep"])
    fig, axs = plt.subplots(1, len(tasks), figsize=(3.6 * len(tasks), 3.3), facecolor=SURF)
    for ax, t in zip(axs, tasks):
        sw = S3["sweep"][t]
        x = sw["N_eff"]
        y = sw["D_shfixed"]
        e = [[yy - c[0] for yy, c in zip(y, sw["D_shfixed_ci"])], [c[1] - yy for yy, c in zip(y, sw["D_shfixed_ci"])]]
        ax.errorbar(x, y, yerr=e, color=COL["det"], lw=2, marker="o", ms=6, capsize=2,
                    label="shared plan-fixed table (sh-fixed)")
        fx, fy, fe = [], [], [[], []]
        for nm, ne in (("shiter_l", 4), ("shiter_g32", 128)):
            if f"{nm}_D" in sw:
                fx.append(ne)
                fy.append(sw[f"{nm}_D"])
                fe[0].append(sw[f"{nm}_D"] - sw[f"{nm}_D_ci"][0])
                fe[1].append(sw[f"{nm}_D_ci"][1] - sw[f"{nm}_D"])
        sh = S3["cells"][t]["variants"]["shiter"]
        fx.append(6144)
        fy.append(sh["D"])
        fe[0].append(sh["D"] - sh["D_ci"][0])
        fe[1].append(sh["D_ci"][1] - sh["D"])
        ax.errorbar(fx, fy, yerr=fe, color=COL["shiter"], lw=0, elinewidth=2, marker="s", ms=6, capsize=2,
                    label="fresh draw, same granularity")
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(["4", "128", "2048", "6144"])
        style(ax, f"{t}: Spearman = {sw['spearman']:.2f}", "N_eff (independent offsets per cost)", ylab(t))
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, fontsize=8, frameon=False, labelcolor=INK)
    fig.suptitle("E3 TD-MPC2 H4a table-size sweep (95% paired-bootstrap CI)", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    fig.savefig(path, dpi=110, facecolor=SURF)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2", required=True)
    ap.add_argument("--e3", required=True)
    ap.add_argument("--out2", required=True)
    ap.add_argument("--out3", required=True)
    a = ap.parse_args()
    S2, S3 = json.load(open(a.e2)), json.load(open(a.e3))
    fig_iterations(S2, f"{a.out2}/e2_iteration_curves.png")
    fig_nsweep(S2, f"{a.out2}/e2_h2c_nsweep.png")
    fig_table(S3, f"{a.out3}/e3_table_sweep.png")
    print("figures written")


if __name__ == "__main__":
    main()
