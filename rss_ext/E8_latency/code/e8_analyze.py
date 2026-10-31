"""E8 analysis (rss_ext/DESIGN_ext.md 1.4-1.5), written before the main measurement.

Pre-registered computations (graph-WM):
  T(condition) = median of all block values (block value = median per-call time of the block); ratio = ratio of T
  pair overhead of a cell = median over its 10 pairs of (shiter block - det block) / det block; 95% CI = percentile
  bootstrap of the median, 10,000 resamples of the pairs, numpy RandomState(0) per cell (E7 post-hoc convention)
  L1 holds if T(int-det) / T(fp32) <= 0.80 in at least one of the 12 cells
  L2 holds if T(int-det) / min(T(tf32), T(bf16)) <= 0.90 in at least one cell
  L3 holds if the median pair overhead <= 5% in all 12 cells and its CI upper bound <= 10% in at least 10 cells
  (threshold comparisons with tolerance 1e-9)
Descriptive (defined before the main measurement; DESIGN_ext 1.5 and PLAN.md):
  crossover per size: the smallest N with T(int-det) < T(fp32), and with T(int-det) < min(T(tf32), T(bf16))
  eager-WM vs graph-WM (ratios and absolute times), eager-step ratios (N = 512)
  (5M, N = 512): E7 reproduction (eager-step int-det / fp32 of the new path and of E7's fused kernels vs E7's A5000
  value) and E7 fused vs the new general path in every mode and N
  round-bootstrap 95% CI of the ratios (10,000 resamples of the 10 rounds, all conditions resampled jointly by round,
  RandomState(0)); measurement state (SM clock, temperature, power, clock-event reasons, host load, CPU busy) and the
  share of blocks more than 15% slower than their condition's T
usage: python e8_analyze.py --main <dir with e8_main.json> --smoke <smoke dir> --outdir <dir>
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

SIZES = (5, 19, 48, 317)
NS = (512, 2048, 8192)
FLOATS = ("fp32", "tf32", "bf16")
NB = 10000
EPS = 1e-9
E7_A5000 = {"label": "E7 (RTX A5000, eager-step, N 512, trained 5M checkpoints): int det / fp32 latency, median over "
                     "4 tasks x MPPI iterations {3, 6, 12} (reports/E7_tdmpc2.md 6)", "ratio_int4_det": 1.177,
            "ratio_int8_det": 1.182}


def le(x, thr):
    return x <= thr + EPS


def boot_median_ci(x):
    x = np.asarray(x, float)
    idx = np.random.RandomState(0).randint(0, len(x), size=(NB, len(x)))
    m = np.median(x[idx], axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def key(mode, size, N, prec, impl):
    return f"{mode}|{size}|{N}|{prec}|{impl}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--smoke", default=None)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    M = json.load(open(f"{a.main}/e8_main.json"))
    sel = M["settings"]["selection"]
    blocks, pairs = M["blocks"], M["pairs"]
    rounds = sorted({b["round"] for b in blocks})
    byk = defaultdict(dict)                       # key -> round -> block value
    for b in blocks:
        byk[b["key"]][b["round"]] = b["median_ms"]
    T = {k: float(np.median(list(v.values()))) for k, v in byk.items()}

    def cond_keys(mode, size, N):
        impl = sel[str(size)]
        d = {p: key(mode, size, N, p, "-") for p in FLOATS}
        d["int-det"], d["int-shiter"] = key(mode, size, N, "int-det", impl), key(mode, size, N, "int-shiter", impl)
        if size == 5:
            d["e7-det"], d["e7-shiter"] = key(mode, size, N, "e7-det", "e7"), key(mode, size, N, "e7-shiter", "e7")
        return {p: k for p, k in d.items() if k in T}

    def ratios_of(t):
        r = {f"{p}/fp32": t[p] / t["fp32"] for p in t if p != "fp32"}
        fast = min(t["tf32"], t["bf16"])
        r["int-det/minfast"] = t["int-det"] / fast
        r["int-shiter/minfast"] = t["int-shiter"] / fast
        r["minfast_is"] = "tf32" if t["tf32"] <= t["bf16"] else "bf16"
        if "e7-det" in t:
            r["e7-det/int-det"] = t["e7-det"] / t["int-det"]
            r["e7-det/minfast"] = t["e7-det"] / fast
        return r

    # round-bootstrap of the ratios (descriptive): resample rounds jointly
    rs = np.random.RandomState(0)
    ridx = rs.randint(0, len(rounds), size=(NB, len(rounds)))
    rarr = {k: np.array([v.get(r, np.nan) for r in rounds]) for k, v in byk.items()}

    def boot_T(k):
        x = rarr[k][ridx]
        return np.nanmedian(x, axis=1)

    cells = {}
    for mode in ("graph", "eager", "step"):
        for size in SIZES:
            for N in (NS if mode != "step" else (512,)):
                ck = cond_keys(mode, size, N)
                if not {"fp32", "tf32", "bf16", "int-det", "int-shiter"} <= set(ck):
                    continue
                t = {p: T[k] for p, k in ck.items()}
                c = {"mode": mode, "size": size, "N": N, "impl": sel[str(size)], "T_ms": t, "ratios": ratios_of(t)}
                bt = {p: boot_T(k) for p, k in ck.items()}
                fastb = np.minimum(bt["tf32"], bt["bf16"])
                c["ratio_ci_rounds"] = {
                    "int-det/fp32": [float(np.percentile(bt["int-det"] / bt["fp32"], q)) for q in (2.5, 97.5)],
                    "int-det/minfast": [float(np.percentile(bt["int-det"] / fastb, q)) for q in (2.5, 97.5)]}
                for prec, impl in (("int", sel[str(size)]), ("e7", "e7")):
                    ov = [p["overhead"] for p in pairs if p["mode"] == mode and p["size"] == size and p["N"] == N
                          and p["prec"] == prec and p["impl"] == impl]
                    if ov:
                        c[f"overhead_{prec}"] = {"pairs": ov, "median": float(np.median(ov)),
                                                 "ci": boot_median_ci(ov) if len(ov) > 1 else [float("nan")] * 2,
                                                 "n": len(ov),
                                                 "pooled": float((T[ck[f"{prec}-shiter"]] - T[ck[f"{prec}-det"]])
                                                                 / T[ck[f"{prec}-det"]])}
                cells[f"{mode}|{size}|{N}"] = c

    g = {k: v for k, v in cells.items() if v["mode"] == "graph"}
    l1_cells = [k for k, v in g.items() if le(v["ratios"]["int-det/fp32"], 0.80)]
    l2_cells = [k for k, v in g.items() if le(v["ratios"]["int-det/minfast"], 0.90)]
    med_ok = [k for k, v in g.items() if le(v["overhead_int"]["median"], 0.05)]
    ci_ok = [k for k, v in g.items() if le(v["overhead_int"]["ci"][1], 0.10)]
    verdict = {
        "n_graph_cells": len(g),
        "L1": {"holds": len(l1_cells) >= 1, "cells": l1_cells,
               "min_ratio": min(v["ratios"]["int-det/fp32"] for v in g.values())},
        "L2": {"holds": len(l2_cells) >= 1, "cells": l2_cells,
               "min_ratio": min(v["ratios"]["int-det/minfast"] for v in g.values())},
        "L3": {"holds": len(med_ok) == len(g) == 12 and len(ci_ok) >= 10, "cells_median_le_5pct": len(med_ok),
               "cells_ci_upper_le_10pct": len(ci_ok),
               "max_median": max(v["overhead_int"]["median"] for v in g.values()),
               "max_ci_upper": max(v["overhead_int"]["ci"][1] for v in g.values())}}
    cross = {}
    for mode in ("graph", "eager"):
        for size in SIZES:
            vs = [cells.get(f"{mode}|{size}|{N}") for N in NS]
            f1 = [N for N, v in zip(NS, vs) if v and v["ratios"]["int-det/fp32"] < 1.0]
            f2 = [N for N, v in zip(NS, vs) if v and v["ratios"]["int-det/minfast"] < 1.0]
            cross[f"{mode}|{size}"] = {"vs_fp32": f1[0] if f1 else None, "vs_minfast": f2[0] if f2 else None}
    eager_vs_graph = {}
    for size in SIZES:
        for N in NS:
            ge, ee = cells.get(f"graph|{size}|{N}"), cells.get(f"eager|{size}|{N}")
            if ge and ee:
                eager_vs_graph[f"{size}|{N}"] = {
                    "int-det/fp32": [ge["ratios"]["int-det/fp32"], ee["ratios"]["int-det/fp32"]],
                    "int-det/minfast": [ge["ratios"]["int-det/minfast"], ee["ratios"]["int-det/minfast"]],
                    "eager_over_graph_fp32": ee["T_ms"]["fp32"] / ge["T_ms"]["fp32"],
                    "eager_over_graph_int_det": ee["T_ms"]["int-det"] / ge["T_ms"]["int-det"]}
    s512 = cells.get("step|5|512")
    e7rep = {"reference": E7_A5000}
    if s512:
        e7rep.update({"new_int_det_over_fp32": s512["ratios"]["int-det/fp32"],
                      "e7_det_over_fp32": s512["ratios"].get("e7-det/fp32"),
                      "e7_det_over_new_int_det": s512["ratios"].get("e7-det/int-det")})
    e7_vs_new = {k: {"e7-det/int-det": v["ratios"].get("e7-det/int-det"), "e7-det/fp32": v["ratios"].get("e7-det/fp32"),
                     "int-det/fp32": v["ratios"]["int-det/fp32"]} for k, v in cells.items() if v["size"] == 5}

    # measurement state
    def f(b, part, k):
        v = (b.get(part) or {}).get(k)
        return None if v is None else float(v)

    state = {"n_blocks": len(blocks), "rounds": len(rounds)}
    for k in ("sm_mhz", "temp_c", "power_w"):
        vals = [f(b, p, k) for b in blocks for p in ("gpu_start", "gpu_mid", "gpu_end") if f(b, p, k) is not None]
        if vals:
            state[k] = [min(vals), float(np.median(vals)), max(vals)]
    thr = [int((b.get(p) or {}).get("throttle", 0)) for b in blocks for p in ("gpu_start", "gpu_mid", "gpu_end")]
    state["clock_event_reasons_seen"] = sorted({hex(t) for t in thr})
    state["share_readings_with_power_cap_reason"] = float(np.mean([(t & 0x4) != 0 for t in thr])) if thr else None
    state["share_readings_with_thermal_reason"] = float(np.mean([(t & (0x20 | 0x40)) != 0 for t in thr])) if thr else None
    state["load1"] = [min(b["load_start"][0] for b in blocks), float(np.median([b["load_start"][0] for b in blocks])),
                      max(b["load_start"][0] for b in blocks)]
    state["cpu_busy"] = [min(b["cpu_busy"] for b in blocks), float(np.median([b["cpu_busy"] for b in blocks])),
                         max(b["cpu_busy"] for b in blocks)]
    slow = [b["median_ms"] / T[b["key"]] for b in blocks]
    state["share_blocks_slower_15pct"] = float(np.mean(np.array(slow) > 1.15))
    state["share_blocks_faster_15pct"] = float(np.mean(np.array(slow) < 1 / 1.15))
    for mode in ("graph", "eager", "step"):
        sm = [s for s, b in zip(slow, blocks) if b["mode"] == mode]
        state[f"share_slower_15pct_{mode}"] = float(np.mean(np.array(sm) > 1.15)) if sm else None
    state["secs"] = M.get("secs")
    state["foreign_procs_max"] = max(int((b.get("gpu_end") or {}).get("procs", -1)) for b in blocks)

    out = {"label": "E8 analysis (DESIGN_ext 1.4-1.5)", "verdict": verdict, "cells": cells, "crossover": cross,
           "eager_vs_graph": eager_vs_graph, "e7_reproduction": e7rep, "e7_vs_new": e7_vs_new, "state": state,
           "selection": sel, "rounds_done": M.get("rounds_done"), "env": M.get("env")}
    json.dump(out, open(f"{a.outdir}/e8_summary.json", "w"), indent=1)

    # CSV of every cell
    with open(f"{a.outdir}/e8_cells.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mode", "size", "N", "impl", "T_fp32_ms", "T_tf32_ms", "T_bf16_ms", "T_int_det_ms",
                    "T_int_shiter_ms", "T_e7_det_ms", "T_e7_shiter_ms", "int_det_over_fp32", "int_det_over_minfast",
                    "minfast", "overhead_median", "overhead_ci_lo", "overhead_ci_hi", "e7_overhead_median",
                    "ratio_ci_rounds_lo", "ratio_ci_rounds_hi"])
        for c in cells.values():
            t, r = c["T_ms"], c["ratios"]
            o, o7 = c.get("overhead_int", {}), c.get("overhead_e7", {})
            w.writerow([c["mode"], c["size"], c["N"], c["impl"], t["fp32"], t["tf32"], t["bf16"], t["int-det"],
                        t["int-shiter"], t.get("e7-det", ""), t.get("e7-shiter", ""), r["int-det/fp32"],
                        r["int-det/minfast"], r["minfast_is"], o.get("median", ""), (o.get("ci") or ["", ""])[0],
                        (o.get("ci") or ["", ""])[1], o7.get("median", ""), c["ratio_ci_rounds"]["int-det/fp32"][0],
                        c["ratio_ci_rounds"]["int-det/fp32"][1]])

    # markdown tables
    L = ["# E8 결과 표 (자동 생성)", "", f"- 회전 {len(rounds)}, block {len(blocks)}, GEMM 경로 {sel}", "",
         "## graph-WM (주) / eager-WM / eager-step", "",
         "| 방식 | 크기 | N | fp32 ms | tf32 ms | bf16 ms | int-det ms | int-sh-iter ms | int-det/fp32 [회전 CI] | "
         "int-det/min(tf32,bf16) | 짝 오버헤드 [CI] |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells.values():
        t, r, o = c["T_ms"], c["ratios"], c["overhead_int"]
        ci = c["ratio_ci_rounds"]["int-det/fp32"]
        L.append(f"| {c['mode']} | {c['size']}M | {c['N']} | {t['fp32']:.3f} | {t['tf32']:.3f} | {t['bf16']:.3f} | "
                 f"{t['int-det']:.3f} | {t['int-shiter']:.3f} | {r['int-det/fp32']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | "
                 f"{r['int-det/minfast']:.3f} ({r['minfast_is']}) | {100 * o['median']:+.2f}% "
                 f"[{100 * o['ci'][0]:+.2f}, {100 * o['ci'][1]:+.2f}] |")
    L += ["", "## E7 융합 kernel (5M, 기술)", "", "| 방식 | N | e7-det ms | e7-sh-iter ms | e7-det/int-det | e7-det/fp32 | "
          "e7 짝 오버헤드 [CI] |", "|---|---|---|---|---|---|---|"]
    for c in cells.values():
        if "e7-det" in c["T_ms"]:
            o = c["overhead_e7"]
            L.append(f"| {c['mode']} | {c['N']} | {c['T_ms']['e7-det']:.3f} | {c['T_ms']['e7-shiter']:.3f} | "
                     f"{c['ratios']['e7-det/int-det']:.3f} | {c['ratios']['e7-det/fp32']:.3f} | "
                     f"{100 * o['median']:+.2f}% [{100 * o['ci'][0]:+.2f}, {100 * o['ci'][1]:+.2f}] |")
    L += ["", "## 판정", "", "```", json.dumps(verdict, indent=1), "```", "", "## 교차점", "", "```",
          json.dumps(cross, indent=1), "```", "", "## 측정 상태", "", "```", json.dumps(state, indent=1), "```"]
    open(f"{a.outdir}/e8_tables.md", "w").write("\n".join(L) + "\n")
    plots(cells, sel, a.outdir)
    print(json.dumps({"verdict": verdict, "crossover": cross, "e7_reproduction": e7rep}, indent=1))


def plots(cells, sel, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    series = [("tf32", "TF32", "#2a78d6", "o"), ("bf16", "BF16 (autocast)", "#eb6834", "s"),
              ("int-det", "int W8A4 det", "#1baf7a", "D"), ("int-shiter", "int W8A4 sh-iter", "#6250d6", "^"),
              ("e7-det", "E7 fused det (5M)", "#e34948", "v")]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2,
                         "ytick.color": ink2, "axes.spines.top": False, "axes.spines.right": False})
    for mode, title in (("graph", "graph-WM (CUDA graph replay)"), ("eager", "eager-WM")):
        fig, axs = plt.subplots(1, 4, figsize=(11, 3.0), sharey=True)
        for ax, size in zip(axs, SIZES):
            vs = [cells.get(f"{mode}|{size}|{N}") for N in NS]
            if not all(vs):
                continue
            ax.axhline(1.0, color=ink2, lw=1.0, ls="--", zorder=1)
            ax.axhline(0.8, color=grid, lw=1.0, ls=":", zorder=1)
            for p, lab, col, mk in series:
                if p not in vs[0]["T_ms"]:
                    continue
                y = [v["T_ms"][p] / v["T_ms"]["fp32"] for v in vs]
                ax.plot(NS, y, color=col, lw=2, marker=mk, ms=6, label=lab, zorder=3,
                        markeredgecolor="white", markeredgewidth=1.0)
            ax.set_xscale("log", base=2)
            ax.set_xticks(NS)
            ax.set_xticklabels([str(n) for n in NS])
            ax.set_title(f"{size}M (GEMM {sel[str(size)]})", color=ink, fontsize=9)
            ax.set_xlabel("candidates N")
            ax.grid(axis="y", color=grid, lw=0.6)
        axs[0].set_ylabel("latency / fp32")
        h, lb = [], []
        for ax in axs:
            for hh, ll in zip(*ax.get_legend_handles_labels()):
                if ll not in lb:
                    h.append(hh)
                    lb.append(ll)
        fig.legend(h, lb, loc="upper center", ncol=len(lb), frameon=False, bbox_to_anchor=(0.5, 1.04))
        fig.suptitle(f"E8 {title}: latency relative to fp32 (A40); dashed 1.0, dotted 0.8", y=1.12, fontsize=10,
                     color=ink)
        fig.savefig(f"{outdir}/e8_ratio_{mode}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    # pair overhead with CI (graph and eager)
    fig, ax = plt.subplots(figsize=(9, 3.0))
    labels, x = [], 0
    for mode, col in (("graph", "#1baf7a"), ("eager", "#2a78d6")):
        for size in SIZES:
            for N in NS:
                c = cells.get(f"{mode}|{size}|{N}")
                if not c:
                    continue
                o = c["overhead_int"]
                ax.errorbar([x], [100 * o["median"]], yerr=[[100 * (o["median"] - o["ci"][0])],
                                                           [100 * (o["ci"][1] - o["median"])]],
                            fmt="o", color=col, ms=5, capsize=2, lw=1.2,
                            label=f"{mode}-WM" if (size, N) == (5, 512) else None)
                labels.append(f"{size}M\n{N}")
                x += 1
        x += 1
        labels.append("")
    ax.axhline(5, color="#e34948", lw=1, ls=":")
    ax.axhline(10, color="#e34948", lw=1, ls="--")
    ax.axhline(0, color=ink2, lw=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("sh-iter overhead vs det (%)")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", color=grid, lw=0.6)
    ax.set_title("E8 paired dither overhead: median of 10 pairs, bootstrap 95% CI (dotted 5%, dashed 10%)",
                 fontsize=9, color=ink)
    fig.savefig(f"{outdir}/e8_overhead.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # absolute graph-WM latency
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.0))
    allser = [("fp32", "FP32", ink2, "x")] + series
    for ax, size in zip(axs, SIZES):
        vs = [cells.get(f"graph|{size}|{N}") for N in NS]
        if not all(vs):
            continue
        for p, lab, col, mk in allser:
            if p not in vs[0]["T_ms"]:
                continue
            ax.plot(NS, [v["T_ms"][p] for v in vs], color=col, lw=2, marker=mk, ms=6, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(NS)
        ax.set_xticklabels([str(n) for n in NS])
        ax.set_title(f"{size}M", fontsize=9, color=ink)
        ax.set_xlabel("candidates N")
        ax.grid(color=grid, lw=0.6)
    axs[0].set_ylabel("graph-WM latency (ms)")
    h, lb = axs[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="upper center", ncol=len(lb), frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.savefig(f"{outdir}/e8_latency_graph.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
