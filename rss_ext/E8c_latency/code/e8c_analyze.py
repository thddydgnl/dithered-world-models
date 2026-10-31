"""E8c analysis (rss_ext/DESIGN_E8c.md 4-5), written before the main measurement.

Pre-registered computations (DESIGN_E8c 4-5):
  T(condition) = median of its block values (10 rounds; block value = median per-call time); ratio = ratio of T
  pair overhead of a cell = median over its 10 pairs of (shiter block - det block) / det block; 95% CI = percentile
  bootstrap of the median, 10,000 resamples of the pairs, numpy RandomState(0) per cell (E8)
  min(float) = min(T(tf32), T(bf16), T(bf16cast)) of the same cell and mode; threshold comparisons with tolerance 1e-9
  L2c (graph-WM):               T(int-det) / min(float) <= 0.90 in at least one of the 12 cells
  S1  (graph-step, rest-fp32):  T(int-det) / T(fp32) <= 0.80 in at least one of the 12 cells
  S2  (graph-step, rest-fp32):  T(int-det) / min(float) <= 0.90 in at least one cell
  S3  (graph-step, rest-fp32):  median pair overhead <= 5% in all 12 cells and CI upper bound <= 10% in >= 10 cells
Descriptive (DESIGN_E8c 5; defined before the main measurement in PLAN.md):
  same-pod replication of E8's L1-L3 on graph-WM (L1: T(int-det)/T(fp32) <= 0.80; L2: T(int-det)/min(T(tf32),
    T(bf16)) <= 0.90; L3: the overhead rule) and, with --e8, E8's values of the same conditions (other pod)
  per cell and mode T(bf16cast)/T(bf16), T(int-det)/T(bf16cast); graph-WM T(int-det-L0tf32)/T(int-det)
  graph-step vs eager-step (N 512, rest-fp32): T ratio per path, and int-det/fp32, int-det/min(float) in both modes
  WM share of the planning step: T(graph-WM, path) / T(graph-step, path) for the same cell (rest-fp32 and rest-tf32)
  rest-tf32 graph-step: T(int-det)/min(float), T(int-det)/T(tf32), pair overhead [CI]
  crossover per size: smallest N with T(int-det) < T(fp32) and with T(int-det) < min(float) (graph-step rest-fp32;
    also graph-WM and graph-step rest-tf32 (min(float) only))
  round-bootstrap 95% CI of the main ratios (10,000 resamples of the rounds, all conditions jointly, RandomState(0))
  measurement state: SM clock, temperature, power, clock-event reasons, host load, CPU busy, share of blocks more than
    15% slower than their condition's T, per mode
Figures (results/analysis): e8c_ratio_graph.png, e8c_ratio_gstep.png, e8c_overhead.png, e8c_wm_share.png,
  e8c_latency_gstep.png
usage: python e8c_analyze.py --main <dir with e8c_main.json> --outdir <dir> [--e8 <E8 e8_summary.json>]
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

SIZES = (5, 19, 48, 317)
NS = (512, 2048, 8192)
STEP_N = 512
NB = 10000
EPS = 1e-9
MODES = (("graph", "-"), ("gstep", "fp32"), ("gstep", "tf32"), ("step", "fp32"))
PRECS = {("graph", "-"): ("fp32", "tf32", "bf16", "bf16cast", "int-det-L0tf32", "int-det", "int-shiter"),
         ("gstep", "fp32"): ("fp32", "tf32", "bf16", "bf16cast", "int-det", "int-shiter"),
         ("gstep", "tf32"): ("tf32", "bf16", "bf16cast", "int-det", "int-shiter"),
         ("step", "fp32"): ("fp32", "tf32", "bf16", "bf16cast", "int-det", "int-shiter")}
FLOATS_MIN = ("tf32", "bf16", "bf16cast")
MODE_LABEL = {"graph|-": "graph-WM", "gstep|fp32": "graph-step rest-fp32", "gstep|tf32": "graph-step rest-tf32",
              "step|fp32": "eager-step"}


def le(x, thr):
    return x <= thr + EPS


def boot_median_ci(x):
    x = np.asarray(x, float)
    idx = np.random.RandomState(0).randint(0, len(x), size=(NB, len(x)))
    m = np.median(x[idx], axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def key(mode, rest, size, N, prec, impl):
    return f"{mode}|{rest}|{size}|{N}|{prec}|{impl}"


def rng_of(vals):
    vals = [v for v in vals if v is not None]
    return [min(vals), max(vals)] if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--e8", default=None, help="E8 results/analysis/e8_summary.json (other pod), for side-by-side")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    M = json.load(open(f"{a.main}/e8c_main.json"))
    sel = M["settings"]["selection"]
    blocks, pairs = M["blocks"], M["pairs"]
    rounds = sorted({b["round"] for b in blocks})
    byk = defaultdict(dict)                       # key -> round -> block value
    for b in blocks:
        byk[b["key"]][b["round"]] = b["median_ms"]
    T = {k: float(np.median(list(v.values()))) for k, v in byk.items()}
    nblk = {k: len(v) for k, v in byk.items()}

    def cond_keys(mode, rest, size, N):
        impl = sel[str(size)]
        d = {p: key(mode, rest, size, N, p, impl if p.startswith("int") else "-") for p in PRECS[(mode, rest)]}
        return {p: k for p, k in d.items() if k in T}

    def ratios_of(t):
        r = {}
        if "fp32" in t:
            for p in t:
                if p != "fp32":
                    r[f"{p}/fp32"] = t[p] / t["fp32"]
        mf = min(FLOATS_MIN, key=lambda p: t[p])
        r["minfloat_is"] = mf
        r["int-det/minfloat"] = t["int-det"] / t[mf]
        r["int-shiter/minfloat"] = t["int-shiter"] / t[mf]
        fast = "tf32" if t["tf32"] <= t["bf16"] else "bf16"
        r["minfast_is"] = fast
        r["int-det/minfast"] = t["int-det"] / t[fast]
        r["int-det/tf32"] = t["int-det"] / t["tf32"]
        r["bf16cast/bf16"] = t["bf16cast"] / t["bf16"]
        r["int-det/bf16cast"] = t["int-det"] / t["bf16cast"]
        if "int-det-L0tf32" in t:
            r["int-det-L0tf32/int-det"] = t["int-det-L0tf32"] / t["int-det"]
        return r

    # round bootstrap of T (descriptive): resample rounds jointly for all conditions
    rs = np.random.RandomState(0)
    ridx = rs.randint(0, len(rounds), size=(NB, len(rounds)))
    rarr = {k: np.array([v.get(r, np.nan) for r in rounds]) for k, v in byk.items()}

    def boot_T(k):
        return np.nanmedian(rarr[k][ridx], axis=1)

    def pq(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    cells = {}
    for (mode, rest) in MODES:
        for size in SIZES:
            for N in (NS if mode != "step" else (STEP_N,)):
                ck = cond_keys(mode, rest, size, N)
                if not set(PRECS[(mode, rest)]) <= set(ck):
                    continue
                t = {p: T[k] for p, k in ck.items()}
                c = {"mode": mode, "rest": rest, "size": size, "N": N, "impl": sel[str(size)], "T_ms": t,
                     "n_blocks": {p: nblk[k] for p, k in ck.items()}, "ratios": ratios_of(t)}
                bt = {p: boot_T(k) for p, k in ck.items()}
                minf = np.minimum(np.minimum(bt["tf32"], bt["bf16"]), bt["bf16cast"])
                ci = {"int-det/minfloat": pq(bt["int-det"] / minf),
                      "int-det/minfast": pq(bt["int-det"] / np.minimum(bt["tf32"], bt["bf16"])),
                      "bf16cast/bf16": pq(bt["bf16cast"] / bt["bf16"])}
                if "fp32" in bt:
                    ci["int-det/fp32"] = pq(bt["int-det"] / bt["fp32"])
                c["ratio_ci_rounds"] = ci
                ov = [p["overhead"] for p in pairs if p["mode"] == mode and p["rest"] == rest and p["size"] == size
                      and p["N"] == N and p["prec"] == "int"]
                c["overhead"] = {"pairs": ov, "median": float(np.median(ov)),
                                 "ci": boot_median_ci(ov) if len(ov) > 1 else [float("nan")] * 2, "n": len(ov),
                                 "pooled": (t["int-shiter"] - t["int-det"]) / t["int-det"],
                                 "abs_ms_median": float(np.median([p["shiter_ms"] - p["det_ms"] for p in pairs
                                                                   if p["mode"] == mode and p["rest"] == rest
                                                                   and p["size"] == size and p["N"] == N]))}
                cells[f"{mode}|{rest}|{size}|{N}"] = c

    def sel_cells(mode, rest):
        return {k: v for k, v in cells.items() if v["mode"] == mode and v["rest"] == rest}

    def count_rule(cs, ratio, thr):
        ok = [k for k, v in cs.items() if le(v["ratios"][ratio], thr)]
        vals = [v["ratios"][ratio] for v in cs.values()]
        return {"holds": len(ok) >= 1, "n_cells": len(cs), "n_pass": len(ok), "cells_pass": ok,
                "range": rng_of(vals), "min_cell": min(cs, key=lambda k: cs[k]["ratios"][ratio]) if cs else None}

    def overhead_rule(cs):
        med_ok = [k for k, v in cs.items() if le(v["overhead"]["median"], 0.05)]
        ci_ok = [k for k, v in cs.items() if le(v["overhead"]["ci"][1], 0.10)]
        return {"holds": len(med_ok) == len(cs) == 12 and len(ci_ok) >= 10, "n_cells": len(cs),
                "cells_median_le_5pct": len(med_ok), "cells_ci_upper_le_10pct": len(ci_ok),
                "median_range": rng_of([v["overhead"]["median"] for v in cs.values()]),
                "max_ci_upper": max(v["overhead"]["ci"][1] for v in cs.values()) if cs else None}

    g, s, s_tf, st = sel_cells("graph", "-"), sel_cells("gstep", "fp32"), sel_cells("gstep", "tf32"), \
        sel_cells("step", "fp32")
    verdict = {"L2c": count_rule(g, "int-det/minfloat", 0.90), "S1": count_rule(s, "int-det/fp32", 0.80),
               "S2": count_rule(s, "int-det/minfloat", 0.90), "S3": overhead_rule(s)}
    for k in ("L2c", "S1", "S2"):
        verdict[k]["holds"] = verdict[k]["holds"] and verdict[k]["n_cells"] == 12
    e8rep = {"L1": count_rule(g, "int-det/fp32", 0.80), "L2": count_rule(g, "int-det/minfast", 0.90),
             "L3": overhead_rule(g)}
    rest_tf32 = {"int-det/minfloat": count_rule(s_tf, "int-det/minfloat", 0.90),
                 "int-det/tf32_range": rng_of([v["ratios"]["int-det/tf32"] for v in s_tf.values()]),
                 "overhead": overhead_rule(s_tf)}

    # descriptive tables
    bf16cast = {k: {"bf16cast/bf16": v["ratios"]["bf16cast/bf16"], "int-det/bf16cast": v["ratios"]["int-det/bf16cast"],
                    "minfloat_is": v["ratios"]["minfloat_is"]} for k, v in cells.items()}
    l0 = {k: v["ratios"].get("int-det-L0tf32/int-det") for k, v in g.items()}
    g_vs_e = {}
    for size in SIZES:
        cg, ce = cells.get(f"gstep|fp32|{size}|{STEP_N}"), cells.get(f"step|fp32|{size}|{STEP_N}")
        if cg and ce:
            g_vs_e[str(size)] = {"gstep_over_step": {p: cg["T_ms"][p] / ce["T_ms"][p] for p in PRECS[("step", "fp32")]},
                                 "int-det/fp32": [cg["ratios"]["int-det/fp32"], ce["ratios"]["int-det/fp32"]],
                                 "int-det/minfloat": [cg["ratios"]["int-det/minfloat"],
                                                      ce["ratios"]["int-det/minfloat"]],
                                 "overhead_median": [cg["overhead"]["median"], ce["overhead"]["median"]]}
    share = {}
    for size in SIZES:
        for N in NS:
            cg = cells.get(f"graph|-|{size}|{N}")
            for rest in ("fp32", "tf32"):
                cs_ = cells.get(f"gstep|{rest}|{size}|{N}")
                if cg and cs_:
                    share[f"{rest}|{size}|{N}"] = {p: cg["T_ms"][p] / cs_["T_ms"][p] for p in cs_["T_ms"]}
    cross = {}
    for (mode, rest) in (("graph", "-"), ("gstep", "fp32"), ("gstep", "tf32")):
        for size in SIZES:
            vs = [cells.get(f"{mode}|{rest}|{size}|{N}") for N in NS]
            f1 = [N for N, v in zip(NS, vs) if v and "int-det/fp32" in v["ratios"] and v["ratios"]["int-det/fp32"] < 1]
            f2 = [N for N, v in zip(NS, vs) if v and v["ratios"]["int-det/minfloat"] < 1]
            cross[f"{mode}|{rest}|{size}"] = {"vs_fp32": (f1[0] if f1 else None) if mode != "gstep" or rest == "fp32"
                                              else "n/a", "vs_minfloat": f2[0] if f2 else None}
    e8cmp = None
    if a.e8 and os.path.exists(a.e8):
        E8 = json.load(open(a.e8))
        e8cmp = {}
        combos = [("graph", "graph", "-", N) for N in NS] + [("step", "step", "fp32", STEP_N)]
        for size in SIZES:
            for mode8, mode, rest, N in combos:
                c8, cc = E8["cells"].get(f"{mode8}|{size}|{N}"), cells.get(f"{mode}|{rest}|{size}|{N}")
                if not (c8 and cc):
                    continue
                t8, tc = c8["T_ms"], cc["T_ms"]
                e8cmp[f"{mode8}|{size}|{N}"] = {
                    "T_e8c_over_e8": {p: tc[p] / t8[p] for p in ("fp32", "tf32", "bf16", "int-det", "int-shiter")},
                    "int-det/fp32": [c8["ratios"]["int-det/fp32"], cc["ratios"]["int-det/fp32"]],
                    "int-det/minfast": [c8["ratios"]["int-det/minfast"], cc["ratios"]["int-det/minfast"]],
                    "overhead_median": [c8["overhead_int"]["median"], cc["overhead"]["median"]]}

    # measurement state
    def fval(b, part, k):
        v = (b.get(part) or {}).get(k)
        return None if v is None else float(v)

    state = {"n_blocks": len(blocks), "rounds": len(rounds), "n_pairs": len(pairs)}
    for k in ("sm_mhz", "temp_c", "power_w"):
        vals = [fval(b, p, k) for b in blocks for p in ("gpu_start", "gpu_mid", "gpu_end") if fval(b, p, k) is not None]
        if vals:
            state[k] = [min(vals), float(np.median(vals)), max(vals)]
    thr = [int((b.get(p) or {}).get("throttle", 0)) for b in blocks for p in ("gpu_start", "gpu_mid", "gpu_end")]
    state["clock_event_reasons_seen"] = sorted({hex(t) for t in thr})
    state["share_readings_with_power_cap_reason"] = float(np.mean([(t & 0x4) != 0 for t in thr])) if thr else None
    state["share_readings_with_thermal_reason"] = float(np.mean([(t & (0x20 | 0x40)) != 0 for t in thr])) if thr \
        else None
    state["load1"] = [min(b["load_start"][0] for b in blocks), float(np.median([b["load_start"][0] for b in blocks])),
                      max(b["load_start"][0] for b in blocks)]
    state["cpu_busy"] = [min(b["cpu_busy"] for b in blocks), float(np.median([b["cpu_busy"] for b in blocks])),
                         max(b["cpu_busy"] for b in blocks)]
    slow = [b["median_ms"] / T[b["key"]] for b in blocks]
    state["share_blocks_slower_15pct"] = float(np.mean(np.array(slow) > 1.15))
    state["share_blocks_faster_15pct"] = float(np.mean(np.array(slow) < 1 / 1.15))
    for (mode, rest) in MODES:
        sm = [x for x, b in zip(slow, blocks) if b["mode"] == mode and b["rest"] == rest]
        state[f"share_slower_15pct_{mode}|{rest}"] = float(np.mean(np.array(sm) > 1.15)) if sm else None
        state[f"max_block_over_T_{mode}|{rest}"] = float(max(sm)) if sm else None
    state["secs"] = M.get("secs")
    state["foreign_procs_max"] = max(int((b.get("gpu_end") or {}).get("procs", -1)) for b in blocks)

    out = {"label": "E8c analysis (DESIGN_E8c 4-5)", "verdict": verdict, "e8_replication_same_pod": e8rep,
           "rest_tf32_gstep": rest_tf32, "cells": cells, "bf16cast": bf16cast, "l0tf32_over_int_det": l0,
           "gstep_vs_eager_step": g_vs_e, "wm_share": share, "crossover": cross, "e8_vs_e8c": e8cmp, "state": state,
           "selection": sel, "rounds_done": M.get("rounds_done"), "env": M.get("env"),
           "settings": {k: v for k, v in M["settings"].items() if k not in ("n_reps", "gemm_cfgs")}}
    json.dump(out, open(f"{a.outdir}/e8c_summary.json", "w"), indent=1)

    # CSV of every cell
    allp = ("fp32", "tf32", "bf16", "bf16cast", "int-det-L0tf32", "int-det", "int-shiter")
    with open(f"{a.outdir}/e8c_cells.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mode", "rest", "size", "N", "impl"] + [f"T_{p}_ms" for p in allp]
                   + ["int_det_over_fp32", "int_det_over_minfloat", "minfloat_is", "int_det_over_minfast",
                      "bf16cast_over_bf16", "int_det_over_bf16cast", "l0tf32_over_int_det", "overhead_median",
                      "overhead_ci_lo", "overhead_ci_hi", "overhead_abs_ms_median"])
        for c in cells.values():
            t, r, o = c["T_ms"], c["ratios"], c["overhead"]
            w.writerow([c["mode"], c["rest"], c["size"], c["N"], c["impl"]] + [t.get(p, "") for p in allp]
                       + [r.get("int-det/fp32", ""), r["int-det/minfloat"], r["minfloat_is"], r["int-det/minfast"],
                          r["bf16cast/bf16"], r["int-det/bf16cast"], r.get("int-det-L0tf32/int-det", ""), o["median"],
                          o["ci"][0], o["ci"][1], o["abs_ms_median"]])

    # markdown tables
    def f3(x):
        return "" if x is None else f"{x:.3f}"

    def pct(o):
        return f"{100 * o['median']:+.2f}% [{100 * o['ci'][0]:+.2f}, {100 * o['ci'][1]:+.2f}]"

    L = ["# E8c 결과 표 (자동 생성, `e8c_analyze.py`)", "",
         f"- 회전 {len(rounds)}, block {len(blocks)}, 짝 {len(pairs)}, GEMM 경로 {sel}",
         "- T: 조건 block 값(block 안 회당 시간의 중앙값)의 중앙값, ms. min(float) = min(tf32, bf16, bf16cast). "
         "min(fast) = min(tf32, bf16)(E8 정의). 짝 오버헤드: 중앙값 [짝 bootstrap 95% CI].", "",
         "## 판정 [사전등록]", "", "| 가설 | 판정 | 통과 칸 | 비율 범위 (최소 칸) |", "|---|---|---|---|"]
    for name, rule in (("L2c graph-WM det/min(float) ≤ 0.90", verdict["L2c"]),
                       ("S1 graph-step rest-fp32 det/fp32 ≤ 0.80", verdict["S1"]),
                       ("S2 graph-step rest-fp32 det/min(float) ≤ 0.90", verdict["S2"])):
        L.append(f"| {name} | {'성립' if rule['holds'] else '불성립'} | {rule['n_pass']}/{rule['n_cells']} | "
                 f"{f3(rule['range'][0])} ~ {f3(rule['range'][1])} ({rule['min_cell']}) |")
    s3 = verdict["S3"]
    L.append(f"| S3 graph-step rest-fp32 짝 오버헤드 | {'성립' if s3['holds'] else '불성립'} | 중앙값 ≤ 5%: "
             f"{s3['cells_median_le_5pct']}/{s3['n_cells']}, CI 상한 ≤ 10%: {s3['cells_ci_upper_le_10pct']} | "
             f"중앙값 {100 * s3['median_range'][0]:+.2f}% ~ {100 * s3['median_range'][1]:+.2f}%, CI 상한 최대 "
             f"{100 * s3['max_ci_upper']:+.2f}% |")
    L += ["", "## E8 L1~L3 같은 pod 재계산 (graph-WM) [기술]", "", "| 가설 | 판정 | 통과 칸 | 비율 범위 |",
          "|---|---|---|---|"]
    for name in ("L1", "L2"):
        rule = e8rep[name]
        L.append(f"| {name} | {'성립' if rule['holds'] else '불성립'} | {rule['n_pass']}/{rule['n_cells']} | "
                 f"{f3(rule['range'][0])} ~ {f3(rule['range'][1])} |")
    r3 = e8rep["L3"]
    L.append(f"| L3 | {'성립' if r3['holds'] else '불성립'} | 중앙값 ≤ 5%: {r3['cells_median_le_5pct']}, CI 상한 ≤ 10%: "
             f"{r3['cells_ci_upper_le_10pct']} | 중앙값 {100 * r3['median_range'][0]:+.2f}% ~ "
             f"{100 * r3['median_range'][1]:+.2f}%, CI 상한 최대 {100 * r3['max_ci_upper']:+.2f}% |")
    hdr = {("graph", "-"): ["fp32", "tf32", "bf16", "bf16cast", "int-det-L0tf32", "int-det", "int-shiter"],
           ("gstep", "fp32"): ["fp32", "tf32", "bf16", "bf16cast", "int-det", "int-shiter"],
           ("gstep", "tf32"): ["tf32", "bf16", "bf16cast", "int-det", "int-shiter"],
           ("step", "fp32"): ["fp32", "tf32", "bf16", "bf16cast", "int-det", "int-shiter"]}
    for (mode, rest) in MODES:
        ps = hdr[(mode, rest)]
        extra = ["det/fp32"] if "fp32" in ps else ["det/tf32"]
        extra += ["det/min(float)", "det/min(fast)", "bf16cast/bf16", "det/bf16cast"]
        if mode == "graph":
            extra += ["L0tf32/det"]
        L += ["", f"## {MODE_LABEL[f'{mode}|{rest}']}", "",
              "| 크기 | N | " + " | ".join(f"{p} ms" for p in ps) + " | " + " | ".join(extra) + " | 짝 오버헤드 [CI] |",
              "|" + "---|" * (2 + len(ps) + len(extra) + 1)]
        for c in sel_cells(mode, rest).values():
            t, r = c["T_ms"], c["ratios"]
            vals = [f3(r.get("int-det/fp32")) if "fp32" in ps else f3(r["int-det/tf32"]),
                    f"{f3(r['int-det/minfloat'])} ({r['minfloat_is']})", f"{f3(r['int-det/minfast'])}",
                    f3(r["bf16cast/bf16"]), f3(r["int-det/bf16cast"])]
            if mode == "graph":
                vals.append(f3(r.get("int-det-L0tf32/int-det")))
            L.append(f"| {c['size']}M | {c['N']} | " + " | ".join(f3(t[p]) for p in ps) + " | " + " | ".join(vals)
                     + f" | {pct(c['overhead'])} |")
    L += ["", "## graph-step 대 eager-step (N 512, rest-fp32) [기술]", "",
          "| 크기 | graph-step/eager-step: fp32 | tf32 | bf16 | bf16cast | int-det | int-shiter | det/fp32 (graph-step → "
          "eager-step) | det/min(float) (graph-step → eager-step) |", "|---|---|---|---|---|---|---|---|---|"]
    for size, v in g_vs_e.items():
        q = v["gstep_over_step"]
        L.append(f"| {size}M | " + " | ".join(f3(q[p]) for p in ("fp32", "tf32", "bf16", "bf16cast", "int-det",
                                                                 "int-shiter"))
                 + f" | {f3(v['int-det/fp32'][0])} → {f3(v['int-det/fp32'][1])} | {f3(v['int-det/minfloat'][0])} → "
                   f"{f3(v['int-det/minfloat'][1])} |")
    L += ["", "## 계획 step에서 WM rollout의 몫: T(graph-WM) / T(graph-step), 같은 경로 [기술]", "",
          "| rest | 크기 | N | fp32 | tf32 | bf16 | bf16cast | int-det | int-shiter |", "|---|---|---|---|---|---|---|---|---|"]
    for k, v in share.items():
        rest, size, N = k.split("|")
        L.append(f"| {rest} | {size}M | {N} | " + " | ".join(f3(v.get(p)) for p in ("fp32", "tf32", "bf16", "bf16cast",
                                                                                   "int-det", "int-shiter")) + " |")
    if e8cmp:
        L += ["", "## E8 대 E8c 같은 조건 (다른 pod) [기술]", "",
              "| 방식 | 크기 | N | E8c/E8 시간비 fp32 · tf32 · bf16 · int-det · int-shiter | det/fp32 E8 → E8c | "
              "det/min(fast) E8 → E8c | 짝 오버헤드 E8 → E8c |", "|---|---|---|---|---|---|---|"]
        for k, v in e8cmp.items():
            mode8, size, N = k.split("|")
            q = v["T_e8c_over_e8"]
            L.append(f"| {mode8} | {size}M | {N} | " + " · ".join(f3(q[p]) for p in ("fp32", "tf32", "bf16", "int-det",
                                                                                     "int-shiter"))
                     + f" | {f3(v['int-det/fp32'][0])} → {f3(v['int-det/fp32'][1])} | {f3(v['int-det/minfast'][0])} → "
                       f"{f3(v['int-det/minfast'][1])} | {100 * v['overhead_median'][0]:+.2f}% → "
                       f"{100 * v['overhead_median'][1]:+.2f}% |")
    L += ["", "## rest-tf32 graph-step [기술]", "", "```", json.dumps(rest_tf32, indent=1), "```",
          "", "## 교차점 [기술]", "", "```", json.dumps(cross, indent=1), "```", "", "## 측정 상태", "", "```",
          json.dumps(state, indent=1), "```"]
    open(f"{a.outdir}/e8c_tables.md", "w").write("\n".join(L) + "\n")
    plots(cells, sel, share, a.outdir)
    print(json.dumps({"verdict": verdict, "e8_replication_same_pod": e8rep, "rest_tf32": rest_tf32,
                      "crossover": cross}, indent=1))


def plots(cells, sel, share, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    # categorical slots of the validated reference palette, in its fixed order (adjacent pairs pass in light mode)
    col = {"tf32": "#2a78d6", "bf16": "#eb6834", "int-det": "#1baf7a", "bf16cast": "#eda100",
           "int-shiter": "#e87ba4", "int-det-L0tf32": "#008300", "fp32": ink2}
    lab = {"fp32": "FP32", "tf32": "TF32", "bf16": "BF16 (autocast)", "bf16cast": "BF16 (pre-cast)",
           "int-det": "int W8A4 det", "int-shiter": "int W8A4 sh-iter", "int-det-L0tf32": "int det, layer-0 TF32"}
    mk = {"fp32": "x", "tf32": "o", "bf16": "s", "bf16cast": "P", "int-det": "D", "int-shiter": "^",
          "int-det-L0tf32": "v"}
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2,
                         "ytick.color": ink2, "axes.spines.top": False, "axes.spines.right": False})

    def legend(fig, axs, ncol=None):
        h, lb = [], []
        for ax in np.ravel(axs):
            for hh, ll in zip(*ax.get_legend_handles_labels()):
                if ll not in lb:
                    h.append(hh)
                    lb.append(ll)
        fig.legend(h, lb, loc="upper center", ncol=ncol or len(lb), frameon=False, bbox_to_anchor=(0.5, 1.06))

    def xaxis(ax):
        ax.set_xscale("log", base=2)
        ax.set_xticks(NS)
        ax.set_xticklabels([str(n) for n in NS])
        ax.set_xlabel("candidates N")
        ax.grid(axis="y", color=grid, lw=0.6)

    # graph-WM: latency / fp32
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.0), sharey=True)
    for ax, size in zip(axs, SIZES):
        vs = [cells.get(f"graph|-|{size}|{N}") for N in NS]
        if not all(vs):
            continue
        ax.axhline(1.0, color=ink2, lw=1.0, ls="--", zorder=1)
        for p in ("tf32", "bf16", "int-det", "bf16cast", "int-shiter", "int-det-L0tf32"):
            ax.plot(NS, [v["T_ms"][p] / v["T_ms"]["fp32"] for v in vs], color=col[p], lw=2, marker=mk[p], ms=6,
                    label=lab[p], zorder=3, markeredgecolor="white", markeredgewidth=0.8)
        xaxis(ax)
        ax.set_title(f"{size}M", color=ink, fontsize=9)
    axs[0].set_ylabel("graph-WM latency / FP32")
    legend(fig, axs)
    fig.suptitle("E8c graph-WM (world-model rollout, CUDA graph; A40): latency relative to FP32", y=1.14, fontsize=10,
                 color=ink)
    fig.savefig(f"{outdir}/e8c_ratio_graph.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # graph-step: latency / T(fp32, rest-fp32); top rest-fp32, bottom rest-tf32
    fig, axs = plt.subplots(2, 4, figsize=(11, 5.6), sharey=True)
    for row, rest in enumerate(("fp32", "tf32")):
        for ax, size in zip(axs[row], SIZES):
            vs = [cells.get(f"gstep|{rest}|{size}|{N}") for N in NS]
            base = [cells.get(f"gstep|fp32|{size}|{N}") for N in NS]
            if not (all(vs) and all(base)):
                continue
            ax.axhline(1.0, color=ink2, lw=1.0, ls="--", zorder=1)
            ax.axhline(0.8, color=grid, lw=1.0, ls=":", zorder=1)
            for p in ("tf32", "bf16", "int-det", "bf16cast", "int-shiter"):
                ax.plot(NS, [v["T_ms"][p] / b["T_ms"]["fp32"] for v, b in zip(vs, base)], color=col[p], lw=2,
                        marker=mk[p], ms=6, label=lab[p], zorder=3, markeredgecolor="white", markeredgewidth=0.8)
            xaxis(ax)
            ax.set_title(f"{size}M, rest {rest.upper()}", color=ink, fontsize=9)
        axs[row][0].set_ylabel("graph-step latency /\nFP32 (rest FP32)")
    legend(fig, axs)
    fig.suptitle("E8c graph-step (whole planning step, CUDA graph; A40): latency relative to all-FP32; dotted 0.8",
                 y=1.08, fontsize=10, color=ink)
    fig.tight_layout()
    fig.savefig(f"{outdir}/e8c_ratio_gstep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # pair overhead with CI
    fig, ax = plt.subplots(figsize=(11, 3.0))
    labels, x = [], 0
    mcol = {"graph|-": "#2a78d6", "gstep|fp32": "#eb6834", "gstep|tf32": "#1baf7a", "step|fp32": "#eda100"}
    for (mode, rest) in MODES:
        first = True
        for size in SIZES:
            for N in (NS if mode != "step" else (STEP_N,)):
                c = cells.get(f"{mode}|{rest}|{size}|{N}")
                if not c:
                    continue
                o = c["overhead"]
                ax.errorbar([x], [100 * o["median"]], yerr=[[100 * (o["median"] - o["ci"][0])],
                                                           [100 * (o["ci"][1] - o["median"])]],
                            fmt="o", color=mcol[f"{mode}|{rest}"], ms=5, capsize=2, lw=1.2,
                            label=MODE_LABEL[f"{mode}|{rest}"] if first else None)
                first = False
                labels.append(f"{size}M\n{N}")
                x += 1
        x += 1
        labels.append("")
    ax.axhline(5, color=ink2, lw=1, ls=":")
    ax.axhline(10, color=ink2, lw=1, ls="--")
    ax.axhline(0, color=ink2, lw=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("sh-iter overhead vs det (%)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)
    ax.grid(axis="y", color=grid, lw=0.6)
    ax.set_title("E8c paired dither overhead: median of 10 pairs, bootstrap 95% CI (dotted 5%, dashed 10%)",
                 fontsize=9, color=ink)
    fig.savefig(f"{outdir}/e8c_overhead.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # WM share of the planning step (rest-fp32)
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.0), sharey=True)
    for ax, size in zip(axs, SIZES):
        vs = [share.get(f"fp32|{size}|{N}") for N in NS]
        if not all(vs):
            continue
        for p in ("fp32", "int-det", "bf16cast"):
            ax.plot(NS, [v[p] for v in vs], color=col[p], lw=2, marker=mk[p], ms=6, label=lab[p], zorder=3,
                    markeredgecolor="white", markeredgewidth=0.8)
        xaxis(ax)
        ax.set_ylim(0, 1)
        ax.set_title(f"{size}M", color=ink, fontsize=9)
    axs[0].set_ylabel("T(graph-WM) / T(graph-step)")
    legend(fig, axs)
    fig.suptitle("E8c: world-model rollout share of the planning step (same WM path, rest FP32)", y=1.14, fontsize=10,
                 color=ink)
    fig.savefig(f"{outdir}/e8c_wm_share.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # absolute graph-step latency (rest-fp32)
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.0))
    for ax, size in zip(axs, SIZES):
        vs = [cells.get(f"gstep|fp32|{size}|{N}") for N in NS]
        if not all(vs):
            continue
        for p in ("fp32", "tf32", "bf16", "int-det", "bf16cast", "int-shiter"):
            ax.plot(NS, [v["T_ms"][p] for v in vs], color=col[p], lw=2, marker=mk[p], ms=6, label=lab[p])
        xaxis(ax)
        ax.set_yscale("log")
        ax.grid(color=grid, lw=0.6)
        ax.set_title(f"{size}M", fontsize=9, color=ink)
    axs[0].set_ylabel("graph-step latency (ms)")
    legend(fig, axs)
    fig.suptitle("E8c graph-step latency, rest FP32 (A40)", y=1.14, fontsize=10, color=ink)
    fig.savefig(f"{outdir}/e8c_latency_gstep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
