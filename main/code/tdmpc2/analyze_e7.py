"""E7 analysis of the TD-MPC2 family, DESIGN_main v5 7.4. Written before any E7 outcome existed; checked on synthetic
data only.

  - success with the real kernels (E7 seeds 70000-70029, 90 pairs, the same pairs in every configuration): mean
    success (Meta-World) / return (DMC) per (task, precision, iterations) with CIs, and D against fp32 at the same
    iteration count (paired)
  - latency (e7_latency.json, exclusive GPU, eager): median and IQR per condition; int4 has two kernels (codes in int8
    containers = the success-run kernel, and packed int4 = the v5 target); both are reported
  - H5 part 1: dither overhead (median latency, dither vs det, same task and iterations) <= 10% for int8 and int4:
    reported per condition (all <= 10%) and as the median over the 12 (task, iterations) conditions
  - H5 part 2: Pareto dominance of a "dither + fewer bits" configuration (int4 sh-iter dither, any iteration count) over
    a "det + more bits" configuration (int8 det, any iteration count) of the same task: latency <= and success >= with
    at least one strict (point estimates; the paired bootstrap CI of the success difference is given with each case)
usage (server): python analyze_e7.py --res ~/pmx/A/tdmpc2_main/out_e7 --latency ../v5/e7_latency.json \
                    --outdir ~/pmx/A/tdmpc2_main/analysis_e7

Changes of 2026-10-24 (E7 analysis worker, after the runs ended; the latency summary had been seen in the log, the
success outcomes not; rules unchanged, see worklogs/E7_tdmpc2.md):
  - the 10% comparisons use v5_common.le (the shared tolerance, coordinator instruction)
  - the H5 verdict is computed explicitly for every reading the pre-run note left open: int4 kernel (container =
    the success-run kernel / packed = the v5 target) x overhead reading (every condition <= 10% / median over the
    conditions <= 10%), each with the Pareto part (>= 1 case with the latency of that kernel)
  - descriptive additions: per-round overheads (from the per-round block medians), the latency of every condition
    relative to fp32, a robustness view of every Pareto case (the success-difference CI and whether the latency order
    holds in all 5 rounds), the same-iteration subset of the Pareto cases, family_points_e7_tdmpc2.json, a CSV
"""
import argparse
import json
import os

import numpy as np

import v5_common as C

PRECS = [("fp32", "none", 0), ("bf16", "none", 0), ("int8", "det", 8), ("int8", "shiter", 8), ("int4", "det", 4),
         ("int4", "shiter", 4)]
ITERS = [3, 6, 12]
THR = 0.10


def round_medians(L, key):
    return [b["median_ms"] for b in sorted(L["blocks"][key], key=lambda b: b["round"])] if key in L.get("blocks", {}) \
        else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--latency", default=None)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    E = C.load_episodes(a.res, prefix="e7_")
    Lj = json.load(open(a.latency)) if a.latency and os.path.exists(a.latency) else {}
    lat = Lj.get("summary", {})
    over = Lj.get("dither_overhead", {})
    tasks = sorted({k[1] for k in E})
    out = {"n_episodes": len(E), "success": {}, "latency": lat, "H5_overhead": {}, "H5_pareto": {},
           "overhead_per_round": {}, "latency_rel_fp32": {}}
    for t in tasks:
        mw = C.is_mw(t)
        keys = {(f"e7{p}it{it}", t, k, v) for p, v, k in PRECS for it in ITERS}
        P = None
        for kv in keys:
            s = {(kk[4], kk[5]) for kk in E if kk[:4] == kv}
            P = s if P is None else P & s
        P = C.Pairs(P or set())
        if P.n == 0:
            continue
        ys = {}
        for p, v, k in PRECS:
            for it in ITERS:
                kv = (f"e7{p}it{it}", t, k, v)
                ys[(p, v, it)] = P.arr(E, lambda c, e, kv=kv: kv + (c, e), mw=mw)
        for (p, v, it), y in ys.items():
            r = P.mean_ci(y)
            r["n_pairs"] = P.n
            st, _ = P.drop_ci(ys[("fp32", "none", it)], y, mw)
            r.update(D_vs_fp32=st["D"], D_vs_fp32_ci=st["D_ci"])
            for kern in (["int4", "int4p"] if p == "int4" else [p]):
                lk = f"{t}|{kern}_{v}|it{it}"
                if lk in lat:
                    r[f"latency_ms_{kern}"] = lat[lk]["median_ms"]
                    r[f"latency_iqr_{kern}"] = lat[lk]["iqr_ms"]
                    r[f"latency_round_medians_{kern}"] = round_medians(Lj, lk)
            out["success"][f"{t}|{p}_{v}|it{it}"] = r
        # det vs dither at the same precision and iterations (descriptive, paired)
        for p in ("int8", "int4"):
            for it in ITERS:
                d = ys[(p, "det", it)] - ys[(p, "shiter", it)]          # success(det) - success(dither)
                st_d, Dd = P.drop_ci(ys[("fp32", "none", it)], ys[(p, "det", it)], mw)
                st_s, Ds = P.drop_ci(ys[("fp32", "none", it)], ys[(p, "shiter", it)], mw)
                out["success"][f"{t}|{p}_det_minus_dither|it{it}"] = {
                    "D_det": st_d["D"], "D_dither": st_s["D"], "D_det_minus_D_dither": st_d["D"] - st_s["D"],
                    "D_det_minus_D_dither_ci": C.A1.ci95(Dd - Ds), "mean_success_diff_det_minus_dither": float(d.mean())}
        # Pareto: int4 dither (kernel container or packed) vs int8 det
        cases = []
        for kern in ("int4", "int4p"):
            for ia in ITERS:
                A_ = out["success"][f"{t}|int4_shiter|it{ia}"]
                for ib in ITERS:
                    B_ = out["success"][f"{t}|int8_det|it{ib}"]
                    la, lb = A_.get(f"latency_ms_{kern}"), B_.get("latency_ms_int8")
                    if la is None or lb is None:
                        continue
                    sa, sb = A_["mean"], B_["mean"]
                    if la <= lb and sa >= sb and (la < lb or sa > sb):
                        d = ys[("int4", "shiter", ia)] - ys[("int8", "det", ib)]
                        ra, rb = A_.get(f"latency_round_medians_{kern}"), B_.get("latency_round_medians_int8")
                        cases.append({"int4_kernel": kern, "int4_dither_iters": ia, "int8_det_iters": ib,
                                      "latency_ms": [la, lb], "success": [sa, sb],
                                      "success_diff": float(d.mean()), "success_diff_ci": C.A1.ci95(d[P.idx].mean(1)),
                                      "latency_lower_in_all_rounds": bool(ra and rb and all(x < y for x, y in zip(ra, rb))),
                                      "same_iterations": ia == ib})
        out["H5_pareto"][t] = cases
    # latency relative to fp32 at the same task and iterations (descriptive)
    for key, s in lat.items():
        t, pv, it = key.split("|")
        f = lat.get(f"{t}|fp32_none|{it}")
        if f:
            out["latency_rel_fp32"][key] = s["median_ms"] / f["median_ms"] - 1.0
    # H5 part 1 per kernel: every condition and the median over conditions
    for kern in ("int8", "int4", "int4p"):
        vals = {k: v for k, v in over.items() if k.split("|")[1] == kern}
        if vals:
            med = float(np.median(list(vals.values())))
            out["H5_overhead"][kern] = {"per_condition": vals, "max": float(max(vals.values())),
                                        "min": float(min(vals.values())),
                                        "all_le_10pct": bool(all(C.le(x, THR) for x in vals.values())),
                                        "median": med, "median_le_10pct": C.le(med, THR)}
        # per-round overheads from the block medians (descriptive)
        for key in vals:
            t, _, it = key.split("|")
            rd, rv = round_medians(Lj, f"{t}|{kern}_det|{it}"), round_medians(Lj, f"{t}|{kern}_shiter|{it}")
            if rd and rv:
                out["overhead_per_round"][key] = [(y - x) / x for x, y in zip(rd, rv)]
    # H5 verdict under every reading (container / packed int4) x (every condition / median of conditions)
    n_cases = {kern: sum(1 for c in out["H5_pareto"].values() for x in c if x["int4_kernel"] == kern)
               for kern in ("int4", "int4p")}
    readings = {}
    for kern in ("int4", "int4p"):
        for rd_name, fld in (("every_condition", "all_le_10pct"), ("median_of_conditions", "median_le_10pct")):
            o8 = out["H5_overhead"].get("int8", {}).get(fld, False)
            o4 = out["H5_overhead"].get(kern, {}).get(fld, False)
            readings[f"{kern}|{rd_name}"] = {"overhead_int8": o8, f"overhead_{kern}": o4, "pareto_cases": n_cases[kern],
                                             "holds": bool(o8 and o4 and n_cases[kern] >= 1)}
    out["H5"] = {"readings": readings, "holds_all_readings": bool(readings and all(r["holds"] for r in readings.values())),
                 "holds_any_reading": bool(any(r["holds"] for r in readings.values())),
                 "pareto_cases": sum(len(c) for c in out["H5_pareto"].values()),
                 "pareto_cases_by_kernel": n_cases,
                 "rule": "v5 7.4: (int8 dither - int8 det) / int8 det of the median latency <= 10% (int4 likewise) and "
                         ">= 1 Pareto dominance case of dither + fewer bits over det + more bits"}
    C.write_json(out, f"{a.outdir}/e7_tdmpc2_summary.json")
    C.write_json(family_points(out), f"{a.outdir}/family_points_e7_tdmpc2.json")
    write_md(out, f"{a.outdir}/e7_tdmpc2_tables.md")
    write_csv(out, f"{a.outdir}/e7_tdmpc2_rows.csv")
    print({"H5": {k: v for k, v in out["H5"].items() if k != "rule"},
           "overhead": {k: {kk: v[kk] for kk in ("max", "median", "all_le_10pct")} for k, v in out["H5_overhead"].items()}})


def family_points(out):
    return {"source": "analyze_e7.py (TD-MPC2; success: E7 seeds 70000-70029, 90 pairs; latency: e7_latency.json, "
                      "checkpoint 1, exclusive GPU 0, eager, 5 rounds x 200 steps)",
            "rule": out["H5"]["rule"], "H5": {k: v for k, v in out["H5"].items() if k != "rule"},
            "overhead": out["H5_overhead"], "pareto_cases": out["H5_pareto"],
            "units": "overhead: fraction of the det median latency; success: Meta-World success rate, DMC return; "
                     "latency: ms per control step"}


def write_md(out, path):
    f2, ci = C.f2, C.ci_txt
    h = out["H5"]
    L = ["# E7 TD-MPC2 (v5 7.4)", "", "## H5", ""]
    for k, r in h["readings"].items():
        L.append(f"- {k}: overhead int8 {r['overhead_int8']}, int4 kernel {r.get('overhead_' + k.split('|')[0])}, "
                 f"Pareto {r['pareto_cases']} → {r['holds']}")
    L += ["", "| kernel | 오버헤드 최소 / 중앙값 / 최대 | 모두 ≤ 10% |", "|---|---|---|"]
    for kern, r in out["H5_overhead"].items():
        L.append(f"| {kern} | {r['min']:+.3f} / {r['median']:+.3f} / {r['max']:+.3f} | {r['all_le_10pct']} |")
    L += ["", "| 과제·kernel·반복 | 오버헤드 | 회전별 오버헤드 |", "|---|---|---|"]
    for kern, r in out["H5_overhead"].items():
        for key, v in r["per_condition"].items():
            pr = out["overhead_per_round"].get(key, [])
            L.append(f"| {key} | {v:+.3f} | {', '.join(f'{x:+.3f}' for x in pr)} |")
    L += ["", "## Pareto 사례(int4 dither가 int8 det를 지배)", "",
          "| 과제 | int4 kernel | 반복 int4 dither / int8 det | 지연 ms | 성공 | 성공 차 [CI] | 5회전 모두 지연이 낮음 |",
          "|---|---|---|---|---|---|---|"]
    for t, cs in out["H5_pareto"].items():
        for c in cs:
            L.append(f"| {t} | {c['int4_kernel']} | {c['int4_dither_iters']} / {c['int8_det_iters']} | "
                     f"{c['latency_ms'][0]:.2f} / {c['latency_ms'][1]:.2f} | {c['success'][0]:.3f} / {c['success'][1]:.3f} | "
                     f"{c['success_diff']:+.3f} {ci(c['success_diff_ci'], 1, 3)} | {c['latency_lower_in_all_rounds']} |")
    L += ["", "## 성공률(실제 kernel)", "", "| 과제·정밀도·반복 | 성공(평균) [CI] | D vs fp32 [CI] | 지연 ms | fp32 대비 지연 |",
          "|---|---|---|---|---|"]
    for key, r in out["success"].items():
        if "det_minus_dither" in key:
            continue
        lats = ", ".join(f"{k[11:]} {v:.2f}" for k, v in r.items() if k.startswith("latency_ms_"))
        t, pv, it = key.split("|")
        prec, var = pv.split("_")
        rels = []
        for kern in (["int4", "int4p"] if prec == "int4" else [prec]):
            x = out["latency_rel_fp32"].get(f"{t}|{kern}_{var}|{it}")
            rels.append(f"{kern} {x:+.3f}" if x is not None else f"{kern} —")
        rel = ", ".join(rels)
        L.append(f"| {key} | {r['mean']:.3f} {ci(r['ci'], 1, 3)} | {f2(r['D_vs_fp32'], 3)} "
                 f"{ci(r['D_vs_fp32_ci'], 1, 3)} | {lats} | {rel} |")
    L += ["", "| 과제·정밀도·반복 | D_det − D_dither [CI] |", "|---|---|"]
    for key, r in out["success"].items():
        if "det_minus_dither" in key:
            L.append(f"| {key} | {f2(r['D_det_minus_D_dither'], 3)} {ci(r['D_det_minus_D_dither_ci'], 1, 3)} |")
    open(path, "w").write("\n".join(L) + "\n")


def write_csv(out, path):
    fmt = lambda x: ("[" + ";".join(fmt(y) for y in x) + "]") if isinstance(x, (list, tuple)) else (
        f"{x:.6g}" if isinstance(x, float) else ("" if x is None else str(x)))
    cols = ["config", "n_pairs", "mean", "ci", "D_vs_fp32", "D_vs_fp32_ci", "latency_ms_main", "latency_ms_int4p"]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for key, r in out["success"].items():
            if "det_minus_dither" in key:
                continue
            p = key.split("|")[1].split("_")[0]
            row = {"config": key, "n_pairs": r.get("n_pairs"), "mean": r["mean"], "ci": r["ci"],
                   "D_vs_fp32": r["D_vs_fp32"], "D_vs_fp32_ci": r["D_vs_fp32_ci"],
                   "latency_ms_main": r.get(f"latency_ms_{p}"), "latency_ms_int4p": r.get("latency_ms_int4p")}
            f.write(",".join(fmt(row[c]) for c in cols) + "\n")


if __name__ == "__main__":
    main()
