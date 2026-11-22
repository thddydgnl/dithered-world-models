"""X5 analysis (tmlr_ext/DESIGN_tmlr.md 11.1, D6; post hoc, descriptive, no thresholds; frozen design, commit 1d73e10).
Meta-World det+noise bit curve from pod td.7 (A40), compared with E12's k* (td.5 / td.6, same GPU type, other pods:
not paired, so every comparison of k* is descriptive).
  - D per (task, range kind, k): none - det+noise success rate on the cell's pairs (90), paired bootstrap 10,000
    resamples RandomState(0), percentile 95% CI (analyze_td.Comp)
  - k*(det+noise) per (task, range kind): E12's definition, the smallest measured k with D <= tau = 0.10 such that every
    larger measured k also has D <= tau (tolerance 1e-9); none -> 9 ("> 8")
  - E12 k*(det), k*(det+BC), k*(sh-iter), k*(det+shift) from TD_tdmpc2/results/analysis/td_e12_cells.csv
  - per range kind, numbers of the 8 tasks with (1) k*(noise) <= k*(sh-iter), (2) k*(noise) <= min(k*(sh-iter),
    k*(det+shift)), (3) G_noise = min(k*(det), k*(det+BC)) - k*(noise) >= 1
  - reproduction: the E11 primary-k PT det+noise cell and none measured again here vs E11 (td.5 / td.6 job files):
    D and CI of both, and the share of episodes whose return / success are bit-identical
Usage: python analyze_x5.py --res <td.7 out> --e11 <dir with E11 MW job files> --e12 <td_e12_cells.csv> --outdir <dir>
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_td as A  # noqa: E402
import plan_td as P  # noqa: E402

TAU, TOL, KS = P.TAU, P.TOL, P.KMAX + 1


def kstar(curve):
    ks = sorted(curve)
    for k in ks:
        if all(curve[kk] <= TAU + TOL for kk in ks if kk >= k):
            return k
    return KS


def e12_kstar(path):
    out = {}
    for r in csv.DictReader(open(path)):
        out[(r["task"], r["range"], r["rounding"])] = int(r["kstar"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--e11", required=True)
    ap.add_argument("--e12", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    E, _, meta = A.load([a.res])
    E11, _, meta11 = A.load([a.e11])
    k12 = e12_kstar(a.e12)
    curves, ks_noise, rows = {}, {}, []
    for t in P.MW8:
        for rk in ("PT", "PC"):
            ks = sorted({int(cell.split("_")[0][3:]) for (tt, c, cell, pt) in E if tt == t and pt == "base"
                         and cell.startswith(rk + "k") and cell.endswith("_detnoise")})
            cur = {}
            for k in ks:
                nm = P.cell_name("detnoise", rk, k)
                C = A.Comp(E, t, "base", [nm])
                if not C.ok:
                    continue
                D, Db = C.D(nm)
                cur[k] = {"D": float(D), "D_ci": A.ci95(Db), "n": C.n, "mean": float(C.Y[nm].mean()),
                          "none_mean": float(C.Y["none"].mean())}
                rows.append({"task": t, "range": rk, "k": k, **cur[k]})
            curves[f"{t}|{rk}"] = cur
            if cur:
                ks_noise[(t, rk)] = kstar({k: v["D"] for k, v in cur.items()})
    table, counts = {}, {}
    for rk in ("PT", "PC"):
        c1 = c2 = c3 = 0
        n_def = 0
        for t in P.MW8:
            kn = ks_noise.get((t, rk))
            kd, kb, ksh, kds = (k12.get((t, rk, v)) for v in ("det", "detbc", "shiter", "detshift"))
            if kn is None or None in (kd, kb, ksh, kds):
                table[f"{t}|{rk}"] = {"kstar_detnoise": kn, "complete": False}
                continue
            n_def += 1
            g = min(kd, kb) - kn
            f1, f2, f3 = kn <= ksh, kn <= min(ksh, kds), g >= 1
            c1 += f1
            c2 += f2
            c3 += f3
            table[f"{t}|{rk}"] = {"kstar_detnoise": kn, "kstar_det": kd, "kstar_detbc": kb, "kstar_shiter": ksh,
                                  "kstar_detshift": kds, "G_noise": g, "noise_le_shiter": f1,
                                  "noise_le_min_shiter_detshift": f2, "G_noise_ge_1": f3,
                                  "highest_k_measured": max(curves[f"{t}|{rk}"])}
        counts[rk] = {"n_tasks": n_def, "noise_le_shiter": c1, "noise_le_min_shiter_detshift": c2,
                      "G_noise_ge_1": c3}
    repro = {}
    for t in P.MW8:
        k = P.K1[t]
        nm = P.cell_name("detnoise", "PT", k)
        row = {"k": k}
        for lab, EE in (("x5", E), ("e11", E11)):
            C = A.Comp(EE, t, "base", [nm])
            if C.ok:
                D, Db = C.D(nm)
                row[lab] = {"D": float(D), "D_ci": A.ci95(Db), "n": C.n}
        same = {}
        for cell in ("none", nm):
            tot = eq = 0
            for c in P.CKPTS:
                x, y = E.get((t, c, cell, "base"), {}), E11.get((t, c, cell, "base"), {})
                for ep in set(x) & set(y):
                    tot += 1
                    eq += int(x[ep]["ret"] == y[ep]["ret"] and x[ep]["success"] == y[ep]["success"])
            same["none" if cell == "none" else "detnoise"] = {"n": tot, "identical": eq}
        row["bit_identical_episodes"] = same
        repro[t] = row
    S = {"source": "analyze_x5.py (D6, descriptive)", "n_episodes": sum(len(d) for d in E.values()),
         "meta": {k: meta[k] for k in ("files", "gpus", "seed_mismatch", "pairing_one_host_per_task_ckpt",
                                       "decisions")},
         "curves": curves, "kstar": table, "counts": counts, "repro_e11": repro,
         "note": "E12 k* come from other pods (td.5 / td.6, same GPU type): not paired, descriptive only"}
    json.dump(S, open(f"{a.outdir}/x5_summary.json", "w"), indent=1, default=float)
    with open(f"{a.outdir}/x5_curve.csv", "w") as f:
        f.write("task,range,k,n,none_mean,mean,D,ci_lo,ci_hi\n")
        for r in rows:
            f.write(f"{r['task']},{r['range']},{r['k']},{r['n']},{r['none_mean']:.6g},{r['mean']:.6g},{r['D']:.6g},"
                    f"{r['D_ci'][0]:.6g},{r['D_ci'][1]:.6g}\n")
    with open(f"{a.outdir}/x5_kstar.csv", "w") as f:
        cols = ["kstar_detnoise", "kstar_det", "kstar_detbc", "kstar_shiter", "kstar_detshift", "G_noise",
                "noise_le_shiter", "noise_le_min_shiter_detshift", "G_noise_ge_1", "highest_k_measured"]
        f.write("task,range," + ",".join(cols) + "\n")
        for key, x in table.items():
            t, rk = key.split("|")
            f.write(f"{t},{rk}," + ",".join("" if x.get(c) is None else str(x.get(c)) for c in cols) + "\n")
    L = ["# X5 (D6): MW det+noise 비트 곡선 (analyze_x5.py 자동 생성, 기술)", "",
         "| 과제 | 범위 | k\\*(det+noise) | E12 k\\*: det / det+BC / sh-iter / det+shift | G_noise | noise ≤ sh-iter | "
         "noise ≤ min(sh-iter, det+shift) | det+noise D(k) [CI] |", "|---|---|---|---|---|---|---|---|"]
    for key, x in table.items():
        t, rk = key.split("|")
        cur = curves[key]
        cs = "; ".join(f"{k}: {100 * v['D']:+.1f} [{100 * v['D_ci'][0]:.1f}, {100 * v['D_ci'][1]:.1f}]"
                       for k, v in sorted(cur.items()))
        L.append(f"| {t} | {rk} | {x.get('kstar_detnoise')} | {x.get('kstar_det')} / {x.get('kstar_detbc')} / "
                 f"{x.get('kstar_shiter')} / {x.get('kstar_detshift')} | {x.get('G_noise')} | {x.get('noise_le_shiter')} | "
                 f"{x.get('noise_le_min_shiter_detshift')} | {cs} |")
    L += ["", "| 범위 | 과제 | noise ≤ sh-iter | noise ≤ min(sh-iter, det+shift) | G_noise ≥ 1 |", "|---|---|---|---|---|"]
    for rk, c in counts.items():
        L.append(f"| {rk} | {c['n_tasks']} | {c['noise_le_shiter']} | {c['noise_le_min_shiter_detshift']} | "
                 f"{c['G_noise_ge_1']} |")
    L += ["", "## E11 재현 점검 (PT 1차 k det+noise, 같은 seed·같은 기종 다른 pod)", "",
          "| 과제 | k | D X5 [CI] | D E11 [CI] | 같은 episode(none) | 같은 episode(det+noise) |", "|---|---|---|---|---|---|"]
    for t, r in repro.items():
        def dt(x):
            return "" if not x else f"{100 * x['D']:+.1f} [{100 * x['D_ci'][0]:.1f}, {100 * x['D_ci'][1]:.1f}] (n {x['n']})"
        b = r["bit_identical_episodes"]
        L.append(f"| {t} | {r['k']} | {dt(r.get('x5'))} | {dt(r.get('e11'))} | {b['none']['identical']}/{b['none']['n']} | "
                 f"{b['detnoise']['identical']}/{b['detnoise']['n']} |")
    open(f"{a.outdir}/x5_tables.md", "w").write("\n".join(L) + "\n")
    print(json.dumps({"counts": counts, "repro": {t: r["bit_identical_episodes"] for t, r in repro.items()},
                      "n_episodes": S["n_episodes"]}, default=float))


if __name__ == "__main__":
    main()
