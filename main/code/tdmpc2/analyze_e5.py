"""E5 analysis of the TD-MPC2 family, DESIGN_main v5 7.3 (+ v5.1 8.1). Written before any E5 outcome existed; checked
on synthetic data only (test_v5_analyses.py).

  - baseline selection (held split, seeds 12000-12019, 60 pairs): D_det(B) of the 4 PTQ baselines per cell and the
    selected B (from <res>/e5/selection.json, written by e5_select.py before the evaluation)
  - H3b (E5 seeds 60000-60029, 90 pairs, 19 cells): per cell D(B+det) - D(B+sh-iter) with a paired bootstrap 95% CI
    (10,000 resamples, RandomState(0)); the cell passes if the CI lower bound > 0; H3b (TD-MPC2) holds if >= 10 of
    the 19 cells pass. Also D(B+det), D(B+sh-iter), red(B) = (D(B+det) - D(B+sh-iter)) / D(B+det)
  - weight-only (exploratory; 4 sweep tasks, 90 pairs): D of w8rtn, w4rtn, w4awq, w4sr1/2/4 against the E5 none
  - QAT (11 cells): D(QAT+det), D(QAT+sh-iter) against the E5 none; "QAT removes the damage" in a cell if the
    D(QAT+det) CI upper bound < 0.05 (5 pp for Meta-World, relative return 0.05 for DMC); dither on top of QAT:
    D(QAT+det) - D(QAT+sh-iter) with CI; the training-loss history of every QAT job
usage (server): python analyze_e5.py --res ~/pmx/A/tdmpc2_main/out_e5 --outdir ~/pmx/A/tdmpc2_main/analysis_e5

Changes of 2026-10-23 (E5 analysis worker, after the runs ended but before any E5 outcome was read; rules unchanged,
see worklogs/E5_tdmpc2.md): the H3b count is also reported per sub-family (Meta-World 12 cells, DMC 7 cells); McNemar
exact p of B+det vs B+sh-iter (Meta-World success, auxiliary, design 6.8); the QAT reporting condition is summarised
(cells meeting it, any); descriptive paired contrasts of the weight variants (w4rtn - w4srK, w4rtn - w4awq,
w8rtn - w4rtn); first / last QAT training loss; family_points_e5_tdmpc2.json and a CSV of the rows.
"""
import argparse
import glob
import gzip
import json
import os

import numpy as np

import v5_common as C

W_VARS = ("w8rtn", "w4rtn", "w4awq", "w4sr1", "w4sr2", "w4sr4")
W_CONTRASTS = (("w4rtn", "w4sr1"), ("w4rtn", "w4sr2"), ("w4rtn", "w4sr4"), ("w4rtn", "w4awq"), ("w8rtn", "w4rtn"))
THR_QAT = 0.05      # v5 7.3 / design 6.5: 5 pp (Meta-World success-rate fraction) or relative return 0.05 (DMC)


def comp(E, P, t, mw, kv_a, kv_b=None):
    """D of kv_a (and kv_b) against the E5 none, and D_a - D_b, on the pair set P."""
    y0 = P.arr(E, lambda c, e: ("base", t, 0, "none", c, e), mw=mw)
    ya = P.arr(E, lambda c, e: kv_a + (c, e), mw=mw)
    sa, Da = P.drop_ci(y0, ya, mw)
    sa["mean"] = float(ya.mean())
    out = {"n_pairs": P.n, "none_mean": float(y0.mean()), "a": sa}
    if kv_b is not None:
        yb = P.arr(E, lambda c, e: kv_b + (c, e), mw=mw)
        sb, Db = P.drop_ci(y0, yb, mw)
        sb["mean"] = float(yb.mean())
        out["b"] = sb
        diff = Da - Db
        out["Da_minus_Db"] = float(sa["D"] - sb["D"])
        out["Da_minus_Db_ci"] = C.A1.ci95(diff)
        out["red"] = float((sa["D"] - sb["D"]) / sa["D"]) if sa["D"] > 0 else None
        if mw:   # auxiliary (design 6.8): McNemar exact test of the paired successes, b-only vs a-only
            n_b, n_a, p = C.A1.mcnemar(yb.astype(bool), ya.astype(bool))
            out["mcnemar"] = {"b_only": n_b, "a_only": n_a, "p": p}
    return out


def pairs_for(E, t, kvs):
    s = None
    for kv in kvs:
        p = {(k[4], k[5]) for k in E if k[:4] == kv}
        s = p if s is None else s & p
    return C.Pairs(s or set())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    E = C.load_episodes(a.res, prefix="e5_")
    sel_p = f"{a.res}/e5/selection.json"
    sel = json.load(open(sel_p)) if os.path.exists(sel_p) else {"cells": {}}
    out = {"n_episodes": len(E), "selection": sel["cells"], "H3b_cells": {}, "weights": {}, "weight_contrasts": {},
           "qat": {}, "qat_train": {}}
    # H3b
    cells = sorted({(k[1], k[2]) for k in E if k[0] == "e5best"})
    for t, k in cells:
        mw = C.is_mw(t)
        kd, ks = ("e5best", t, k, "det"), ("e5best", t, k, "shiter")
        P = pairs_for(E, t, [("base", t, 0, "none"), kd, ks])
        if P.n == 0:
            continue
        r = comp(E, P, t, mw, kd, ks)
        r["B"] = sel["cells"].get(f"{t}|{k}", {}).get("best")
        r["B_resolved"] = sorted({E[key]["extra"].get("ptq_resolved") for key in E if key[:4] in (kd, ks)})
        r["family"] = "mw" if mw else "dmc"
        r["passes"] = bool(r["Da_minus_Db_ci"][0] > 0)
        out["H3b_cells"][f"{t}|{k}"] = r
    H = out["H3b_cells"].values()
    n_pass = sum(r["passes"] for r in H)
    out["H3b"] = {"cells": len(out["H3b_cells"]), "passing": n_pass,
                  "holds": bool(len(out["H3b_cells"]) == 19 and n_pass >= 10),
                  "rule": "D(B+det) - D(B+sh-iter) 95% CI lower > 0 in >= 10 of the 19 cells",
                  "by_family": {f: {"cells": sum(r["family"] == f for r in H),
                                    "passing": sum(r["passes"] for r in H if r["family"] == f)} for f in ("mw", "dmc")},
                  "passing_cells": [c for c, r in out["H3b_cells"].items() if r["passes"]],
                  "failing_cells": [c for c, r in out["H3b_cells"].items() if not r["passes"]]}
    # weights (exploratory)
    for t in sorted({k[1] for k in E if k[0].startswith("e5w")}):
        mw = C.is_mw(t)
        kvs = {w: (f"e5{w}", t, 0, "none") for w in W_VARS}
        P = pairs_for(E, t, [("base", t, 0, "none")] + list(kvs.values()))
        if not P.n:
            continue
        for w, kv in kvs.items():
            out["weights"][f"{t}|{w}"] = dict(comp(E, P, t, mw, kv)["a"], n_pairs=P.n)
        for wa, wb in W_CONTRASTS:
            r = comp(E, P, t, mw, kvs[wa], kvs[wb])
            out["weight_contrasts"][f"{t}|{wa}-{wb}"] = {"D_a": r["a"]["D"], "D_b": r["b"]["D"],
                                                          "Da_minus_Db": r["Da_minus_Db"],
                                                          "Da_minus_Db_ci": r["Da_minus_Db_ci"]}
    # QAT
    for t, k in sorted({(k[1], k[2]) for k in E if k[0] == "e5qat"}):
        mw = C.is_mw(t)
        kd, ks = ("e5qat", t, k, "det"), ("e5qat", t, k, "shiter")
        P = pairs_for(E, t, [("base", t, 0, "none"), kd, ks])
        if P.n == 0:
            continue
        r = comp(E, P, t, mw, kd, ks)
        r["family"] = "mw" if mw else "dmc"
        r["qat_removes_damage"] = bool(r["a"]["D_ci"][1] < THR_QAT)
        out["qat"][f"{t}|{k}"] = r
    meet = [c for c, r in out["qat"].items() if r["qat_removes_damage"]]
    out["qat_summary"] = {"cells": len(out["qat"]), "qat_removes_damage": len(meet), "cells_meeting": meet,
                          "condition_triggered_any_cell": bool(meet),
                          "rule": "v5 7.3 / design 6.5: if D(QAT+det) CI upper < 5 pp (DMC 0.05), the value of "
                                  "dither is reported as a training-free PTQ prescription"}
    for f in sorted(glob.glob(f"{a.res}/jobs/qat_train/*.json.gz")):
        d = json.load(gzip.open(f, "rt"))
        h = d.get("hist") or []
        out["qat_train"][d["spec"]["id"]] = {"hist": h, "n_windows": d.get("n_windows"),
                                             "loss_first": h[0]["loss"] if h else None,
                                             "loss_last": h[-1]["loss"] if h else None}
    C.write_json(out, f"{a.outdir}/e5_tdmpc2_summary.json")
    C.write_json(family_points(out), f"{a.outdir}/family_points_e5_tdmpc2.json")
    write_md(out, f"{a.outdir}/e5_tdmpc2_tables.md")
    write_csv(out, f"{a.outdir}/e5_tdmpc2_rows.csv")
    print({"H3b": {kk: out["H3b"][kk] for kk in ("cells", "passing", "holds", "by_family")}, "qat": out["qat_summary"],
           "weights": len(out["weights"])})


def family_points(out):
    P = {"source": "analyze_e5.py (TD-MPC2; E5 seeds 60000-60029, 90 pairs; B selected on held 12000-12019)",
         "rule_H3b": out["H3b"]["rule"], "rule_qat": out["qat_summary"]["rule"],
         "units": "D: Meta-World success-rate drop (fraction), DMC relative return drop (fraction), vs the E5 none",
         "H3b": {kk: out["H3b"][kk] for kk in ("cells", "passing", "holds", "by_family")}, "H3b_cells": {},
         "qat_cells": {}, "qat_condition_triggered_any_cell": out["qat_summary"]["condition_triggered_any_cell"]}
    for c, r in out["H3b_cells"].items():
        P["H3b_cells"][c] = {"family": r["family"], "B": r["B"], "held_D_det_B": out["selection"].get(c, {}).get("D_det"),
                             "D_Bdet": r["a"]["D"], "D_Bdet_ci": r["a"]["D_ci"], "D_Bshiter": r["b"]["D"],
                             "D_Bshiter_ci": r["b"]["D_ci"], "diff": r["Da_minus_Db"], "diff_ci": r["Da_minus_Db_ci"],
                             "passes": r["passes"], "red": r["red"], "mcnemar": r.get("mcnemar")}
    for c, r in out["qat"].items():
        P["qat_cells"][c] = {"family": r["family"], "D_qat_det": r["a"]["D"], "D_qat_det_ci": r["a"]["D_ci"],
                             "D_qat_shiter": r["b"]["D"], "D_qat_shiter_ci": r["b"]["D_ci"], "diff": r["Da_minus_Db"],
                             "diff_ci": r["Da_minus_Db_ci"], "qat_removes_damage": r["qat_removes_damage"]}
    return P


def write_md(out, path):
    f2, ci = C.f2, C.ci_txt
    h = out["H3b"]
    L = ["# E5 TD-MPC2 (v5 7.3)", "",
         "D: Meta-World 성공률 하락(비율), DMC 상대 return 하락(비율), E5 none 대비. CI: 짝 bootstrap 10,000회 95%.", "",
         f"H3b: {h['passing']} / {h['cells']} 칸 통과 → {h['holds']} (MW {h['by_family']['mw']['passing']}/"
         f"{h['by_family']['mw']['cells']}, DMC {h['by_family']['dmc']['passing']}/{h['by_family']['dmc']['cells']})", "",
         "## 기준선 선택(held, 60짝)", "", "| 칸 | B | D_det perchan | mseclip | smooth | mixed |", "|---|---|---|---|---|---|"]
    for c, s in out["selection"].items():
        D = s["D_det"]
        L.append(f"| {c} | {s['best']} | " + " | ".join(f2(D.get(b), 3) for b in ("perchan", "mseclip", "smooth", "mixed"))
                 + " |")
    L += ["", "## H3b", "", "| 칸 | B | D(B+det) [CI] | D(B+sh-iter) [CI] | 차 [CI] | red | McNemar p | 통과 |",
          "|---|---|---|---|---|---|---|---|"]
    for cid, r in out["H3b_cells"].items():
        mc = r.get("mcnemar")
        L.append(f"| {cid} | {r['B']} | {f2(r['a']['D'], 3)} {ci(r['a']['D_ci'], 1, 3)} | "
                 f"{f2(r['b']['D'], 3)} {ci(r['b']['D_ci'], 1, 3)} | {f2(r['Da_minus_Db'], 3)} "
                 f"{ci(r['Da_minus_Db_ci'], 1, 3)} | {f2(r['red'])} | {'' if not mc else format(mc['p'], '.4f')} | "
                 f"{r['passes']} |")
    L += ["", "## QAT", "", "| QAT 칸 | D(QAT+det) [CI] | D(QAT+sh-iter) [CI] | 차 [CI] | QAT+det CI 상한 < 0.05 |",
          "|---|---|---|---|---|"]
    for cid, r in out["qat"].items():
        L.append(f"| {cid} | {f2(r['a']['D'], 3)} {ci(r['a']['D_ci'], 1, 3)} | {f2(r['b']['D'], 3)} "
                 f"{ci(r['b']['D_ci'], 1, 3)} | {f2(r['Da_minus_Db'], 3)} {ci(r['Da_minus_Db_ci'], 1, 3)} | "
                 f"{r['qat_removes_damage']} |")
    L += ["", "| QAT 학습 job | 손실 처음 → 끝 |", "|---|---|"]
    for jid, r in out["qat_train"].items():
        L.append(f"| {jid} | {r['loss_first']:.4f} → {r['loss_last']:.4f} |")
    L += ["", "## 가중치(탐색)", "", "| 과제·변형 | D [CI] |", "|---|---|"]
    for cid, r in out["weights"].items():
        L.append(f"| {cid} | {f2(r['D'], 3)} {ci(r['D_ci'], 1, 3)} |")
    L += ["", "| 과제·대조 | D_a − D_b [CI] |", "|---|---|"]
    for cid, r in out["weight_contrasts"].items():
        L.append(f"| {cid} | {f2(r['Da_minus_Db'], 3)} {ci(r['Da_minus_Db_ci'], 1, 3)} |")
    open(path, "w").write("\n".join(L) + "\n")


def write_csv(out, path):
    fmt = lambda x: ("[" + ";".join(fmt(y) for y in x) + "]") if isinstance(x, (list, tuple)) else (
        f"{x:.6g}" if isinstance(x, float) else ("" if x is None else str(x)))
    cols = ["part", "cell", "B", "n_pairs", "none_mean", "mean_a", "D_a", "D_a_ci", "mean_b", "D_b", "D_b_ci",
            "Da_minus_Db", "Da_minus_Db_ci", "red", "passes_or_flag"]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for part, src in (("H3b", out["H3b_cells"]), ("QAT", out["qat"])):
            for c, r in src.items():
                row = {"part": part, "cell": c, "B": r.get("B", "qat"), "n_pairs": r["n_pairs"],
                       "none_mean": r["none_mean"], "mean_a": r["a"]["mean"], "D_a": r["a"]["D"], "D_a_ci": r["a"]["D_ci"],
                       "mean_b": r["b"]["mean"], "D_b": r["b"]["D"], "D_b_ci": r["b"]["D_ci"],
                       "Da_minus_Db": r["Da_minus_Db"], "Da_minus_Db_ci": r["Da_minus_Db_ci"], "red": r["red"],
                       "passes_or_flag": r.get("passes", r.get("qat_removes_damage"))}
                f.write(",".join(fmt(row[c_]) for c_ in cols) + "\n")
        for c, r in out["weights"].items():
            row = {"part": "weights", "cell": c, "B": "", "n_pairs": r["n_pairs"], "none_mean": None, "mean_a": r["mean"],
                   "D_a": r["D"], "D_a_ci": r["D_ci"], "mean_b": None, "D_b": None, "D_b_ci": None, "Da_minus_Db": None,
                   "Da_minus_Db_ci": None, "red": None, "passes_or_flag": None}
            f.write(",".join(fmt(row[c_]) for c_ in cols) + "\n")


if __name__ == "__main__":
    main()
