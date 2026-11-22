"""R5: persistence-cost table with CIs, recomputed from episodes and checked against the published contrast files.

  TD-MPC2 5M (E11, 12 cells, PT first-order k, 90 pairs): D(sh-ep) - D(sh-iter) and
      Delta_pers = D(det+shift-fixed) - D(det+shift). Check: td_e11_contrasts.csv.
      Extra (same family, E16 CEM-style planner, 3 cells): D(sh-ep) - D(sh-iter). Check: td_summary.json E16.
  TD-MPC2 317M (E14, 8 tasks, 90 seeds): D(sh-ep) - D(sh-iter). Check: coordinator/verify_mt_e14.txt values.
  DINO-WM: E11 cells (PushT 4-bit, PointMaze 3-bit, Wall 3-bit; CEM): D(sh-plan) - D(sh-iter) and Delta_pers;
      E16 MPPI-style cells (PointMaze 3-bit, Wall 3-bit): D(sh-plan) - D(sh-iter). Check: analysis_dw.json.
Statistics as published: paired bootstrap 10,000, RandomState(0), one index matrix per cell (shared by the variants of
the cell), percentile 95% CI; MW / DINO-WM in success-rate points, DMC as relative return drop.
Side by side: DINO-WM CI upper bounds vs TD-MPC2 CI lower bounds (same units: MW and 317M success points).
Output: r5.json, r5_persistence.csv, r5_side_by_side.csv.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

OUT = C.HERE
E11V = ["det", "detbc", "shep", "detshiftfix", "shiterb", "shiter", "detshift", "detnoise"]


def boot_W(n):
    return C.counts(np.random.RandomState(0).randint(0, n, size=(C.NB, n)), n)


def td_cell(E, t, ptag, variants, k):
    """paired arrays over the pairs common to none and every variant (as analyze_td.Comp)."""
    mw = C.is_mw(t)
    j = 1 if mw else 0
    none = E[(t, "none", ptag)]
    cellsd = {v: E[(t, f"PTk{k}_{v}", ptag)] for v in variants}
    keys = sorted(set(none).intersection(*[set(d) for d in cellsd.values()]))
    y0 = np.array([none[p][j] for p in keys])
    Y = {v: np.array([cellsd[v][p][j] for p in keys]) for v in variants}
    W = boot_W(len(keys))
    W1 = C.ones(len(keys))
    m = np.ones(len(keys), bool)
    D = {v: float(C.d_w(W1, m, y0, Y[v], mw)[0]) for v in variants}
    Db = {v: C.d_w(W, m, y0, Y[v], mw) for v in variants}
    return len(keys), D, Db


def contrast(D, Db, a, b):
    return dict(est=D[a] - D[b], ci=C.ci95(Db[a] - Db[b]))


def main():
    rows, checks = [], []
    out = dict(meta=dict(B=C.NB, seed="RandomState(0), one index matrix per cell"), td5m={}, td5m_cem={}, mt={}, dw={})
    # ------------------------------------------------------------ TD-MPC2 5M E11
    E = C.load_td()
    pub = {}
    with open(C.TD_CON) as f:
        for r in csv.DictReader(f):
            pub[(r["task"], r["contrast"])] = (float(r["D"]), float(r["ci_lo"]), float(r["ci_hi"]))
    for t in C.MW8 + C.DMC4:
        n, D, Db = td_cell(E, t, "base", E11V, C.K1[t])
        sp = contrast(D, Db, "shep", "shiter")
        pe = contrast(D, Db, "detshiftfix", "detshift")
        out["td5m"][t] = dict(k=C.K1[t], n=n, shep_minus_shiter=sp, delta_pers=pe, D_det=D["det"])
        unit = "MW" if C.is_mw(t) else "DMC"
        for lab, x, key in (("D(sh-ep)-D(sh-iter)", sp, "shep-shiter"), ("Delta_pers", pe, "D_pers=detshiftfix-detshift")):
            p = pub[(t, key)]
            dev = max(abs(x["est"] - p[0]), abs(x["ci"][0] - p[1]), abs(x["ci"][1] - p[2]))
            checks.append(["TD-MPC2 5M E11", t, lab, round(dev, 7), dev < 1e-5])
            rows.append(["TD-MPC2 5M", unit, "E11 default MPPI", t, C.K1[t], n, lab, round(x["est"], 4),
                         round(x["ci"][0], 4), round(x["ci"][1], 4), x["ci"][0] > C.TOL])
    # ------------------------------------------------------------ TD-MPC2 5M E16 CEM-style (extra)
    S = json.load(open(C.TD_SUM))["E16"]
    for t in ("mw-push", "mw-pick-place", "cheetah-run"):
        n, D, Db = td_cell(E, t, "cem", ["det", "shiter", "shep", "detshift"], C.K1[t])
        sp = contrast(D, Db, "shep", "shiter")
        out["td5m_cem"][t] = dict(k=C.K1[t], n=n, shep_minus_shiter=sp)
        p = S[t]["shep_minus_shiter"]
        dev = max(abs(sp["est"] - p["D"]), abs(sp["ci"][0] - p["ci"][0]), abs(sp["ci"][1] - p["ci"][1]))
        checks.append(["TD-MPC2 5M E16 CEM", t, "D(sh-ep)-D(sh-iter)", round(dev, 7), dev < 1e-5])
        rows.append(["TD-MPC2 5M", "MW" if C.is_mw(t) else "DMC", "E16 CEM-style", t, C.K1[t], n,
                     "D(sh-ep)-D(sh-iter)", round(sp["est"], 4), round(sp["ci"][0], 4), round(sp["ci"][1], 4),
                     sp["ci"][0] > C.TOL])
    # ------------------------------------------------------------ 317M E14
    M = C.load_mt()
    pub_mt = {"mw-button-press-topdown": (0.789, 0.700, 0.867), "mw-door-open": (0.700, 0.589, 0.800),
              "mw-drawer-open": (0.244, 0.122, 0.367), "mw-peg-insert-side": (0.822, 0.744, 0.900),
              "mw-pick-place": (0.622, 0.522, 0.722), "mw-push": (0.033, 0.000, 0.078),
              "mw-sweep-into": (0.522, 0.422, 0.622), "mw-window-open": (0.633, 0.533, 0.733)}  # verify_mt_e14.txt
    seeds = list(range(102000, 102090))
    W = boot_W(90)
    W1 = C.ones(90)
    m = np.ones(90, bool)
    for t in sorted({t for t, _ in M}):
        y0 = np.array([M[(t, "none")][s] for s in seeds])
        Y = {v: np.array([M[(t, v)][s] for s in seeds]) for v in ("det", "shiter", "shep")}
        D = {v: float(C.d_w(W1, m, y0, Y[v], True)[0]) for v in Y}
        Db = {v: C.d_w(W, m, y0, Y[v], True) for v in Y}
        sp = contrast(D, Db, "shep", "shiter")
        out["mt"][t] = dict(n=90, shep_minus_shiter=sp, D_det=D["det"])
        p = pub_mt[t]
        dev = max(abs(sp["est"] - p[0]), abs(sp["ci"][0] - p[1]), abs(sp["ci"][1] - p[2]))
        checks.append(["TD-MPC2 317M E14", t, "D(sh-ep)-D(sh-iter)", round(dev, 7), dev < 5e-4])
        rows.append(["TD-MPC2 317M", "MW", "E14 default MPPI", t, "", 90, "D(sh-ep)-D(sh-iter)", round(sp["est"], 4),
                     round(sp["ci"][0], 4), round(sp["ci"][1], 4), sp["ci"][0] > C.TOL])
    # ------------------------------------------------------------ DINO-WM
    cells, _ = C.load_dw()
    A = json.load(open(C.DW_AN))
    pub_dw11 = {c["task"]: c for c in A["E11"]["cells"]}
    for task, k, pod, pl, lab in (("pusht", 4, "dw.1", "cem", "E11 CEM"), ("pointmaze", 3, "dw.3", "cem", "E11 CEM"),
                                  ("wall", 3, "dw.4", "cem", "E11 CEM"), ("pointmaze", 3, "dw.3", "mppi2", "E16 MPPI-style"),
                                  ("wall", 3, "dw.4", "mppi1", "E16 MPPI-style")):
        vs = ["det", "shstep", "shiter", "detshiftfix", "detshift"] if pl == "cem" else ["det", "shstep", "shiter", "detshift"]
        nn = C.dw_none(cells, pod, task, pl)
        cd = {v: cells[(pod, task, "eval8", "main", v, k, pl)] for v in vs}
        eps = sorted(set(nn).intersection(*[set(c) for c in cd.values()]))
        y0 = np.array([nn[e] for e in eps])
        Y = {v: np.array([cd[v][e] for e in eps]) for v in vs}
        Wd = boot_W(len(eps))
        m = np.ones(len(eps), bool)
        D = {v: float(C.d_w(C.ones(len(eps)), m, y0, Y[v], True)[0]) for v in vs}
        Db = {v: C.d_w(Wd, m, y0, Y[v], True) for v in vs}
        sp = contrast(D, Db, "shstep", "shiter")
        ent = dict(k=k, pod=pod, planner=pl, n=len(eps), none=float(y0.mean()), shplan_minus_shiter=sp)
        rows.append(["DINO-WM", "DW", lab, task, k, len(eps), "D(sh-plan)-D(sh-iter)", round(sp["est"], 4),
                     round(sp["ci"][0], 4), round(sp["ci"][1], 4), sp["ci"][0] > C.TOL])
        if pl == "cem":
            pe = contrast(D, Db, "detshiftfix", "detshift")
            ent["delta_pers"] = pe
            rows.append(["DINO-WM", "DW", lab, task, k, len(eps), "Delta_pers", round(pe["est"], 4),
                         round(pe["ci"][0], 4), round(pe["ci"][1], 4), pe["ci"][0] > C.TOL])
            # checks against analysis_dw.json E11 (d_pers, aux D(sh-plan)-D(sh-iter))
            pc_ = pub_dw11[task]
            for x, p, nm in ((pe, pc_["d_pers"], "Delta_pers"), (sp, pc_["aux"]["D(sh-plan)-D(sh-iter)"], "D(sh-plan)-D(sh-iter)")):
                dev = max(abs(x["est"] - p["diff"]), abs(x["ci"][0] - p["ci"][0]), abs(x["ci"][1] - p["ci"][1]))
                checks.append(["DINO-WM E11", task, nm, round(dev, 7), dev < 1e-9])
            if task == "pusht":     # D1 auxiliary: cluster bootstrap over the 10 eval8 overlap components
                comp = C.dw_components()
                Wc = C.w_cluster(np.random.RandomState(0), [comp[e] for e in eps])
                Dc = {v: C.d_w(Wc, m, y0, Y[v], True) for v in vs}
                ent["cluster_ci"] = {"shplan_minus_shiter": C.ci95(Dc["shstep"] - Dc["shiter"]),
                                     "delta_pers": C.ci95(Dc["detshiftfix"] - Dc["detshift"])}
                p = pc_["cluster"]["d_pers"]["ci"]
                dev = max(abs(ent["cluster_ci"]["delta_pers"][0] - p[0]), abs(ent["cluster_ci"]["delta_pers"][1] - p[1]))
                checks.append(["DINO-WM E11 cluster", task, "Delta_pers cluster CI", round(dev, 7), dev < 1e-9])
                for nm, ci in (("D(sh-plan)-D(sh-iter)", ent["cluster_ci"]["shplan_minus_shiter"]),
                               ("Delta_pers", ent["cluster_ci"]["delta_pers"])):
                    est = sp["est"] if nm.startswith("D(") else pe["est"]
                    rows.append(["DINO-WM", "DW", lab + " (D1 cluster CI)", task, k, len(eps), nm, round(est, 4),
                                 round(ci[0], 4), round(ci[1], 4), ci[0] > C.TOL])
        else:
            p = A["E16"][task]["D(sh-plan)-D(sh-iter)"]["mppi"]
            dev = max(abs(sp["est"] - p["diff"]), abs(sp["ci"][0] - p["ci"][0]), abs(sp["ci"][1] - p["ci"][1]))
            checks.append(["DINO-WM E16", task, "D(sh-plan)-D(sh-iter) MPPI", round(dev, 7), dev < 1e-9])
            pc = A["E16"][task]["D(sh-plan)-D(sh-iter)"]["cem"]
            e11 = out["dw"][f"{task}|cem"]["shplan_minus_shiter"]
            dev = max(abs(e11["est"] - pc["diff"]), abs(e11["ci"][0] - pc["ci"][0]), abs(e11["ci"][1] - pc["ci"][1]))
            checks.append(["DINO-WM E11 (via E16 file)", task, "D(sh-plan)-D(sh-iter) CEM", round(dev, 7), dev < 1e-9])
        out["dw"][f"{task}|{pl}"] = ent
    # ------------------------------------------------------------ side by side
    side = []
    mw_lo = {("shep", t): out["td5m"][t]["shep_minus_shiter"]["ci"][0] for t in C.MW8}
    mw_lo.update({("pers", t): out["td5m"][t]["delta_pers"]["ci"][0] for t in C.MW8})
    mt_lo = {t: out["mt"][t]["shep_minus_shiter"]["ci"][0] for t in out["mt"]}
    for key, ent in out["dw"].items():
        for lab, x, kind in (("D(sh-plan)-D(sh-iter)", ent["shplan_minus_shiter"], "shep"),
                             ("Delta_pers", ent.get("delta_pers"), "pers")):
            if x is None:
                continue
            hi = x["ci"][1]
            n_mw = sum(mw_lo[(kind, t)] > hi + C.TOL for t in C.MW8)
            n_mt = sum(v > hi + C.TOL for v in mt_lo.values()) if kind == "shep" else None
            side.append(dict(dw_cell=key, contrast=lab, est=x["est"], ci_hi=hi,
                             n_mw5m_lower_above=n_mw, n_mt317m_lower_above=n_mt))
    out["side_by_side"] = side
    out["td_lower_bounds"] = dict(
        mw5m_shep=sorted(((round(out["td5m"][t]["shep_minus_shiter"]["ci"][0], 4), C.SHORT[t]) for t in C.MW8)),
        mw5m_pers=sorted(((round(out["td5m"][t]["delta_pers"]["ci"][0], 4), C.SHORT[t]) for t in C.MW8)),
        dmc5m_shep=sorted(((round(out["td5m"][t]["shep_minus_shiter"]["ci"][0], 4), C.SHORT[t]) for t in C.DMC4)),
        dmc5m_pers=sorted(((round(out["td5m"][t]["delta_pers"]["ci"][0], 4), C.SHORT[t]) for t in C.DMC4)),
        mt317m_shep=sorted(((round(v, 4), C.SHORT[t]) for t, v in mt_lo.items())))
    out["checks"] = checks
    C.dump_json(os.path.join(OUT, "r5.json"), out)
    C.write_csv(os.path.join(OUT, "r5_persistence.csv"),
                ["family", "unit", "setting", "task", "k", "n", "contrast", "estimate", "ci_lo", "ci_hi", "ci_lo_gt_0"], rows)
    C.write_csv(os.path.join(OUT, "r5_side_by_side.csv"),
                ["dw_cell", "contrast", "dw_estimate", "dw_ci_hi", "n_of_8_MW5M_cells_with_ci_lo_above",
                 "n_of_8_317M_tasks_with_ci_lo_above"],
                [[s["dw_cell"], s["contrast"], round(s["est"], 4), round(s["ci_hi"], 4), s["n_mw5m_lower_above"],
                  "" if s["n_mt317m_lower_above"] is None else s["n_mt317m_lower_above"]] for s in side])
    for c in checks:
        print("check", c)
    for r in rows:
        print(r)
    for s in side:
        print(s)
    print(out["td_lower_bounds"])


if __name__ == "__main__":
    main()
