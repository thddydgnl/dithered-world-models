"""R2: tau sensitivity of k*, G and the Fr1/Fr2 counts (E12 point estimates, no resampling).

tau in {0.05, 0.075, 0.10, 0.125, 0.15, 0.20}, plus "0.10_strict" (D < 0.10 instead of D <= 0.10; four MW cells have
D = 9/90 = 0.100 exactly). Dither bases: best = min(k*(sh-iter), k*(det+shift)) (the published G),
sh-iter alone, det+shift alone; the det side is always min(k*(det), k*(det+BC)). DINO-WM has sh-iter only (RC: det
side = k*(det)). The measured k grids were chosen by the extension rule at tau = 0.10, so at other tau a k* can be
censored: "<=k" = the lowest measured k already passes (k > 1), ">=k" = no measured k passes (k = highest measured + 1).
G is reported at the boundary convention with the interval [G_lo, G_hi]; a Fr count is reported as point [lo, hi].
Also lists every E12 cell whose D lies within 0.01 of tau (|D - tau| <= 0.01).
Output: r2.json, r2_G.csv, r2_fr.csv, r2_near_tau.csv.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

OUT = C.HERE
STRICT = 0.10 - 2 * C.TOL        # "D < 0.10" (the published rule is D <= 0.10 with tolerance 1e-9)
TAUS = [0.05, 0.075, 0.10, STRICT, 0.125, 0.15, 0.20]


def tkey(tau):
    return "0.10_strict" if tau == STRICT else str(tau)
BASES = ("best", "shiter", "detshift")
FR = {"Fr1": ("PT", 6), "Fr2": ("PC", 4)}


def kfmt(x):
    k = int(x["k"][0])
    if x["cens_dn"][0]:
        return f"<={k}"
    if x["cens_up"][0]:
        return f">={k}" if k <= C.KMAX else ">8"
    return str(k)


def gfmt(g):
    G, lo, hi = int(g["G"][0]), int(g["lo"][0]), int(g["hi"][0])
    return str(G) if lo == hi else f"{G} [{lo},{hi}]"


def main():
    E = C.load_td()
    S = C.td_struct(E)
    cells, _ = C.load_dw()
    SD = C.dw_struct(cells)
    out = dict(meta=dict(taus=TAUS, tol=C.TOL, near_window=0.01), tau={})
    rows_G, rows_fr, rows_near = [], [], []
    # point D of every E12 cell (tau-independent)
    td_D = {}
    for t in C.MW8 + C.DMC4:
        St = S[t]
        W1 = C.ones(len(St["pairs"]))
        for (rk, v), curve in St["cells"].items():
            for k, (m, yv) in curve.items():
                td_D[(t, rk, v, k)] = (float(C.d_w(W1, m, St["y0"], yv, St["mw"])[0]), int(m.sum()))
    dw_D = {}
    for t in C.DW_TASKS:
        St = SD[t]
        W1 = C.ones(len(St["eps"]))
        for (rk, v), curve in St["cells"].items():
            for k, (m, y0, yv, pod) in curve.items():
                dw_D[(t, rk, v, k)] = (float(C.d_w(W1, m, y0, yv, True)[0]), int(m.sum()), pod)
    for tau in TAUS:
        T = dict(td={}, dw={}, fr={}, near=[])
        for t in C.MW8 + C.DMC4:
            St = S[t]
            ks, _ = C.td_kstars(St, C.ones(len(St["pairs"])), tau)
            ent = {}
            for rk in ("PT", "PC"):
                kst = {v: kfmt(ks[(rk, v)]) for v in C.TD_ROUND}
                for b in BASES:
                    g = C.td_G(ks, rk, b)
                    ent[f"{rk}|{b}"] = dict(G=int(g["G"][0]), G_lo=int(g["lo"][0]), G_hi=int(g["hi"][0]))
                    rows_G.append([tkey(tau), "TD-MPC2", t, rk, b, int(g["G"][0]), int(g["lo"][0]), int(g["hi"][0]),
                                   kst["det"], kst["detbc"], kst["shiter"], kst["detshift"]])
                ent[f"{rk}|kstar"] = kst
            T["td"][t] = ent
        for b in BASES:
            for lab, (rk, thr) in FR.items():
                pt = sum(T["td"][t][f"{rk}|{b}"]["G"] >= 1 for t in C.MW8)
                lo = sum(T["td"][t][f"{rk}|{b}"]["G_lo"] >= 1 for t in C.MW8)
                hi = sum(T["td"][t][f"{rk}|{b}"]["G_hi"] >= 1 for t in C.MW8)
                T["fr"][f"{lab}|{b}"] = dict(count=pt, count_lo=lo, count_hi=hi, threshold=thr,
                                             holds=pt >= thr, holds_lo=lo >= thr, holds_hi=hi >= thr)
                rows_fr.append([tkey(tau), lab, b, pt, lo, hi, thr, pt >= thr, lo >= thr, hi >= thr])
        for t in C.DW_TASKS:
            St = SD[t]
            ks, _ = C.dw_kstars(St, C.ones(len(St["eps"])), tau)
            ent = {}
            for rk in C.DW_RANGES[t]:
                g = C.dw_G(ks, rk)
                kst = {v: kfmt(ks[(rk, v)]) for v in (("det", "shiter") if rk == "RC" else ("det", "detbc", "shiter"))}
                ent[rk] = dict(G=int(g["G"][0]), G_lo=int(g["lo"][0]), G_hi=int(g["hi"][0]), kstar=kst)
                rows_G.append([tkey(tau), "DINO-WM", t, rk, "shiter", int(g["G"][0]), int(g["lo"][0]), int(g["hi"][0]),
                               kst["det"], kst.get("detbc", "-"), kst["shiter"], "-"])
            T["dw"][t] = ent
        if tau == STRICT:
            out["tau"][tkey(tau)] = T
            continue
        for (t, rk, v, k), (D, n) in sorted(td_D.items()):
            if abs(D - tau) <= 0.01 + C.TOL:
                T["near"].append(["TD-MPC2", t, rk, v, k, n, round(D, 6)])
                rows_near.append([tau, "TD-MPC2", t, rk, v, k, n, round(D, 6), round(D - tau, 6),
                                  abs(D - tau) <= C.TOL])
        for (t, rk, v, k), (D, n, pod) in sorted(dw_D.items()):
            if abs(D - tau) <= 0.01 + C.TOL:
                T["near"].append(["DINO-WM", t, rk, v, k, n, round(D, 6)])
                rows_near.append([tau, "DINO-WM", t, rk, v, k, n, round(D, 6), round(D - tau, 6),
                                  abs(D - tau) <= C.TOL])
        out["tau"][tkey(tau)] = T
    out["meta"]["taus"] = [tkey(t) for t in TAUS]
    C.dump_json(os.path.join(OUT, "r2.json"), out)
    C.write_csv(os.path.join(OUT, "r2_G.csv"),
                ["tau", "family", "task", "range", "dither_basis", "G", "G_lo", "G_hi", "kstar_det", "kstar_detbc",
                 "kstar_shiter", "kstar_detshift"], rows_G)
    C.write_csv(os.path.join(OUT, "r2_fr.csv"),
                ["tau", "label", "dither_basis", "count", "count_lo", "count_hi", "threshold_count", "holds",
                 "holds_lo", "holds_hi"], rows_fr)
    C.write_csv(os.path.join(OUT, "r2_near_tau.csv"),
                ["tau", "family", "task", "range", "rounding", "k", "n", "D", "D_minus_tau", "exactly_tau"], rows_near)
    # console digest
    for tau in TAUS:
        T = out["tau"][tkey(tau)]
        fr = " ".join(f"{lab}:{b}={x['count']}" + (f"[{x['count_lo']},{x['count_hi']}]" if x['count_lo'] != x['count_hi'] else "")
                      for lab in FR for b in BASES for x in [T["fr"][f"{lab}|{b}"]])
        pt = " ".join(f"{C.SHORT[t]}:{gfmt_d(T['td'][t]['PT|best'])}/{gfmt_d(T['td'][t]['PC|best'])}" for t in C.MW8)
        dm = " ".join(f"{C.SHORT[t]}:{gfmt_d(T['td'][t]['PT|best'])}/{gfmt_d(T['td'][t]['PC|best'])}" for t in C.DMC4)
        dw = " ".join(f"{C.SHORT[t]}-{rk}:{gfmt_d(T['dw'][t][rk])}" for t in C.DW_TASKS for rk in C.DW_RANGES[t])
        print(f"tau={tkey(tau)}: {fr}\n   MW G PT/PC best: {pt}\n   DMC: {dm}\n   DW: {dw}\n   near-tau cells: {len(T['near'])}")


def gfmt_d(x):
    return str(x["G"]) if x["G_lo"] == x["G_hi"] else f"{x['G']}[{x['G_lo']},{x['G_hi']}]"


if __name__ == "__main__":
    main()
