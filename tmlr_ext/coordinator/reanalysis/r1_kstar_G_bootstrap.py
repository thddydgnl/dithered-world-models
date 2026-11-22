"""R1: bootstrap distribution of k* and G (E12), TD-MPC2 5M (MW, DMC) and DINO-WM (PushT PT/PC/RC, Wall, PointMaze).

Step 0 reproduces the published point values (every E12 cell's n, D, k*; G; Fr1 7/8, Fr2 7/8; DINO-WM G) and stops
if any differs. Resampling (see common.py): one count-weight matrix per task shared by every cell of the task, so the
resampled D at all k, roundings and ranges of a task use the same pairs; tasks are resampled independently.
  primary  TD: RandomState(0), tasks drawn in the order MW8 + DMC4; MW simple bootstrap of the 90 pairs, DMC stratified
           by stage (45 stage-1 pairs = episodes 0-14 x 3 checkpoints, 45 stage-2 pairs) -> a 45-pair cell sees exactly
           45 draws, all of them shared with the 90-pair cells.
           DW: RandomState(0), order PushT, PointMaze, Wall; PushT stratified (eval8 pairs 0-49 / 50-99), others simple.
  checks   (a) DMC / PushT unstratified: simple bootstrap over all pairs, half cells use the draws that fall in their
           half (random size); (b) PushT cluster bootstrap over the 10 eval8 overlap components (D1 auxiliary).
Output: r1.json, r1_G.csv, r1_kstar.csv, r1_fr.csv in this folder.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

OUT = C.HERE
BASES = ("best", "shiter", "detshift")
FR = {"Fr1": ("PT", 6), "Fr2": ("PC", 4)}          # 75% and 50% of the 8 MW tasks


def reproduce(S, SD):
    """Published point values; returns a list of problems (empty = reproduced)."""
    bad = []
    pub = {}
    with open(C.TD_E12) as f:
        for r in csv.DictReader(f):
            pub[(r["task"], r["range"], r["rounding"], int(r["k"]))] = (int(r["n"]), float(r["D"]), int(r["kstar"]))
    ncell = 0
    for t in C.MW8 + C.DMC4:
        St = S[t]
        W1 = C.ones(len(St["pairs"]))
        ks, Ds = C.td_kstars(St, W1)
        for (rk, v), curve in St["cells"].items():
            for j, k in enumerate(sorted(curve)):
                n, Dp, kp = pub[(t, rk, v, k)]
                ncell += 1
                if n != int(curve[k][0].sum()) or abs(Ds[(rk, v)][0, j] - Dp) > 1e-5 or int(ks[(rk, v)]["k"][0]) != kp:
                    bad.append(f"TD cell {t} {rk} {v} k{k}")
        for rk in ("PT", "PC"):
            g = C.td_G(ks, rk)
            if int(g["G"][0]) != C.PUB_TD_G[rk][t] or g["lo"][0] != g["hi"][0]:
                bad.append(f"TD G {t} {rk}: {int(g['G'][0])} vs {C.PUB_TD_G[rk][t]}")
    if ncell != len(pub):
        bad.append(f"TD cell count {ncell} vs {len(pub)}")
    for lab, (rk, thr) in FR.items():
        n_ok = sum(C.PUB_TD_G[rk][t] >= 1 for t in C.MW8)
        if n_ok != 7:
            bad.append(f"{lab} count {n_ok}")
    for t in C.DW_TASKS:
        St = SD[t]
        ks, _ = C.dw_kstars(St, C.ones(len(St["eps"])))
        for rk in C.DW_RANGES[t]:
            g = C.dw_G(ks, rk)
            if int(g["G"][0]) != C.PUB_DW_G[(t, rk)]:
                bad.append(f"DW G {t} {rk}: {int(g['G'][0])} vs {C.PUB_DW_G[(t, rk)]}")
    return bad, ncell


def summarize_G(g, valid=None):
    G, lo, hi = g["G"], g["lo"], g["hi"]
    if valid is not None:
        G, lo, hi = G[valid], lo[valid], hi[valid]
    return dict(n=int(len(G)), dist=C.dist(G), P_ge1=float(np.mean(G >= 1)), P_ge1_lo=float(np.mean(lo >= 1)),
                P_ge1_hi=float(np.mean(hi >= 1)), P_ge2=float(np.mean(G >= 2)), P_le0=float(np.mean(G <= 0)),
                q025=float(np.percentile(G, 2.5)), q50=float(np.percentile(G, 50)), q975=float(np.percentile(G, 97.5)),
                frac_censored=float(np.mean(lo != hi)))


def summarize_k(x):
    return dict(ks=[int(k) for k in x["ks"]], dist=C.dist(x["k"]), P_cens_up=float(np.mean(x["cens_up"])),
                P_cens_dn=float(np.mean(x["cens_dn"])))


def td_boot(S, scheme):
    rs = np.random.RandomState(0)
    W = C.td_weights(S, rs, scheme)
    res = {}
    for t in C.MW8 + C.DMC4:
        ks, _ = C.td_kstars(S[t], W[t])
        res[t] = dict(ks=ks, G={(rk, b): C.td_G(ks, rk, b) for rk in ("PT", "PC") for b in BASES})
    return res


def fr_probs(res, basis):
    out = {}
    for lab, (rk, thr) in FR.items():
        cnt = sum((res[t]["G"][(rk, basis)]["G"] >= 1).astype(int) for t in C.MW8)
        clo = sum((res[t]["G"][(rk, basis)]["lo"] >= 1).astype(int) for t in C.MW8)
        chi = sum((res[t]["G"][(rk, basis)]["hi"] >= 1).astype(int) for t in C.MW8)
        out[lab] = dict(range=rk, threshold_count=thr, P_holds=float(np.mean(cnt >= thr)),
                        P_holds_lo=float(np.mean(clo >= thr)), P_holds_hi=float(np.mean(chi >= thr)),
                        count_dist=C.dist(cnt), count_q025=float(np.percentile(cnt, 2.5)),
                        count_q975=float(np.percentile(cnt, 97.5)))
    return out


def dw_boot(SD, scheme, comp=None, tasks=C.DW_TASKS):
    rs = np.random.RandomState(0)
    if scheme == "cluster":
        St = SD["pusht"]
        W = {"pusht": C.w_cluster(rs, [comp[e] for e in St["eps"]])}
    else:
        W = C.dw_weights(SD, rs, scheme)
    res = {}
    for t in tasks:
        if t not in W:
            continue
        ks, Ds = C.dw_kstars(SD[t], W[t])
        valid = np.all(np.isfinite(np.concatenate([D for D in Ds.values()], 1)), 1)
        res[t] = dict(ks=ks, valid=valid, G={rk: C.dw_G(ks, rk) for rk in C.DW_RANGES[t]})
    return res


def main():
    E = C.load_td()
    S = C.td_struct(E)
    cells, gpus = C.load_dw()
    SD = C.dw_struct(cells)
    bad, ncell = reproduce(S, SD)
    print(f"step 0: reproduced {ncell} TD E12 cells (n, D, k*), TD G, Fr1/Fr2 7/8, DW G -> problems: {bad or 'none'}")
    if bad:
        sys.exit("published point values not reproduced; stopping")
    out = dict(meta=dict(B=C.NB, tau=C.TAU, tol=C.TOL, seed="numpy.random.RandomState(0)",
                         td_scheme="MW simple over 90 pairs; DMC stratified 45 (stage 1, ep 0-14) + 45 (stage 2)",
                         dw_scheme="PushT stratified 50 (pairs 0-49) + 50; PointMaze, Wall simple over 100 pairs",
                         censoring="k* boundary convention (lowest measured k / highest measured k + 1); G_lo/G_hi "
                                   "leave the unmeasured side open (k down to 1, up to 9)",
                         dw_pods=C.DW_PODS, dw_gpus=gpus, reproduced_cells=ncell), td={}, dw={}, fr={})
    rows_G, rows_k, rows_fr = [], [], []
    # ------------------------------------------------------------------ TD-MPC2 5M
    res = td_boot(S, "strat")
    res_simple = td_boot(S, "simple")          # identical MW draws; DMC unstratified (check a)
    W1 = {t: C.ones(len(S[t]["pairs"])) for t in S}
    for t in C.MW8 + C.DMC4:
        ks0, _ = C.td_kstars(S[t], W1[t])
        ent = {}
        for rk in ("PT", "PC"):
            for b in BASES:
                g0 = C.td_G(ks0, rk, b)
                sm = summarize_G(res[t]["G"][(rk, b)])
                sm["G_point"] = int(g0["G"][0])
                if not S[t]["mw"]:
                    sm["unstratified"] = summarize_G(res_simple[t]["G"][(rk, b)])
                ent[f"{rk}|{b}"] = sm
                if b == "best":
                    rows_G.append(["TD-MPC2", t, rk, b, "primary", sm["G_point"], round(sm["P_ge1"], 4),
                                   round(sm["P_ge1_lo"], 4), round(sm["P_ge1_hi"], 4), round(sm["P_ge2"], 4),
                                   sm["q025"], sm["q50"], sm["q975"], round(sm["frac_censored"], 4),
                                   ";".join(f"{k}:{v:.4f}" for k, v in sm["dist"].items())])
                    if not S[t]["mw"]:
                        u = sm["unstratified"]
                        rows_G.append(["TD-MPC2", t, rk, b, "unstratified", sm["G_point"], round(u["P_ge1"], 4),
                                       round(u["P_ge1_lo"], 4), round(u["P_ge1_hi"], 4), round(u["P_ge2"], 4),
                                       u["q025"], u["q50"], u["q975"], round(u["frac_censored"], 4),
                                       ";".join(f"{k}:{v:.4f}" for k, v in u["dist"].items())])
            for v in C.TD_ROUND:
                sk = summarize_k(res[t]["ks"][(rk, v)])
                sk["k_point"] = int(ks0[(rk, v)]["k"][0])
                ent[f"kstar|{rk}|{v}"] = sk
                rows_k.append(["TD-MPC2", t, rk, v, " ".join(map(str, sk["ks"])), sk["k_point"],
                               ";".join(f"{k}:{p:.4f}" for k, p in sk["dist"].items()),
                               round(sk["P_cens_up"], 4), round(sk["P_cens_dn"], 4)])
        out["td"][t] = ent
    for b in BASES:
        out["fr"][b] = fr_probs(res, b)
        for lab, x in out["fr"][b].items():
            pt = sum(C.td_G(C.td_kstars(S[t], W1[t])[0], x["range"], b)["G"][0] >= 1 for t in C.MW8)
            x["count_point"] = int(pt)
            rows_fr.append([lab, b, x["range"], x["threshold_count"], int(pt), round(x["P_holds"], 4),
                            round(x["P_holds_lo"], 4), round(x["P_holds_hi"], 4), x["count_q025"], x["count_q975"],
                            ";".join(f"{k}:{v:.4f}" for k, v in x["count_dist"].items())])
    # ------------------------------------------------------------------ DINO-WM
    comp = C.dw_components()
    rd = dw_boot(SD, "strat")
    rd_simple = dw_boot(SD, "simple", tasks=["pusht"])
    rd_clu = dw_boot(SD, "cluster", comp=comp, tasks=["pusht"])
    for t in C.DW_TASKS:
        ks0, _ = C.dw_kstars(SD[t], C.ones(len(SD[t]["eps"])))
        ent = {}
        for rk in C.DW_RANGES[t]:
            g0 = C.dw_G(ks0, rk)
            sm = summarize_G(rd[t]["G"][rk], rd[t]["valid"])
            sm["G_point"] = int(g0["G"][0])
            sm["n_invalid_resamples"] = int((~rd[t]["valid"]).sum())
            variants = [("primary", sm)]
            if t == "pusht":
                sm["unstratified"] = summarize_G(rd_simple[t]["G"][rk], rd_simple[t]["valid"])
                sm["cluster"] = summarize_G(rd_clu[t]["G"][rk], rd_clu[t]["valid"])
                sm["cluster"]["n_invalid_resamples"] = int((~rd_clu[t]["valid"]).sum())
                variants += [("unstratified", sm["unstratified"]), ("cluster", sm["cluster"])]
            ent[rk] = sm
            for lab, u in variants:
                rows_G.append(["DINO-WM", t, rk, "shiter", lab, sm["G_point"], round(u["P_ge1"], 4),
                               round(u["P_ge1_lo"], 4), round(u["P_ge1_hi"], 4), round(u["P_ge2"], 4),
                               u["q025"], u["q50"], u["q975"], round(u["frac_censored"], 4),
                               ";".join(f"{k}:{v:.4f}" for k, v in u["dist"].items())])
            for v in (("det", "shiter") if rk == "RC" else ("det", "detbc", "shiter")):
                sk = summarize_k(rd[t]["ks"][(rk, v)])
                sk["k_point"] = int(ks0[(rk, v)]["k"][0])
                ent[f"kstar|{rk}|{v}"] = sk
                rows_k.append(["DINO-WM", t, rk, v, " ".join(map(str, sk["ks"])), sk["k_point"],
                               ";".join(f"{k}:{p:.4f}" for k, p in sk["dist"].items()),
                               round(sk["P_cens_up"], 4), round(sk["P_cens_dn"], 4)])
        out["dw"][t] = ent
    C.dump_json(os.path.join(OUT, "r1.json"), out)
    C.write_csv(os.path.join(OUT, "r1_G.csv"),
                ["family", "task", "range", "dither_basis", "scheme", "G_point", "P_G_ge1", "P_G_ge1_lo", "P_G_ge1_hi",
                 "P_G_ge2", "G_q025", "G_q50", "G_q975", "frac_resamples_censored", "G_dist"], rows_G)
    C.write_csv(os.path.join(OUT, "r1_kstar.csv"),
                ["family", "task", "range", "rounding", "measured_k", "kstar_point", "kstar_dist", "P_censored_up",
                 "P_censored_down"], rows_k)
    C.write_csv(os.path.join(OUT, "r1_fr.csv"),
                ["label", "dither_basis", "range", "threshold_count", "count_point", "P_holds", "P_holds_lo",
                 "P_holds_hi", "count_q025", "count_q975", "count_dist"], rows_fr)
    # console digest
    for t in C.MW8 + C.DMC4:
        for rk in ("PT", "PC"):
            x = out["td"][t][f"{rk}|best"]
            print(f"TD {C.SHORT[t]:12s} {rk} G={x['G_point']} P(G>=1)={x['P_ge1']:.3f} [{x['P_ge1_lo']:.3f},{x['P_ge1_hi']:.3f}] "
                  f"dist={ {k: round(v, 3) for k, v in x['dist'].items()} }")
    for b in BASES:
        for lab, x in out["fr"][b].items():
            print(f"{lab} basis={b}: count {x['count_point']}/8, P(holds)={x['P_holds']:.3f} "
                  f"[{x['P_holds_lo']:.3f},{x['P_holds_hi']:.3f}] dist={ {k: round(v, 3) for k, v in x['count_dist'].items()} }")
    for t in C.DW_TASKS:
        for rk in C.DW_RANGES[t]:
            x = out["dw"][t][rk]
            extra = ""
            if t == "pusht":
                extra = f" | unstrat {x['unstratified']['P_ge1']:.3f} | cluster {x['cluster']['P_ge1']:.3f}"
            print(f"DW {t:9s} {rk} G={x['G_point']} P(G>=1)={x['P_ge1']:.3f} [{x['P_ge1_lo']:.3f},{x['P_ge1_hi']:.3f}] "
                  f"P(G>=2)={x['P_ge2']:.3f} dist={ {k: round(v, 3) for k, v in x['dist'].items()} }{extra}")


if __name__ == "__main__":
    main()
