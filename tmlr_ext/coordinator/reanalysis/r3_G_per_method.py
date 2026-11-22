"""R3: G per dither method (E12, tau = 0.10): sh-iter alone and det+shift alone, PT and PC, MW and DMC.
DINO-WM has sh-iter only (its published G already uses sh-iter alone).

G_sh = min(k*(det), k*(det+BC)) - k*(sh-iter); G_ds = min(k*(det), k*(det+BC)) - k*(det+shift); G_best (published) =
the same det side minus min of the two. Point values plus the R1 bootstrap (same resamples, r1_kstar_G_bootstrap.td_boot:
RandomState(0), MW simple, DMC stratified by stage) for P(G >= 1) per task and P(Fr1 holds) / P(Fr2 holds) per method.
Output: r3.json, r3_G_per_method.csv.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import r1_kstar_G_bootstrap as R1  # noqa: E402

OUT = C.HERE
BASES = ("best", "shiter", "detshift")


def main():
    E = C.load_td()
    S = C.td_struct(E)
    res = R1.td_boot(S, "strat")
    out = dict(meta=dict(tau=C.TAU, B=C.NB, resampling="as R1 primary"), td={}, counts={}, fr=R1.fr_probs(res, "best"),
               fr_by_basis={b: R1.fr_probs(res, b) for b in BASES}, dw={})
    rows = []
    for t in C.MW8 + C.DMC4:
        ks, _ = C.td_kstars(S[t], C.ones(len(S[t]["pairs"])))
        ent = {}
        for rk in ("PT", "PC"):
            for b in BASES:
                g0 = C.td_G(ks, rk, b)
                gb = res[t]["G"][(rk, b)]
                ent[f"{rk}|{b}"] = dict(G=int(g0["G"][0]), det_side=int(g0["det"][0]), dither=int(g0["dith"][0]),
                                        P_ge1=float(np.mean(gb["G"] >= 1)), dist=C.dist(gb["G"]))
                rows.append(["TD-MPC2", "MW" if C.is_mw(t) else "DMC", t, rk, b, int(g0["G"][0]),
                             int(ks[(rk, "det")]["k"][0]), int(ks[(rk, "detbc")]["k"][0]),
                             int(ks[(rk, "shiter")]["k"][0]), int(ks[(rk, "detshift")]["k"][0]),
                             round(float(np.mean(gb["G"] >= 1)), 4),
                             ";".join(f"{k}:{v:.4f}" for k, v in C.dist(gb["G"]).items())])
        out["td"][t] = ent
    for unit, ts in (("MW", C.MW8), ("DMC", C.DMC4)):
        for rk in ("PT", "PC"):
            for b in BASES:
                gs = {C.SHORT[t]: out["td"][t][f"{rk}|{b}"]["G"] for t in ts}
                out["counts"][f"{unit}|{rk}|{b}"] = dict(G=gs, n_ge1=sum(g >= 1 for g in gs.values()), n=len(ts))
    cells, _ = C.load_dw()
    SD = C.dw_struct(cells)
    rd = R1.dw_boot(SD, "strat")
    for t in C.DW_TASKS:
        ks, _ = C.dw_kstars(SD[t], C.ones(len(SD[t]["eps"])))
        for rk in C.DW_RANGES[t]:
            g0 = C.dw_G(ks, rk)
            gb = rd[t]["G"][rk]
            out["dw"][f"{t}|{rk}"] = dict(G=int(g0["G"][0]), P_ge1=float(np.mean(gb["G"] >= 1)), dist=C.dist(gb["G"]))
            rows.append(["DINO-WM", "DW", t, rk, "shiter", int(g0["G"][0]), int(ks[(rk, "det")]["k"][0]),
                         int(ks[(rk, "detbc")]["k"][0]) if rk != "RC" else "-", int(ks[(rk, "shiter")]["k"][0]), "-",
                         round(float(np.mean(gb["G"] >= 1)), 4),
                         ";".join(f"{k}:{v:.4f}" for k, v in C.dist(gb["G"]).items())])
    C.dump_json(os.path.join(OUT, "r3.json"), out)
    C.write_csv(os.path.join(OUT, "r3_G_per_method.csv"),
                ["family", "unit", "task", "range", "dither_basis", "G", "kstar_det", "kstar_detbc", "kstar_shiter",
                 "kstar_detshift", "P_G_ge1_boot", "G_boot_dist"], rows)
    for key, x in out["counts"].items():
        print(f"{key:16s} G>=1 in {x['n_ge1']}/{x['n']}: {x['G']}")
    for b in BASES:
        for lab, x in out["fr_by_basis"][b].items():
            print(f"{lab} {b}: P(holds) {x['P_holds']:.4f}  count dist {x['count_dist']}")
    for key, x in out["dw"].items():
        print(f"DW {key}: G {x['G']} P(G>=1) {x['P_ge1']:.3f}")


if __name__ == "__main__":
    main()
