"""R4 [post hoc]: DINO-WM, E1 + eval8 pooled for the three E11 cells PushT 4-bit, PointMaze 3-bit, Wall 3-bit.

Wall 3-bit was an exploratory E1 cell. The two splits ran on different GPUs (E1: RTX A5000, eval8: RTX 4090) and use
different pairs, so they are not paired: each split is bootstrapped on its own pairs (10,000, RandomState(0), the
statistics of a split share one index matrix) and the splits are combined as independent estimates.
Statistics per split: D_det = success(none) - success(det) and red(sh-iter) = (D_det - D_sh) / D_det (resamples with
D_det* <= 0 dropped, as in the published red CIs). Variance = variance of the split's bootstrap replicates.
Pooling: inverse-variance (fixed effect) estimate, SE = sqrt(1 / sum w), 95% CI = estimate +- 1.96 SE; Cochran's Q
(df = 1, p from chi-square), I^2 = max(0, (Q - df) / Q).
Sensitivities: (a) red as the ratio of the pooled (D_det - D_sh) to the pooled D_det (point only); (b) PushT eval8 with
the D1 cluster-bootstrap variance (10 overlap components; E1's eval6 has no component file here, so pair level there).
Output: r4.json, r4_pooled.csv.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

OUT = C.HERE
CELLS = [("pusht", 4, "dw.1", "confirmatory"), ("pointmaze", 3, "dw.3", "confirmatory"), ("wall", 3, "dw.4", "exploratory")]


def chi2_sf_df1(q):
    return math.erfc(math.sqrt(max(q, 0.0) / 2.0))


def split_stats(Sn, Sd, Ss, W):
    """W: (B, n) count weights (rows sum to n). Returns point and replicate arrays."""
    n = len(Sn)
    Dd = float(np.mean(Sn - Sd))
    Dsh = float(np.mean(Sn - Ss))
    tot = W.sum(1)
    Ddb = (W @ (Sn - Sd)) / tot
    Dsb = (W @ (Sn - Ss)) / tot
    with np.errstate(invalid="ignore", divide="ignore"):
        redb = np.where(Ddb > 0, (Ddb - Dsb) / np.where(Ddb > 0, Ddb, 1), np.nan)
    red = (Dd - Dsh) / Dd if Dd > 0 else float("nan")
    return dict(n=n, none=float(Sn.mean()), D_det=Dd, D_sh=Dsh, red=red, Ddb=Ddb, Dsb=Dsb, redb=redb)


def summ(est, rep):
    r = rep[np.isfinite(rep)]
    return dict(est=float(est), se=float(np.std(r, ddof=1)), var=float(np.var(r, ddof=1)), ci=C.ci95(r),
                n_valid=int(len(r)), n_dropped=int(len(rep) - len(r)))


def pool(items):
    th = np.array([x["est"] for x in items])
    w = np.array([1.0 / x["var"] for x in items])
    est = float(np.sum(w * th) / np.sum(w))
    se = float(math.sqrt(1.0 / np.sum(w)))
    Q = float(np.sum(w * (th - est) ** 2))
    df = len(items) - 1
    return dict(est=est, se=se, ci=[est - 1.96 * se, est + 1.96 * se], Q=Q, df=df, p_Q=chi2_sf_df1(Q),
                I2=float(max(0.0, (Q - df) / Q)) if Q > 0 else 0.0, weights=(w / w.sum()).tolist())


def main():
    R1, e1_gpus = C.load_e1()
    cells, dw_gpus = C.load_dw()
    comp = C.dw_components()
    out = dict(meta=dict(label="[post hoc]", B=C.NB, seed="RandomState(0) per split", e1_gpus=e1_gpus,
                         eval8_gpus={p: dw_gpus.get(p) for p in ("dw.1", "dw.3", "dw.4")},
                         note="splits unpaired (different pairs and GPUs); inverse-variance fixed-effect pooling"),
               cells={})
    rows = []
    for task, k, pod, role in CELLS:
        # E1 split (pairs = E1 eval pairs; PushT eval6, Wall/PointMaze E1 eval split)
        eps1 = sorted(e for (t, v, kk, e) in R1 if t == task and v == "det" and kk == k
                      and (t, "none", 0, e) in R1 and (t, "shiter", k, e) in R1)
        Sn1 = np.array([R1[(task, "none", 0, e)] for e in eps1])
        Sd1 = np.array([R1[(task, "det", k, e)] for e in eps1])
        Ss1 = np.array([R1[(task, "shiter", k, e)] for e in eps1])
        W1 = C.counts(np.random.RandomState(0).randint(0, len(eps1), size=(C.NB, len(eps1))), len(eps1))
        a = split_stats(Sn1, Sd1, Ss1, W1)
        # eval8 split (E11 cell of its pod)
        nn = C.dw_none(cells, pod, task)
        cd = cells[(pod, task, "eval8", "main", "det", k, "cem")]
        cs = cells[(pod, task, "eval8", "main", "shiter", k, "cem")]
        eps8 = sorted(set(nn) & set(cd) & set(cs))
        Sn8 = np.array([nn[e] for e in eps8])
        Sd8 = np.array([cd[e] for e in eps8])
        Ss8 = np.array([cs[e] for e in eps8])
        W8 = C.counts(np.random.RandomState(0).randint(0, len(eps8), size=(C.NB, len(eps8))), len(eps8))
        b = split_stats(Sn8, Sd8, Ss8, W8)
        ent = dict(role_in_E1=role, k=k, eval8_pod=pod, n_E1=a["n"], n_eval8=b["n"],
                   none_E1=a["none"], none_eval8=b["none"], D_sh_E1=a["D_sh"], D_sh_eval8=b["D_sh"])
        for stat, ka, kb in (("D_det", "Ddb", "Ddb"), ("red_shiter", "redb", "redb")):
            ea = a["D_det"] if stat == "D_det" else a["red"]
            eb = b["D_det"] if stat == "D_det" else b["red"]
            sa, sb = summ(ea, a[ka]), summ(eb, b[kb])
            pl = pool([sa, sb])
            ent[stat] = dict(E1=sa, eval8=sb, pooled=pl)
            for lab, x in (("E1", sa), ("eval8", sb)):
                rows.append([task, k, role, stat, lab, "pair", round(x["est"], 4), round(x["se"], 4),
                             round(x["ci"][0], 4), round(x["ci"][1], 4), x["n_valid"], x["n_dropped"], "", "", "", ""])
            rows.append([task, k, role, stat, "pooled", "pair", round(pl["est"], 4), round(pl["se"], 4),
                         round(pl["ci"][0], 4), round(pl["ci"][1], 4), "", "", round(pl["Q"], 4), round(pl["p_Q"], 4),
                         round(pl["I2"], 4), ";".join(f"{w:.3f}" for w in pl["weights"])])
        # (a) ratio of pooled differences
        dd = summ(a["D_det"] - a["D_sh"], a["Ddb"] - a["Dsb"]), summ(b["D_det"] - b["D_sh"], b["Ddb"] - b["Dsb"])
        pdd = pool(list(dd))
        ent["red_ratio_of_pooled"] = dict(diff_pooled=pdd["est"], D_det_pooled=ent["D_det"]["pooled"]["est"],
                                          red=pdd["est"] / ent["D_det"]["pooled"]["est"], Q_diff=pdd["Q"],
                                          p_Q_diff=pdd["p_Q"])
        # (b) PushT eval8 cluster variance
        if task == "pusht":
            Wc = C.w_cluster(np.random.RandomState(0), [comp[e] for e in eps8])
            bc = split_stats(Sn8, Sd8, Ss8, Wc)
            sens = {}
            for stat, kb in (("D_det", "Ddb"), ("red_shiter", "redb")):
                sa = ent[stat]["E1"]
                sb = summ(b["D_det"] if stat == "D_det" else b["red"], bc[kb])
                pl = pool([sa, sb])
                sens[stat] = dict(eval8_cluster=sb, pooled=pl)
                rows.append([task, k, role, stat, "eval8", "cluster", round(sb["est"], 4), round(sb["se"], 4),
                             round(sb["ci"][0], 4), round(sb["ci"][1], 4), sb["n_valid"], sb["n_dropped"], "", "", "", ""])
                rows.append([task, k, role, stat, "pooled", "E1 pair + eval8 cluster", round(pl["est"], 4),
                             round(pl["se"], 4), round(pl["ci"][0], 4), round(pl["ci"][1], 4), "", "", round(pl["Q"], 4),
                             round(pl["p_Q"], 4), round(pl["I2"], 4), ";".join(f"{w:.3f}" for w in pl["weights"])])
            ent["sensitivity_eval8_cluster"] = sens
        out["cells"][task] = ent
    C.dump_json(os.path.join(OUT, "r4.json"), out)
    C.write_csv(os.path.join(OUT, "r4_pooled.csv"),
                ["task", "k", "E1_role", "statistic", "split", "variance_from", "estimate", "se", "ci_lo", "ci_hi",
                 "n_boot_valid", "n_boot_dropped", "Q", "p_Q", "I2", "weights_E1_eval8"], rows)
    for task, ent in out["cells"].items():
        for stat in ("D_det", "red_shiter"):
            x = ent[stat]
            p = x["pooled"]
            print(f"{task:9s} {stat:10s} E1 {x['E1']['est']:+.3f} (se {x['E1']['se']:.3f}, CI {x['E1']['ci'][0]:+.3f},{x['E1']['ci'][1]:+.3f}, drop {x['E1']['n_dropped']}) "
                  f"eval8 {x['eval8']['est']:+.3f} (se {x['eval8']['se']:.3f}, CI {x['eval8']['ci'][0]:+.3f},{x['eval8']['ci'][1]:+.3f}, drop {x['eval8']['n_dropped']}) "
                  f"pooled {p['est']:+.3f} [{p['ci'][0]:+.3f},{p['ci'][1]:+.3f}] Q {p['Q']:.2f} p {p['p_Q']:.4f} I2 {100 * p['I2']:.0f}% w {p['weights'][0]:.2f}/{p['weights'][1]:.2f}")
        print(f"          red ratio-of-pooled {ent['red_ratio_of_pooled']['red']:.3f}; none E1 {ent['none_E1']:.2f} eval8 {ent['none_eval8']:.2f}")
        if "sensitivity_eval8_cluster" in ent:
            for stat, x in ent["sensitivity_eval8_cluster"].items():
                p = x["pooled"]
                print(f"          cluster: {stat} eval8 se {x['eval8_cluster']['se']:.3f} CI {x['eval8_cluster']['ci']} -> pooled {p['est']:+.3f} "
                      f"[{p['ci'][0]:+.3f},{p['ci'][1]:+.3f}] Q {p['Q']:.2f} p {p['p_Q']:.4f} I2 {100 * p['I2']:.0f}%")


if __name__ == "__main__":
    main()
