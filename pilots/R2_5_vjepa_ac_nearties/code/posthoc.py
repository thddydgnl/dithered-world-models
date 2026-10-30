"""Post-hoc analyses (사후 분석, not part of the pre-registered criteria). CPU only, reads eval_raw.npz.

1. decision regret: original energy of the intervened top-1 minus the original minimum, in units of the per-instance
   candidate energy std (how bad the new choice is under the unablated model)
2. distinct-candidate ranking: tau / top-1 flips among candidates that are not near-duplicates of the executed sequence
   (drop the 0.1, 0.25, 0.5 sigma Gaussian perturbations; keep exec, zero, flip, 1 and 2 sigma, other windows: 32 candidates)
3. common-mode vs differential energy change: d = E_cond - E_base over the 64 candidates; |mean(d)| and std(d) in units of
   std(E_base). std(d) is the rank-relevant part; at matched prediction change a candidate-consistent perturbation has a
   small std(d) relative to its mean shift
4. near-tie localisation with the normalized gap (gap / candidate std) quartiles
Usage: python posthoc.py <prep_dir> <run_dir>
"""
import csv
import json
import sys

import numpy as np
from scipy.stats import kendalltau

PREP, RUN = sys.argv[1], sys.argv[2]
ALPHAS = [0.05, 0.1, 0.2]
R = np.load(f"{RUN}/eval_raw.npz", allow_pickle=True)
E = R["E"]
names = [str(x) for x in R["cond_names"]]
ci = {n: i for i, n in enumerate(names)}
types = np.array(json.load(open(f"{PREP}/prep.json"))["cand_types"])
distinct = np.isin(types, ["exec", "zero", "flip", "gauss1.0", "gauss2.0", "other"])
nW = E.shape[0]
E0 = E[:, 0]
sd0 = E0.std(-1)                                   # [nW, 2]
srt = np.sort(E0, -1)
gapn = (srt[..., 1] - srt[..., 0]) / sd0
qn = np.zeros((nW, 2), int)
for k in range(2):
    qn[:, k] = np.searchsorted(np.quantile(gapn[:, k], [0.25, 0.5, 0.75]), gapn[:, k], side="right")


def one(c):
    reg, taud, flipd, cm, dif, flip, newtop_near = [], [], [], [], [], [], []
    fq = np.zeros((4, 2))
    for w in range(nW):
        for k in range(2):
            e0, e1 = E0[w, k], E[w, c, k]
            if np.isnan(e1).any():
                continue
            i0, i1 = np.argmin(e0), np.argmin(e1)
            reg.append((e0[i1] - e0[i0]) / sd0[w, k])
            f = float(i0 != i1)
            flip.append(f)
            if f:
                newtop_near.append(float(not distinct[i1]))
            fq[qn[w, k], 0] += f
            fq[qn[w, k], 1] += 1
            a, b = e0[distinct], e1[distinct]
            taud.append(kendalltau(a, b)[0])
            flipd.append(float(np.argmin(a) != np.argmin(b)))
            d = e1 - e0
            cm.append(abs(d.mean()) / sd0[w, k])
            dif.append(d.std() / sd0[w, k])
    return dict(regret_mean=float(np.mean(reg)), regret_p90=float(np.quantile(reg, 0.9)), regret_gt_half_sd=float(np.mean(np.array(reg) > 0.5)),
                rd_distinct=float(1 - np.mean(taud)), flip_distinct=float(np.mean(flipd)), flip_all=float(np.mean(flip)),
                newtop1_is_near_duplicate=float(np.mean(newtop_near)) if newtop_near else float("nan"),
                common_mode_over_sd=float(np.mean(cm)), differential_over_sd=float(np.mean(dif)),
                flip_by_normgap_quartile=(fq[:, 0] / np.maximum(fq[:, 1], 1)).tolist(),
                q1q4_normgap=float(((fq[0, 0] + 0.5) / (fq[0, 1] + 1)) / ((fq[3, 0] + 0.5) / (fq[3, 1] + 1))))


def fam(prefix, a):
    if prefix in ("F1", "F2"):
        return [f"{prefix}s{s}@{a}" for s in range(3) if f"{prefix}s{s}@{a}" in ci]
    return [f"{prefix}@{a}"] if f"{prefix}@{a}" in ci else []


out = {}
for a in ALPHAS:
    for p in ["A", "B", "C", "F1", "F2", "D1", "D2"]:
        ms = [one(ci[n]) for n in fam(p, a)]
        if ms:
            out[f"{p}@{a}"] = {k: float(np.nanmean([m[k] for m in ms])) if not isinstance(ms[0][k], list) else np.mean([m[k] for m in ms], 0).tolist()
                               for k in ms[0]}
for n in names:
    if n.startswith("NUM_"):
        out[n] = one(ci[n])
base_types = {t: float(np.mean([(E0[w, k, types == t].mean() - E0[w, k, 0]) / sd0[w, k] for w in range(nW) for k in range(2)]))
              for t in sorted(set(types))}
top1_type = {t: float(np.mean([types[np.argmin(E0[w, k])] == t for w in range(nW) for k in range(2)])) for t in sorted(set(types))}
res = dict(note="사후 분석 (post hoc)", n_windows=nW, baseline_energy_minus_exec_by_type_in_sd=base_types, baseline_top1_type_share=top1_type,
           baseline_gapnorm_median=float(np.median(gapn)), groups=out)
json.dump(res, open(f"{RUN}/posthoc.json", "w"), indent=1)
with open(f"{RUN}/table_posthoc.csv", "w", newline="") as f:
    wr = csv.writer(f)
    keys = ["regret_mean", "regret_p90", "regret_gt_half_sd", "rd_distinct", "flip_distinct", "flip_all", "newtop1_is_near_duplicate",
            "common_mode_over_sd", "differential_over_sd", "q1q4_normgap"]
    wr.writerow(["condition"] + keys + ["flip_normgap_Q1..Q4"])
    for k, v in out.items():
        wr.writerow([k] + [f"{v[x]:.4f}" for x in keys] + [np.round(v["flip_by_normgap_quartile"], 3).tolist()])
print(json.dumps(dict(base_types=base_types, top1_type=top1_type), indent=1))
for k, v in out.items():
    print(f"{k:12s} regret {v['regret_mean']:.3f} rd_dist {v['rd_distinct']:.3f} flip_dist {v['flip_distinct']:.3f} "
          f"near-dup new top1 {v['newtop1_is_near_duplicate']:.2f} common {v['common_mode_over_sd']:.3f} diff {v['differential_over_sd']:.3f} "
          f"q1q4n {v['q1q4_normgap']:.2f}")
print("POSTHOC_DONE")
