"""POST-HOC (decided after seeing the E1 TD-MPC2 results; not part of the pre-registered judgements): decomposition of
the TD-MPC2 OEI into
  sel_last  = e(executed elite) - mean e(last-iteration population)        (selection within the last evaluation)
  drift     = mean e(last-iteration population) - mean e(first-iteration sampled candidates)
              (the population moved to where the model is optimistic: persistent exploitation and/or region effect)
  OEI       = sel_last + drift  (exactly, per plan, since oei = e_chosen - f_e_rand)
plus oei_first = elite-weighted e of the first iteration - mean e of its sampled candidates (one-shot selection).
Episode means of the per-plan values (records: oei_mean e_chosen, l_e_rand, f_e_rand, oei_first), mean over the 90
pairs, paired bootstrap 95% CI (RandomState(0), as analyze_e1).
usage: python analyze_e1_posthoc.py --res <out> --outdir <dir>
"""
import argparse
import json
import os

import numpy as np

from analyze_e1 import CONF, VARS, boot_idx, ci95, is_mw, load, pairs_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    E, _, _ = load(a.res, "eval")
    out, lines = {}, []
    lines.append("| 칸 | 변형 | OEI | 마지막 반복 안 선택 sel_last [95% CI] | 모집단 이동 drift [95% CI] | 첫 반복 선택 oei_first | 정규화 OEI (÷ none 후보 가치 std) |")
    lines.append("|---|---|---|---|---|---|---|")
    for t, ks in CONF.items():
        sd = [E[k]["oei"].get("f_sd_v0") for k in E if k[0] == t and k[3] == "none"]
        sd = float(np.mean([x for x in sd if x is not None]))
        for k in ks:
            prs = pairs_of(E, t, k, VARS)
            idx = boot_idx(len(prs))
            for v in VARS:
                g = lambda kk: np.array([E[(t, c, k, v, e)]["oei"][kk] for c, e in prs], float)
                ec, lr, fr, of = g("e_chosen"), g("l_e_rand"), g("f_e_rand"), g("oei_first")
                sel, drift = ec - lr, lr - fr
                r = {"oei": float((ec - fr).mean()), "sel_last": float(sel.mean()), "sel_last_ci": ci95(sel[idx].mean(1)),
                     "drift": float(drift.mean()), "drift_ci": ci95(drift[idx].mean(1)), "oei_first": float(of.mean()),
                     "oei_norm": float((ec - fr).mean() / sd)}
                out[f"{t}:{k}:{v}"] = r
                lines.append(f"| {t} k={k} | {v} | {r['oei']:.2f} | {r['sel_last']:.2f} [{r['sel_last_ci'][0]:.2f}, "
                             f"{r['sel_last_ci'][1]:.2f}] | {r['drift']:.2f} [{r['drift_ci'][0]:.2f}, {r['drift_ci'][1]:.2f}] | "
                             f"{r['oei_first']:.2f} | {r['oei_norm']:.3f} |")
    os.makedirs(a.outdir, exist_ok=True)
    json.dump(out, open(f"{a.outdir}/e1_tdmpc2_posthoc_oei_decomposition.json", "w"), indent=1)
    open(f"{a.outdir}/e1_tdmpc2_posthoc_oei_decomposition.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(l for l in lines if any(s in l for s in ("mw-push k=3", "mw-pick-place k=3", "cheetah-run k=5", "| 칸", "|---"))))


if __name__ == "__main__":
    main()
