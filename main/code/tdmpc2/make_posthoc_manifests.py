"""Post-hoc manifests of the TD-MPC2 part (DESIGN_main v5.3 section 10, user-approved 2026-10-24; descriptive checks,
no pre-registered verdict depends on them).

qatnone (v5.3 10.2): the E5 QAT weights (11 cells x checkpoints 1-3) evaluated WITHOUT activation quantization
  (variant none) on the E5 evaluation seeds 60000-60029 (90 pairs). Same job form as the E5 QAT evaluation
  (make_manifest.m_e5qateval: 'qat': True makes runner.job_episodes load <out>/qat/<task>_s<ckpt>_k<k>.pt, so k stays
  the cell's QAT k); only the variant is 'none' and the ptag is e5qatnone. The output folder gets the usual symlinks
  scales / calib and qat -> out_e5/qat (read only).
usage: python make_posthoc_manifests.py --out ~/pmx/A/tdmpc2_main/posthoc/manifests
"""
import argparse
import json
import os
from collections import Counter

import make_manifest as MM


def m_qatnone():
    jobs = []
    for ch_i, eps in enumerate(MM.chunks(30, 5)):
        for t in MM.dmc_first(list(MM.E5_QAT)):
            k = MM.E5_QAT[t]
            for c in (1, 2, 3):
                jobs.append(MM.ep_job("e5", t, c, k, "none", eps, ch_i, ptag="e5qatnone", instr_on=False,
                                      extra={"qat": True}))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = m_qatnone()
    ids = [j["id"] for j in jobs]
    assert len(ids) == len(set(ids)), "duplicate job ids"
    p = f"{a.out}/qatnone.jsonl"
    with open(p, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"{len(jobs)} jobs -> {p}: {dict(Counter(j['task'] for j in jobs))}")


if __name__ == "__main__":
    main()
