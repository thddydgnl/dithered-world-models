"""Model-competence check before the main run: executed-sequence ranking of the unablated model under the two input
crops (square = notebook/phaseB, train = DROID training config). Same windows and candidates in both prep dirs.
Usage: python crop_test.py <prep_square> <prep_train> <out_json>
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r25lib as L  # noqa: E402

_, pred, info = L.load_models(need_encoder=False)
out = {}
for tag, d in [("square", sys.argv[1]), ("train", sys.argv[2])]:
    P = torch.load(f"{d}/prep.pt", weights_only=False)
    meta = json.load(open(f"{d}/prep.json"))
    types = np.array(meta["cand_types"])
    R, BZ, BF, G, EX = [], [], [], {t: [] for t in sorted(set(types))}, []
    for wi in range(len(meta["windows"])):
        z = P["lat"][wi].cuda()
        zh1, zh2 = L.rollout(pred, z[0], P["states"][wi].numpy(), P["cands"][wi].numpy(), "fp16")
        E = np.stack([L.energy(zh1, z[1]).double().cpu().numpy(), L.energy(zh2, z[2]).double().cpu().numpy()])
        for k in range(2):
            R.append([k, int(1 + (E[k] < E[k, 0]).sum())])
            BZ.append(float(E[k, 0] < E[k, 1]))
            BF.append(float(E[k, 0] < E[k, 2]))
            EX.append(float(E[k, 0]))
            for t in G:
                G[t].append(float((E[k, types == t].mean() - E[k, 0]) / E[k].std()))
    R = np.array(R)
    out[tag] = dict(n_windows=len(meta["windows"]), exec_top1=float((R[:, 1] == 1).mean()),
                    exec_top1_k=[float((R[R[:, 0] == k, 1] == 1).mean()) for k in range(2)],
                    exec_rank_mean=float(R[:, 1].mean()), exec_rank_median=float(np.median(R[:, 1])),
                    exec_rank_k=[float(R[R[:, 0] == k, 1].mean()) for k in range(2)],
                    exec_beats_zero=float(np.mean(BZ)), exec_beats_flip=float(np.mean(BF)), energy_exec_mean=float(np.mean(EX)),
                    mean_energy_minus_exec_by_type_in_std=({t: float(np.mean(v)) for t, v in G.items()}))
json.dump(out, open(sys.argv[3], "w"), indent=1)
print(json.dumps(out, indent=1))
print("CROP_TEST_DONE")
