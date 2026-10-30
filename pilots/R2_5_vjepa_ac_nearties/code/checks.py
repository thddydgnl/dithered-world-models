"""Smoke checks: precision agreement and throughput, determinism, hook no-op, official compute_new_pose replica.
Usage: python checks.py <prep_dir> <out_json>
"""
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import kendalltau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r25lib as L  # noqa: E402

PREP, OUT = sys.argv[1], sys.argv[2]
P = torch.load(f"{PREP}/prep.pt", weights_only=False)
meta = json.load(open(f"{PREP}/prep.json"))
ev = [w["wi"] for w in meta["windows"] if w["split"] == "eval"]
lat, cands, states = P["lat"], P["cands"].numpy(), P["states"].numpy()
_, pred, info = L.load_models(need_encoder=False)
res = dict(load_info=info)

# official compute_new_pose vs replica
try:
    from notebooks.utils.mpc_utils import compute_new_pose as cnp_official
    rng = np.random.default_rng(0)
    pose = np.concatenate([rng.normal(0, 0.3, (50, 3)), rng.uniform(-3, 3, (50, 3)), rng.uniform(0, 1, (50, 1))], 1)
    act = np.concatenate([rng.normal(0, 0.03, (50, 3)), rng.normal(0, 0.1, (50, 3)), rng.normal(0, 0.3, (50, 1))], 1)
    off = cnp_official(torch.tensor(pose)[:, None], torch.tensor(act)[:, None])[:, 0].numpy()
    res["compute_new_pose_maxdiff"] = float(np.abs(off - L.compute_new_pose_np(pose, act)).max())
except Exception as e:  # noqa: BLE001
    res["compute_new_pose_maxdiff"] = f"import failed: {e!r}"


def energies(wi, prec):
    z = lat[wi].cuda()
    zh1, zh2 = L.rollout(pred, z[0], states[wi], cands[wi], prec)
    return np.stack([L.energy(zh1, z[1]).double().cpu().numpy(), L.energy(zh2, z[2]).double().cpu().numpy()])


for prec in ["fp32", "tf32", "bf16", "fp16"]:  # warmup
    energies(ev[0], prec)
torch.cuda.synchronize()
E, T = {}, {}
for prec in ["fp32", "tf32", "bf16", "fp16"]:
    t0 = time.time()
    E[prec] = np.stack([energies(wi, prec) for wi in ev])  # [nW, 2, 64]
    torch.cuda.synchronize()
    T[prec] = (time.time() - t0) / len(ev)
res["s_per_window_rollout"] = T
cmp = {}
for prec in ["tf32", "bf16", "fp16"]:
    taus, top1, rel = [], [], []
    for w in range(len(ev)):
        for k in range(2):
            a, b = E["fp32"][w, k], E[prec][w, k]
            taus.append(kendalltau(a, b)[0])
            top1.append(int(np.argmin(a) == np.argmin(b)))
            rel.append(float(np.abs(a - b).max() / np.abs(a).mean()))
    cmp[prec] = dict(mean_1_minus_tau=float(1 - np.mean(taus)), top1_agree=float(np.mean(top1)), max_rel_diff=float(np.max(rel)),
                     any_nan=bool(np.isnan(E[prec]).any()))
res["vs_fp32"] = cmp
# determinism: repeat
res["repeat_identical"] = {p: bool(np.array_equal(E[p], np.stack([energies(wi, p) for wi in ev]))) for p in ["tf32", "bf16"]}
# hook no-op: with Intervention hooks registered (mode None) vs without
iv = L.Intervention(pred)
with_hooks = np.stack([energies(wi, "tf32") for wi in ev])
for h in iv.handles:
    h.remove()
res["hook_noop_identical_tf32"] = bool(np.array_equal(with_hooks, E["tf32"]))
# energy structure in the baseline (tf32): executed rank, gap, spread by candidate type
types = meta["cand_types"]
ex_rank = [[int((E["tf32"][w, k] < E["tf32"][w, k, 0]).sum() + 1) for k in range(2)] for w in range(len(ev))]
gaps = [[float(np.sort(E["tf32"][w, k])[1] - np.sort(E["tf32"][w, k])[0]) for k in range(2)] for w in range(len(ev))]
by_type = {}
for t in sorted(set(types)):
    idx = [i for i, x in enumerate(types) if x == t]
    by_type[t] = [float(E["tf32"][:, k, idx].mean() - E["tf32"][:, k, 0].mean()) for k in range(2)]
res["baseline_tf32"] = dict(exec_rank=ex_rank, top1_gap=gaps, energy_minus_exec_by_type=by_type,
                            energy_exec=E["tf32"][:, :, 0].tolist(), energy_range=[[float(E["tf32"][w, k].min()), float(E["tf32"][w, k].max())]
                                                                                for w in range(len(ev)) for k in range(2)])
res["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "load_info"}, indent=1))
print("CHECKS_DONE")
