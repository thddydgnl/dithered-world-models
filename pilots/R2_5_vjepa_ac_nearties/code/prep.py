"""Select windows, build the 64 candidate 2-step action sequences per window and encode frames t, t+1, t+2.

Windows: context frame t, goals t+1 (k=1) and t+2 (k=2) at the 3.75 fps stride. Only "moving" windows
(|xyz step| >= 5 mm or |gripper step| >= 0.05 on the executed first action). Calibration windows come from episodes
with ep % 10 == 9, evaluation windows from the other episodes (<= 3 per episode, >= 3 frames apart).
Candidates (identical index layout in every window): 0 executed, 1 zero, 2 sign flip, 3..42 Gaussian perturbations of
the executed sequence (xyz + gripper dims, scales 0.1/0.25/0.5/1/2 x per-dim std, 8 each), 43..63 executed sequences of
21 random windows from other episodes. k=1 uses the first step of every candidate, k=2 the full 2-step sequence.
Usage: python prep.py <out_dir> <n_eval> <n_cal> [square|train]
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r25lib as L  # noqa: E402

OUT, N_EVAL, N_CAL = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
CROP = sys.argv[4] if len(sys.argv) > 4 else "square"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261021)

eps = {}
for f in sorted(glob.glob(f"{L.DATA}/ep*.npz")):
    z = np.load(f)
    ei = int(os.path.basename(f)[2:5])
    st = np.concatenate([z["cart"], z["grip"][:, None]], 1).astype(np.float64)  # [n_sub, 7]
    if len(st) < 4:
        continue
    eps[ei] = dict(f=f, states=st, actions=L.poses_to_diffs(st))  # actions[t] = step t -> t+1

elig = []  # (ep, t): t+2 exists and the first executed action moves
for ei, e in eps.items():
    a = e["actions"]
    for t in range(0, len(e["states"]) - 2):
        if np.linalg.norm(a[t, :3]) >= 0.005 or abs(a[t, 6]) >= 0.05:
            elig.append((ei, t))
A1 = np.stack([eps[ei]["actions"][t] for ei, t in elig])
sigma = A1.std(0)
L.log("episodes", len(eps), "eligible windows", len(elig), "sigma", np.round(sigma, 4).tolist())


def pick(pool, n, max_per_ep=3, min_gap=3):
    order = rng.permutation(len(pool))
    got, per = [], {}
    for i in order:
        ei, t = pool[i]
        if per.get(ei, 0) >= max_per_ep or any(e2 == ei and abs(t2 - t) < min_gap for e2, t2 in got):
            continue
        got.append((ei, t))
        per[ei] = per.get(ei, 0) + 1
        if len(got) == n:
            break
    return got


cal = pick([w for w in elig if w[0] % 10 == 9], N_CAL, max_per_ep=2)
ev = pick([w for w in elig if w[0] % 10 != 9], N_EVAL)
wins = [(ei, t, "cal") for ei, t in cal] + [(ei, t, "eval") for ei, t in ev]
L.log("calibration", len(cal), "eval", len(ev))

pert = np.zeros(7)
pert[L.PERT_DIMS] = 1.0
cands, states, meta = [], [], []
for wi, (ei, t, split) in enumerate(wins):
    e = eps[ei]
    exe = e["actions"][t:t + 2]  # [2, 7]
    C, types = [exe, np.zeros_like(exe), -exe], ["exec", "zero", "flip"]
    for s in L.GAUSS_SCALES:
        for _ in range(L.N_PER_SCALE):
            C.append(exe + rng.standard_normal((2, 7)) * sigma * s * pert)
            types.append(f"gauss{s}")
    others = [w for w in elig if w[0] != ei]
    for j in rng.choice(len(others), L.N_OTHER, replace=False):
        e2, t2 = others[j]
        C.append(eps[e2]["actions"][t2:t2 + 2])
        types.append("other")
    C = np.stack(C)
    assert C.shape == (L.NCAND, 2, 7), C.shape
    cands.append(C)
    states.append(e["states"][t])
    meta.append(dict(wi=wi, ep=ei, t=t, split=split, exec_xyz_mm=[float(np.linalg.norm(exe[i, :3]) * 1000) for i in range(2)],
                     exec_grip=[float(exe[i, 6]) for i in range(2)]))

# check: the official pose update applied to the executed action reproduces the next recorded state
s_next = L.compute_new_pose_np(np.stack(states), np.stack([c[0, 0] for c in cands]))
s_true = np.stack([eps[ei]["states"][t + 1] for ei, t, _ in wins])
pose_err = float(np.abs(s_next[:, :3] - s_true[:, :3]).max())

enc, _, info = L.load_models(need_encoder=True)
L.log("loaded", info)
tf = L.make_transform(CROP)
frames0 = np.load(eps[wins[0][0]]["f"])["img1"][wins[0][1]:wins[0][1] + 3]
x1, x2 = tf(frames0), tf(frames0)
tf_det = float((x1 - x2).abs().max())
lat = torch.zeros(len(wins), 3, L.NTOK, 1408)
cache = {}
for wi, (ei, t, _) in enumerate(wins):
    if ei not in cache:
        cache = {ei: np.load(eps[ei]["f"])["img1"]}
    lat[wi] = L.encode_frames(enc, tf, cache[ei][t:t + 3], prec="fp32").cpu()
    if wi % 50 == 0:
        L.log("encoded", wi, len(wins))
torch.save(dict(lat=lat, cands=torch.tensor(np.stack(cands)), states=torch.tensor(np.stack(states)), sigma=sigma.tolist()),
           f"{OUT}/prep.pt")
json.dump(dict(windows=meta, cand_types=types, sigma=sigma.tolist(), n_eligible=len(elig), pose_err_max=pose_err,
               transform_max_diff=tf_det, crop=CROP, load_info=info), open(f"{OUT}/prep.json", "w"), indent=1)
L.log("PREP_DONE", len(wins), "pose_err", pose_err, "transform diff", tf_det)
