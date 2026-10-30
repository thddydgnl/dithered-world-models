"""사후/부가 (coordinator add-on 2): massive-coordinate check on V-JEPA 2-AC planning latents.

(a) encoder output tokens: per-coordinate mean |z| and variance over all tokens of the prepared frames (layer-normed,
    exactly the planning latents) and of the raw pre-LN encoder output for a subset of frames
(b) predictor output tokens (pre-LN and post-LN) of unablated candidate rollouts (k=1 and k=2)
(c) share of the between-candidate energy variance carried by the top-1% coordinates (14 of 1408) by variance:
    E_c = sum_d e_cd with e_cd = mean_tokens |zh_c - z_goal|_d / 1408; covariance share sum_{d in S} Cov_c(e_d, E) / Var_c(E)
    (additive, sums to 1 over all coordinates) and own share Var_c(sum_{d in S} e_d) / Var_c(E); random-14 baseline
Usage: python massive.py <prep_dir> <out_json> [n_windows=40] [n_raw_windows=32]
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r25lib as L  # noqa: E402

PREP, OUT = sys.argv[1], sys.argv[2]
NWIN = int(sys.argv[3]) if len(sys.argv) > 3 else 40
NRAW = int(sys.argv[4]) if len(sys.argv) > 4 else 32
P = torch.load(f"{PREP}/prep.pt", weights_only=False)
meta = json.load(open(f"{PREP}/prep.json"))
lat, cands, states = P["lat"], P["cands"].numpy(), P["states"].numpy()
D = lat.shape[-1]
K = max(1, int(round(0.01 * D)))


class Acc:
    def __init__(self):
        self.s1 = torch.zeros(D, dtype=torch.float64, device="cuda")
        self.s2 = torch.zeros(D, dtype=torch.float64, device="cuda")
        self.sa = torch.zeros(D, dtype=torch.float64, device="cuda")
        self.n = 0

    def add(self, x):
        x = x.reshape(-1, D).float()
        self.s1 += x.sum(0).double()
        self.s2 += (x * x).sum(0).double()
        self.sa += x.abs().sum(0).double()
        self.n += x.shape[0]

    def report(self):
        m = self.sa / self.n
        v = self.s2 / self.n - (self.s1 / self.n) ** 2
        om, ov = torch.argsort(m, descending=True), torch.argsort(v, descending=True)
        return dict(n_tokens=int(self.n), mean_abs_max_over_median=float(m.max() / m.median()), var_max_over_median=float(v.max() / v.median()),
                    top10_mean_abs_idx=om[:10].tolist(), top10_mean_abs_vals=[round(float(m[i]), 4) for i in om[:10]],
                    top10_var_idx=ov[:10].tolist(), top10_var_vals=[round(float(v[i]), 4) for i in ov[:10]],
                    median_mean_abs=float(m.median()), median_var=float(v.median()),
                    n_mean_abs_gt10x_median=int((m > 10 * m.median()).sum()), n_var_gt10x_median=int((v > 10 * v.median()).sum()),
                    top1pct_var_share_of_total_var=float(v[ov[:K]].sum() / v.sum())), v


res = dict(note="사후/부가 분석 (coordinator add-on)", D=D, top_k=K)
# (a) layer-normed planning latents of all prepared frames
acc = Acc()
for i in range(lat.shape[0]):
    acc.add(lat[i].cuda())
res["encoder_ln"], v_enc = acc.report()
res["encoder_ln"]["n_frames"] = int(lat.shape[0] * 3)

enc, pred, info = L.load_models(need_encoder=True)
tf = L.make_transform("square")
acc = Acc()
for wi in range(min(NRAW, lat.shape[0])):
    w = meta["windows"][wi]
    frames = np.load(f"{L.DATA}/ep{w['ep']:03d}.npz")["img1"][w["t"]:w["t"] + 3]
    x = tf(frames).permute(1, 0, 2, 3).unsqueeze(2).repeat(1, 1, 2, 1, 1).cuda()
    with torch.no_grad(), L.precision("fp32"):
        acc.add(enc(x))
res["encoder_raw_preLN"], _ = acc.report()
res["encoder_raw_preLN"]["n_frames"] = 3 * min(NRAW, lat.shape[0])
del enc
torch.cuda.empty_cache()

# (b, c) predictor outputs of unablated rollouts and the energy-variance decomposition
ev = [w["wi"] for w in meta["windows"] if w["split"] == "eval"][:NWIN]
acc_pre, acc_post = Acc(), Acc()
rng = np.random.default_rng(0)
rand_sets = [torch.tensor(rng.choice(D, K, replace=False), device="cuda") for _ in range(20)]
shares = {"enc_var_top": [], "pred_var_top": [], "random": [], "own_enc_var_top": [], "own_pred_var_top": []}
per_inst_e = []
for wi in ev:
    z = lat[wi].cuda()
    B = cands.shape[1]
    s0b = np.repeat(states[wi][None], B, 0)
    s1 = L.compute_new_pose_np(s0b, cands[wi][:, 0])
    a = torch.tensor(cands[wi], dtype=torch.float32, device="cuda")
    st = torch.tensor(np.stack([s0b, s1], 1), dtype=torch.float32, device="cuda")
    x = z[0].unsqueeze(0).expand(B, -1, -1).contiguous()
    with torch.no_grad(), L.precision("fp16"):
        o1 = pred(x, a[:, :1], st[:, :1])[:, -L.NTOK:].float()
        zh1 = F.layer_norm(o1, (D,))
        o2 = pred(torch.cat([x, zh1], 1), a, st)[:, -L.NTOK:].float()
        zh2 = F.layer_norm(o2, (D,))
    for o, zh in [(o1, zh1), (o2, zh2)]:
        acc_pre.add(o)
        acc_post.add(zh)
    per_inst_e.append([((zh1 - z[1]).abs().mean(1) / D).double(), ((zh2 - z[2]).abs().mean(1) / D).double()])  # e_cd [64, D]
res["predictor_out_preLN"], _ = acc_pre.report()
res["predictor_out_postLN"], v_pred = acc_post.report()
top_enc = torch.argsort(v_enc, descending=True)[:K]
top_pred = torch.argsort(v_pred, descending=True)[:K]
for pair in per_inst_e:
    for e in pair:
        Etot = e.sum(1)
        vE = Etot.var()
        cov = ((e - e.mean(0)) * (Etot - Etot.mean())[:, None]).sum(0) / (e.shape[0] - 1)
        shares["enc_var_top"].append(float(cov[top_enc].sum() / vE))
        shares["pred_var_top"].append(float(cov[top_pred].sum() / vE))
        shares["random"].append(float(np.mean([float(cov[r].sum() / vE) for r in rand_sets])))
        shares["own_enc_var_top"].append(float(e[:, top_enc].sum(1).var() / vE))
        shares["own_pred_var_top"].append(float(e[:, top_pred].sum(1).var() / vE))
res["energy_variance_share_top1pct"] = {k: dict(mean=float(np.mean(v)), median=float(np.median(v)), p90=float(np.quantile(v, 0.9)))
                                        for k, v in shares.items()}
res["energy_variance_share_top1pct"]["n_instances"] = len(shares["random"])
res["top1pct_coords_enc_var"] = top_enc.tolist()
res["top1pct_coords_pred_var"] = top_pred.tolist()
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
print("MASSIVE_DONE")
