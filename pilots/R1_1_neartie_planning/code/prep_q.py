"""Phase B' (amendment 1, PLAN.md section 7): reference scales and prediction metrics for the Q-ensemble scope.

Planning-time Q inputs are (z_H, pi(z_H)) at the end of each candidate rollout. From the recorded candidate sequences we
roll out the unablated dynamics to z_3 and sample the policy action with a fixed seed (as the planner does), then
  calibration split -> m_j (99.9th pct of |a_j|) for both hidden layers of every Q member (q{k}_0, q{k}_1)
  held-out split    -> errq = mean over members k of E(Q_k,abl - Q_k)^2 / Var(Q_k) (two-hot decoded), removed fraction,
                       for weak / rand_count / strong at each alpha, and the prediction-matched scale r_q (bisection).
Uses the same calibration / held-out episodes and the same held-out subsample as prep.py.
Writes <out>/results/prep_q_<task>.json and <out>/data/scales_q_<task>.pt
"""
import argparse
import json
import time

import numpy as np
import torch

import nt_lib as L
from prep import CHUNK, MATCH_SEED, N_HELD, calib_eps, load_split, match_scale


@torch.no_grad()
def terminal_inputs(model, Z0, A, seed):
    """z_3 from the unablated rollout of each candidate sequence, and a = pi(z_3) sampled with a fixed seed."""
    torch.manual_seed(seed)
    zs, acts = [], []
    for s in range(0, len(Z0), CHUNK):
        z = Z0[s:s + CHUNK]
        for h in range(A.shape[1]):
            z = model.next(z, A[s:s + CHUNK, h], None)
        a, _ = model.pi(z, None)
        zs.append(z)
        acts.append(a)
    return torch.cat(zs), torch.cat(acts)


@torch.no_grad()
def q_values(model, cfg, z, a):
    """Two-hot decoded Q of every member: (num_q, N)."""
    out = []
    for s in range(0, len(z), CHUNK):
        logits = model.Q(z[s:s + CHUNK], a[s:s + CHUNK], None, return_type="all")  # (num_q, B, bins)
        out.append(L.two_hot_inv(logits, cfg))
    return torch.cat(out, 1)


@torch.no_grad()
def compute_m_q(model, ab, z, a, q=0.999):
    n_rows = len(z)
    k = int(np.ceil((1 - q) * n_rows)) + 1
    top = {}
    ab.set("record")
    for s in range(0, n_rows, CHUNK):
        ab.last.clear()
        model.Q(z[s:s + CHUNK], a[s:s + CHUNK], None, return_type="all")
        for name, v in ab.last.items():
            cat = v.abs() if name not in top else torch.cat([top[name], v.abs()], 0)
            top[name] = torch.topk(cat, min(k, cat.shape[0]), dim=0).values
    ab.set(None)
    return {name: t[-1].clamp(min=1e-6).contiguous() for name, t in top.items()}


@torch.no_grad()
def errq(model, cfg, ab, z, a, Qref, var, setting, seed):
    ab.set(**setting, seed=seed)
    Qa = q_values(model, cfg, z, a)
    frac = ab.fractions()
    ab.set(None)
    e = (((Qa - Qref) ** 2).mean(1) / var).mean()
    return {"errq": float(e), "frac_q": frac["q"]}


def run_task(task, out, alphas):
    t0 = time.time()
    agent, cfg = L.build_agent(task)
    model = agent.model
    ab = L.Ablator(model, with_q=True)
    eps = calib_eps(out, task)
    half = max(1, len(eps) // 2)
    cal_eps, held_eps = eps[:half], eps[half:] if len(eps) > 1 else eps
    Zu, idx, A = load_split(out, task, cal_eps)
    Z0 = torch.as_tensor(Zu, device=L.DEV)[torch.as_tensor(idx, device=L.DEV)]
    zc, ac = terminal_inputs(model, Z0, torch.as_tensor(A, device=L.DEV), seed=11)
    m_q = compute_m_q(model, ab, zc, ac)
    ab.m = m_q
    del Z0, zc, ac
    Zu, idx, A = load_split(out, task, held_eps)
    sel = np.random.default_rng(0).choice(len(idx), size=min(N_HELD, len(idx)), replace=False)  # as in prep.py
    Z0 = torch.as_tensor(Zu, device=L.DEV)[torch.as_tensor(idx[sel], device=L.DEV)]
    z, a = terminal_inputs(model, Z0, torch.as_tensor(A[sel], device=L.DEV), seed=12)
    Qref = q_values(model, cfg, z, a)
    var = Qref.var(1).clamp(min=1e-8)
    res = {"task": task, "n_held": len(z), "q_var_per_member": var.tolist(),
           "m_q_median": {k: float(v.median()) for k, v in m_q.items()}, "alphas": {}}
    pred_scale_q = {}
    for alpha in alphas:
        k = L.akey(alpha)
        r = {c: errq(model, cfg, ab, z, a, Qref, var, L.setting_for(c, alpha, {"pred_scale": {k: {"dyn": 1, "rew": 1}}},
                                                                       scope="q"), MATCH_SEED)
             for c in ("weak", "rand_count", "strong")}
        fq = lambda x: errq(model, cfg, ab, z, a, Qref, var, {"mode": "rand", "alpha": alpha, "r_q": x, "scope": "q"},
                            MATCH_SEED)["errq"]
        r_q, e_q, st = match_scale(fq, r["weak"]["errq"])
        pred_scale_q[k] = {"q": r_q}
        r["rand_pred"] = errq(model, cfg, ab, z, a, Qref, var, {"mode": "rand", "alpha": alpha, "r_q": r_q, "scope": "q"},
                              MATCH_SEED + 17)
        r["rand_pred"].update(r_q=r_q, search_errq=e_q, status=st)
        res["alphas"][k] = r
        print(f"{task} Q alpha {k}: frac weak {r['weak']['frac_q']:.3f} errq weak {r['weak']['errq']:.4f} "
              f"rand_count {r['rand_count']['errq']:.4f} strong {r['strong']['errq']:.4f} | pred-matched r_q {r_q:.4f} "
              f"-> errq {r['rand_pred']['errq']:.4f} (frac {r['rand_pred']['frac_q']:.3f})", flush=True)
    torch.save({"m_q": {k: v.cpu() for k, v in m_q.items()}, "pred_scale_q": pred_scale_q},
               f"{out}/data/scales_q_{task}.pt")
    res["secs"] = time.time() - t0
    json.dump(res, open(f"{out}/results/prep_q_{task}.json", "w"), indent=1)
    print(f"PREP_Q {task} done in {res['secs']:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--alphas", default="0.05,0.1,0.2,0.3,0.5")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    for task in a.tasks.split(","):
        run_task(task, a.out, [float(x) for x in a.alphas.split(",")])
    print("PREP_Q_DONE", flush=True)
