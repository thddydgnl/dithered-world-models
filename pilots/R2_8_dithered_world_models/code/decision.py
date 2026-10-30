"""Phase D (R2-3, copied from R1-1): decision-level analysis on decision states from the unablated calibration rollouts.

For each stored state (50 per episode) and each candidate set (the full 512-candidate set of the FIRST MPPI iteration =
primary for S3, the LAST iteration = secondary; same fixed candidates for every condition) we compute the value of every
candidate with the agent's own estimator agent._estimate_value (H-step reward + discounted terminal Q of the policy
action) under the original and the ablated world model, with the same torch seed (same Q-head subset and policy noise),
and record: top1-top2 gap of the original values, argmax flip, Spearman rank correlation, elite (top-64) Jaccard,
normalized value error. 'seednoise' = original model with a different seed (the estimator's own stochasticity).
Full-plan check: rerun agent._plan from the stored obs and previous mean with the same seed under each condition and
record the RMS deviation of the executed action and of the first step of the final MPPI mean.

Usage: python decision.py --tasks walker-walk --alphas ... --conds weak,rand_count,rand_pred,strong --max_states 500
Writes <out>/results/decision_<task>.csv
"""
import argparse
import csv
import glob
import os
import time
from multiprocessing import get_context

import numpy as np


def rank(x):
    import torch
    r = torch.empty_like(x)
    r[x.argsort()] = torch.arange(len(x), device=x.device, dtype=x.dtype)
    return r


def mppi_weights(v, n_elite, temp):
    """MPPI elite weights as in TDMPC2._plan: softmax(temperature * (V_elite - max V)) over the top-n_elite."""
    import torch
    val, idx = torch.topk(v, n_elite)
    w = torch.exp(temp * (val - val.max()))
    w = w / w.sum()
    full = torch.zeros_like(v)
    full[idx] = w
    ent = float(-(w * torch.log(w.clamp(min=1e-30))).sum())
    return full, ent


def metrics(v0, v, n_elite, temp=0.5, act0=None):
    import torch
    o0 = torch.argsort(v0, descending=True)
    gap = float(v0[o0[0]] - v0[o0[1]])
    pick = int(v.argmax())
    flip = int(int(o0[0]) != pick)
    regret = float(v0[o0[0]] - v0[pick])  # amendment 1: original-model value lost by the ablated model's argmax
    w0, h0 = mppi_weights(v0, n_elite, temp)
    w1, h1 = mppi_weights(v, n_elite, temp)
    e0 = set(o0[:n_elite].tolist())
    e1 = set(torch.topk(v, n_elite).indices.tolist())
    r0, r1 = rank(v0), rank(v)
    r0, r1 = r0 - r0.mean(), r1 - r1.mean()
    spear = float((r0 * r1).sum() / torch.sqrt((r0 ** 2).sum() * (r1 ** 2).sum()))
    var0 = v0.var().clamp(min=1e-12)
    dv = v - v0
    nerr = float((dv ** 2).mean() / var0)
    nerr_c = float(((dv - dv.mean()) ** 2).mean() / var0)  # without the common-mode shift (rank-relevant part)
    vstd = float(v0.std())
    extra = {}
    if act0 is not None:  # R2-8 S1 metric: shift of the MPPI elite-weighted mean of the first planned action
        mu0 = (w0.unsqueeze(1) * act0).sum(0)
        mu1 = (w1.unsqueeze(1) * act0).sum(0)
        extra["dmu"] = float(((mu1 - mu0) ** 2).mean().sqrt())
    return {**extra, "gap": gap, "flip": flip, "jac": len(e0 & e1) / len(e0 | e1), "spear": spear, "nerr": nerr, "nerr_c": nerr_c,
            "vstd": vstd, "vtop": float(v0[o0[0]]), "regret": regret, "regret_n": regret / max(vstd, 1e-12),
            "ent0": h0, "ent1": h1, "dent": h1 - h0, "w_overlap": float(torch.minimum(w0, w1).sum())}


def run_task(args):
    task, out, condlist, plan_conds, max_states, plan_states = args
    import torch
    import nt_lib as L
    torch.set_num_threads(1)
    t_start = time.time()
    agent, cfg = L.build_agent(task)
    sc = L.load_scales(out, task)
    ab = L.Ablator(agent.model)
    L.attach_scales(ab, sc)
    temp = float(cfg.temperature)
    files = sorted(glob.glob(f"{out}/data/calib_{task}_ep*.npz"))
    keys = ("dec_t", "dec_t0", "dec_obs", "dec_prev", "dec_z", "dec_c0", "dec_cL")
    D = {k: [] for k in keys}
    epi = []
    for f in files:
        d = np.load(f)
        ep = int(f.rsplit("_ep", 1)[1].split(".")[0])
        for k in keys:
            D[k].append(d[k])
        epi.append(np.full(len(d["dec_t"]), ep))
    D = {k: np.concatenate(v) for k, v in D.items()}
    epi = np.concatenate(epi)
    n = len(epi)
    sel = np.arange(n) if n <= max_states else np.sort(np.random.default_rng(0).choice(n, max_states, replace=False))
    N, n_elite = int(cfg.num_samples), int(cfg.num_elites)
    condlist = [("wm", c, a) for c, a in condlist]
    rows = []
    for j, i in enumerate(sel):
        seed = 7_000_000 + int(i)
        z = torch.as_tensor(D["dec_z"][i], device=L.DEV).unsqueeze(0).repeat(N, 1)
        base = {"task": task, "state": int(i), "ep": int(epi[i]), "t": int(D["dec_t"][i])}
        for set_name, key in (("last", "dec_cL"), ("first", "dec_c0")):
            cand = torch.as_tensor(D[key][i], device=L.DEV)
            ab.set(None)
            torch.manual_seed(seed)
            v0 = agent._estimate_value(z, cand, None).squeeze(-1).nan_to_num(0)
            torch.manual_seed(seed + 1)
            vn = agent._estimate_value(z, cand, None).squeeze(-1).nan_to_num(0)
            act0 = cand[0]
            rows.append({**base, "set": set_name, "scope": "wm", "cond": "seednoise", "alpha": 0.0,
                         **metrics(v0, vn, n_elite, temp, act0)})
            for scope, cond, alpha in condlist:
                ab.set(**L.setting_for(cond, alpha, sc), seed=seed)
                torch.manual_seed(seed)
                v = agent._estimate_value(z, cand, None).squeeze(-1).nan_to_num(0)
                fr = ab.fractions()
                ab.set(None)
                rows.append({**base, "set": set_name, "scope": scope, "cond": cond, "alpha": alpha, "frac": fr["all"],
                             **metrics(v0, v, n_elite, temp, act0)})
        if j % max(1, len(sel) // plan_states) != 0:  # full-plan check on an evenly spaced subset
            continue
        # full-plan deviation
        obs = torch.as_tensor(D["dec_obs"][i], device=L.DEV).unsqueeze(0)
        prev = torch.as_tensor(D["dec_prev"][i], device=L.DEV)
        t0 = bool(D["dec_t0"][i])

        def plan(s, setting):
            ab.set(**setting, seed=s)
            agent._prev_mean.copy_(prev)
            torch.manual_seed(s)
            a = agent._plan(obs, t0=t0, eval_mode=True).flatten().clone()
            mu = agent._prev_mean[0].clone()
            ab.set(None)
            return a, mu

        a0, m0 = plan(seed, {"mode": None})
        rms = lambda x, y: float(((x - y) ** 2).mean().sqrt())
        an, mn = plan(seed + 1, {"mode": None})
        rows.append({**base, "set": "plan", "scope": "wm", "cond": "seednoise", "alpha": 0.0, "dev_act": rms(an, a0),
                     "dev_mean": rms(mn, m0)})
        for scope, cond, alpha in condlist:
            if (cond, alpha) not in plan_conds:
                continue
            a, mu = plan(seed, L.setting_for(cond, alpha, sc))
            rows.append({**base, "set": "plan", "scope": scope, "cond": cond, "alpha": alpha, "dev_act": rms(a, a0),
                         "dev_mean": rms(mu, m0)})
        if j % 50 == 0:
            print(f"{task}: state {j}/{len(sel)} ({time.time() - t_start:.0f}s)", flush=True)
    cols = sorted({k for r in rows for k in r})
    with open(f"{out}/results/decision_{task}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return task, len(sel), time.time() - t_start


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--condlist", required=True)    # comma list of cond:alpha (alpha = k for quantization)
    ap.add_argument("--plan_conds", default="")      # cond:alpha items that also get the full-plan rerun
    ap.add_argument("--max_states", type=int, default=300)
    ap.add_argument("--plan_states", type=int, default=150)

    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks = a.tasks.split(",")
    parse = lambda txt: [(x.split(":")[0], float(x.split(":")[1])) for x in txt.split(",") if x]
    cl, pc = parse(a.condlist), set(parse(a.plan_conds))
    jobs = [(t, a.out, cl, pc, a.max_states, a.plan_states) for t in tasks]
    with get_context("spawn").Pool(len(jobs)) as pool:
        for task, n, secs in pool.imap_unordered(run_task, jobs):
            print(f"DECISION {task}: {n} states in {secs:.0f}s", flush=True)
    print("DECISION_DONE", flush=True)
