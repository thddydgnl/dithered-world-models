"""R2-14 secondary: final-plan deviation vs MPPI iteration count (one-step plan reruns from stored calibration states).

For each iteration count n in --iters and each of --max_states calibration decision states (same selection rule as
R2-8b decision.py: rng(0) subset of the 50-per-episode decision states of the 10 calibration episodes), rerun
agent._plan from the stored obs and previous plan mean with the same torch seed under 'none' (seeds s and s+1: the
seed-noise floor at that n) and under every variant; record the RMS deviation of the first step of the final MPPI mean
(dev_mean) and of the executed action (dev_act) from the unquantized plan with the same seed and the same n.
Usage: python plandev_r214.py --task mw-push --k 3 --iters 1,3,6,12 --variants det,shstep,shfwd --max_states 150 --out ..
Writes <out>/results/plandev_<task>.csv
"""
import argparse
import csv
import glob
import time

import numpy as np
import torch

import nt_lib as L


def main(task, k, iters, variants, max_states, out):
    torch.set_num_threads(1)
    agent, cfg = L.build_agent(task)
    sc = L.load_scales(out, task)
    ab = L.Ablator(agent.model, horizon=int(cfg.horizon))
    L.attach_scales(ab, sc)
    keys = ("dec_t0", "dec_obs", "dec_prev")
    D = {kk: [] for kk in keys}
    for f in sorted(glob.glob(f"{out}/data/calib_{task}_ep*.npz")):
        d = np.load(f)
        for kk in keys:
            D[kk].append(d[kk])
    D = {kk: np.concatenate(v) for kk, v in D.items()}
    n = len(D["dec_obs"])
    sel = np.arange(n) if n <= max_states else np.sort(np.random.default_rng(0).choice(n, max_states, replace=False))
    rms = lambda x, y: float(((x - y) ** 2).mean().sqrt())
    rows, t_start = [], time.time()
    for it in iters:
        agent.cfg.iterations = int(it)
        for i in sel:
            seed = 7_000_000 + int(i)
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
            an, mn = plan(seed + 1, {"mode": None})
            base = {"task": task, "iters": int(it), "state": int(i)}
            rows.append({**base, "cond": "seednoise", "dev_act": rms(an, a0), "dev_mean": rms(mn, m0)})
            for v in variants:
                c = f"Q{v}{k}"
                a, mu = plan(seed, L.setting_for(c, float(k), sc))
                rows.append({**base, "cond": c, "dev_act": rms(a, a0), "dev_mean": rms(mu, m0)})
        print(f"{task} iters {it} done ({time.time() - t_start:.0f}s)", flush=True)
    with open(f"{out}/results/plandev_{task}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"PLANDEV {task} done in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--iters", default="1,3,6,12")
    ap.add_argument("--variants", default="det,shstep,shfwd")
    ap.add_argument("--max_states", type=int, default=150)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.task, a.k, [int(x) for x in a.iters.split(",")], a.variants.split(","), a.max_states, a.out)
