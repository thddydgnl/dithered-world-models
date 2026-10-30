"""Phase A: unablated MPC evaluation episodes that record planning-time inputs.

Episode k of every task uses seed EP_SEED0 + k (grid.py uses the same seeds for every ablation condition).
Per decision step we keep: root latent z, K random candidate action sequences from each MPPI iteration (the inputs
the dynamics and reward MLPs see during planning), obs, executed action, reward. Every `stride` steps (50 states per
episode) we keep a decision state: obs, previous plan mean, t0 flag, the full candidate sets of the first and last
MPPI iteration, the executed action and the new plan mean.

Usage: python calib.py --tasks walker-walk,cheetah-run --episodes 10 --workers 8 --out /workspace/out_main
Writes <out>/data/calib_<task>_ep<k>.npz and appends to <out>/results/calib_episodes.jsonl
"""
import argparse
import json
import os
import time
from multiprocessing import get_context

import numpy as np

K_PER_ITER = 16
STATES_PER_EP = 50
_cache = {}


def job(args):
    task, ep, out = args
    import torch
    import nt_lib as L
    torch.set_num_threads(1)
    path = f"{out}/data/calib_{task}_ep{ep}.npz"
    if os.path.exists(path):
        return json.loads(str(np.load(path)["summary"]))
    if task not in _cache:
        _cache[task] = L.build_agent(task)
    agent, cfg = _cache[task]
    T = int(cfg.episode_length)
    stride = max(1, T // STATES_PER_EP)
    dec_steps = set(range(stride // 2, T, stride))
    rec = L.CandRecorder(agent, K_PER_ITER, seed=ep)
    S = {k: [] for k in ("z", "cand", "obs", "act", "dec_t", "dec_obs", "dec_prev", "dec_z", "dec_c0", "dec_cL",
                         "dec_act", "dec_mean")}

    def before(t, obs):
        rec.start_step(t in dec_steps)
        if t in dec_steps:
            S["dec_prev"].append(agent._prev_mean.detach().clone())

    def after(t, obs, action):
        z, it, first, last = rec.end_step()
        S["z"].append(z)
        S["cand"].append(it)
        S["obs"].append(obs.clone())
        S["act"].append(action.clone())
        if t in dec_steps:
            S["dec_t"].append(t)
            S["dec_obs"].append(obs.clone())
            S["dec_z"].append(z)
            S["dec_c0"].append(first)
            S["dec_cL"].append(last)
            S["dec_act"].append(action.clone())
            S["dec_mean"].append(agent._prev_mean.detach().clone())

    t0 = time.time()
    res = L.run_episode(agent, cfg, L.EP_SEED0 + ep, before, after)
    rec.detach()
    np_ = lambda xs: torch.stack([torch.as_tensor(x) for x in xs]).float().cpu().numpy()
    obs_seq = np_(S["obs"] + [res["final_obs"]])
    summary = {"task": task, "ep": ep, "seed": L.EP_SEED0 + ep, "ret": res["ret"], "success": res["success"],
               "success_any": res["success_any"], "steps": res["steps"], "secs": time.time() - t0,
               "n_dec": len(S["dec_t"])}
    tmp = path + ".tmp.npz"
    np.savez_compressed(
        tmp, z_root=np_(S["z"]), cand=np_(S["cand"]), obs_seq=obs_seq, act_seq=np_(S["act"]),
        rew_seq=np.asarray(res["rewards"], np.float32), dec_t=np.asarray(S["dec_t"], np.int32),
        dec_t0=np.asarray([t == 0 for t in S["dec_t"]], bool), dec_obs=np_(S["dec_obs"]), dec_prev=np_(S["dec_prev"]),
        dec_z=np_(S["dec_z"]), dec_c0=np_(S["dec_c0"]), dec_cL=np_(S["dec_cL"]), dec_act=np_(S["dec_act"]),
        dec_mean=np_(S["dec_mean"]), summary=json.dumps(summary))
    os.replace(tmp, path)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(f"{a.out}/data", exist_ok=True)
    os.makedirs(f"{a.out}/results", exist_ok=True)
    tasks = a.tasks.split(",")
    jobs = [(t, ep, a.out) for ep in range(a.episodes) for t in tasks]
    t0 = time.time()
    with get_context("spawn").Pool(min(a.workers, len(jobs)), maxtasksperchild=None) as pool, \
            open(f"{a.out}/results/calib_episodes.jsonl", "a") as f:
        for s in pool.imap_unordered(job, jobs):
            f.write(json.dumps(s) + "\n")
            f.flush()
            print(f"{time.time() - t0:7.0f}s calib {s['task']:<18} ep{s['ep']} ret {s['ret']:8.1f} "
                  f"succ {s['success']:.0f} steps {s['steps']} ({s['secs']:.0f}s)", flush=True)
    print("CALIB_DONE", flush=True)
