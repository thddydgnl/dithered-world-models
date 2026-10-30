"""Phase C (R2-3): MPC planning episodes under each condition (paired fixed seeds across conditions).

Conditions: none (baseline) and, for each alpha, A (zero weak), B (conditional-mean replacement), C (bias only),
D1z / D3z (random zeroing matched to A on 1-step / 3-step error), D1m / D3m (mean-corrected random replacement
matched to B on 1-step / 3-step error). Episode k uses seed EP_SEED0 + k in every condition (same initial state, same
planner noise until the trajectories diverge); random masks use a private generator seeded per episode.
Job order follows the coordinator's priority: MW tasks + cheetah at alpha=0.2, then their other alphas, then walker.
Results are appended to <out>/results/episodes.jsonl as they finish; finished jobs are skipped on restart.

Usage: python grid.py --tasks mw-push,mw-pick-place,cheetah-run,walker-walk --episodes 10 --alphas 0.1,0.2,0.3 \
         --workers 7 --out /workspace/out_main
"""
import argparse
import json
import os
import time
from multiprocessing import get_context

_cache = {}


def job(args):
    task, cond, alpha, ep, out = args
    import torch
    import nt_lib as L
    torch.set_num_threads(1)
    if task not in _cache:
        agent, cfg = L.build_agent(task)
        sc = L.load_scales(out, task)
        ab = L.Ablator(agent.model)
        L.attach_scales(ab, sc)
        _cache[task] = (agent, cfg, ab, sc)
    agent, cfg, ab, sc = _cache[task]
    seed = L.EP_SEED0 + ep
    ab.set(**L.setting_for(cond, alpha, sc), seed=seed * 7 + 13)
    t0 = time.time()
    res = L.run_episode(agent, cfg, seed)
    frac = ab.fractions()
    ab.set(None)
    for k in ("rewards", "final_obs"):
        res.pop(k)
    res.update(task=task, cond=cond, alpha=alpha, ep=ep, seed=seed, frac=frac, secs=time.time() - t0)
    return res


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                keys.add((r["task"], r["cond"], float(r["alpha"]), int(r["ep"])))
    return keys


def ordered_jobs(tasks, conds, alphas, episodes, out, have):
    prio_tasks = [t for t in tasks if t != "walker-walk"]
    groups = [[(t, 0.0) for t in tasks]]                                   # baselines first
    groups.append([(t, 0.2) for t in prio_tasks])                          # priority 1
    groups.append([(t, a) for t in prio_tasks for a in alphas if a != 0.2])  # priority 2
    groups.append([(t, a) for t in tasks if t == "walker-walk" for a in alphas])  # priority 3
    jobs = []
    for g in groups:
        for ep in range(episodes):  # episode-major inside a group
            for task, a in g:
                for cond in (["none"] if a == 0.0 else [c for c in conds if c != "none"]):
                    if a != 0.0 and a not in alphas:
                        continue
                    if (task, cond, a, ep) not in have:
                        jobs.append((task, cond, a, ep, out))
    return jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--alphas", default="0.1,0.2,0.3")
    ap.add_argument("--conds", default="none,A,B,C,D1z,D3z,D1m,D3m")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks = a.tasks.split(",")
    alphas = [float(x) for x in a.alphas.split(",")]
    conds = a.conds.split(",")
    path = f"{a.out}/results/episodes.jsonl"
    have = done_keys(path)
    jobs = ordered_jobs(tasks, conds, alphas, a.episodes, a.out, have)
    print(f"{len(jobs)} jobs ({len(have)} already done)", flush=True)
    t0 = time.time()
    n = 0
    if jobs:
        with get_context("spawn").Pool(min(a.workers, len(jobs))) as pool, open(path, "a") as f:
            for r in pool.imap_unordered(job, jobs, chunksize=1):
                f.write(json.dumps(r) + "\n")
                f.flush()
                n += 1
                fr = r["frac"]["all"]
                print(f"{time.time() - t0:7.0f}s [{n}/{len(jobs)}] {r['task']:<14} {r['cond']:<5} a={r['alpha']:<4g} "
                      f"ep{r['ep']} ret {r['ret']:8.1f} succ {r['success']:.0f} frac {fr if fr is None else round(fr, 3)} "
                      f"({r['secs']:.0f}s)", flush=True)
    print("GRID_DONE", flush=True)
