"""Phase C (R2-3b): MPC planning episodes for the deterministic, candidate-consistent controls.

Conditions per cell (task, alpha in {0.1, 0.2}): A, B, D1z, D3z (R2-3 conditions, rerun on this pod because its GPU
differs from the R2-3 pod and episodes are not bit-identical across GPU models) and Dfix1_s{0,1,2}, Dfix3_s{0,1,2},
Ddir1_s{0,1,2} (prep.py), 10 episodes each with the shared seeds 1000-1009; baselines first.
Order: check jobs, then alpha=0.2 for mw-push / mw-pick-place / cheetah-run, then their alpha=0.1, then walker-walk.

Usage: python grid.py --tasks mw-push,mw-pick-place,cheetah-run,walker-walk --episodes 10 --alphas 0.1,0.2 --workers 7 --out ...
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


CONDS_3B = ("A", "B", "D1z", "D3z") + tuple(f"{c}_s{k}" for c in ("Dfix1", "Dfix3", "Ddir1") for k in (0, 1, 2))


def ordered_jobs(tasks, conds, alphas, episodes, out, have):
    jobs = []
    add = lambda j: jobs.append(j) if j[:4] not in have else None
    for ep in range(episodes):  # baselines (also compared with the R2-3 A5000 episodes: cross-GPU determinism)
        for task in tasks:
            add((task, "none", 0.0, ep, out))
    prio = [t for t in tasks if t != "walker-walk"]
    groups = [[(t, 0.2) for t in prio], [(t, 0.1) for t in prio],
              [(t, a) for t in tasks if t == "walker-walk" for a in sorted(alphas, reverse=True)]]
    for g in groups:
        for ep in range(episodes):
            for task, a in g:
                if a not in alphas:
                    continue
                for cond in conds:
                    add((task, cond, a, ep, out))
    return jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--alphas", default="0.1,0.2")
    ap.add_argument("--conds", default=",".join(CONDS_3B))
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks = a.tasks.split(",")
    alphas = [float(x) for x in a.alphas.split(",")]
    conds = a.conds.split(",")
    path = f"{a.out}/results/episodes_3b.jsonl"
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
                print(f"{time.time() - t0:7.0f}s [{n}/{len(jobs)}] {r['task']:<14} {r['cond']:<8} a={r['alpha']:<4g} "
                      f"ep{r['ep']} ret {r['ret']:8.1f} succ {r['success']:.0f} frac {fr if fr is None else round(fr, 3)} "
                      f"({r['secs']:.0f}s)", flush=True)
    print("GRID_DONE", flush=True)
