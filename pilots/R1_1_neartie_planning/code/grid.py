"""Phase C: MPC planning episodes under each ablation condition (paired fixed seeds across conditions).

Conditions: none (baseline), and for each alpha: weak, rand_count (count-matched random), rand_pred (prediction-matched
random), strong (count-matched top-k). Episode k uses seed EP_SEED0 + k in every condition (same initial state, same
planner noise until the trajectories diverge). The ablation masks use a private generator seeded per episode.
Results are appended to <out>/results/episodes.jsonl as they finish; finished jobs are skipped on restart.

Usage: python grid.py --tasks walker-walk --episodes 10 --alphas 0.05,0.1,0.2,0.3,0.5 \
         --conds none,weak,rand_count,rand_pred,strong --workers 8 --out /workspace/out_main
"""
import argparse
import json
import os
import time
from multiprocessing import get_context

_cache = {}


def job(args):
    task, cond, alpha, ep, out, scope = args
    import torch
    import nt_lib as L
    torch.set_num_threads(1)
    with_q = scope != "wm"
    if (task, with_q) not in _cache:
        agent, cfg = L.build_agent(task)
        sc = L.load_scales(out, task, with_q=with_q)
        ab = L.Ablator(agent.model, with_q=with_q)
        ab.m = sc["m"]
        _cache[(task, with_q)] = (agent, cfg, ab, sc)
    agent, cfg, ab, sc = _cache[(task, with_q)]
    seed = L.EP_SEED0 + ep
    ab.set(**L.setting_for(cond, alpha, sc, scope=scope), seed=seed * 7 + 13)
    t0 = time.time()
    res = L.run_episode(agent, cfg, seed)
    frac = ab.fractions()
    ab.set(None)
    for k in ("rewards", "final_obs"):
        res.pop(k)
    res.update(task=task, cond=cond, alpha=alpha, ep=ep, seed=seed, frac=frac, scope=scope, secs=time.time() - t0)
    return res


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                keys.add((r["task"], r["cond"], float(r["alpha"]), int(r["ep"]), r.get("scope", "wm")))
    return keys


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--alphas", default="0.05,0.1,0.2,0.3,0.5")
    ap.add_argument("--conds", default="none,weak,rand_count,rand_pred,strong")
    ap.add_argument("--short_conds", default="strong,rand_count")  # controls not used by any criterion
    ap.add_argument("--short_episodes", type=int, default=5)
    ap.add_argument("--scope", default="wm")  # comma list of wm / q / all (q, all: amendment 1, PLAN.md section 7)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks = a.tasks.split(",")
    alphas = [float(x) for x in a.alphas.split(",")]
    conds = a.conds.split(",")
    scopes = a.scope.split(",")
    path = f"{a.out}/results/episodes.jsonl" if scopes == ["wm"] else f"{a.out}/results/episodes_q.jsonl"
    have = done_keys(path)
    jobs = []
    for ep in range(a.episodes):  # episode-major order: every condition gets early episodes first
        for task in tasks:
            for scope in scopes:
                for cond in conds:
                    for alpha in ([0.0] if cond == "none" else alphas):
                        if cond in a.short_conds.split(",") and ep >= a.short_episodes:
                            continue
                        if (task, cond, alpha, ep, scope) not in have:
                            jobs.append((task, cond, alpha, ep, a.out, scope))
    print(f"{len(jobs)} jobs ({len(have)} already done)", flush=True)
    t0 = time.time()
    n = 0
    if jobs:
        with get_context("spawn").Pool(min(a.workers, len(jobs))) as pool, open(path, "a") as f:
            for r in pool.imap_unordered(job, jobs):
                f.write(json.dumps(r) + "\n")
                f.flush()
                n += 1
                fr = r["frac"]["all"]
                print(f"{time.time() - t0:7.0f}s [{n}/{len(jobs)}] {r['task']:<16} {r['scope']}:{r['cond']:<10} a={r['alpha']:<5g} "
                      f"ep{r['ep']} ret {r['ret']:8.1f} succ {r['success']:.0f} frac {fr if fr is None else round(fr, 3)} "
                      f"({r['secs']:.0f}s)", flush=True)
    print("GRID_DONE", flush=True)
