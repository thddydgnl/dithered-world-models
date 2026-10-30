"""Phase C (R2-14): MPC planning episodes vs MPPI iteration count under k-bit activation quantization
(paired fixed seeds, all conditions on the same pod). Copied from R2-8b grid.py; adds the iteration axis.

Cells: task:k. Per cell and per MPPI iteration count n in --iters: none (unquantized baseline with n iterations) and
Q{v}{k} for v in det, shstep, shfwd (definitions in nt_lib). Every other MPPI hyperparameter stays at the TD-MPC2
default (512 samples, 24 policy-prior trajectories, 64 elites, horizon 3, temperature 0.5); only agent.cfg.iterations
is changed (read by TDMPC2._plan at every call). Episode e uses seed seed0 + e for the env, torch and numpy (planner
noise paired across conditions and iteration counts); the private dither generator is seeded with 7 * seed + 13.
shstep draws a new dither table before every control step (ab.begin_step() in before_act) and reuses it in all n
iterations of that step. Per episode we also store the wall-clock spent inside agent.act (plan_secs).
Results are appended to <out>/results/episodes.jsonl one episode at a time (resumable).
Usage: python grid.py --cells mw-push:3,cheetah-run:4 --iters 1,3,6,12 --episodes 15 --seed0 3000 --workers 20 \
         --out /workspace/out_main
Set NT_LOG_CALLS=1 to store the per-control-step forward-call counts (smoke test).
"""
import argparse
import json
import os
import time
from multiprocessing import get_context

_cache = {}
LOG_CALLS = os.environ.get("NT_LOG_CALLS", "0") == "1"


def job(args):
    task, cond, alpha, iters, ep, seed0, out = args
    import torch
    import nt_lib as L
    torch.set_num_threads(1)
    if task not in _cache:
        agent, cfg = L.build_agent(task)
        sc = L.load_scales(out, task)
        ab = L.Ablator(agent.model, horizon=int(cfg.horizon))
        L.attach_scales(ab, sc)
        _cache[task] = (agent, cfg, ab, sc)
    agent, cfg, ab, sc = _cache[task]
    agent.cfg.iterations = int(iters)
    seed = seed0 + ep
    ab.set(**L.setting_for(cond, alpha, sc), seed=seed * 7 + 13)
    ab.call_log = [] if LOG_CALLS else None
    tick = {"t": 0.0, "sum": 0.0}

    def before(t, obs):
        ab.begin_step()
        tick["t"] = time.perf_counter()

    def after(t, obs, action):
        tick["sum"] += time.perf_counter() - tick["t"]   # action comes back on the CPU, so the GPU work is done

    t0 = time.time()
    res = L.run_episode(agent, cfg, seed, before_act=before, after_act=after)
    ab.flush_calls()
    n_refresh, calls = ab.n_refresh, ab.call_log
    ab.call_log = None
    ab.set(None)
    for k in ("rewards", "final_obs"):
        res.pop(k)
    res.update(task=task, cond=cond, alpha=alpha, iters=int(iters), cfg_iterations=int(agent.cfg.iterations), ep=ep,
               seed=seed, n_refresh=n_refresh, plan_secs=tick["sum"], plan_ms_per_step=1000 * tick["sum"] / res["steps"],
               secs=time.time() - t0)
    if LOG_CALLS:
        res.update(calls_patterns=sorted({json.dumps(c, sort_keys=True) for c in calls}) if calls else None)
    return res


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                keys.add((r["task"], r["cond"], int(r["iters"]), int(r["ep"])))
    return keys


def ordered_jobs(cells, iters, episodes, seed0, out, have, variants):
    jobs = []
    for ep in range(episodes):  # episode-major: every condition gets its early episodes first
        for task, k in cells:
            for n in iters:
                for cond, alpha in [("none", 0.0)] + [(f"Q{v}{k}", float(k)) for v in variants]:
                    if (task, cond, int(n), ep) not in have:
                        jobs.append((task, cond, alpha, int(n), ep, seed0, out))
    return jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)          # task:k,task:k,...
    ap.add_argument("--iters", default="1,3,6,12")
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--seed0", type=int, default=3000)
    ap.add_argument("--variants", default="det,shstep,shfwd")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cells = [(c.split(":")[0], int(c.split(":")[1])) for c in a.cells.split(",")]
    iters = [int(x) for x in a.iters.split(",")]
    os.makedirs(f"{a.out}/results", exist_ok=True)
    path = f"{a.out}/results/episodes.jsonl"
    have = done_keys(path)
    jobs = ordered_jobs(cells, iters, a.episodes, a.seed0, a.out, have, a.variants.split(","))
    print(f"{len(jobs)} jobs ({len(have)} already done)", flush=True)
    t0 = time.time()
    n = 0
    if jobs:
        with get_context("spawn").Pool(min(a.workers, len(jobs))) as pool, open(path, "a") as f:
            for r in pool.imap_unordered(job, jobs, chunksize=1):
                f.write(json.dumps(r) + "\n")
                f.flush()
                n += 1
                print(f"{time.time() - t0:7.0f}s [{n}/{len(jobs)}] {r['task']:<14} it{r['iters']:<3} {r['cond']:<10} "
                      f"ep{r['ep']:<3} seed {r['seed']} ret {r['ret']:8.1f} succ {r['success']:.0f} refresh {r['n_refresh']} "
                      f"plan {r['plan_ms_per_step']:.1f}ms/step ({r['secs']:.0f}s)", flush=True)
    print("GRID_DONE", flush=True)
