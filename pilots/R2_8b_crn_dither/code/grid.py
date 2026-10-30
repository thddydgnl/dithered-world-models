"""Phase C (R2-8b): MPC planning episodes under k-bit activation quantization with five dither-sharing scopes
(paired fixed seeds, all conditions on the same pod).

Cells: task:k. Per cell: none (unquantized baseline, this pod) and Q{v}{k} for v in det, ind, shfwd, shstep, shep
(definitions in nt_lib). Episode e uses seed seed0 + e for the env, torch and numpy, so the planner noise is paired
across conditions; the private dither generator is seeded with 7 * seed + 13. shstep draws a new dither table before
every control step (ab.begin_step() in before_act); shep keeps the table drawn when the episode starts.
Results are appended to <out>/results/episodes.jsonl one episode at a time (resumable).
Usage: python grid.py --cells mw-push:3,mw-pick-place:3,cheetah-run:4,walker-walk:4 --episodes 20 --seed0 2000 \
         --workers 16 --out /workspace/out_main
Set NT_LOG_CALLS=1 to store the per-control-step forward-call counts and dither-table signatures (smoke test).
"""
import argparse
import json
import os
import time
from multiprocessing import get_context

_cache = {}
LOG_CALLS = os.environ.get("NT_LOG_CALLS", "0") == "1"


def job(args):
    task, cond, alpha, ep, seed0, out = args
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
    seed = seed0 + ep
    ab.set(**L.setting_for(cond, alpha, sc), seed=seed * 7 + 13)
    ab.call_log = [] if LOG_CALLS else None
    sig = []

    def before(t, obs):
        ab.begin_step()
        if LOG_CALLS and ab.mode == "quant" and ab.qvar in ("shstep", "shep") and t < 4:
            sig.append([round(float(ab.table[n].sum()), 4) for n in ("dyn0", "rew1")])

    t0 = time.time()
    res = L.run_episode(agent, cfg, seed, before_act=before)
    ab.flush_calls()
    n_refresh, calls = ab.n_refresh, ab.call_log
    frac = ab.fractions()
    ab.call_log = None
    ab.set(None)
    for k in ("rewards", "final_obs"):
        res.pop(k)
    res.update(task=task, cond=cond, alpha=alpha, ep=ep, seed=seed, frac=frac, n_refresh=n_refresh,
               secs=time.time() - t0)
    if LOG_CALLS:
        res.update(calls_first=calls[:2] if calls else None,
                   calls_patterns=sorted({json.dumps(c, sort_keys=True) for c in calls}) if calls else None,
                   table_sig=sig)
    return res


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                keys.add((r["task"], r["cond"], float(r["alpha"]), int(r["ep"])))
    return keys


def ordered_jobs(cells, episodes, seed0, out, have, variants):
    jobs = []
    add = lambda j: jobs.append(j) if j[:4] not in have else None
    for ep in range(episodes):
        for task, _ in cells:
            add((task, "none", 0.0, ep, seed0, out))
    for ep in range(episodes):  # episode-major: every condition gets its early episodes first
        for task, k in cells:
            for v in variants:
                add((task, f"Q{v}{k}", float(k), ep, seed0, out))
    return jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)          # task:k,task:k,...
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=2000)
    ap.add_argument("--variants", default="det,ind,shfwd,shstep,shep")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cells = [(c.split(":")[0], int(c.split(":")[1])) for c in a.cells.split(",")]
    os.makedirs(f"{a.out}/results", exist_ok=True)
    path = f"{a.out}/results/episodes.jsonl"
    have = done_keys(path)
    jobs = ordered_jobs(cells, a.episodes, a.seed0, a.out, have, a.variants.split(","))
    print(f"{len(jobs)} jobs ({len(have)} already done)", flush=True)
    t0 = time.time()
    n = 0
    if jobs:
        with get_context("spawn").Pool(min(a.workers, len(jobs))) as pool, open(path, "a") as f:
            for r in pool.imap_unordered(job, jobs, chunksize=1):
                f.write(json.dumps(r) + "\n")
                f.flush()
                n += 1
                print(f"{time.time() - t0:7.0f}s [{n}/{len(jobs)}] {r['task']:<14} {r['cond']:<10} ep{r['ep']:<3} "
                      f"seed {r['seed']} ret {r['ret']:8.1f} succ {r['success']:.0f} refresh {r['n_refresh']} "
                      f"({r['secs']:.0f}s)", flush=True)
    print("GRID_DONE", flush=True)
