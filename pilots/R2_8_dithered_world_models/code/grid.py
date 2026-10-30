"""Phase C (R2-8): MPC planning episodes under k-bit activation quantization (paired fixed seeds, same pod).

Conditions per task: none (unquantized baseline, run on this pod) and, for each k in --kbits, Qdet{k} (deterministic
rounding), Qind{k} (per-candidate independent stochastic rounding), Qsh{k} (stochastic rounding with the dither shared by
all candidates of a forward call), Qsub{k} (per-candidate independent subtractive dither). The 'alpha' field carries k.
Usage: python grid.py --tasks mw-push,mw-pick-place,cheetah-run,walker-walk --episodes 10 --kbits 3,4,5 --workers 7 --out ...
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


QVARS = ("det", "ind", "sh", "sub")


def ordered_jobs(tasks, kbits, episodes, out, have, variants=QVARS):
    jobs = []
    add = lambda j: jobs.append(j) if j[:4] not in have else None
    for ep in range(episodes):
        for task in tasks:
            add((task, "none", 0.0, ep, out))
    for ep in range(episodes):  # episode-major: every condition gets its early episodes first
        for task in tasks:
            for k in kbits:
                for v in variants:
                    add((task, f"Q{v}{k}", float(k), ep, out))
    return jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--kbits", default="3,4,5")
    ap.add_argument("--variants", default=",".join(QVARS))
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks = a.tasks.split(",")
    kbits = [int(x) for x in a.kbits.split(",")]
    path = f"{a.out}/results/episodes.jsonl"
    have = done_keys(path)
    jobs = ordered_jobs(tasks, kbits, a.episodes, a.out, have, a.variants.split(","))
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
