"""R2-14: planning wall-clock per control step vs MPPI iteration count (single process; run when nothing else uses the
GPU). For each cell, iteration count n and condition (none + variants): fresh env (seed --seed), --warm control steps,
then --steps timed calls of agent.act with torch.cuda.synchronize() around each call.
Usage: python timing_r214.py --cells mw-push:3,cheetah-run:4 --iters 1,3,6,12 --variants det,shstep,shfwd --out ..
Writes <out>/results/timing.json
"""
import argparse
import copy
import json
import time

import numpy as np
import torch

import nt_lib as L


def main(cells, iters, variants, warm, steps, seed, out):
    torch.set_num_threads(1)
    res = []
    for task, k in cells:
        agent, cfg = L.build_agent(task)
        sc = L.load_scales(out, task)
        ab = L.Ablator(agent.model, horizon=int(cfg.horizon))
        L.attach_scales(ab, sc)
        for it in iters:
            agent.cfg.iterations = int(it)
            for cond in ["none"] + [f"Q{v}{k}" for v in variants]:
                c = copy.deepcopy(cfg)
                c.seed = seed
                env = L.make_env(c)
                L.seed_all(seed)
                ab.set(**L.setting_for(cond, float(k), sc), seed=seed * 7 + 13)
                obs = env.reset()
                ts = []
                for t in range(warm + steps):
                    ab.begin_step()
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    act = agent.act(obs, t0=(t == 0), eval_mode=True)
                    torch.cuda.synchronize()
                    if t >= warm:
                        ts.append(time.perf_counter() - t0)
                    obs, _, done, _ = env.step(act)
                    if done:
                        obs = env.reset()
                ab.set(None)
                r = {"task": task, "cond": cond, "iters": int(it), "ms_mean": 1000 * float(np.mean(ts)),
                     "ms_median": 1000 * float(np.median(ts)), "ms_p90": 1000 * float(np.percentile(ts, 90)), "n": len(ts)}
                res.append(r)
                print(json.dumps(r), flush=True)
    json.dump(res, open(f"{out}/results/timing.json", "w"), indent=1)
    print("TIMING_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--iters", default="1,3,6,12")
    ap.add_argument("--variants", default="det,shstep,shfwd")
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=3900)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main([(c.split(":")[0], int(c.split(":")[1])) for c in a.cells.split(",")], [int(x) for x in a.iters.split(",")],
         a.variants.split(","), a.warm, a.steps, a.seed, a.out)
