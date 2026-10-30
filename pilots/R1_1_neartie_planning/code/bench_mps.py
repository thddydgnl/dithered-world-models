"""Throughput check: N worker processes each run `steps` MPC steps on walker-walk (weak ablation hooks on).
Usage: python bench_mps.py <n_procs> <steps>"""
import sys
import time
from multiprocessing import get_context


def worker(args):
    steps, seed = args
    import copy
    import torch
    import nt_lib as L
    from envs import make_env
    torch.set_num_threads(1)
    agent, cfg = L.build_agent("walker-walk")
    ab = L.Ablator(agent.model)
    ab.m = {k: torch.full((512,), 3.0, device="cuda") for k in L.LAYERS}
    ab.set("weak", alpha=0.1)
    c = copy.deepcopy(cfg)
    c.seed = seed
    env = make_env(c)
    obs = env.reset()
    for t in range(5):
        obs, *_ = env.step(agent.act(obs, t0=t == 0, eval_mode=True))
    torch.cuda.synchronize()
    t0 = time.time()
    for t in range(steps):
        obs, *_ = env.step(agent.act(obs, t0=False, eval_mode=True))
    torch.cuda.synchronize()
    return time.time() - t0


if __name__ == "__main__":
    n, steps = int(sys.argv[1]), int(sys.argv[2])
    with get_context("spawn").Pool(n) as pool:
        t0 = time.time()
        secs = pool.map(worker, [(steps, s) for s in range(n)])
    print(f"n={n} per-proc ms/step {[round(1000 * s / steps) for s in secs]} -> aggregate steps/s "
          f"{n * steps / max(secs):.1f}", flush=True)
