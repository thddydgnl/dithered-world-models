"""v5 cost factors (single process, no MPS; GPU 0 is shared, so only ratios are used): wall-clock per control step of
the E2 planner settings relative to the default, with the E2 instrumentation (OEI every 4 / 10 steps with the
independent re-evaluation, plan reruns every 20 / 50 steps) and without it. Mirrors qlib.run_episode's per-step
instrumentation; stops after --steps control steps. Smoke seeds; returns are not looked at.
Usage: python timing_v5.py --tasks mw-push:3,cheetah-run:5 --steps 60 --out <file.json>
"""
import argparse
import copy
import json
import os
import time

import torch

import qlib as L
from make_manifest import e2_settings

R28 = os.path.expanduser("~/pmx/A/tdmpc2_main/inputs/r28")
OUT = os.path.expanduser("~/pmx/A/tdmpc2_main/out")


def ranges(task, ckpt):
    p = f"{OUT}/scales/{task}_s{ckpt}.json"
    qr = json.load(open(p))["quant"]
    return {k: (float(v[0]), float(v[1])) for k, v in qr.items()}


@torch.no_grad()
def time_steps(agent, cfg, quant, probe, v, k, seed, steps, instr, warm, oei_stride, pd_stride):
    c = copy.deepcopy(cfg)
    c.seed = seed
    env = L.make_env(c)
    L.seed_all(seed)
    quant.set(v, k, seed=seed * 7 + 13)
    obs = env.reset()
    ts = []
    for t in range(steps):
        quant.begin_step()
        torch.cuda.synchronize()
        t_0 = time.perf_counter()
        m_oei = instr and t % oei_stride == 0
        m_pd = instr and t % pd_stride == 0
        t0 = (t == 0) or not warm
        if m_pd:
            rng_pre, prev = L.rng_get(), agent._prev_mean.clone()
        probe.start(m_oei, sn_seed=1_000_003 + 101 * seed + t)
        action = agent.act(obs, t0=t0, eval_mode=True)
        if m_oei:
            probe.summary(indep_seed=3_000_017 + 101 * seed + t)
        if m_pd:
            rng_post, mean_q = L.rng_get(), agent._prev_mean.clone()
            L.plan_rerun(agent, quant, probe, obs.to(agent.device).unsqueeze(0), t0, prev, rng_pre, 2_000_003 + t)
            agent._prev_mean.copy_(mean_q)
            L.rng_set(rng_post)
        probe.start(False)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t_0)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
    quant.off()
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="mw-push:3,cheetah-run:5")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    res = []
    for tk in a.tasks.split(","):
        task, k = tk.split(":")[0], int(tk.split(":")[1])
        agent, cfg = L.build_agent(task, 1)
        quant = L.Quantizer(agent.model, horizon=int(cfg.horizon))
        quant.qrange = ranges(task, 1)
        probe = L.PlanProbe(agent, quant)
        mw = task.startswith("mw-")
        for ptag, ov, warm in e2_settings():
            L.set_planner(agent, ov)
            for instr in (False, True):
                ts = time_steps(agent, cfg, quant, probe, "shiter", k, 19005, a.steps, instr, warm,
                                4 if mw else 10, 20 if mw else 50)
                ts = sorted(ts[5:])
                r = {"task": task, "setting": ptag, "instr": instr, "ms_median": 1000 * ts[len(ts) // 2],
                     "ms_mean": 1000 * sum(ts) / len(ts), "n": len(ts)}
                res.append(r)
                print(json.dumps(r), flush=True)
        L.set_planner(agent, None)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
