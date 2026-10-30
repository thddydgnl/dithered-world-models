"""Policy-prior share of the unquantized TD-MPC2 planner (planner structure for the v5 expectation on indstep / indep;
smoke seeds, not an outcome of any hypothesis). TDMPC2._plan fills candidate slots 0..num_pi_trajs-1 with policy-prior
trajectories once per control step, before the MPPI loop, so these slots keep their action sequence in every
iteration; slots num_pi_trajs.. are redrawn from N(mean, std) every iteration. Per measured control step: share of the
last-iteration elites that are policy-prior slots (elite_pi), their MPPI weight (elite_pi_w) and whether the executed
elite is one (chosen_pi). One episode per task, checkpoint 1.
usage (server): python pi_share.py [--tasks a,b] [--seed 19005] [--stride 2] [--out ../v5/pi_share.json]
"""
import argparse
import json

import numpy as np
import torch

import qlib as L

TASKS = ["mw-push", "mw-pick-place", "mw-door-open", "mw-drawer-open", "mw-button-press-topdown", "mw-peg-insert-side",
         "mw-window-open", "mw-sweep-into", "cheetah-run", "walker-walk", "quadruped-walk", "finger-turn-hard"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--seed", type=int, default=19005)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", default="../v5/pi_share.json")
    a = ap.parse_args()
    res = {}
    for task in a.tasks.split(","):
        agent, cfg = L.build_agent(task, 1)
        quant = L.Quantizer(agent.model, horizon=int(cfg.horizon))
        probe = L.PlanProbe(agent, quant)
        r = L.run_episode(agent, cfg, quant, "none", 0, a.seed, probe=probe, oei_stride=a.stride)
        s = r["steps_rec"]
        res[task] = {kk: float(np.mean([x[kk] for x in s])) for kk in ("chosen_pi", "elite_pi", "elite_pi_w")}
        res[task].update(n_steps=len(s), ret=r["ret"], n_pi=int(agent.cfg.num_pi_trajs),
                         n_samples=int(agent.cfg.num_samples))
        print(task, json.dumps(res[task]), flush=True)
        probe.detach()
        quant.remove()
        del agent
        torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1)
    print("PI_SHARE_DONE", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
