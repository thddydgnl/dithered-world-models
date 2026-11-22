"""X6 (D7) check: TD-MPC2's mpc=False path never calls the world model's dynamics or reward MLPs (the layers the
quantizer acts on). Counts forward calls of dyn0, dyn1, rew0, rew1 (plain counting hooks, quantizer off) during one
policy episode per task (checkpoint 1, seed 100000), and for contrast during the first 3 control steps with mpc=True.
Also prints TDMPC2.act's source lines (the mpc switch). Usage: python check_policy_calls.py --out <json>
"""
import argparse
import copy
import inspect
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

import plan_td as P  # noqa: E402
import qlib as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_num_threads(1)
from tdmpc2 import TDMPC2  # noqa: E402
src = [ln.rstrip() for ln in inspect.getsource(TDMPC2.act).splitlines()]
print("\n".join(src), flush=True)
res = {"act_source": src, "tasks": {}}
with torch.no_grad():
    for t in P.MW8 + P.DMC4:
        agent, cfg = L.build_agent(t, 1)
        quant = L.Quantizer(agent.model, horizon=int(cfg.horizon))
        cnt = {n: 0 for n in L.LAYERS}
        hs = [quant.mods[n].register_forward_hook(lambda m, i, o, n=n: cnt.__setitem__(n, cnt[n] + 1))
              for n in L.LAYERS]
        agent.cfg.mpc = False
        r = L.run_episode(agent, cfg, quant, "none", 0, P.SEED0["eval"])
        pol = dict(cnt)
        for n in cnt:
            cnt[n] = 0
        agent.cfg.mpc = True
        c = copy.deepcopy(cfg)
        c.seed = P.SEED0["eval"]
        env = L.make_env(c)
        L.seed_all(P.SEED0["eval"])
        obs = env.reset()
        for step in range(3):
            act = agent.act(obs, t0=(step == 0), eval_mode=True)
            obs, _, _, _ = env.step(act)
        mpc = dict(cnt)
        for h in hs:
            h.remove()
        quant.remove()
        res["tasks"][t] = {"policy_episode_calls": pol, "policy_steps": r["steps"], "policy_return": r["ret"],
                           "mpc_3_steps_calls": mpc, "zero_calls": all(v == 0 for v in pol.values())}
        print(t, "policy calls", pol, "steps", r["steps"], "| mpc 3 steps", mpc, flush=True)
res["all_zero"] = all(x["zero_calls"] for x in res["tasks"].values())
json.dump(res, open(a.out, "w"), indent=1, default=float)
print("CHECK_POLICY_CALLS all_zero", res["all_zero"], flush=True)
