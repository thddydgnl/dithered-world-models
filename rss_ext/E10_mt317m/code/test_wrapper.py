"""E10 wrapper check (smoke): the same episode through
  (a) the official multi-task path: envs.make_env(task=mt80) -> MultitaskWrapper (all 80 members built with
      cfg.seed = seed), agent built from that cfg, env.reset(task_idx), agent.act(..., task=task_idx), env.step
  (b) mtlib: the member environment alone (padding + action cut), agent built from the stored dims (mt_dims.py),
      mtlib.run_episode_mt
must give bit-identical action sequences, rewards and success (none variant; eval_mode=True in both). Also compares the
two agents' state dicts and discount vectors.
usage: python test_wrapper.py <task1,task2,...> <seed> <out.json>
"""
import json
import sys

import torch

import mtlib as M
import qlib as L
from envs import make_env
from tdmpc2 import TDMPC2


@torch.no_grad()
def main():
    tasks, seed, out = sys.argv[1].split(","), int(sys.argv[2]), sys.argv[3]
    res = {"tasks": tasks, "seed": seed}
    cfg = M.base_cfg(M.MT, 317, seed=seed)
    env = make_env(cfg)                       # official multi-task env, members built with cfg.seed = seed
    torch.manual_seed(0)
    agent_o = TDMPC2(cfg)
    agent_o.load(M.CKPT)
    agent_o.model.eval()
    agent_m, cfg_m = M.build_agent_mt()
    so, sm = agent_o.model.state_dict(), agent_m.model.state_dict()
    diff = []
    for k in so:
        a, b = so[k], sm.get(k)
        if torch.is_tensor(a):
            if b is None or not torch.equal(a, b):
                diff.append(k)
        elif a != b:
            diff.append(k)
    res["state_dict_same_keys"] = set(so) == set(sm)
    res["state_dict_diff"] = diff
    res["discount_equal"] = bool(torch.equal(agent_o.discount, agent_m.discount))
    res["cfg_equal"] = {k: (getattr(cfg, k) == getattr(cfg_m, k)) for k in
                        ("obs_shapes", "action_dims", "episode_lengths", "obs_shape", "action_dim", "episode_length",
                         "task_dim", "latent_dim", "mlp_dim", "num_q", "iterations", "horizon", "num_samples")}
    quant = L.Quantizer(agent_m.model, horizon=int(cfg_m.horizon))
    obs_dim = cfg_m.obs_shape["state"][0]
    res["episodes"] = {}
    for task in tasks:
        tidx = M.task_index(task)
        L.seed_all(seed)
        obs, done, t, ret, acts, rews, info = env.reset(task_idx=tidx), False, 0, 0.0, [], [], {}
        obs0 = obs.clone()
        while not done:
            a = agent_o.act(obs, t0=(t == 0), eval_mode=True, task=tidx)
            obs, r, done, info = env.step(a)
            acts.append(a)
            rews.append(float(r))
            ret += float(r)
            t += 1
        A_o = torch.stack(acts)
        r_m = M.run_episode_mt(agent_m, quant, task, "none", 0, seed, obs_dim)
        same_act = bool(torch.equal(A_o, r_m["_act"]))
        res["episodes"][task] = {
            "idx": tidx, "obs0_dim": int(obs0.shape[0]), "obs0_equal": bool(torch.equal(obs0, r_m["_obs"][0])),
            "steps_official": t, "steps_ours": r_m["steps"], "ret_official": ret, "ret_ours": r_m["ret"],
            "success_official": float(info.get("success", 0.0)), "success_ours": r_m["success"],
            "actions_bit_identical": same_act,
            "max_abs_action_diff": float((A_o - r_m["_act"]).abs().max()) if A_o.shape == r_m["_act"].shape else None,
            "masked_dims_zero": bool((A_o[:, 4:] == 0).all()) if A_o.shape[1] > 4 else None,
            "act_ms_median_ours": r_m["act_ms_median"]}
        print(task, res["episodes"][task], flush=True)
    res["all_identical"] = all(e["actions_bit_identical"] and e["ret_official"] == e["ret_ours"]
                               for e in res["episodes"].values()) and not diff and res["discount_equal"]
    json.dump(res, open(out, "w"), indent=1)
    print("ALL_IDENTICAL", res["all_identical"], "state_dict_diff", diff, "discount_equal", res["discount_equal"])


if __name__ == "__main__":
    main()
