"""E10: build the official mt80 multi-task environment once (envs.make_env with task=mt80 -> make_multitask_env ->
MultitaskWrapper) and store the cfg fields it fills in (per-task observation dims, action dims, episode lengths, padded
observation shape, max action dim), so that workers can build the mt80-317M agent without building 80 environments.
Also inspects the checkpoint (format, key names, shapes, stored buffers) and the agent built from the stored dims.
usage: python mt_dims.py <dims.json> <report.json>
"""
import json
import sys
import time

import torch

import mtlib as M
import qlib as L
from envs import make_env


def main():
    out, rep_path = sys.argv[1], sys.argv[2]
    rep = {}
    t0 = time.time()
    cfg = M.base_cfg(M.MT, 317, seed=0)
    env = make_env(cfg)          # official multi-task build (all 80 members)
    dims = M.dims_from_cfg(cfg)
    rep["env_build_secs"] = time.time() - t0
    json.dump(dims, open(out, "w"), indent=1)
    rep["obs_shape"], rep["action_dim"], rep["episode_length"] = dims["obs_shape"], dims["action_dim"], \
        dims["episode_length"]
    rep["members"] = {t: {"idx": i, "obs": dims["obs_shapes"][i], "act": dims["action_dims"][i],
                          "ep_len": dims["episode_lengths"][i]} for i, t in enumerate(dims["tasks"])}
    del env

    # checkpoint as stored
    sd = torch.load(M.CKPT, map_location="cpu", weights_only=False)
    rep["ckpt_top_keys"] = sorted(sd.keys()) if isinstance(sd, dict) else str(type(sd))
    sd = sd["model"] if "model" in sd else sd
    rep["ckpt_n_keys"] = len(sd)
    rep["ckpt_new_api"] = "_detach_Qs_params.0.weight" in sd
    rep["ckpt_has_action_masks"] = "_action_masks" in sd
    rep["ckpt_shapes"] = {k: list(v.shape) for k, v in sd.items() if torch.is_tensor(v) and not k.startswith(
        ("_Qs.", "_target_Qs"))}
    rep["ckpt_q_keys_sample"] = [k for k in sd if k.startswith(("_Qs", "_target_Qs"))][:12]
    n_params_ckpt = sum(v.numel() for k, v in sd.items() if torch.is_tensor(v) and not k.startswith("_target_Qs"))
    rep["ckpt_numel_without_target_Qs"] = int(n_params_ckpt)
    masks_ckpt = sd.get("_action_masks")

    # agent from the stored dims
    agent, acfg = M.build_agent_mt(out)
    m = agent.model
    rep["agent_total_params"] = int(m.total_params)
    rep["agent_iterations"] = int(acfg.iterations)
    rep["agent_task_dim"] = int(acfg.task_dim)
    rep["agent_latent_dim"], rep["agent_mlp_dim"] = int(acfg.latent_dim), int(acfg.mlp_dim)
    rep["dynamics"] = str(m._dynamics)
    rep["reward"] = str(m._reward)
    rep["encoder"] = str(m._encoder)
    rep["quant_layers"] = {"dyn0": str(m._dynamics[0]), "dyn1": str(m._dynamics[1]),
                           "rew0": str(m._reward[0]), "rew1": str(m._reward[1])}
    am = m._action_masks.detach().cpu()
    if masks_ckpt is not None:
        rep["action_masks_equal_ckpt"] = bool(torch.equal(am, masks_ckpt.cpu().float()))
    cand = ["mw-push", "mw-pick-place", "mw-door-open", "mw-drawer-open", "mw-button-press-topdown",
            "mw-peg-insert-side", "mw-window-open", "mw-sweep-into", "mw-reach", "mw-button-press", "mw-door-close",
            "mw-drawer-close", "mw-faucet-open", "mw-faucet-close", "mw-plate-slide", "mw-coffee-button",
            "mw-handle-press", "mw-window-close"]
    rep["candidates"] = {t: {"idx": M.task_index(t), "mask": am[M.task_index(t)].tolist(),
                             "discount": float(agent.discount[M.task_index(t)]),
                             "obs": dims["obs_shapes"][M.task_index(t)], "act": dims["action_dims"][M.task_index(t)],
                             "ep_len": dims["episode_lengths"][M.task_index(t)]} for t in cand}
    q = L.Quantizer(m, horizon=int(acfg.horizon))
    rep["quantizer_mods"] = {n: [int(q.mods[n].in_features), int(q.mods[n].out_features)] for n in L.LAYERS}
    q.remove()
    rep["gpu"] = torch.cuda.get_device_name(0)
    rep["gpu_mem_alloc_gb"] = torch.cuda.memory_allocated() / 1e9
    json.dump(rep, open(rep_path, "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("members", "ckpt_shapes")}, indent=1))


if __name__ == "__main__":
    main()
