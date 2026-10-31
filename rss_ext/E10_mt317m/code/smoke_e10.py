"""E10 smoke (pod, one process, nothing else on the GPU): function checks only, never interpreted.
Seeds 99000+ (smoke split; never the held 91000-91029 or eval 90000-90089 seeds).
  1. none episodes on the given tasks (success rate of the wrapper, per-decision planning latency fp32)
  2. HeldRecorder on one none episode: arrays, and the episode equals the unrecorded one (actions bit-identical)
  3. ranges (quant_ranges_mt) from that one recording, then det / sh-iter at k=3 on one episode with the call log
     (expected per control step: 18 calls x 512 rows on dyn0, dyn1, rew0, rew1 and 2 calls x 24 rows on dyn0, dyn1)
  4. pt_err1h code path (plan_inputs_mt + pred_errors_mt) on that recording for det and sh-iter at k=3
usage: python smoke_e10.py <out.json> <task1,task2,...> <n_eps>
"""
import json
import os
import sys
import time

import numpy as np
import torch

import mtlib as M
import qlib as L


@torch.no_grad()
def main():
    out, tasks, n_eps = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3])
    torch.set_num_threads(1)
    agent, cfg = M.build_agent_mt()
    quant = L.Quantizer(agent.model, horizon=int(cfg.horizon))
    obs_dim = cfg.obs_shape["state"][0]
    res = {"gpu": torch.cuda.get_device_name(0), "obs_dim": obs_dim, "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
           "mods": {n: [quant.mods[n].in_features, quant.mods[n].out_features, type(quant.mods[n].act).__name__]
                    for n in L.LAYERS}, "none": {}}
    # 1. none episodes
    acts_ref = {}
    for task in tasks:
        eps = []
        for i in range(n_eps):
            seed = 99000 + i
            t0 = time.time()
            r = M.run_episode_mt(agent, quant, task, "none", 0, seed, obs_dim)
            eps.append({"seed": seed, "success": r["success"], "success_any": r["success_any"], "ret": r["ret"],
                        "steps": r["steps"], "secs": time.time() - t0, "act_ms_median": r["act_ms_median"],
                        "act_ms_mean": r["act_ms_mean"], "act_ms_first": r["act_ms_first"]})
            if i == 0:
                acts_ref[task] = r["_act"]
            print(task, eps[-1], flush=True)
        res["none"][task] = {"episodes": eps, "success_rate": float(np.mean([e["success"] for e in eps])),
                             "secs_mean": float(np.mean([e["secs"] for e in eps[1:] or eps])),
                             "act_ms_median": float(np.median([e["act_ms_median"] for e in eps[1:] or eps]))}
    # 2. recorder invariance + arrays
    task = tasks[0]
    rec = M.HeldRecorder(agent, seed_rec=0)
    try:
        r = M.run_episode_mt(agent, quant, task, "none", 0, 99000, obs_dim, recorder=rec)
    finally:
        rec.detach()
    arr = rec.arrays()
    os.makedirs("/workspace/e10/smoke", exist_ok=True)
    f = "/workspace/e10/smoke/rec_smoke.npz"
    np.savez_compressed(f, **arr)
    res["recorder"] = {"actions_identical_to_unrecorded": bool(torch.equal(r["_act"], acts_ref[task])),
                       "shapes": {k: list(v.shape) for k, v in arr.items()}, "pt_t": arr["pt_t"].tolist(),
                       "estimate_value_restored": "_estimate_value" not in agent.__dict__}
    print("recorder", res["recorder"], flush=True)
    # 3. ranges + det / sh-iter episodes with the call log
    qr, info = M.quant_ranges_mt(agent, quant, [f], task)
    res["ranges_smoke"] = {"quant": qr, "info": info}
    print("ranges", qr, flush=True)
    quant.qrange = qr
    res["quant_eps"] = {}
    for v in ("det", "shiter"):
        t0 = time.time()
        rq = M.run_episode_mt(agent, quant, task, v, 3, 99000, obs_dim, log_calls=True)
        res["quant_eps"][v] = {"success": rq["success"], "ret": rq["ret"], "nq": rq["nq"], "secs": time.time() - t0,
                               "act_ms_median": rq["act_ms_median"], "calls_patterns": rq["calls_patterns"],
                               "differs_from_none": not bool(torch.equal(rq["_act"], acts_ref[task]))}
        print(v, res["quant_eps"][v], flush=True)
    # 4. pt_err1h path on the smoke recording
    cands = [(torch.as_tensor(arr["pt_z"][i], device=L.DEV), torch.as_tensor(arr["pt_cand"][i], device=L.DEV))
             for i in range(len(arr["pt_t"]))]
    tt = M.task_tensor(task)
    t0 = time.time()
    Z0, Ap = M.plan_inputs_mt(agent, cands, tt, K=5)
    seeds = [9_000_001 + 1000 * 99000 + j for j in range(3)]
    res["perr_smoke"] = {v: M.pred_errors_mt(agent, cfg, quant, v, 3, Z0, Ap, seeds, tt, h1=int(cfg.horizon))
                         for v in ("det", "shiter")}
    res["perr_smoke_secs"] = time.time() - t0
    res["perr_inputs"] = [list(Z0.shape), list(Ap.shape)]
    print("perr", res["perr_smoke"], res["perr_smoke_secs"], flush=True)
    res["gpu_mem_max_gb"] = torch.cuda.max_memory_allocated() / 1e9
    json.dump(res, open(out, "w"), indent=1)
    print("SMOKE_DONE")


if __name__ == "__main__":
    main()
