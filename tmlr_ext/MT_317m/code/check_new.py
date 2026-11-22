"""Check 0.7-2 for E14 (new code only, one process, smoke seeds that are never analysed):
  a. det+BC with b = 0 gives the det episode bit for bit (the new branch adds nothing but the bias)
  b. the planner's global RNG is unchanged: the torch CPU and CUDA RNG states at the end of the same-seed episode are
     equal for none, det, det+BC (b = 0 and the task's real b), sh-iter, sh-ep and det+shift (the dither generator is
     private; det+BC draws nothing)
  c. sh-ep: one table per episode (n_refresh = 1 at every control step, table bytes unchanged over the 100 steps), a
     new table in another episode; call pattern per control step (18 x 512 rows per layer, 2 x 24 rows on dyn)
  d. det+BC with the real b: call pattern and nq as det, and the episode differs from det (b is applied)
  e. the det+BC bias of this task: the residual after the third sweep is below 1% of delta in every layer (bias job)
usage: python check_new.py --task <task> --seed 108001 --out <json> [--res /workspace/atm/out]
"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch

import mt14_jobs as J
import mtlib as M
import qlib as L

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--seed", type=int, default=108001)
ap.add_argument("--out", required=True)
ap.add_argument("--res", default=os.environ.get("ATM_OUT", "/workspace/atm/out"))
a = ap.parse_args()


def sha(t):
    return hashlib.sha1(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def rng_hash():
    return hashlib.sha1(torch.get_rng_state().numpy().tobytes() + torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()


torch.set_num_threads(1)
agent, cfg = M.build_agent_mt()
quant = L.Quantizer(agent.model, horizon=int(cfg.horizon))
obs_dim = int(cfg.obs_shape["state"][0])
task, k, S = a.task, J.PRIMARY_K[a.task], a.seed
quant.qrange = J.load_ranges(task)
b_real = J.load_bias(a.res, task, k)
b_zero = {n: torch.zeros_like(v) for n, v in b_real.items()}
res = {"task": task, "k": k, "seed": S, "gpu": torch.cuda.get_device_name(0), "qlib": L.__file__}


def run(v, bias=None, seed=S, log_calls=False, table_log=None):
    quant.bias = bias if bias is not None else {}
    orig = None
    if table_log is not None:
        orig = quant.begin_step

        def wrapped():
            orig()
            table_log.append([int(quant.n_refresh)] + [sha(quant.table[n]) for n in L.LAYERS])
        quant.begin_step = wrapped
    try:
        r = M.run_episode_mt(agent, quant, task, v, 0 if v == "none" else k, seed, obs_dim, log_calls=log_calls)
    finally:
        if orig is not None:
            del quant.begin_step         # back to the class method
        quant.bias = {}
    r["rng"] = rng_hash()
    r["n_refresh"] = int(quant.n_refresh)
    return r


with torch.no_grad():
    R = {}
    R["none"] = run("none")
    R["det"] = run("det", log_calls=True)
    R["detbc_b0"] = run("detbc", b_zero, log_calls=True)
    R["detbc"] = run("detbc", b_real, log_calls=True)
    R["shiter"] = run("shiter")
    tabA, tabB = [], []
    R["shep"] = run("shep", log_calls=True, table_log=tabA)
    R["shep_other_seed"] = run("shep", seed=S + 1, table_log=tabB)
    R["detshift"] = run("detshift")
    eq = lambda x, y: bool(torch.equal(x["_act"], y["_act"]) and x["_act"].numpy().tobytes() == y["_act"].numpy().tobytes())
    res["episodes"] = {key: {"ret": r["ret"], "success": r["success"], "nq": r["nq"], "n_refresh": r["n_refresh"],
                             "rng": r["rng"], "act_ms_median": r["act_ms_median"]} for key, r in R.items()}
    # a
    res["a_detbc_b0_bit_identical_to_det"] = eq(R["det"], R["detbc_b0"]) and R["det"]["ret"] == R["detbc_b0"]["ret"]
    # b
    same_seed = [key for key in R if key != "shep_other_seed"]
    res["b_global_rng_end_state_equal"] = len({R[key]["rng"] for key in same_seed}) == 1
    # c
    res["c_shep_steps"] = len(tabA)
    res["c_shep_n_refresh_values"] = sorted({row[0] for row in tabA})
    res["c_shep_table_fixed_within_episode"] = bool(len(tabA) == 100 and all(row == tabA[0] for row in tabA)
                                                    and tabA[0][0] == 1)
    res["c_shep_other_episode_fixed_and_new"] = bool(len(tabB) == 100 and all(row == tabB[0] for row in tabB)
                                                     and all(tabB[0][i] != tabA[0][i] for i in range(1, 5)))
    res["c_shep_table_shapes"] = {n: list(quant.table[n].shape) for n in L.LAYERS}
    res["c_call_patterns"] = {key: R[key].get("calls_patterns") for key in ("det", "detbc_b0", "detbc", "shep")}
    exp = {"dyn0x24": 2, "dyn0x512": 18, "dyn1x24": 2, "dyn1x512": 18, "rew0x512": 18, "rew1x512": 18}
    res["c_call_pattern_expected"] = str(sorted(exp.items()))
    res["c_call_patterns_ok"] = all(R[key].get("calls_patterns") == [str(sorted(exp.items()))]
                                    for key in ("det", "detbc_b0", "detbc", "shep"))
    # d
    res["d_detbc_nq"] = R["detbc"]["nq"]
    res["d_detbc_differs_from_det"] = not eq(R["det"], R["detbc"])
    # e
    bj = J.read_json_gz(f"{a.res}/jobs/bias/bias_{task}_k{k}.json.gz")
    res["e_bias_residual_after_sweep3"] = {n: x["mean_abs_over_delta"] for n, x in bj["sweeps"][-1]["layers"].items()}
    res["e_bias_sweeps_mean_abs_over_delta"] = [{n: x["mean_abs_over_delta"] for n, x in s["layers"].items()}
                                               for s in bj["sweeps"]]
    res["e_bias_residual_ok"] = bool(bj["residual_ok"])
    res["b_real_mean_abs_over_delta"] = {n: float(b_real[n].abs().mean()) / bj["sweeps"][-1]["layers"][n]["delta"]
                                         for n in L.LAYERS}
    res["all_pass"] = bool(res["a_detbc_b0_bit_identical_to_det"] and res["b_global_rng_end_state_equal"]
                           and res["c_shep_table_fixed_within_episode"] and res["c_shep_other_episode_fixed_and_new"]
                           and res["c_call_patterns_ok"] and res["d_detbc_nq"] == 7600 and res["e_bias_residual_ok"])
json.dump(res, open(a.out, "w"), indent=1, default=float)
print("CHECK2 " + json.dumps({k_: v for k_, v in res.items() if not k_.startswith(("episodes", "c_call_patterns"))},
                             default=float), flush=True)
