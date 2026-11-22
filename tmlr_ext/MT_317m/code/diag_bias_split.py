"""DIAGNOSTIC (recommended by D2, never used in the run): out-of-sample residual of the D2 bias vector.
b is computed by the D2 staged rule (bclib.bias_staged, same rule file) from the held episodes 103000-103004 only, then
one pass on the other five held episodes 103005-103009 measures the residual mean_j |m_j| / delta_j per layer (and the
same for b = 0 and for the main-run b of all ten episodes, read from <out>/bias/).
usage: python diag_bias_split.py --task <task> --out <json> [--res /workspace/atm/out]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import bclib as B
import mt14_jobs as J
import mtlib as M
import qlib as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", default=os.environ.get("ATM_OUT", "/workspace/atm/out"))
    a = ap.parse_args()
    torch.set_num_threads(1)
    agent, cfg = M.build_agent_mt()
    quant = L.Quantizer(agent.model, horizon=int(cfg.horizon))
    task, k = a.task, J.PRIMARY_K[a.task]
    quant.qrange = J.load_ranges(task)
    rule = J.bias_rule()
    files = [J.calib_path(task, s) for s in J.HELD_SEEDS]
    fit, test = files[:5], files[5:]
    t0 = time.time()
    b_fit, rep_fit, _ = B.bias_staged(agent, quant, fit, task, k, rule["stages"])
    Zu, idx, A = M.load_rows(test)
    Zu, idx, A = (torch.as_tensor(x, device=L.DEV) for x in (Zu, idx, A))
    W = {n: int(quant.mods[n].out_features) for n in L.LAYERS}
    dl = B.deltas(quant.qrange, k)
    tt = M.task_tensor(task)
    tap = B.OutTap(quant)
    out = {"task": task, "k": k, "fit_seeds": J.HELD_SEEDS[:5], "test_seeds": J.HELD_SEEDS[5:],
           "fit": {x: rep_fit[x] for x in ("converged", "bc_unconverged", "final_eta", "final_n_updates",
                                           "residual_final")}, "test": {}}
    try:
        bs = {"b0": {n: torch.zeros(W[n], device=L.DEV) for n in L.LAYERS}, "b_fit5": b_fit}
        main_b = J.bias_path(a.res, task, k)
        if os.path.exists(main_b):
            bs["b_main10"] = J.load_bias(a.res, task, k)
        for name, b in bs.items():
            m, mse, zerr = B._measure(agent.model, quant, tap, Zu, idx, A, tt, b, k, W, 8000)
            out["test"][name] = {"r_mean": {n: float(m[n].abs().mean()) / dl[n] for n in L.LAYERS},
                                 "r_max": {n: float(m[n].abs().max()) / dl[n] for n in L.LAYERS},
                                 "mse_over_delta2": {n: mse[n] / dl[n] ** 2 for n in L.LAYERS}, "latent_err": zerr}
    finally:
        tap.remove()
        quant.off()
        quant.bias = {}
    out["secs"] = time.time() - t0
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("SPLIT", task, "fit", out["fit"]["final_eta"], out["fit"]["final_n_updates"], out["fit"]["converged"],
          "| test r_mean " + " ".join(f"{nm}:" + ",".join(f"{v['r_mean'][n]:.4f}" for n in L.LAYERS)
                                      for nm, v in out["test"].items()), flush=True)


if __name__ == "__main__":
    main()
