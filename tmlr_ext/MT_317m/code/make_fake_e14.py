"""Fake E14 result directories for the analysis dry run (never analysed as data). Same file layout and record fields
as mt14_jobs.py: <out>/<tag>/jobs/{held,bias,eval}/<id>.json.gz, tasks split over two "pods" (4090 layout).
Successes are coupled across variants through one uniform per (task, seed) (paired episodes), with made-up rates.
usage: python make_fake_e14.py --out <dir> [--drop_task <task>]   (drop_task: leave that task's det+BC jobs out)
"""
import argparse
import hashlib
import os

import numpy as np

import mt14_driver as Dr
import mt14_jobs as J

RATES = {"none": 0.95, "det": 0.35, "shiter": 0.90, "shep": 0.50, "detshift": 0.85, "detbc": 0.45}


def episode(seed, succ, v, rng):
    h = hashlib.sha1(f"{seed}".encode()).hexdigest()
    return {"seed": seed, "ret": float(1000 * succ + rng.normal(500, 50)), "success": float(succ),
            "success_any": float(succ), "steps": 100, "nq": 0 if v == "none" else 7600,
            "n_refresh": 1 if v == "shep" else 0, "act_ms_mean": 75.0, "act_ms_median": 75.0,
            "act_sha1": hashlib.sha1(f"{seed}{v}{succ}".encode()).hexdigest(), "rng_sha1": h, "secs": 8.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop_task", default="")
    a = ap.parse_args()
    rng = np.random.default_rng(123)
    pods = {"mt.1": J.TASKS8[:4], "mt.2": J.TASKS8[4:]}
    for tag, tasks in pods.items():
        out = f"{a.out}/{tag}"
        for ti, t in enumerate(tasks):
            k = J.PRIMARY_K[t]
            u = {s: rng.random() for s in J.EVAL_SEEDS}
            shift = 0.1 * (ti - 1.5)
            for s in Dr.held_specs(t):
                eps = [episode(sd, 1.0, "none", rng) for sd in s["seeds"]]
                J.write_json_gz(J.result_path(out, s), {"episodes": eps, "spec": s, "env": {"gpu": "FAKE"}, "ok": True})
            B = Dr.bias_spec(t, k)
            LY = ("dyn0", "dyn1", "rew0", "rew1")
            unconv = t == "mw-pick-place"           # one flagged task exercises the two-way det+BC report
            lay = lambda v: {n: v for n in LY}
            passes = [{"pass": i, "r_mean": lay(0.5 * 0.6 ** i), "r_max": lay(2 * 0.6 ** i),
                       "mse_over_delta2": lay(1.0 * 0.7 ** i), "b_mean_abs_over_delta": lay(0.3), "latent_err": [0.1, 0.2, 0.3]}
                      for i in range(12)]
            stages = [{"eta": 1.0, "until_below": 0.01, "max_sweeps": 60, "converged": False, "n_updates": 60,
                       "passes": passes[:3], "worst_r_last": 1.3},
                      {"eta": 0.5, "until_below": 0.01, "max_sweeps": 60, "converged": not unconv, "n_updates": 11,
                       "passes": passes, "worst_r_last": 0.5 * 0.6 ** 11}]
            rep = {"task": t, "k": k, "n_rows": 96000, "rule_id": "D2", "converged": not unconv,
                   "bc_unconverged": unconv, "final_stage": 1, "final_eta": 0.5, "final_n_updates": 11,
                   "residual_final": lay(0.002), "residual_final_max": lay(0.02), "residual_ok": not unconv,
                   "b_mean_abs_over_delta": lay(0.3), "mse_over_delta2_b0": lay(1.0), "mse_over_delta2_final": lay(0.2),
                   "latent_err_b0": [0.1, 0.2, 0.3], "latent_err_final": [0.05, 0.1, 0.15], "stages": stages,
                   "spec": B, "env": {"gpu": "FAKE"}, "ok": True}
            J.write_json_gz(J.result_path(out, B), rep)
            for v in J.VARIANTS:
                if v == "detbc" and t == a.drop_task:
                    continue
                p = min(max(RATES[v] + (0 if v == "none" else shift), 0.0), 1.0)
                for s in Dr.eval_specs(t, v, k):
                    eps = [episode(sd, float(u[sd] < p), v, rng) for sd in s["seeds"]]
                    J.write_json_gz(J.result_path(out, s), {"episodes": eps, "spec": s, "env": {"gpu": "FAKE"},
                                                            "ok": True})
    print("FAKE_DONE", a.out)


if __name__ == "__main__":
    main()
