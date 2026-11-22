"""Synthetic result files in the runner's format for the analysis dry run (analyze_td.py end to end before the main run;
the numbers are random and never interpreted). Writes the job layout of all six td pods into one folder:
E11 / E12 (with some stage-2 and extension cells) / E15 / E16 episode jobs, E13 jobs, timing, prep, decisions.
Usage: python make_fake.py --out <dir> [--seed 0]
"""
import argparse
import gzip
import json
import os

import numpy as np

import plan_td as P

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
DAMAGE = {"none": 0.0, "det": 0.4, "detbc": 0.35, "shep": 0.45, "detshiftfix": 0.4, "shiterb": 0.2, "shiter": 0.05,
          "detshift": 0.05, "detnoise": 0.38, "ind": 0.25, "shstep": 0.35}


def wjson(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(obj, f, default=float)


def episodes(task, v, k, ptag, eps, rk):
    mw = P.is_mw(task)
    dmg = DAMAGE.get(v, 0.3) * (2.0 ** (P.K1[task] - k) if v != "none" else 0.0)
    if rk == "PC":
        dmg *= 0.4
    if ptag in ("it12", "it24") and v in ("shiter", "detshift"):
        dmg *= 0.5 if ptag == "it12" else 0.3
    out = []
    for ep in eps:
        if mw:
            succ = float(rng.random() < max(0.0, min(1.0, 0.98 - dmg)))
            ret = 1500 * succ + rng.normal(300, 50)
        else:
            ret = max(0.0, 950 * (1 - min(dmg, 0.95)) + rng.normal(0, 25))
            succ = 0.0
        rec = {"ep": ep, "seed": P.SEED0["eval"] + ep, "ret": float(ret), "success": succ, "success_any": succ,
               "steps": 100 if mw else 500, "n_refresh": 1, "secs_loop": float(rng.uniform(3, 25))}
        if task in P.MECH and ptag == "base":
            dr = -50.0 * dmg if v in ("det", "detbc", "shep", "detshiftfix", "detnoise") else -3.0 * dmg
            rec["oei_mean"] = {"oei": dr * 1.5 + rng.normal(0, 3), "drift": dr + rng.normal(0, 3),
                               "sel_last": dr * 0.5 + rng.normal(0, 2), "oei_indep": dr + rng.normal(0, 2),
                               "sel_indep": rng.normal(0, 1), "f_sd_v0": 10.0}
        out.append(rec)
    return out


def ep_file(j):
    s = {k: v for k, v in j.items() if k not in ("deps", "prio", "cost", "exclusive", "gate")}
    d = {"spec": s, "episodes": episodes(j["task"], j["variant"], j["k"], j["ptag"], j["eps"], j["rk"]),
         "env": {"gpu": "NVIDIA A40", "host": f"pod-{j['task']}"}, "t_start": 0.0, "secs": 1.0, "ok": True}
    wjson(f"{a.out}/jobs/episodes/{j['id']}.json.gz", d)


def e13_file(j):
    eps = []
    for ep in j["eps"]:
        cells = {}
        for v, rk, k in j["cells"]:
            if v == "none":
                continue
            key = P.cell_name(v, rk, k)
            base = DAMAGE.get(v, 0.3)
            cells[key] = {sn: {"err1": base * rng.uniform(0.5, 1.5), "err3": base * 2, "err5": base * 3,
                               "err1h": base * rng.uniform(0.5, 1.5), "errr": base, "rew_med": base / 3,
                               "rew_mean": base / 2, "e_mean": 0.1, "common_frac": 0.3, "sigma_e": base,
                               "sigma_J": 1.0, "ratio": base * rng.uniform(0.5, 1.5),
                               "rho_p": 1.0 if v in ("det", "detbc", "shep", "detshiftfix", "shstep") else 0.1,
                               "prod": 0.1, "spear": 0.9, "elite_overlap": 0.7, "nseeds": 3} for sn in ("first", "last")}
        eps.append({"ep": ep, "seed": j["seed0"] + ep, "ret": 900.0, "success": 1.0, "success_any": 1.0, "steps": 100,
                    "n_kept": 25, "state_t": list(range(8)), "secs_loop": 5.0, "secs": 9.0, "cells": cells,
                    "sets": {"first": {"sigma_J": 1.0, "gap": 0.01}, "last": {"sigma_J": 0.5, "gap": 0.05}}})
    s = {k: v for k, v in j.items() if k not in ("deps", "prio", "cost", "exclusive")}
    wjson(f"{a.out}/jobs/e13/{j['id']}.json.gz", {"spec": s, "episodes": eps, "env": {"gpu": "NVIDIA A40",
                                                                                      "host": f"pod-{j['task']}"}})


n = 0
dec = []
for tag, tasks in P.POD_TASKS.items():
    js = P.static_jobs(tasks)
    for j in js:
        if j["type"] == "episodes":
            ep_file(j)
            n += 1
        elif j["type"] == "e13":
            e13_file(j)
        elif j["type"] == "timing":
            tm = {f"it{it}|{v}": {"iterations": it, "variant": v, "n": 60, "ms_median": 10.0 * it / 6,
                                  "ms_mean": 10.0 * it / 6, "ms_p90": 12.0 * it / 6}
                  for it in j["iterations"] for v in j["variants"]}
            wjson(f"{a.out}/jobs/timing/{j['id']}.json.gz", {"spec": j, "timing": tm})
        elif j["type"] == "prep":
            wjson(f"{a.out}/jobs/prep/{j['id']}.json.gz",
                  {"spec": j, "ranges": {ly: {"hi_minus_e0_rel_width": 0.01, "lo_minus_e0": 0.0,
                                              "pc_width_mean_rel_pt": 0.5} for ly in ("dyn0", "dyn1", "rew0", "rew1")}})
        elif j["type"] == "bias":
            # D2 format; a few flagged cells (bc_unconverged) exercise the two readings of the analysis
            flag = {("mw-push", 1, "PT", 3), ("mw-window-open", 2, "PC", 2), ("cheetah-run", 3, "PT", 5)}
            info = {rk: {str(k): {"resid_mean_abs_rel_delta_max_layer": 1e-3, "n_updates": 12, "alpha": 0.5,
                                  "bc_unconverged": (j["task"], j["ckpt"], rk, k) in flag,
                                  "converged": (j["task"], j["ckpt"], rk, k) not in flag}
                         for k in range(1, 9)} for rk in ("PT", "PC")}
            wjson(f"{a.out}/jobs/bias/{j['id']}.json.gz",
                  {"spec": j, "rule": {"stages": [{"eta": 1.0}, {"eta": 0.5}, {"eta": 0.25}]}, "worst_resid": 1e-3,
                   "unconverged": [f"{rk}k{k}" for rk in info for k in info[rk] if info[rk][k]["bc_unconverged"]],
                   "cells_by_stage": {"alpha=0.5": 16}, "max_updates": 12, "bias": info})
    # adaptive jobs implied by the synthetic results (stage 2 and extension), applied until nothing new
    res = P.Results(a.out)
    known = {j["id"] for j in js}
    for j in js:
        res.add_job(j)
    for t in tasks:
        for _ in range(20):
            new, d = P.adaptive_jobs(t, res, known)
            if not new:
                break
            for j in new:
                ep_file(j)
                known.add(j["id"])
                res.add_job(j)
                n += 1
            dec += d
os.makedirs(f"{a.out}/driver", exist_ok=True)
with open(f"{a.out}/driver/decisions.jsonl", "w") as f:
    for d in dec:
        f.write(json.dumps(d, default=float) + "\n")
print(f"fake: {n} episode files, {len(dec)} adaptive decisions -> {a.out}")
