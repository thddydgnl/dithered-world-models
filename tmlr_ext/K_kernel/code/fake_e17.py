"""Synthetic result files for the dry run of analyze_e17.py (before the main run; the numbers are random and never
analysed): one result file per job of main.jsonl in <dst>/jobs/episodes/ with the runner's format, random successes /
returns (Meta-World: Bernoulli, DMC: normal around 900 scaled by a random damage per cell), plus a synthetic fake-quant
CSV for the --fq_csv path (pt and pc rows).
usage: python fake_e17.py --manifest <out>/manifests/main.jsonl --dst /workspace/atm/big/fake [--seed 0]
"""
import argparse
import csv
import gzip
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
jobs = [json.loads(line) for line in open(a.manifest) if line.strip()]
os.makedirs(f"{a.dst}/jobs/episodes", exist_ok=True)
damage = {}
for j in jobs:
    key = (j["task"], j["prec"], j["k"], j["variant"])
    if key not in damage:
        damage[key] = 0.0 if j["prec"] == "fp32" else float(rng.uniform(0, 0.6))
    eps = []
    for ep in j["eps"]:
        if j["task"].startswith("mw-"):
            s = float(rng.random() > damage[key])
            r = 1500 * s + float(rng.normal(300, 50))
        else:
            s = 0.0
            r = float(max(0.0, rng.normal(900 * (1 - damage[key]), 40)))
        eps.append({"ep": ep, "seed": j["seed0"] + ep, "ret": r, "success": s, "success_any": s, "steps": 100,
                    "n_refresh": 0, "secs_loop": 1.0, "secs": 1.0})
    res = {"episodes": eps, "spec": j, "env": {"host": "fakehost", "gpu": "NVIDIA A40", "torch": "x", "cvd": "0",
                                               "mps": None, "pid": 0}, "t_start": 0.0, "secs": 1.0, "ok": True}
    if "e7" in j:
        res["e7"] = {"prec": j["e7"], "variant": j["variant"], "k": j["k"]}
    with gzip.open(f"{a.dst}/jobs/episodes/{j['id']}.json.gz", "wt") as f:
        json.dump(res, f)
rows = []
for task in ("mw-push", "mw-pick-place", "cheetah-run", "walker-walk"):
    kprim = {"mw-push": 3, "mw-pick-place": 3, "cheetah-run": 5, "walker-walk": 3}[task]
    cells = [("pt", 0, "none"), ("pt", kprim, "det"), ("pt", kprim, "detshift")] + \
            [("pc", k, v) for k in (2, 3) for v in ("det", "shiter")]
    for rg, k, v in cells:
        dmg = 0.0 if v == "none" else float(rng.uniform(0, 0.5))
        for c in (1, 2, 3):
            for s in range(100000, 100030):
                if task.startswith("mw-"):
                    ok = float(rng.random() > dmg)
                    rows.append([task, c, s, v, k, ok, 1500 * ok, rg])
                else:
                    rows.append([task, c, s, v, k, 0.0, float(max(0.0, rng.normal(900 * (1 - dmg), 40))), rg])
with open(f"{a.dst}/fq_fake.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["task", "ckpt", "seed", "variant", "k", "success", "ret", "range"])
    w.writerows(rows)
with open(f"{a.dst}/fq_fake_td.csv", "w", newline="") as f:     # the td worker's td_episodes.csv layout
    w = csv.writer(f)
    w.writerow(["task", "ckpt", "ep", "seed", "cell", "ptag", "ret", "success", "secs", "gpu"])
    for task, c, s, v, k, ok, ret, rg in rows:
        cell = "none" if v == "none" else f"{rg.upper()}k{k}_{v}"
        w.writerow([task, c, s - 100000, s, cell, "base", ret, ok, 1.0, "NVIDIA A40"])
        w.writerow([task, c, s - 100000, s, cell, "it12", ret, ok, 1.0, "NVIDIA A40"])   # must be ignored
print(f"fake results: {len(jobs)} jobs -> {a.dst}; fake fq rows {len(rows)}")
