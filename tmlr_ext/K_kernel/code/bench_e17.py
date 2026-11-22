"""Throughput of the E17 closed-loop precisions (0.7 check 4) from the pre driver's jobs (bench seeds 108000+ and the
FP32 none eval jobs, all run with the same worker count under MPS): mean seconds per episode (secs_loop) per family
(Meta-World / DMC) and precision type, the wall-clock episode rate of the run, and the time estimate of the remaining
main manifest = sum over its unfinished jobs of episodes x mean secs of the job's type / workers.
usage: python bench_e17.py --res /workspace/atm/out --workers 8 [--out <json>]
"""
import argparse
import glob
import gzip
import json
import os
from collections import defaultdict

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--res", required=True)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--out", default=None)
a = ap.parse_args()
secs, ends, starts = defaultdict(list), [], []
for f in glob.glob(f"{a.res}/jobs/episodes/*.json.gz"):
    d = json.load(gzip.open(f, "rt"))
    s = d["spec"]
    fam = "mw" if s["task"].startswith("mw-") else "dmc"
    typ = f"{s.get('prec', s.get('e7', 'fp32'))}_{s['variant']}"
    for e in d["episodes"]:
        secs[(fam, typ)].append(float(e["secs_loop"]))
    ends.append(d["t_start"] + d["secs"])
    starts.append(d["t_start"])
calib = defaultdict(list)
for f in glob.glob(f"{a.res}/jobs/calib/*.json.gz"):
    d = json.load(gzip.open(f, "rt"))
    fam = "mw" if d["spec"]["task"].startswith("mw-") else "dmc"
    calib[fam].append(float(d["secs"]))
mean = {f"{k[0]}|{k[1]}": float(np.mean(v)) for k, v in secs.items()}
ratio = {}
for k, v in mean.items():
    fam = k.split("|")[0]
    base = mean.get(f"{fam}|fp32_none")
    if base:
        ratio[k] = v / base
est = None
mpath = f"{a.res}/manifests/main.jsonl"
if os.path.exists(mpath):
    tot, n_left, eps_left = 0.0, 0, 0
    for line in open(mpath):
        j = json.loads(line)
        if os.path.exists(f"{a.res}/jobs/episodes/{j['id']}.json.gz"):
            continue
        fam = "mw" if j["task"].startswith("mw-") else "dmc"
        typ = f"{j['prec']}_{j['variant']}"
        typ_b = typ.replace("intpc", "intk")   # per-channel cells: the intk cost of the same rounding (no bench)
        m = mean.get(f"{fam}|{typ}") or mean.get(f"{fam}|{typ_b}") or mean.get(f"{fam}|intk_det")
        tot += len(j["eps"]) * (m or 0.0)
        n_left += 1
        eps_left += len(j["eps"])
    est = {"jobs_left": n_left, "episodes_left": eps_left, "worker_seconds": tot,
           "wall_hours_at_workers": tot / a.workers / 3600}
span = (max(ends) - min(starts)) if ends else 0.0
out = {"mean_secs_per_episode": mean, "ratio_to_fp32_none": ratio,
       "n_episodes": {f"{k[0]}|{k[1]}": len(v) for k, v in secs.items()},
       "calib_secs_mean": {k: float(np.mean(v)) for k, v in calib.items()},
       "wall_span_s": span, "episodes_total": sum(len(v) for v in secs.values()),
       "episodes_per_hour_wall": (sum(len(v) for v in secs.values()) / span * 3600) if span else None,
       "workers": a.workers, "main_estimate": est}
print(json.dumps(out, indent=1))
if a.out:
    json.dump(out, open(a.out, "w"), indent=1)
