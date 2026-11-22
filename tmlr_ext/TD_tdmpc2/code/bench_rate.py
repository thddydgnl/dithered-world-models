"""Throughput of a fixed-time bench run (one-episode jobs): completion times from the result files.
rate = (n - W) / (t_n - t_W) episodes per second after the first W completions (steady state); mean seconds per
episode (secs_loop, under the bench's concurrency) per (task, planner tag, instrumented, cell).
Usage: python bench_rate.py --res <dir> --prefix bench_ --workers 8
"""
import argparse
import glob
import gzip
import json
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--res", required=True)
ap.add_argument("--prefix", required=True)
ap.add_argument("--workers", type=int, required=True)
a = ap.parse_args()
ends, secs = [], defaultdict(list)
for f in glob.glob(f"{a.res}/jobs/episodes/{a.prefix}*.json.gz"):
    d = json.load(gzip.open(f, "rt"))
    s = d["spec"]
    ends.append(d["t_start"] + d["secs"])
    grp = f"{s['task']}|{s.get('ptag')}|{'instr' if s.get('instr') else 'plain'}"
    cell = "none" if s["variant"] == "none" else f"{s.get('rk')}k{s['k']}_{s['variant']}"
    secs[grp].append(d["episodes"][0]["secs_loop"])
    secs[f"{grp}|{cell}"].append(d["episodes"][0]["secs_loop"])
ends.sort()
n, W = len(ends), a.workers
rate = (n - W) / (ends[-1] - ends[W - 1]) if n > 2 * W else float("nan")
out = {"n": n, "workers": W, "eps_per_hour": rate * 3600, "span_s": ends[-1] - ends[0] if ends else 0,
       "mean_secs_per_episode": {k: [round(sum(x) / len(x), 2), len(x)] for k, x in sorted(secs.items())
                                 if k.count("|") == 2}}
print(json.dumps(out, indent=1))
