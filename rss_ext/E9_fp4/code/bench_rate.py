"""Throughput of a fixed-time bench run (one-episode jobs): completion times from the result files.
rate = (n - W) / (t_n - t_W) episodes per second after the first W completions (steady state), per variant mix.
Usage: python bench_rate.py --res /workspace/results --prefix bench_mw --workers 16
"""
import argparse
import glob
import gzip
import json
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--res", required=True)
ap.add_argument("--prefix", required=True)
ap.add_argument("--workers", type=int, required=True)
a = ap.parse_args()
ends, per_var, secs = [], Counter(), {}
for f in glob.glob(f"{a.res}/jobs/episodes/{a.prefix}*.json.gz"):
    d = json.load(gzip.open(f, "rt"))
    ends.append(d["t_start"] + d["secs"])
    v = d["spec"]["variant"]
    per_var[v] += 1
    secs.setdefault(v, []).append(d["episodes"][0]["secs_loop"])
ends.sort()
n, W = len(ends), a.workers
rate = (n - W) / (ends[-1] - ends[W - 1]) if n > 2 * W else float("nan")
out = {"prefix": a.prefix, "n": n, "workers": W, "eps_per_hour": rate * 3600,
       "span_s": ends[-1] - ends[0] if ends else 0, "per_variant": dict(per_var),
       "mean_secs_per_episode": {v: sum(x) / len(x) for v, x in secs.items()}}
print(json.dumps(out, indent=1))
