"""Progress of a manifest: done / pending jobs and episodes per (type, split, task family), failed jobs, and a rough
remaining-time estimate from the finished jobs' wall-clock (per-worker seconds / workers).
Usage: python status.py --manifest <out>/manifests/e1.jsonl --out <out> [--workers 16]
"""
import argparse
import glob
import gzip
import json
import os
from collections import defaultdict

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    jobs = [json.loads(l) for l in open(a.manifest) if l.strip()]
    agg = defaultdict(lambda: {"jobs": 0, "done": 0, "eps": 0, "eps_done": 0, "secs": 0.0})
    for j in jobs:
        fam = ("mw" if j.get("task", "").startswith("mw-") else "dmc")
        key = (j["type"], j.get("split", "-"), fam)
        g = agg[key]
        g["jobs"] += 1
        n_eps = len(j.get("eps", [])) or 1
        g["eps"] += n_eps
        p = f"{a.out}/jobs/{j['type']}/{j['id']}.json.gz"
        if os.path.exists(p):
            g["done"] += 1
            g["eps_done"] += n_eps
            try:
                g["secs"] += float(json.load(gzip.open(p, "rt"))["secs"])
            except Exception:
                pass
    remaining = 0.0
    print(f"{'type':9s} {'split':6s} {'fam':4s} {'jobs':>11s} {'episodes':>13s} {'s/ep/worker':>12s}")
    for key, g in sorted(agg.items()):
        per = g["secs"] / g["eps_done"] if g["eps_done"] else float("nan")
        if g["eps_done"]:
            remaining += per * (g["eps"] - g["eps_done"])
        print(f"{key[0]:9s} {key[1]:6s} {key[2]:4s} {g['done']:5d}/{g['jobs']:<5d} {g['eps_done']:6d}/{g['eps']:<6d} "
              f"{per:12.1f}")
    errs = glob.glob(f"{a.out}/errors/*.txt")
    pending_ids = {j["id"] for j in jobs if not os.path.exists(f"{a.out}/jobs/{j['type']}/{j['id']}.json.gz")}
    errs = [e for e in errs if os.path.basename(e)[:-4] in pending_ids]
    print(f"failed (pending) jobs with an error file: {len(errs)}" + (f", e.g. {errs[0]}" if errs else ""))
    print(f"rough remaining wall-clock with {a.workers} workers: {remaining / a.workers / 3600:.2f} h "
          f"(only job families with finished jobs counted)")
