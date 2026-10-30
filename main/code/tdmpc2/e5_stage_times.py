"""E5 stage time spans and job-second sums from the result files (reporting helper, no statistics).
usage: python e5_stage_times.py <out_e5> <v5/manifests>"""
import glob, gzip, json, os, sys, time
res, mdir = sys.argv[1], sys.argv[2]
fmt = lambda x: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(x))
for m in ("e5_stats", "e5_held", "e5_qatdata", "e5_qattrain", "e5_eval", "e5_w", "e5_qateval"):
    lo, hi, tot = float("inf"), 0.0, 0.0
    for l in open(f"{mdir}/{m}.jsonl"):
        if not l.strip():
            continue
        j = json.loads(l)
        sub = {"episodes": "episodes", "e5stats": "e5stats", "qat_train": "qat_train"}[j["type"]]
        d = json.load(gzip.open(f"{res}/jobs/{sub}/{j['id']}.json.gz", "rt"))
        lo, hi, tot = min(lo, d["t_start"]), max(hi, d["t_start"] + d["secs"]), tot + d["secs"]
    print(m, fmt(lo), fmt(hi), f"job_secs_sum={tot:.0f}")
