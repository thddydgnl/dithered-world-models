"""Run chosen E14 main-run jobs in this process (phase-A checks: the held and bias jobs of a task before the driver
starts; results go to the same <out>/jobs files the driver uses, so the driver skips them later).
usage: python run_jobs.py --out /workspace/atm/out --task <task> --stages held,bias
"""
import argparse
import json

import mt14_driver as Dr
import mt14_jobs as J

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--task", required=True)
ap.add_argument("--stages", default="held,bias")
a = ap.parse_args()
J.init_worker()
k = J.PRIMARY_K[a.task]
specs = []
for st in a.stages.split(","):
    specs += Dr.held_specs(a.task) if st == "held" else [Dr.bias_spec(a.task, k)] if st == "bias" else []
for s in specs:
    r = J.run_job(s, a.out)
    print("JOB " + json.dumps(r), flush=True)
    if not r["ok"]:
        raise SystemExit(f"job failed: {r['id']}")
print("RUN_JOBS_DONE", flush=True)
