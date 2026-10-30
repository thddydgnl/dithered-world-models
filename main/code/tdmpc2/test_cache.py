"""E0 check of the worker agent cache (runner.get_agent keeps 2 agents and evicts the oldest): instrumented episode
jobs for three (task, ckpt) pairs in one process must all record the planner's chosen elite (no NaN e_chosen) and give
the same returns as before (smoke seed 19000; det k=3 on mw-push s1 = 116.36395911313593 from test_equiv / smoke).
Usage (server): python test_cache.py <out>
"""
import json
import math
import sys

import torch

import runner

out = sys.argv[1]
specs = [("mw-push", 1), ("mw-pick-place", 1), ("mw-push", 2), ("mw-push", 1)]
with torch.no_grad():
    for task, ckpt in specs:
        spec = {"type": "episodes", "task": task, "ckpt": ckpt, "k": 3, "variant": "det", "eps": [0], "seed0": 19000,
                "scales": "r28" if ckpt == 1 else "main", "planner": {},
                "instr": {"oei_stride": 4, "pd_stride": 10, "perr": False}}
        r = runner.job_episodes(spec, out)["episodes"][0]
        ec = r["oei_steps"]["e_chosen"]
        n_nan = sum(1 for x in ec if x is None or (isinstance(x, float) and math.isnan(x)))
        print(json.dumps({"task": task, "ckpt": ckpt, "ret": r["ret"], "n_steps_rec": len(ec), "n_nan_e_chosen": n_nan,
                          "cache": [list(k) for k in runner._cache.keys()]}), flush=True)
