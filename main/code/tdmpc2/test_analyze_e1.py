"""Code-path dry run of analyze_e1.py on SYNTHETIC data (no E1 outcome is read): builds a fake <res>/jobs tree with
the E1 layout (12 tasks x 3 confirmatory k x 6 variants + none, checkpoints 1-3 x 30 episodes) whose episode records
are random draws in the exact record format of runner.py, and fake prederr / decision jobs copied from the smoke jobs
(relabelled). Usage (server): python test_analyze_e1.py <smoke out dir with jobs/> <fake res dir>
"""
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_e1 import CONF, VARS  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(123)
for sub in ("episodes", "prederr", "decision"):
    os.makedirs(f"{dst}/jobs/{sub}", exist_ok=True)
tmpl_p = json.load(gzip.open(f"{src}/jobs/prederr/smoke_prederr_mw-push_s1_k3.json.gz", "rt"))
tmpl_d = json.load(gzip.open(f"{src}/jobs/decision/smoke_decision_mw-push_s1_k3.json.gz", "rt"))


def ep_rec(task, v, ep, k):
    mw = task.startswith("mw-")
    p = {"none": 0.97, "det": 0.5 if k <= 3 else 0.9, "ind": 0.8, "shiter": 0.93, "shstep": 0.6, "shep": 0.5,
         "detshift": 0.92}[v]
    ret = float(rng.normal(900 * (p if not mw else 1.0), 40)) if not mw else float(rng.normal(1500 * p, 200))
    succ = float(rng.random() < p) if mw else 0.0
    rec = {"ep": ep, "seed": 20000 + ep, "ret": ret, "success": succ, "success_any": succ, "steps": 100 if mw else 500,
           "n_refresh": 0, "secs_loop": 1.0, "secs": 1.0}
    oei = {"det": -8.0, "shep": -6.0, "shstep": -4.0}.get(v, 0.0) if v != "none" else 0.0
    rec["oei_mean"] = {"oei": float(rng.normal(oei, 3)) if v != "none" else 0.0, "oei_w": float(rng.normal(oei, 3)),
                       "oei_top1": float(rng.normal(oei, 3)), "oei_first": float(rng.normal(oei, 3)),
                       "f_e_rand": float(rng.normal(2, 1)), "f_sd_v0": float(abs(rng.normal(15, 2))),
                       "f_spear": float(rng.random()), "l_spear": float(rng.random()), "f_flip": 1.0,
                       "f_sn_spear": 0.9, "e_chosen": 0.0}
    rec["pd_mean"] = {"dev_mean": float(abs(rng.normal(0.5, 0.1))) if v != "none" else 0.0,
                      "sn_mean": float(abs(rng.normal(0.15, 0.03))), "dev_act": 0.5, "sn_act": 0.2}
    if v != "none":
        e1 = {"det": 0.6, "detshift": 0.3}.get(v, 0.55)
        for src_ in ("perr_plan", "perr_traj"):
            rec[src_] = {"err1": float(abs(rng.normal(e1, 0.1))), "err1h": float(abs(rng.normal(e1, 0.1))),
                         "err3": float(abs(rng.normal(2 * e1, 0.2))), "err5": float(abs(rng.normal(3 * e1, 0.3))),
                         "errr": 0.1, "nseeds": 3}
    return rec


n_files = 0
for task, ks in CONF.items():
    for ck in (1, 2, 3):
        conds = [(0, "none")] + [(k, v) for k in ks for v in VARS]
        for k, v in conds:
            for c in range(6):
                eps = list(range(5 * c, 5 * c + 5))
                jid = f"eval_{task}_s{ck}_k{k}_{v}_base_c{c:02d}"
                spec = {"id": jid, "type": "episodes", "split": "eval", "task": task, "ckpt": ck, "k": k, "variant": v,
                        "eps": eps, "seed0": 20000, "scales": "main", "planner": {}, "instr": {}}
                obj = {"episodes": [ep_rec(task, v, e, k) for e in eps], "spec": spec, "env": {"gpu": "fake"}}
                with gzip.open(f"{dst}/jobs/episodes/{jid}.json.gz", "wt") as f:
                    json.dump(obj, f)
                n_files += 1
        for k in ks:
            p = dict(tmpl_p, task=task, ckpt=ck, k=k)
            with gzip.open(f"{dst}/jobs/prederr/prederr_{task}_s{ck}_k{k}.json.gz", "wt") as f:
                json.dump(p, f)
            d = dict(tmpl_d, task=task, ckpt=ck, k=k)
            with gzip.open(f"{dst}/jobs/decision/decision_{task}_s{ck}_k{k}.json.gz", "wt") as f:
                json.dump(d, f)
print("fake episode job files:", n_files)
