"""DESIGN_tmlr 0.7 check 1 (code paths unchanged): the same episodes through a given qlib (the untouched E1 copy in
inputs/e1_reference or this folder's TMLR qlib) on the same pod; the two runs must give bit-identical episodes for
none, det, sh-iter, sh-ep, det+shift (mw-push k = 3 and cheetah-run k = 5, checkpoint 1, seed 12000, E0 ranges).
Per episode: return, success, n_refresh, the sha256 of the executed action and reward sequences. Prints one JSON line.
Usage: python check_e1_paths.py --qlib_dir <dir> [--tasks mw-push:3,cheetah-run:5]
"""
import argparse
import gzip
import hashlib
import json
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--qlib_dir", required=True)
ap.add_argument("--ranges", default="/workspace/atm/code/inputs/ranges")
ap.add_argument("--tasks", default="mw-push:3,cheetah-run:5")
ap.add_argument("--seed", type=int, default=12000)
a = ap.parse_args()
sys.path.insert(0, a.qlib_dir)
import torch  # noqa: E402
import qlib as L  # noqa: E402

torch.set_num_threads(1)
out = {"qlib": L.__file__}
for tk in a.tasks.split(","):
    task, k = tk.split(":")
    k = int(k)
    agent, cfg = L.build_agent(task, 1)
    quant = L.Quantizer(agent.model, horizon=int(cfg.horizon))
    qr = json.load(gzip.open(f"{a.ranges}/ranges_{task}_s1.json.gz", "rt"))["quant"]
    quant.qrange = {kk: (float(v[0]), float(v[1])) for kk, v in qr.items()}
    with torch.no_grad():
        for v in ("none", "det", "shiter", "shep", "detshift"):
            kk = 0 if v == "none" else k
            r = L.run_episode(agent, cfg, quant, v, kk, a.seed)
            ha = hashlib.sha256(r["_act"].numpy().tobytes()).hexdigest()[:16]
            hr = hashlib.sha256(r["_rew"].numpy().tobytes()).hexdigest()[:16]
            out[f"{task}|{v}_k{kk}"] = [r["ret"], r["success"], r["n_refresh"], ha, hr]
    quant.remove()
print("E1PATHS " + json.dumps(out), flush=True)
