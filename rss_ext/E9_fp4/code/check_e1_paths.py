"""E1 code paths unchanged: the same episodes through a given qlib (the untouched E1 copy in inputs/e1_reference or
this folder's E9 qlib) on the same pod; the two runs must give bit-identical returns. Prints one JSON line.
Usage: python check_e1_paths.py --qlib_dir <dir>
"""
import argparse
import gzip
import json
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--qlib_dir", required=True)
ap.add_argument("--ranges", default="/root/e9/code/inputs/ranges")
a = ap.parse_args()
sys.path.insert(0, a.qlib_dir)
import torch  # noqa: E402
import qlib as L  # noqa: E402

torch.set_num_threads(1)
agent, cfg = L.build_agent("mw-push", 1)
quant = L.Quantizer(agent.model, horizon=int(cfg.horizon))
qr = json.load(gzip.open(f"{a.ranges}/ranges_mw-push_s1.json.gz", "rt"))["quant"]
quant.qrange = {k: (float(v[0]), float(v[1])) for k, v in qr.items()}
out = {"qlib": L.__file__}
with torch.no_grad():
    for v, k in (("none", 0), ("det", 3), ("shiter", 3), ("shstep", 3)):
        r = L.run_episode(agent, cfg, quant, v, k, 12000)
        out[f"{v}_k{k}"] = [r["ret"], r["success"], r["n_refresh"]]
print("E1PATHS " + json.dumps(out), flush=True)
