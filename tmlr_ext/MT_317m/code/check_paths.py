"""Check 0.7-1 (code paths unchanged): the same episodes through E10's untouched qlib.py/mtlib.py (code/e10_ref) and
through this folder's qlib.py/mtlib.py, run as two processes on the same pod. Variants none, det, sh-iter, sh-ep,
det+shift at the task's E10 primary k with the E10 range; smoke seeds (never analysed).
  run:      python check_paths.py --impl <dir with qlib.py and mtlib.py> --tasks t1,t2 --seed 108000 --out <prefix>
            -> <prefix>.npz (action sequences, rewards-free) and <prefix>.json (returns, success, nq, refreshes, RNG)
  compare:  python check_paths.py --compare <a.npz> <b.npz>   -> one JSON line, all_identical true/false
Bit identity is checked on the raw float32 bytes of every action sequence, and on the returns (exact float equality).
"""
import argparse
import json
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--impl", default="")
ap.add_argument("--tasks", default="")
ap.add_argument("--seed", type=int, default=108000)
ap.add_argument("--out", default="")
ap.add_argument("--compare", nargs=2, default=None)
a = ap.parse_args()

if a.compare:
    import numpy as np
    A, B = np.load(a.compare[0]), np.load(a.compare[1])
    ja = json.load(open(a.compare[0][:-4] + ".json"))
    jb = json.load(open(a.compare[1][:-4] + ".json"))
    res = {"keys_equal": sorted(A.files) == sorted(B.files), "episodes": {}}
    for key in sorted(set(A.files) & set(B.files)):
        x, y = A[key], B[key]
        same = x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes()
        ea, eb = ja["episodes"][key], jb["episodes"][key]
        res["episodes"][key] = {"actions_bit_identical": bool(same), "ret_equal": ea["ret"] == eb["ret"],
                                "success": [ea["success"], eb["success"]], "nq": [ea["nq"], eb["nq"]],
                                "n_refresh": [ea["n_refresh"], eb["n_refresh"]], "rng_equal": ea["rng"] == eb["rng"]}
    res["all_identical"] = bool(res["keys_equal"] and all(
        e["actions_bit_identical"] and e["ret_equal"] and e["rng_equal"] and e["nq"][0] == e["nq"][1]
        and e["n_refresh"][0] == e["n_refresh"][1] for e in res["episodes"].values()))
    res["files"] = [ja["qlib"], jb["qlib"], ja["mtlib"], jb["mtlib"]]
    print("CHECK1 " + json.dumps(res), flush=True)
    sys.exit(0)

sys.path.insert(0, a.impl)
import hashlib  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import mtlib as M  # noqa: E402
import qlib as L  # noqa: E402
import mt14_jobs as J  # noqa: E402   (constants and the E10 range loader only)

assert os.path.dirname(os.path.abspath(L.__file__)) == os.path.abspath(a.impl), L.__file__
assert os.path.dirname(os.path.abspath(M.__file__)) == os.path.abspath(a.impl), M.__file__
torch.set_num_threads(1)
agent, cfg = M.build_agent_mt()
quant = L.Quantizer(agent.model, horizon=int(cfg.horizon))
obs_dim = int(cfg.obs_shape["state"][0])
arrs, eps = {}, {}
with torch.no_grad():
    for task in a.tasks.split(","):
        k = J.PRIMARY_K[task]
        quant.qrange = J.load_ranges(task)
        for v in ("none", "det", "shiter", "shep", "detshift"):
            kk = 0 if v == "none" else k
            r = M.run_episode_mt(agent, quant, task, v, kk, a.seed, obs_dim)
            key = f"{task}__{v}__k{kk}__s{a.seed}"
            arrs[key] = r["_act"].numpy()
            eps[key] = {"ret": r["ret"], "success": r["success"], "steps": r["steps"], "nq": r["nq"],
                        "n_refresh": int(quant.n_refresh), "act_ms_median": r["act_ms_median"],
                        "rng": hashlib.sha1(torch.get_rng_state().numpy().tobytes() +
                                            torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()}
            print(key, eps[key], flush=True)
np.savez(a.out + ".npz", **arrs)
json.dump({"qlib": L.__file__, "mtlib": M.__file__, "gpu": torch.cuda.get_device_name(0), "episodes": eps},
          open(a.out + ".json", "w"), indent=1)
print("PATHS_DONE", a.out, flush=True)
