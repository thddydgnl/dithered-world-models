"""E5 baseline selection (DESIGN_main v5 7.3): for every E5 cell (task, k), the PTQ baseline B with the smallest det
drop D_det(B) on the held split (seeds 12000-12019, checkpoints 1-3, 60 pairs; none from the same held jobs); a tie
goes to the earlier baseline in the v5 list (perchan, mseclip, smooth, mixed). D as in v4 6.1: Meta-World success-rate
difference, DMC relative return drop. Writes <res>/e5/selection.json, which the E5 evaluation jobs ('ptq': 'best')
read at run time. Refuses (exit 1) if any cell is incomplete.
usage: python e5_select.py --res <out_e5> --manifest <e5_held manifest>
"""
import argparse
import gzip
import json
import os
from collections import defaultdict

import numpy as np

PTQ = ["perchan", "mseclip", "smooth", "mixed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--manifest", required=True)
    a = ap.parse_args()
    jobs = [json.loads(l) for l in open(a.manifest) if l.strip()]
    vals = defaultdict(dict)       # (task, k, B) -> {(ckpt, ep): metric}
    missing = []
    for j in jobs:
        p = f"{a.res}/jobs/episodes/{j['id']}.json.gz"
        if not os.path.exists(p):
            missing.append(j["id"])
            continue
        mw = j["task"].startswith("mw-")
        key = (j["task"], int(j["k"]), j.get("ptq", "none"))
        for e in json.load(gzip.open(p, "rt"))["episodes"]:
            vals[key][(int(j["ckpt"]), int(e["ep"]))] = float(e["success"] if mw else e["ret"])
    if missing:
        print(f"REFUSED: {len(missing)} held jobs missing, e.g. {missing[:3]}")
        raise SystemExit(1)
    cells = sorted({(t, k) for (t, k, b) in vals if b != "none"})
    out = {"rule": "argmin_B D_det(B) on held (12000-12019, 60 pairs); tie -> earlier in " + ",".join(PTQ), "cells": {}}
    for t, k in cells:
        mw = t.startswith("mw-")
        none = vals[(t, 0, "none")]
        prs = sorted(none)
        r0 = float(np.mean([none[x] for x in prs]))
        D = {}
        for b in PTQ:
            v = vals[(t, k, b)]
            if sorted(v) != prs:
                print(f"REFUSED: pairs differ for {t} k={k} {b}")
                raise SystemExit(1)
            rv = float(np.mean([v[x] for x in prs]))
            D[b] = (r0 - rv) if mw else (r0 - rv) / r0
        best = min(PTQ, key=lambda b: (D[b], PTQ.index(b)))
        out["cells"][f"{t}|{k}"] = {"best": best, "D_det": D, "none_mean": r0, "n_pairs": len(prs)}
    os.makedirs(f"{a.res}/e5", exist_ok=True)
    p = f"{a.res}/e5/selection.json"
    tmp = p + f".tmp{os.getpid()}"
    json.dump(out, open(tmp, "w"), indent=1)
    os.replace(tmp, p)
    print(json.dumps({c: v["best"] for c, v in out["cells"].items()}))
    print("E5_SELECT_DONE")


if __name__ == "__main__":
    main()
