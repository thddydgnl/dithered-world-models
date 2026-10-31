"""E9 smoke checks on the pod's result folder (prints a report and writes <res>/smoke/check_smoke.json).
- repro: this code on this pod vs the stored E0 held episodes (A5000, E0 ranges): none and uniform det, seed 12000
- FP4 smoke: every FP4 variant on mw-push / cheetah-run (seed 89000) with the Triton kernel and the PyTorch reference;
  the two must give the identical episode (return, success); forward-call patterns, time per episode
- heldperr smoke: per-variant planning-time / trajectory errors of one held episode per task
Usage: python check_smoke.py --res /workspace/results
"""
import argparse
import glob
import gzip
import json
import os

E0 = {("mw-push", "none"): (1784.2798, 1.0), ("mw-push", "det"): (946.2061, 0.0),
      ("cheetah-run", "none"): (861.5378, 0.0), ("cheetah-run", "det"): (321.3105, 0.0)}


def load(path):
    return json.load(gzip.open(path, "rt"))


ap = argparse.ArgumentParser()
ap.add_argument("--res", required=True)
a = ap.parse_args()
out = {"repro": {}, "fp4": {}, "held": {}}
for f in sorted(glob.glob(f"{a.res}/jobs/episodes/repro_*.json.gz")):
    d = load(f)
    s, e = d["spec"], d["episodes"][0]
    exp = E0[(s["task"], s["variant"])]
    ok = round(e["ret"], 4) == exp[0] and e["success"] == exp[1]
    out["repro"][s["id"]] = {"ret": e["ret"], "success": e["success"], "e0_ret": exp[0], "match": ok,
                             "secs": e["secs_loop"], "gpu": d["env"]["gpu"]}
    print(f"REPRO {'OK ' if ok else 'DIFF'} {s['id']}: ret {e['ret']:.4f} (E0 {exp[0]}) succ {e['success']} "
          f"{e['secs_loop']:.1f}s", flush=True)
R = {}
for f in sorted(glob.glob(f"{a.res}/jobs/episodes/smoke_*.json.gz")):
    d = load(f)
    s, e = d["spec"], d["episodes"][0]
    R[(s["task"], s["variant"], s["fp4_impl"])] = (e, d)
for (task, v, impl), (e, d) in sorted(R.items()):
    if impl != "triton":
        continue
    er = R.get((task, v, "ref"), (None, None))[0]
    same = er is not None and er["ret"] == e["ret"] and er["success"] == e["success"]
    out["fp4"][f"{task}/{v}"] = {"ret_triton": e["ret"], "ret_ref": er["ret"] if er else None, "identical": same,
                                 "success": e["success"], "secs_triton": e["secs_loop"],
                                 "secs_ref": er["secs_loop"] if er else None, "calls_patterns": e.get("calls_patterns"),
                                 "n_refresh": e["n_refresh"]}
    print(f"FP4 {'SAME' if same else 'DIFF'} {task} {v}: ret triton {e['ret']:.4f} ref "
          f"{er['ret'] if er else float('nan'):.4f} succ {e['success']} | sec/ep triton {e['secs_loop']:.1f} ref "
          f"{er['secs_loop'] if er else float('nan'):.1f} | calls {e.get('calls_patterns')}", flush=True)
for f in sorted(glob.glob(f"{a.res}/jobs/heldperr/smoke*.json.gz")):
    d = load(f)
    s = d["spec"]
    for e in d["episodes"]:
        row = {v: {kk: e["perr_plan"][v].get(kk) for kk in ("err1h", "err1", "err3", "err5", "errr", "n_states",
                                                              "nseeds")} for v in s["variants"]}
        rowt = {v: {kk: e["perr_traj"].get(v, {}).get(kk) for kk in ("err1h", "err1", "true1", "true0_1")}
                for v in s["variants"]}
        out["held"][s["id"]] = {"ret": e["ret"], "success": e["success"], "state_t": e.get("perr_state_t"),
                                "plan": row, "traj": rowt, "secs": e["secs"], "secs_loop": e["secs_loop"]}
        print(f"HELD {s['id']} ret {e['ret']:.1f} succ {e['success']} states {e.get('perr_state_t')} "
              f"secs {e['secs']:.1f} (loop {e['secs_loop']:.1f})", flush=True)
        for v in s["variants"]:
            print(f"   {v:14s} plan err1h {row[v]['err1h']:.4g} err1 {row[v]['err1']:.4g} err3 {row[v]['err3']:.4g} "
                  f"errr {row[v]['errr']:.4g} nseeds {row[v]['nseeds']} | traj err1h {rowt[v]['err1h']}", flush=True)
os.makedirs(f"{a.res}/smoke", exist_ok=True)
json.dump(out, open(f"{a.res}/smoke/check_smoke.json", "w"), indent=1)
n_bad = sum(not v["match"] for v in out["repro"].values()) + sum(not v["identical"] for v in out["fp4"].values())
print(f"CHECK_SMOKE_DONE repro {len(out['repro'])} fp4 {len(out['fp4'])} held {len(out['held'])} n_bad {n_bad}")
