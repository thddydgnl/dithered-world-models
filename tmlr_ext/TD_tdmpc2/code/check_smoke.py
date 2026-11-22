"""td smoke report (phase A, smoke folders only; smoke seeds are never analysed). Writes <res>/smoke_report.json.
- repro: the runner job path (repro_* jobs, seed 12000, E0 ranges) gives the same episode as check_e1_paths.py's
  direct run_episode with the untouched E1 qlib and with this qlib (return, success)
- episodes: every smoke episode job finished; forward-call patterns (default planner: dyn0 x512 18 + x24 2 per step;
  cem: x512 only); table signatures of the table variants constant inside the episode (t < 4); n_refresh;
  instrumentation fields (drift, sel_last, oei_indep, sel_indep) on the mechanism tasks; seconds per episode
- prep: residual of the 3-sweep bias procedure, re-derived per-tensor range minus the E0 file, PC range summary
- e13: per cell first / last set metrics (rho_p of det, det+BC and the table variants must be 1), seconds per episode
- timing: ms per decision by iterations
Usage: python check_smoke.py --res <smoke out> --e1paths <dir with e1paths_e1.txt, e1paths_td.txt>
"""
import argparse
import glob
import gzip
import json
import os


def load(p):
    return json.load(gzip.open(p, "rt"))


ap = argparse.ArgumentParser()
ap.add_argument("--res", required=True)
ap.add_argument("--e1paths", required=True)
a = ap.parse_args()
R = {"repro": {}, "episodes": {}, "prep": {}, "e13": {}, "timing": {}, "problems": []}


def prob(msg):
    R["problems"].append(msg)
    print("PROBLEM", msg, flush=True)


# ------------------------------------------------------------------ check 1 outputs and the job path
paths = {}
for nm in ("e1", "td"):
    f = f"{a.e1paths}/e1paths_{nm}.txt"
    if os.path.exists(f):
        for line in open(f):
            if line.startswith("E1PATHS "):
                paths[nm] = json.loads(line[len("E1PATHS "):])
if len(paths) == 2:
    keys = [k for k in paths["e1"] if k != "qlib"]
    same = {k: paths["e1"][k] == paths["td"].get(k) for k in keys}
    R["repro"]["e1_vs_td_qlib"] = {"identical": all(same.values()), "per_episode": same,
                                   "values_e1": {k: paths["e1"][k] for k in keys}}
    print(f"CHECK1 untouched E1 qlib vs td qlib: {'IDENTICAL' if all(same.values()) else 'DIFFERENT'} {same}")
    if not all(same.values()):
        prob("check 1: E1 qlib and td qlib differ")
else:
    prob(f"check 1 outputs missing: {list(paths)}")
for f in sorted(glob.glob(f"{a.res}/jobs/episodes/repro_*.json.gz")):
    d = load(f)
    s, e = d["spec"], d["episodes"][0]
    key = f"{s['task']}|{s['variant']}_k{s['k']}"
    ref = paths.get("e1", {}).get(key)
    ok = ref is not None and ref[0] == e["ret"] and ref[1] == e["success"]
    R["repro"][s["id"]] = {"ret": e["ret"], "check1_ret": ref[0] if ref else None, "same": ok}
    print(f"JOBPATH {'SAME' if ok else 'DIFF'} {s['id']} ret {e['ret']:.4f} direct {ref[0] if ref else None}")
    if not ok:
        prob(f"job path differs from direct run_episode: {s['id']}")

# ------------------------------------------------------------------ smoke episodes
for f in sorted(glob.glob(f"{a.res}/jobs/episodes/smoke_*.json.gz")):
    d = load(f)
    s, e = d["spec"], d["episodes"][0]
    rec = {"task": s["task"], "variant": s["variant"], "rk": s.get("rk"), "k": s["k"], "ptag": s.get("ptag"),
           "ret": e["ret"], "success": e["success"], "secs": e["secs_loop"], "n_refresh": e["n_refresh"],
           "calls": e.get("calls_patterns"), "table_sig": e.get("table_sig"), "gpu": d["env"]["gpu"]}
    om = e.get("oei_mean") or {}
    rec["oei_fields"] = sorted(k for k in om if k in ("oei", "drift", "sel_last", "oei_indep", "sel_indep", "e_fresh"))
    if s.get("instr", {}).get("oei_stride") and s["variant"] != "none":
        for fld in ("drift", "sel_last", "oei_indep", "sel_indep"):
            if fld not in om:
                prob(f"{s['id']}: instrumentation field {fld} missing")
    if rec["table_sig"]:
        sig = rec["table_sig"]
        rec["table_constant"] = all(x == sig[0] for x in sig)
        if not rec["table_constant"] and s["variant"] in ("shep", "detshiftfix"):
            prob(f"{s['id']}: episode table changed inside the episode {sig}")
    calls = rec["calls"] or []
    if s.get("ptag") == "cem" and any("x24" in c for c in calls):
        prob(f"{s['id']}: policy-prior calls under num_pi_trajs 0")
    R["episodes"][s["id"]] = rec
    print(f"EP {s['id']}: ret {e['ret']:.2f} succ {e['success']} {e['secs_loop']:.1f}s refresh {e['n_refresh']} "
          f"calls {calls[:1]} sig {rec['table_sig'][:2] if rec['table_sig'] else ''} oei {rec['oei_fields']}")

# ------------------------------------------------------------------ prep
for f in sorted(glob.glob(f"{a.res}/jobs/prep/prep_*.json.gz")):
    d = load(f)
    s = d["spec"]
    rr = {n: {kk: v[kk] for kk in ("lo_minus_e0", "hi_minus_e0", "hi_minus_e0_rel_width", "pc_width_mean_rel_pt")}
          for n, v in d["ranges"].items()}
    R["prep"][s["id"]] = {"ranges": rr, "secs": d["secs"]}
    print(f"PREP {s['id']}: range diff {  {n: round(v['hi_minus_e0_rel_width'], 4) for n, v in rr.items()} }, "
          f"{d['secs']:.1f}s")
for f in sorted(glob.glob(f"{a.res}/jobs/bias/bias_*.json.gz")):
    d = load(f)
    s = d["spec"]
    resid = {f"{rk}k{k}": x["resid_mean_abs_rel_delta_max_layer"] for rk, byk in d["bias"].items()
             for k, x in byk.items()}
    R.setdefault("bias", {})[s["id"]] = {"rule": d.get("rule"), "resid": resid,
                                          "n_updates": {f"{rk}k{k}": x.get("n_updates") for rk, byk in d["bias"].items()
                                                        for k, x in byk.items()}}
    print(f"BIAS {s['id']}: rule {d.get('rule')} worst resid/delta {max(resid.values()):.2e}")
    if max(resid.values()) >= 0.01:
        prob(f"{s['id']}: bias residual >= 1% of delta")

# ------------------------------------------------------------------ e13
for f in sorted(glob.glob(f"{a.res}/jobs/e13/*.json.gz")):
    d = load(f)
    s = d["spec"]
    e = d["episodes"][0]
    cells = {}
    for key, sets in e["cells"].items():
        cells[key] = {sn: {m: sets[sn][m] for m in ("err1h", "err3", "rew_med", "ratio", "rho_p", "prod", "spear",
                                                     "elite_overlap", "common_frac")} for sn in sets}
        v = key.split("_", 1)[1] if "_" in key else key
        if v in ("det", "detbc", "shep", "shstep", "detshiftfix"):
            for sn in sets:
                if abs(sets[sn]["rho_p"] - 1) > 1e-4:
                    prob(f"{s['id']} {key} {sn}: rho_p {sets[sn]['rho_p']} != 1 for a persistent variant")
    R["e13"][s["id"]] = {"ret": e["ret"], "state_t": e["state_t"], "n_kept": e["n_kept"], "secs": e["secs"],
                         "secs_loop": e["secs_loop"], "sets": e["sets"], "cells": cells, "n_cells": len(cells)}
    print(f"E13 {s['id']}: {len(cells)} cells, states {e['state_t']}, {e['secs']:.1f}s (loop {e['secs_loop']:.1f}s)")
    for key in list(cells)[:12]:
        c = cells[key]
        print(f"   {key:22s} first err1h {c['first']['err1h']:.4f} rew_med {c['first']['rew_med']:.4f} ratio "
              f"{c['first']['ratio']:.4f} rho_p {c['first']['rho_p']:.3f} | last ratio {c['last']['ratio']:.4f} "
              f"rho_p {c['last']['rho_p']:.3f} spear {c['last']['spear']:.3f} overlap {c['last']['elite_overlap']:.3f}")

# ------------------------------------------------------------------ timing
for f in sorted(glob.glob(f"{a.res}/jobs/timing/*.json.gz")):
    d = load(f)
    R["timing"][d["spec"]["id"]] = d["timing"]
    print("TIMING", d["spec"]["id"], {k: round(v["ms_median"], 2) for k, v in d["timing"].items()})
errs = sorted(glob.glob(f"{a.res}/errors/*.txt"))
for f in errs:
    prob(f"error file {os.path.basename(f)}: {open(f).read().strip().splitlines()[-1]}")
json.dump(R, open(f"{a.res}/smoke_report.json", "w"), indent=1, default=float)
print(f"CHECK_SMOKE_DONE episodes {len(R['episodes'])} prep {len(R['prep'])} e13 {len(R['e13'])} "
      f"problems {len(R['problems'])}")
