"""E7 TD-MPC2 integrity check (no success outcome is printed).
Success runs: every job of e7.jsonl has one result whose spec equals the manifest entry; the grid against DESIGN_main
v5 7.4 (written from the design text): {fp32, bf16, int8 det, int8 dither, int4 det, int4 dither} x iterations
{3, 6, 12} x tasks {mw-push, mw-pick-place, cheetah-run, walker-walk}, checkpoints 1-3, seeds 70000-70029 (90 pairs);
the recorded kernel info; GPU / CVD / host / MPS; error and lock files.
Latency (e7_latency.json) against the v5 7.4 protocol: single process, exclusive GPU (no foreign compute process in
any block), warmup 50 + 200 timed steps per block, 5 rounds, a new random order per round (reconstructed from the
script's numpy seed 0), 1,000 timed steps per condition, the 8 latency precisions (int4 twice: container and packed);
host load and GPU clock / temperature / power per block are summarised (descriptive).
usage (server, CPU only): CUDA_VISIBLE_DEVICES="" python integrity_e7.py --res <out_e7> --manifest <e7.jsonl>
                              --latency <e7_latency.json> --out <json>
"""
import argparse
import glob
import gzip
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

TASKS = ["mw-push", "mw-pick-place", "cheetah-run", "walker-walk"]
PRECS = [("fp32", 0, "none"), ("bf16", 0, "none"), ("int8", 8, "det"), ("int8", 8, "shiter"), ("int4", 4, "det"),
         ("int4", 4, "shiter")]
LAT_PRECS = PRECS + [("int4p", 4, "det"), ("int4p", 4, "shiter")]
ITERS = [3, 6, 12]
CKPTS = [1, 2, 3]


def ptag_of(job_id):
    return job_id.split("_")[-2]


def parse_gpu(s):
    """'idx, sm MHz, mem MHz, temp C, power W' -> dict (None if unparsable)."""
    try:
        i, sm, mem, temp, pw = [x.strip() for x in s.split(",")]
        return {"idx": int(i), "sm": float(sm), "mem": float(mem), "temp": float(temp), "power": float(pw)}
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--latency", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    jobs = [json.loads(l) for l in open(a.manifest) if l.strip()]
    rep = {"manifest_jobs": len(jobs)}
    bad = defaultdict(list)
    env = Counter()
    eps_cond = defaultdict(set)
    ids = set()
    want = {(t, c, p, k, v, it) for t in TASKS for c in CKPTS for p, k, v in PRECS for it in ITERS}
    seen = set()
    span = [math.inf, -math.inf]
    for j in jobs:
        ids.add(j["id"])
        p_ = f"{a.res}/jobs/episodes/{j['id']}.json.gz"
        if not os.path.exists(p_):
            bad["missing_result"].append(j["id"])
            continue
        d = json.load(gzip.open(p_, "rt"))
        if not d.get("ok"):
            bad["not_ok"].append(j["id"])
        if d.get("spec") != j:
            bad["spec_differs"].append(j["id"])
        e = d.get("env", {})
        env[(e.get("host"), e.get("gpu"), e.get("cvd"), e.get("mps"), e.get("torch"))] += 1
        span[0], span[1] = min(span[0], d["t_start"]), max(span[1], d["t_start"] + d["secs"])
        t, c, k, v, prec = j["task"], int(j["ckpt"]), int(j["k"]), j["variant"], j.get("e7")
        it = int(j.get("planner", {}).get("iterations", -1))
        key = (t, c, prec, k, v, it)
        if key not in want or ptag_of(j["id"]) != f"e7{prec}it{it}" or set(j.get("planner", {})) != {"iterations"}:
            bad["grid"].append(j["id"])
        seen.add(key)
        if j["split"] != "e7" or int(j["seed0"]) != 70000 or j.get("instr"):
            bad["split_seed_instr"].append(j["id"])
        info = d.get("e7", {})
        if info.get("prec") != prec or info.get("variant") != v or int(info.get("k", -1)) != k:
            bad["kernel_info"].append(j["id"])
        if prec in ("int8", "int4") and info.get("act_container") != "int8":
            bad["int4_container"].append(j["id"])
        got = [int(x["ep"]) for x in d["episodes"]]
        if got != [int(x) for x in j["eps"]]:
            bad["episode_ids"].append(j["id"])
        for ep in d["episodes"]:
            if int(ep["seed"]) != 70000 + int(ep["ep"]):
                bad["seed"].append([j["id"], ep["ep"]])
            if not (math.isfinite(ep["ret"]) and math.isfinite(ep["success"])):
                bad["nonfinite"].append([j["id"], ep["ep"]])
            eps_cond[key].add(int(ep["ep"]))
    extra = [os.path.basename(f) for f in glob.glob(f"{a.res}/jobs/episodes/*.json.gz")
             if os.path.basename(f)[:-len(".json.gz")] not in ids]
    import time as _t
    rep.update({
        "grid_complete": seen == want, "grid_conditions": len(seen), "grid_expected": len(want),
        "conditions_not_30_eps": [list(k) + [len(s)] for k, s in eps_cond.items() if s != set(range(30))],
        "env": [list(k) + [n] for k, n in env.items()], "result_files_extra": extra,
        "errors_dir_files": len(glob.glob(f"{a.res}/errors/*")), "lock_files": len(glob.glob(f"{a.res}/locks/*")),
        "success_time_span_utc": [_t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime(x)) for x in span]})
    # ---------------------------------------------------------------- latency
    L = json.load(open(a.latency))
    st = L["settings"]
    lat = {"settings": st, "gpu": L.get("gpu"), "torch": L.get("torch"), "secs": L.get("secs"),
           "blocks_with_foreign_processes": L.get("blocks_with_foreign_processes")}
    conds = [f"{t}|{p}_{v}|it{it}" for t in TASKS for p, k, v in LAT_PRECS for it in ITERS]
    lb = defaultdict(list)
    if (st.get("rounds"), st.get("warmup"), st.get("steps")) != (5, 50, 200) or not st.get("gpu_check") or \
            st.get("force") or st.get("quick") or st.get("precs") != "all" or \
            st.get("tasks") != ",".join(TASKS) or st.get("iters") != "3,6,12":
        lb["settings"].append(st)
    if set(L["summary"]) != set(conds):
        lb["conditions"].append(sorted(set(conds) ^ set(L["summary"])))
    loads, busy, sm, temp, power, foreign = [], [], [], [], [], 0
    for cnd in conds:
        s = L["summary"].get(cnd, {})
        if s.get("n") != 1000 or len(s.get("round_median_ms", [])) != 5:
            lb["n_or_rounds"].append(cnd)
        blks = L["blocks"].get(cnd, [])
        if sorted(b["round"] for b in blks) != list(range(5)):
            lb["block_rounds"].append(cnd)
        for b in blks:
            foreign += int(b["foreign_procs"] > 0)
            loads.append(b["loadavg1"])
            busy.append(b["cpu_busy"])
            for g in (parse_gpu(b["gpu_start"]), parse_gpu(b["gpu_end"])):
                if g is None or g["idx"] != 0:
                    lb["gpu_state_parse_or_index"].append(cnd)
                    continue
                sm.append(g["sm"])
                temp.append(g["temp"])
                power.append(g["power"])
    # the script's order: numpy default_rng(0).permutation(96) once per round
    rng = np.random.default_rng(0)
    perms = [rng.permutation(len(conds)) for _ in range(st.get("rounds", 5))]
    lat.update({
        "conditions": len(L["summary"]), "foreign_blocks_recount": foreign,
        "orders_differ_between_rounds": bool(all(not np.array_equal(perms[i], perms[j])
                                                 for i in range(len(perms)) for j in range(i + 1, len(perms)))),
        "host_loadavg1": {"min": float(np.min(loads)), "median": float(np.median(loads)), "max": float(np.max(loads))},
        "host_cpu_busy": {"min": float(np.min(busy)), "median": float(np.median(busy)), "max": float(np.max(busy))},
        "gpu_sm_clock_mhz": {"min": float(np.min(sm)), "median": float(np.median(sm)), "max": float(np.max(sm)),
                             "share_at_max": float(np.mean(np.array(sm) == np.max(sm)))},
        "gpu_temp_c": {"min": float(np.min(temp)), "max": float(np.max(temp))},
        "gpu_power_w": {"min": float(np.min(power)), "max": float(np.max(power))},
        "problems": {k: (len(v), v[:3]) for k, v in lb.items()}})
    rep["latency"] = lat
    rep["problems"] = {k: (len(v), v[:5]) for k, v in bad.items()}
    rep["ALL_OK"] = bool(not bad and not extra and seen == want and not rep["conditions_not_30_eps"] and len(env) == 1
                         and rep["errors_dir_files"] == 0 and rep["lock_files"] == 0 and not lb and foreign == 0
                         and L.get("blocks_with_foreign_processes") == 0 and lat["orders_differ_between_rounds"])
    s = json.dumps(rep, indent=1, default=str)
    if a.out:
        open(a.out, "w").write(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
