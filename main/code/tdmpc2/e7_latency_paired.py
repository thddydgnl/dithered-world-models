"""POST-HOC E7 latency re-measurement in paired blocks (DESIGN_main v5.3 10.3, user-approved 2026-10-24; descriptive,
the H5 verdict stays the first measurement's pre-defined computation). The first file v5/e7_latency.json is never
written by this script.

Protocol (the first measurement's, e7_latency.py, unless stated): exclusive GPU (no other compute process, no MPS),
single process, control step = agent.act(obs, eval_mode=True) on the fixed observation stream of calibration episode 0
(cycled), CUDA events around every step (GPU-synchronized), block = warmup 50 + 200 timed steps, eager only, the same
96 conditions (4 tasks x 8 latency precisions x MPPI iterations {3, 6, 12}; checkpoint 1).
Changes (v5.3 10.3): 10 rounds; one unit = the det block and the dither block of the same (task, iterations, kernel
bits) measured back to back (int8, int4 container, packed int4), in a random order inside the unit; fp32 and bf16
are single-block units; the units are shuffled anew every round (numpy default_rng(--seed), default 0).
Recorded per block: round, unit position, position inside the unit, block median, start time, host 1-min load, CPU busy
share over the block, GPU SM clock / memory clock / temperature / power at the block start and end, foreign compute
processes on the GPU at the block end. Per pair: the overhead (dither block median - det block median) / det block
median. The statistics (per-condition median of the pair overheads with a bootstrap CI, and the first measurement's
pooled-median computation on the same data) are computed by analyze_e7_posthoc_latency.py.
usage (server, exclusive GPU 0, MPS stopped): python e7_latency_paired.py --gpu_check [--wait_exclusive 7200]
                                                  [--rounds 10] [--out ../v5/e7_latency_posthoc.json]
       code-path check on a shared GPU (numbers meaningless): python e7_latency_paired.py --quick --force --out <file>
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import e7_latency as E7L
import e7lib as E7
import qlib as L

FIRST = os.path.realpath(os.path.expanduser("~/pmx/A/tdmpc2_main/v5/e7_latency.json"))
PAIRS = [("int8", 8), ("int4", 4), ("int4p", 4)]
SINGLES = [("fp32", 0, "none"), ("bf16", 0, "none")]


def block(agents, t, p, k, v, it, rd, warmup, steps):
    agent, cfg, quant, obs = agents[t]
    L.set_planner(agent, {"iterations": it})
    e7 = E7.install(agent, quant, E7L.OUT_DIR, {"e7": p, "variant": v, "k": k, "task": t, "ckpt": 1})
    try:
        e7.new_episode(7 * (70000 + rd) + 13)
        t0 = time.time()
        load0, (tot0, idle0), gs0 = os.getloadavg()[0], E7L.cpu_times(), E7L.gpu_state()
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ms = []
        for s in range(warmup + steps):
            o = obs[s % len(obs)]
            ev0.record()
            agent.act(o, t0=(s == 0), eval_mode=True)
            ev1.record()
            torch.cuda.synchronize()
            if s >= warmup:
                ms.append(ev0.elapsed_time(ev1))
        tot1, idle1 = E7L.cpu_times()
        blk = {"round": rd, "median_ms": float(np.median(ms)), "t_start": t0, "loadavg1": load0,
               "cpu_busy": 1.0 - (idle1 - idle0) / max(tot1 - tot0, 1), "gpu_start": gs0, "gpu_end": E7L.gpu_state(),
               "foreign_procs": E7L.gpu_procs() - 1}
    finally:
        e7.restore()
        quant.layer_k, quant.wq = {}, None
        L.set_planner(agent, None)
    return ms, blk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(E7L.TASKS))
    ap.add_argument("--iters", default="3,6,12")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="../v5/e7_latency_posthoc.json")
    ap.add_argument("--gpu_check", action="store_true")
    ap.add_argument("--wait_exclusive", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if os.path.realpath(os.path.abspath(a.out)) == FIRST:
        raise SystemExit("REFUSED: --out is the first measurement's file")
    if a.quick:
        a.tasks, a.iters, a.rounds, a.warmup, a.steps = "mw-push", "6", 1, 5, 20
    if os.environ.get("CUDA_MPS_PIPE_DIRECTORY") and not a.force:
        raise SystemExit("REFUSED: MPS pipe set; the latency protocol needs a non-MPS exclusive GPU")
    if a.gpu_check and not a.force:
        waited = 0
        while E7L.gpu_procs() > 0:
            if waited >= a.wait_exclusive:
                print(f"GAVE_UP: {E7L.gpu_procs()} other compute processes on the GPU after {waited} s", flush=True)
                raise SystemExit(2)
            print(f"{time.strftime('%H:%M:%S', time.gmtime())} waiting for an exclusive GPU "
                  f"({E7L.gpu_procs()} other compute processes)", flush=True)
            time.sleep(300)
            waited += 300
    tasks, iters = a.tasks.split(","), [int(x) for x in a.iters.split(",")]
    torch.set_num_threads(1)
    agents = {}
    for t in tasks:
        agent, cfg = L.build_agent(t, 1)
        quant = L.Quantizer(agent.model, horizon=int(cfg.horizon))
        qr = json.load(open(f"{E7L.OUT_DIR}/scales/{t}_s1.json"))["quant"]
        quant.qrange = {k: (float(v[0]), float(v[1])) for k, v in qr.items()}
        obs = np.load(f"{E7L.OUT_DIR}/calib/{t}_s1_ep0.npz")["obs_seq"]
        agents[t] = (agent, cfg, quant, torch.as_tensor(obs, dtype=torch.float32))
    units = [("pair", t, p, k, it) for t in tasks for p, k in PAIRS for it in iters] + \
            [("single", t, p, k, it) for t in tasks for p, k, _ in SINGLES for it in iters]
    rng = np.random.default_rng(a.seed)
    keys = {}
    for kind, t, p, k, it in units:
        for v in (("det", "shiter") if kind == "pair" else ("none",)):
            keys[f"{t}|{p}_{v}|it{it}"] = {"steps_ms": [], "blocks": []}
    pairs = {f"{t}|{p}|it{it}": [] for kind, t, p, k, it in units if kind == "pair"}
    t_start = time.time()
    with torch.no_grad():
        for rd in range(a.rounds):
            for upos, ui in enumerate(rng.permutation(len(units))):
                kind, t, p, k, it = units[ui]
                vs = ["det", "shiter"] if kind == "pair" else ["none"]
                if kind == "pair" and rng.random() < 0.5:
                    vs = vs[::-1]
                med = {}
                for vpos, v in enumerate(vs):
                    ms, blk = block(agents, t, p, k, v, it, rd, a.warmup, a.steps)
                    blk.update(unit_pos=int(upos), pos_in_unit=vpos, first_in_unit=vs[0])
                    r = keys[f"{t}|{p}_{v}|it{it}"]
                    r["steps_ms"] += ms
                    r["blocks"].append(blk)
                    med[v] = blk["median_ms"]
                if kind == "pair":
                    pairs[f"{t}|{p}|it{it}"].append({"round": rd, "first": vs[0], "det_ms": med["det"],
                                                     "dither_ms": med["shiter"],
                                                     "overhead": (med["shiter"] - med["det"]) / med["det"]})
            print(f"round {rd} done {time.time() - t_start:.0f}s", flush=True)
    summ = {}
    for key, r in keys.items():
        x = np.array(r["steps_ms"])
        summ[key] = {"median_ms": float(np.median(x)), "iqr_ms": [float(np.percentile(x, 25)),
                                                                   float(np.percentile(x, 75))],
                     "n": int(len(x)), "round_median_ms": [b["median_ms"] for b in r["blocks"]]}
    out = {"label": "post-hoc paired re-measurement (DESIGN_main v5.3 10.3); descriptive", "summary": summ,
           "pairs": pairs, "blocks": {k: r["blocks"] for k, r in keys.items()},
           "steps_ms": {k: r["steps_ms"] for k, r in keys.items()},
           "blocks_with_foreign_processes": sum(b["foreign_procs"] > 0 for r in keys.values() for b in r["blocks"]),
           "settings": vars(a), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "secs": time.time() - t_start, "t_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start))}
    tmp = f"{a.out}.tmp{os.getpid()}"
    json.dump(out, open(tmp, "w"))
    os.replace(tmp, a.out)
    print(json.dumps({"blocks": sum(len(r["blocks"]) for r in keys.values()),
                      "foreign_blocks": out["blocks_with_foreign_processes"], "secs": round(out["secs"])}))
    print("E7_LATENCY_POSTHOC_DONE", flush=True)


if __name__ == "__main__":
    main()
