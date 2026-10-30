"""E7 latency (DESIGN_main v5 7.4): control-step latency of the TD-MPC2 planner per precision, on an EXCLUSIVE GPU
(no other process, no MPS), single process. Not part of the GPU-0 queue; the coordinator schedules the window.

Protocol
  - condition = (task, precision, MPPI iterations); precisions fp32, bf16, int8 det, int8 dither, int4 det, int4 dither
    (e7lib; int4 twice: codes in int8 containers as in the success runs, and packed two per byte = the v5 target),
    iterations {3, 6, 12}, tasks mw-push, mw-pick-place, cheetah-run, walker-walk, checkpoint 1
  - one block = warmup `--warmup` (50) control steps, then `--steps` (200) timed control steps; each control step is
    agent.act(obs, eval_mode=True) on a fixed observation stream (calibration episode 0 of the task, cycled), timed
    with CUDA events around the call (GPU-synchronized after each step)
  - `--rounds` (5) rounds; in every round the conditions run in a new random order (numpy seed 0)
  - per condition: median and IQR of all timed steps (pooled over rounds), per-round medians; host load (1-min
    loadavg, CPU busy share from /proc/stat over the block) and GPU SM clock / temperature / power at every block
  - H5 inputs: dither overhead = (median int8 dither - median int8 det) / median int8 det per (task, iterations); same
    for int4
  - eager only (confirmatory). CUDA graphs (auxiliary) are not implemented: the dither seed is a host-side value per
    forward call and would be frozen inside a captured graph.
usage (server, exclusive GPU): python e7_latency.py --gpu_check [--wait_exclusive 7200] [--rounds 5]
                                     [--out ../v5/e7_latency.json]   (the GPU-0 queue runs it after stopping MPS)
       code-path check on a shared GPU (numbers meaningless): python e7_latency.py --quick --force
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch

import qlib as L
import e7lib as E7

TASKS = ["mw-push", "mw-pick-place", "cheetah-run", "walker-walk"]
PRECS = [("fp32", 0, "none"), ("bf16", 0, "none"), ("int8", 8, "det"), ("int8", 8, "shiter"), ("int4", 4, "det"),
         ("int4", 4, "shiter"), ("int4p", 4, "det"), ("int4p", 4, "shiter")]   # int4p: packed int4 (the v5 target)
OUT_DIR = os.path.expanduser("~/pmx/A/tdmpc2_main/out")


def gpu_state():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,clocks.sm,clocks.mem,temperature.gpu,power.draw",
                            "--format=csv,noheader,nounits", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0")],
                           capture_output=True, text=True, timeout=20).stdout.strip()
        return q
    except Exception as ex:  # noqa: BLE001
        return repr(ex)


def gpu_procs():
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], capture_output=True,
                       text=True, timeout=20).stdout.split()
    me = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True,
                        text=True, timeout=20).stdout
    idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    uuid = [l.split(", ")[1] for l in me.strip().splitlines() if l.split(", ")[0] == idx][0]
    return sum(1 for u in q if u.strip(",") == uuid)


def cpu_times():
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    return sum(v), v[3] + v[4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--iters", default="3,6,12")
    ap.add_argument("--precs", default="all")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out", default="../v5/e7_latency.json")
    ap.add_argument("--gpu_check", action="store_true", help="refuse unless this process is alone on the GPU")
    ap.add_argument("--wait_exclusive", type=int, default=0,
                    help="with --gpu_check: wait up to this many seconds (5-minute polls) for an exclusive GPU, then "
                         "give up with exit code 2")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.tasks, a.iters, a.rounds, a.warmup, a.steps = "mw-push", "6", 1, 5, 20
    if os.environ.get("CUDA_MPS_PIPE_DIRECTORY") and not a.force:
        raise SystemExit("REFUSED: MPS pipe set; the latency protocol needs a non-MPS exclusive GPU")
    if a.gpu_check and not a.force:
        waited = 0
        while gpu_procs() > 0:
            if waited >= a.wait_exclusive:
                print(f"GAVE_UP: {gpu_procs()} other compute processes on the GPU after {waited} s", flush=True)
                raise SystemExit(2)
            print(f"{time.strftime('%H:%M:%S', time.gmtime())} waiting for an exclusive GPU "
                  f"({gpu_procs()} other compute processes)", flush=True)
            time.sleep(300)
            waited += 300
    precs = PRECS if a.precs == "all" else [p for p in PRECS if f"{p[0]}_{p[2]}" in a.precs.split(",")]
    iters = [int(x) for x in a.iters.split(",")]
    tasks = a.tasks.split(",")
    torch.set_num_threads(1)
    agents = {}
    for t in tasks:
        agent, cfg = L.build_agent(t, 1)
        quant = L.Quantizer(agent.model, horizon=int(cfg.horizon))
        qr = json.load(open(f"{OUT_DIR}/scales/{t}_s1.json"))["quant"]
        quant.qrange = {k: (float(v[0]), float(v[1])) for k, v in qr.items()}
        obs = np.load(f"{OUT_DIR}/calib/{t}_s1_ep0.npz")["obs_seq"]
        agents[t] = (agent, cfg, quant, torch.as_tensor(obs, dtype=torch.float32))
    conds = [(t, p, k, v, it) for t in tasks for (p, k, v) in precs for it in iters]
    rng = np.random.default_rng(0)
    res = {f"{t}|{p}_{v}|it{it}": {"steps_ms": [], "round_median_ms": [], "blocks": []} for t, p, k, v, it in conds}
    t_start = time.time()
    with torch.no_grad():
        for rd in range(a.rounds):
            for ci in rng.permutation(len(conds)):
                t, p, k, v, it = conds[ci]
                agent, cfg, quant, obs = agents[t]
                L.set_planner(agent, {"iterations": it})
                e7 = E7.install(agent, quant, OUT_DIR, {"e7": p, "variant": v, "k": k, "task": t, "ckpt": 1})
                try:
                    e7.new_episode(7 * (70000 + rd) + 13)
                    load0, (tot0, idle0), gs0 = os.getloadavg()[0], cpu_times(), gpu_state()
                    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    ms = []
                    for s in range(a.warmup + a.steps):
                        o = obs[s % len(obs)]
                        ev0.record()
                        agent.act(o, t0=(s == 0), eval_mode=True)
                        ev1.record()
                        torch.cuda.synchronize()
                        if s >= a.warmup:
                            ms.append(ev0.elapsed_time(ev1))
                    tot1, idle1 = cpu_times()
                    blk = {"round": rd, "median_ms": float(np.median(ms)), "loadavg1": load0,
                           "cpu_busy": 1.0 - (idle1 - idle0) / max(tot1 - tot0, 1), "gpu_start": gs0,
                           "gpu_end": gpu_state(), "foreign_procs": gpu_procs() - 1}
                finally:
                    e7.restore()
                    quant.layer_k, quant.wq = {}, None
                    L.set_planner(agent, None)
                r = res[f"{t}|{p}_{v}|it{it}"]
                r["steps_ms"] += ms
                r["round_median_ms"].append(blk["median_ms"])
                r["blocks"].append(blk)
            print(f"round {rd} done {time.time() - t_start:.0f}s", flush=True)
    summ = {}
    for key, r in res.items():
        x = np.array(r["steps_ms"])
        summ[key] = {"median_ms": float(np.median(x)), "iqr_ms": [float(np.percentile(x, 25)), float(np.percentile(x, 75))],
                     "n": int(len(x)), "round_median_ms": r["round_median_ms"]}
    over = {}
    for t in tasks:
        for it in iters:
            for p in ("int8", "int4", "int4p"):
                kd, kv = f"{t}|{p}_det|it{it}", f"{t}|{p}_shiter|it{it}"
                if kd in summ and kv in summ:
                    over[f"{t}|{p}|it{it}"] = (summ[kv]["median_ms"] - summ[kd]["median_ms"]) / summ[kd]["median_ms"]
    foreign = sum(b["foreign_procs"] > 0 for r in res.values() for b in r["blocks"])
    out = {"summary": summ, "dither_overhead": over, "blocks": {k: r["blocks"] for k, r in res.items()},
           "blocks_with_foreign_processes": foreign,
           "settings": vars(a), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "secs": time.time() - t_start}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({"summary": {k: round(v["median_ms"], 3) for k, v in summ.items()}, "dither_overhead": over}))
    print("E7_LATENCY_DONE", flush=True)


if __name__ == "__main__":
    main()
