"""E8c main measurement (rss_ext/DESIGN_E8c.md 3-4, protocol of DESIGN_ext 1.4 / E8 PLAN 6) on one exclusive GPU,
single process.

Units (e8clib.units_c; 200 units, 240 blocks per round):
  per (size, N)  graph-WM   singles fp32, tf32, bf16, bf16cast, int-det-L0tf32; pair int-det / int-shiter
                 graph-step rest-fp32: singles fp32, tf32, bf16, bf16cast; pair int-det / int-shiter
                 graph-step rest-tf32: singles tf32, bf16, bf16cast; pair int-det / int-shiter
  per size       eager-step (N 512, rest-fp32): singles fp32, tf32, bf16, bf16cast; pair int-det / int-shiter
Protocol (E8): a block = 30 warmup calls, then n timed calls (n from the smoke calibration: max(10, ceil(2 s / t_rep)),
pairs use the det n), each call between a CUDA event pair followed by a device sync; block value = median per-call
time. A graph-step call = copy of the next observation into the static obs buffer + graph replay. Units are shuffled
anew in every round (numpy default_rng(0): one permutation per round, one draw per pair for its order); 10 rounds.
Recorded per block: round, unit position, position in the unit, the per-call times (npz), block statistics, start / end
time, host 1/5/15-min load, host CPU busy share, GPU SM / memory clock, temperature, power, clock-event reasons and
visible compute processes at the block start, middle and end. The dither state of the int MLPs is reset to base
7 * (70000 + round) + 13 before every block (E8). All graphs are built once before round 0 (graph-WM: private pools as
E8; graph-step: one shared pool, e8clib.GraphStep). Results are rewritten after every round.
usage: python e8c_main.py --smoke <smoke dir> --e8in <E8 inputs dir> --out <dir> [--rounds 10]
       [--quick --sizes 5,19]   (code-path check: 1 round, n = 3, warmup 2; numbers void)
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch

import e8lib as E
import e8clib as C


def log(*a):
    print(time.strftime("%H:%M:%S", time.gmtime()), *a, flush=True)


def dump_json(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", default="/workspace/results/smoke")
    ap.add_argument("--e8in", default="/workspace/e8c/e8_inputs")
    ap.add_argument("--out", default="/workspace/results/main")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sizes", default="5,19,48,317")
    ap.add_argument("--quick", action="store_true", help="code-path check: 1 round, n = 3, warmup 2 (numbers void)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    gemm_cfgs = {tuple(int(t) for t in k.split(",")): tuple(v)
                 for k, v in json.load(open(f"{a.e8in}/gemm_cfg.json")).items()}
    selection = json.load(open(f"{a.e8in}/selection.json"))["selection"]
    e8_ranges = json.load(open(f"{a.e8in}/ranges.json"))
    smoke_ranges = json.load(open(f"{a.smoke}/ranges.json"))
    calib = json.load(open(f"{a.smoke}/calib.json"))
    n_reps = calib["n_reps"]
    sizes = tuple(int(s) for s in a.sizes.split(","))
    if a.quick:
        a.rounds, a.warmup = 1, 2
        n_reps = {k: 3 for k in n_reps}
    else:
        assert sizes == E.SIZES, sizes
    gpu = E.GpuState()
    pool = E.EventPool(max(n_reps.values()) + 8)
    t_start = time.time()
    log(f"selection {selection}; GPU {torch.cuda.get_device_name(0)}; sizes {sizes}")

    # models, ranges (must equal E8's and the smoke's), WM paths, graphs, runners
    agents, ranges = {}, {}
    for size in sizes:
        agents[size] = E.build_agent(size)[0]
        ranges[size] = E.calib_ranges(agents[size].model)
        for ref in (e8_ranges, smoke_ranges):
            sr = {k: tuple(v) for k, v in ref[str(size)].items()}
            assert all(abs(ranges[size][k][i] - sr[k][i]) <= 1e-6 * max(1.0, abs(sr[k][i])) for k in sr
                       for i in (0, 1)), (size, ranges[size], sr)
    units = C.units_c(selection, sizes)
    obs = E.obs_stream()
    t0 = time.time()
    with torch.no_grad():
        built = C.build_all(agents, ranges, gemm_cfgs, units, obs, log=log)
    t_build = time.time() - t0
    conds = built["conds"]
    missing = [C.cond_key(*c) for c in conds if C.cond_key(*c) not in n_reps]
    assert not missing, missing[:5]
    smi_procs = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                               capture_output=True, text=True).stdout.strip()

    blocks, pairs, reps = [], [], {}
    rng = np.random.default_rng(a.seed)
    settings = {"rounds": a.rounds, "warmup": a.warmup, "seed": a.seed, "quick": a.quick, "sizes": list(sizes),
                "n_reps": {C.cond_key(*c): n_reps[C.cond_key(*c)] for c in conds}, "selection": selection,
                "gemm_cfgs": {",".join(map(str, k)): list(v) for k, v in gemm_cfgs.items()},
                "ranges": {str(k): v for k, v in ranges.items()}, "n_units_per_round": len(units),
                "n_blocks_per_round": len(conds), "build_s": t_build, "graph_build_s": built["t_graph_s"],
                "gstep_build_s": built["t_gstep_s"], "mem_after_build": built["mem"]}
    env = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
           "triton": __import__("triton").__version__, "nvidia_smi_compute_apps_at_start": smi_procs,
           "gpu_state_at_start": gpu.snap()}

    def run_block(c, rd, upos, vpos, first):
        mode, rest, size, N, prec, impl = c
        key = C.cond_key(*c)
        n = n_reps[key]
        wm = built["wms"][(size, prec, impl)]
        wm.new_episode(7 * (70000 + rd) + 13)
        model = agents[size].model
        fn, inst = C.runner_for(built, agents, c)
        if inst is not None:
            E.install(model, inst)
        load0 = os.getloadavg()
        tot0, idle0 = E.cpu_times()
        g0 = gpu.snap()
        tb0 = time.time()
        try:
            with torch.no_grad():
                ms, gmid = E.time_block(fn, a.warmup, n, pool, mid=gpu.snap)
        finally:
            if inst is not None:
                E.restore(model)
        tb1 = time.time()
        tot1, idle1 = E.cpu_times()
        g1 = gpu.snap()
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert agents[size].cfg.num_samples == C.STEP_N
        idx = len(blocks)
        reps[f"b{idx}"] = np.asarray(ms, np.float32)
        blk = {"idx": idx, "key": key, "mode": mode, "rest": rest, "size": size, "N": N, "prec": prec, "impl": impl,
               "round": rd, "unit_pos": int(upos), "pos_in_unit": int(vpos), "first_in_unit": first, "n_reps": n,
               "t_start": tb0, "t_end": tb1, "load_start": list(load0), "load_end": list(os.getloadavg()),
               "cpu_busy": 1.0 - (idle1 - idle0) / max(tot1 - tot0, 1), "gpu_start": g0, "gpu_mid": gmid,
               "gpu_end": g1}
        blk.update(E.summarize(ms))
        blocks.append(blk)
        return blk

    def save(done):
        out = {"label": "E8c main measurement (DESIGN_E8c 3-4)", "settings": settings, "env": env, "blocks": blocks,
               "pairs": pairs, "rounds_done": done,
               "t_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)), "secs": time.time() - t_start}
        dump_json(f"{a.out}/e8c_main.json", out)
        tmp = f"{a.out}/e8c_reps.tmp{os.getpid()}.npz"
        np.savez_compressed(tmp, **reps)
        os.replace(tmp, f"{a.out}/e8c_reps.npz")

    for rd in range(a.rounds):
        tr = time.time()
        for upos, ui in enumerate(rng.permutation(len(units))):
            u = units[ui]
            cs = C.unit_conds(u)
            if u["kind"] == "pair" and rng.random() < 0.5:
                cs = cs[::-1]
            first = cs[0][4]
            med = {}
            for vpos, c in enumerate(cs):
                blk = run_block(c, rd, upos, vpos, first)
                med[c[4].split("-")[1] if u["kind"] == "pair" else c[4]] = blk["median_ms"]
            if u["kind"] == "pair":
                pairs.append({"mode": u["mode"], "rest": u["rest"], "size": u["size"], "N": u["N"], "prec": u["prec"],
                              "impl": u["impl"], "round": rd, "first": first, "det_ms": med["det"],
                              "shiter_ms": med["shiter"], "overhead": (med["shiter"] - med["det"]) / med["det"]})
        save(rd + 1)
        log(f"round {rd} done in {time.time() - tr:.0f}s (total {time.time() - t_start:.0f}s)")
    log("E8C_MAIN_DONE")


if __name__ == "__main__":
    main()
