"""E8 main measurement (rss_ext/DESIGN_ext.md 1.4) on one exclusive GPU, single process.

Conditions (e8lib.units): modes graph (graph-WM, primary: the WM rollout of one control step captured as one CUDA graph
and replayed), eager (eager-WM: the same rollout run eagerly), step (eager-step: TDMPC2.act(obs, eval_mode=True), the
whole planning step, N = 512 only); sizes 5 / 19 / 48 / 317; N 512 / 2048 / 8192; precisions fp32, tf32, bf16 (single
blocks) and int det / shiter (pairs, on the GEMM path selected per size in the smoke); at 5M also E7's fused kernels
det / shiter (pairs, descriptive).
Protocol: a block = 30 warmup calls, then n timed calls (n from the smoke calibration: max(10, ceil(2 s / t_rep)), pairs
use the det n), each call between a CUDA event pair followed by a device sync; block value = median per-call time.
A unit (single block, or the det and shiter blocks of a pair back to back in random order) is shuffled anew in every
round (numpy default_rng(0): one permutation per round, one draw per pair for its order); 10 rounds. Recorded per
block: round, unit position, position in the unit, the per-call times (npz), block statistics, start / end time, host
1/5/15-min load, host CPU busy share over the block, GPU SM / memory clock, temperature, power, clock-event reasons and
visible compute processes at the block start, middle and end. The dither state of the int / E7 MLPs is reset to
base 7 * (70000 + round) + 13 before every block (as in E7). Results are rewritten after every round.
usage: python e8_main.py --smoke /workspace/results/smoke --out /workspace/results/main [--rounds 10]
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch

import e8lib as E


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
    ap.add_argument("--out", default="/workspace/results/main")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="code-path check: 1 round, n = 3, warmup 2 (numbers void)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    gemm_cfgs = {tuple(int(t) for t in k.split(",")): tuple(v)
                 for k, v in json.load(open(f"{a.smoke}/gemm_cfg.json")).items()}
    calib = json.load(open(f"{a.smoke}/calib.json"))
    selection = calib["selection"]
    n_reps = calib["n_reps"]
    smoke_ranges = json.load(open(f"{a.smoke}/ranges.json"))
    if a.quick:
        a.rounds, a.warmup = 1, 2
        n_reps = {k: 3 for k in n_reps}
    gpu = E.GpuState()
    pool = E.EventPool(max(n_reps.values()) + 8)
    t_start = time.time()
    log(f"selection {selection}; GPU {torch.cuda.get_device_name(0)}")

    # models, ranges (must equal the smoke's), WM paths, static inputs, graphs, step runners
    agents, ranges, wms, inputs = {}, {}, {}, {}
    for size in E.SIZES:
        agents[size] = E.build_agent(size)[0]
        ranges[size] = E.calib_ranges(agents[size].model)
        sr = {k: tuple(v) for k, v in smoke_ranges[str(size)].items()}
        assert all(abs(ranges[size][k][i] - sr[k][i]) <= 1e-6 * max(1.0, abs(sr[k][i])) for k in sr for i in (0, 1)), \
            (size, ranges[size], sr)
    units = E.units(selection)
    conds = [c for u in units for c in E.unit_conds(u)]
    for (mode, size, N, prec, impl) in conds:
        if (size, prec, impl) not in wms:
            wms[(size, prec, impl)] = E.make_wm(agents[size].model, prec, ranges[size], impl, gemm_cfgs)
        if mode != "step" and (size, N) not in inputs:
            inputs[(size, N)] = E.wm_inputs(agents[size].model, N)
    graphs = {}
    t0 = time.time()
    for (mode, size, N, prec, impl) in conds:
        if mode == "graph":
            wm, inp = wms[(size, prec, impl)], inputs[(size, N)]
            wm.new_episode(1)
            graphs[E.cond_key(mode, size, N, prec, impl)] = E.GraphRunner(lambda w=wm, i=inp: E.wm_rollout(w, *i))
    torch.cuda.synchronize()
    log(f"built {len(graphs)} graphs in {time.time() - t0:.0f}s; allocated {torch.cuda.memory_allocated() / 2**30:.1f}"
        f" GiB, reserved {torch.cuda.memory_reserved() / 2**30:.1f} GiB")
    obs = E.obs_stream()
    steps = {size: E.StepRunner(agents[size], obs) for size in E.SIZES}
    smi_procs = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                               capture_output=True, text=True).stdout.strip()

    blocks, pairs, reps = [], [], {}
    rng = np.random.default_rng(a.seed)
    settings = {"rounds": a.rounds, "warmup": a.warmup, "seed": a.seed, "quick": a.quick, "n_reps": n_reps,
                "selection": selection, "gemm_cfgs": {",".join(map(str, k)): list(v) for k, v in gemm_cfgs.items()},
                "ranges": {str(k): v for k, v in ranges.items()}, "n_units_per_round": len(units),
                "n_blocks_per_round": len(conds)}
    env = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
           "triton": __import__("triton").__version__, "nvidia_smi_compute_apps_at_start": smi_procs,
           "gpu_state_at_start": gpu.snap()}

    def run_block(c, rd, upos, vpos, first):
        mode, size, N, prec, impl = c
        key = E.cond_key(*c)
        n = n_reps[key]
        wm = wms[(size, prec, impl)]
        wm.new_episode(7 * (70000 + rd) + 13)
        model = agents[size].model
        if mode == "graph":
            fn = graphs[key]
        elif mode == "eager":
            inp = inputs[(size, N)]
            fn = (lambda: E.wm_rollout(wm, *inp))
        else:
            fn = steps[size]
            fn.reset()
            E.install(model, wm)
        load0 = os.getloadavg()
        tot0, idle0 = E.cpu_times()
        g0 = gpu.snap()
        tb0 = time.time()
        try:
            with torch.no_grad():
                ms, gmid = E.time_block(fn, a.warmup, n, pool, mid=gpu.snap)
        finally:
            if mode == "step":
                E.restore(model)
        tb1 = time.time()
        tot1, idle1 = E.cpu_times()
        g1 = gpu.snap()
        assert torch.backends.cuda.matmul.allow_tf32 is False
        idx = len(blocks)
        reps[f"b{idx}"] = np.asarray(ms, np.float32)
        blk = {"idx": idx, "key": key, "mode": mode, "size": size, "N": N, "prec": prec, "impl": impl, "round": rd,
               "unit_pos": int(upos), "pos_in_unit": int(vpos), "first_in_unit": first, "n_reps": n,
               "t_start": tb0, "t_end": tb1, "load_start": list(load0), "load_end": list(os.getloadavg()),
               "cpu_busy": 1.0 - (idle1 - idle0) / max(tot1 - tot0, 1), "gpu_start": g0, "gpu_mid": gmid,
               "gpu_end": g1}
        blk.update(E.summarize(ms))
        blocks.append(blk)
        return blk

    def save(done):
        out = {"label": "E8 main measurement (DESIGN_ext 1.4)", "settings": settings, "env": env, "blocks": blocks,
               "pairs": pairs, "rounds_done": done, "t_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                                 time.gmtime(t_start)),
               "secs": time.time() - t_start}
        dump_json(f"{a.out}/e8_main.json", out)
        tmp = f"{a.out}/e8_reps.tmp{os.getpid()}.npz"
        np.savez_compressed(tmp, **reps)
        os.replace(tmp, f"{a.out}/e8_reps.npz")

    for rd in range(a.rounds):
        tr = time.time()
        for upos, ui in enumerate(rng.permutation(len(units))):
            u = units[ui]
            cs = E.unit_conds(u)
            if u["kind"] == "pair" and rng.random() < 0.5:
                cs = cs[::-1]
            first = cs[0][3]
            med = {}
            for vpos, c in enumerate(cs):
                blk = run_block(c, rd, upos, vpos, first)
                med[c[3].split("-")[1] if u["kind"] == "pair" else c[3]] = blk["median_ms"]
            if u["kind"] == "pair":
                pairs.append({"mode": u["mode"], "size": u["size"], "N": u["N"], "prec": u["prec"],
                              "impl": u["impl"], "round": rd, "first": first, "det_ms": med["det"],
                              "shiter_ms": med["shiter"], "overhead": (med["shiter"] - med["det"]) / med["det"]})
        save(rd + 1)
        log(f"round {rd} done in {time.time() - tr:.0f}s (total {time.time() - t_start:.0f}s)")
    log("E8_MAIN_DONE")


if __name__ == "__main__":
    main()
