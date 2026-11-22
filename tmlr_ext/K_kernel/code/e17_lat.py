"""E17 latency (tmlr_ext/DESIGN_tmlr.md 7, K1) on one exclusive A40, single process, no MPS: E8's protocol.

Conditions: graph-WM (E8's primary mode: the world-model rollout of one control step = policy-prior rollout (batch 24,
dynamics x 2) + 6 MPPI iterations x horizon 3 x (reward, dynamics) at batch N, captured as one CUDA graph and replayed)
for model sizes 5M and 317M (E8's randomly initialized official TD-MPC2 models, mw-push dims, E8's random-input
activation ranges) x N in {512, 2048, 8192}; precisions int-det, int-shiter (E8's code, e8lib unchanged) and
int-detshift (e17lat); GEMM path 'a' (E8's selection for every size) with E8's A40 tile configs (inputs/gemm_cfg_e8.json);
4-bit codes (E8's W8A4 latency setting; the code width inside the int8 container does not change the arithmetic cost).
Units of one round: per (size, N) two pairs, (det, detshift) and (det, shiter); a pair = the det block and the variant
block back to back in random order. 12 units, 24 blocks per round, shuffled anew every round (numpy default_rng(0):
one permutation per round, one draw per pair for its order), 10 rounds.
Block (E8): 30 warmup replays, then n timed replays, each between a CUDA event pair followed by a device sync; block
value = median per-replay time; n = max(10, ceil(2000 ms / t_rep(det))) from the smoke calibration, both blocks of a
pair use the det n. The dither state of the int MLPs is reset to base 7 (70000 + round) + 13 before every block (E8).
Recorded per block: round, unit position, position in the unit, all per-replay times (npz), block statistics, start /
end time, host load, host CPU busy share, GPU SM / memory clock, temperature, power, clock-event reasons and visible
compute processes at the block start, middle and end. Results are rewritten after every round.

Stages
  smoke   env, ranges (asserted equal to E8's smoke values), correctness of int-detshift against an fp32 fake-quant
          reference with the same W8 weights and offsets (teacher forcing per layer, c_eff of both layers, counters,
          fresh offsets per forward, shared by rows, graph replay of one MLP forward and of the rollout), E8's det /
          sh-iter rollout checks, a descriptive row-kernel bench (det vs det+shift without / with the fused GEMV), and
          the block-length calibration -> <out>/smoke/{env,ranges,checks,rowbench,calib}.json
  main    the measurement -> <out>/main/e17_lat_main.json, e17_lat_reps.npz
  quick   code-path check of main (1 round, n = 3, warmup 2; numbers void) -> <out>/quick/
usage: python e17_lat.py smoke|main|quick --out /workspace/atm/out/latency
"""
import argparse
import copy
import json
import math
import os
import platform
import subprocess
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import triton

import e8lib as E
import e17lat as S

SIZES = (5, 317)
NS = (512, 2048, 8192)
PRECS = ("int-det", "int-shiter", "int-detshift")
IMPL = "a"
TOL_MISMATCH = 1e-3     # E8 smoke tolerances
TOL_HEAD = 1e-4
HERE = os.path.dirname(os.path.abspath(__file__))
E8_RANGES = f"{HERE}/inputs/ranges_e8_latency.json"   # E8 results/smoke/ranges.json (same models, same seeds)


def log(*a):
    print(time.strftime("%H:%M:%S", time.gmtime()), *a, flush=True)


def dump(path, obj, indent=1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp, path)


def progress(out, **kw):
    kw["updated"] = E.now_utc()
    dump(f"{out}/progress.json", kw, indent=None)


def gemm_cfgs():
    d = json.load(open(f"{HERE}/inputs/gemm_cfg_e8.json"))
    return {tuple(int(t) for t in k.split(",")): tuple(v) for k, v in d.items()}


def env_info():
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,power.limit,"
                          "memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "triton": triton.__version__,
            "gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
            "sm_count": torch.cuda.get_device_properties(0).multi_processor_count, "nvidia_smi": smi,
            "python": platform.python_version(), "nproc": os.cpu_count(), "tdmpc2_dir": E.TDMPC2_DIR,
            "allow_tf32_default": torch.backends.cuda.matmul.allow_tf32, "time": E.now_utc(),
            "mps_pipe_env": os.environ.get("CUDA_MPS_PIPE_DIRECTORY")}


def gpu_compute_procs():
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                       capture_output=True, text=True).stdout.strip()
    return [x for x in r.splitlines() if x.strip()]


def build_models():
    agents, ranges = {}, {}
    for size in SIZES:
        agents[size] = E.build_agent(size)[0]
        ranges[size] = E.calib_ranges(agents[size].model)
    return agents, ranges


# ---------------------------------------------------------------------------------------------------- checks
@torch.no_grad()
def ref_chain(mlp, x, c0, c1, o0, o1):
    """fp32 fake-quant det+shift reference of one forward with the same W8 weights and offsets. Teacher forcing: the
    layer-1 reference reads the kernel's layer-0 codes c0, the head reads the kernel's layer-1 codes c1."""
    L0, L1, L2 = mlp.seq[0], mlp.seq[1], mlp.seq[2]
    W1hat = mlp.W1q.float() * mlp.s1[:, None]
    W2hat = mlp.W2q.float() * mlp.s2[:, None]

    def codes(ylin, ln, lo, hi, d, o):
        m = F.mish(F.layer_norm(ylin, (ylin.shape[-1],), ln.weight, ln.bias, E.LN_EPS))
        xg = (m.clamp(lo, hi) - lo) * (1.0 / d)
        return torch.round(xg - o).clamp(0, mlp.nl)

    def head(c_in):
        y2 = F.linear(mlp.lo1 + mlp.d1 * (c_in + o1), W2hat, L2.bias)
        return L2.act(L2.ln(y2)) if mlp.dyn_head else y2

    r0 = codes(F.linear(x, L0.weight, L0.bias), L0.ln, mlp.lo0, mlp.hi0, mlp.d0, o0)
    r1 = codes(F.linear(mlp.lo0 + mlp.d0 * (c0 + o0), W1hat, L1.bias), L1.ln, mlp.lo1, mlp.hi1, mlp.d1, o1)
    y2 = head(c1)
    e1 = codes(F.linear(mlp.lo0 + mlp.d0 * (r0 + o0), W1hat, L1.bias), L1.ln, mlp.lo1, mlp.hi1, mlp.d1, o1)
    ye = head(e1)
    c1ref = mlp.c1.double() + mlp.alpha1.double() * (mlp.W1q.double() @ o0.double())
    c2ref = mlp.c2.double() + mlp.alpha2.double() * (mlp.W2q.double() @ o1.double())
    return r0, r1, y2, ye, c1ref, c2ref


@torch.no_grad()
def check_forward_s(mlp, x, fwd=None):
    st0 = mlp.state.clone()
    o0 = E.u_vector(mlp.state, mlp.width, mlp.salt + 0) - 0.5
    o1 = E.u_vector(mlp.state, mlp.width, mlp.salt + 1) - 0.5
    q0, q1, out = (fwd or (lambda: mlp._forward(x)))()
    q0, q1, out = q0.clone(), q1.clone(), out.clone()
    st1 = mlp.state.clone()
    c0, c1 = q0.float() + mlp.zp, q1.float() + mlp.zp
    r0, r1, y2, ye, c1ref, c2ref = ref_chain(mlp, x, c0, c1, o0, o1)
    scale = max(1.0, float(y2.abs().max()))
    rel = lambda a, b: float((a.double() - b).abs().max() / b.abs().max().clamp(min=1e-12))
    d = {"mismatch0": float((c0 != r0).float().mean()), "maxdiff0": float((c0 - r0).abs().max()),
         "mismatch1": float((c1 != r1).float().mean()), "maxdiff1": float((c1 - r1).abs().max()),
         "head_maxabs": float((out.reshape(y2.shape) - y2).abs().max()), "head_scale": scale,
         "c1eff_relmax": rel(mlp.c1eff, c1ref), "c2eff_relmax": rel(mlp.c2eff, c2ref),
         "e2e_rel_l2": float((out.reshape(ye.shape) - ye).norm() / ye.norm().clamp(min=1e-12)),
         "counter_step": int(st1[1] - st0[1]),
         "o_min": float(min(o0.min(), o1.min())), "o_max": float(max(o0.max(), o1.max()))}
    d["pass"] = bool(d["mismatch0"] <= TOL_MISMATCH and d["maxdiff0"] <= 1 and d["mismatch1"] <= TOL_MISMATCH
                     and d["maxdiff1"] <= 1 and d["head_maxabs"] <= TOL_HEAD * scale and d["c1eff_relmax"] <= 1e-5
                     and d["c2eff_relmax"] <= 1e-5 and d["counter_step"] == 1 and d["o_min"] >= -0.5
                     and d["o_max"] < 0.5)
    return d, (q0, q1, out)


def randomized_reward(seq, seed=1):
    """copy of the reward MLP with the zero-initialized head re-randomized (checks only; E8 smoke)."""
    s2 = copy.deepcopy(seq)
    g = torch.Generator(device=s2[2].weight.device)
    g.manual_seed(seed)
    with torch.no_grad():
        w = torch.randn(s2[2].weight.shape, generator=g, device=s2[2].weight.device) * 0.02
        s2[2].weight.copy_(w.clamp(-0.04, 0.04))
    return s2


@torch.no_grad()
def checks_size(size, agent, qr, cfgs):
    model = agent.model
    out = {}
    g = torch.Generator(device="cuda")
    g.manual_seed(100 + size)
    obs = torch.randn(512, E.OBS_DIM, device="cuda", generator=g)
    a = torch.rand(512, E.ACT_DIM, device="cuda", generator=g) * 2 - 1
    x = torch.cat([model.encode(obs, None), a], -1)
    x24 = x[:24].contiguous()
    xsame = x[:1].repeat(512, 1).contiguous()
    rew_seq = randomized_reward(model._reward)
    for name, seq, salt, dyn in (("dyn", model._dynamics, 0, True), ("rew", rew_seq, 4, False)):
        m = S.GenIntMLPShift(seq, qr[f"{name}0"], qr[f"{name}1"], salt, dyn, IMPL, cfgs)
        m.new_episode(4242)
        r = {}
        r["M512"], (q0a, q1a, _) = check_forward_s(m, x)
        if dyn:
            r["M24"], _ = check_forward_s(m, x24)
        r["M512_second"], (q0b, q1b, _) = check_forward_s(m, x)
        r["new_o_share_changed0"] = float((q0a != q0b).float().mean())
        r["new_o_share_changed1"] = float((q1a != q1b).float().mean())
        _, (s0, s1, so) = check_forward_s(m, xsame)
        r["identical_rows_equal"] = bool((s0 == s0[:1]).all() and (s1 == s1[:1]).all() and (so == so[:1]).all())
        md = E.GenIntMLP(seq, qr[f"{name}0"], qr[f"{name}1"], False, salt, dyn, IMPL, cfgs)
        r["det_vs_detshift_share_diff0"] = float((md._forward(x)[0] != m._forward(x)[0]).float().mean())
        r["pass_shift"] = bool(r["new_o_share_changed0"] > 0.05 and r["new_o_share_changed1"] > 0.05
                               and r["identical_rows_equal"] and r["det_vs_detshift_share_diff0"] > 0.05)
        # graph replay of one MLP forward: every replay matches its reference (fresh offsets per replay)
        m.new_episode(999)
        xs = x.clone()
        gr = E.GraphRunner(lambda: m._forward(xs), warmup=1)
        reps, prev = [], None
        for _ in range(3):
            d, (q0, _, _) = check_forward_s(m, xs, fwd=lambda: (gr(), gr.out)[1])
            d["changed_vs_prev0"] = None if prev is None else float((q0 != prev).float().mean())
            prev = q0
            reps.append(d)
        r["graph_mlp"] = {"replays": reps, "pass": bool(all(z["pass"] for z in reps)
                                                         and all(z["changed_vs_prev0"] > 0.05 for z in reps[1:]))}
        del gr
        r["pass"] = bool(r["M512"]["pass"] and r["M512_second"]["pass"] and (not dyn or r["M24"]["pass"])
                         and r["pass_shift"] and r["graph_mlp"]["pass"])
        out[name] = r
        log(f"check {size} {name}: pass={r['pass']} " + json.dumps({k: (v["pass"] if isinstance(v, dict) else v)
                                                                   for k, v in r.items()}))
    return out


@torch.no_grad()
def checks_rollout(size, agent, qr, cfgs):
    """graph replay of the whole rollout at N 512: counter steps 20 (dynamics) / 18 (reward) per replay; det replays
    bitwise equal and equal to eager; sh-iter / det+shift replays differ (E8 smoke)."""
    model = agent.model
    inp = E.wm_inputs(model, 512)
    res = {}
    for prec in PRECS:
        wm = S.make_wm_s(model, prec, qr, IMPL, cfgs)
        wm.new_episode(31337)
        gr = E.GraphRunner(lambda: E.wm_rollout(wm, *inp), warmup=1)
        s0 = (wm.dyn.state.clone(), wm.rew.state.clone())
        gr()
        z1 = gr.out[0].clone()
        s1 = (wm.dyn.state.clone(), wm.rew.state.clone())
        gr()
        z2 = gr.out[0].clone()
        s2 = (wm.dyn.state.clone(), wm.rew.state.clone())
        ze = E.wm_rollout(wm, *inp)[0]
        step = [int(s1[0][1] - s0[0][1]), int(s1[1][1] - s0[1][1]), int(s2[0][1] - s1[0][1]), int(s2[1][1] - s1[1][1])]
        r = {"counter_steps_dyn_rew_rep1_rep2": step, "replays_differ": bool((z1 != z2).any()),
             "finite": bool(torch.isfinite(z1).all() and torch.isfinite(z2).all())}
        if prec == "int-det":
            r["graph_vs_eager_maxabs"] = float((z1 - ze).abs().max())
            r["pass"] = bool(step == [0, 0, 0, 0] and not r["replays_differ"] and r["graph_vs_eager_maxabs"] <= 1e-5)
        else:
            r["pass"] = bool(step == [20, 18, 20, 18] and r["replays_differ"] and r["finite"])
        res[prec] = r
        log(f"rollout check {size} {prec}: {r}")
        del gr
    return res


@torch.no_grad()
def rowbench(cfgs):
    """descriptive: the layer-0 row kernel (MODE 0 with the next-u draw) of det+shift without and with the fused GEMV,
    and E8's det / sh-iter row kernel, do_bench ms at the shapes of the plan."""
    res = {}
    st = torch.zeros(2, dtype=torch.int64, device="cuda")
    for M, W, gn in ((24, 512, 512), (512, 512, 512), (2048, 512, 512), (8192, 512, 512), (24, 4096, 4096),
                     (512, 4096, 4096), (2048, 4096, 4096), (8192, 4096, 4096), (512, 4096, 1376), (512, 4096, 101)):
        y = torch.randn(M, W, device="cuda")
        gg, bb = torch.ones(W, device="cuda"), torch.zeros(W, device="cuda")
        q = torch.empty(M, W, device="cuda", dtype=torch.int8)
        ub = torch.rand(2, W, device="cuda")
        gw = torch.randint(-127, 128, (gn, W), device="cuda", dtype=torch.int8)
        ga, gc, go = torch.rand(gn, device="cuda"), torch.rand(gn, device="cuda"), torch.empty(gn, device="cuda")
        fns = {"det": lambda: E.row_launch(y, None, None, gg, bb, q, st, M, W, -0.3, 3.0, 0.2, 15.0, 0.0, 0, False,
                                           False),
               "shiter_gen": lambda: E.row_launch(y, None, None, gg, bb, q, st, M, W, -0.3, 3.0, 0.2, 15.0, 0.0, 0,
                                                  False, True, u=ub[0], ug=ub[1], wg=W, gsalt=1),
               "detshift_gen_nogemv": lambda: S.row_launch_s(y, gg, bb, q, st, M, W, -0.3, 3.0, 0.2, 15.0, 0.0, ub[0],
                                                             ug=ub[1], wg=W, gsalt=1),
               "detshift_gen_gemv": lambda: S.row_launch_s(y, gg, bb, q, st, M, W, -0.3, 3.0, 0.2, 15.0, 0.0, ub[0],
                                                           ug=ub[1], wg=W, gsalt=1, gw=gw, ga=ga, gc=gc, gout=go,
                                                           gn=gn)}
        for name, fn in fns.items():
            fn()
            torch.cuda.synchronize()
            res[f"M{M}|W{W}|GN{gn}|{name}"] = float(triton.testing.do_bench(fn, warmup=10, rep=50))
        log(f"rowbench M{M} W{W} GN{gn}: " + " ".join(f"{k.split('|')[-1]}={v * 1e3:.1f}us"
                                                      for k, v in res.items() if k.startswith(f"M{M}|W{W}|GN{gn}|")))
    return res


def quick_rep(fn, pool, min_s=0.3, min_n=3, max_n=200, warm=3):
    for _ in range(warm):
        fn()
        torch.cuda.synchronize()
    ms = []
    while len(ms) < min_n or (sum(ms) < 1e3 * min_s and len(ms) < max_n):
        x, _ = E.time_block(fn, 0, 1, pool)
        ms += x
    return float(np.median(ms))


def cond_key(size, N, prec):
    return f"graph|{size}|{N}|{prec}|{IMPL}"


def units():
    """one round: per (size, N) the pairs (det, detshift) and (det, shiter)."""
    return [{"size": s, "N": n, "variant": v} for s in SIZES for n in NS for v in ("int-detshift", "int-shiter")]


def build_graphs(agents, ranges, cfgs):
    wms, graphs, inputs = {}, {}, {}
    for size in sorted(SIZES, reverse=True):            # heaviest first (E8c work order)
        for N in sorted(NS, reverse=True):
            inputs[(size, N)] = E.wm_inputs(agents[size].model, N)
            for prec in PRECS:
                if (size, prec) not in wms:
                    wms[(size, prec)] = S.make_wm_s(agents[size].model, prec, ranges[size], IMPL, cfgs)
                wm = wms[(size, prec)]
                wm.new_episode(1)
                graphs[cond_key(size, N, prec)] = E.GraphRunner(lambda w=wm, i=inputs[(size, N)]: E.wm_rollout(w, *i))
    torch.cuda.synchronize()
    return wms, graphs


def calibrate(graphs, pool):
    t = {k: quick_rep(g, pool) for k, g in graphs.items()}
    n_reps = {}
    est = 0.0
    for u in units():
        kd, kv = cond_key(u["size"], u["N"], "int-det"), cond_key(u["size"], u["N"], u["variant"])
        n = max(10, math.ceil(2000.0 / t[kd]))
        n_reps[f"{kd}|pair_{u['variant']}"] = n
        est += (30 + n) * (t[kd] + t[kv]) / 1e3 + 0.1
    return {"t_rep_ms": t, "n_reps": n_reps, "est_round_s": est, "est_main_s_10_rounds": 10 * est,
            "rule": "n = max(10, ceil(2000 ms / t_rep(det))); both blocks of a pair use the det n"}


def stage_smoke(a):
    out = f"{a.out}/smoke"
    os.makedirs(out, exist_ok=True)
    progress(a.out, stage="smoke", note="latency smoke")
    dump(f"{out}/env.json", env_info())
    cfgs = gemm_cfgs()
    agents, ranges = build_models()
    e8r = json.load(open(E8_RANGES))
    cmp_ = {str(s): {k: [ranges[s][k][i] - e8r[str(s)][k][i] for i in (0, 1)] for k in ranges[s]} for s in SIZES}
    dump(f"{out}/ranges.json", {str(k): v for k, v in ranges.items()})
    dump(f"{out}/ranges_vs_e8.json", {"diff_this_minus_e8": cmp_, "max_abs": max(abs(x) for s in cmp_.values()
                                                                                for v in s.values() for x in v)})
    log(f"ranges vs E8 pod: max abs diff {max(abs(x) for s in cmp_.values() for v in s.values() for x in v):.3g}")
    allc = {}
    for size in SIZES:
        try:
            allc[str(size)] = {"mlp": checks_size(size, agents[size], ranges[size], cfgs),
                               "rollout": checks_rollout(size, agents[size], ranges[size], cfgs)}
        except Exception:  # noqa: BLE001
            allc[str(size)] = {"error": traceback.format_exc()}
            log(traceback.format_exc())
        dump(f"{out}/checks.json", allc)
        torch.cuda.empty_cache()
    ok = all("error" not in v and all(r["pass"] for r in v["mlp"].values())
             and all(r["pass"] for r in v["rollout"].values()) for v in allc.values())
    allc["ALL_PASS"] = bool(ok)
    dump(f"{out}/checks.json", allc)
    log(f"checks ALL_PASS={ok}")
    dump(f"{out}/rowbench.json", rowbench(cfgs))
    pool = E.EventPool(4096)
    wms, graphs = build_graphs(agents, ranges, cfgs)
    cal = calibrate(graphs, pool)
    cal["gpu_compute_procs"] = gpu_compute_procs()
    dump(f"{out}/calib.json", cal)
    log(f"calib: est round {cal['est_round_s']:.0f} s, 10 rounds {cal['est_main_s_10_rounds'] / 60:.1f} min")
    progress(a.out, stage="smoke_done", note="latency smoke done")
    log("E17_LAT_SMOKE_DONE")


def stage_main(a, quick=False):
    out = f"{a.out}/{'quick' if quick else 'main'}"
    os.makedirs(out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    cal = json.load(open(f"{a.out}/smoke/calib.json"))
    n_reps = cal["n_reps"]
    rounds, warmup = (1, 2) if quick else (a.rounds, 30)
    if quick:
        n_reps = {k: 3 for k in n_reps}
    if os.environ.get("CUDA_MPS_PIPE_DIRECTORY") and not quick:
        raise SystemExit("REFUSED: MPS pipe set; the latency protocol needs a non-MPS exclusive GPU")
    procs0 = gpu_compute_procs()
    if procs0 and not quick:
        raise SystemExit(f"REFUSED: other compute processes on the GPU: {procs0}")
    cfgs = gemm_cfgs()
    agents, ranges = build_models()
    sr = json.load(open(f"{a.out}/smoke/ranges.json"))
    for size in SIZES:
        for k, v in sr[str(size)].items():
            assert all(abs(ranges[size][k][i] - v[i]) <= 1e-6 * max(1.0, abs(v[i])) for i in (0, 1)), (size, k)
    gpu = E.GpuState()
    pool = E.EventPool(max(n_reps.values()) + 8)
    t_start = time.time()
    t0 = time.time()
    wms, graphs = build_graphs(agents, ranges, cfgs)
    t_build = time.time() - t0
    log(f"built {len(graphs)} graphs in {t_build:.0f}s; allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    us = units()
    settings = {"rounds": rounds, "warmup": warmup, "seed": a.seed, "quick": quick, "n_reps": n_reps,
                "impl": IMPL, "sizes": list(SIZES), "ns": list(NS), "kbits": E.KBITS,
                "gemm_cfgs": {",".join(map(str, k)): list(v) for k, v in cfgs.items()},
                "ranges": {str(k): v for k, v in ranges.items()}, "n_units_per_round": len(us),
                "n_blocks_per_round": 2 * len(us), "graph_build_s": t_build}
    env = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
           "triton": triton.__version__, "gpu_compute_procs_at_start": procs0, "gpu_state_at_start": gpu.snap()}
    blocks, pairs, reps = [], [], {}
    rng = np.random.default_rng(a.seed)

    def run_block(size, N, prec, rd, upos, vpos, first, n):
        key = cond_key(size, N, prec)
        wms[(size, prec)].new_episode(7 * (70000 + rd) + 13)
        fn = graphs[key]
        load0 = os.getloadavg()
        tot0, idle0 = E.cpu_times()
        g0 = gpu.snap()
        tb0 = time.time()
        with torch.no_grad():
            ms, gmid = E.time_block(fn, warmup, n, pool, mid=gpu.snap)
        tb1 = time.time()
        tot1, idle1 = E.cpu_times()
        g1 = gpu.snap()
        assert torch.backends.cuda.matmul.allow_tf32 is False
        idx = len(blocks)
        reps[f"b{idx}"] = np.asarray(ms, np.float32)
        blk = {"idx": idx, "key": key, "size": size, "N": N, "prec": prec, "impl": IMPL, "round": rd,
               "unit_pos": int(upos), "pos_in_unit": int(vpos), "first_in_unit": first, "n_reps": n,
               "t_start": tb0, "t_end": tb1, "load_start": list(load0), "load_end": list(os.getloadavg()),
               "cpu_busy": 1.0 - (idle1 - idle0) / max(tot1 - tot0, 1), "gpu_start": g0, "gpu_mid": gmid,
               "gpu_end": g1}
        blk.update(E.summarize(ms))
        blocks.append(blk)
        return blk

    def save(done):
        res = {"label": "E17 latency main measurement (DESIGN_tmlr 7, K1; E8 protocol)", "settings": settings,
               "env": env, "blocks": blocks, "pairs": pairs, "rounds_done": done,
               "t_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)), "secs": time.time() - t_start,
               "gpu_compute_procs_at_end": gpu_compute_procs()}
        dump(f"{out}/e17_lat_main.json", res, indent=None)
        tmp = f"{out}/e17_lat_reps.tmp{os.getpid()}.npz"
        np.savez_compressed(tmp, **reps)
        os.replace(tmp, f"{out}/e17_lat_reps.npz")

    for rd in range(rounds):
        tr = time.time()
        progress(a.out, stage="quick" if quick else "main", round=rd, rounds=rounds,
                 note=f"latency round {rd + 1}/{rounds} (exclusive)")
        for upos, ui in enumerate(rng.permutation(len(us))):
            u = us[ui]
            vs = ["int-det", u["variant"]]
            if rng.random() < 0.5:
                vs = vs[::-1]
            n = n_reps[f"{cond_key(u['size'], u['N'], 'int-det')}|pair_{u['variant']}"]
            med = {}
            for vpos, prec in enumerate(vs):
                med[prec] = run_block(u["size"], u["N"], prec, rd, upos, vpos, vs[0], n)["median_ms"]
            pairs.append({"size": u["size"], "N": u["N"], "variant": u["variant"], "round": rd, "first": vs[0],
                          "det_ms": med["int-det"], "var_ms": med[u["variant"]],
                          "overhead": (med[u["variant"]] - med["int-det"]) / med["int-det"]})
        save(rd + 1)
        log(f"round {rd} done in {time.time() - tr:.0f}s (total {time.time() - t_start:.0f}s)")
    progress(a.out, stage="quick_done" if quick else "main_done", note="latency done")
    log("E17_LAT_MAIN_DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["smoke", "main", "quick"])
    ap.add_argument("--out", default="/workspace/atm/out/latency")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    if a.stage == "smoke":
        stage_smoke(a)
    else:
        stage_main(a, quick=a.stage == "quick")


if __name__ == "__main__":
    main()
