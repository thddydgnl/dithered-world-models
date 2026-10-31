"""E8 smoke on the pod (before PLAN.md and the main measurement). Sections (each writes its JSON in --out):
  env        torch / triton / GPU / driver versions
  tune       tile config per (M, N, K) of GEMM 'a' (Triton int8), int-det GEMM time only (do_bench); also the
             torch._int_mm time of the same shapes (descriptive)            -> gemm_cfg.json, gemm_tune.json
  checks     correctness on every size, GEMM 'a' and 'b' (and E7's fused kernels at 5M):
               int-det vs the fp32 fake-quant reference on the same grid (W8 weights dequantized, teacher forcing per
               layer: codes of layer 0 / 1 and the head output), plus the end-to-end difference
               int-shiter vs the same reference with u = Philox of the pre-forward counter state; new u every forward;
               identical codes for identical rows (u shared by the candidates); 'a' and 'b' give the same codes for
               the same state; graph replay of one MLP forward (3 replays) and of the whole rollout (counter steps
               20 dynamics / 18 reward forwards per replay, det replays bitwise equal, shiter replays differ)
             The official init zeroes the reward head; the reward checks use a copy with a re-randomized head
             (trunc_normal std 0.02) so that the head GEMM is exercised.        -> checks.json
  select     GEMM path per size by int-det graph-WM latency only: 3 rounds x {a, b} x N in {512, 2048, 8192},
             block = 30 warmup + max(10, ceil(1 s / t_rep)) timed replays; T = median of the 3 block medians;
             per size the path with the smaller geometric mean of T over N wins   -> selection.json
  calib      rep time of every main-run condition (selected path), block lengths n = max(10, ceil(2 s / t_rep)) and
             the estimated main-run time                                        -> calib.json
  rowbench   row-kernel time det vs dither at a few shapes (descriptive)      -> rowbench.json
usage: python e8_smoke.py --out /workspace/results/smoke [--sections env,tune,checks,select,calib,rowbench]
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

TOL_MISMATCH = 1e-3     # max share of codes that differ from the reference (floating-point order at the boundaries)
TOL_HEAD = 1e-4         # max |out - ref| / max(1, max |ref|) of the head given identical codes


def log(*a):
    print(time.strftime("%H:%M:%S", time.gmtime()), *a, flush=True)


def dump(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def env_info():
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,power.limit,"
                          "memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "triton": triton.__version__,
            "gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
            "sm_count": torch.cuda.get_device_properties(0).multi_processor_count, "nvidia_smi": smi,
            "python": platform.python_version(), "nproc": os.cpu_count(), "tdmpc2_dir": E.TDMPC2_DIR,
            "allow_tf32_default": torch.backends.cuda.matmul.allow_tf32, "time": E.now_utc()}


# ---------------------------------------------------------------------------------------------------- reference
@torch.no_grad()
def ref_codes(ylin, ln_w, ln_b, lo, hi, d, nl, u=None):
    m = F.mish(F.layer_norm(ylin, (ylin.shape[-1],), ln_w, ln_b, E.LN_EPS))
    x = (m.clamp(lo, hi) - lo) * (1.0 / d)
    q = torch.floor(x + u) if u is not None else torch.round(x)
    return q.clamp(0, nl)


@torch.no_grad()
def compare(mlp, x, q0, q1, out, u0=None, u1=None):
    """teacher-forced per-layer comparison of one forward's (q0, q1, out) with the fp32 fake-quant reference."""
    L0, L1, L2 = mlp.seq[0], mlp.seq[1], mlp.seq[2]
    c0, c1 = q0.float() + mlp.zp, q1.float() + mlp.zp
    r0 = ref_codes(F.linear(x, L0.weight, L0.bias), L0.ln.weight, L0.ln.bias, mlp.lo0, mlp.hi0, mlp.d0, mlp.nl, u0)
    W1hat = mlp.W1q[: mlp.width].float() * mlp.s1[:, None]
    W2hat = mlp.W2q[: mlp.n_out].float() * mlp.s2[:, None]
    r1 = ref_codes(F.linear(mlp.lo0 + mlp.d0 * c0, W1hat, L1.bias), L1.ln.weight, L1.ln.bias, mlp.lo1, mlp.hi1,
                   mlp.d1, mlp.nl, u1)
    y2 = F.linear(mlp.lo1 + mlp.d1 * c1, W2hat, L2.bias)
    if mlp.dyn_head:
        y2 = L2.act(L2.ln(y2))
    # end to end (no teacher forcing)
    e1 = ref_codes(F.linear(mlp.lo0 + mlp.d0 * r0, W1hat, L1.bias), L1.ln.weight, L1.ln.bias, mlp.lo1, mlp.hi1,
                   mlp.d1, mlp.nl, u1)
    ye = F.linear(mlp.lo1 + mlp.d1 * e1, W2hat, L2.bias)
    if mlp.dyn_head:
        ye = L2.act(L2.ln(ye))
    scale = max(1.0, float(y2.abs().max()))
    d = {"mismatch0": float((c0 != r0).float().mean()), "maxdiff0": float((c0 - r0).abs().max()),
         "mismatch1": float((c1 != r1).float().mean()), "maxdiff1": float((c1 - r1).abs().max()),
         "head_maxabs": float((out.reshape(y2.shape) - y2).abs().max()), "head_scale": scale,
         "e2e_rel_l2": float((out.reshape(ye.shape) - ye).norm() / ye.norm().clamp(min=1e-12)),
         "e2e_maxabs": float((out.reshape(ye.shape) - ye).abs().max()),
         "out_std": float(out.float().std())}
    d["pass"] = bool(d["mismatch0"] <= TOL_MISMATCH and d["maxdiff0"] <= 1 and d["mismatch1"] <= TOL_MISMATCH
                     and d["maxdiff1"] <= 1 and d["head_maxabs"] <= TOL_HEAD * scale)
    return d


@torch.no_grad()
def check_forward(mlp, x, fwd=None):
    """one forward (eager, or fwd() = a graph replay returning (q0, q1, out)) checked against the reference; for
    dither, u0 / u1 are the Philox vectors of the counter state before the forward."""
    st0 = mlp.state.clone()
    u0 = u1 = None
    if mlp.dither:
        u0 = E.u_vector(mlp.state, mlp.width, mlp.salt + 0)
        u1 = E.u_vector(mlp.state, mlp.width, mlp.salt + 1)
    q0, q1, out = (fwd or (lambda: mlp._forward(x)))()
    q0, q1, out = q0.clone(), q1.clone(), out.clone()
    st1 = mlp.state.clone()
    d = compare(mlp, x, q0, q1, out, u0, u1)
    d["counter_step"] = int(st1[1] - st0[1])
    d["pass"] = d["pass"] and d["counter_step"] == (1 if mlp.dither else 0)
    return d, (q0, q1, out)


def randomized_reward(seq, seed=1):
    """copy of the reward MLP with the zero-initialized head re-randomized (checks only)."""
    s2 = copy.deepcopy(seq)
    g = torch.Generator(device=s2[2].weight.device)
    g.manual_seed(seed)
    with torch.no_grad():
        w = torch.randn(s2[2].weight.shape, generator=g, device=s2[2].weight.device) * 0.02
        s2[2].weight.copy_(w.clamp(-0.04, 0.04))
    return s2


@torch.no_grad()
def checks_size(size, agent, qr, gemm_cfgs):
    model = agent.model
    out = {}
    g = torch.Generator(device="cuda")
    g.manual_seed(100 + size)
    obs = torch.randn(512, E.OBS_DIM, device="cuda", generator=g)
    a = torch.rand(512, E.ACT_DIM, device="cuda", generator=g) * 2 - 1
    x = torch.cat([model.encode(obs, None), a], -1)            # 512 distinct rows
    x24 = x[:24].contiguous()
    xsame = x[:1].repeat(512, 1).contiguous()                  # identical rows
    rew_seq = randomized_reward(model._reward)
    impls = ["a", "b"] + (["e7"] if size == 5 else [])
    mlps = {}
    for impl in impls:
        for dith in (False, True):
            for name, seq, salt, dyn in (("dyn", model._dynamics, 0, True), ("rew", rew_seq, 4, False)):
                if impl == "e7":
                    import e7dev
                    m = e7dev.E7IntMLP(seq, qr[f"{name}0"], qr[f"{name}1"], dith, salt, dyn)
                else:
                    m = E.GenIntMLP(seq, qr[f"{name}0"], qr[f"{name}1"], dith, salt, dyn, impl, gemm_cfgs)
                m.new_episode(4242)
                mlps[(impl, dith, name)] = m
    for (impl, dith, name), m in mlps.items():
        key = f"{impl}|{'shiter' if dith else 'det'}|{name}"
        r = {}
        r["M512"], (q0a, q1a, _) = check_forward(m, x)
        if name == "dyn":
            r["M24"], _ = check_forward(m, x24)
        if dith:
            r["M512_second"], (q0b, q1b, _) = check_forward(m, x)
            r["new_u_share_changed0"] = float((q0a != q0b).float().mean())
            r["new_u_share_changed1"] = float((q1a != q1b).float().mean())
            _, (s0, s1, so) = check_forward(m, xsame)
            r["identical_rows_equal"] = bool((s0 == s0[:1]).all() and (s1 == s1[:1]).all()
                                             and (so == so[:1]).all())
            r["pass_dither"] = bool(r["new_u_share_changed0"] > 0.05 and r["new_u_share_changed1"] > 0.05
                                    and r["identical_rows_equal"])
        out[key] = r
        log(f"check {size} {key}: " + " ".join(f"{k}={v['pass'] if isinstance(v, dict) else v}" for k, v in r.items()))
    # det vs shiter codes differ (the dither is applied)
    for impl in impls:
        for name in ("dyn", "rew"):
            md, ms = mlps[(impl, False, name)], mlps[(impl, True, name)]
            q0d = md._forward(x)[0]
            q0s = ms._forward(x)[0]
            out[f"{impl}|det_vs_shiter_share_diff0|{name}"] = float((q0d != q0s).float().mean())
    # a and b: same codes for the same state
    for dith in (False, True):
        for name in ("dyn", "rew"):
            ma, mb = mlps[("a", dith, name)], mlps[("b", dith, name)]
            ma.new_episode(777)
            mb.new_episode(777)
            q0a, q1a, oa = [t.clone() for t in ma._forward(x)]
            q0b, q1b, ob = [t.clone() for t in mb._forward(x)]
            out[f"a_vs_b|{'shiter' if dith else 'det'}|{name}"] = {
                "q0_equal": bool((q0a == q0b).all()), "q1_mismatch": float((q1a != q1b).float().mean()),
                "out_maxabs": float((oa - ob).abs().max()),
                "pass": bool((q0a == q0b).all() and float((q1a != q1b).float().mean()) <= TOL_MISMATCH)}
    # graph replay of one MLP forward (shiter): fresh u at every replay, each replay matches its reference
    for impl in impls:
        m = mlps[(impl, True, "dyn")]
        m.new_episode(999)
        xs = x.clone()
        gr = E.GraphRunner(lambda: m._forward(xs), warmup=1)
        reps, prev = [], None
        for i in range(3):
            d, (q0, _, _) = check_forward(m, xs, fwd=lambda: (gr(), gr.out)[1])
            d["changed_vs_prev0"] = None if prev is None else float((q0 != prev).float().mean())
            prev = q0
            reps.append(d)
        out[f"graph_mlp|{impl}|shiter|dyn"] = {
            "replays": reps, "pass": bool(all(r["pass"] for r in reps)
                                          and all(r["changed_vs_prev0"] > 0.05 for r in reps[1:]))}
        del gr
    return out


@torch.no_grad()
def checks_rollout(size, agent, qr, gemm_cfgs, impls):
    """graph replay of the whole WM rollout at N = 512: counters and replay-to-replay behaviour; graph vs eager."""
    model = agent.model
    inp = E.wm_inputs(model, 512)
    res = {}
    for impl in impls:
        for prec in ("int-det", "int-shiter"):
            wm = E.make_wm(model, prec.replace("int", "e7") if impl == "e7" else prec, qr, impl, gemm_cfgs)
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
            step = [int(s1[0][1] - s0[0][1]), int(s1[1][1] - s0[1][1]), int(s2[0][1] - s1[0][1]),
                    int(s2[1][1] - s1[1][1])]
            dith = prec.endswith("shiter")
            r = {"counter_steps_dyn_rew_rep1_rep2": step, "replays_differ": bool((z1 != z2).any()),
                 "replay_maxabs_diff": float((z1 - z2).abs().max())}
            if dith:
                r["pass"] = bool(step == [20, 18, 20, 18] and r["replays_differ"])
            else:
                r["graph_vs_eager_maxabs"] = float((z1 - ze).abs().max())
                r["pass"] = bool(step == [0, 0, 0, 0] and not r["replays_differ"]
                                 and r["graph_vs_eager_maxabs"] <= 1e-5)
            res[f"{impl}|{prec}"] = r
            log(f"rollout check {size} {impl} {prec}: {r}")
            del gr
    for prec in E.FLOAT_PRECS:
        wm = E.make_wm(model, prec)
        gr = E.GraphRunner(lambda: E.wm_rollout(wm, *inp), warmup=1)
        gr()
        z1 = gr.out[0].clone()
        ze = E.wm_rollout(wm, *inp)[0]
        res[prec] = {"graph_vs_eager_maxabs": float((z1 - ze).abs().max()), "finite": bool(torch.isfinite(z1).all())}
        del gr
    return res


# ---------------------------------------------------------------------------------------------------- timing
def quick_rep(fn, pool, min_s=0.3, min_n=3, max_n=200, warm=3):
    for _ in range(warm):
        fn()
        torch.cuda.synchronize()
    ms, t0 = [], time.time()
    while len(ms) < min_n or (sum(ms) < 1e3 * min_s and len(ms) < max_n):
        x, _ = E.time_block(fn, 0, 1, pool)
        ms += x
    return float(np.median(ms))


def build_runner(mode, agent, wm, inp, obs):
    if mode == "graph":
        return E.GraphRunner(lambda: E.wm_rollout(wm, *inp))
    if mode == "eager":
        return lambda: E.wm_rollout(wm, *inp)
    return E.StepRunner(agent, obs)


def select_paths(agents, ranges, gemm_cfgs, pool, rounds=3):
    rng = np.random.default_rng(1)
    res = {}
    for size in E.SIZES:
        model = agents[size][0].model
        wms = {impl: E.make_wm(model, "int-det", ranges[size], impl, gemm_cfgs) for impl in ("a", "b")}
        runs, trep = {}, {}
        for N in E.NS:
            inp = E.wm_inputs(model, N)
            for impl in ("a", "b"):
                runs[(N, impl)] = E.GraphRunner(lambda w=wms[impl], i=inp: E.wm_rollout(w, *i))
                trep[(N, impl)] = quick_rep(runs[(N, impl)], pool)
        blocks = {k: [] for k in runs}
        for rd in range(rounds):
            order = list(runs.keys())
            for j in rng.permutation(len(order)):
                k = order[j]
                n = max(10, math.ceil(1000.0 / trep[k]))
                ms, _ = E.time_block(runs[k], 30, n, pool)
                blocks[k].append(float(np.median(ms)))
        T = {f"{N}|{impl}": float(np.median(v)) for (N, impl), v in blocks.items()}
        geo = {impl: float(np.exp(np.mean([np.log(T[f"{N}|{impl}"]) for N in E.NS]))) for impl in ("a", "b")}
        choice = "a" if geo["a"] < geo["b"] else "b"
        res[str(size)] = {"T_ms": T, "blocks_ms": {f"{N}|{impl}": v for (N, impl), v in blocks.items()},
                          "t_rep_quick_ms": {f"{N}|{impl}": v for (N, impl), v in trep.items()},
                          "geomean_ms": geo, "choice": choice}
        log(f"select {size}: T={T} geo={geo} -> {choice}")
        del runs
        torch.cuda.empty_cache()
    return res


def calibrate(agents, ranges, gemm_cfgs, selection, pool, obs):
    """rep time of every main-run condition; block length n = max(10, ceil(2 s / t_rep)) (pairs: the det t_rep)."""
    res = {}
    for u in E.units(selection):
        for (mode, size, N, prec, impl) in E.unit_conds(u):
            agent = agents[size][0]
            model = agent.model
            wm = E.make_wm(model, prec, ranges[size], impl, gemm_cfgs)
            wm.new_episode(1)
            inp = E.wm_inputs(model, N) if mode != "step" else None
            t0 = time.time()
            runner = build_runner(mode, agent, wm, inp, obs)
            tb = time.time() - t0
            if mode == "step":
                E.install(model, wm)
            try:
                tr = quick_rep(runner, pool)
            finally:
                if mode == "step":
                    E.restore(model)
            res[E.cond_key(mode, size, N, prec, impl)] = {"t_rep_ms": tr, "build_s": tb}
            log(f"calib {E.cond_key(mode, size, N, prec, impl)}: {tr:.3f} ms (build {tb:.1f}s)")
            del runner
        torch.cuda.empty_cache()
    blocks, est = {}, 0.0
    for u in E.units(selection):
        cs = E.unit_conds(u)
        tdet = res[E.cond_key(*cs[0])]["t_rep_ms"]
        n = max(10, math.ceil(2000.0 / tdet))
        for c in cs:
            k = E.cond_key(*c)
            blocks[k] = n
            est += (30 + n) * res[k]["t_rep_ms"] / 1e3 + 0.05
    build = sum(v["build_s"] for k, v in res.items() if k.startswith("graph|"))
    return {"t_rep": res, "n_reps": blocks, "est_round_s": est, "est_main_s_10_rounds": 10 * est + build,
            "graph_build_s": build, "rule": "n = max(10, ceil(2000 ms / t_rep)); pairs use the det t_rep"}


@torch.no_grad()
def rowbench(pool):
    """row-kernel time (do_bench, ms) of the layer-0 style (MODE 0) launch: det, dither reading u, dither reading u and
    drawing the next u (layer 0 of a dither forward), dither reading u and bumping the counter (layer 1)."""
    res = {}
    st = torch.zeros(2, dtype=torch.int64, device="cuda")
    for M, W in ((24, 512), (512, 512), (8192, 512), (512, 4096), (2048, 4096), (8192, 4096)):
        y = torch.randn(M, W, device="cuda")
        acc = torch.randint(-50000, 50000, (M, W), device="cuda", dtype=torch.int32)
        g, b = torch.ones(W, device="cuda"), torch.zeros(W, device="cuda")
        al, c = torch.full((W,), 1e-5, device="cuda"), torch.zeros(W, device="cuda")
        q = torch.empty(M, W, device="cuda", dtype=torch.int8)
        ub = torch.rand(2, W, device="cuda")
        for deq in (False, True):
            xin = acc if deq else y
            variants = {"det": dict(dither=False), "dither": dict(dither=True, u=ub[0]),
                        "dither_gen": dict(dither=True, u=ub[0], ug=ub[1], wg=W, gsalt=1),
                        "dither_bump": dict(dither=True, u=ub[0], bump=True)}
            for name, kw in variants.items():
                kw = dict(kw)
                dith = kw.pop("dither")
                fn = (lambda xin=xin, deq=deq, dith=dith, kw=kw: E.row_launch(
                    xin, al, c, g, b, q, st, M, W, -0.3, 3.0, 0.2, 15.0, 0.0, 0, deq, dith, **kw))
                fn()
                res[f"M{M}|W{W}|deq{int(deq)}|{name}"] = float(triton.testing.do_bench(fn, warmup=10, rep=50))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/results/smoke")
    ap.add_argument("--sections", default="env,tune,checks,select,calib,rowbench")
    ap.add_argument("--sizes", default="5,19,48,317")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sec = set(a.sections.split(","))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    pool = E.EventPool(4096)
    t_all = time.time()
    if "env" in sec:
        dump(f"{a.out}/env.json", env_info())
        log("env", env_info())
    agents, ranges = {}, {}
    for size in E.SIZES:
        agents[size] = E.build_agent(size)
        ranges[size] = E.calib_ranges(agents[size][0].model)
        log(f"model {size}: dims {E.model_dims(size)} ranges {ranges[size]} "
            f"mem {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    dump(f"{a.out}/ranges.json", {str(k): v for k, v in ranges.items()})
    cfg_path = f"{a.out}/gemm_cfg.json"
    if "tune" in sec:
        shapes = sorted({s for size in E.SIZES for s in E.gemm_shapes(size)})
        best, table = E.tune_gemm_a(shapes, log=log)
        intmm = {}
        for (M, N, K) in shapes:
            if N % 8:
                N8 = N + (8 - N % 8)
            else:
                N8 = N
            A = torch.randint(0, 16, (M, K), dtype=torch.int8, device="cuda")
            Wt = torch.randint(-127, 128, (N8, K), dtype=torch.int8, device="cuda")
            intmm[f"{M},{N},{K}"] = float(triton.testing.do_bench(lambda: torch._int_mm(A, Wt.t()), warmup=10, rep=40))
        dump(cfg_path, {f"{M},{N},{K}": v[0] for (M, N, K), v in best.items()})
        dump(f"{a.out}/gemm_tune.json", {"triton_best": {f"{M},{N},{K}": v for (M, N, K), v in best.items()},
                                         "triton_all": table, "int_mm_ms": intmm})
    gemm_cfgs = {}
    if os.path.exists(cfg_path):
        gemm_cfgs = {tuple(int(t) for t in k.split(",")): tuple(v) for k, v in json.load(open(cfg_path)).items()}
    if "checks" in sec:
        allc = {}
        for size in E.SIZES:
            try:
                allc[str(size)] = {"mlp": checks_size(size, agents[size][0], ranges[size], gemm_cfgs),
                                   "rollout": checks_rollout(size, agents[size][0], ranges[size], gemm_cfgs,
                                                             ["a", "b"] + (["e7"] if size == 5 else []))}
            except Exception:  # noqa: BLE001
                allc[str(size)] = {"error": traceback.format_exc()}
                log(traceback.format_exc())
            dump(f"{a.out}/checks.json", allc)
            torch.cuda.empty_cache()
    sel_path = f"{a.out}/selection.json"
    if "select" in sec:
        sel = select_paths(agents, ranges, gemm_cfgs, pool)
        dump(sel_path, {"rule": "per size: GEMM path with the smaller geometric mean over N of T(int-det, graph-WM); "
                                "T = median of 3 block medians (30 warmup + max(10, ceil(1 s / t_rep)) replays)",
                        "sizes": sel, "selection": {k: v["choice"] for k, v in sel.items()}, "time": E.now_utc()})
    if "calib" in sec:
        selection = json.load(open(sel_path))["selection"]
        obs = E.obs_stream()
        cal = calibrate(agents, ranges, gemm_cfgs, selection, pool, obs)
        cal["selection"] = selection
        dump(f"{a.out}/calib.json", cal)
        log(f"calib: est round {cal['est_round_s']:.0f} s, main (10 rounds + graph builds) "
            f"{cal['est_main_s_10_rounds'] / 60:.1f} min")
    if "rowbench" in sec:
        dump(f"{a.out}/rowbench.json", rowbench(pool))
    log(f"SMOKE_DONE {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
