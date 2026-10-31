"""E8c smoke on the pod (before PLAN.md and the main measurement). Sections (each writes its JSON in --out):
  env       torch / triton / GPU / driver versions
  ranges    quantization ranges of every size (e8lib.calib_ranges, model seed 0, input seed 0) compared with E8's
            ranges.json (must be equal within 1e-6 relative, as E8's main asserted against its smoke)
  randperm  torch.randperm (as in WorldModel.Q(return_type='avg')) inside a CUDA graph: capture works, replays draw new
            permutations from the default CUDA generator, frequencies of the first two indices close to uniform
            (chi-square below the p = 0.001 critical value), and WorldModel.Q(..., 'avg') graph == eager under the same
            CUDA seed, replays differ
  checks    functional checks on "check agents": the same models (seed 0) with the zero-initialized reward head and Q
            heads re-randomized (trunc normal 0.02, as E8's reward checks), so that the planner's values depend on the
            WM and Q computations (with the official init every candidate value is 0)
              bf16cast vs bf16 (autocast) vs fp32: single dynamics / reward forwards on 512 random inputs and the WM
              rollout (graph-WM inputs, N 512), graph-WM replay == eager for bf16cast
              int-det-L0tf32 vs int-det: TF32 effective in the layer-0 GEMM, layer-0 / layer-1 code mismatch, int-det
              unchanged by a TF32-on rest (layer 0 pinned), graph-WM replay == eager
              graph-step: for every size at N 512 (and 5M at N 2048 and 8192) all 11 graph-step paths (rest-fp32: fp32,
              tf32, bf16, bf16cast, int-det, int-shiter; rest-tf32: tf32, bf16, bf16cast, int-det, int-shiter)
              captured into one shared pool, then checked in random order:
                eager vs graph   same CUDA seed, _prev_mean, observation and dither state -> same action and final MPPI
                                 mean (|diff| <= 1e-4) and dither counters
                same seed again  (dither counter continues) det / float paths: bitwise equal; int-shiter: different
                                 (new u); counters +20 dynamics / +18 reward per replay
                no reseed        (generator offset continues) -> different (new MPPI / policy / Gumbel / Q-subset draws)
                rest-tf32 vs rest-fp32 of the same WM path under the same state -> different (TF32 in the rest)
  calib     every main-run condition built as in the main run (latency agents = official init; graph-WM graphs in
            private pools, all graph-step graphs in one shared pool, heaviest first) and its call time measured
            (quick_rep: 3 warmup calls, then >= 3 calls and >= 0.3 s); block lengths n = max(10, ceil(2 s / t_rep))
            (pairs: the det t_rep); estimated main-run time
usage: python e8c_smoke.py --e8in <dir with E8's gemm_cfg.json, ranges.json, selection.json> --out <dir>
       [--sections env,ranges,randperm,checks,calib]
"""
import argparse
import json
import os
import platform
import subprocess
import time
import traceback
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import triton

import e8lib as E
import e8clib as C

TOL_PLAN = 1e-4          # max |eager - graph| of the action and the final MPPI mean (same random inputs)
TOL_GRAPH = 1e-6         # max |graph - eager| of a WM rollout
CHI2_P001 = {19: 43.82, 55: 93.17}     # chi-square critical values at p = 0.001


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


def ranges_equal(r, ref):
    return all(abs(r[k][i] - ref[k][i]) <= 1e-6 * max(1.0, abs(ref[k][i])) for k in ref for i in (0, 1))


# ---------------------------------------------------------------------------------------------------- randperm
@torch.no_grad()
def check_randperm(agents, reps=2000):
    res = {}
    for nq in (5, 8):
        r = {"num_q": nq, "reps": reps}
        try:
            gr = E.GraphRunner(lambda nq=nq: torch.randperm(nq, device="cuda")[:2], warmup=1)
            seen = []
            for _ in range(reps):
                gr()
                seen.append(tuple(int(v) for v in gr.out.tolist()))
            cnt = Counter(seen)
            k = nq * (nq - 1)
            exp = reps / k
            chi2 = sum((cnt.get((i, j), 0) - exp) ** 2 / exp for i in range(nq) for j in range(nq) if i != j)
            changed = float(np.mean([seen[i] != seen[i - 1] for i in range(1, reps)]))
            r.update({"captured": True, "distinct_pairs": len(cnt), "expected_distinct": k, "chi2": chi2,
                      "dof": k - 1, "chi2_crit_p001": CHI2_P001[k - 1], "share_consecutive_changed": changed,
                      "pass": bool(len(cnt) == k and chi2 < CHI2_P001[k - 1] and changed > 0.5)})
            del gr
        except Exception:  # noqa: BLE001
            r.update({"captured": False, "error": traceback.format_exc()[-3000:], "pass": False})
        res[f"randperm_{nq}"] = r
        log(f"randperm {nq}: {r}")
    for size, agent in agents.items():
        model = agent.model
        r = {}
        try:
            g = torch.Generator(device="cuda")
            g.manual_seed(5)
            z = model.encode(torch.randn(512, E.OBS_DIM, device="cuda", generator=g), None)
            a = torch.rand(512, E.ACT_DIM, device="cuda", generator=g) * 2 - 1
            fq = (lambda: model.Q(z, a, None, return_type="avg"))
            gr = E.GraphRunner(fq, warmup=1)
            torch.cuda.manual_seed(123)
            qe = fq().clone()
            torch.cuda.manual_seed(123)
            gr()
            q1 = gr.out.clone()
            gr()
            q2 = gr.out.clone()
            r = {"captured": True, "graph_vs_eager_maxabs": C.maxabs(qe, q1), "replays_differ": bool((q1 != q2).any()),
                 "q_std": float(q1.std())}
            r["pass"] = bool(r["graph_vs_eager_maxabs"] <= 1e-5 and r["replays_differ"])
            del gr
        except Exception:  # noqa: BLE001
            r = {"captured": False, "error": traceback.format_exc()[-3000:], "pass": False}
        res[f"Q_avg_{size}"] = r
        log(f"Q avg {size}: {r}")
    return res


# ---------------------------------------------------------------------------------------------------- check agents
def check_agent(size):
    """the latency model (seed 0) with the zero-initialized reward head and Q heads re-randomized (trunc normal 0.02,
    E8's randomized_reward rule), so that candidate values depend on the WM and Q computations."""
    agent, _ = E.build_agent(size)
    m = agent.model
    g = torch.Generator(device="cuda")
    g.manual_seed(1)
    with torch.no_grad():
        for w in (m._reward[2].weight, m._Qs.params["2", "weight"]):
            w.copy_((torch.randn(w.shape, generator=g, device=w.device) * 0.02).clamp(-0.04, 0.04))
    return agent


def check_inputs(model, size, n=512):
    g = torch.Generator(device="cuda")
    g.manual_seed(100 + size)
    obs = torch.randn(n, E.OBS_DIM, device="cuda", generator=g)
    a = torch.rand(n, E.ACT_DIM, device="cuda", generator=g) * 2 - 1
    return model.encode(obs, None), a


@torch.no_grad()
def check_bf16cast(size, agent):
    model = agent.model
    z, a = check_inputs(model, size)
    w32, wbf, wc = E.make_wm(model, "fp32"), E.make_wm(model, "bf16"), C.WMbf16cast(model)
    dts = sorted({str(p.dtype) for p in list(wc.dyn.parameters()) + list(wc.rew.parameters())})
    r = {"param_dtypes": dts}
    ok = dts == ["torch.bfloat16"]

    def within(c, b):
        return c <= 3.0 * b + 1e-3

    for name in ("next", "reward"):
        o32, obf, oc = getattr(w32, name)(z, a), getattr(wbf, name)(z, a), getattr(wc, name)(z, a)
        d = {"out_dtype": str(oc.dtype), "rel_bf16_vs_fp32": C.rel_l2(obf, o32), "rel_cast_vs_fp32": C.rel_l2(oc, o32),
             "rel_cast_vs_bf16": C.rel_l2(oc, obf), "maxabs_cast_vs_fp32": C.maxabs(oc, o32),
             "finite": bool(torch.isfinite(oc).all())}
        d["pass"] = bool(d["out_dtype"] == "torch.float32" and d["finite"]
                         and within(d["rel_cast_vs_fp32"], d["rel_bf16_vs_fp32"])
                         and within(d["rel_cast_vs_bf16"], d["rel_bf16_vs_fp32"]))
        ok = ok and d["pass"]
        r[name] = d
    inp = E.wm_inputs(model, 512)
    outs = {}
    for nm, wm in (("fp32", w32), ("bf16", wbf), ("bf16cast", wc)):
        zz, rr = E.wm_rollout(wm, *inp)
        outs[nm] = (zz.clone(), rr.clone())
    d = {}
    for j, part in ((0, "z"), (1, "r")):
        d[f"rel_bf16_vs_fp32_{part}"] = C.rel_l2(outs["bf16"][j], outs["fp32"][j])
        d[f"rel_cast_vs_fp32_{part}"] = C.rel_l2(outs["bf16cast"][j], outs["fp32"][j])
        d[f"rel_cast_vs_bf16_{part}"] = C.rel_l2(outs["bf16cast"][j], outs["bf16"][j])
    gr = E.GraphRunner(lambda: E.wm_rollout(wc, *inp), warmup=1)
    gr()
    d["graph_vs_eager_maxabs_z"] = C.maxabs(gr.out[0], outs["bf16cast"][0])
    d["graph_vs_eager_maxabs_r"] = C.maxabs(gr.out[1], outs["bf16cast"][1])
    del gr
    d["pass"] = bool(all(within(d[f"rel_cast_vs_fp32_{p}"], d[f"rel_bf16_vs_fp32_{p}"])
                         and within(d[f"rel_cast_vs_bf16_{p}"], d[f"rel_bf16_vs_fp32_{p}"]) for p in ("z", "r"))
                     and d["graph_vs_eager_maxabs_z"] <= TOL_GRAPH and d["graph_vs_eager_maxabs_r"] <= TOL_GRAPH)
    r["rollout"] = d
    r["pass"] = bool(ok and d["pass"])
    return r


@torch.no_grad()
def check_l0tf32(size, agent, qr, gemm_cfgs, impl):
    model = agent.model
    z, a = check_inputs(model, size)
    x = torch.cat([z, a], -1).contiguous()
    wd = C.WMintX(model, qr, False, impl, gemm_cfgs, l0_tf32=False)
    wl = C.WMintX(model, qr, False, impl, gemm_cfgs, l0_tf32=True)
    r, ok = {}, True
    for name, md, ml in (("dyn", wd.dyn, wl.dyn), ("rew", wd.rew, wl.rew)):
        with C.tf32_mode(False):
            q0d, q1d, od = [t.clone() for t in md._forward(x)]
        with C.tf32_mode(True):
            q0l, q1l, ol = [t.clone() for t in ml._forward(x)]
        L0 = md.seq[0]
        with C.tf32_mode(False):
            y32 = F.linear(x, L0.weight, L0.bias)
        with C.tf32_mode(True):
            ytf = F.linear(x, L0.weight, L0.bias)
        fn = wd.next if name == "dyn" else wd.reward
        with C.tf32_mode(True):
            o_on = fn(z, a).clone()
        with C.tf32_mode(False):
            o_off = fn(z, a).clone()
        d = {"l0_gemm_rel_tf32_vs_fp32": C.rel_l2(ytf, y32), "q0_mismatch": float((q0d != q0l).float().mean()),
             "q0_maxdiff": float((q0d.float() - q0l.float()).abs().max()),
             "q1_mismatch": float((q1d != q1l).float().mean()), "out_rel_l2": C.rel_l2(ol, od),
             "int_det_global_tf32_on_vs_off_maxabs": C.maxabs(o_on, o_off)}
        d["pass"] = bool(d["l0_gemm_rel_tf32_vs_fp32"] > 1e-7 and d["q0_mismatch"] <= 0.05 and d["q0_maxdiff"] <= 1
                         and d["int_det_global_tf32_on_vs_off_maxabs"] == 0.0)
        ok = ok and d["pass"]
        r[name] = d
    inp = E.wm_inputs(model, 512)
    ze, re_ = [t.clone() for t in E.wm_rollout(wl, *inp)]
    gr = E.GraphRunner(lambda: E.wm_rollout(wl, *inp), warmup=1)
    gr()
    r["graph_vs_eager_maxabs_z"] = C.maxabs(gr.out[0], ze)
    r["graph_vs_eager_maxabs_r"] = C.maxabs(gr.out[1], re_)
    del gr
    r["pass"] = bool(ok and r["graph_vs_eager_maxabs_z"] <= TOL_GRAPH and r["graph_vs_eager_maxabs_r"] <= TOL_GRAPH)
    return r


def dither_counters(wm):
    if isinstance(wm, E.WMint) and wm.dyn.dither:
        return [int(wm.dyn.state[1]), int(wm.rew.state[1])]
    return [0, 0]


@torch.no_grad()
def check_gstep(size, agent, qr, gemm_cfgs, impl, Ns, obs, rng):
    """graph-step functional checks (see module docstring)."""
    out = {}
    H, A = agent.cfg.horizon, agent.cfg.action_dim
    gp = torch.Generator(device="cuda")
    gp.manual_seed(7)
    pm0 = (torch.randn(H, A, device="cuda", generator=gp) * 0.3).clamp(-1, 1)
    obs0 = obs[0].reshape(1, -1).cuda()
    for N in Ns:
        pool = torch.cuda.graph_pool_handle()
        items = []
        t0 = time.time()
        for rest in ("fp32", "tf32"):
            for prec in C.GSTEP_SINGLES[rest] + ("int-det", "int-shiter"):
                wm = C.make_wm_c(agent.model, prec, qr, impl if prec.startswith("int") else "-", gemm_cfgs)
                wm.new_episode(1)
                items.append((rest, prec, wm, C.GraphStep(agent, wm, N, rest == "tf32", obs, pool=pool)))
        t_build = time.time() - t0

        def prep(wm, reset_dither=True, reseed=True):
            agent._prev_mean.copy_(pm0)
            if reset_dither:
                wm.new_episode(4242)
            if reseed:
                torch.cuda.manual_seed(1234)

        res, r1 = {}, {}
        for j in rng.permutation(len(items)):
            rest, prec, wm, gs = items[j]
            prep(wm)
            ae = C.plan_eager(agent, wm, N, rest == "tf32", obs0).clone()
            me, ce = agent._prev_mean.clone(), dither_counters(wm)
            prep(wm)
            gs.static_obs.copy_(obs0)
            gs.replay()
            a1, m1, c1 = gs.out.clone(), agent._prev_mean.clone(), dither_counters(wm)
            prep(wm, reset_dither=False)
            gs.replay()
            a2, m2, c2 = gs.out.clone(), agent._prev_mean.clone(), dither_counters(wm)
            prep(wm, reset_dither=False, reseed=False)
            gs.replay()
            a3, m3, c3 = gs.out.clone(), agent._prev_mean.clone(), dither_counters(wm)
            shiter = prec == "int-shiter"
            d = {"eager_vs_graph": {"action_maxabs": C.maxabs(ae, a1), "mean_maxabs": C.maxabs(me, m1),
                                    "bitwise": bool(torch.equal(ae, a1) and torch.equal(me, m1)),
                                    "counters_eager": ce, "counters_graph": c1},
                 "same_seed_replay": {"action_maxabs": C.maxabs(a1, a2), "mean_maxabs": C.maxabs(m1, m2),
                                      "counters": c2},
                 "new_rng_replay": {"action_maxabs": C.maxabs(a2, a3), "mean_maxabs": C.maxabs(m2, m3),
                                    "counters": c3},
                 "action": a1.tolist(), "mean_row0": m1[0].tolist()}
            p = (d["eager_vs_graph"]["action_maxabs"] <= TOL_PLAN and d["eager_vs_graph"]["mean_maxabs"] <= TOL_PLAN
                 and ce == c1 and d["new_rng_replay"]["mean_maxabs"] > 0)
            if shiter:
                p = p and c1 == [20, 18] and c2 == [40, 36] and c3 == [60, 54] and d["same_seed_replay"]["mean_maxabs"] > 0
            else:
                p = p and c1 == c2 == c3 == [0, 0] and d["same_seed_replay"]["mean_maxabs"] == 0.0 \
                    and d["same_seed_replay"]["action_maxabs"] == 0.0
            d["pass"] = bool(p)
            res[f"{rest}|{prec}"] = d
            r1[(rest, prec)] = m1
            log(f"gstep check {size} N{N} {rest}|{prec}: pass={d['pass']} eager-graph "
                f"{d['eager_vs_graph']['action_maxabs']:.2e}/{d['eager_vs_graph']['mean_maxabs']:.2e} "
                f"same-seed {d['same_seed_replay']['mean_maxabs']:.2e} new-rng {d['new_rng_replay']['mean_maxabs']:.2e} "
                f"counters {c1} {c2} {c3}")
        eff = {p: C.maxabs(r1[("fp32", p)], r1[("tf32", p)]) for p in ("tf32", "bf16", "bf16cast", "int-det",
                                                                       "int-shiter")}
        res["rest_tf32_vs_fp32_mean_maxabs"] = eff
        res["rest_tf32_effective"] = bool(all(v > 0 for v in eff.values()))
        res["build_s"] = t_build
        res["pass"] = bool(all(v["pass"] for k, v in res.items() if isinstance(v, dict) and "pass" in v)
                           and res["rest_tf32_effective"])
        out[str(N)] = res
        del items
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------------------------------- calib
def quick_rep(fn, pool, min_s=0.3, min_n=3, max_n=200, warm=3):
    """E8's calibration timing: warm calls, then single timed calls until >= min_n calls and >= min_s seconds."""
    for _ in range(warm):
        fn()
        torch.cuda.synchronize()
    ms = []
    while len(ms) < min_n or (sum(ms) < 1e3 * min_s and len(ms) < max_n):
        x, _ = E.time_block(fn, 0, 1, pool)
        ms += x
    return float(np.median(ms))


@torch.no_grad()
def calibrate(agents, ranges, gemm_cfgs, selection, obs, pool):
    units = C.units_c(selection)
    t0 = time.time()
    built = C.build_all(agents, ranges, gemm_cfgs, units, obs, log=log)
    t_build_all = time.time() - t0
    res = {}
    for c in built["conds"]:
        mode, rest, size, N, prec, impl = c
        key = C.cond_key(*c)
        wm = built["wms"][(size, prec, impl)]
        wm.new_episode(1)
        fn, inst = C.runner_for(built, agents, c)
        if inst is not None:
            E.install(agents[size].model, inst)
        try:
            tr = quick_rep(fn, pool)
        finally:
            if inst is not None:
                E.restore(agents[size].model)
        assert torch.backends.cuda.matmul.allow_tf32 is False
        res[key] = {"t_rep_ms": tr, "build_s": built["build_s"].get(key, 0.0)}
        log(f"calib {key}: {tr:.3f} ms")
    blocks, est, per_mode = {}, 0.0, {}
    for u in units:
        cs = C.unit_conds(u)
        n = C.block_len(res[C.cond_key(*cs[0])]["t_rep_ms"])
        for c in cs:
            k = C.cond_key(*c)
            blocks[k] = n
            t = (30 + n) * res[k]["t_rep_ms"] / 1e3 + 0.05
            est += t
            mk = f"{c[0]}|{c[1]}"
            per_mode[mk] = per_mode.get(mk, 0.0) + t
    out = {"t_rep": res, "n_reps": blocks, "est_round_s": est, "est_round_s_by_mode": per_mode,
           "build_all_s": t_build_all, "t_graph_build_s": built["t_graph_s"], "t_gstep_build_s": built["t_gstep_s"],
           "est_main_s_10_rounds": 10 * est + t_build_all + 60.0, "mem_after_build": built["mem"],
           "n_units_per_round": len(units), "n_blocks_per_round": len(built["conds"]),
           "rule": "n = max(10, ceil(2000 ms / t_rep)); pairs use the det t_rep; est = sum (30 + n) t_rep + 0.05 s"}
    del built
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e8in", default="/workspace/e8c/e8_inputs")
    ap.add_argument("--out", default="/workspace/results/smoke")
    ap.add_argument("--sections", default="env,ranges,randperm,checks,calib")
    ap.add_argument("--check-sizes", default="5,19,48,317")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sec = set(a.sections.split(","))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    pool = E.EventPool(4096)
    t_all = time.time()
    gemm_cfgs = {tuple(int(t) for t in k.split(",")): tuple(v)
                 for k, v in json.load(open(f"{a.e8in}/gemm_cfg.json")).items()}
    selection = json.load(open(f"{a.e8in}/selection.json"))["selection"]
    e8_ranges = json.load(open(f"{a.e8in}/ranges.json"))
    if "env" in sec:
        dump(f"{a.out}/env.json", env_info())
        log("env", env_info())
    agents, ranges = {}, {}
    for size in E.SIZES:
        agents[size] = E.build_agent(size)[0]
        ranges[size] = E.calib_ranges(agents[size].model)
        log(f"model {size}: dims {E.model_dims(size)} ranges {ranges[size]} "
            f"mem {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    if "ranges" in sec:
        cmp = {str(s): ranges_equal(ranges[s], {k: tuple(v) for k, v in e8_ranges[str(s)].items()}) for s in E.SIZES}
        dump(f"{a.out}/ranges.json", {str(k): v for k, v in ranges.items()})
        dump(f"{a.out}/ranges_vs_e8.json", {"equal_to_e8_within_1e-6": cmp, "pass": all(cmp.values())})
        log(f"ranges equal to E8: {cmp}")
        assert all(cmp.values()), cmp
    if "randperm" in sec:
        cas = {s: check_agent(s) for s in (5, 317)}      # Q heads re-randomized (official init: every Q value is 0)
        rp = check_randperm(cas)
        rp["pass"] = bool(all(v["pass"] for v in rp.values()))
        dump(f"{a.out}/randperm.json", rp)
        log(f"randperm / Q avg pass: {rp['pass']}")
        del cas
        torch.cuda.empty_cache()
    if "checks" in sec:
        obs = E.obs_stream()
        rng = np.random.default_rng(3)
        allc = {}
        for size in [int(s) for s in a.check_sizes.split(",")]:
            try:
                ca = check_agent(size)
                impl = selection[str(size)]
                r = {"bf16cast": check_bf16cast(size, ca)}
                log(f"bf16cast check {size}: {json.dumps(r['bf16cast'])[:600]}")
                r["l0tf32"] = check_l0tf32(size, ca, ranges[size], gemm_cfgs, impl)
                log(f"L0tf32 check {size}: {json.dumps(r['l0tf32'])[:600]}")
                Ns = (512, 2048, 8192) if size == 5 else (512,)
                r["gstep"] = check_gstep(size, ca, ranges[size], gemm_cfgs, impl, Ns, obs, rng)
                r["pass"] = bool(r["bf16cast"]["pass"] and r["l0tf32"]["pass"]
                                 and all(v["pass"] for v in r["gstep"].values()))
                allc[str(size)] = r
                del ca
            except Exception:  # noqa: BLE001
                allc[str(size)] = {"error": traceback.format_exc(), "pass": False}
                log(traceback.format_exc())
            dump(f"{a.out}/checks.json", allc)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        log(f"checks pass: { {k: v.get('pass') for k, v in allc.items()} }")
    if "calib" in sec:
        obs = E.obs_stream()
        cal = calibrate(agents, ranges, gemm_cfgs, selection, obs, pool)
        cal["selection"] = selection
        dump(f"{a.out}/calib.json", cal)
        log(f"calib: est round {cal['est_round_s']:.0f} s {cal['est_round_s_by_mode']}, main (10 rounds + builds) "
            f"{cal['est_main_s_10_rounds'] / 60:.1f} min")
    log(f"SMOKE_DONE {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
