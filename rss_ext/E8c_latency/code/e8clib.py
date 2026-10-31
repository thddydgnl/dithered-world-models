"""E8c (rss_ext/DESIGN_E8c.md): additions to E8's e8lib for the extra latency measurement.

e8lib.py is E8's file, unchanged (md5 7dd7b30d...). Everything of the integer path (Triton int8 tile GEMM 'a' with E8's
per-shape tile configs, row kernels, device-counter dither, W8 RTN weights, 4-bit codes), the fp32 / tf32 / bf16
(autocast) WM paths, the model dims, seeds, rollout inputs, the observation stream, the timing helpers and the range
estimation come from it.

World-model (WM = dynamics + reward MLP) precision paths
  fp32, tf32, bf16    e8lib.WMfp32 / WMtf32 / WMbf16 (E8 PLAN 3: tf32 = TF32 allowed inside the WM calls only;
                      bf16 = torch.autocast(bfloat16, cache_enabled=False): the weights are cast at every call)
  bf16cast            (new) copies of the dynamics and reward MLPs with every parameter (weights, biases, LayerNorm
                      affine) cast to bf16 once, before the measurement; the input [z, a] is cast to bf16 and the outputs
                      (next latent, reward logits) back to fp32; no autocast
  int-det, int-shiter E8's integer path (e8lib.WMint on GEMM path 'a') with its layer-0 fp32 GEMM pinned to TF32 off
                      inside the MLP call. This is E8's definition (layer 0 = fp32 GEMM); the pin only matters when the
                      rest of the planner runs with TF32 allowed (rest-tf32).
  int-det-L0tf32      (new, descriptive, graph-WM only) int-det with TF32 allowed for the layer-0 GEMM only
  The only cuBLAS call inside an integer MLP forward is the layer-0 F.linear (the int8 GEMMs of path 'a' are Triton
  kernels), so setting the TF32 flag around the whole MLP call sets it for layer 0 only.
Rest of the planner (encoder, policy prior, Q-ensemble, MPPI arithmetic)
  rest-fp32  TF32 off (E7, E8 eager-step)
  rest-tf32  (graph-step only, descriptive) TF32 allowed for every matmul outside the WM calls; the WM paths keep their
             own setting (tf32: all TF32; bf16 / bf16cast: bf16 GEMMs; int-*: layer 0 with TF32 off)
Modes
  graph  graph-WM (E8): the WM rollout of one control step as one CUDA graph (e8lib.GraphRunner, private pools)
  gstep  graph-step: one CUDA graph of TDMPC2._plan(obs, t0=False, eval_mode=True) at N candidates (GraphStep)
  step   eager-step (E8): TDMPC2.act(obs, t0=(first call of the block), eval_mode=True) at N 512 (e8lib.StepRunner)
"""
import copy
import math
import time

import numpy as np
import torch

import e8lib as E

STEP_N = 512                                        # eager-step candidates (E8)
GWM_SINGLES = ("fp32", "tf32", "bf16", "bf16cast", "int-det-L0tf32")
GSTEP_SINGLES = {"fp32": ("fp32", "tf32", "bf16", "bf16cast"), "tf32": ("tf32", "bf16", "bf16cast")}
ESTEP_SINGLES = ("fp32", "tf32", "bf16", "bf16cast")
FLOATS_MIN = ("tf32", "bf16", "bf16cast")           # min(float) = min(T(tf32), T(bf16), T(bf16cast))


# ---------------------------------------------------------------------------------------------------- WM paths
class tf32_mode:
    """context: torch.backends.cuda.matmul.allow_tf32 = flag inside, the previous value restored on exit."""

    def __init__(self, flag):
        self.flag = bool(flag)

    def __enter__(self):
        self.prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = self.flag
        return self

    def __exit__(self, *exc):
        torch.backends.cuda.matmul.allow_tf32 = self.prev
        return False


class WMbf16cast(E.WMfp32):
    """pre-cast BF16: bf16 copies of the dynamics / reward MLPs made once; bf16 input, fp32 outputs, no autocast."""
    name = "bf16cast"

    def __init__(self, model):
        super().__init__(model)
        self.dyn = copy.deepcopy(model._dynamics).to(torch.bfloat16).eval()
        self.rew = copy.deepcopy(model._reward).to(torch.bfloat16).eval()

    def next(self, z, a):
        return self.dyn(torch.cat([z, a], dim=-1).to(torch.bfloat16)).float()

    def reward(self, z, a):
        return self.rew(torch.cat([z, a], dim=-1).to(torch.bfloat16)).float()


class WMintX(E.WMint):
    """E8's integer WM path with the TF32 flag of its layer-0 fp32 GEMM pinned (False: int-det / int-shiter as defined
    in E8; True: int-det-L0tf32)."""

    def __init__(self, model, qr, dither, impl, gemm_cfgs, l0_tf32=False):
        super().__init__(model, qr, dither, impl, gemm_cfgs)
        assert impl in ("a", "b"), impl
        self.l0_tf32 = bool(l0_tf32)
        if self.l0_tf32:
            self.name = self.name + "-L0tf32"

    def next(self, z, a):
        with tf32_mode(self.l0_tf32):
            return self.dyn(torch.cat([z, a], dim=-1))

    def reward(self, z, a):
        with tf32_mode(self.l0_tf32):
            return self.rew(torch.cat([z, a], dim=-1))


def make_wm_c(model, prec, qr=None, impl=None, gemm_cfgs=None):
    if prec in ("fp32", "tf32", "bf16"):
        return E.make_wm(model, prec)
    if prec == "bf16cast":
        return WMbf16cast(model)
    if prec in ("int-det", "int-shiter"):
        return WMintX(model, qr, prec == "int-shiter", impl, gemm_cfgs, l0_tf32=False)
    if prec == "int-det-L0tf32":
        return WMintX(model, qr, False, impl, gemm_cfgs, l0_tf32=True)
    raise ValueError(prec)


# ---------------------------------------------------------------------------------------------------- planning step
class plan_setup:
    """context of one planning call or capture: the WM path installed on the agent's model (e8lib.install), the agent's
    cfg.num_samples = N, the TF32 flag of the rest = rest_tf32; all restored on exit (outside, cfg.num_samples stays
    512 for eager-step and TF32 stays off)."""

    def __init__(self, agent, wm, N, rest_tf32):
        self.agent, self.wm, self.N, self.rest_tf32 = agent, wm, int(N), bool(rest_tf32)

    def __enter__(self):
        self.n0 = self.agent.cfg.num_samples
        self.tf0 = torch.backends.cuda.matmul.allow_tf32
        E.install(self.agent.model, self.wm)
        self.agent.cfg.num_samples = self.N
        torch.backends.cuda.matmul.allow_tf32 = self.rest_tf32
        return self

    def __exit__(self, *exc):
        torch.backends.cuda.matmul.allow_tf32 = self.tf0
        self.agent.cfg.num_samples = self.n0
        E.restore(self.agent.model)
        return False


@torch.no_grad()
def plan_eager(agent, wm, N, rest_tf32, obs_dev):
    """one eager TDMPC2._plan(obs, t0=False, eval_mode=True) with the same settings as the graph-step capture."""
    with plan_setup(agent, wm, N, rest_tf32):
        return agent._plan(obs_dev, t0=False, eval_mode=True)


_WARM_STREAM = None


def warm_stream():
    """one side stream for every graph-step warmup (its cached blocks are reused by the next warmup)."""
    global _WARM_STREAM
    if _WARM_STREAM is None:
        _WARM_STREAM = torch.cuda.Stream()
    return _WARM_STREAM


class GraphStep:
    """graph-step: TDMPC2._plan(static_obs, t0=False, eval_mode=True) at N candidates, WM path wm, rest TF32 flag
    rest_tf32, captured as one CUDA graph after `warmup` eager calls on a side stream (as e8lib.GraphRunner).
    A timed call = copy the next observation of the fixed host stream (E8's 200 N(0, 1) observations, pageable host
    memory, non_blocking=True as in TDMPC2.act) into the static obs buffer, then replay.
    MPPI samples, policy noise, Gumbel selection and the Q-subset draw use the default CUDA generator (graph-safe: every
    replay continues its offset, so every replay draws new numbers); the dither counter of the int MLPs is a device
    tensor (E8), so every replay draws new u vectors (dynamics +20, reward +18 forwards per replay).
    Memory: all graph-step graphs may share one CUDA-graph pool (pool=torch.cuda.graph_pool_handle()). A replay only
    reads tensors outside the pool (weights, static obs, _prev_mean, dither state and u buffers) or tensors written
    earlier in the same replay, so it computes the same values whatever graph ran before it; only a graph's static
    output may be overwritten by later replays of other graphs (the timing blocks never read it; the smoke checks read
    it right after the replay)."""

    def __init__(self, agent, wm, N, rest_tf32, obs, pool=None, warmup=3):
        self.agent, self.wm, self.N, self.rest_tf32 = agent, wm, int(N), bool(rest_tf32)
        self.obs = [o.reshape(1, -1).contiguous() for o in obs]
        self.s = 0
        self.static_obs = torch.zeros(1, E.OBS_DIM, device="cuda")
        self.static_obs.copy_(self.obs[0])
        sobs = self.static_obs

        def fn():
            return agent._plan(sobs, t0=False, eval_mode=True)

        with torch.no_grad(), plan_setup(agent, wm, N, rest_tf32):
            s = warm_stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(warmup):
                    fn()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            self.g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g, pool=pool):
                self.out = fn()
            torch.cuda.synchronize()

    def reset(self):
        self.s = 0

    def replay(self):
        self.g.replay()

    def __call__(self):
        self.static_obs.copy_(self.obs[self.s % len(self.obs)], non_blocking=True)
        self.g.replay()
        self.s += 1


# ---------------------------------------------------------------------------------------------------- units
def cond_key(mode, rest, size, N, prec, impl):
    return f"{mode}|{rest}|{size}|{N}|{prec}|{impl}"


def units_c(selection, sizes=E.SIZES, ns=E.NS):
    """measurement units of one round: single blocks and int det / shiter pairs.
    per (size, N): graph-WM fp32, tf32, bf16, bf16cast, int-det-L0tf32 + int pair (7 blocks);
                   graph-step rest-fp32 fp32, tf32, bf16, bf16cast + int pair (6 blocks);
                   graph-step rest-tf32 tf32, bf16, bf16cast + int pair (5 blocks)
    per size:      eager-step (N 512, rest-fp32) fp32, tf32, bf16, bf16cast + int pair (6 blocks)
    -> 200 units, 240 blocks per round for 4 sizes x 3 N."""
    out = []
    for size in sizes:
        impl = selection[str(size)]
        for N in ns:
            for p in GWM_SINGLES:
                out.append({"kind": "single", "mode": "graph", "rest": "-", "size": size, "N": N, "prec": p,
                            "impl": impl if p.startswith("int") else "-"})
            out.append({"kind": "pair", "mode": "graph", "rest": "-", "size": size, "N": N, "prec": "int",
                        "impl": impl})
            for rest in ("fp32", "tf32"):
                for p in GSTEP_SINGLES[rest]:
                    out.append({"kind": "single", "mode": "gstep", "rest": rest, "size": size, "N": N, "prec": p,
                                "impl": "-"})
                out.append({"kind": "pair", "mode": "gstep", "rest": rest, "size": size, "N": N, "prec": "int",
                            "impl": impl})
        for p in ESTEP_SINGLES:
            out.append({"kind": "single", "mode": "step", "rest": "fp32", "size": size, "N": STEP_N, "prec": p,
                        "impl": "-"})
        out.append({"kind": "pair", "mode": "step", "rest": "fp32", "size": size, "N": STEP_N, "prec": "int",
                    "impl": impl})
    return out


def unit_conds(u):
    if u["kind"] == "single":
        return [(u["mode"], u["rest"], u["size"], u["N"], u["prec"], u["impl"])]
    return [(u["mode"], u["rest"], u["size"], u["N"], f"int-{v}", u["impl"]) for v in ("det", "shiter")]


def work_order(conds):
    """build order of the graphs: heaviest first (317M ... 5M, N 8192 ... 512), so that the large CUDA-graph pool
    segments are made first and reused by the smaller captures."""
    return sorted(conds, key=lambda c: (-c[2], -c[3]))


def build_all(agents, ranges, gemm_cfgs, units, obs, log=print):
    """WM paths, graph-WM graphs (E8 GraphRunner, private pools), graph-step graphs (GraphStep, one shared pool) and
    eager-step runners for every condition of `units`. Returns dict with wms, inputs, graphs, gsteps, steps, times."""
    conds = [c for u in units for c in unit_conds(u)]
    wms, inputs, graphs, gsteps, times = {}, {}, {}, {}, {}
    for (mode, rest, size, N, prec, impl) in conds:
        if (size, prec, impl) not in wms:
            wms[(size, prec, impl)] = make_wm_c(agents[size].model, prec, ranges[size], impl, gemm_cfgs)
        if mode == "graph" and (size, N) not in inputs:
            inputs[(size, N)] = E.wm_inputs(agents[size].model, N)
    t0 = time.time()
    for c in work_order([c for c in conds if c[0] == "graph"]):
        mode, rest, size, N, prec, impl = c
        wm, inp = wms[(size, prec, impl)], inputs[(size, N)]
        wm.new_episode(1)
        tb = time.time()
        graphs[cond_key(*c)] = E.GraphRunner(lambda w=wm, i=inp: E.wm_rollout(w, *i))
        times[cond_key(*c)] = time.time() - tb
    torch.cuda.synchronize()
    t_graph = time.time() - t0
    torch.cuda.empty_cache()
    log(f"built {len(graphs)} graph-WM graphs in {t_graph:.0f}s; allocated {torch.cuda.memory_allocated() / 2**30:.1f}"
        f" GiB, reserved {torch.cuda.memory_reserved() / 2**30:.1f} GiB")
    pool = torch.cuda.graph_pool_handle()
    t0 = time.time()
    for c in work_order([c for c in conds if c[0] == "gstep"]):
        mode, rest, size, N, prec, impl = c
        wm = wms[(size, prec, impl)]
        wm.new_episode(1)
        tb = time.time()
        gsteps[cond_key(*c)] = GraphStep(agents[size], wm, N, rest == "tf32", obs, pool=pool)
        times[cond_key(*c)] = time.time() - tb
    torch.cuda.synchronize()
    t_gstep = time.time() - t0
    torch.cuda.empty_cache()
    log(f"built {len(gsteps)} graph-step graphs in {t_gstep:.0f}s (shared pool); allocated "
        f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB, reserved {torch.cuda.memory_reserved() / 2**30:.1f} GiB")
    steps = {size: E.StepRunner(agents[size], obs) for size in {c[2] for c in conds if c[0] == "step"}}
    return {"wms": wms, "inputs": inputs, "graphs": graphs, "gsteps": gsteps, "steps": steps, "build_s": times,
            "t_graph_s": t_graph, "t_gstep_s": t_gstep, "conds": conds,
            "mem": {"allocated_gib": torch.cuda.memory_allocated() / 2**30,
                    "reserved_gib": torch.cuda.memory_reserved() / 2**30,
                    "max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}}


def runner_for(built, agents, c):
    """(fn, install_wm) for one condition: fn() is one timed call; install_wm is the WM to install for eager-step."""
    mode, rest, size, N, prec, impl = c
    key = cond_key(*c)
    if mode == "graph":
        return built["graphs"][key], None
    if mode == "gstep":
        g = built["gsteps"][key]
        g.reset()
        return g, None
    assert mode == "step" and N == STEP_N and rest == "fp32"
    assert agents[size].cfg.num_samples == STEP_N
    st = built["steps"][size]
    st.reset()
    return st, built["wms"][(size, prec, impl)]


def block_len(t_rep_ms):
    """E8 block rule: n = max(10, ceil(2000 ms / t_rep))."""
    return max(10, math.ceil(2000.0 / t_rep_ms))


def rel_l2(x, ref):
    x, ref = x.float(), ref.float()
    return float((x - ref).norm() / ref.norm().clamp(min=1e-12))


def maxabs(x, y):
    return float((x.float() - y.float()).abs().max())


def np_float(x):
    return float(np.asarray(x))
