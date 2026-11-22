"""E17 pre-run checks of the closed-loop kernel paths (tmlr_ext/DESIGN_tmlr.md 0.7 checks 1 and 2), single process on
the pod, smoke seeds 109000+ (never analysed). Sections (each writes <out>/<section>.json and prints PASS / FAIL):
  paths    check 1: the untouched E7 kernel path. inputs/e7_reference/e7lib.py (main/code/tdmpc2/e7lib.py before any E17
           change) and this folder's e7lib.py on the same pod, same episodes: E7's own precisions int8 / int4 det and
           sh-iter (mw-push ckpt 1), and the E17 'intk' det / sh-iter at k = 3 (mw-push) and k = 5 (cheetah-run)
           against the reference IntMLP at the same k; returns and successes must be bitwise equal. Also the v2 kernels
           with det / sh-iter rounding must give E7's codes bitwise (sanity of the shared v2 code).
  shift    check 2 (det+shift): offset draws o ~ U[-1/2, 1/2) (moments, range, histogram), one row shared by all rows
           of a call, new offsets every forward, the planner's global RNG untouched; the integer det+shift forward
           (dynamics and reward MLPs) equals the fp32 fake-quant det+shift with the same W8 weights and offsets
           (teacher forcing per layer: codes, c_eff, head) within float tolerance; the element-wise error law of the
           kernel's dequantized values (mean ~ 0, variance delta^2 / 12, interior entries); closed-loop smoke episodes
           and the restore check (fp32 episode before and after the integer jobs bitwise equal).
  pc       check 2 (per-channel fold-in, needs the pcstats results): the folded integer path equals the fake-quant
           per-channel computation with the same folded W8 weights (teacher forcing: codes and head; end to end:
           relative L2) within tolerance; descriptive: relative L2 against the fake-quant per-channel model with fp32
           weights (E12's configuration) and against fp32; closed-loop smoke episodes.
  bf16p    pre-cast BF16: forward relative L2 against fp32, smoke episodes, restore check.
usage: python e17_checks.py --sections paths,shift,pc,bf16p --out /workspace/atm/out/checks [--res /workspace/atm/out]
"""
import argparse
import gzip
import importlib.util
import json
import os
import sys
import tempfile
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qlib as L  # noqa: E402
import e7lib as E7  # noqa: E402
import runner as R  # noqa: E402

TOL_MISMATCH = 1e-3
TOL_HEAD = 1e-4
SMOKE_SEED = 109000


def log(*a):
    print(time.strftime("%H:%M:%S", time.gmtime()), *a, flush=True)


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def load_ref_e7lib():
    spec = importlib.util.spec_from_file_location("e7lib_ref", f"{HERE}/inputs/e7_reference/e7lib.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@triton.jit
def k_urow(O, seed, N: tl.constexpr):
    offs = tl.arange(0, N)
    tl.store(O + offs, tl.rand(seed, offs))


def urow(seed, n):
    u = torch.empty(n, device="cuda", dtype=torch.float32)
    k_urow[(1,)](u, seed, N=n)
    return u


AGENTS = {}


def agent_of(task, ckpt=1):
    if (task, ckpt) not in AGENTS:
        agent, cfg = L.build_agent(task, ckpt)
        quant = L.Quantizer(agent.model, horizon=int(cfg.horizon))
        AGENTS[(task, ckpt)] = (agent, cfg, quant)
    return AGENTS[(task, ckpt)]


def episode(task, ckpt, seed, install=None, variant="none", k=0):
    """one closed-loop episode (qlib.run_episode, E1 seeding) with an optional precision install(agent, quant)."""
    agent, cfg, quant = agent_of(task, ckpt)
    obj = install(agent, quant) if install is not None else None
    try:
        t0 = time.time()
        r = L.run_episode(agent, cfg, quant, L.canon_variant(variant), int(k), int(seed))
        return {"ret": r["ret"], "success": r["success"], "steps": r["steps"],
                "ms_per_step": (time.time() - t0) / r["steps"] * 1e3}
    finally:
        if obj is not None:
            obj.restore()
        quant.layer_k, quant.wq = {}, None
        quant.off()


_PLAN_IN = {}


@torch.no_grad()
def plan_inputs(task, ckpt=1, n=512, seed=0):
    """planning-time-like MLP inputs [z, a]: encoder latents of the observations of one unquantized smoke episode
    (evenly spaced states) and actions ~ U[-1, 1] (fixed generator); cached per (task, ckpt, n, seed)."""
    key = (task, ckpt, n, seed)
    if key in _PLAN_IN:
        return _PLAN_IN[key]
    agent, cfg, quant = agent_of(task, ckpt)
    r = L.run_episode(agent, cfg, quant, "none", 0, SMOKE_SEED + 99)
    obs = r["_obs"]
    idx = torch.linspace(0, len(obs) - 1, n).round().long()
    z = agent.model.encode(obs[idx].to(L.DEV), None)
    g = torch.Generator(device=L.DEV)
    g.manual_seed(seed)
    a = torch.rand((n, cfg.action_dim), generator=g, device=L.DEV) * 2 - 1
    _PLAN_IN[key] = (z, a)
    return z, a


# ---------------------------------------------------------------------------------------------------- check 1
class RefK:
    """E7's class for the reference lib at any k (the reference E7 class only allows k = 8 / 4): the reference IntMLP
    (unchanged E7 code) installed exactly as E7 installs it."""

    def __init__(self, ref, agent, quant, qr, k, dither):
        self.model, self.quant, self.base = agent.model, quant, 0
        m = self.model
        self.dyn = ref.IntMLP(m._dynamics, ("dyn0", "dyn1"), qr, k, dither, self, 0, True, False)
        self.rew = ref.IntMLP(m._reward, ("rew0", "rew1"), qr, k, dither, self, 4, False, False)
        m.next = lambda z, a, task: self.dyn(torch.cat([z, a], dim=-1))
        m.reward = lambda z, a, task: self.rew(torch.cat([z, a], dim=-1))
        quant.wq = self

    def new_episode(self, seed):
        self.base = int(seed)
        self.dyn.calls = self.rew.calls = 0

    def begin_step(self):
        pass

    def restore(self):
        for name in ("next", "reward"):
            if name in self.model.__dict__:
                delattr(self.model, name)
        if self.quant.wq is self:
            self.quant.wq = None


def sec_paths(a):
    ref = load_ref_e7lib()
    res = {"episodes": {}, "v2_codes": {}}
    scales = tempfile.mkdtemp(prefix="e17scales_")
    os.makedirs(f"{scales}/scales", exist_ok=True)
    for task in ("mw-push", "cheetah-run"):
        qr = R.load_ranges(None, task, 1, "e1")
        json.dump({"quant": {k: list(v) for k, v in qr.items()}}, open(f"{scales}/scales/{task}_s1.json", "w"))
    seed = SMOKE_SEED
    # E7's own precisions, both libs (the reference class reads <out>/scales)
    for prec, k in (("int8", 8), ("int4", 4)):
        for v in ("det", "shiter"):
            spec = {"e7": prec, "variant": v, "k": k, "task": "mw-push", "ckpt": 1}
            er = episode("mw-push", 1, seed, lambda ag, qu: ref.install(ag, qu, scales, spec), v, k)
            en = episode("mw-push", 1, seed, lambda ag, qu: E7.install(ag, qu, scales, spec), v, k)
            same = er["ret"] == en["ret"] and er["success"] == en["success"]
            res["episodes"][f"mw-push|{prec}|{v}"] = {"ref": er, "new": en, "identical": same}
            log(f"paths mw-push {prec} {v}: ref {er['ret']:.4f} new {en['ret']:.4f} identical={same}")
    # E17 intk det / shiter vs the reference IntMLP at the same k
    for task, k in (("mw-push", 3), ("cheetah-run", 5)):
        qr = R.load_ranges(None, task, 1, "e1")
        for v in ("det", "shiter"):
            spec = {"e7": "intk", "variant": v, "k": k, "task": task, "ckpt": 1}
            er = episode(task, 1, seed, lambda ag, qu: RefK(ref, ag, qu, qr, k, v == "shiter"), v, k)
            en = episode(task, 1, seed, lambda ag, qu: E7.install(ag, qu, None, spec, qr=qr), v, k)
            same = er["ret"] == en["ret"] and er["success"] == en["success"]
            res["episodes"][f"{task}|intk{k}|{v}"] = {"ref": er, "new": en, "identical": same}
            log(f"paths {task} intk{k} {v}: ref {er['ret']:.4f} new {en['ret']:.4f} identical={same}")
    # v2 kernels with det / sh-iter rounding = E7's kernels (codes), trained mw-push MLP, k = 3
    agent, cfg, quant = agent_of("mw-push", 1)
    qr = R.load_ranges(None, "mw-push", 1, "e1")
    z, act = plan_inputs("mw-push")
    x = torch.cat([z, act], -1)
    with torch.no_grad():
        for v in ("det", "shiter"):
            mlp = E7.IntMLP(agent.model._dynamics, ("dyn0", "dyn1"), qr, 3, v == "shiter", type("O", (), {"base": 7})(),
                            0, True, False)
            y0 = F.linear(x, mlp.W0, mlp.b0).contiguous()
            q0 = E7.ln_mish_quant(y0, mlp.g0, mlp.be0, mlp.lo0, mlp.hi0, mlp.d0, mlp.nl, mlp.zp, 1234, v == "shiter")
            q0b = E7.ln_mish_quant2(y0, mlp.g0, mlp.be0, mlp.lo0, mlp.hi0, mlp.d0, mlp.nl, mlp.zp, 1234, v)
            q1 = E7.i8gemm_ln_mish_quant(q0, mlp.W1qT, mlp.alpha1, mlp.c1, mlp.g1, mlp.be1, mlp.lo1, mlp.hi1, mlp.d1,
                                         mlp.nl, mlp.zp, 5678, v == "shiter")
            q1b = E7.i8gemm_ln_mish_quant2(q0, mlp.W1qT, mlp.alpha1, mlp.c1, mlp.g1, mlp.be1, mlp.lo1, mlp.hi1,
                                           mlp.d1, mlp.nl, mlp.zp, 5678, v)
            res["v2_codes"][v] = {"layer0_equal": bool(torch.equal(q0, q0b)), "layer1_equal": bool(torch.equal(q1, q1b))}
    res["PASS"] = bool(all(e["identical"] for e in res["episodes"].values())
                       and all(c["layer0_equal"] and c["layer1_equal"] for c in res["v2_codes"].values()))
    dump(f"{a.out}/paths.json", res)
    log(f"PATHS {'PASS' if res['PASS'] else 'FAIL'}")
    return res


# ---------------------------------------------------------------------------------------------------- check 2: shift
@torch.no_grad()
def shift_reference(mlp, x, c0, c1, o0, o1):
    """fp32 fake-quant det+shift of one MLP forward with the kernel's W8 weights and offsets (qlib detshift formula:
    value lo + delta (clamp(round(x - o), 0, nl) + o)); teacher forcing per layer."""
    W1hat = (mlp.W1qT.t().float() * (mlp.alpha1 / mlp.d0)[:, None])            # s1 Wq1 (N1, K)
    n2 = mlp.n_out
    W2hat = (mlp.W2qT.t().float()[:n2] * (mlp.alpha2[:n2] / mlp.d1)[:, None])  # s2 Wq2 (n_out, K)
    b1 = mlp.c1 - (mlp.lo0 + mlp.zp * mlp.d0) * (W1hat.sum(1))                # bias of layer 1
    b2 = mlp.c2[:n2] - (mlp.lo1 + mlp.zp * mlp.d1) * (W2hat.sum(1))

    def codes(ylin, g, b, lo, hi, d, o):
        m = F.mish(F.layer_norm(ylin, (ylin.shape[-1],), g, b, E7.LN_EPS))
        xg = (m.clamp(lo, hi) - lo) * (1.0 / d)
        return torch.round(xg - o).clamp(0, mlp.nl), m

    r0, a0 = codes(F.linear(x, mlp.W0, mlp.b0), mlp.g0, mlp.be0, mlp.lo0, mlp.hi0, mlp.d0, o0)
    r1, _ = codes(F.linear(mlp.lo0 + mlp.d0 * (c0 + o0), W1hat, b1), mlp.g1, mlp.be1, mlp.lo1, mlp.hi1, mlp.d1, o1)
    y2 = F.linear(mlp.lo1 + mlp.d1 * (c1 + o1), W2hat, b2)
    if mlp.dyn_head:
        y2 = mlp.L2.act(mlp.L2.ln(y2))
    c1ref = mlp.c1.double() + mlp.Wa1.double() @ o0.double()
    c2ref = mlp.c2.double() + mlp.Wa2.double() @ o1.double()
    return r0, r1, y2, c1ref, c2ref, a0


def sec_shift(a):
    res = {}
    # offsets: law, sharing, freshness
    n_draw = 2000
    O = torch.stack([E7.offsets((1_000_003 * s + 17) & 0x7FFFFFFF, 512, L.DEV) for s in range(n_draw)])
    hist = torch.histc(O.flatten(), bins=20, min=-0.5, max=0.5)
    exp = O.numel() / 20
    res["offsets"] = {"n": int(O.numel()), "mean": float(O.mean()), "var": float(O.var()), "var_rel_diff":
                      float(O.var() / (1 / 12) - 1), "min": float(O.min()), "max": float(O.max()),
                      "chi2_20bins": float(((hist - exp) ** 2 / exp).sum()), "chi2_df": 19,
                      # correlation between the offset rows of different calls (200 rows x 512 units)
                      "mean_abs_corr_between_calls": float(torch.corrcoef(O[:200]).fill_diagonal_(0).abs().sum()
                                                           / (200 * 199)),
                      "corr_ref_1_over_sqrt_units": float(1 / 512 ** 0.5)}
    oo = res["offsets"]
    oo["pass"] = bool(abs(oo["mean"]) < 4 * (1 / 12) ** 0.5 / O.numel() ** 0.5 and abs(oo["var_rel_diff"]) < 0.02
                      and oo["min"] >= -0.5 and oo["max"] < 0.5 and oo["chi2_20bins"] < 60)
    log(f"offsets: {oo}")
    for task, k in (("mw-push", 3), ("cheetah-run", 5)):
        agent, cfg, quant = agent_of(task, 1)
        qr = R.load_ranges(None, task, 1, "e1")
        z, act = plan_inputs(task)
        x = torch.cat([z, act], -1)
        xsame = x[:1].repeat(len(x), 1).contiguous()
        owner = type("O", (), {"base": 4242})()
        r = {}
        with torch.no_grad():
            for name, seq, salt, dyn in (("dyn", agent.model._dynamics, 0, True), ("rew", agent.model._reward, 4,
                                                                                 False)):
                mlp = E7.IntMLP2(seq, (f"{name}0", f"{name}1"), qr, k, "detshift", owner, salt, dyn)
                rng0 = L.rng_get()
                out1 = mlp(x).clone()
                rng1 = L.rng_get()
                q0, q1 = (t.clone() for t in mlp.last_codes)
                o0, o1 = (t.clone() for t in mlp.last_offsets)
                c0, c1 = q0.float() + mlp.zp, q1.float() + mlp.zp
                r0, r1, y2, c1ref, c2ref, act0 = shift_reference(mlp, x, c0, c1, o0, o1)
                c1k = torch.addmv(mlp.c1, mlp.Wa1, o0)
                c2k = torch.addmv(mlp.c2, mlp.Wa2, o1)
                scale = max(1.0, float(y2.abs().max()))
                rel = lambda u, v: float((u.double() - v).abs().max() / v.abs().max().clamp(min=1e-12))
                d = {"mismatch0": float((c0 != r0).float().mean()), "maxdiff0": float((c0 - r0).abs().max()),
                     "mismatch1": float((c1 != r1).float().mean()), "maxdiff1": float((c1 - r1).abs().max()),
                     "head_maxabs": float((out1.reshape(y2.shape) - y2).abs().max()), "head_scale": scale,
                     "c1eff_relmax": rel(c1k, c1ref), "c2eff_relmax": rel(c2k, c2ref),
                     "global_rng_unchanged": bool(torch.equal(rng0[0], rng1[0]) and torch.equal(rng0[1], rng1[1]))}
                # second forward on the same input: new offsets, codes change; offsets = Philox(seed) - 1/2
                mlp(x)
                q0b = mlp.last_codes[0].clone()
                o0b = mlp.last_offsets[0].clone()
                d["new_offsets_differ"] = bool((o0b != o0).any())
                d["codes_changed_share0"] = float((q0b != q0).float().mean())
                # identical rows -> identical codes and outputs (one offset row per call)
                outs = mlp(xsame)
                d["identical_rows_equal"] = bool((mlp.last_codes[0] == mlp.last_codes[0][:1]).all()
                                                 and (mlp.last_codes[1] == mlp.last_codes[1][:1]).all()
                                                 and (outs == outs[:1]).all())
                # det vs det+shift codes differ (the shift is applied)
                mdet = E7.IntMLP(seq, (f"{name}0", f"{name}1"), qr, k, False, owner, salt, dyn, False)
                y0 = F.linear(x, mdet.W0, mdet.b0).contiguous()
                qdet = E7.ln_mish_quant(y0, mdet.g0, mdet.be0, mdet.lo0, mdet.hi0, mdet.d0, mdet.nl, mdet.zp, 0, False)
                d["det_vs_detshift_share_diff0"] = float((qdet != q0).float().mean())
                # element-wise error law of the dequantized layer-0 values over many calls (interior entries)
                errs = []
                for _ in range(40):
                    mlp(x)
                    qq, oo_ = mlp.last_codes[0].float() + mlp.zp, mlp.last_offsets[0]
                    deq = mlp.lo0 + mlp.d0 * (qq + oo_[None, :])
                    inner = (act0 > mlp.lo0 + mlp.d0) & (act0 < mlp.hi0 - mlp.d0)
                    errs.append(((deq - act0) / mlp.d0)[inner])
                e = torch.cat(errs)
                d["err_mean_over_delta"] = float(e.mean())
                d["err_var_over_delta2"] = float(e.var())
                d["err_var_rel_diff_vs_1_12"] = float(e.var() * 12 - 1)
                d["err_n"] = int(e.numel())
                d["pass"] = bool(d["mismatch0"] <= TOL_MISMATCH and d["maxdiff0"] <= 1 and d["mismatch1"] <= TOL_MISMATCH
                                 and d["maxdiff1"] <= 1 and d["head_maxabs"] <= TOL_HEAD * scale
                                 and d["c1eff_relmax"] <= 1e-5 and d["c2eff_relmax"] <= 1e-5
                                 and d["global_rng_unchanged"] and d["new_offsets_differ"]
                                 and d["codes_changed_share0"] > 0.05 and d["identical_rows_equal"]
                                 and d["det_vs_detshift_share_diff0"] > 0.05 and abs(d["err_mean_over_delta"]) < 0.01
                                 and abs(d["err_var_rel_diff_vs_1_12"]) < 0.02)
                r[name] = d
                log(f"shift {task} {name}: {d}")
        # closed loop: det+shift smoke episode, restore check (fp32 before / after bitwise equal)
        seed = SMOKE_SEED + 1
        f0 = episode(task, 1, seed)
        spec = {"e7": "intk", "variant": "detshift", "k": k, "task": task, "ckpt": 1}
        eds = episode(task, 1, seed, lambda ag, qu: E7.install(ag, qu, None, spec, qr=qr), "detshift", k)
        eds2 = episode(task, 1, seed, lambda ag, qu: E7.install(ag, qu, None, spec, qr=qr), "detshift", k)
        f1 = episode(task, 1, seed)
        r["closed_loop"] = {"fp32_before": f0, "detshift": eds, "detshift_repeat": eds2, "fp32_after": f1,
                            "detshift_repeat_identical": eds["ret"] == eds2["ret"],
                            "restore_identical": f0["ret"] == f1["ret"] and f0["success"] == f1["success"]}
        log(f"shift {task} closed loop: {r['closed_loop']}")
        r["pass"] = bool(r["dyn"]["pass"] and r["rew"]["pass"] and r["closed_loop"]["restore_identical"]
                         and r["closed_loop"]["detshift_repeat_identical"])
        res[task] = r
    res["PASS"] = bool(res["offsets"]["pass"] and all(res[t]["pass"] for t in ("mw-push", "cheetah-run")))
    dump(f"{a.out}/shift.json", res)
    log(f"SHIFT {'PASS' if res['PASS'] else 'FAIL'}")
    return res


# ---------------------------------------------------------------------------------------------------- check 2: pc
@torch.no_grad()
def pc_reference(mlp, x, c0, c1, rnd, s0, s1):
    """fake-quant per-channel computation with the folded W8 weights: layer-1 / head pre-activations
    s' (Wq' q) + (b + W lo) (fp32 torch), per-channel grids, det rint or sh-iter floor(x + u) with u = Philox(seed);
    teacher forcing per layer (the kernel's codes feed the next layer)."""
    def codes(ylin, g, b, lov, hiv, dv, seed):
        m = F.mish(F.layer_norm(ylin, (ylin.shape[-1],), g, b, E7.LN_EPS))
        xg = (torch.minimum(torch.maximum(m, lov), hiv) - lov) * (1.0 / dv)
        q = torch.floor(xg + urow(seed, xg.shape[-1])[None, :]) if rnd == "shiter" else torch.round(xg)
        return q.clamp(0, mlp.nl)

    Wq1, s1v = mlp.W1fold
    Wq2, s2v = mlp.W2fold
    n2 = mlp.n_out
    r0 = codes(F.linear(x, mlp.W0, mlp.b0), mlp.g0, mlp.be0, mlp.lo0v, mlp.hi0v, mlp.d0v, s0)
    y1 = (c0 @ Wq1.t()) * s1v + mlp.c1
    r1 = codes(y1, mlp.g1, mlp.be1, mlp.lo1v, mlp.hi1v, mlp.d1v, s1)
    y2 = (c1 @ Wq2.t()) * s2v + mlp.c2[:n2]
    if mlp.dyn_head:
        y2 = mlp.L2.act(mlp.L2.ln(y2))
    # end to end through the dequantized grid values with W_eff = (s' Wq') diag(1 / delta), b_eff = b + (W - W_eff) lo
    return r0, r1, y2


@torch.no_grad()
def pc_e2e(seq, pcr, names, k, rnd, x, mlp, s0, s1, weights="folded"):
    """end-to-end fake-quant per-channel forward: weights 'folded' (W_eff from the folded W8 weights, b_eff) or 'fp32'
    (the model's own weights = E12's fake-quant per-channel configuration); same grids and the same u rows."""
    L0, L1, L2 = seq[0], seq[1], seq[2]
    nl = (1 << k) - 1

    def q(m, lov, hiv, seed):
        dv = (hiv - lov) / nl
        xg = (torch.minimum(torch.maximum(m, lov), hiv) - lov) * (1.0 / dv)
        c = torch.floor(xg + urow(seed, xg.shape[-1])[None, :]) if rnd == "shiter" else torch.round(xg)
        return lov + dv * c.clamp(0, nl)

    lo0, hi0 = (torch.as_tensor(v, device=L.DEV) for v in pcr[names[0]])
    lo1, hi1 = (torch.as_tensor(v, device=L.DEV) for v in pcr[names[1]])
    if weights == "folded":
        Wq1, s1v = mlp.W1fold
        Wq2, s2v = mlp.W2fold
        d0, d1 = (hi0 - lo0) / nl, (hi1 - lo1) / nl
        W1e = (Wq1 * s1v[:, None]) / d0[None, :]
        W2e = (Wq2 * s2v[:, None]) / d1[None, :]
        b1e = L1.bias + (L1.weight - W1e) @ lo0
        b2e = L2.bias + (L2.weight - W2e) @ lo1
    else:
        W1e, W2e, b1e, b2e = L1.weight, L2.weight, L1.bias, L2.bias
    h0 = q(F.mish(L0.ln(F.linear(x, L0.weight, L0.bias))), lo0, hi0, s0)
    h1 = q(F.mish(L1.ln(F.linear(h0, W1e, b1e))), lo1, hi1, s1)
    y2 = F.linear(h1, W2e, b2e)
    return L2.act(L2.ln(y2)) if hasattr(L2, "ln") else y2


def sec_pc(a):
    res = {}
    for task in ("mw-push", "cheetah-run"):
        p = R.pc_result_path(a.res, task, 1)
        if not os.path.exists(p):
            res[task] = {"error": f"missing {p} (run the pcstats stage first)"}
            continue
        pcr = R.load_pc(a.res, task, 1)
        agent, cfg, quant = agent_of(task, 1)
        z, act = plan_inputs(task)
        x = torch.cat([z, act], -1)
        r = {}
        with torch.no_grad():
            f_dyn = agent.model._dynamics(x)
            f_rew = agent.model._reward(x)
            for k in (2, 3):
                for rnd in ("det", "shiter"):
                    for name, seq, salt, dyn, fref in (("dyn", agent.model._dynamics, 0, True, f_dyn),
                                                       ("rew", agent.model._reward, 4, False, f_rew)):
                        owner = type("O", (), {"base": 777})()
                        mlp = E7.IntMLP2(seq, (f"{name}0", f"{name}1"), None, k, rnd, owner, salt, dyn, pcr)
                        calls0 = mlp.calls
                        out = mlp(x).clone()
                        s0 = (owner.base * 1_000_003 + (calls0 + 1) * 8 + salt + 0) & 0x7FFFFFFF
                        s1 = (owner.base * 1_000_003 + (calls0 + 2) * 8 + salt + 1) & 0x7FFFFFFF
                        q0, q1 = (t.clone() for t in mlp.last_codes)
                        c0, c1 = q0.float(), q1.float()
                        r0, r1, y2 = pc_reference(mlp, x, c0, c1, rnd, s0, s1)
                        scale = max(1.0, float(y2.abs().max()))
                        e2e = pc_e2e(seq, pcr, (f"{name}0", f"{name}1"), k, rnd, x, mlp, s0, s1, "folded")
                        fq = pc_e2e(seq, pcr, (f"{name}0", f"{name}1"), k, rnd, x, mlp, s0, s1, "fp32")
                        rl2 = lambda u, v: float((u - v).norm() / v.norm().clamp(min=1e-12))
                        d = {"mismatch0": float((c0 != r0).float().mean()), "maxdiff0": float((c0 - r0).abs().max()),
                             "mismatch1": float((c1 != r1).float().mean()), "maxdiff1": float((c1 - r1).abs().max()),
                             "head_maxabs": float((out.reshape(y2.shape) - y2).abs().max()), "head_scale": scale,
                             "e2e_rel_l2_vs_folded_fakequant": rl2(out, e2e),
                             "rel_l2_vs_fp32weight_pc_fakequant": rl2(out, fq),
                             "fp32weight_pc_fakequant_rel_l2_vs_fp32": rl2(fq, fref), "rel_l2_vs_fp32": rl2(out, fref)}
                        d["pass"] = bool(d["mismatch0"] <= TOL_MISMATCH and d["maxdiff0"] <= 1
                                         and d["mismatch1"] <= TOL_MISMATCH and d["maxdiff1"] <= 1
                                         and d["head_maxabs"] <= TOL_HEAD * scale
                                         and d["e2e_rel_l2_vs_folded_fakequant"] <= 1e-2)
                        r[f"k{k}|{rnd}|{name}"] = d
                        log(f"pc {task} k{k} {rnd} {name}: {d}")
        seed = SMOKE_SEED + 2
        f0 = episode(task, 1, seed)
        eps = {}
        for rnd in ("det", "shiter"):
            spec = {"e7": "intpc", "variant": rnd, "k": 3, "task": task, "ckpt": 1}
            eps[rnd] = episode(task, 1, seed, lambda ag, qu: E7.install(ag, qu, None, spec, pcr=pcr), rnd, 3)
        f1 = episode(task, 1, seed)
        r["closed_loop"] = {"fp32_before": f0, "k3": eps, "fp32_after": f1,
                            "restore_identical": f0["ret"] == f1["ret"] and f0["success"] == f1["success"]}
        log(f"pc {task} closed loop: {r['closed_loop']}")
        r["pass"] = bool(all(v["pass"] for kk, v in r.items() if kk.startswith("k"))
                         and r["closed_loop"]["restore_identical"])
        res[task] = r
    res["PASS"] = bool(all(isinstance(v, dict) and v.get("pass") for v in res.values()))
    dump(f"{a.out}/pc.json", res)
    log(f"PC {'PASS' if res['PASS'] else 'FAIL'}")
    return res


# ---------------------------------------------------------------------------------------------------- bf16p
def sec_bf16p(a):
    res = {}
    for task in ("mw-push", "cheetah-run"):
        agent, cfg, quant = agent_of(task, 1)
        z, act = plan_inputs(task)
        with torch.no_grad():
            f_dyn, f_rew = agent.model.next(z, act, None), agent.model.reward(z, act, None)
            spec = {"e7": "bf16p", "variant": "none", "k": 0, "task": task, "ckpt": 1}
            e7 = E7.install(agent, quant, None, spec)
            b_dyn, b_rew = agent.model.next(z, act, None), agent.model.reward(z, act, None)
            e7.restore()
            quant.wq = None
            restored = ("next" not in agent.model.__dict__) and ("reward" not in agent.model.__dict__)
            n_dyn = agent.model.next(z, act, None)
        rl2 = lambda u, v: float((u - v).norm() / v.norm().clamp(min=1e-12))
        r = {"dyn_rel_l2_vs_fp32": rl2(b_dyn, f_dyn), "rew_rel_l2_vs_fp32": rl2(b_rew, f_rew),
             "reward_value_maxabs": float((L.two_hot_inv(b_rew, cfg) - L.two_hot_inv(f_rew, cfg)).abs().max()),
             "methods_restored": restored, "fp32_after_restore_equal": bool(torch.equal(n_dyn, f_dyn)),
             "dtype_out": str(b_dyn.dtype)}
        seed = SMOKE_SEED + 3
        f0 = episode(task, 1, seed)
        eb = episode(task, 1, seed, lambda ag, qu: E7.install(ag, qu, None, spec), "none", 0)
        f1 = episode(task, 1, seed)
        r["closed_loop"] = {"fp32_before": f0, "bf16p": eb, "fp32_after": f1,
                            "restore_identical": f0["ret"] == f1["ret"] and f0["success"] == f1["success"]}
        r["pass"] = bool(restored and r["fp32_after_restore_equal"] and r["closed_loop"]["restore_identical"]
                         and r["dyn_rel_l2_vs_fp32"] < 0.05 and r["dtype_out"] == "torch.float32")
        res[task] = r
        log(f"bf16p {task}: {r}")
    res["PASS"] = bool(all(v["pass"] for v in res.values()))
    dump(f"{a.out}/bf16p.json", res)
    log(f"BF16P {'PASS' if res['PASS'] else 'FAIL'}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="paths,shift,bf16p")
    ap.add_argument("--out", default="/workspace/atm/out/checks")
    ap.add_argument("--res", default="/workspace/atm/out", help="results root with jobs/pcstats (pc section)")
    a = ap.parse_args()
    torch.set_num_threads(1)
    fns = {"paths": sec_paths, "shift": sec_shift, "pc": sec_pc, "bf16p": sec_bf16p}
    summary = {}
    for s in a.sections.split(","):
        try:
            summary[s] = fns[s](a)["PASS"]
        except Exception:  # noqa: BLE001
            summary[s] = False
            log(traceback.format_exc())
            dump(f"{a.out}/{s}.json", {"error": traceback.format_exc(), "PASS": False})
    log("E17_CHECKS " + json.dumps(summary))


if __name__ == "__main__":
    main()
