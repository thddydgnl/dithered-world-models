"""E5 (DESIGN_main v5 7.3) for the TD-MPC2 part of paper A: activation PTQ baselines, weight-only quantization and QAT.

Activation PTQ baselines (same k, same 4 layers dyn0, dyn1, rew0, rew1 as E1; statistics from the E0 range rows: the
12,000-row subsample of the planning-time inputs of calibration episodes 0-4, x 3 horizon steps):
  perchan   per-channel (per unit) range [min_j, 99.99th percentile_j]
  mseclip   per-tensor [lo, alpha * hi], (lo, hi) = the E0 range (min, 99.99th pct), alpha in {0.10, 0.15, ..., 0.45}
            U {0.5, 0.6, ..., 1.1} (v5 grid extended downwards, coordinator amendment) with the smallest k-bit det
            (round-to-nearest) MSE on the same activations (per layer and k)
  smooth    SmoothQuant-style migration, alpha 0.5: s_j = max|a_j|^0.5 / max_i |W_next[i, j]|^0.5 with W_next the next
            Linear (dyn0 -> dyn1, dyn1 -> dynamics head, rew0 -> rew1, rew1 -> reward head); a / s is quantized on a
            per-tensor grid [min, 99.99th pct of a / s] and s is folded into the next Linear. With fp32 weights the fold
            is exact, so this is implemented as the equivalent per-channel grid lo_j = lo_s s_j, step d_j = d_s s_j
  mixed     the E0 ranges, dyn1 and rew1 kept at 8 bits (dyn0, rew0 at k)
Weight-only quantization (activations in fp32), the Linear weights of the same 4 layers (output heads in fp32):
  w8rtn / w4rtn   symmetric per-output-channel round-to-nearest
  w4awq           AWQ-style: s_j = (mean |x_j|)^0.5 over the layer's calibration inputs, W_eff = Q(W diag(s)) diag(1/s)
  w4sr<K>         K independent stochastic roundings of W drawn per episode; MPPI iteration i of a control step uses
                  copy i mod K (the policy-prior rollout uses copy 0)
QAT: det fake-quant with a straight-through estimator on the 4 hidden activations at k (E0 ranges), only the dynamics
and reward MLPs are trained (encoder, Q, policy frozen), TD-MPC2 consistency + reward losses (its coefficients, rho
and horizon), Adam, on unquantized-agent episodes (seeds 65000-65019).
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import qlib as L

PTQ = ("perchan", "mseclip", "smooth", "mixed")
WVARS = ("w8rtn", "w4rtn", "w4awq", "w4sr1", "w4sr2", "w4sr4")
# MSE clip grid: v5 7.3 {0.5, ..., 1.1} extended downwards by the coordinator's pre-run amendment (2026-10-22, v5.1):
# {0.10, 0.15, ..., 0.45} (the k=3 optimum sat on the old lower end 0.5 in 102 of 144 cases)
ALPHAS = tuple(round(0.10 + 0.05 * i, 2) for i in range(8)) + (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1)
Q_HI = 0.9999
NEXT = {"dyn0": ("_dynamics", 1), "dyn1": ("_dynamics", 2), "rew0": ("_reward", 1), "rew1": ("_reward", 2)}
EPS_RANGE = 1e-6


def stats_path(out, task, ckpt):
    return f"{out}/e5stats/{task}_s{int(ckpt)}.pt"


def qat_path(out, task, ckpt, k):
    return f"{out}/qat/{task}_s{int(ckpt)}_k{int(k)}.pt"


def traj_path(out, task, ckpt, ep):
    return f"{out}/qatdata/{task}_s{int(ckpt)}_ep{int(ep)}.npz"


def _kth_largest(v, q=Q_HI, dim=None):
    if dim is None:
        k = max(1, int(round((1 - q) * v.numel())))
        return torch.topk(v.reshape(-1), k).values[-1]
    k = max(1, int(round((1 - q) * v.shape[dim])))
    return torch.topk(v, k, dim=dim).values[-1]


def _det_mse(v, lo, hi, nl):
    d = (hi - lo) / nl
    q = lo + d * torch.round((v.clamp(lo, hi) - lo) / d)
    return float(((q - v) ** 2).mean())


# ---------------------------------------------------------------------------------------------------- statistics
@torch.no_grad()
def collect(agent, quant, out, task, ckpt, cal_eps, max_rows=12000):
    """Layer outputs (rows, 512) and mean |input| per input channel of the 4 layers on the E0 range rows (the same
    subsample and procedure as calib_lib.quant_ranges_from_calib)."""
    from calib_lib import load_split
    model = agent.model
    Zu, idx, A = load_split(out, task, ckpt, cal_eps)
    Zu, idx, A = (torch.as_tensor(x, device=L.DEV) for x in (Zu, idx, A))
    rng = np.random.default_rng(1)
    sel = torch.as_tensor(np.sort(rng.choice(len(idx), size=min(max_rows, len(idx)), replace=False)), device=L.DEV)
    acts = {n: [] for n in L.LAYERS}
    insum = {n: None for n in L.LAYERS}
    incnt = {n: 0 for n in L.LAYERS}

    def pre(n):
        def f(mod, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            s = x.abs().sum(0)
            insum[n] = s if insum[n] is None else insum[n] + s
            incnt[n] += x.shape[0]
        return f

    hs = [quant.mods[n].register_forward_pre_hook(pre(n)) for n in L.LAYERS]
    try:
        quant.set("record")
        z = Zu[idx[sel]]
        a = A[sel]
        for h in range(A.shape[1]):
            quant.last.clear()
            model.reward(z, a[:, h], None)
            z_next = model.next(z, a[:, h], None)
            for n in L.LAYERS:
                acts[n].append(quant.last[n].float())
            z = z_next
        quant.off()
    finally:
        for h_ in hs:
            h_.remove()
    return ({n: torch.cat(acts[n]) for n in L.LAYERS},
            {n: (insum[n] / max(incnt[n], 1)) for n in L.LAYERS})


@torch.no_grad()
def compute_stats(agent, quant, out, spec):
    """e5stats job: everything the PTQ baselines and w4awq need, saved to <out>/e5stats/<task>_s<ckpt>.pt."""
    task, ckpt = spec["task"], int(spec["ckpt"])
    ks = [int(k) for k in spec.get("ks", [3, 4])]
    acts, inmean = collect(agent, quant, out, task, ckpt, spec.get("cal_eps", [0, 1, 2, 3, 4]))
    e0 = json.load(open(f"{out}/scales/{task}_s{ckpt}.json"))["quant"]
    model = agent.model
    st = {"base": {}, "perchan": {}, "mse_alpha": {k: {} for k in ks}, "mse": {k: {} for k in ks}, "smooth": {},
          "awq_absmean": {}, "check": {}}
    for n in L.LAYERS:
        a = acts[n]
        lo, hi = float(a.min()), float(_kth_largest(a))
        st["base"][n] = (lo, hi)
        st["check"][n] = {"lo_minus_e0": lo - float(e0[n][0]), "hi_minus_e0": hi - float(e0[n][1])}
        # per-channel range
        lo_c = a.min(0).values
        hi_c = torch.maximum(_kth_largest(a, dim=0), lo_c + EPS_RANGE)
        st["perchan"][n] = (lo_c.cpu(), hi_c.cpu())
        # MSE-optimal clip factor per k
        for k in ks:
            nl = (1 << k) - 1
            errs = {al: _det_mse(a, lo, max(al * hi, lo + EPS_RANGE), nl) for al in ALPHAS}
            st["mse"][k][n] = errs
            st["mse_alpha"][k][n] = min(ALPHAS, key=lambda al: (errs[al], ALPHAS.index(al)))
        # SmoothQuant scales and the equivalent per-channel grid
        mod_name, li = NEXT[n]
        W = getattr(model, mod_name)[li].weight.detach().float()
        amax = a.abs().max(0).values.clamp(min=1e-5)
        wmax = W.abs().max(0).values.clamp(min=1e-5)
        s = amax.sqrt() / wmax.sqrt()
        xs = a / s
        lo_s, hi_s = float(xs.min()), float(_kth_largest(xs))
        st["smooth"][n] = ((lo_s * s).cpu(), torch.maximum(hi_s * s, lo_s * s + EPS_RANGE).cpu(), lo_s, hi_s,
                           s.cpu())
        st["awq_absmean"][n] = inmean[n].cpu()
    os.makedirs(f"{out}/e5stats", exist_ok=True)
    p = stats_path(out, task, ckpt)
    tmp = f"{p}.tmp{os.getpid()}"
    torch.save(st, tmp)
    os.replace(tmp, p)
    return {"path": p, "mse_alpha": {str(k): v for k, v in st["mse_alpha"].items()}, "check": st["check"],
            "n_rows": int(acts["dyn0"].shape[0])}


def load_stats(out, task, ckpt):
    return torch.load(stats_path(out, task, ckpt), map_location="cpu")


def ptq_config(stats, ptq, k):
    """-> (qrange {layer: (lo, hi)} with floats or (512,) CUDA tensors, layer_k {layer: bits})."""
    k = int(k)
    if ptq == "perchan":
        return {n: (stats["perchan"][n][0].to(L.DEV), stats["perchan"][n][1].to(L.DEV)) for n in L.LAYERS}, {}
    if ptq == "mseclip":
        out = {}
        for n in L.LAYERS:
            lo, hi = stats["base"][n]
            out[n] = (lo, max(stats["mse_alpha"][k][n] * hi, lo + EPS_RANGE))
        return out, {}
    if ptq == "smooth":
        return {n: (stats["smooth"][n][0].to(L.DEV), stats["smooth"][n][1].to(L.DEV)) for n in L.LAYERS}, {}
    if ptq == "mixed":
        return {n: tuple(stats["base"][n]) for n in L.LAYERS}, {"dyn1": 8, "rew1": 8}
    raise ValueError(ptq)


def resolve_best(out, task, k):
    sel = json.load(open(f"{out}/e5/selection.json"))
    return sel["cells"][f"{task}|{int(k)}"]["best"]


# ---------------------------------------------------------------------------------------------------- weights
def rtn_per_channel(W, bits):
    qmax = (1 << (bits - 1)) - 1
    s = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return torch.round(W / s).clamp(-qmax - 1, qmax) * s


def sr_per_channel(W, bits, gen):
    qmax = (1 << (bits - 1)) - 1
    s = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    u = torch.rand(W.shape, generator=gen, device=W.device, dtype=W.dtype)
    return torch.floor(W / s + u).clamp(-qmax - 1, qmax) * s


class WeightQuant:
    """Puts quantized Linear weights of dyn0, dyn1, rew0, rew1 in place for one job (restore() puts the fp32 weights
    back). For w4sr<K>, forward pre-hooks select copy (MPPI iteration mod K) at every call: within a control step the
    j-th call of a layer with num_samples rows belongs to iteration j // horizon; other row counts (the policy-prior
    rollout) use copy 0. Quantizer.begin_step() resets the counters, Quantizer.set() draws new copies per episode."""

    def __init__(self, quant, var, n_samples, horizon, stats=None):
        if var not in WVARS:
            raise ValueError(var)
        self.quant, self.var, self.N, self.H = quant, var, int(n_samples), int(horizon)
        self.mods = quant.mods
        self.orig = {n: m.weight.detach().clone() for n, m in self.mods.items()}
        self.K = int(var[4:]) if var.startswith("w4sr") else 1
        self.calls, self.loaded, self.copies, self.handles = {}, {}, None, []
        if var in ("w8rtn", "w4rtn"):
            bits = 8 if var == "w8rtn" else 4
            self._put({n: rtn_per_channel(self.orig[n].float(), bits) for n in self.mods})
        elif var == "w4awq":
            ws = {}
            for n in self.mods:
                s = stats["awq_absmean"][n].to(self.orig[n].device).float().clamp(min=1e-6).sqrt()
                ws[n] = rtn_per_channel(self.orig[n].float() * s[None], 4) / s[None]
            self._put(ws)
        else:
            self.new_episode(0)
            if self.K > 1:
                self.handles = [m.register_forward_pre_hook(self._pre(n)) for n, m in self.mods.items()]

    def _put(self, ws):
        for n, m in self.mods.items():
            m.weight.data.copy_(ws[n].to(m.weight.dtype))
            self.loaded[n] = None

    def new_episode(self, seed):
        if self.copies is None and not self.var.startswith("w4sr"):
            return
        g = torch.Generator(device=L.DEV)
        g.manual_seed(5_000_011 + int(seed))
        self.copies = [{n: sr_per_channel(self.orig[n].float(), 4, g) for n in self.mods} for _ in range(self.K)]
        for n, m in self.mods.items():
            m.weight.data.copy_(self.copies[0][n])
            self.loaded[n] = 0
        self.calls = {}

    def begin_step(self):
        self.calls = {}

    def _pre(self, n):
        def f(mod, inp):
            rows = int(inp[0].reshape(-1, inp[0].shape[-1]).shape[0])
            key = (n, rows)
            c = self.calls.get(key, 0)
            self.calls[key] = c + 1
            idx = (c // self.H) % self.K if rows == self.N else 0
            if idx != self.loaded.get(n):
                mod.weight.data.copy_(self.copies[idx][n])
                self.loaded[n] = idx
        return f

    def restore(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        for n, m in self.mods.items():
            m.weight.data.copy_(self.orig[n])


# ---------------------------------------------------------------------------------------------------- QAT
def model_state(agent):
    return {"_dynamics": {k: v.detach().clone() for k, v in agent.model._dynamics.state_dict().items()},
            "_reward": {k: v.detach().clone() for k, v in agent.model._reward.state_dict().items()}}


def load_state(agent, st):
    agent.model._dynamics.load_state_dict(st["_dynamics"])
    agent.model._reward.load_state_dict(st["_reward"])


def save_traj(out, task, ckpt, ep, r):
    p = traj_path(out, task, ckpt, ep)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = f"{p}.tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, obs=r["_obs"].numpy(), act=r["_act"].numpy(), rew=r["_rew"].numpy())
    os.replace(tmp, p)
    return p


def qat_train(agent, cfg, quant, out, spec):
    """qat_train job (see the module docstring). Trains on a copy of the cached agent's weights and restores them."""
    from common import math as tmath
    task, ckpt, k = spec["task"], int(spec["ckpt"]), int(spec["k"])
    H, rho = int(cfg.horizon), float(cfg.rho)
    cc, rc = float(cfg.consistency_coef), float(cfg.reward_coef)
    obs, act, rew = [], [], []
    for ep in spec["data_eps"]:
        d = np.load(traj_path(out, task, ckpt, ep))
        obs.append(d["obs"])
        act.append(d["act"])
        rew.append(d["rew"])
    starts = [(i, t) for i, a in enumerate(act) for t in range(a.shape[0] - H + 1)]
    O = [torch.as_tensor(o, device=L.DEV) for o in obs]
    Aa = [torch.as_tensor(a, device=L.DEV) for a in act]
    Rr = [torch.as_tensor(r, device=L.DEV) for r in rew]
    model = agent.model
    saved = model_state(agent)
    req = {nm: p.requires_grad for nm, p in model.named_parameters()}
    params = list(model._dynamics.parameters()) + list(model._reward.parameters())
    pids = {id(p) for p in params}
    for p in model.parameters():
        p.requires_grad_(id(p) in pids)
    quant.qrange = {n: (float(v[0]), float(v[1])) for n, v in
                    json.load(open(f"{out}/scales/{task}_s{ckpt}.json"))["quant"].items()}
    opt = torch.optim.Adam(params, lr=float(spec["lr"]))
    rng = np.random.default_rng(int(spec.get("seed", 0)))
    torch.manual_seed(int(spec.get("seed", 0)))
    B, steps, clip = int(spec["batch"]), int(spec["steps"]), float(spec["grad_clip"])
    hist = []
    try:
        with torch.enable_grad():
            quant.set_ste(k)
            for step in range(steps):
                pick = rng.integers(0, len(starts), size=B)
                ob = torch.stack([O[starts[j][0]][starts[j][1]:starts[j][1] + H + 1] for j in pick], dim=1)
                ac = torch.stack([Aa[starts[j][0]][starts[j][1]:starts[j][1] + H] for j in pick], dim=1)
                rw = torch.stack([Rr[starts[j][0]][starts[j][1]:starts[j][1] + H] for j in pick], dim=1)
                with torch.no_grad():
                    target = model.encode(ob[1:], None)
                    z = model.encode(ob[0], None)
                cons = rl = 0.0
                for h in range(H):
                    r_pred = model.reward(z, ac[h], None)
                    z = model.next(z, ac[h], None)
                    cons = cons + F.mse_loss(z, target[h]) * rho ** h
                    rl = rl + tmath.soft_ce(r_pred, rw[h].unsqueeze(-1), cfg).mean() * rho ** h
                loss = (cc * cons + rc * rl) / H
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(params, clip)
                opt.step()
                if step % 250 == 0 or step == steps - 1:
                    hist.append({"step": step, "loss": float(loss), "cons": float(cons) / H, "rew": float(rl) / H,
                                 "grad_norm": float(gn)})
        quant.off()
        trained = model_state(agent)
        qp = qat_path(out, task, ckpt, k)
        os.makedirs(os.path.dirname(qp), exist_ok=True)
        tmp = f"{qp}.tmp{os.getpid()}"
        torch.save({kk: {n: v.cpu() for n, v in d.items()} for kk, d in trained.items()}, tmp)
        os.replace(tmp, qp)
    finally:
        quant.off()
        load_state(agent, saved)
        for nm, prm in model.named_parameters():
            prm.requires_grad_(req[nm])
    return {"path": qp, "n_windows": len(starts), "hist": hist}


def load_qat(agent, out, task, ckpt, k):
    """Puts the QAT weights in place; returns the fp32 state to restore with load_state."""
    saved = model_state(agent)
    st = torch.load(qat_path(out, task, ckpt, k), map_location=L.DEV)
    load_state(agent, st)
    return saved
