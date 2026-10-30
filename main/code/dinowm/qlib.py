"""Paper A main experiments, DINO-WM part: k-bit quantization of the predictor MLP hidden activations (GELU outputs
of all 6 predictor blocks) with the 7 rounding variants of EXPERIMENT_DESIGN.md 4.2, batched CEM with DINO-WM
CEMPlanner semantics, and per-episode diagnostics (planning-time 1/3/5-step errors, OEI inputs).
Builds on the pilot qlib13.py (R2-13): the quantizer kernels of det/ind/shared dithers, the pilot variants (their
seed formulas included) and the CEM loop are unchanged; new are shstep, shep, detshift and the diagnostics.

Variants. Same per-layer k-bit uniform grid [lo, hi], D = (hi - lo)/(2^k - 1), y = (clamp(x, lo, hi) - lo)/D:
  none      unquantized
  det       deterministic rounding round(y) (RTN)
  ind       stochastic rounding floor(y + u), u i.i.d. per entry (candidate x token x unit), fresh every forward
  shiter    floor(y + u), u shared by the 300 candidates of an episode (independent across tokens and units), fresh
            at every predictor forward call = every CEM iteration and every horizon step     [main prescription]
  shstep    floor(y + u), u shared by the candidates; one table per (layer, horizon step) drawn at the start of every
            plan (control step) and reused in all CEM iterations of that plan
  shep      as shstep, but the table is drawn once per episode.
            DINO-WM is planned open-loop here (one plan per episode), so shstep and shep have the same distribution
            and differ only in the draw (independent seeds).
  detshift  deterministic rounding on a randomly shifted grid: clamp(round(y - o), 0, 2^k - 1) + o (2^k levels), with
            an offset table o ~ U[-1/2, 1/2) of the same shape and schedule as shiter's u: one value per (token, unit),
            shared by the 300 candidates of an episode, fresh at every predictor forward call. Same input -> same
            error within a call; no error persists across CEM iterations. Same construction as the TD-MPC2 part
            (code/tdmpc2/qlib.py: per-unit offset = the granularity of its shiter u).
  shfixed   the pilot R2-13 sh-fixed (= shstep with the pilot seed formula); kept for the reproduction runs
ind/shiter/shstep/shep/shfixed have identical per-entry error marginals (mean 0, mean MSE D^2/6); det and detshift
have mean MSE D^2/12 (detshift error is uniform on [-D/2, D/2) for interior values and unbiased over o; det is not).
Dither randomness never touches the planner's random stream (CPU generators); shared tables come from per-call CUDA
generators seeded by seed_of(...); ind uses the default CUDA generator re-seeded at begin_plan (as in the pilot).
Runs on the server only.
"""
import os, sys, json, time, zlib, random
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wmlib as L

VARIANTS = ('none', 'det', 'ind', 'shiter', 'shstep', 'shep', 'detshift')    # design 4.2, in this order
# E3 additions (DESIGN_main v5 draft): indplan = independent per candidate slot, fixed for the plan; shunit /
# shgroup / shlayer = sh-plan with a smaller fixed table per (layer, horizon step): one value per unit (2048), per
# block of 16 units (128 values), per layer-step (1 value), shared by all tokens and candidates
QVARS = ('det', 'ind', 'shiter', 'shstep', 'shep', 'detshift', 'shfixed', 'indplan', 'shunit', 'shgroup', 'shlayer')
STOCH = ('ind', 'shiter', 'shstep', 'shep', 'shfixed', 'indplan', 'shunit', 'shgroup', 'shlayer')   # stochastic rounding
RAND = STOCH + ('detshift',)          # randomized variants: prediction errors are averaged over dither draws
SHARED_U = ('shiter', 'shstep', 'shep', 'shfixed', 'shunit', 'shgroup', 'shlayer')
UNITS_PER_GROUP = 16                  # shgroup: 2048 units / 16 = 128 independent values per (layer, horizon step)
# E5 weight-quantization variants (activations unquantized; ptq.WeightQuant must be attached as Quantizer.wq)
WVARS = ('w8rtn', 'w4rtn', 'w4awq', 'w4sr1', 'w4sr2', 'w4sr4')
FRAMESKIP = 5
# start/goal splits: pair seed -> L.sample_pairs (DINO-WM PlanWorkspace.prepare_targets), planner seed plan_seed + e.
#   held  = pilot held split (R2-7, R2-13): range/k selection only, never used for verdicts
#   repro = pilot R2-13 eval split (first 50 pairs of pair seed 4): reproduction only
#   eval  = E1 main evaluation split (pair seed 5: not used by any pilot; pilots used 1, 2, 3, 4, 7, 11)
#   smoke = code smoke tests only (pair seed 7 = the R2-1 smoke pairs)
# salt = string mixed into the dither seeds (repro keeps the pilot's 'eval' so its shared tables are the pilot's).
SPLITS = {'held': dict(pair_seed=3, n=20, plan_seed=1000, salt='held'),
          'repro': dict(pair_seed=4, n=50, plan_seed=2000, salt='eval'),
          'eval': dict(pair_seed=5, n=100, plan_seed=3000, salt='eval5'),
          'smoke': dict(pair_seed=7, n=4, plan_seed=9000, salt='smoke'),
          # PushT E1 eval split of DESIGN_main v3 (user-approved 2026-10-21T22:54Z): pair seed 6, drawn like DINO-WM
          # prepare_targets but sharing no dataset frame with the held pairs (and the E0 smoke pairs) and, whenever
          # possible, none with the pilot eval pairs (pair seed 4, pairs 0-49); env seeds 6001-6100, planner seeds
          # 4000-4099, nonealt 4500-4599, dither salt 'eval6' (all disjoint from earlier splits). PushT only.
          'eval6': dict(pair_seed=6, n=100, plan_seed=4000, salt='eval6', sampler='constrained',
                        hard=(('held', 20), ('smoke', 4)), soft=(('repro', 50),))}
ALT_SEED_OFFSET = 500      # 'nonealt': unquantized with planner seed plan_seed + 500 + e (seed-noise floor)
ERR_STEPS = (1, 3, 5)      # planning-time k-step errors (horizon steps of the iteration-0 rollout)


def seed_of(*parts):
    return zlib.crc32(repr(parts).encode()) & 0x7fffffff


def plan_seed(split, var, e):
    sp = SPLITS[split]
    return sp['plan_seed'] + (ALT_SEED_OFFSET if var == 'nonealt' else 0) + int(e)


# ------------------------------------------------------------------------------------------------ quantizer kernels
def _qdet(x, lo, d, nl):
    y = ((x.float() - lo) / d).clamp(min=0.0).clamp(max=nl)
    return (lo + d * torch.round(y)).to(x.dtype)


def _qind(x, lo, d, nl):
    y = ((x.float() - lo) / d).clamp(min=0.0).clamp(max=nl)
    q = torch.floor(y + torch.rand_like(y)).clamp(max=nl)
    return (lo + d * q).to(x.dtype)


def _qsh(x4, u4, lo, d, nl):
    """x4 (G, N, L, C); u4 (G, 1, L, C) float32 uniform: shared by the N rows of a group."""
    y = ((x4.float() - lo) / d).clamp(min=0.0).clamp(max=nl)
    q = torch.floor(y + u4).clamp(max=nl)
    return (lo + d * q).to(x4.dtype)


def _qshift(x4, o4, lo, d, nl):
    """x4 (G, N, L, C); o4 (G, 1, L, C) float32 offsets in [-1/2, 1/2), shared by the N rows of a group:
    RTN on the grid shifted by o (levels o, 1 + o, ..., nl + o)."""
    y = ((x4.float() - lo) / d).clamp(min=0.0).clamp(max=nl)
    q = torch.round(y - o4).clamp(min=0.0).clamp(max=nl) + o4
    return (lo + d * q).to(x4.dtype)


_CH = 64   # eager fallback: rows per chunk (bounds the float32 temporaries)


def _chunked(fn):
    def run(x, *args):
        out = torch.empty_like(x)
        for s in range(0, x.shape[0], _CH):
            out[s:s + _CH] = fn(x[s:s + _CH], *args)
        return out
    return run


def _grouped_eager(fn):
    def run(x4, u4, lo, d, nl):
        out = torch.empty_like(x4)
        for g in range(x4.shape[0]):
            for s in range(0, x4.shape[1], _CH):
                out[g, s:s + _CH] = fn(x4[g, s:s + _CH][None], u4[g][None], lo, d, nl)[0]
        return out
    return run


def make_kernels(use_compile=True):
    if use_compile:
        kw = dict(dynamic=True, fullgraph=True)
        return (torch.compile(_qdet, **kw), torch.compile(_qind, **kw), torch.compile(_qsh, **kw),
                torch.compile(_qshift, **kw))
    return _chunked(_qdet), _chunked(_qind), _grouped_eager(_qsh), _grouped_eager(_qshift)


class Quantizer:
    """Forward hooks on the GELU of the 6 predictor MLPs (blk[1].net[2], as in R2-7/R2-13).
    mode None: off; 'record': keep the GELU outputs (calibration); 'quant': k-bit quantization with variant self.var.
    Rows of a forward call are grouped as (G episodes) x (N candidates); shared dithers are shared inside a group.
    self.it (CEM iteration) and self.h (horizon step) are set by the caller before every predictor call."""

    def __init__(self, m, use_compile=True):
        self.mods = [blk[1].net[2] for blk in m.predictor.transformer.layers]
        self.handles = [mod.register_forward_hook(self._hook(li)) for li, mod in enumerate(self.mods)]
        self.mode, self.var, self.k = None, 'none', 0
        self.ranges = None
        self.ep_ids, self.salt, self.it, self.h, self.plan_idx = [0], 'x', 0, 0, 0
        self.kdet, self.kind, self.ksh, self.kshift = make_kernels(use_compile)
        self.rec = None
        self.n_calls = 0
        self.chunked = False       # set by rollout_chunks while a candidate chunk is processed
        # E5 options (defaults keep the E1 behaviour): mixed precision, weight variants, straight-through (QAT)
        self.last_bits = None      # e.g. 8: the last block's MLP hidden is quantized with this many bits
        self.wq = None             # ptq.WeightQuant for the weight variants
        self.ste = False           # QAT: forward quantized, gradient straight through
        self.pre_handle = (m.predictor.register_forward_pre_hook(self._pre_predict)
                           if hasattr(m.predictor, 'register_forward_pre_hook') else None)

    # ---- configuration
    def set_ranges(self, ranges):
        """per layer (lo, hi): floats (per-tensor grid, E1) or equal-length lists (per-channel grid, E5)."""
        self.ranges = [(float(lo), float(hi)) if not isinstance(lo, (list, tuple)) else (list(lo), list(hi))
                       for lo, hi in ranges]

    def _pre_predict(self, module, inp):
        if self.wq is not None and self.var in WVARS:      # weight SR rotation: iteration it uses copy it mod K
            self.wq.at_iteration(self.it)

    def configure(self, var, k=0):
        self.var, self.k = var, int(k)
        if self.wq is not None and self.wq.active is not None and self.wq.active != var:
            self.wq.restore()                              # unquantized reference forwards use the fp weights
        if var in ('none', 'nonealt'):
            self.mode = None
            return
        if var in WVARS:
            assert self.wq is not None, 'weight variant without an attached ptq.WeightQuant'
            self.mode = None                               # activations unquantized; weights put in place by apply_weights
            return
        assert var in QVARS and self.ranges is not None, var
        nl = (1 << self.k) - 1
        self.mode = 'quant'
        self.nl_t = torch.tensor(float(nl), device='cuda')
        if all(not isinstance(lo, list) for lo, hi in self.ranges):     # E1 path (unchanged)
            self.lo_t = [torch.tensor(lo, device='cuda') for lo, hi in self.ranges]
            self.d_t = [torch.tensor((hi - lo) / nl, device='cuda') for lo, hi in self.ranges]
        else:                                                            # per-channel grids
            self.lo_t = [torch.tensor(lo, device='cuda', dtype=torch.float32) for lo, hi in self.ranges]
            self.d_t = [(torch.tensor(hi, device='cuda', dtype=torch.float32) - torch.tensor(lo, device='cuda', dtype=torch.float32)) / nl
                        for lo, hi in self.ranges]
        self.nl_l = [self.nl_t] * len(self.ranges)
        if self.last_bits:                                               # mixed precision: last block at last_bits
            nl8 = (1 << int(self.last_bits)) - 1
            lo, hi = self.ranges[-1]
            self.nl_l = self.nl_l[:-1] + [torch.tensor(float(nl8), device='cuda')]
            self.d_t = self.d_t[:-1] + [(torch.tensor(hi, device='cuda', dtype=torch.float32) - torch.tensor(lo, device='cuda', dtype=torch.float32)) / nl8]

    def apply_weights(self, var, seed):
        """put the quantized weights of weight variant `var` in place (SR copies drawn from `seed`)."""
        self.wq.apply(var, seed=seed)

    def begin_plan(self, ep_ids, salt, plan_idx=0):
        """ep_ids: one id per row group (episode); salt: run identifier; plan_idx: control step (0 for open loop)."""
        self.ep_ids, self.salt, self.plan_idx = [int(e) for e in ep_ids], str(salt), int(plan_idx)
        self.it, self.h = 0, 0
        if self.var == 'ind':   # in-kernel RNG of the compiled kernel draws from the default CUDA generator
            torch.cuda.manual_seed(seed_of(self.salt, 'ind', *self.ep_ids))
        if self.var in WVARS:   # E5: quantized weights for this plan (SR copies seeded per episode block)
            assert len(self.ep_ids) == 1 or not self.var.startswith('w4sr'), 'weight SR copies need E=1'
            self.apply_weights(self.var, seed_of(self.salt, self.var, *self.ep_ids))

    # ---- calibration
    def start_record(self):
        self.mode = 'record'
        self.rec = [[] for _ in self.mods]

    def finish_record(self, q_hi=0.9999):
        self.mode = None
        out = []
        for li in range(len(self.mods)):
            v = torch.cat([a.reshape(-1) for a in self.rec[li]]).float()
            k = max(1, int(round((1 - q_hi) * v.numel())))
            hi = float(torch.topk(v, k).values[-1])
            out.append(dict(lo=float(v.min()), hi=hi, max=float(v.max()), n=int(v.numel()),
                            frac_below_0=float((v < 0).float().mean()), p50=float(v.median()),
                            p99=float(torch.topk(v, max(1, int(round(0.01 * v.numel())))).values[-1])))
            self.rec[li] = None
        self.rec = None
        return out

    # ---- shared dither tables and grid offsets
    def _table_seed(self, e, li):
        v = self.var
        if v == 'shiter':      # pilot formula: fresh at every (iteration, horizon step)
            return seed_of(self.salt, 'shiter', e, self.it, self.h, li)
        if v == 'shfixed':     # pilot formula: one table per (layer, horizon step) for the whole plan
            return seed_of(self.salt, 'shfixed', e, self.h, li)
        if v == 'shstep':      # one table per (layer, horizon step) per plan (control step)
            return seed_of(self.salt, 'shstep', e, self.plan_idx, self.h, li)
        if v == 'shep':        # one table per (layer, horizon step) per episode
            return seed_of(self.salt, 'shep', e, self.h, li)
        if v in ('shunit', 'shgroup', 'shlayer', 'indplan'):   # E3: fixed for the plan (control step)
            return seed_of(self.salt, v, e, self.plan_idx, self.h, li)
        raise ValueError(v)

    def _shared_u(self, li, Lt, C):
        G = len(self.ep_ids)
        u = torch.empty((G, 1, Lt, C), device='cuda', dtype=torch.float32)
        for g, e in enumerate(self.ep_ids):
            gen = torch.Generator(device='cuda')
            gen.manual_seed(self._table_seed(e, li))
            if self.var == 'shunit':
                u[g, 0] = torch.rand((1, C), generator=gen, device='cuda').expand(Lt, C)
            elif self.var == 'shgroup':
                u[g, 0] = torch.rand((1, C // UNITS_PER_GROUP), generator=gen, device='cuda').repeat_interleave(
                    UNITS_PER_GROUP, dim=1).expand(Lt, C)
            elif self.var == 'shlayer':
                u[g, 0] = torch.rand((1, 1), generator=gen, device='cuda').expand(Lt, C)
            else:                                  # E1 variants: unchanged
                u[g, 0] = torch.rand((Lt, C), generator=gen, device='cuda')
        return u

    def _indplan_u(self, li, Nr, Lt, C):
        """indplan: u independent per (candidate slot, token, unit), fixed for the plan; (G, Nr, Lt, C). Candidates are
        resampled every CEM iteration, so the persistence is per slot (slot 0 = current mean), not per action sequence."""
        assert not self.chunked, 'indplan does not support candidate chunking'
        G = len(self.ep_ids)
        u = torch.empty((G, Nr, Lt, C), device='cuda', dtype=torch.float32)
        for g, e in enumerate(self.ep_ids):
            gen = torch.Generator(device='cuda')
            gen.manual_seed(self._table_seed(e, li))
            u[g] = torch.rand((Nr, Lt, C), generator=gen, device='cuda')
        return u

    def _shift(self, li, Lt, C):
        """detshift offsets: same table shape and schedule as shiter (fresh at every forward call)."""
        G = len(self.ep_ids)
        o = torch.empty((G, 1, Lt, C), device='cuda', dtype=torch.float32)
        for g, e in enumerate(self.ep_ids):
            gen = torch.Generator(device='cuda')
            gen.manual_seed(seed_of(self.salt, 'detshift', e, self.plan_idx, self.it, self.h, li))
            o[g, 0] = torch.rand((Lt, C), generator=gen, device='cuda') - 0.5
        return o

    def _hook(self, li):
        def hook(module, inp, out):
            if self.mode is None:
                return None
            if self.mode == 'record':
                self.rec[li].append(out.detach().reshape(-1, out.shape[-1]))
                return None
            self.n_calls += 1
            lo, d, nl = self.lo_t[li], self.d_t[li], self.nl_l[li]
            if self.ste:                                   # QAT: forward det-quantized, gradient straight through
                return out + (self.kdet(out.detach(), lo, d, nl) - out).detach()
            if self.var == 'det':
                return self.kdet(out, lo, d, nl)
            if self.var == 'ind':
                return self.kind(out, lo, d, nl)
            B, Lt, C = out.shape
            G = len(self.ep_ids)
            assert B % G == 0, (B, G)
            x4 = out.reshape(G, B // G, Lt, C)
            if self.var == 'detshift':
                return self.kshift(x4, self._shift(li, Lt, C), lo, d, nl).reshape(B, Lt, C)
            if self.var == 'indplan':
                return self.ksh(x4, self._indplan_u(li, B // G, Lt, C), lo, d, nl).reshape(B, Lt, C)
            return self.ksh(x4, self._shared_u(li, Lt, C), lo, d, nl).reshape(B, Lt, C)
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        if self.pre_handle is not None:
            self.pre_handle.remove()


# ------------------------------------------------------------------------------------------------ world model
def _tile(m, vis, pro, act_emb):
    P = vis.shape[2]
    return torch.cat([vis, pro.unsqueeze(2).expand(-1, -1, P, -1).repeat(1, 1, 1, m.num_proprio_repeat),
                      act_emb.unsqueeze(2).expand(-1, -1, P, -1).repeat(1, 1, 1, m.num_action_repeat)], dim=3)


@torch.no_grad()
def rollout_q(m, Q, z0v, z0p, actions, all_frames=False):
    """r27lib.rollout_batched (= VWorldModel.rollout with the start latent tiled) with the horizon index exposed to the
    quantizer. z0v (E,1,P,C), z0p (E,1,Dp), actions (E,N,H,A). Returns final visual (E,N,P,C) and proprio (E,N,Dp);
    with all_frames also the predicted visual/proprio of every horizon step (E,N,H,P,C), (E,N,H,Dp)."""
    E, N, H, A = actions.shape
    act = actions.reshape(E * N, H, A)
    vis = z0v.repeat_interleave(N, 0)
    pro = z0p.repeat_interleave(N, 0)
    z = _tile(m, vis, pro, m.encode_act(act[:, :1]))
    action = act[:, 1:]
    for t in range(action.shape[1]):
        Q.h = t
        z_pred = m.predict(z[:, -m.num_hist:])
        z_new = m.replace_actions_from_z(z_pred[:, -1:, ...], action[:, t:t + 1, :])
        z = torch.cat([z, z_new], dim=1)
    Q.h = action.shape[1]
    z_pred = m.predict(z[:, -m.num_hist:])
    z = torch.cat([z, z_pred[:, -1:, ...]], dim=1)
    z_obses, _ = m.separate_emb(z)
    v, p = z_obses['visual'], z_obses['proprio']
    out = (v[:, -1].reshape(E, N, *v.shape[2:]), p[:, -1].reshape(E, N, -1))
    if all_frames:
        out = out + (v[:, 1:].reshape(E, N, H, *v.shape[2:]), p[:, 1:].reshape(E, N, H, -1))
    return out


@torch.no_grad()
def rollout_chunks(m, Q, z0v, z0p, actions, all_frames=False, max_rows=None):
    """rollout_q over candidate chunks of at most max_rows (memory bound for large N, E2). Rows are computed
    independently (batch-size invariance checked bitwise in E0), and the shared dither tables depend only on
    (episode, iteration, step, layer), so chunking does not change det / shared-dither results (ind would draw a
    different in-kernel random stream; indplan refuses chunking). max_rows None: one call (E1 behaviour)."""
    N = actions.shape[1]
    if not max_rows or N <= max_rows:
        return rollout_q(m, Q, z0v, z0p, actions, all_frames=all_frames)
    Q.chunked = True
    try:
        outs = [rollout_q(m, Q, z0v, z0p, actions[:, s:s + max_rows], all_frames=all_frames)
                for s in range(0, N, int(max_rows))]
    finally:
        Q.chunked = False
    return tuple(torch.cat([o[i] for o in outs], dim=1) for i in range(len(outs[0])))


@torch.no_grad()
def episode_inputs(m, pre, pairs, e):
    """Same as r27lib.episode_inputs: start latent under bf16 autocast, goal latent in float32."""
    o0 = L.trans_obs(pre, pairs['obs0_visual'][e:e + 1], pairs['obs0_proprio'][e:e + 1])
    og = L.trans_obs(pre, pairs['obsg_visual'][e:e + 1], pairs['obsg_proprio'][e:e + 1])
    with torch.autocast('cuda', dtype=torch.bfloat16):
        zs = m.encode_obs(o0)
    gz = m.encode_obs(og)
    return zs['visual'], zs['proprio'], gz['visual'][0, 0], gz['proprio'][0, 0]


def cost(zv, zp, gv, gp, alpha):
    """DINO-WM objective_fn_last: visual MSE + alpha * proprio-embedding MSE. zv (N,P,C), zp (N,Dp)."""
    V = ((zv - gv[None]) ** 2).mean((1, 2))
    Qc = ((zp - gp[None]) ** 2).mean(-1)
    return V + alpha * Qc, V, Qc


def _step_errors(qva, qpa, rva, rpa, alpha):
    """Per-episode planning-time errors at horizon steps ERR_STEPS, in cost units (visual MSE + alpha proprio MSE between
    the quantized and the unquantized prediction of the same candidate, averaged over the candidates). qva/rva
    (E,N,H,P,C) bf16, qpa/rpa (E,N,H,Dp). Returns {step: tensor (E,)}."""
    out = {}
    for s in ERR_STEPS:
        h = s - 1
        ev = ((qva[:, :, h].float() - rva[:, :, h].float()) ** 2).mean((-1, -2))
        ep = ((qpa[:, :, h].float() - rpa[:, :, h].float()) ** 2).mean(-1)
        out[s] = (ev + alpha * ep).mean(1)
    return out


def _pred_var(rva, rpa, alpha):
    """Per-episode variance of the unquantized predictions across the candidates (normalizer of rel errors, as the
    pilot's calib13.pred_var). Returns {step: tensor (E,)}."""
    out = {}
    for s in ERR_STEPS:
        h = s - 1
        v = rva[:, :, h].float(); p = rpa[:, :, h].float()
        out[s] = ((v - v.mean(1, keepdim=True)) ** 2).mean((1, 2, 3)) + alpha * ((p - p.mean(1, keepdim=True)) ** 2).mean((1, 2))
    return out


# ------------------------------------------------------------------------------------------------ CEM
def _clone_gen(g):
    c = torch.Generator()
    c.set_state(g.get_state())
    return c


def run_cem(m, Q, inputs, gens, alpha, opt_steps, ep_ids, var, k, salt, diag=True, n_err_draws=3, n_samples=None,
            topk=None, max_rows=None):
    """Batched CEM with DINO-WM CEMPlanner semantics (300 samples, top-30, var_scale 1, first sample = mu, open loop)
    for E episodes under one variant. Candidate sampling uses the per-episode CPU generators, so every variant sees
    the same random stream (identical iteration-0 population). With diag (all after/before the planning draws, never
    changing them):
      - unquantized rollout of the iteration-0 population -> J_none(it0) and the reference predictions
      - planning-time errors at horizon steps 1/3/5 vs that reference: the planning draw of iteration 0 plus, for the
        randomized variants (RAND), n_err_draws - 1 extra dither draws (salts '<salt>|errdraw<j>')
      - unquantized cost J_none of the last population (as R2-13) and of the final plan mean (elite mean)
      - variant cost of the final plan mean (shared tables: the plan's; fresh dithers: one more draw, it = opt_steps)
    n_samples / topk override the design values (300 / 30) for planner sweeps (E2); None keeps them.
    Returns one dict per episode."""
    E = len(inputs)
    H, N, A = L.GOAL_H, int(n_samples or L.N_SAMPLES), m.action_encoder.patch_embed.in_channels
    TOPK = int(topk or L.TOPK)
    z0v = torch.cat([x[0] for x in inputs]); z0p = torch.cat([x[1] for x in inputs])
    out = [dict(pops={}, best=[]) for _ in range(E)]
    quant = var not in ('none', 'nonealt')
    diag_q = diag and quant          # the unquantized runs need no reference rollout (their errors are 0)
    ref = None
    if diag_q:
        a0 = torch.stack([torch.randn(N, H, A, generator=_clone_gen(gens[kk])).cuda() for kk in range(E)])
        a0[:, 0] = 0.0
        Q.configure('none')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            rv, rp, rva, rpa = rollout_chunks(m, Q, z0v, z0p, a0, all_frames=True, max_rows=max_rows)
        rv, rp = rv.float(), rp.float()
        pv = _pred_var(rva, rpa, alpha)
        for kk in range(E):
            J, _, _ = cost(rv[kk], rp[kk], inputs[kk][2], inputs[kk][3], alpha)
            out[kk]['J_none_it0'] = J.cpu().numpy()
            out[kk]['pred_var'] = {s: float(pv[s][kk]) for s in ERR_STEPS}
            out[kk]['err_draws'] = {s: [] for s in ERR_STEPS}
        ref = (rva, rpa)
        if var in RAND:
            for j in range(1, n_err_draws):
                Q.configure(var, k); Q.begin_plan(ep_ids, f'{salt}|errdraw{j}')
                Q.it = 0
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    _, _, qva, qpa = rollout_chunks(m, Q, z0v, z0p, a0, all_frames=True, max_rows=max_rows)
                er = _step_errors(qva, qpa, rva, rpa, alpha)
                del qva, qpa
                for kk in range(E):
                    for s in ERR_STEPS:
                        out[kk]['err_draws'][s].append(float(er[s][kk]))
        del rv, rp
    mu = torch.zeros(E, H, A, device='cuda'); sigma = L.VAR_SCALE * torch.ones(E, H, A, device='cuda')
    Q.configure(var, k)
    Q.begin_plan(ep_ids, salt)
    last = opt_steps - 1
    last_actions = None
    for i in range(opt_steps):
        action = torch.stack([torch.randn(N, H, A, generator=gens[kk]).cuda() * sigma[kk] + mu[kk] for kk in range(E)])
        action[:, 0] = mu
        Q.it = i
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if i == 0 and diag_q:
                assert torch.equal(action, a0), 'iteration-0 population differs from the preview'
                zv, zp, qva, qpa = rollout_chunks(m, Q, z0v, z0p, action, all_frames=True, max_rows=max_rows)
            else:
                zv, zp = rollout_chunks(m, Q, z0v, z0p, action, max_rows=max_rows)
        if i == 0 and diag_q:
            er = _step_errors(qva, qpa, ref[0], ref[1], alpha)
            for kk in range(E):
                for s in ERR_STEPS:
                    out[kk]['err_draws'][s].insert(0, float(er[s][kk]))   # planning draw first
            del qva, qpa
            ref = None
        zv, zp = zv.float(), zp.float()
        for kk in range(E):
            J, V, Qc = cost(zv[kk], zp[kk], inputs[kk][2], inputs[kk][3], alpha)
            idx = torch.argsort(J)[:TOPK]
            out[kk]['best'].append(float(J[idx[0]]))
            if i in (0, last):
                out[kk]['pops'][i] = dict(actions=action[kk].cpu().numpy().astype(np.float32),
                                          J=J.cpu().numpy(), V=V.cpu().numpy(), Q=Qc.cpu().numpy())
            el = action[kk][idx]
            mu[kk] = el.mean(0); sigma[kk] = el.std(0)
        if i == last:
            last_actions = action
    if diag:
        mf = mu[:, None].clone()                          # final plan mean (E,1,H,A)
        if quant:                                         # variant cost of the final plan (still configured)
            Q.it = opt_steps
            with torch.autocast('cuda', dtype=torch.bfloat16):
                fv, fp = rollout_q(m, Q, z0v, z0p, mf)
            fv, fp = fv.float(), fp.float()
            for kk in range(E):
                J, _, _ = cost(fv[kk], fp[kk], inputs[kk][2], inputs[kk][3], alpha)
                out[kk]['J_v_mf'] = float(J[0])
        Q.configure('none')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            fv, fp = rollout_q(m, Q, z0v, z0p, mf)
            lv, lp = rollout_chunks(m, Q, z0v, z0p, last_actions, max_rows=max_rows) if quant else (None, None)
        fv, fp = fv.float(), fp.float()
        for kk in range(E):
            J, _, _ = cost(fv[kk], fp[kk], inputs[kk][2], inputs[kk][3], alpha)
            out[kk]['J_none_mf'] = float(J[0])
            if not quant:
                out[kk]['J_v_mf'] = float(J[0])
                out[kk]['pops'][last]['J_none'] = out[kk]['pops'][last]['J']
                out[kk]['J_none_it0'] = out[kk]['pops'][0]['J']
            else:
                Jl, _, _ = cost(lv[kk].float(), lp[kk].float(), inputs[kk][2], inputs[kk][3], alpha)
                out[kk]['pops'][last]['J_none'] = Jl.cpu().numpy()
    Q.configure('none')
    for kk in range(E):
        out[kk]['mu_final'] = mu[kk].cpu().numpy().copy()
    return out


# ------------------------------------------------------------------------------------------------ dataset windows
def dataset_windows(T, env, split, n, seed, n_frames, n_act=5):
    """n windows along dataset trajectories of the given split ('train' or 'valid'), rendered by the simulator from the
    dataset states with the dataset actions: frames 0, fs, ..., (n_frames-1)*fs after a random offset, and the next
    n_act model actions (each = fs consecutive normalized low-level actions). Returns numpy arrays
    visual (n, n_frames, H, W, C), proprio (n, n_frames, D), act (n, n_act, fs*A). (Unchanged from qlib13.)"""
    task = T['task']
    dset = T['dset'] if split == 'valid' else T['dset_train']
    pre = T['pre']
    fs = FRAMESKIP
    rng = random.Random(seed)
    need = fs * max(n_act, n_frames - 1) + 1
    out = dict(visual=[], proprio=[], act=[], traj=[])
    for j in range(n):
        while True:
            ti = rng.randint(0, len(dset) - 1)
            base, bi = L._base_traj(dset, ti)
            Lq = base.get_seq_length(bi)
            if Lq - need >= 0:
                break
        off = rng.randint(0, Lq - need)
        env.update_env(L.env_info_of(task, base, bi))
        acts = base.actions[bi, off:off + fs * n_act]                    # normalized (fs*n_act, A)
        init_state = base.states[bi][off].numpy()
        es = seed * 100000 + j
        if n_frames > 1:
            exec_acts = pre.denormalize_actions(acts[:fs * (n_frames - 1)]).numpy()
            obses, _ = env.rollout(es, init_state, exec_acts)
            vis = obses['visual'][::fs][:n_frames]
            pro = obses['proprio'][::fs][:n_frames]
        else:
            o, _ = env.prepare(es, init_state)
            vis, pro = o['visual'][None], o['proprio'][None]
        vis = vis.numpy() if torch.is_tensor(vis) else np.asarray(vis)
        pro = pro.numpy() if torch.is_tensor(pro) else np.asarray(pro)
        out['visual'].append(vis); out['proprio'].append(pro.astype(np.float32))
        out['act'].append(acts.reshape(n_act, -1).numpy().astype(np.float32))
        out['traj'].append((int(bi), int(off)))
    for k in ('visual', 'proprio', 'act'):
        out[k] = np.stack(out[k])
    return out


@torch.no_grad()
def encode_windows(m, pre, W, idx):
    """bf16 autocast encodings of window frames idx: visual (b,t,P,C), proprio (b,t,Dp)."""
    o = pre.transform_obs({'visual': W['visual'][idx], 'proprio': W['proprio'][idx]})
    o = {k: v.float().cuda() for k, v in o.items()}
    with torch.autocast('cuda', dtype=torch.bfloat16):
        zs = m.encode_obs(o)
    return zs['visual'], zs['proprio']


@torch.no_grad()
def window_predict(m, zv, zp, act):
    """One predictor call on a window of t frames (t = num_hist): prediction of the next frame from the last position.
    zv (b,t,P,C), zp (b,t,Dp), act (b,t,A) normalized. Returns visual (b,P,C), proprio (b,Dp) float32."""
    with torch.autocast('cuda', dtype=torch.bfloat16):
        z = _tile(m, zv, zp, m.encode_act(act))
        z_pred = m.predict(z)
        zo, _ = m.separate_emb(z_pred[:, -1:])
    return zo['visual'][:, 0].float(), zo['proprio'][:, 0].float()


# ------------------------------------------------------------------------------------------------ pairs cache
def get_pairs(T, env, split, cache_dir):
    """Start/goal pairs of a split (L.sample_pairs), cached as npz + json so that every job of a split uses identical
    pairs. Returns the pairs dict of L.sample_pairs."""
    task = T['task']
    sp = SPLITS[split]
    os.makedirs(cache_dir, exist_ok=True)
    fz, fj = f'{cache_dir}/pairs_{task}_{split}.npz', f'{cache_dir}/pairs_{task}_{split}.json'
    if os.path.exists(fz) and os.path.exists(fj):
        z = np.load(fz)
        meta = json.load(open(fj))
        pairs = {k: z[k] for k in z.files}
        pairs['env_info'] = meta['env_info']; pairs['eval_seed'] = meta['eval_seed']; pairs['traj'] = meta['traj']
        return pairs
    extra = {}
    if sp.get('sampler') == 'constrained':
        hard = [seg for name, n in sp['hard'] for seg in L.pair_segments(T, n, SPLITS[name]['pair_seed'])]
        soft = [seg for name, n in sp['soft'] for seg in L.pair_segments(T, n, SPLITS[name]['pair_seed'])]
        pairs = L.sample_pairs_constrained(T, env, sp['n'], sp['pair_seed'], hard, soft)
        extra = dict(sampler='constrained', hard=[list(x) for x in sp['hard']], soft=[list(x) for x in sp['soft']],
                     sampler_stats=pairs['sampler_stats'])
    else:
        pairs = L.sample_pairs(T, env, sp['n'], seed=sp['pair_seed'])
    tmp = f'{fz}.tmp{os.getpid()}.npz'
    np.savez_compressed(tmp, **{k: pairs[k] for k in ('obs0_visual', 'obs0_proprio', 'obsg_visual', 'obsg_proprio',
                                                       'state0', 'stateg')})
    os.replace(tmp, fz)
    L.jdump(dict(task=task, split=split, pair_seed=sp['pair_seed'], n=sp['n'], plan_seed=sp['plan_seed'],
                 env_info=pairs['env_info'], eval_seed=pairs['eval_seed'], traj=pairs['traj'], **extra), fj)
    return pairs
