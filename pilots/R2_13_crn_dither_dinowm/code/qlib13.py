"""R2-13 library: k-bit quantization of the DINO-WM predictor MLP hidden activations (GELU outputs of all 6 blocks) with
five rounding variants that differ only in how the dither is shared, batched CEM with population recording, and
dataset windows rendered by the simulator (for range calibration and prediction errors). Builds on r21lib (R2-1/R2-7).
Runs on the pod only.

Variants (same per-layer k-bit uniform grid [lo, hi], lo = min, hi = 99.99th percentile of dataset activations):
  none     unquantized
  det      deterministic rounding (round to nearest)
  ind      stochastic rounding floor(x + u), u independent per entry (per candidate, token, unit), fresh every forward
  shiter   u shared by the 300 candidates of one episode (independent across tokens and units), fresh every forward
           call, i.e. fresh at every CEM iteration (and at every horizon step inside an iteration)
  shfixed  u shared by the 300 candidates; one table per (layer, horizon step) drawn once per plan (episode) and reused
           in all opt_steps CEM iterations of that plan
"""
import os, sys, json, time, zlib, random
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r21lib as L

RES = '/workspace/results'
POPS = '/workspace/pops'
VARIANTS = ('none', 'det', 'ind', 'shiter', 'shfixed')
QVARS = ('det', 'ind', 'shiter', 'shfixed')
SPLITS = {'held': dict(pair_seed=3, n=20, plan_seed=1000), 'eval': dict(pair_seed=4, n=50, plan_seed=2000)}
FRAMESKIP = 5


def seed_of(*parts):
    return zlib.crc32(repr(parts).encode()) & 0x7fffffff


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


_CH = 64   # eager fallback: rows per chunk (bounds the float32 temporaries)


def _chunked(fn):
    def run(x, *args):
        out = torch.empty_like(x)
        for s in range(0, x.shape[0], _CH):
            out[s:s + _CH] = fn(x[s:s + _CH], *args)
        return out
    return run


def _qsh_eager(x4, u4, lo, d, nl):
    out = torch.empty_like(x4)
    for g in range(x4.shape[0]):
        for s in range(0, x4.shape[1], _CH):
            out[g, s:s + _CH] = _qsh(x4[g, s:s + _CH][None], u4[g][None], lo, d, nl)[0]
    return out


def make_kernels(use_compile=True):
    if use_compile:
        kw = dict(dynamic=True, fullgraph=True)
        return torch.compile(_qdet, **kw), torch.compile(_qind, **kw), torch.compile(_qsh, **kw)
    return _chunked(_qdet), _chunked(_qind), _qsh_eager


class Quantizer:
    """Forward hooks on the GELU of the 6 predictor MLPs (blk[1].net[2], as R2-7's MLPAblator).
    mode None: off; 'record': keep the GELU outputs (calibration); 'quant': k-bit quantization with variant self.var.
    Rows of a forward call are grouped as (G episodes) x (N candidates); shared dithers are shared inside a group."""

    def __init__(self, m, use_compile=True):
        self.mods = [blk[1].net[2] for blk in m.predictor.transformer.layers]
        self.handles = [mod.register_forward_hook(self._hook(li)) for li, mod in enumerate(self.mods)]
        self.mode, self.var, self.k = None, 'none', 0
        self.ranges = None
        self.ep_ids, self.salt, self.it, self.h = [0], 'x', 0, 0
        self.kdet, self.kind, self.ksh = make_kernels(use_compile)
        self.rec = None
        self.n_calls = 0

    # ---- configuration
    def set_ranges(self, ranges):
        self.ranges = [(float(lo), float(hi)) for lo, hi in ranges]

    def configure(self, var, k=0):
        self.var, self.k = var, int(k)
        if var == 'none':
            self.mode = None
            return
        assert var in QVARS and self.ranges is not None
        nl = (1 << self.k) - 1
        self.mode = 'quant'
        self.nl_t = torch.tensor(float(nl), device='cuda')
        self.lo_t = [torch.tensor(lo, device='cuda') for lo, hi in self.ranges]
        self.d_t = [torch.tensor((hi - lo) / nl, device='cuda') for lo, hi in self.ranges]

    def begin_plan(self, ep_ids, salt):
        """ep_ids: one id per row group (episode); salt: run identifier (split name)."""
        self.ep_ids, self.salt = [int(e) for e in ep_ids], str(salt)
        self.it, self.h = 0, 0
        if self.var == 'ind':   # in-kernel RNG of the compiled kernel draws from the default CUDA generator
            torch.cuda.manual_seed(seed_of(self.salt, 'ind', *self.ep_ids))

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

    # ---- shared dither
    def _shared_u(self, li, Lt, C):
        G = len(self.ep_ids)
        u = torch.empty((G, 1, Lt, C), device='cuda', dtype=torch.float32)
        for g, e in enumerate(self.ep_ids):
            if self.var == 'shiter':
                s = seed_of(self.salt, 'shiter', e, self.it, self.h, li)
            else:   # shfixed: no iteration index -> the same table in every CEM iteration of the plan
                s = seed_of(self.salt, 'shfixed', e, self.h, li)
            gen = torch.Generator(device='cuda')
            gen.manual_seed(s)
            u[g, 0] = torch.rand((Lt, C), generator=gen, device='cuda')
        return u

    def _hook(self, li):
        def hook(module, inp, out):
            if self.mode is None:
                return None
            if self.mode == 'record':
                self.rec[li].append(out.detach().reshape(-1, out.shape[-1]))
                return None
            self.n_calls += 1
            lo, d, nl = self.lo_t[li], self.d_t[li], self.nl_t
            if self.var == 'det':
                return self.kdet(out, lo, d, nl)
            if self.var == 'ind':
                return self.kind(out, lo, d, nl)
            B, Lt, C = out.shape
            G = len(self.ep_ids)
            assert B % G == 0, (B, G)
            u = self._shared_u(li, Lt, C)
            return self.ksh(out.reshape(G, B // G, Lt, C), u, lo, d, nl).reshape(B, Lt, C)
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


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


# ------------------------------------------------------------------------------------------------ CEM
def run_cem(m, Q, inputs, gens, alpha, opt_steps, ep_ids, var, k, salt, rec=True):
    """Batched CEM with DINO-WM CEMPlanner semantics (300 samples, top-30, var_scale 1, first sample = mu, open loop)
    for E episodes under one variant. Candidate sampling uses the per-episode CPU generators, so every variant sees
    the same random stream. Records the populations of the first and the last iteration (actions, J) and, for the
    last one, the unquantized cost J_none of the same candidates."""
    E = len(inputs)
    H, N, A = L.GOAL_H, L.N_SAMPLES, m.action_encoder.patch_embed.in_channels
    z0v = torch.cat([x[0] for x in inputs]); z0p = torch.cat([x[1] for x in inputs])
    mu = torch.zeros(E, H, A, device='cuda'); sigma = L.VAR_SCALE * torch.ones(E, H, A, device='cuda')
    Q.configure(var, k)
    Q.begin_plan(ep_ids, salt)
    last = opt_steps - 1
    out = [dict(pops={}, best=[]) for _ in range(E)]
    for i in range(opt_steps):
        action = torch.stack([torch.randn(N, H, A, generator=gens[kk]).cuda() * sigma[kk] + mu[kk] for kk in range(E)])
        action[:, 0] = mu
        Q.it = i
        with torch.autocast('cuda', dtype=torch.bfloat16):
            zv, zp = rollout_q(m, Q, z0v, z0p, action)
        zv, zp = zv.float(), zp.float()
        for kk in range(E):
            J, V, Qc = cost(zv[kk], zp[kk], inputs[kk][2], inputs[kk][3], alpha)
            idx = torch.argsort(J)[:L.TOPK]
            out[kk]['best'].append(float(J[idx[0]]))
            if rec and i in (0, last):
                out[kk]['pops'][i] = dict(actions=action[kk].cpu().numpy().astype(np.float32),
                                          J=J.cpu().numpy(), V=V.cpu().numpy(), Q=Qc.cpu().numpy())
            el = action[kk][idx]
            mu[kk] = el.mean(0); sigma[kk] = el.std(0)
    if rec and var != 'none':
        acts = torch.stack([torch.as_tensor(out[kk]['pops'][last]['actions']).cuda() for kk in range(E)])
        Q.configure('none')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            zv, zp = rollout_q(m, Q, z0v, z0p, acts)
        zv, zp = zv.float(), zp.float()
        for kk in range(E):
            J, _, _ = cost(zv[kk], zp[kk], inputs[kk][2], inputs[kk][3], alpha)
            out[kk]['pops'][last]['J_none'] = J.cpu().numpy()
    Q.configure('none')
    for kk in range(E):
        out[kk]['mu_final'] = mu[kk].cpu().numpy().copy()
    return out


# ------------------------------------------------------------------------------------------------ dataset windows
def dataset_windows(T, env, split, n, seed, n_frames, n_act=5):
    """n windows along dataset trajectories of the given split ('train' or 'valid'), rendered by the simulator from the
    dataset states with the dataset actions: frames 0, fs, ..., (n_frames-1)*fs after a random offset, and the next
    n_act model actions (each = fs consecutive normalized low-level actions). Returns numpy arrays
    visual (n, n_frames, H, W, C), proprio (n, n_frames, D), act (n, n_act, fs*A)."""
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
        if task == 'pusht':
            info = {'shape': base.shapes[bi]}
        else:
            info = {'fix_door_location': base.door_locations[bi, 0], 'fix_wall_location': base.wall_locations[bi, 0]}
        env.update_env(info)
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
