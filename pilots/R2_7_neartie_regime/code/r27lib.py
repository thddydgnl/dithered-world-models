"""R2-7 library: batched CEM with ground-truth-free resolution proxies, resolution-aware schedules, population
recording, and predictor-MLP ablation hooks (S4). Builds on r21lib (copied from R2-1). Runs on the pod only."""
import os, sys, json, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r21lib as L

RES = '/workspace/results'
POPS = '/workspace/pops'
STEPS_REC = {'pusht': (0, 4, 9, 14, 19, 24, 29), 'wall': (0, 2, 4, 6, 9)}
MID = {'pusht': 14, 'wall': 4}
N_BOOT = 20


# ------------------------------------------------------------------------------------------------ world model
@torch.no_grad()
def rollout_batched(m, z0v, z0p, actions):
    """Same as VWorldModel.rollout with obs_0 repeated N times (start image encoded once, latents tiled).
    z0v (E,1,P,C), z0p (E,1,Dp), actions (E,N,H,A) -> final visual (E,N,P,C), proprio-emb (E,N,Dp)."""
    E, N, H, A = actions.shape
    act = actions.reshape(E * N, H, A)
    vis = z0v.repeat_interleave(N, 0)
    pro = z0p.repeat_interleave(N, 0)
    act_emb = m.encode_act(act[:, :1])
    P = vis.shape[2]
    z = torch.cat([vis, pro.unsqueeze(2).expand(-1, -1, P, -1).repeat(1, 1, 1, m.num_proprio_repeat),
                   act_emb.unsqueeze(2).expand(-1, -1, P, -1).repeat(1, 1, 1, m.num_action_repeat)], dim=3)
    action = act[:, 1:]
    for t in range(action.shape[1]):
        z_pred = m.predict(z[:, -m.num_hist:])
        z_new = m.replace_actions_from_z(z_pred[:, -1:, ...], action[:, t:t + 1, :])
        z = torch.cat([z, z_new], dim=1)
    z_pred = m.predict(z[:, -m.num_hist:])
    z = torch.cat([z, z_pred[:, -1:, ...]], dim=1)
    z_obses, _ = m.separate_emb(z)
    return (z_obses['visual'][:, -1].reshape(E, N, *z_obses['visual'].shape[2:]),
            z_obses['proprio'][:, -1].reshape(E, N, -1))


@torch.no_grad()
def episode_inputs(m, pre, pairs, e):
    o0 = L.trans_obs(pre, pairs['obs0_visual'][e:e + 1], pairs['obs0_proprio'][e:e + 1])
    og = L.trans_obs(pre, pairs['obsg_visual'][e:e + 1], pairs['obsg_proprio'][e:e + 1])
    with torch.autocast('cuda', dtype=torch.bfloat16):
        zs = m.encode_obs(o0)
    gz = m.encode_obs(og)
    return zs['visual'], zs['proprio'], gz['visual'][0, 0], gz['proprio'][0, 0]


def cost_parts(zv, zp, gv, gp):
    """zv (N,P,C) float, zp (N,Dp). Returns V (N,), Q (N,), per-token visual contributions b (N,P) (sum_p b = V)."""
    N, P, C = zv.shape
    e = (zv - gv[None]) ** 2
    b = e.sum(2) / (P * C)
    V = b.sum(1)
    Q = ((zp - gp[None]) ** 2).mean(-1)
    return V, Q, b


def boot_counts(P=196, B=N_BOOT, seed=12345):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, P, (B, P), generator=g)
    C = torch.zeros(B, P)
    C.scatter_add_(1, idx, torch.ones(B, P))
    return C           # rows sum to P


def _spearman_t(x, y):
    rx = x.argsort().argsort().float(); ry = y.argsort().argsort().float()
    rx = rx - rx.mean(); ry = ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm() + 1e-12))


def proxies(V, Q, b, alpha, Cb):
    """Ground-truth-free resolution proxies of one population.
    R_boot: median over candidates of the patch-bootstrap sd of the centred cost / population sd of the cost.
    boot_rank: mean Spearman between bootstrap costs and the cost. rho_VQ: Spearman(visual, proprio)."""
    J = V + alpha * Q
    Jb = b @ Cb.T.to(b) + alpha * Q[:, None]            # (N, B)
    Jb = Jb - Jb.mean(0, keepdim=True)
    sd_pop = J.std()
    r_boot = float((Jb.std(1).median() / (sd_pop + 1e-12)))
    br = float(np.mean([_spearman_t(Jb[:, k], J) for k in range(min(8, Jb.shape[1]))]))
    return dict(R_boot=r_boot, boot_rank=br, rho_VQ=_spearman_t(V, Q), sd_J=float(sd_pop), sd_V=float(V.std()),
                sd_Q=float((alpha * Q).std()))


# ------------------------------------------------------------------------------------------------ CEM
def run_cem_batch(m, inputs, gens, task, schedule, alpha, opt_steps, rec_steps=(), Cb=None):
    """Batched CEM (DINO-WM CEMPlanner semantics: 300 samples, top-30, var_scale 1, first sample = mu) for E episodes.
    schedule: dict(kind in {'base','fv','ak'}, tau, topk_after). Proxy R_boot is computed at every step.
    Returns per-episode dicts: final mu, mu after each step, per-step proxies, trigger step, recorded populations."""
    E = len(inputs)
    H, N, A = L.GOAL_H, L.N_SAMPLES, m.action_encoder.patch_embed.in_channels
    z0v = torch.cat([x[0] for x in inputs]); z0p = torch.cat([x[1] for x in inputs])
    mu = torch.zeros(E, H, A, device='cuda'); sigma = L.VAR_SCALE * torch.ones(E, H, A, device='cuda')
    out = [dict(mu_traj=[], prox=[], trig=None, pops={}) for _ in range(E)]
    frozen = [None] * E
    for i in range(opt_steps):
        action = torch.stack([torch.randn(N, H, A, generator=gens[k]).cuda() * sigma[k] + mu[k] for k in range(E)])
        action[:, 0] = mu
        with torch.autocast('cuda', dtype=torch.bfloat16):
            zv, zp = rollout_batched(m, z0v, z0p, action)
        zv, zp = zv.float(), zp.float()
        for k in range(E):
            gv, gp = inputs[k][2], inputs[k][3]
            V, Q, b = cost_parts(zv[k], zp[k], gv, gp)
            J = V + alpha * Q
            px = proxies(V, Q, b, alpha, Cb)
            px['step'] = i
            o = out[k]
            if o['trig'] is None and schedule['kind'] != 'base' and i >= 1 and px['R_boot'] > schedule['tau']:
                o['trig'] = i
            kk = L.TOPK
            if schedule['kind'] == 'ak' and o['trig'] is not None:
                kk = schedule.get('topk_after', 90)
            idx = torch.argsort(J)[:kk]
            px['best'] = float(J[idx[0]])
            o['prox'].append(px)
            if i in rec_steps:
                o['pops'][i] = dict(actions=action[k].cpu().numpy().astype(np.float32), J=J.cpu().numpy(),
                                    V=V.cpu().numpy(), Q=Q.cpu().numpy())
            el = action[k][idx]
            new_mu, new_sigma = el.mean(0), el.std(0)
            if schedule['kind'] == 'fv' and o['trig'] is not None:
                if frozen[k] is None:
                    frozen[k] = sigma[k].clone()
                new_sigma = frozen[k]
            mu[k] = new_mu; sigma[k] = new_sigma
            o['mu_traj'].append(mu[k].cpu().numpy().copy())
    for k in range(E):
        out[k]['mu_final'] = mu[k].cpu().numpy().copy()
    return out


# ------------------------------------------------------------------------------------------------ S4 hooks
class MLPAblator:
    """Hooks on the GELU outputs of the 6 predictor MLPs. modes: None, 'meanrep' (weak set |a|<alpha*m_j -> mu_j),
    'dfix' (fixed random unit set U, identical for every row/candidate -> abar_j), 'stats' (collect calibration)."""

    def __init__(self, m):
        self.m = m
        self.mods = [blk[1].net[2] for blk in m.predictor.transformer.layers]
        self.mode = None
        self.handles = [mod.register_forward_hook(self._hook(li)) for li, mod in enumerate(self.mods)]
        self.alpha = 0.2
        self.mvals = None; self.mu = None; self.abar = None; self.U = None
        self.acc = None

    def _hook(self, li):
        def hook(module, inp, out):
            if self.mode is None:
                return None
            if self.mode == 'stats':
                a = out.detach().float().reshape(-1, out.shape[-1])
                self.acc[li].append(a[torch.randperm(a.shape[0], device=a.device)[:256]].cpu())
                return None
            if self.mode == 'meanrep':
                weak = out.abs() < (self.alpha * self.mvals[li]).to(out.dtype)
                return torch.where(weak, self.mu[li].to(out.dtype).expand_as(out), out)
            if self.mode == 'dfix':
                o = out.clone()
                o[..., self.U[li]] = self.abar[li][self.U[li]].to(out.dtype)
                return o
            return None
        return hook

    def start_stats(self):
        self.mode = 'stats'; self.acc = [[] for _ in self.mods]

    def finish_stats(self, alpha):
        self.mode = None
        self.alpha = alpha
        self.mvals, self.mu, self.abar, self.weak_frac = [], [], [], []
        for li in range(len(self.mods)):
            a = torch.cat(self.acc[li])                                   # (n, 2048)
            kq = max(1, int(np.ceil(0.999 * a.shape[0])))
            mj = a.abs().kthvalue(kq, dim=0).values                       # 99.9th percentile (quantile() size limit)
            weak = a.abs() < alpha * mj
            mu = (a * weak).sum(0) / weak.sum(0).clamp(min=1)
            self.mvals.append(mj.cuda()); self.mu.append(mu.cuda()); self.abar.append(a.mean(0).cuda())
            self.weak_frac.append(float(weak.float().mean()))
        self.acc = None

    def set_dfix(self, frac, seed=777):
        g = torch.Generator().manual_seed(seed)
        self.U = []
        for li, mod in enumerate(self.mods):
            n = self.abar[li].shape[0]
            perm = torch.randperm(n, generator=g)
            self.U.append(perm[:max(1, int(round(frac * n)))].cuda())
        self.mode = 'dfix'

    def remove(self):
        for h in self.handles:
            h.remove()
