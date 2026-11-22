"""DINO-WM shared library for paper A main experiments (server version of the pilot r21lib.py, R2-1/R2-7/R2-13).
Load DINO-WM (PushT, Wall; PointMaze only if mujoco_py + d4rl import), sample start/goal pairs, simulator
ground-truth costs. Only the paths differ from the pilot copy (server user space instead of /root, /workspace);
every function body that the pilots used is unchanged. Runs on the server only."""
import os, sys, types, json, time, math, random
import numpy as np
import torch
import torch.nn.functional as F

_H = os.path.expanduser('~')
MAIN = os.environ.get('DINOWM_MAIN', f'{_H}/pmx/A/dinowm_main')
DINO_WM = os.environ.get('DINOWM_REPO', f'{MAIN}/dino_wm')          # gaoyuezhou/dino_wm @ 0a9492f
CKPT_BASE = os.environ.get('DINOWM_CKPT', f'{_H}/pmx/data/dinowm')  # holds outputs/{pusht,wall_single,point_maze}
DS_DIR = os.environ.get('DINOWM_DS', f'{MAIN}/ds')                  # holds symlinks pusht_noise, wall_single
RES = f'{MAIN}/results'
TASKS = {
    # model dir, planning-config values copied from conf/plan_<task>.yaml
    'pusht': dict(model='pusht', goal_source='dset', alpha=1.0, opt_steps=30, plan_seed=99),
    'wall': dict(model='wall_single', goal_source='random_state', alpha=1.0, opt_steps=10, plan_seed=99),
    'pointmaze': dict(model='point_maze', goal_source='random_state', alpha=0.0, opt_steps=10, plan_seed=99),
}
GOAL_H = 5
N_SAMPLES, TOPK, VAR_SCALE = 300, 30, 1.0
WITH_POINTMAZE = os.environ.get('DINOWM_POINTMAZE', '0') == '1'


def hub_dir():
    return os.path.join(torch.hub.get_dir(), 'facebookresearch_dinov2_main')


def setup_paths():
    if DINO_WM not in sys.path:
        sys.path.insert(0, DINO_WM)
    os.environ['DATASET_DIR'] = DS_DIR
    # env/__init__.py imports env.pointmaze (needs mujoco_py + d4rl): stub it unless PointMaze is enabled
    if not WITH_POINTMAZE and 'env.pointmaze' not in sys.modules:
        pm = types.ModuleType('env.pointmaze')
        pm.U_MAZE = None
        pm.PointMazeWrapper = None
        sys.modules['env.pointmaze'] = pm
    hub = hub_dir()
    if os.path.isdir(hub) and hub not in sys.path:
        sys.path.insert(0, hub)   # needed to unpickle the encoder
    import env  # noqa: F401  registers pusht / wall (/ point_maze)


# ----------------------------------------------------------------------------------------------------------
# exact-math attention replacement (SDPA with the same frame-causal boolean mask); verified against the original
def _sdpa_attention_forward(self, x):
    from einops import rearrange
    B, T, C = x.size()
    x = self.norm(x)
    qkv = self.to_qkv(x).chunk(3, dim=-1)
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
    mask = (self.bias[0, 0, :T, :T] != 0)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, scale=self.scale)
    out = rearrange(out, 'b h n d -> b n (h d)')
    return self.to_out(out)


def patch_attention(use_sdpa=True):
    setup_paths()
    from models import vit
    if use_sdpa:
        if not hasattr(vit.Attention, '_orig_forward'):
            vit.Attention._orig_forward = vit.Attention.forward
        vit.Attention.forward = _sdpa_attention_forward
    elif hasattr(vit.Attention, '_orig_forward'):
        vit.Attention.forward = vit.Attention._orig_forward


# ----------------------------------------------------------------------------------------------------------
def load_task(task, device='cuda', load_dset=True):
    """Returns dict(cfg, model, dset (valid traj dataset), pre (Preprocessor))."""
    setup_paths()
    import hydra
    from omegaconf import OmegaConf
    from preprocessor import Preprocessor
    if not os.path.isdir(hub_dir()):
        torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        setup_paths()
    mdir = f"{CKPT_BASE}/outputs/{TASKS[task]['model']}"
    cfg = OmegaConf.load(f'{mdir}/hydra.yaml')
    payload = torch.load(f'{mdir}/checkpoints/model_latest.pth', map_location=device, weights_only=False)
    res = {k: payload[k].to(device) for k in ['encoder', 'predictor', 'proprio_encoder', 'action_encoder'] if k in payload}
    train_flags = {k: bool(v.training) for k, v in res.items()}
    train_flags['payload_keys'] = sorted(payload.keys())
    if 'encoder' not in res:   # frozen DINOv2 is not stored in the checkpoint (same as plan.load_model)
        res['encoder'] = hydra.utils.instantiate(cfg.encoder).to(device)
    model = hydra.utils.instantiate(
        cfg.model, encoder=res['encoder'], proprio_encoder=res['proprio_encoder'],
        action_encoder=res['action_encoder'], predictor=res['predictor'], decoder=None,
        proprio_dim=cfg.proprio_emb_dim, action_dim=cfg.action_emb_dim, concat_dim=cfg.concat_dim,
        num_action_repeat=cfg.num_action_repeat, num_proprio_repeat=cfg.num_proprio_repeat)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    out = dict(cfg=cfg, model=model, task=task, ckpt_train_flags=train_flags, epoch=payload.get('epoch'))
    if load_dset:
        _, tdset = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist, num_pred=cfg.num_pred,
                                    frameskip=cfg.frameskip)
        dset = tdset['valid']
        out['dset'] = dset
        out['dset_train'] = tdset['train']
        out['pre'] = Preprocessor(action_mean=dset.action_mean, action_std=dset.action_std,
                                  state_mean=dset.state_mean, state_std=dset.state_std,
                                  proprio_mean=dset.proprio_mean, proprio_std=dset.proprio_std,
                                  transform=dset.transform)
    return out


def make_env(task):
    setup_paths()
    import gym
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(f"{CKPT_BASE}/outputs/{TASKS[task]['model']}/hydra.yaml")
    return gym.make(cfg.env.name, *cfg.env.args, **cfg.env.kwargs).unwrapped


# ----------------------------------------------------------------------------------------------------------
def _base_traj(dset, i):
    """Underlying (dataset, index) for a valid-split trajectory (Wall uses a TrajSubset)."""
    if hasattr(dset, 'indices'):
        return dset.dataset, dset.indices[i]
    return dset, i


def env_info_of(task, base, bi):
    """Per-trajectory env settings used by update_env (PushT: block shape; Wall: door/wall positions; PointMaze: none)."""
    if task == 'pusht':
        return {'shape': base.shapes[bi]}
    if task == 'wall':
        return {'fix_door_location': base.door_locations[bi, 0], 'fix_wall_location': base.wall_locations[bi, 0]}
    return {}


def sample_pairs(T, env, n, seed, frameskip=5):
    """Start/goal pairs following PlanWorkspace.prepare_targets (goal_source from the task's plan config).
    Returns dict of numpy arrays: obs0_visual (n,H,W,C), obs0_proprio, obsg_visual, obsg_proprio, state0, stateg,
    env_info list, eval_seed list."""
    task = T['task']
    dset, pre = T['dset'], T['pre']
    rng = random.Random(seed)
    eval_seed = [seed * 1000 + j + 1 for j in range(n)]
    out = dict(obs0_visual=[], obs0_proprio=[], obsg_visual=[], obsg_proprio=[], state0=[], stateg=[],
               env_info=[], eval_seed=eval_seed, traj=[])
    traj_len = frameskip * GOAL_H + 1
    for j in range(n):
        base, bi = None, None
        while True:
            ti = rng.randint(0, len(dset) - 1)
            base, bi = _base_traj(dset, ti)
            L = base.get_seq_length(bi)
            if L - traj_len >= 0:
                break
        offset = rng.randint(0, L - traj_len)
        info = env_info_of(task, base, bi)
        env.update_env(info)
        if TASKS[task]['goal_source'] == 'dset':
            states = base.states[bi].numpy()
            acts = base.actions[bi, offset:offset + frameskip * GOAL_H]          # normalized
            exec_acts = pre.denormalize_actions(acts).numpy()
            init_state = states[offset]
            obses, sts = env.rollout(eval_seed[j], init_state, exec_acts)
            o0 = {k: v[0] for k, v in obses.items()}
            og = {k: v[-1] for k, v in obses.items()}
            s0, sg = init_state, sts[-1]
            out['traj'].append((int(bi), int(offset)))
        else:  # random_state (Wall)
            s0, sg = env.sample_random_init_goal_states(eval_seed[j])
            o0, _ = env.prepare(eval_seed[j], s0)
            og, _ = env.prepare(eval_seed[j], sg)
            out['traj'].append((int(bi), -1))
        for k_src, o in (('obs0', o0), ('obsg', og)):
            vis = o['visual']
            vis = vis.numpy() if torch.is_tensor(vis) else np.asarray(vis)
            pro = o['proprio']
            pro = pro.numpy() if torch.is_tensor(pro) else np.asarray(pro)
            out[f'{k_src}_visual'].append(vis)
            out[f'{k_src}_proprio'].append(pro.astype(np.float32))
        out['state0'].append(np.asarray(s0, dtype=np.float64))
        out['stateg'].append(np.asarray(sg, dtype=np.float64))
        out['env_info'].append({k: (v.item() if torch.is_tensor(v) else v) for k, v in info.items()})
    for k in ['obs0_visual', 'obs0_proprio', 'obsg_visual', 'obsg_proprio', 'state0', 'stateg']:
        out[k] = np.stack(out[k])
    return out


def pair_segments(T, n, seed, frameskip=5):
    """(trajectory, offset) of the first n pairs that sample_pairs(T, env, n, seed) draws for a dset goal source (PushT),
    without rendering: the same random.Random(seed) draws in the same order."""
    dset = T['dset']
    rng = random.Random(seed)
    traj_len = frameskip * GOAL_H + 1
    out = []
    for j in range(n):
        while True:
            ti = rng.randint(0, len(dset) - 1)
            base, bi = _base_traj(dset, ti)
            L = base.get_seq_length(bi)
            if L - traj_len >= 0:
                break
        out.append((int(bi), int(rng.randint(0, L - traj_len))))
    return out


def shared_frames(o1, o2, traj_len):
    """Dataset frames shared by two segments [o, o + traj_len - 1] of the same trajectory."""
    return max(0, traj_len - abs(int(o1) - int(o2)))


def sample_pairs_constrained(T, env, n, seed, hard, soft, frameskip=5, max_tries=5000):
    """PushT start/goal pairs (goal_source dset) drawn like sample_pairs (uniform validation trajectory, then uniform
    offset, rng = random.Random(seed)) but (1) rejecting every segment that shares a frame with a segment in `hard`,
    (2) rejecting exact duplicates of pairs already chosen, and (3) preferring segments that share no frame with
    `soft`: draws continue until a candidate with zero soft overlap appears; if none appears within max_tries draws the
    candidate with the fewest shared soft frames (first such) is kept. hard/soft: lists of (trajectory, offset).
    Rendering of the start/goal observations is the same as sample_pairs. Returns the sample_pairs dict plus
    'sampler_stats' (per pair: draws, hard rejections, duplicate rejections, soft frames shared)."""
    task = T['task']
    assert TASKS[task]['goal_source'] == 'dset'
    dset, pre = T['dset'], T['pre']
    rng = random.Random(seed)
    eval_seed = [seed * 1000 + j + 1 for j in range(n)]
    traj_len = frameskip * GOAL_H + 1
    out = dict(obs0_visual=[], obs0_proprio=[], obsg_visual=[], obsg_proprio=[], state0=[], stateg=[],
               env_info=[], eval_seed=eval_seed, traj=[], sampler_stats=[])
    chosen = set()
    for j in range(n):
        best, draws, rej_hard, rej_dup = None, 0, 0, 0
        while draws < max_tries:
            draws += 1
            while True:
                ti = rng.randint(0, len(dset) - 1)
                base, bi = _base_traj(dset, ti)
                L = base.get_seq_length(bi)
                if L - traj_len >= 0:
                    break
            offset = rng.randint(0, L - traj_len)
            if any(int(bi) == hb and shared_frames(offset, ho, traj_len) > 0 for hb, ho in hard):
                rej_hard += 1
                continue
            if (int(bi), int(offset)) in chosen:
                rej_dup += 1
                continue
            ov = sum(shared_frames(offset, so, traj_len) for sb, so in soft if sb == int(bi))
            if best is None or ov < best[3]:
                best = (base, int(bi), int(offset), ov)
            if ov == 0:
                break
        assert best is not None, f'no admissible pair after {max_tries} draws (pair {j})'
        base, bi, offset, ov = best
        chosen.add((bi, offset))
        info = env_info_of(task, base, bi)
        env.update_env(info)
        states = base.states[bi].numpy()
        acts = base.actions[bi, offset:offset + frameskip * GOAL_H]          # normalized
        exec_acts = pre.denormalize_actions(acts).numpy()
        init_state = states[offset]
        obses, sts = env.rollout(eval_seed[j], init_state, exec_acts)
        o0 = {k: v[0] for k, v in obses.items()}
        og = {k: v[-1] for k, v in obses.items()}
        for k_src, o in (('obs0', o0), ('obsg', og)):
            vis = o['visual']
            vis = vis.numpy() if torch.is_tensor(vis) else np.asarray(vis)
            pro = o['proprio']
            pro = pro.numpy() if torch.is_tensor(pro) else np.asarray(pro)
            out[f'{k_src}_visual'].append(vis)
            out[f'{k_src}_proprio'].append(pro.astype(np.float32))
        out['state0'].append(np.asarray(init_state, dtype=np.float64))
        out['stateg'].append(np.asarray(sts[-1], dtype=np.float64))
        out['env_info'].append({k: (v.item() if torch.is_tensor(v) else v) for k, v in info.items()})
        out['traj'].append((bi, offset))
        out['sampler_stats'].append(dict(draws=draws, hard_rejections=rej_hard, duplicate_rejections=rej_dup,
                                         soft_shared_frames=ov))
    for k in ['obs0_visual', 'obs0_proprio', 'obsg_visual', 'obsg_proprio', 'state0', 'stateg']:
        out[k] = np.stack(out[k])
    return out


def trans_obs(pre, visual, proprio, device='cuda'):
    """visual (b,H,W,C) uint8-range, proprio (b,D) raw -> model input dict with t=1."""
    o = pre.transform_obs({'visual': visual[:, None], 'proprio': proprio[:, None]})
    return {k: v.float().to(device) for k, v in o.items()}


# ----------------------------------------------------------------------------------------------------------
@torch.no_grad()
def encode_with_prenorm(model, obs):
    """Encoder outputs plus DINOv2 pre-final-LayerNorm patch tokens (for high-norm token selection)."""
    visual = obs['visual']
    b = visual.shape[0]
    v = visual.reshape(-1, *visual.shape[2:])
    v = model.encoder_transform(v)
    ff = model.encoder.base_model.forward_features(v)
    post = ff['x_norm_patchtokens']
    nreg = getattr(model.encoder.base_model, 'num_register_tokens', 0)
    prenorm = ff['x_prenorm'][:, 1 + nreg:]
    return post.reshape(b, -1, *post.shape[1:]), prenorm.reshape(b, -1, *prenorm.shape[1:])


@torch.no_grad()
def rollout_last(model, obs0, actions):
    """obs0: dict with t=1, batch 1 (expanded to N). actions (N, H, A) normalized. Returns last visual (N,P,C) and
    proprio-emb (N,Dp) of the predicted final frame (identical to VWorldModel.rollout)."""
    N = actions.shape[0]
    o = {k: v.expand(N, *v.shape[1:]) for k, v in obs0.items()}
    z_obses, _ = model.rollout(obs_0=o, act=actions)
    return z_obses['visual'][:, -1], z_obses['proprio'][:, -1]


@torch.no_grad()
def decompose(zv, zp, gv, gp, alpha, R=None):
    """zv (N,P,C), zp (N,Dp); gv (P,C), gp (Dp). Returns per-coordinate (N,C) and per-token (N,P) shares of the
    visual MSE (each sums to the visual cost), proprio MSE (N,), and optional rotated-basis per-coordinate (N,C)."""
    N, P, C = zv.shape
    d = zv - gv[None]
    e = d * d
    a = e.sum(1) / (P * C)
    b = e.sum(2) / (P * C)
    jp = ((zp - gp[None]) ** 2).mean(-1)
    out = dict(a=a, b=b, jp=jp)
    if R is not None:
        dr = d.reshape(N * P, C) @ R
        out['a_rot'] = (dr * dr).reshape(N, P, C).sum(1) / (P * C)
    return out


# ----------------------------------------------------------------------------------------------------------
# simulator ground truth (fast replays; verified against env.rollout in the smoke test)
_ENV = {}


def _get_env(task):
    if task not in _ENV:
        _ENV[task] = make_env(task)
    return _ENV[task]


def replay_final_states(task, env_info, seed, init_state, exec_actions, fast=True):
    """exec_actions (N, T, 2) denormalized env actions. Returns final states (N, D)."""
    env = _get_env(task)
    env.update_env(env_info if task == 'pusht' else {k: torch.tensor(v) for k, v in env_info.items()})
    finals = []
    if task == 'pointmaze':
        for seq in exec_actions:
            if fast:      # same physics as env.rollout without rendering (rendering does not touch the sim state)
                env.return_value = 'state'
                env.seed(seed)
                env.set_init_state(np.asarray(init_state))
                env.reset()
                for a in seq:
                    env.step(a)
                finals.append(env._get_obs()['proprio'].astype(np.float64))   # float32 state, as env.rollout
            else:
                _, sts = env.rollout(seed, np.asarray(init_state), seq)
                finals.append(np.asarray(sts[-1], dtype=np.float64))
        return np.stack(finals)
    if task == 'pusht':
        from pymunk.vec2d import Vec2d
        dt = 1.0 / env.sim_hz
        n_sub = env.sim_hz // env.control_hz
        for seq in exec_actions:
            if fast:
                env.seed(seed)
                env._setup()
                env._set_state(np.asarray(init_state))
                env.n_contact_points = 0
                for a in seq:
                    act = np.array(a) * env.action_scale
                    if env.relative:
                        act = env.agent.position + act
                    for _ in range(n_sub):
                        acc = env.k_p * (act - env.agent.position) + env.k_v * (Vec2d(0, 0) - env.agent.velocity)
                        env.agent.velocity += acc * dt
                        env.space.step(dt)
                finals.append(env._get_obs().astype(np.float64))
            else:
                _, sts = env.rollout(seed, np.asarray(init_state), seq)
                finals.append(sts[-1].astype(np.float64))
    else:
        for seq in exec_actions:
            if fast:
                env.seed(seed)
                env.set_init_state(np.asarray(init_state))
                env.reset()
                for a in seq:
                    env.dot_position = env._calculate_next_position(torch.as_tensor(a))
                finals.append(env.dot_position.double().numpy().copy())
            else:
                _, sts = env.rollout(seed, np.asarray(init_state), seq)
                finals.append(np.asarray(sts[-1], dtype=np.float64))
    return np.stack(finals)


def true_costs(task, stateg, finals):
    """Evaluator's own definitions: state_dist (primary), success; plus success-margin (secondary)."""
    env = _get_env(task)
    sd, sc, mg = [], [], []
    for f in finals:
        r = env.eval_state(np.asarray(stateg), f)
        sd.append(float(r['state_dist']))
        sc.append(bool(r['success']))
        if task == 'pusht':
            pos = np.linalg.norm(stateg[:4] - f[:4])
            ang = abs(stateg[4] - f[4])
            ang = min(ang, 2 * np.pi - ang)
            mg.append(max(pos / 20.0, ang / (np.pi / 9)))
        elif task == 'pointmaze':                       # success: |dxy| < 0.5
            mg.append(np.linalg.norm(stateg[:2] - f[:2]) / 0.5)
        else:
            mg.append(np.linalg.norm(stateg[:2] - f[:2]) / 4.5)
    return np.array(sd), np.array(sc), np.array(mg)


def _replay_job(args):
    task, env_info, seed, init_state, exec_actions, stateg = args
    torch.set_num_threads(1)
    setup_paths()
    finals = replay_final_states(task, env_info, seed, init_state, exec_actions)
    sd, sc, mg = true_costs(task, stateg, finals)
    return finals, sd, sc, mg


def to_exec_actions(pre, actions, frameskip=5):
    """normalized (N, H, frameskip*2) -> denormalized env actions (N, H*frameskip, 2)."""
    from einops import rearrange
    a = rearrange(actions.detach().cpu(), 'b t (f d) -> b (t f) d', f=frameskip)
    return pre.denormalize_actions(a).numpy()


# ----------------------------------------------------------------------------------------------------------
def spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(x, y).correlation
    return float(r) if np.isfinite(r) else float('nan')


def jdump(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------------------------
def rot_matrix(C=384, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(C, C, generator=g, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r))[None]
    return q.float()


def visual_cost_variant(zv, gv, variant, k=0, coord_rank=None, w=None, tok_rank=None, coord_idx=None):
    """Visual part of the cost under a variant. zv (N,P,C), gv (P,C). Returns (N,).
    base: mean over p,c; excl_coord: mean over c not in top-k of coord_rank; std: weighted by w (mean 1);
    excl_tok: mean over p not in top-k of tok_rank (per-pair ranking); excl_idx: exclude the given coord indices."""
    e = (zv - gv[None]) ** 2
    N, P, C = e.shape
    if variant == 'base':
        return e.mean((1, 2))
    if variant in ('excl_coord', 'excl_idx'):
        idx = coord_rank[:k] if variant == 'excl_coord' else coord_idx
        keep = torch.ones(C, dtype=torch.bool, device=e.device)
        keep[torch.as_tensor(idx, device=e.device)] = False
        return e[:, :, keep].mean((1, 2))
    if variant == 'std':
        return (e * w[None, None]).mean((1, 2))
    if variant == 'excl_tok':
        keep = torch.ones(P, dtype=torch.bool, device=e.device)
        keep[torch.as_tensor(tok_rank[:k], device=e.device)] = False
        return e[:, keep, :].mean((1, 2))
    raise ValueError(variant)
