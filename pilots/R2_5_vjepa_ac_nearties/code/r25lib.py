"""Shared code for R2-5 (near-ties in V-JEPA 2-AC). Runs on the pod only.

Official recipe (vjepa2 repo, notebooks/utils/world_model_wrapper.py + mpc_utils.py, app/vjepa_droid):
- encoder: every frame alone as a 2-frame tubelet (256 tokens), then F.layer_norm over the feature dim
- predictor: per-frame tokens [action, state, 256 patches], frame-causal; next latent = layer_norm(pred(...)[:, -256:])
- rollout (cem with rollout=2): append the predicted latent, the action and compute_new_pose(state, action), call again
- energy: l1 = mean |final predicted latent - goal latent| over all tokens and dims (objective(final.flatten(1), goal.flatten(1)))
- actions: poses_to_diffs (xyz difference, relative rotation as euler xyz, gripper difference)
"""
import glob
import math
import sys
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

REPO = "/workspace/vjepa2"
W = "/workspace/r25"
DATA = f"{W}/data"
RES = f"{W}/results"
CKPT_GLOB = "/workspace/cache/torch/hub/checkpoints/vjepa2-ac-vitg*.pt"
NTOK, COND = 256, 2
NCAND = 64
ALPHAS = [0.05, 0.1, 0.2]
QPCT = 0.999
GAUSS_SCALES = [0.1, 0.25, 0.5, 1.0, 2.0]
N_PER_SCALE = 8
N_OTHER = NCAND - 3 - len(GAUSS_SCALES) * N_PER_SCALE  # 21
PERT_DIMS = [0, 1, 2, 6]  # xyz and gripper: the action space the official CEM samples (rotation fixed)
sys.path.insert(0, REPO)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


@contextmanager
def precision(mode):
    """bf16 / fp16: autocast (bf16 is the official inference setting); tf32: fp32 activations with TF32 matmuls;
    fp32: strict fp32."""
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = (mode == "tf32")
    try:
        if mode in ("bf16", "fp16"):
            with torch.autocast("cuda", dtype=torch.bfloat16 if mode == "bf16" else torch.float16):
                yield
        else:
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


# ----------------------------------------------------------------------------------------------------------- poses
def poses_to_diffs(poses):
    """Official DROID loader convention on [T, 7] float64: xyz diff, relative rotation R_{t+1} R_t^T (euler xyz), gripper diff."""
    xyz = poses[1:, :3] - poses[:-1, :3]
    R = Rotation.from_euler("xyz", poses[:, 3:6], degrees=False)
    ang = (R[1:] * R[:-1].inv()).as_euler("xyz", degrees=False)
    return np.concatenate([xyz, ang, poses[1:, -1:] - poses[:-1, -1:]], axis=1)


def compute_new_pose_np(pose, action):
    """Replica of the official compute_new_pose on [B, 7] float64: xyz add, R_new = dR @ R, gripper add clipped to [0, 1]."""
    new_xyz = pose[:, :3] + action[:, :3]
    R = Rotation.from_euler("xyz", pose[:, 3:6], degrees=False)
    dR = Rotation.from_euler("xyz", action[:, 3:6], degrees=False)
    new_ang = (dR * R).as_euler("xyz", degrees=False)
    new_g = np.clip(pose[:, -1:] + action[:, -1:], 0, 1)
    return np.concatenate([new_xyz, new_ang, new_g], axis=1)


# ----------------------------------------------------------------------------------------------------------- model
def _clean(sd):
    return {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}


def load_models(need_encoder=True):
    t0 = time.time()
    enc, pred = torch.hub.load(REPO, "vjepa2_ac_vit_giant", source="local", pretrained=False)
    sd = torch.load(glob.glob(CKPT_GLOB)[0], map_location="cpu", weights_only=False)
    info = dict(ckpt_keys=list(sd.keys()))
    for name, mod, key in [("enc", enc, "encoder"), ("pred", pred, "predictor")]:
        if name == "enc" and not need_encoder:
            continue
        m, u = mod.load_state_dict(_clean(sd[key]), strict=False)
        info[f"{name}_missing"], info[f"{name}_unexpected"] = len(m), len(u)
        info[f"{name}_examples"] = [list(m)[:3], list(u)[:3]]
    del sd
    pred = pred.cuda().eval()
    enc = enc.cuda().eval() if need_encoder else None
    info["load_s"] = round(time.time() - t0, 1)
    info["pred_mlp"] = str(pred.predictor_blocks[0].mlp).replace("\n", " ")
    info["n_blocks"] = len(pred.predictor_blocks)
    info["pred_params_M"] = round(sum(p.numel() for p in pred.parameters()) / 1e6, 1)
    return enc, pred, info


def make_transform(kind="square"):
    """square: notebook / phaseB setting (ratio 1, scale 1) -> central 180x180 crop of a 180x320 DROID frame.
    train: the DROID training config (configs/train/vitg16/droid-256px-8f.yaml: scale 1.777, ratio 0.75-1.35); on 16:9
    frames every random draw is rejected and the loader falls back to a deterministic central 243x180 crop (aspect 1.35)."""
    from app.vjepa_droid.transforms import make_transforms
    if kind == "train":
        return make_transforms(random_horizontal_flip=False, random_resize_aspect_ratio=(0.75, 1.35), random_resize_scale=(1.777, 1.777),
                               reprob=0.0, auto_augment=False, motion_shift=False, crop_size=256)
    return make_transforms(random_horizontal_flip=False, random_resize_aspect_ratio=(1.0, 1.0), random_resize_scale=(1.0, 1.0),
                           reprob=0.0, auto_augment=False, motion_shift=False, crop_size=256)


@torch.no_grad()
def encode_frames(enc, transform, frames, prec="fp32", bs=16):
    """frames uint8 [T, H, W, 3] -> layer-normed latents [T, 256, 1408]; each frame alone as a 2-frame tubelet."""
    x = transform(frames).permute(1, 0, 2, 3)  # [T, C, 256, 256]
    out = []
    for s in range(0, len(x), bs):
        c = x[s:s + bs].unsqueeze(2).repeat(1, 1, 2, 1, 1).cuda()
        with precision(prec):
            h = enc(c)
        out.append(F.layer_norm(h.float(), (h.size(-1),)))
    return torch.cat(out)


@torch.no_grad()
def rollout(pred, z0, s0, acts, prec, iv=None, seed=None):
    """Official 2-step planning rollout for a batch of candidate sequences.
    z0 [256, D] context latent (cuda, fp32, layer-normed); s0 [7] float64; acts [B, 2, 7] float64.
    Returns layer-normed predictions zh1 (t+1, uses only the first action) and zh2 (t+2), both [B, 256, D] fp32."""
    B = acts.shape[0]
    dev = z0.device
    if iv is not None and iv.mode in ("D1", "D2"):  # independent masks: reproducible stream per (window, mask seed)
        iv.gen.manual_seed(int(seed))
    s0b = np.repeat(s0[None], B, 0)
    s1 = compute_new_pose_np(s0b, acts[:, 0])
    a = torch.tensor(acts, dtype=torch.float32, device=dev)
    st = torch.tensor(np.stack([s0b, s1], 1), dtype=torch.float32, device=dev)
    x = z0.unsqueeze(0).expand(B, -1, -1).contiguous()
    with precision(prec):
        o1 = pred(x, a[:, :1], st[:, :1])[:, -NTOK:]
    zh1 = F.layer_norm(o1.float(), (o1.size(-1),))
    with precision(prec):
        o2 = pred(torch.cat([x, zh1], 1), a, st)[:, -NTOK:]
    zh2 = F.layer_norm(o2.float(), (o2.size(-1),))
    return zh1, zh2


def energy(zh, goal):
    return (zh - goal.unsqueeze(0)).abs().mean((1, 2))


# ---------------------------------------------------------------------------------------------------- intervention
class Intervention:
    """forward_pre_hook on every predictor block's mlp.fc2, i.e. on the post-GELU hidden activations h [B, N, 4096].

    modes: None (no-op) | A: zero weak entries (|h| < thr_j) | B: weak entries -> mu_j | C: h + delta_j everywhere |
           random masks, entry-wise Bernoulli(p):
           D1 / D2   : independent mask for every candidate, token and unit (fresh draws; zero-fill / abar_j-fill)
           F1 / F2   : "D-fix", one fixed mask over (block, token position, unit) applied identically to all candidates
                       and windows (u < p with a fixed uniform tensor u per seed; zero-fill / abar_j-fill)"""

    def __init__(self, pred):
        self.blocks = pred.predictor_blocks
        self.NB = len(self.blocks)
        self.H = self.blocks[0].mlp.fc1.out_features
        self.mode = None
        self.collector = None
        self.count = False
        self.nmod = torch.zeros(self.NB, dtype=torch.float64, device="cuda")
        self.ntot = torch.zeros(self.NB, dtype=torch.float64, device="cuda")
        self.thr = self.fill = self.delta = self.u = None
        self.p = 0.0
        self.gen = None
        self.handles = [blk.mlp.fc2.register_forward_pre_hook(self._make(b)) for b, blk in enumerate(self.blocks)]

    def set(self, mode=None, thr=None, fill=None, delta=None, u=None, p=0.0):
        self.mode, self.thr, self.fill, self.delta, self.u, self.p = mode, thr, fill, delta, u, float(p)
        if mode in ("D1", "D2") and self.gen is None:
            self.gen = torch.Generator(device="cuda")

    def reset_count(self):
        self.nmod.zero_()
        self.ntot.zero_()

    def modfrac(self):
        return float(self.nmod.sum() / self.ntot.sum().clamp(min=1))

    def _make(self, b):
        def hook(mod, inp):
            h = inp[0]
            if self.collector is not None:
                self.collector(b, h)
            mode = self.mode
            if mode is None:
                return None
            if mode == "C":
                return (h + self.delta[b].to(h.dtype),)
            if mode in ("A", "B"):
                m = h.abs() < self.thr[b].to(h.dtype)
            elif mode in ("F1", "F2"):
                m = (self.u[b, :h.shape[1]] < self.p).unsqueeze(0)
            elif mode in ("D1", "D2"):
                m = torch.rand(h.shape, generator=self.gen, device=h.device) < self.p
            else:
                raise ValueError(mode)
            if self.count:
                self.nmod[b] += m.sum() * (h.shape[0] if m.shape[0] == 1 else 1)
                self.ntot[b] += h.numel()
            if mode in ("A", "D1", "F1"):
                return (h.masked_fill(m, 0.0),)
            return (torch.where(m, self.fill[b].to(h.dtype), h),)
        return hook


def shared_uniform(NB, L, H, seed):
    g = torch.Generator(device="cuda")
    g.manual_seed(int(seed))
    return torch.rand((NB, L, H), generator=g, device="cuda")


class Pass1:
    """Random subsample of |h| rows per block (for the per-unit 99.9th percentile m_j) and the exact E[h_j]."""

    def __init__(self, NB, H, rows=1024, seed=0):
        self.samples = [[] for _ in range(NB)]
        self.sum = torch.zeros(NB, H, dtype=torch.float64, device="cuda")
        self.n = torch.zeros(NB, dtype=torch.float64, device="cuda")
        self.g = torch.Generator(device="cuda")
        self.g.manual_seed(seed)
        self.rows = rows

    def __call__(self, b, h):
        x = h.reshape(-1, h.shape[-1])
        idx = torch.randint(0, x.shape[0], (self.rows,), generator=self.g, device=x.device)
        self.samples[b].append(x[idx].abs().half())
        self.sum[b] += x.float().sum(0).double()
        self.n[b] += x.shape[0]

    def quantile(self, q=QPCT):
        out = []
        for s in self.samples:
            X = torch.cat(s).float()
            k = max(1, min(X.shape[0], int(math.ceil(q * X.shape[0]))))
            out.append(torch.kthvalue(X, k, dim=0).values)
        return torch.stack(out)  # [NB, H]


class Pass2:
    """Exact per-unit weak-regime sums for every alpha: sum h 1(|h|<thr), count 1(|h|<thr); also cond-token counts."""

    def __init__(self, thr):  # thr [nA, NB, H]
        self.thr = thr
        nA, NB, H = thr.shape
        self.sw = torch.zeros(nA, NB, H, dtype=torch.float64, device="cuda")
        self.nw = torch.zeros(nA, NB, H, dtype=torch.float64, device="cuda")
        self.n = torch.zeros(NB, dtype=torch.float64, device="cuda")
        self.nw_cond = torch.zeros(nA, NB, dtype=torch.float64, device="cuda")
        self.n_cond = torch.zeros(NB, dtype=torch.float64, device="cuda")

    def __call__(self, b, h):
        N = h.shape[1]
        cond = (torch.arange(N, device=h.device) % (NTOK + COND)) < COND
        hc = h[:, cond].reshape(-1, h.shape[-1]).float()
        x = h.reshape(-1, h.shape[-1]).float()
        for i in range(self.thr.shape[0]):
            w = x.abs() < self.thr[i, b]
            self.sw[i, b] += (x * w).sum(0).double()
            self.nw[i, b] += w.sum(0).double()
            self.nw_cond[i, b] += (hc.abs() < self.thr[i, b]).sum().double()
        self.n[b] += x.shape[0]
        self.n_cond[b] += hc.shape[0]
