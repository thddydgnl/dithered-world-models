"""Shared helpers for pilot R1-1 (near-tie planning), run on the pod.

- build_agent(task): single-task TD-MPC2 (model_size 5, state obs, MPC on, compile off) built from the official repo
  and the released checkpoint, without a hydra app.
- run_episode(...): one evaluation episode with a fixed seed (fresh env seeded with the episode seed, torch/numpy
  seeded with the same seed), exactly like the official eval loop (agent.act(obs, t0, eval_mode=True)).
- Ablator: forward hooks on the hidden NormedLinear layers (Linear -> LayerNorm -> Mish) of the dynamics and reward
  MLPs of the world model. Encoder, Q-ensemble and policy prior are never touched.
- CandRecorder: temporarily wraps agent._estimate_value to record the MPPI candidate sets (planning-time inputs).
"""
import copy
import os
import random
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
TDMPC2_DIR = os.environ.get("TDMPC2_DIR", "/root/tdmpc2/tdmpc2")
sys.path.insert(0, TDMPC2_DIR)

import hydra.utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

hydra.utils.get_original_cwd = lambda: os.getcwd()  # parse_cfg expects a running hydra app

from common import math as tmath  # noqa: E402
from common.parser import parse_cfg  # noqa: E402
from envs import make_env  # noqa: E402
from tdmpc2 import TDMPC2  # noqa: E402
from envs.wrappers.timeout import Timeout as _Timeout  # noqa: E402

# compat shim: envs/metaworld.py assigns env.max_episode_steps, a read-only property of Timeout in this repo version
# (same value; no behavioral change)
if _Timeout.max_episode_steps.fset is None:
    _Timeout.max_episode_steps = _Timeout.max_episode_steps.setter(
        lambda self, v: setattr(self, "_max_episode_steps", v))

CKPT = os.environ.get("NT_CKPT", "/workspace/ckpt") + "/{task}-1.pt"
LAYERS = ("dyn0", "dyn1", "rew0", "rew1")
CONDS = ("weak", "rand_count", "rand_pred", "strong")
EP_SEED0 = 1000
DEV = "cuda"


def akey(alpha):
    return f"{float(alpha):g}"


def make_cfg(task, seed=EP_SEED0):
    cfg = OmegaConf.load(os.path.join(TDMPC2_DIR, "config.yaml"))
    cfg.task = task
    cfg.model_size = 5
    cfg.checkpoint = CKPT.format(task=task)
    cfg.compile = False
    cfg.mpc = True
    cfg.seed = seed
    cfg.obs = "state"
    cfg.enable_wandb = False
    cfg.save_video = False
    cfg.data_dir = "none"
    cfg.wandb_project = "none"
    cfg.wandb_entity = "none"
    return parse_cfg(cfg)


def build_agent(task):
    """Returns (agent, cfg). make_env fills obs/action dims and episode length into cfg before the model is built."""
    cfg = make_cfg(task)
    make_env(cfg)
    torch.manual_seed(0)
    agent = TDMPC2(cfg)
    agent.load(cfg.checkpoint)
    agent.model.eval()
    return agent, cfg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def run_episode(agent, cfg, seed, before_act=None, after_act=None):
    """One eval episode. The env is created with cfg.seed = seed, so episode k has the same initial state in every
    condition, and torch is seeded with the same seed, so the planner's noise is paired across conditions."""
    c = copy.deepcopy(cfg)
    c.seed = seed
    env = make_env(c)
    seed_all(seed)
    obs, done, ret, t, info, succ_any, rews = env.reset(), False, 0.0, 0, {}, 0.0, []
    while not done:
        if before_act is not None:
            before_act(t, obs)
        action = agent.act(obs, t0=(t == 0), eval_mode=True)
        if after_act is not None:
            after_act(t, obs, action)
        obs, reward, done, info = env.step(action)
        rews.append(float(reward))
        ret += float(reward)
        succ_any = max(succ_any, float(info.get("success", 0.0)))
        t += 1
    return {"ret": ret, "success": float(info.get("success", 0.0)), "success_any": succ_any, "steps": t,
            "rewards": rews, "final_obs": obs}


SCOPES = {"wm": ("dyn", "rew"), "q": ("q",), "all": ("dyn", "rew", "q")}


class Ablator:
    """Forward hooks on hidden layers of the world model.

    Layers: dyn0, dyn1 (dynamics MLP), rew0, rew1 (reward MLP) and, with with_q=True, q{k}_0, q{k}_1 for the k-th
    Q-ensemble member. Each hooked module is a NormedLinear whose output is Mish(LayerNorm(Linear(x))); a hook replaces
    that output. With with_q=True the vmapped Ensemble forward is replaced by a loop over members with the same
    parameters (numerically the same up to float rounding) so that each member's hook can use its own m_j.
    scope: 'wm' (dyn + rew, the original condition), 'q' (Q members only), 'all' (both).
    m[name] is the per-unit reference scale (99.9th percentile of |a_j| during planning). For one row (one input sample)
    the weak set is {j : |a_j| < alpha * m_j}; c = its size, n = layer width.
    modes
      None      off
      'record'  keep the last output of each hooked layer in self.last
      'weak'    zero the weak set
      'rand'    zero each unit of the row independently with probability p = min(1, r * c / n) (uniformly random
                units, count-matched to the weak set row by row in expectation; r per module family: dyn / rew / q).
                r = 1 is the count-matched control, r = r_pred(alpha) the prediction-matched control
      'strong'  zero the c units of the row with the largest |a_j| / m_j (count-matched top-k, sanity check)
    Random draws use a private CUDA generator, so the planner's own RNG stream is identical in every condition.
    Zeroed-entry counts are accumulated on the GPU (no sync in the planning loop).
    """

    def __init__(self, model, device=DEV, with_q=False):
        self.mods = {"dyn0": model._dynamics[0], "dyn1": model._dynamics[1],
                     "rew0": model._reward[0], "rew1": model._reward[1]}
        self.device = device
        self._m, self.inv_m = {}, {}
        self.mode, self.alpha, self.r, self.active = None, 0.0, {"dyn": 1.0, "rew": 1.0, "q": 1.0}, SCOPES["wm"]
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(0)
        self.last = {}
        self.names = list(LAYERS)
        self.handles = [mod.register_forward_hook(self._hook(k, k[:3])) for k, mod in self.mods.items()]
        self.q_member = None
        if with_q:
            self._install_q(model._Qs)
        self.zeroed = {k: torch.zeros((), dtype=torch.float64, device=device) for k in self.names}
        self.total = {k: 0 for k in self.names}

    def _install_q(self, ens):
        n, base, params = len(ens), ens.module, ens.params

        def forward_loop(*args, **kwargs):
            outs = []
            for k in range(n):
                self.q_member = k
                with params[k].to_module(base):
                    outs.append(base(*args, **kwargs))
            self.q_member = None
            return torch.stack(outs)

        ens.forward = forward_loop
        self.nq = n
        self.names += [f"q{k}_{li}" for k in range(n) for li in (0, 1)]
        for li in (0, 1):
            self.handles.append(base[li].register_forward_hook(self._hook(None, "q", li)))

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, value):
        self._m = {k: v.clamp(min=1e-6) for k, v in value.items()}
        self.inv_m = {k: 1.0 / v for k, v in self._m.items()}

    def set(self, mode=None, alpha=0.0, r_dyn=1.0, r_rew=1.0, r_q=1.0, scope="wm", seed=None):
        self.mode, self.alpha = mode, float(alpha)
        self.r = {"dyn": float(r_dyn), "rew": float(r_rew), "q": float(r_q)}
        self.active = SCOPES[scope]
        if seed is not None:
            self.gen.manual_seed(int(seed))
        for k in self.names:
            self.zeroed[k].zero_()
            self.total[k] = 0

    def fractions(self):
        z = {k: float(v) for k, v in self.zeroed.items()}
        out = {k: (z[k] / self.total[k] if self.total[k] else None) for k in self.names if k in LAYERS}
        groups = (("dyn", ("dyn0", "dyn1")), ("rew", ("rew0", "rew1")), ("all", LAYERS),
                  ("q", [k for k in self.names if k.startswith("q")]), ("all6", self.names))
        for grp, keys in groups:
            n = sum(self.total[k] for k in keys)
            out[grp] = sum(z[k] for k in keys) / n if n else None
        return out

    def _hook(self, name, kind, li=None):
        def hook(module, inputs, out):
            mode = self.mode
            if mode is None:
                return None
            nm = name if name is not None else (None if self.q_member is None else f"q{self.q_member}_{li}")
            if nm is None:
                return None
            if mode == "record":
                self.last[nm] = out
                return None
            if kind not in self.active:
                return None
            score = out.abs() * self.inv_m[nm]
            weak = score < self.alpha
            if mode == "weak":
                mask = weak
            elif mode == "rand":
                p = weak.sum(-1, keepdim=True) * (self.r[kind] / out.shape[-1])
                mask = torch.rand(out.shape, generator=self.gen, device=out.device, dtype=out.dtype) < p
            elif mode == "strong":
                c = weak.sum(-1, keepdim=True)
                thr = score.sort(dim=-1, descending=True).values.gather(-1, (c - 1).clamp(min=0))
                mask = (score >= thr) & (c > 0)
            else:
                raise ValueError(mode)
            self.zeroed[nm] += mask.sum()
            self.total[nm] += mask.numel()
            return out.masked_fill(mask, 0.0)

        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def setting_for(cond, alpha, scales=None, scope="wm"):
    if cond == "none":
        return {"mode": None}
    if cond == "weak":
        return {"mode": "weak", "alpha": alpha, "scope": scope}
    if cond == "strong":
        return {"mode": "strong", "alpha": alpha, "scope": scope}
    if cond == "rand_count":
        return {"mode": "rand", "alpha": alpha, "r_dyn": 1.0, "r_rew": 1.0, "r_q": 1.0, "scope": scope}
    if cond == "rand_pred":
        s = scales["pred_scale"][akey(alpha)]
        rq = scales.get("pred_scale_q", {}).get(akey(alpha), {}).get("q", 1.0) if scope != "wm" else 1.0
        return {"mode": "rand", "alpha": alpha, "r_dyn": s["dyn"], "r_rew": s["rew"], "r_q": rq, "scope": scope}
    raise ValueError(cond)


def load_scales(out, task, with_q=False):
    sc = torch.load(f"{out}/data/scales_{task}.pt", map_location="cpu", weights_only=False)
    if with_q:
        sq = torch.load(f"{out}/data/scales_q_{task}.pt", map_location="cpu", weights_only=False)
        sc["m"].update(sq["m_q"])
        sc["pred_scale_q"] = sq["pred_scale_q"]
    sc["m"] = {k: v.to(DEV) for k, v in sc["m"].items()}
    return sc


class CandRecorder:
    """Temporarily replaces agent._estimate_value to record MPPI candidate sets.

    _estimate_value(z, actions, task) is called once per MPPI iteration with z = root latent repeated num_samples times
    and actions of shape (horizon, num_samples, action_dim) (the first num_pi_trajs columns are policy-prior
    trajectories). Per call we keep K uniformly chosen candidate sequences; on 'full' steps we also keep the whole
    candidate set of the first and the last iteration. Indices come from numpy, so torch's RNG is not consumed.
    """

    def __init__(self, agent, K, seed):
        self.agent = agent
        self.inner = agent._estimate_value
        agent._estimate_value = self
        self.K = K
        self.rng = np.random.default_rng(seed)
        self.start_step(False)

    def start_step(self, full=False):
        self.full, self.it, self.first, self.last, self.z, self.i = full, [], None, None, None, 0

    def __call__(self, z, actions, task):
        n = actions.shape[1]
        idx = torch.as_tensor(self.rng.choice(n, self.K, replace=False), device=actions.device)
        self.it.append(actions.index_select(1, idx).transpose(0, 1).clone())  # (K, H, A)
        if self.z is None:
            self.z = z[0].clone()
        if self.full:
            if self.i == 0:
                self.first = actions.clone()
            self.last = actions.clone()
        self.i += 1
        return self.inner(z, actions, task)

    def end_step(self):
        return self.z, torch.stack(self.it), self.first, self.last

    def detach(self):
        del self.agent._estimate_value


def two_hot_inv(x, cfg):
    return tmath.two_hot_inv(x, cfg).squeeze(-1)
