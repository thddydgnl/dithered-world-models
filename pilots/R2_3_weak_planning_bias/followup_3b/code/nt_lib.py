"""Shared helpers for pilot R2-3b (candidate-consistent random controls), run on the pod.
Copied from R2-3 (R2_3_weak_planning_bias/code/nt_lib.py) and extended with the deterministic controls
D-fix (fixed random unit subsets zeroed for every candidate) and D-dir (fixed random direction added).

- build_agent(task): single-task TD-MPC2 (model_size 5, state obs, MPC on, compile off) built from the official repo
  and the released checkpoint, without a hydra app.
- run_episode(...): one evaluation episode with a fixed seed (fresh env seeded with the episode seed, torch/numpy
  seeded with the same seed), exactly like the official eval loop (agent.act(obs, t0, eval_mode=True)).
- Ablator: forward hooks on the hidden NormedLinear layers (Linear -> LayerNorm -> Mish) of the dynamics and reward
  MLPs of the world model (conditions A, B, C, D*). Encoder, Q-ensemble and policy prior are never touched.
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


CONDS = ("A", "B", "C", "D1z", "D3z", "D1m", "D3m")
COND_DESC = {
    "A": "zero the weak activations (R1-1 weak)",
    "B": "replace weak activations by the per-unit conditional mean E[a | weak]",
    "C": "bias only: add the per-unit mean shift of A, -E[a 1(weak)], to every activation",
    "D1z": "random zeroing, matched to A on 1-step error (R1-1 rand_pred)",
    "D3z": "random zeroing, matched to A on 3-step error",
    "D1m": "random mean replacement (mean-corrected), matched to B on 1-step error",
    "D3m": "random mean replacement (mean-corrected), matched to B on 3-step error",
}


class Ablator:
    """Forward hooks on the hidden layers dyn0, dyn1 (dynamics MLP) and rew0, rew1 (reward MLP) of the world model.

    Each hooked module is a NormedLinear whose output is Mish(LayerNorm(Linear(x))); a hook replaces that output.
    m[name]: per-unit reference scale (99.9th percentile of |a_j| during planning). Weak set of a row (one input
    sample): {j : |a_j| < alpha * m_j}, c = its size, n = layer width. Per-alpha statistics (unablated planning-time
    activations of the calibration split): mu[j] = E[a_j | weak], delta[j] = -E[a_j 1(weak)]; abar[j] = E[a_j].
    modes
      None       off
      'record'   keep the last output of each hooked layer in self.last
      'weak'     A: zero the weak set
      'meanrep'  B: replace the weak set by mu (fluctuations removed, mean kept)
      'shift'    C: add delta to every activation (A's mean shift only, fluctuations kept)
      'rand'     D: each unit of the row is selected independently with probability min(1, r * c / n) (uniform
                 random units, count structure of the weak set) and replaced by 0 (fill='zero') or by abar
                 (fill='mean', the mean-corrected version); r per module family (dyn / rew)
    Random draws use a private CUDA generator, so the planner's own RNG stream is identical in every condition.
    """

    def __init__(self, model, device=DEV):
        self.mods = {"dyn0": model._dynamics[0], "dyn1": model._dynamics[1],
                     "rew0": model._reward[0], "rew1": model._reward[1]}
        self.device = device
        self._m, self.inv_m, self.stats, self.abar = {}, {}, {}, {}
        self.mode, self.alpha, self.r, self.fill = None, 0.0, {"dyn": 1.0, "rew": 1.0}, "zero"
        self.cur = None
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(0)
        self.last = {}
        self.names = list(LAYERS)
        self.handles = [mod.register_forward_hook(self._hook(k, k[:3])) for k, mod in self.mods.items()]
        self.zeroed = {k: torch.zeros((), dtype=torch.float64, device=device) for k in self.names}
        self.total = {k: 0 for k in self.names}

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, value):
        self._m = {k: v.clamp(min=1e-6) for k, v in value.items()}
        self.inv_m = {k: 1.0 / v for k, v in self._m.items()}

    def set(self, mode=None, alpha=0.0, r_dyn=1.0, r_rew=1.0, fill="zero", seed=None, f_dyn=0.0, f_rew=0.0,
            sub_seed=0, lam_dyn=0.0, lam_rew=0.0):
        self.mode, self.alpha, self.fill = mode, float(alpha), fill
        self.r = {"dyn": float(r_dyn), "rew": float(r_rew)}
        if mode == "fix":
            self.gain, self.nzero = {}, {}
            for name in self.names:
                f = float(f_dyn) if name.startswith("dyn") else float(f_rew)
                self.gain[name], self.nzero[name] = self._fix_gain(name, f, int(sub_seed))
        if mode == "dir":
            self.shift = {name: self._dir_vec(name, int(sub_seed)) * (float(lam_dyn) if name.startswith("dyn") else float(lam_rew))
                          for name in self.names}
        self.cur = self.stats.get(akey(alpha)) if mode in ("meanrep", "shift") else None
        if mode in ("meanrep", "shift") and self.cur is None:
            raise ValueError(f"no statistics for alpha {alpha}")
        if seed is not None:
            self.gen.manual_seed(int(seed))
        for k in self.names:
            self.zeroed[k].zero_()
            self.total[k] = 0

    def _gen(self, name, sub_seed):
        g = torch.Generator()  # CPU generator: the fixed subsets/directions do not depend on the device
        g.manual_seed(1_000_003 * (int(sub_seed) + 1) + 7919 * self.names.index(name))
        return g

    def _fix_gain(self, name, f, sub_seed):
        """Fixed random unit subset: the first floor(f n) units of a seeded permutation are zeroed, the next one is
        scaled by 1 - frac(f n) (continuous, nested in f). Same subset for every candidate and planning step."""
        n = self.mods[name].out_features
        perm = torch.randperm(n, generator=self._gen(name, sub_seed))
        x = min(max(f, 0.0), 1.0) * n
        k = int(x)
        g = torch.ones(n)
        g[perm[:k]] = 0.0
        if k < n:
            g[perm[k]] = 1.0 - (x - k)
        return g.to(self.device), k

    def _dir_vec(self, name, sub_seed):
        """Fixed random unit-norm direction in the layer's activation space (same for every candidate and step)."""
        n = self.mods[name].out_features
        z = torch.randn(n, generator=self._gen(name, sub_seed + 1000))
        return (z / z.norm()).to(self.device)

    def fractions(self):
        z = {k: float(v) for k, v in self.zeroed.items()}
        out = {k: (z[k] / self.total[k] if self.total[k] else None) for k in self.names}
        for grp, keys in (("dyn", ("dyn0", "dyn1")), ("rew", ("rew0", "rew1")), ("all", LAYERS)):
            n = sum(self.total[k] for k in keys)
            out[grp] = sum(z[k] for k in keys) / n if n else None
        return out

    def _hook(self, name, kind):
        def hook(module, inputs, out):
            mode = self.mode
            if mode is None:
                return None
            if mode == "record":
                self.last[name] = out
                return None
            if mode == "shift":
                self.total[name] += out.numel()  # every entry is modified; 'zeroed' stays 0
                return out + self.cur["delta"][name]
            if mode == "fix":
                self.zeroed[name] += self.nzero[name] * (out.numel() // out.shape[-1])
                self.total[name] += out.numel()
                return out * self.gain[name]
            if mode == "dir":
                self.total[name] += out.numel()
                return out + self.shift[name]
            weak = (out.abs() * self.inv_m[name]) < self.alpha
            if mode == "weak":
                mask, fillv = weak, None
            elif mode == "meanrep":
                mask, fillv = weak, self.cur["mu"][name]
            elif mode == "rand":
                p = weak.sum(-1, keepdim=True) * (self.r[kind] / out.shape[-1])
                mask = torch.rand(out.shape, generator=self.gen, device=out.device, dtype=out.dtype) < p
                fillv = self.abar[name] if self.fill == "mean" else None
            else:
                raise ValueError(mode)
            self.zeroed[name] += mask.sum()
            self.total[name] += mask.numel()
            if fillv is None:
                return out.masked_fill(mask, 0.0)
            return torch.where(mask, fillv.expand_as(out), out)

        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def setting_for(cond, alpha, scales=None):
    if cond == "none":
        return {"mode": None}
    if cond == "A":
        return {"mode": "weak", "alpha": alpha}
    if cond == "B":
        return {"mode": "meanrep", "alpha": alpha}
    if cond == "C":
        return {"mode": "shift", "alpha": alpha}
    if cond in ("D1z", "D3z", "D1m", "D3m"):
        r = scales["match"][akey(alpha)][cond]
        return {"mode": "rand", "alpha": alpha, "r_dyn": r["dyn"], "r_rew": r["rew"],
                "fill": "mean" if cond.endswith("m") else "zero"}
    if cond.startswith("Dfix") or cond.startswith("Ddir"):  # e.g. Dfix1_s0, Dfix3_s2, Ddir1_s1
        p = scales["match3b"][akey(alpha)][cond]
        sub = int(cond.split("_s")[1])
        if cond.startswith("Dfix"):
            return {"mode": "fix", "alpha": alpha, "f_dyn": p["dyn"], "f_rew": p["rew"], "sub_seed": sub}
        return {"mode": "dir", "alpha": alpha, "lam_dyn": p["dyn"], "lam_rew": p["rew"], "sub_seed": sub}
    raise ValueError(cond)


def attach_scales(ab, sc):
    ab.m = sc["m"]
    ab.stats = sc["stats"]
    ab.abar = sc["abar"]


def load_scales(out, task):
    sc = torch.load(f"{out}/data/scales_{task}.pt", map_location="cpu", weights_only=False)
    sc["m"] = {k: v.to(DEV) for k, v in sc["m"].items()}
    sc["abar"] = {k: v.to(DEV) for k, v in sc.get("abar", {}).items()}
    sc["stats"] = {a: {s: {k: v.to(DEV) for k, v in d.items()} for s, d in st.items()} for a, st in sc.get("stats", {}).items()}
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
