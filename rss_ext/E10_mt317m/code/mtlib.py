"""E10 (paper A, RSS extension, DESIGN_ext section 3): single-task evaluation of the official multi-task TD-MPC2
checkpoint mt80-317M.

The official code (tdmpc2 commit 332f19e) evaluates a multi-task checkpoint only through envs.make_multitask_env (all 80
member environments behind envs/wrappers/multitask.MultitaskWrapper; evaluate.py with task=mt80). This module evaluates
ONE Meta-World member with exactly the inputs the multi-task model gets there:
  - agent: TDMPC2 built from the mt80 config (model_size 317 -> enc/mlp 4096, latent 1376, 5 encoder layers, 8 Qs;
    task_dim 96; 80 task embeddings; action masks from the per-task action dims; one discount per task from the per-task
    episode lengths). The per-task dims (obs_shapes, action_dims, episode_lengths, obs_shape, action_dim, episode_length)
    are the values envs.make_env(task=mt80) writes into the cfg; mt_dims.py builds the official multi-task env once and
    stores them, so no worker has to build 80 environments.
  - environment: the member's own single-task environment (envs.make_env with cfg.task = <member>, exactly how
    make_multitask_env builds each member), observations zero-padded at the end to the largest observation dim
    (MultitaskWrapper._pad_obs), actions cut to the member's action dim before env.step (MultitaskWrapper.step).
  - planning: agent.act(obs, t0, eval_mode=True, task=<index in TASK_SET['mt80']>): the task embedding enters every
    encoder / dynamics / reward / policy / Q call and the task's action mask is applied inside TDMPC2._plan.
test_wrapper.py checks that this gives bit-identical episodes to the official MultitaskWrapper path.

Episode conventions are those of qlib.run_episode (E0/E1): a fresh environment with cfg.seed = episode seed,
random/numpy/torch seeded with the episode seed, dither generator seeded with 7 seed + 13, eval_mode=True, success = the
last step's info['success'].

Quantization: qlib.Quantizer unchanged. Its hooks sit on model._dynamics[0], _dynamics[1], _reward[0], _reward[1] = the two
hidden NormedLinear layers (Linear -> LayerNorm -> Mish, width 4096) of the dynamics and reward MLPs. In the multi-task
model their input is [z, a, task embedding]; the hooked output (Mish activations) is the same kind of tensor as in E1.

The planning-time prediction errors (pred_errors_mt / plan_inputs_mt / _rollout_mt) and the quantizer ranges
(quant_ranges_mt) are copies of qlib.pred_errors / plan_inputs / _rollout and calib_lib.quant_ranges_from_calib whose
only change is that the task index is passed to model.next / reward / pi (the E1 code passes None = single-task model).
"""
import os
import time

import numpy as np
import torch

import qlib as L   # E1 library, verbatim copy: sys.path to TDMPC2_DIR, MUJOCO_GL, Timeout shim, Quantizer, seeding

from omegaconf import OmegaConf  # noqa: E402
from common import TASK_SET  # noqa: E402
from common.parser import parse_cfg  # noqa: E402
from envs import make_env  # noqa: E402
from tdmpc2 import TDMPC2  # noqa: E402

CKPT = os.environ.get("E10_CKPT", "/workspace/ckpt/mt80-317M.pt")
DIMS = os.environ.get("E10_DIMS", "/workspace/e10/mt80_dims.json")
MT = "mt80"
TASKS = list(TASK_SET[MT])
HELD_SEEDS = list(range(91000, 91030))     # DESIGN_ext 0.4: held (qualification, ranges, k selection, pt_err1h inputs)
EVAL_SEEDS = list(range(90000, 90090))     # DESIGN_ext 0.4: eval (90 pairs, one checkpoint)
K_PER_ITER = 16                            # E0 calib_lib: random candidates recorded per MPPI iteration
MAX_ROWS = 12000                           # E0 calib_lib.quant_ranges_from_calib: rows of the range subsample
Q_HI = 0.9999                              # 99.99th percentile
OEI_STRIDE_MW = 4                          # E1 Meta-World OEI states: t % 4 == 0
PERR_STATES = 8                            # E1 runner: perr_plan on 8 evenly spaced OEI states per episode


def pt_steps(T=100, stride=OEI_STRIDE_MW, n=PERR_STATES):
    """Control steps whose first-iteration candidate sets E1 used for perr_plan in a Meta-World episode (runner.py:
    the OEI states t % 4 == 0, then n evenly spaced ones) -> [0, 12, 28, 40, 56, 68, 84, 96]."""
    oei = list(range(0, T, stride))
    sel = np.linspace(0, len(oei) - 1, n).round().astype(int)
    return [oei[i] for i in sel]


PT_STEPS = tuple(pt_steps())


def task_index(task):
    return TASKS.index(task)


def task_tensor(task):
    return torch.tensor([task_index(task)], device=L.DEV)


def base_cfg(task, model_size, seed=0, ckpt=CKPT):
    """Same fields as qlib.make_cfg (E0/E1), parsed by the official parse_cfg."""
    cfg = OmegaConf.load(os.path.join(L.TDMPC2_DIR, "config.yaml"))
    cfg.task = task
    cfg.model_size = model_size
    cfg.checkpoint = ckpt
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


DIM_KEYS = ("obs_shapes", "action_dims", "episode_lengths", "obs_shape", "action_dim", "episode_length", "seed_steps")


def dims_from_cfg(cfg):
    """The cfg fields envs.make_env(mt80) fills in (JSON-able)."""
    return {"obs_shapes": [int(x) for x in cfg.obs_shapes], "action_dims": [int(x) for x in cfg.action_dims],
            "episode_lengths": [int(x) for x in cfg.episode_lengths],
            "obs_shape": {k: [int(d) for d in v] for k, v in cfg.obs_shape.items()},
            "action_dim": int(cfg.action_dim), "episode_length": int(cfg.episode_length),
            "seed_steps": int(cfg.seed_steps), "tasks": list(cfg.tasks)}


def mt_cfg(dims):
    """The mt80-317M agent config: parse_cfg(task=mt80, model_size=317) + the fields make_multitask_env/make_env set."""
    cfg = base_cfg(MT, 317)
    assert cfg.multitask and int(cfg.task_dim) == 96 and list(cfg.tasks) == TASKS == list(dims["tasks"])
    cfg.obs_shapes = list(dims["obs_shapes"])
    cfg.action_dims = list(dims["action_dims"])
    cfg.episode_lengths = list(dims["episode_lengths"])
    cfg.obs_shape = {k: tuple(v) for k, v in dims["obs_shape"].items()}
    cfg.action_dim = int(dims["action_dim"])
    cfg.episode_length = int(dims["episode_length"])
    cfg.seed_steps = int(dims["seed_steps"])
    return cfg


def build_agent_mt(dims_path=DIMS, ckpt=CKPT):
    """Returns (agent, cfg) for the mt80-317M checkpoint (same steps as qlib.build_agent: manual_seed(0), TDMPC2(cfg),
    load, eval)."""
    import json
    dims = json.load(open(dims_path))
    cfg = mt_cfg(dims)
    torch.manual_seed(0)
    agent = TDMPC2(cfg)
    agent.load(ckpt)
    agent.model.eval()
    agent._default_planner = {k: getattr(agent.cfg, k) for k in L.PLANNER_KEYS}
    return agent, cfg


class MemberEnv:
    """One member of the mt80 multi-task environment, built alone: envs.make_env(cfg.task = member, cfg.seed = seed)
    as make_multitask_env does for every member, with MultitaskWrapper's observation padding and action cut."""

    def __init__(self, task, seed, obs_dim):
        self.task = task
        self.env = make_env(base_cfg(task, 5, seed))   # single-task build (TensorWrapper(Timeout(MetaWorldWrapper)))
        self.obs_dim = int(obs_dim)
        self.act_dim = int(self.env.action_space.shape[0])
        self.max_episode_steps = self.env.max_episode_steps

    def _pad_obs(self, obs):        # MultitaskWrapper._pad_obs
        if obs.shape != (self.obs_dim,):
            obs = torch.cat((obs, torch.zeros(self.obs_dim - obs.shape[0], dtype=obs.dtype, device=obs.device)))
        return obs

    def reset(self):                # MultitaskWrapper.reset -> member.reset()
        return self._pad_obs(self.env.reset())

    def step(self, action):         # MultitaskWrapper.step
        obs, reward, done, info = self.env.step(action[:self.act_dim])
        return self._pad_obs(obs), reward, done, info


class HeldRecorder:
    """Wraps agent._estimate_value during held none episodes; uses numpy only (torch's RNG stream is untouched, so the
    episode is the same as without it):
      - every control step: the root latent and K_PER_ITER random candidates of every MPPI iteration (E0
        calib_lib.CandRecorder; rows for the quantizer ranges)
      - steps in PT_STEPS: the full first-iteration candidate set and root latent (E1 PlanProbe keep_cands; inputs of
        the planning-time prediction errors pt_err1h)"""

    def __init__(self, agent, seed_rec, k=K_PER_ITER):
        self.agent = agent
        self.inner = agent._estimate_value
        agent._estimate_value = self
        self.K = k
        self.rng = np.random.default_rng(seed_rec)
        self.Z, self.C, self.pt = [], [], {}
        self.t, self.it, self.zr = None, [], None

    def start_step(self, t):
        self.t, self.it, self.zr = int(t), [], None

    def __call__(self, z, actions, task):
        n = actions.shape[1]
        idx = torch.as_tensor(self.rng.choice(n, self.K, replace=False), device=actions.device)
        self.it.append(actions.index_select(1, idx).transpose(0, 1).clone())   # (K, H, A)
        if self.zr is None:     # first MPPI iteration of this control step
            self.zr = z[0].clone()
            if self.t in PT_STEPS:
                self.pt[self.t] = (z[0].clone(), actions.clone())                 # (L,), (H, N, A)
        return self.inner(z, actions, task)

    def end_step(self):
        self.Z.append(self.zr)
        self.C.append(torch.stack(self.it))   # (I, K, H, A)

    def detach(self):
        if self.agent.__dict__.get("_estimate_value") is self:
            del self.agent._estimate_value

    def arrays(self):
        ts = sorted(self.pt)
        f = lambda xs: torch.stack(xs).float().cpu().numpy()
        return {"z_root": f(self.Z), "cand": f(self.C), "pt_t": np.asarray(ts, np.int32),
                "pt_z": f([self.pt[t][0] for t in ts]), "pt_cand": f([self.pt[t][1] for t in ts])}


@torch.no_grad()
def run_episode_mt(agent, quant, task, variant, k, seed, obs_dim, recorder=None, log_calls=False):
    """One episode of a member task (qlib.run_episode conventions, no OEI / plan-rerun instrumentation)."""
    env = MemberEnv(task, seed, obs_dim)
    L.seed_all(seed)
    quant.set(variant, k, seed=seed * 7 + 13)
    quant.call_log = [] if log_calls else None
    tidx = task_index(task)
    obs, done, ret, t, info, succ_any = env.reset(), False, 0.0, 0, {}, 0.0
    act_s, obs_l, act_l = [], [], []
    while not done:
        quant.begin_step()
        if recorder is not None:
            recorder.start_step(t)
        t1 = time.perf_counter()
        action = agent.act(obs, t0=(t == 0), eval_mode=True, task=tidx)   # returns a CPU tensor (synchronizes)
        act_s.append(time.perf_counter() - t1)
        if recorder is not None:
            recorder.end_step()
        obs_l.append(obs)
        act_l.append(action)
        obs, reward, done, info = env.step(action)
        ret += float(reward)
        succ_any = max(succ_any, float(info.get("success", 0.0)))
        t += 1
    quant.flush_calls()
    obs_l.append(obs)
    a = np.asarray(act_s) * 1000.0
    out = {"ret": ret, "success": float(info.get("success", 0.0)), "success_any": succ_any, "steps": t,
           "nq": int(quant.nq), "act_ms_mean": float(a.mean()), "act_ms_median": float(np.median(a)),
           "act_ms_first": float(a[0]),
           "_obs": torch.stack([torch.as_tensor(o) for o in obs_l]).float(),
           "_act": torch.stack([torch.as_tensor(x) for x in act_l]).float()}
    if log_calls:
        calls = quant.call_log or []
        out["calls_patterns"] = sorted({str(sorted(cc.items())) for cc in calls})
    quant.call_log = None
    return out


# ----------------------------------------------------------------------------------------------------------------------
# quantizer ranges (E0 calib_lib.load_split + quant_ranges_from_calib with the task index)
def load_rows(files):
    """Root latents Zu (T_total, L), row -> root index idx, candidate action sequences A (rows, H, A_dim) of the given
    held recordings (calib_lib.load_split)."""
    zs, idx, acts, base = [], [], [], 0
    for f in files:
        d = np.load(f)
        z, c = d["z_root"], d["cand"]   # (T, L), (T, I, K, H, A)
        T, I, K, H, A = c.shape
        zs.append(z)
        idx.append(base + np.repeat(np.arange(T), I * K))
        acts.append(c.reshape(T * I * K, H, A))
        base += T
    return np.concatenate(zs), np.concatenate(idx), np.concatenate(acts)


@torch.no_grad()
def quant_ranges_mt(agent, quant, files, task, q_hi=Q_HI, max_rows=MAX_ROWS):
    """Per-layer (lo, hi) = (min, 99.99th percentile) of the planning-time activations of a 12000-row subsample of the
    recorded rows (E0 / R2-8 procedure, unchanged apart from the task index)."""
    model = agent.model
    tt = task_tensor(task)
    Zu, idx, A = load_rows(files)
    n_rows = int(len(idx))
    Zu, idx, A = (torch.as_tensor(x, device=L.DEV) for x in (Zu, idx, A))
    rng = np.random.default_rng(1)
    sel = torch.as_tensor(np.sort(rng.choice(len(idx), size=min(max_rows, len(idx)), replace=False)), device=L.DEV)
    acts = {n: [] for n in L.LAYERS}
    quant.set("record")
    z = Zu[idx[sel]]
    a = A[sel]
    for h in range(A.shape[1]):
        quant.last.clear()
        model.reward(z, a[:, h], tt)
        z_next = model.next(z, a[:, h], tt)
        for n in L.LAYERS:
            acts[n].append(quant.last[n].flatten())
        z = z_next
    quant.off()
    qr, info = {}, {}
    for n in L.LAYERS:
        v = torch.cat(acts[n])
        kk = max(1, int(round((1 - q_hi) * v.numel())))
        hi = float(torch.topk(v, kk).values[-1])
        qr[n] = (float(v.min()), hi)
        info[n] = {"max": float(v.max()), "p999": float(torch.quantile(v[:2_000_000].float(), 0.999)),
                   "mean": float(v.mean()), "n": int(v.numel()), "width": int(quant.mods[n].out_features)}
    return qr, {"layers": info, "n_rows_total": n_rows, "n_rows_used": int(len(sel)), "files": len(files)}


# ----------------------------------------------------------------------------------------------------------------------
# planning-time prediction errors (copies of qlib._rollout / plan_inputs / pred_errors with the task index)
def _rollout_mt(model, z, acts, task, quant=None, setting=None, seed=None, H_table=None):
    if quant is not None:
        if setting is None or setting[0] in (None, "none"):
            quant.off()
        else:
            quant.set(setting[0], setting[1], seed=seed, table_rows=H_table)
    zs, rs = [z], []
    for j in range(len(acts)):
        if quant is not None:
            quant.hforce = j if quant.mode == "quant" and quant.qvar in ("shstep", "shep") else None
        rs.append(model.reward(zs[-1], acts[j], task))
        zs.append(model.next(zs[-1], acts[j], task))
    if quant is not None:
        quant.off()
    return zs, rs


@torch.no_grad()
def plan_inputs_mt(agent, cands, task, K=5):
    """qlib.plan_inputs: start latents and length-K action sequences of recorded first-iteration candidate sets; steps
    beyond the planning horizon use the unquantized policy-prior mean along the unquantized rollout."""
    model = agent.model
    Z0 = torch.cat([z.unsqueeze(0).expand(a.shape[1], -1) for z, a in cands])
    A = torch.cat([a for _, a in cands], dim=1)
    H = A.shape[0]
    if K > H:
        zs, _ = _rollout_mt(model, Z0, A, task)
        z = zs[-1]
        ext = []
        for _ in range(K - H):
            _, info = model.pi(z, task)
            a = info["mean"]
            ext.append(a)
            z = model.next(z, a, task)
        A = torch.cat([A, torch.stack(ext)])
    return Z0, A


@torch.no_grad()
def pred_errors_mt(agent, cfg, quant, variant, k, Z0, A, seeds, task, ks=(1, 3, 5), chunk=20000, h1=None):
    """qlib.pred_errors (no Ztrue): relative open-loop j-step latent errors err{j}, the 1-step latent error at every
    horizon position j < h1 of the unquantized trajectory err1h (normalized by the variance of all z_0,j+1) and the
    1-step reward error errr; stochastic variants: mean over the dither seeds (+ per-seed sd)."""
    model = agent.model
    K = A.shape[0]
    h1 = K if h1 is None else min(int(h1), K)
    N = Z0.shape[0]
    quant.off()
    ref_z = [[] for _ in range(K + 1)]
    ref_r = [[] for _ in range(K)]
    for s in range(0, N, chunk):
        zs, rs = _rollout_mt(model, Z0[s:s + chunk], A[:, s:s + chunk], task)
        for j in range(K + 1):
            ref_z[j].append(zs[j])
        for j in range(K):
            ref_r[j].append(L.two_hot_inv(rs[j], cfg))
    ref_z = [torch.cat(x) for x in ref_z]
    ref_r = [torch.cat(x) for x in ref_r]
    var_z = {j: L._var_rows(ref_z[j]) for j in ks}
    var_d = L._var_rows(torch.cat(ref_z[1:h1 + 1]))
    var_r = float(torch.cat(ref_r[:h1]).var())
    out_runs = []
    v = L.canon_variant(variant)
    runs = seeds if v in L.STOCHASTIC else seeds[:1]
    for sd in runs:
        acc = {f"err{j}": 0.0 for j in ks}
        acc.update(err1h=0.0, errr=0.0)
        for s in range(0, N, chunk):
            sl = slice(s, s + chunk)
            zs, _ = _rollout_mt(model, Z0[sl], A[:, sl], task, quant, (v, k), seed=sd,
                                H_table=max(K, int(cfg.horizon)))
            for j in ks:
                acc[f"err{j}"] += float(((zs[j] - ref_z[j][sl]) ** 2).sum())
            quant.set(v, k, seed=sd + 1, table_rows=max(K, int(cfg.horizon)))
            for j in range(h1):
                quant.hforce = j if quant.qvar in ("shstep", "shep") else None
                zin, a = ref_z[j][sl], A[j, sl]
                acc["err1h"] += float(((model.next(zin, a, task) - ref_z[j + 1][sl]) ** 2).sum())
                acc["errr"] += float(((L.two_hot_inv(model.reward(zin, a, task), cfg) - ref_r[j][sl]) ** 2).sum())
            quant.off()
        r = {f"err{j}": acc[f"err{j}"] / N / max(var_z[j], 1e-12) for j in ks}
        r["err1h"] = acc["err1h"] / (h1 * N) / max(var_d, 1e-12)
        r["errr"] = acc["errr"] / (h1 * N) / max(var_r, 1e-12)
        out_runs.append(r)
    res = {m: float(np.mean([r[m] for r in out_runs])) for m in out_runs[0]}
    res["sd"] = {m: float(np.std([r[m] for r in out_runs])) for m in out_runs[0]}
    res["nseeds"] = len(out_runs)
    return res
