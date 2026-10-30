"""Paper A main experiment, TD-MPC2 part: shared library (E0/E1, reused by E2/E3/E5).

Extended copy of the pilot library pilots/R2_8b_crn_dither/code/nt_lib.py. The quantizer keeps the pilot's code paths
bit for bit for det / ind / shfwd (= sh-iter) / shstep / shep (same generator calls in the same order) and adds det+shift.

Quantization (design 4.1-4.2): k-bit uniform grid on the hidden activations (Mish outputs) of the 4 hidden NormedLinear
layers of the dynamics and reward MLPs (dyn0, dyn1, rew0, rew1). Per-layer range [lo, hi] from held calibration data,
delta = (hi - lo) / (2^k - 1), values outside [lo, hi] are clipped; x = (clip(a) - lo) / delta (grid units).
Variants (all at the same k-bit grid):
  none      unquantized
  det       deterministic rounding, round(x)
  ind       stochastic rounding floor(x + u), u independent per entry (candidate x unit), fresh every forward call
  shiter    stochastic rounding, one u vector per forward call shared by all candidates (= R2-8b 'shfwd')
  shstep    stochastic rounding, shared by all candidates; one u row per (layer, horizon index) drawn at the start of
            every control step and reused by every MPPI iteration (and the policy-prior rollout) of that step
  shep      as shstep but drawn once per episode
  detshift  deterministic rounding on a randomly shifted grid: q = round(x - o), value lo + delta (q + o),
            o ~ U[-1/2, 1/2) per unit, shared by all candidates, fresh every forward call (same schedule as shiter).
            Error is a deterministic function of the input within a forward (same input -> same error) but the grid
            moves every forward call, so no error persists across MPPI iterations. 2^k levels (same bits).
Random draws use a private CUDA generator seeded per episode, so the planner's own RNG stream is the same in every
variant (paired episodes).

Instrumentation that never changes the closed loop (checked in the smoke test):
  PlanProbe   wraps agent._estimate_value; on measured control steps it re-evaluates the candidate sets of the first and
              the last MPPI iteration with the unquantized model under the same RNG state (same Q-pair and policy
              noise), and records the planner's chosen elite (wrapped gumbel_softmax_sample). CPU/CUDA RNG states are
              restored, the dither generator is never touched.
  plan reruns on measured steps: the unquantized planner from the same state, previous mean and RNG state
              (deviation) and with another seed (seed-noise floor); all state restored afterwards.
  E2/E3 (v5): the last candidate set valued again by the quantized model with an independent dither generator
              (e_indep; persistent tables kept) and, for the table variants, with every table redrawn (e_fresh); the
              planner's generator, tables, call counters and RNG states are restored (closed loop bit-identical).
"""
import copy
import os
import random
import sys

os.environ.setdefault("MUJOCO_GL", "disable")
TDMPC2_DIR = os.environ.get("TDMPC2_DIR", os.path.expanduser("~/pmx/A/tdmpc2_main/tdmpc2/tdmpc2"))
CKPT_DIR = os.environ.get("NT_CKPT", os.path.expanduser("~/pmx/A/tdmpc2_main/ckpt"))
sys.path.insert(0, TDMPC2_DIR)

import hydra.utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

hydra.utils.get_original_cwd = lambda: os.getcwd()  # parse_cfg expects a running hydra app

import tdmpc2 as _tdmpc2_module  # noqa: E402
from common import math as tmath  # noqa: E402
from common.parser import parse_cfg  # noqa: E402
from envs import make_env  # noqa: E402
from tdmpc2 import TDMPC2  # noqa: E402
from envs.wrappers.timeout import Timeout as _Timeout  # noqa: E402

# compat shim (pilot nt_lib): envs/metaworld.py assigns env.max_episode_steps, a read-only property of Timeout in this
# repo version (same value; no behavioral change)
if _Timeout.max_episode_steps.fset is None:
    _Timeout.max_episode_steps = _Timeout.max_episode_steps.setter(
        lambda self, v: setattr(self, "_max_episode_steps", v))

LAYERS = ("dyn0", "dyn1", "rew0", "rew1")
DEV = "cuda"
VARIANTS = ("none", "det", "ind", "shiter", "shstep", "shep", "detshift")
QVARIANTS = VARIANTS[1:]
# E3 (experiment 3) variants, added after E1; the 7 E1 variants above keep their exact code paths
#   indstep / indep          independent per candidate slot x horizon index x unit, table fixed for the control step /
#                            for the episode (2-axis design: independent x plan-fixed / episode-fixed)
#   shstep_<g> / shep_<g>    shared table of reduced size, fixed for the step / episode: g = u (one u per unit, all
#                            horizon positions), g<G> (G groups of contiguous units), l (one u per layer)
#   shiter_<g>               the same reduced granularity drawn fresh at every forward call (magnitude control for the
#                            table-size sweep: coarser tables also correlate the error across units)
E3_VARIANTS = ("indstep", "indep", "shstep_u", "shstep_g32", "shstep_l", "shep_u", "shep_g32", "shep_l",
               "shiter_g32", "shiter_l")
STOCHASTIC = ("ind", "shiter", "shstep", "shep", "detshift") + E3_VARIANTS
FRESH = ("ind", "shiter", "detshift", "shiter_g32", "shiter_l")    # a new dither draw at every forward call
# dither tables that stay fixed across the MPPI iterations of a control step (E2/E3 metric e_fresh redraws them)
PERSIST_TABLES = ("shstep", "shep", "indstep", "indep", "shstep_u", "shstep_g32", "shstep_l", "shep_u", "shep_g32",
                  "shep_l")
ALIASES = {"shfwd": "shiter", "sh": "shiter", "sh-iter": "shiter", "sh-fwd": "shiter", "sh-step": "shstep",
           "sh-ep": "shep", "det+shift": "detshift"}
PLANNER_KEYS = ("iterations", "num_samples", "num_elites", "num_pi_trajs", "horizon", "temperature", "min_std",
                "max_std")


def canon_variant(v):
    v = ALIASES.get(v, v)
    if v not in VARIANTS and v not in E3_VARIANTS:
        raise ValueError(f"unknown variant {v}")
    return v


def table_spec(v):
    """('step'|'ep'|'fwd', granularity) of a reduced shared-table variant, else None."""
    if v.startswith(("shstep_", "shep_", "shiter_")):
        sched, gran = v.split("_", 1)
        return {"shstep": "step", "shep": "ep", "shiter": "fwd"}[sched], gran
    return None


def reduced_row(gran, w, gen, device, dtype=torch.float32):
    """(1, w) uniform row of granularity u (per unit), g<G> (G groups of contiguous units) or l (one per layer)."""
    if gran == "u":
        return torch.rand((1, w), generator=gen, device=device, dtype=dtype)
    if gran == "l":
        return torch.rand((1, 1), generator=gen, device=device, dtype=dtype).expand(1, w)
    if gran.startswith("g"):
        g = int(gran[1:])
        return torch.rand((1, g), generator=gen, device=device, dtype=dtype).repeat_interleave(w // g, dim=1)
    raise ValueError(gran)


def is_mw(task):
    return task.startswith("mw-")


def ckpt_path(task, ckpt_seed):
    return f"{CKPT_DIR}/{task}-{int(ckpt_seed)}.pt"


def make_cfg(task, ckpt_seed=1, seed=0):
    cfg = OmegaConf.load(os.path.join(TDMPC2_DIR, "config.yaml"))
    cfg.task = task
    cfg.model_size = 5
    cfg.checkpoint = ckpt_path(task, ckpt_seed)
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


def build_agent(task, ckpt_seed=1):
    """Returns (agent, cfg). make_env fills obs/action dims and episode length into cfg before the model is built
    (same order as the pilot nt_lib.build_agent)."""
    cfg = make_cfg(task, ckpt_seed)
    make_env(cfg)
    torch.manual_seed(0)
    agent = TDMPC2(cfg)
    agent.load(cfg.checkpoint)
    agent.model.eval()
    agent._default_planner = {k: getattr(agent.cfg, k) for k in PLANNER_KEYS}
    return agent, cfg


def set_planner(agent, overrides=None):
    """Planner hyperparameters for this job (TDMPC2._plan and _estimate_value read agent.cfg at every call).
    Always resets to the defaults first, so a cached agent never keeps a previous job's overrides."""
    for k, v in agent._default_planner.items():
        setattr(agent.cfg, k, v)
    for k, v in (overrides or {}).items():
        if k not in PLANNER_KEYS:
            raise ValueError(f"unknown planner key {k}")
        setattr(agent.cfg, k, type(agent._default_planner[k])(v))
    if agent.cfg.horizon != agent._default_planner["horizon"]:
        agent._prev_mean = torch.nn.Buffer(torch.zeros(agent.cfg.horizon, agent.cfg.action_dim, device=agent.device))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_get():
    return torch.get_rng_state(), torch.cuda.get_rng_state()


def rng_set(s):
    torch.set_rng_state(s[0])
    torch.cuda.set_rng_state(s[1])


def two_hot_inv(x, cfg):
    return tmath.two_hot_inv(x, cfg).squeeze(-1)


# ----------------------------------------------------------------------------------------------------------------------
class Quantizer:
    """Forward hooks on dyn0, dyn1 (dynamics MLP) and rew0, rew1 (reward MLP); each is a NormedLinear whose output is
    Mish(LayerNorm(Linear(x))). Encoder, Q-ensemble and policy prior are never touched.
    modes: None (off), 'record' (keep the last output of each layer in self.last), 'quant'.
    qrange[name] = (lo, hi) python floats (per-tensor). Tensors of shape (width,) also work (per-channel, E5)."""

    def __init__(self, model, horizon=3, device=DEV):
        self.mods = {"dyn0": model._dynamics[0], "dyn1": model._dynamics[1],
                     "rew0": model._reward[0], "rew1": model._reward[1]}
        self.names = list(LAYERS)
        self.device = device
        self.H = int(horizon)       # planning horizon: horizon index of a call = call counter % H (per row count)
        self.calls = {}             # (layer, rows) -> forward calls since the last set() / begin_step()
        self.table = {}             # layer -> (rows, width) dither table (shstep / shep)
        self.gtable = {}            # layer -> (1, width) reduced shared table (shstep_<g> / shep_<g>)
        self.slot_table = {}        # (layer, rows) -> (H, rows, width) per-slot table (indstep / indep)
        self.hforce = None          # explicit horizon index (prediction-error code); None = infer from call counter
        self.layer_k = {}           # E5 mixed precision: per-layer bit-width override (job level; set() keeps it)
        self.wq = None              # E5 weight quantization (e5lib.WeightQuant; job level)
        self.n_refresh = 0
        self.call_log = None
        self.qk, self.qvar, self.mode = 8, "det", None
        self.qrange = {}
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(0)
        self.last = {}
        self.nq = 0                 # quantized layer calls since set() (sanity)
        self.handles = [mod.register_forward_hook(self._hook(k)) for k, mod in self.mods.items()]
        self.gtable_variants = tuple(v for v in E3_VARIANTS if table_spec(v) and table_spec(v)[0] != "fwd")
        self.fresh_coarse = tuple(v for v in E3_VARIANTS if table_spec(v) and table_spec(v)[0] == "fwd")

    def set(self, variant=None, k=8, seed=None, table_rows=None):
        """variant None/'none' -> off; 'record'; or a quantized variant at k bits. seed: private dither generator."""
        if variant == "record":
            self.mode = "record"
        elif variant is None or variant == "none":
            self.mode = None
        else:
            self.mode, self.qvar, self.qk = "quant", canon_variant(variant), int(k)
        self.calls, self.hforce, self.n_refresh, self.nq = {}, None, 0, 0
        if seed is not None:
            self.gen.manual_seed(int(seed))
            if self.wq is not None:
                self.wq.new_episode(int(seed))   # w4sr<K>: new stochastic-rounding copies per episode
        self.slot_table = {}
        if self.mode == "quant" and self.qvar in ("shstep", "shep"):
            self._refresh_table(table_rows)  # drawn after seeding: the episode's first table depends only on the seed
        if self.mode == "quant" and table_spec(self.qvar) and table_spec(self.qvar)[0] != "fwd":
            self._refresh_gtable()

    def off(self):
        self.set(None)

    def set_ste(self, k):
        """E5 QAT: det fake-quant at k bits with a straight-through gradient (training only)."""
        self.mode, self.qvar, self.qk = "ste", "det", int(k)
        self.calls, self.hforce, self.nq = {}, None, 0

    def _refresh_table(self, rows=None):
        """One uniform dither row per (layer, horizon index), shared by every candidate (row) of a forward call."""
        rows = self.H if rows is None else int(rows)
        self.table = {n: torch.rand((rows, self.mods[n].out_features), generator=self.gen, device=self.device)
                      for n in self.names}
        self.n_refresh += 1

    def _refresh_gtable(self):
        """Reduced shared table (E3 table-size sweep): one (1, width) row per layer, all candidates and horizon
        positions share it; granularity u = per unit, g<G> = G groups of contiguous units, l = one value per layer."""
        gran = table_spec(self.qvar)[1]
        self.gtable = {n: reduced_row(gran, self.mods[n].out_features, self.gen, self.device) for n in self.names}
        self.n_refresh += 1

    def begin_step(self):
        """Call before every control step (agent.act). shstep draws a new table; shep keeps the episode's table.
        E3: shstep_<g> draws a new reduced table, indstep clears its per-slot tables (redrawn at their first use)."""
        self.flush_calls()
        if self.wq is not None:
            self.wq.begin_step()
        if self.mode == "quant" and self.qvar == "shstep":
            self._refresh_table()
        if self.mode == "quant" and self.qvar.startswith("shstep_"):
            self._refresh_gtable()
        if self.mode == "quant" and self.qvar == "indstep":
            self.slot_table = {}
            self.n_refresh += 1

    def flush_calls(self):
        if self.call_log is not None and self.calls:
            self.call_log.append({f"{k[0]}x{k[1]}": v for k, v in sorted(self.calls.items())})
        self.calls = {}

    def _hook(self, name):
        def hook(module, inputs, out):
            mode = self.mode
            if mode is None:
                return None
            if mode == "record":
                self.last[name] = out
                return None
            lo, hi = self.qrange[name]
            nl = (1 << self.layer_k.get(name, self.qk)) - 1
            d = (hi - lo) / nl
            x = (out.clamp(lo, hi) - lo) * (1.0 / d)
            if mode == "ste":     # E5 QAT: det forward, identity backward
                return out + (lo + d * torch.round(x).clamp(0, nl) - out).detach()
            v = self.qvar
            self.nq += 1
            key = (name, int(out.shape[0]))
            c = self.calls.get(key, 0)
            self.calls[key] = c + 1
            if v == "det":        # deterministic rounding: the same error for every candidate with the same input
                q = torch.round(x)
            elif v == "ind":      # stochastic rounding, dither independent per entry, fresh every forward
                q = torch.floor(x + torch.rand(out.shape, generator=self.gen, device=out.device, dtype=out.dtype))
            elif v == "shiter":   # one dither vector per forward call shared by all candidates (pilot 'shfwd')
                q = torch.floor(x + torch.rand((1, out.shape[-1]), generator=self.gen, device=out.device,
                                               dtype=out.dtype))
            elif v in ("shstep", "shep"):  # shared by all candidates, row = horizon index of this forward call
                # MPPI calls (num_samples rows) come in order t = 0..H-1 per iteration; the policy-prior rollout
                # (num_pi_trajs rows) calls dyn for t = 0..H-2, so call counter % H is the horizon index per row count
                h = self.hforce if self.hforce is not None else c % self.H
                q = torch.floor(x + self.table[name][h].to(out.dtype))
            elif v == "detshift":  # deterministic rounding on a grid shifted by a shared offset, fresh every forward
                o = torch.rand((1, out.shape[-1]), generator=self.gen, device=out.device, dtype=out.dtype) - 0.5
                return lo + d * (torch.round(x - o).clamp(0, nl) + o)
            elif v in ("indstep", "indep"):  # E3: independent per candidate slot, fixed for the step / episode
                if self.hforce is not None:  # one-pass error evaluation: same law as ind (i.i.d. per entry)
                    u = torch.rand(out.shape, generator=self.gen, device=out.device, dtype=out.dtype)
                else:
                    tab = self.slot_table.get(key)
                    if tab is None:          # drawn at the first call of this row count in the step / episode
                        tab = torch.rand((self.H,) + tuple(out.shape), generator=self.gen, device=out.device,
                                         dtype=out.dtype)
                        self.slot_table[key] = tab
                    u = tab[c % self.H]
                q = torch.floor(x + u)
            elif v in self.fresh_coarse:     # E3: reduced granularity, fresh every forward call
                q = torch.floor(x + reduced_row(table_spec(v)[1], out.shape[-1], self.gen, out.device, out.dtype))
            elif v in self.gtable_variants:  # E3: reduced shared table (table-size sweep)
                q = torch.floor(x + self.gtable[name].to(out.dtype))
            else:
                raise ValueError(v)
            return lo + d * q.clamp(0, nl)

        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


# ----------------------------------------------------------------------------------------------------------------------
def spearman_t(a, b):
    """Spearman rank correlation of two 1-d tensors (average ranks are not needed: ties have probability ~0)."""
    ra = torch.empty_like(a)
    ra[a.argsort()] = torch.arange(len(a), device=a.device, dtype=a.dtype)
    rb = torch.empty_like(b)
    rb[b.argsort()] = torch.arange(len(b), device=b.device, dtype=b.dtype)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / torch.sqrt((ra ** 2).sum() * (rb ** 2).sum()).clamp(min=1e-12))


def mppi_weights(v, n_elite, temp):
    """Elite indices and MPPI weights exactly as in TDMPC2._plan (topk, softmax(temperature (V - max V)))."""
    val, idx = torch.topk(v, n_elite)
    w = torch.exp(temp * (val - val.max()))
    return idx, w / w.sum()


def set_metrics(vq, v0, n_elite, temp, n_pi, vb=None):
    """Decision-level metrics of one candidate set: quantized values vq vs unquantized v0 (same candidates, same Q-pair
    and policy noise). Cost error e = J_hat - J with cost = -value, i.e. e = v0 - vq. 'rand' = the sampled candidates
    (columns n_pi.., the policy-prior proposals excluded). var_cand (DESIGN_main v5.1 8.3) = the across-candidate
    variance of e over the sampled candidates (= e_rand_sd^2; f_var_cand on the first MPPI iteration's set)."""
    e = v0 - vq
    er = e[n_pi:]
    sd0 = float(v0[n_pi:].std())
    o0 = torch.argsort(v0, descending=True)
    top0, top1 = int(o0[0]), int(o0[1])
    pick = int(vq.argmax())
    iq, wq = mppi_weights(vq, n_elite, temp)
    i0, _ = mppi_weights(v0, n_elite, temp)
    s0, sq = set(i0.tolist()), set(iq.tolist())
    out = {"e_rand": float(er.mean()), "e_rand_sd": float(er.std()), "var_cand": float(er.var()),
           "e_all": float(e.mean()),
           "e_top1": float(e[pick]), "e_elite_w": float((wq * e[iq]).sum()), "e_elite": float(e[iq].mean()),
           "sd_v0": sd0, "v0_mean": float(v0[n_pi:].mean()),
           "spear": spearman_t(vq, v0), "flip": int(pick != top0), "jac": len(s0 & sq) / len(s0 | sq),
           "gap": float(v0[top0] - v0[top1]) / max(sd0, 1e-12),
           "regret": float(v0[top0] - v0[pick]) / max(sd0, 1e-12),
           "regret_w": float(v0[top0] - (wq * v0[iq]).sum()) / max(sd0, 1e-12)}
    if vb is not None:  # seed-noise floor of the value estimator itself (unquantized, other Q-pair / policy noise)
        pb = int(vb.argmax())
        out.update(sn_spear=spearman_t(vb, v0), sn_flip=int(pb != top0),
                   sn_e_rand_sd=float((v0 - vb)[n_pi:].std()))
    return out


# One process-wide pass-through wrapper of common.math.gumbel_softmax_sample (TDMPC2._plan's final elite draw). It
# hands the drawn index to the PlanProbe that is currently planning; installing it once (instead of one wrapper per
# probe) keeps it correct when cached agents and their probes are created and evicted in any order.
_GUMBEL_ORIG = _tdmpc2_module.math.gumbel_softmax_sample
_ACTIVE_PROBE = {"probe": None}


def _gumbel_hook(p, *a, **kw):
    idx = _GUMBEL_ORIG(p, *a, **kw)
    pr = _ACTIVE_PROBE["probe"]
    if pr is not None and pr.active:
        pr.rec["chosen"] = idx
    return idx


_tdmpc2_module.math.gumbel_softmax_sample = _gumbel_hook


class PlanProbe:
    """Wraps agent._estimate_value (instance attribute); receives TDMPC2._plan's final elite draw via _gumbel_hook.

    start(active) before a control step. When active, the candidate sets of the first and the last MPPI iteration are
    re-evaluated with the quantizer off under the RNG state the planner's own call started from (so the unquantized
    value uses the same Q-pair and policy-prior noise); the first set also once more under another seed (estimator
    seed noise). The CPU and CUDA RNG states after the planner's call are restored, so the planner's stream is unchanged;
    the dither generator is not used (quantizer off). keep_cands keeps the first-iteration candidates and root latent
    (planning-time prediction errors at episode end)."""

    def __init__(self, agent, quant):
        self.agent, self.quant = agent, quant
        self.inner = agent._estimate_value
        agent._estimate_value = self
        self.active, self.it, self.rec, self.keep_cands, self.sn_seed = False, 0, {}, False, 0
        self.rec_ref = None         # E2 (b3): a list while an unquantized reference plan records its candidate sets

    def start(self, active, keep_cands=False, sn_seed=0):
        self.active, self.it, self.rec, self.keep_cands, self.sn_seed = bool(active), 0, {}, keep_cands, int(sn_seed)
        if self.active:
            _ACTIVE_PROBE["probe"] = self
        elif _ACTIVE_PROBE["probe"] is self:
            _ACTIVE_PROBE["probe"] = None

    def __call__(self, z, actions, task):
        i = self.it
        self.it += 1
        n_it = int(self.agent.cfg.iterations)
        if self.rec_ref is not None:      # E2 (b3) reference plan: keep the first / last candidate sets and values
            s0 = rng_get()
            v = self.inner(z, actions, task)
            if i == 0 or i == n_it - 1:
                self.rec_ref.append((i, z, actions.clone(), s0, v.squeeze(-1).nan_to_num(0)))
            return v
        if not (self.active and (i == 0 or i == n_it - 1)):
            return self.inner(z, actions, task)
        s0 = rng_get()
        vq = self.inner(z, actions, task)
        s1 = rng_get()
        mode = self.quant.mode
        self.quant.mode = None
        rng_set(s0)
        v0 = self.inner(z, actions, task)
        vb = None
        if i == 0:
            torch.manual_seed(self.sn_seed)
            vb = self.inner(z, actions, task)
        self.quant.mode = mode
        rng_set(s1)
        if i == 0:
            self.rec["first"] = (vq.squeeze(-1).nan_to_num(0), v0.squeeze(-1).nan_to_num(0), vb.squeeze(-1).nan_to_num(0))
            if self.keep_cands:
                self.rec["cand"] = (z[0].clone(), actions.clone())
        if i == n_it - 1:
            self.rec["last"] = (vq.squeeze(-1).nan_to_num(0), v0.squeeze(-1).nan_to_num(0))
            self.rec["last_in"] = (z, actions.clone(), s0)    # for the independent-draw re-evaluation (E2)
        return vq

    def independent_reeval(self, seed, fresh_tables=False):
        """E2 metric: the last iteration's candidates valued again by the quantized model under the RNG state of the
        planner's own call (same Q-pair and policy noise) but with an independent dither draw (a separate generator
        seeded with `seed`, so the planner's dither stream is untouched). Fresh-noise variants (ind, sh-iter, det+shift)
        get a new draw; det and the table variants (sh-step, sh-ep, E3 tables) are deterministic given their table and
        reproduce the planner's values exactly (the DINO-WM rule: persistent error fields are kept).
        fresh_tables: the table variants' tables are redrawn from the independent generator as well (e_fresh: every
        dither redrawn). The quantizer's tables, call counters and generator and all RNG states are restored, so the
        closed loop and the call log are unchanged. Returns the (N,) quantized values."""
        z, acts, s0 = self.rec["last_in"]
        q = self.quant
        post, g_saved, active = rng_get(), q.gen, self.active
        saved = (q.calls, q.nq, q.table, q.gtable, q.slot_table)
        g = torch.Generator(device=q.device)
        g.manual_seed(int(seed))
        q.gen, self.active, q.calls = g, False, dict(q.calls)
        if fresh_tables and q.mode == "quant":
            if q.qvar in ("shstep", "shep"):
                q.table = {n: torch.rand(t.shape, generator=g, device=t.device, dtype=t.dtype) for n, t in q.table.items()}
            elif q.qvar in q.gtable_variants:
                gran = table_spec(q.qvar)[1]
                q.gtable = {n: reduced_row(gran, q.mods[n].out_features, g, q.device) for n in q.names}
            elif q.qvar in ("indstep", "indep"):
                q.slot_table = {}        # redrawn lazily from g at the first call of each row count
        rng_set(s0)
        vq2 = self.inner(z, acts, None).squeeze(-1).nan_to_num(0)
        q.gen, self.active = g_saved, active
        q.calls, q.nq, q.table, q.gtable, q.slot_table = saved
        rng_set(post)
        return vq2

    def summary(self, indep_seed=None):
        """Per-step metrics of the recorded step (call after agent.act). indep_seed: also the E2 independent-draw
        re-evaluation of the executed elite (e_indep, oei_indep, sel_indep) and, for the variants with persistent
        tables, the re-evaluation with every table redrawn (e_fresh, oei_fresh, sel_fresh)."""
        cfg = self.agent.cfg
        n_el, temp, n_pi = int(cfg.num_elites), float(cfg.temperature), int(cfg.num_pi_trajs)
        vq0, v00, vb0 = self.rec["first"]
        vqL, v0L = self.rec["last"]
        f = set_metrics(vq0, v00, n_el, temp, n_pi, vb0)
        last = set_metrics(vqL, v0L, n_el, temp, n_pi)
        iq, wq = mppi_weights(vqL, n_el, temp)
        ch = self.rec.get("chosen")
        e_ch = float((v0L - vqL)[iq[int(ch)]]) if ch is not None else float("nan")
        out = {f"f_{k}": v for k, v in f.items()}
        out.update({f"l_{k}": v for k, v in last.items()})
        # OEI (design 4.3): cost error at the planner's final choice minus the mean cost error of the random
        # candidates of the first distribution of the same plan; negative = the planner picked favourable error
        out.update(e_chosen=e_ch, oei=e_ch - f["e_rand"], oei_w=last["e_elite_w"] - f["e_rand"],
                   oei_top1=last["e_top1"] - f["e_rand"], oei_first=f["e_elite_w"] - f["e_rand"])
        # E2 (recorded in every E2 run): oei = sel_last + drift exactly
        out.update(sel_last=e_ch - last["e_rand"], drift=last["e_rand"] - f["e_rand"])
        if ch is not None:   # policy-prior share (slots 0..n_pi-1 keep their action sequence in every MPPI iteration)
            pi_el = iq < n_pi
            out.update(chosen_pi=float(int(iq[int(ch)]) < n_pi), elite_pi=float(pi_el.float().mean()),
                       elite_pi_w=float(wq[pi_el].sum()))
        if indep_seed is not None and ch is not None:
            vq2 = self.independent_reeval(indep_seed)
            e_in = float((v0L - vq2)[iq[int(ch)]])
            out.update(e_indep=e_in, oei_indep=e_in - f["e_rand"], sel_indep=e_in - last["e_rand"],
                       indep_same=float(torch.equal(vq2, vqL)))
            q = self.quant
            if q.mode == "quant" and q.qvar in PERSIST_TABLES:
                vq3 = self.independent_reeval(int(indep_seed) + 1_000_003, fresh_tables=True)
                e_fr = float((v0L - vq3)[iq[int(ch)]])
                out.update(e_fresh=e_fr, oei_fresh=e_fr - f["e_rand"], sel_fresh=e_fr - last["e_rand"],
                           fresh_same=float(torch.equal(vq3, vqL)))
        return out

    def detach(self):
        if self.agent.__dict__.get("_estimate_value") is self:
            del self.agent._estimate_value
        if _ACTIVE_PROBE["probe"] is self:
            _ACTIVE_PROBE["probe"] = None


# ----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def region_reference(agent, quant, probe, obs_dev, t0, prev_mean, rng_pre, seed):
    """E2 layer (b3) region-effect control (DESIGN_main v5.2 9.1): the unquantized planner from the same observation,
    previous mean and RNG state as the real step; its first and last candidate sets are valued by the quantized model
    under the RNG state of the reference planner's own call (same Q-pair and policy noise), with an independent dither
    generator (fresh variants) or the step's persistent table / det field (as e_indep). drift_ref = mean cost error of
    the sampled candidates of its last set minus that of its first set: how the error field changes along the path the
    planner takes without quantization. The caller restores agent._prev_mean and the RNG state; the quantizer's
    generator, tables and call counters are restored here."""
    mode, active = quant.mode, probe.active
    quant.mode, probe.active, probe.rec_ref, probe.it = None, False, [], 0
    try:
        rng_set(rng_pre)
        agent._prev_mean.copy_(prev_mean)
        agent._plan(obs_dev, t0=t0, eval_mode=True)
        recs = probe.rec_ref
    finally:
        probe.rec_ref = None
        quant.mode, probe.active = mode, active
    n_pi, n_it = int(agent.cfg.num_pi_trajs), int(agent.cfg.iterations)
    q = quant
    saved = (q.gen, q.calls, q.nq, q.table, q.gtable, q.slot_table)
    g = torch.Generator(device=q.device)
    g.manual_seed(int(seed))
    q.gen, q.calls = g, dict(q.calls)
    er = {}
    try:
        for i, z, acts, s0, v0 in recs:
            rng_set(s0)
            vq = probe.inner(z, acts, None).squeeze(-1).nan_to_num(0)
            er[i] = float((v0 - vq)[n_pi:].mean())
    finally:
        q.gen, q.calls, q.nq, q.table, q.gtable, q.slot_table = saved
    f, l = er.get(0), er.get(n_it - 1)
    return {"ref_f_e_rand": f, "ref_l_e_rand": l,
            "drift_ref": (l - f) if f is not None and l is not None else float("nan")}


@torch.no_grad()
def plan_rerun(agent, quant, probe, obs_dev, t0, prev_mean, rng_pre, alt_seed):
    """Unquantized plans from the same observation and previous mean: with the planner RNG state of the real step
    (deviation reference) and with another seed (seed-noise floor). Returns (a_ref, mean_ref, a_alt, mean_alt).
    The caller restores the RNG state and agent._prev_mean afterwards."""
    mode, active = quant.mode, probe.active if probe is not None else False
    quant.mode = None
    if probe is not None:
        probe.active = False
    rng_set(rng_pre)
    agent._prev_mean.copy_(prev_mean)
    a0 = agent._plan(obs_dev, t0=t0, eval_mode=True).flatten().clone()
    m0 = agent._prev_mean.clone()
    torch.manual_seed(int(alt_seed))
    agent._prev_mean.copy_(prev_mean)
    a1 = agent._plan(obs_dev, t0=t0, eval_mode=True).flatten().clone()
    m1 = agent._prev_mean.clone()
    quant.mode = mode
    if probe is not None:
        probe.active = active
    return a0, m0, a1, m1


def rms(x, y):
    return float(((x - y) ** 2).mean().sqrt())


@torch.no_grad()
def run_episode(agent, cfg, quant, variant, k, seed, probe=None, oei_stride=0, pd_stride=0, keep_cands=False,
                warm_start=True, log_calls=False, indep_reeval=False, region_ref=False):
    """One eval episode (fresh env with cfg.seed = seed; random/numpy/torch seeded with seed; dither generator seeded
    with 7 seed + 13, as in the pilots). The policy acts with agent.act(obs, t0, eval_mode=True).
    oei_stride > 0: PlanProbe active on steps t % oei_stride == 0 (OEI and decision metrics).
    pd_stride > 0: plan reruns on steps t % pd_stride == 0 (plan deviation vs unquantized and seed-noise floor).
    warm_start False: every step plans from a zero mean (t0=True), E2 control.
    indep_reeval: on OEI steps also re-evaluate the executed elite with an independent dither draw (E2 metric).
    region_ref: on OEI steps also the E2 (b3) region-effect reference (region_reference; drift_ref, drift_minus_ref)."""
    c = copy.deepcopy(cfg)
    c.seed = seed
    env = make_env(c)
    seed_all(seed)
    quant.set(variant, k, seed=seed * 7 + 13)
    quant.call_log = [] if log_calls else None
    obs, done, ret, t, info, succ_any = env.reset(), False, 0.0, 0, {}, 0.0
    obs_l, act_l, rew_l, steps, pds, sig = [], [], [], [], [], []
    while not done:
        quant.begin_step()
        if log_calls and quant.mode == "quant" and quant.qvar in ("shstep", "shep") and t < 4:
            sig.append([round(float(quant.table[n].sum()), 4) for n in ("dyn0", "rew1")])
        m_oei = probe is not None and oei_stride > 0 and t % oei_stride == 0
        m_pd = pd_stride > 0 and t % pd_stride == 0
        m_ref = bool(region_ref) and m_oei
        t0 = (t == 0) or not warm_start
        if m_pd:
            rng_pre, prev = rng_get(), agent._prev_mean.clone()
        if m_ref:
            rng_pre_r, prev_r = rng_get(), agent._prev_mean.clone()
        if probe is not None:
            probe.start(m_oei, keep_cands=keep_cands, sn_seed=1_000_003 + 101 * seed + t)
        action = agent.act(obs, t0=t0, eval_mode=True)
        if m_oei:
            s = probe.summary(indep_seed=(3_000_017 + 101 * seed + t) if indep_reeval else None)
            s["t"] = t
            if keep_cands:
                s["_cand"] = probe.rec["cand"]
            steps.append(s)
        if m_pd:
            rng_post, mean_q = rng_get(), agent._prev_mean.clone()
            a0, m0, a1, m1 = plan_rerun(agent, quant, probe, obs.to(agent.device).unsqueeze(0), t0, prev, rng_pre,
                                        2_000_003 + 101 * seed + t)
            pds.append({"t": t, "dev_mean": rms(mean_q[0], m0[0]), "dev_act": rms(action.to(a0.device), a0),
                        "sn_mean": rms(m1[0], m0[0]), "sn_act": rms(a1, a0)})
            agent._prev_mean.copy_(mean_q)
            rng_set(rng_post)
        if m_ref:
            rng_post_r, mean_post_r = rng_get(), agent._prev_mean.clone()
            ref = region_reference(agent, quant, probe, obs.to(agent.device).unsqueeze(0), t0, prev_r, rng_pre_r,
                                   4_000_037 + 101 * seed + t)
            s.update(ref)
            s["drift_minus_ref"] = s["drift"] - ref["drift_ref"]
            agent._prev_mean.copy_(mean_post_r)
            rng_set(rng_post_r)
        if probe is not None:
            probe.start(False)
        obs_l.append(obs)
        act_l.append(action)
        obs, reward, done, info = env.step(action)
        rew_l.append(float(reward))
        ret += float(reward)
        succ_any = max(succ_any, float(info.get("success", 0.0)))
        t += 1
    quant.flush_calls()
    obs_l.append(obs)
    out = {"ret": ret, "success": float(info.get("success", 0.0)), "success_any": succ_any, "steps": t,
           "n_refresh": quant.n_refresh, "steps_rec": steps, "pd_rec": pds,
           "_obs": torch.stack([torch.as_tensor(o) for o in obs_l]).float(),
           "_act": torch.stack([torch.as_tensor(a) for a in act_l]).float(),
           "_rew": torch.as_tensor(rew_l, dtype=torch.float32)}
    if log_calls:
        calls = quant.call_log or []
        out.update(calls_patterns=sorted({str(sorted(cc.items())) for cc in calls}), table_sig=sig)
    quant.call_log = None
    return out


# ----------------------------------------------------------------------------------------------------------------------
# prediction errors (design 4.3): relative to the unquantized prediction, normalized by the variance of the reference
def _rollout(model, z, acts, quant=None, setting=None, seed=None, H_table=None):
    """Open-loop latent rollout z_{j+1} = next(z_j, a_j) for j < len(acts), rewards r_j = reward(z_j, a_j) (quantizer
    setting applied when given; shstep/shep use table row j for step j, as in planning)."""
    if quant is not None:
        if setting is None or setting[0] in (None, "none"):
            quant.off()
        else:
            quant.set(setting[0], setting[1], seed=seed, table_rows=H_table)
    zs, rs = [z], []
    for j in range(len(acts)):
        if quant is not None:
            quant.hforce = j if quant.mode == "quant" and quant.qvar in ("shstep", "shep") else None
        rs.append(model.reward(zs[-1], acts[j], None))
        zs.append(model.next(zs[-1], acts[j], None))
    if quant is not None:
        quant.off()
    return zs, rs


def _var_rows(x):
    return float(((x - x.mean(0)) ** 2).sum(-1).mean())


@torch.no_grad()
def pred_errors(agent, cfg, quant, variant, k, Z0, A, seeds, Ztrue=None, ks=(1, 3, 5), chunk=20000, h1=None):
    """Relative k-step errors of the quantized model vs the unquantized one from start latents Z0 (N, L) with action
    sequences A (K, N, A_dim) (K = max(ks)):
      err{j}  open-loop j-step latent:  mean ||z_q,j - z_0,j||^2 / var(z_0,j)
      err1h   1-step latent at every step j < h1 of the unquantized trajectory (R2-8b 'err1' with h1 = H: input
              z_0,j, averaged over j and normalized by the variance of all z_0,j+1, j < h1)
      errr    1-step reward at every step j < h1 of the unquantized trajectory (normalized by the reward variance)
      true{j} (if Ztrue (K+1, N, L) given) j-step error vs the encoder of the true future observation for the
              quantized model, and true0_{j} for the unquantized model (model error, not only quantization error)
    Stochastic variants: mean over the given dither seeds (also returns per-seed sd)."""
    model = agent.model
    K = A.shape[0]
    h1 = K if h1 is None else min(int(h1), K)
    N = Z0.shape[0]
    quant.off()
    ref_z = [[] for _ in range(K + 1)]
    ref_r = [[] for _ in range(K)]
    for s in range(0, N, chunk):
        zs, rs = _rollout(model, Z0[s:s + chunk], A[:, s:s + chunk])
        for j in range(K + 1):
            ref_z[j].append(zs[j])
        for j in range(K):
            ref_r[j].append(two_hot_inv(rs[j], cfg))
    ref_z = [torch.cat(x) for x in ref_z]
    ref_r = [torch.cat(x) for x in ref_r]
    var_z = {j: _var_rows(ref_z[j]) for j in ks}
    var_d = _var_rows(torch.cat(ref_z[1:h1 + 1]))
    var_r = float(torch.cat(ref_r[:h1]).var())
    out_runs = []
    v = canon_variant(variant)
    runs = seeds if v in STOCHASTIC else seeds[:1]
    for sd in runs:
        acc = {f"err{j}": 0.0 for j in ks}
        acc.update(err1h=0.0, errr=0.0)
        if Ztrue is not None:
            acc.update({f"true{j}": 0.0 for j in ks})
        for s in range(0, N, chunk):
            sl = slice(s, s + chunk)
            zs, _ = _rollout(model, Z0[sl], A[:, sl], quant, (v, k), seed=sd, H_table=max(K, int(cfg.horizon)))
            for j in ks:
                acc[f"err{j}"] += float(((zs[j] - ref_z[j][sl]) ** 2).sum())
                if Ztrue is not None:
                    acc[f"true{j}"] += float(((zs[j] - Ztrue[j][sl]) ** 2).sum())
            quant.set(v, k, seed=sd + 1, table_rows=max(K, int(cfg.horizon)))
            for j in range(h1):
                quant.hforce = j if quant.qvar in ("shstep", "shep") else None
                zin, a = ref_z[j][sl], A[j, sl]
                acc["err1h"] += float(((model.next(zin, a, None) - ref_z[j + 1][sl]) ** 2).sum())
                acc["errr"] += float(((two_hot_inv(model.reward(zin, a, None), cfg) - ref_r[j][sl]) ** 2).sum())
            quant.off()
        r = {f"err{j}": acc[f"err{j}"] / N / max(var_z[j], 1e-12) for j in ks}
        r["err1h"] = acc["err1h"] / (h1 * N) / max(var_d, 1e-12)
        r["errr"] = acc["errr"] / (h1 * N) / max(var_r, 1e-12)
        if Ztrue is not None:
            for j in ks:
                vt = max(_var_rows(Ztrue[j]), 1e-12)
                r[f"true{j}"] = acc[f"true{j}"] / N / vt
        out_runs.append(r)
    res = {m: float(np.mean([r[m] for r in out_runs])) for m in out_runs[0]}
    res["sd"] = {m: float(np.std([r[m] for r in out_runs])) for m in out_runs[0]}
    res["nseeds"] = len(out_runs)
    if Ztrue is not None:
        for j in ks:
            res[f"true0_{j}"] = float(((ref_z[j] - Ztrue[j]) ** 2).sum(-1).mean()) / max(_var_rows(Ztrue[j]), 1e-12)
    return res


@torch.no_grad()
def traj_inputs(agent, obs, act, K=5):
    """Dataset-type inputs from a trajectory: start latents enc(o_t), executed action windows a_t..a_{t+K-1} and the
    encoder latents of the true future observations, for every start t with t + K <= T."""
    model = agent.model
    Z = model.encode(obs.to(DEV), None)
    T = act.shape[0]
    n = T - K + 1
    if n <= 0:
        return None
    Ad = act.to(DEV)
    A = torch.stack([Ad[j:j + n] for j in range(K)])          # (K, n, A_dim)
    Zt = torch.stack([Z[j:j + n] for j in range(K + 1)])       # (K+1, n, L)
    return Z[:n], A, Zt


@torch.no_grad()
def plan_inputs(agent, cands, K=5):
    """Planning-time inputs from recorded first-iteration candidate sets [(z_root, actions (H, N, A))]: start latents
    and action sequences of length K; the steps beyond the planning horizon use the unquantized policy-prior mean along
    the unquantized rollout (fixed actions, identical for every variant)."""
    model = agent.model
    Z0 = torch.cat([z.unsqueeze(0).expand(a.shape[1], -1) for z, a in cands])
    A = torch.cat([a for _, a in cands], dim=1)
    H = A.shape[0]
    if K > H:
        zs, _ = _rollout(model, Z0, A)
        z = zs[-1]
        ext = []
        for _ in range(K - H):
            _, info = model.pi(z, None)
            a = info["mean"]
            ext.append(a)
            z = model.next(z, a, None)
        A = torch.cat([A, torch.stack(ext)])
    return Z0, A
