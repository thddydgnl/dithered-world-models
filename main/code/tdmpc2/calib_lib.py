"""Calibration-split jobs for runner.py: calibration rollouts, quantizer ranges, held-set prediction errors and the
fixed-state decision analysis. Logic copied from the pilots (R2-8b calib.py, prep.py / prep_r28b.py, decision.py) and
adapted to qlib (checkpoint seeds, 7 variants, 1/3/5-step errors).

Calibration episodes of one (task, ckpt): ep 0-4 -> quantizer ranges; ep 5-9 -> held prediction-error set and the fixed
decision states. None of them shares a seed with the held (k-selection) or eval episodes (see make_manifest.py).
"""
import copy
import glob
import json
import os
import re
import time

import numpy as np
import torch

import qlib as L

K_PER_ITER = 16
STATES_PER_EP = 50
N_HELD = 60000
CHUNK = 20000
MATCH_SEED = 4242


class CandRecorder:
    """Temporarily wraps agent._estimate_value to record MPPI candidate sets (pilot nt_lib.CandRecorder); restores the
    previous wrapper (PlanProbe) on detach. Indices come from numpy, so torch's RNG is not consumed."""

    def __init__(self, agent, K, seed):
        self.agent = agent
        self.prev = agent.__dict__.get("_estimate_value")
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
        if self.prev is None:
            del self.agent._estimate_value
        else:
            self.agent._estimate_value = self.prev


@torch.no_grad()
def record_calib_episode(agent, cfg, quant, seed, path, seed_rec=0):
    """Unquantized eval episode (same loop as qlib.run_episode) that records the planning-time inputs (pilot calib.py)."""
    quant.off()
    T = int(cfg.episode_length)
    stride = max(1, T // STATES_PER_EP)
    dec_steps = set(range(stride // 2, T, stride))
    rec = CandRecorder(agent, K_PER_ITER, seed=seed_rec)
    S = {k: [] for k in ("z", "cand", "obs", "act", "dec_t", "dec_obs", "dec_prev", "dec_z", "dec_c0", "dec_cL",
                         "dec_act", "dec_mean")}
    c = copy.deepcopy(cfg)
    c.seed = seed
    env = L.make_env(c)
    L.seed_all(seed)
    obs, done, ret, t, info, succ_any, rews = env.reset(), False, 0.0, 0, {}, 0.0, []
    t_start = time.time()
    try:
        while not done:
            rec.start_step(t in dec_steps)
            if t in dec_steps:
                S["dec_prev"].append(agent._prev_mean.detach().clone())
            action = agent.act(obs, t0=(t == 0), eval_mode=True)
            z, it, first, last = rec.end_step()
            S["z"].append(z)
            S["cand"].append(it)
            S["obs"].append(obs.clone())
            S["act"].append(action.clone())
            if t in dec_steps:
                S["dec_t"].append(t)
                S["dec_obs"].append(obs.clone())
                S["dec_z"].append(z)
                S["dec_c0"].append(first)
                S["dec_cL"].append(last)
                S["dec_act"].append(action.clone())
                S["dec_mean"].append(agent._prev_mean.detach().clone())
            obs, reward, done, info = env.step(action)
            rews.append(float(reward))
            ret += float(reward)
            succ_any = max(succ_any, float(info.get("success", 0.0)))
            t += 1
    finally:
        rec.detach()
    np_ = lambda xs: torch.stack([torch.as_tensor(x) for x in xs]).float().cpu().numpy()
    obs_seq = np_(S["obs"] + [obs])
    summary = {"seed": seed, "ret": ret, "success": float(info.get("success", 0.0)), "success_any": succ_any,
               "steps": t, "secs": time.time() - t_start, "n_dec": len(S["dec_t"])}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}.npz"
    np.savez_compressed(
        tmp, z_root=np_(S["z"]), cand=np_(S["cand"]), obs_seq=obs_seq, act_seq=np_(S["act"]),
        rew_seq=np.asarray(rews, np.float32), dec_t=np.asarray(S["dec_t"], np.int32),
        dec_t0=np.asarray([tt == 0 for tt in S["dec_t"]], bool), dec_obs=np_(S["dec_obs"]),
        dec_prev=np_(S["dec_prev"]), dec_z=np_(S["dec_z"]), dec_c0=np_(S["dec_c0"]), dec_cL=np_(S["dec_cL"]),
        dec_act=np_(S["dec_act"]), dec_mean=np_(S["dec_mean"]), summary=json.dumps(summary))
    os.replace(tmp, path)
    return summary


def calib_file(out, task, ckpt, ep):
    return f"{out}/calib/{task}_s{int(ckpt)}_ep{int(ep)}.npz"


def load_split(out, task, ckpt, eps):
    """Planning-time inputs of the given calibration episodes: root latents Zu (T_total, L), row -> root index idx and
    candidate action sequences A (rows, H, A_dim) (pilot prep.load_split)."""
    zs, idx, acts, base = [], [], [], 0
    for ep in eps:
        d = np.load(calib_file(out, task, ckpt, ep))
        z, c = d["z_root"], d["cand"]  # (T, Lz), (T, I, K, H, A)
        T, I, K, H, A = c.shape
        zs.append(z)
        idx.append(base + np.repeat(np.arange(T), I * K))
        acts.append(c.reshape(T * I * K, H, A))
        base += T
    return np.concatenate(zs), np.concatenate(idx), np.concatenate(acts)


@torch.no_grad()
def quant_ranges_from_calib(agent, quant, out, task, ckpt, cal_eps, q_hi=0.9999, max_rows=12000):
    """Per-layer (lo, hi) = (min, 99.99th percentile) of the planning-time activations of a 12000-row subsample of the
    calibration episodes (R2-8 prep_r28.quant_ranges, unchanged)."""
    model = agent.model
    Zu, idx, A = load_split(out, task, ckpt, cal_eps)
    Zu, idx, A = (torch.as_tensor(x, device=L.DEV) for x in (Zu, idx, A))
    rng = np.random.default_rng(1)
    sel = torch.as_tensor(np.sort(rng.choice(len(idx), size=min(max_rows, len(idx)), replace=False)), device=L.DEV)
    acts = {n: [] for n in L.LAYERS}
    quant.set("record")
    z = Zu[idx[sel]]
    a = A[sel]
    for h in range(A.shape[1]):
        quant.last.clear()
        model.reward(z, a[:, h], None)
        z_next = model.next(z, a[:, h], None)
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
                   "mean": float(v.mean()), "n": int(v.numel())}
    return qr, info


def _plan_seqs(agent, Z0, A_hn, K=5):
    """(N, L), (H, N, A) -> action sequences of length K: candidate actions, then the unquantized policy-prior mean."""
    H = A_hn.shape[0]
    if K <= H:
        return A_hn[:K]
    zs, _ = L._rollout(agent.model, Z0, A_hn)
    z, ext = zs[-1], []
    for _ in range(K - H):
        _, info = agent.model.pi(z, None)
        ext.append(info["mean"])
        z = agent.model.next(z, info["mean"], None)
    return torch.cat([A_hn, torch.stack(ext)])


@torch.no_grad()
def held_pred_errors(agent, cfg, quant, out, spec):
    """Held-set prediction errors of every variant at one k: planning-time inputs (60k candidate sequences of the held
    calibration episodes, R2-8b prep) and trajectory inputs (executed actions + true future observations of the same
    episodes). Stochastic variants: mean of nseeds dither seeds."""
    task, ckpt, k = spec["task"], spec["ckpt"], int(spec["k"])
    held = spec.get("held_eps", [5, 6, 7, 8, 9])
    nseeds = int(spec.get("nseeds", 3))
    seeds = [MATCH_SEED + 101 * j for j in range(nseeds)]
    torch.manual_seed(12345)  # the policy-prior extension samples no noise (mean), but keep any RNG use fixed
    Zu, idx, A = load_split(out, task, ckpt, held)
    sel = np.random.default_rng(0).choice(len(idx), size=min(N_HELD, len(idx)), replace=False)
    Z0 = torch.as_tensor(Zu, device=L.DEV)[torch.as_tensor(idx[sel], device=L.DEV)]
    Ah = torch.as_tensor(A[sel], device=L.DEV).transpose(0, 1).contiguous()   # (H, N, A)
    Ap = torch.cat([_plan_seqs(agent, Z0[s:s + CHUNK], Ah[:, s:s + CHUNK]) for s in range(0, len(Z0), CHUNK)], 1)
    trajs = []
    for ep in held:
        d = np.load(calib_file(out, task, ckpt, ep))
        ti = L.traj_inputs(agent, torch.as_tensor(d["obs_seq"]), torch.as_tensor(d["act_seq"]), K=5)
        if ti is not None:
            trajs.append(ti)
    Zt0 = torch.cat([t[0] for t in trajs])
    At = torch.cat([t[1] for t in trajs], 1)
    Ztt = torch.cat([t[2] for t in trajs], 1)
    res = {"task": task, "ckpt": ckpt, "k": k, "held_eps": held, "n_plan": int(len(Z0)), "n_traj": int(len(Zt0)),
           "seeds": seeds, "plan": {}, "traj": {}}
    for v in spec.get("variants", L.QVARIANTS):
        v = L.canon_variant(v)
        res["plan"][v] = L.pred_errors(agent, cfg, quant, v, k, Z0, Ap, seeds, chunk=CHUNK, h1=int(cfg.horizon))
        res["traj"][v] = L.pred_errors(agent, cfg, quant, v, k, Zt0, At, seeds, Ztrue=Ztt, chunk=CHUNK,
                                       h1=int(cfg.horizon))
    quant.off()
    return res


@torch.no_grad()
def fixed_state_decision(agent, cfg, quant, probe, out, spec):
    """R2-8b decision analysis on fixed decision states of the held calibration episodes (same states and candidate
    sets for every variant). For each state: the first and the last MPPI candidate set of the unquantized planner are
    valued by the unquantized model (seed s), by the unquantized model with seed s+1 (estimator seed noise) and by each
    quantized variant (seed s, so the same Q-pair / policy noise); metrics via qlib.set_metrics. Then one-step plan
    reruns from the stored obs and previous mean: unquantized (seed s and s+1) and every variant (seed s); deviation of
    the first step of the final MPPI mean and of the executed action. On the first set also rho_iter (correlation of the
    candidates' cost errors between two consecutive iterations) and common_frac (share of the mean-square error that
    is common to all candidates)."""
    task, ckpt, k = spec["task"], spec["ckpt"], int(spec["k"])
    held = spec.get("held_eps", [5, 6, 7, 8, 9])
    variants = [L.canon_variant(v) for v in spec.get("variants", L.QVARIANTS)]
    keys = ("dec_t", "dec_t0", "dec_obs", "dec_prev", "dec_z", "dec_c0", "dec_cL")
    D = {kk: [] for kk in keys}
    epi = []
    for ep in held:
        d = np.load(calib_file(out, task, ckpt, ep))
        for kk in keys:
            D[kk].append(d[kk])
        epi.append(np.full(len(d["dec_t"]), ep))
    D = {kk: np.concatenate(v) for kk, v in D.items()}
    epi = np.concatenate(epi)
    n = len(epi)
    max_states, plan_states = int(spec.get("max_states", 150)), int(spec.get("plan_states", 150))
    sel = np.arange(n) if n <= max_states else np.sort(np.random.default_rng(0).choice(n, max_states, replace=False))
    N, n_el, temp, n_pi = int(cfg.num_samples), int(cfg.num_elites), float(cfg.temperature), int(cfg.num_pi_trajs)
    est = probe.inner if probe is not None else agent._estimate_value
    if probe is not None:
        probe.start(False)
    rows, plan_rows = [], []
    for j, i in enumerate(sel):
        seed = 7_000_000 + int(i)
        z = torch.as_tensor(D["dec_z"][i], device=L.DEV).unsqueeze(0).repeat(N, 1)
        base = {"state": int(i), "ep": int(epi[i]), "t": int(D["dec_t"][i])}
        for set_name, key in (("first", "dec_c0"), ("last", "dec_cL")):
            cand = torch.as_tensor(D[key][i], device=L.DEV)
            quant.off()
            torch.manual_seed(seed)
            v0 = est(z, cand, None).squeeze(-1).nan_to_num(0)
            torch.manual_seed(seed + 1)
            vb = est(z, cand, None).squeeze(-1).nan_to_num(0)
            for v in variants:
                quant.set(v, k, seed=seed)
                torch.manual_seed(seed)
                vq = est(z, cand, None).squeeze(-1).nan_to_num(0)
                row = {**base, "set": set_name, "variant": v, **L.set_metrics(vq, v0, n_el, temp, n_pi, vb)}
                if set_name == "first":
                    # persistence of the cost error across consecutive MPPI iterations (design 4.3 rho_iter): the same
                    # candidates valued again as the planner's next iteration would (quantizer state continues: fresh
                    # draws for ind / shiter / detshift, the same table for shstep / shep), same Q-pair and policy noise
                    torch.manual_seed(seed)
                    vq2 = est(z, cand, None).squeeze(-1).nan_to_num(0)
                    e1, e2 = (v0 - vq)[n_pi:], (v0 - vq2)[n_pi:]
                    e1c, e2c = e1 - e1.mean(), e2 - e2.mean()
                    row["rho_iter"] = float((e1c * e2c).sum() / torch.sqrt((e1c ** 2).sum() * (e2c ** 2).sum()).clamp(min=1e-12))
                    # share of the error that is common to all candidates (cannot reorder them)
                    row["common_frac"] = float(e1.mean() ** 2 / (e1 ** 2).mean().clamp(min=1e-12))
                    if v in ("indstep", "indep"):
                        # E3 slot tables: the same candidates in the same slots give rho_iter = 1 by construction,
                        # but the planner redraws the sampled slots' action sequences every iteration. rho_iter_perm
                        # values the sampled candidates again in randomly permuted slots (policy-prior slots stay,
                        # as in TDMPC2._plan): persistence attached to the action sequence, not to the slot. For the
                        # other variants it equals rho_iter in distribution (their error does not depend on the slot)
                        gp = torch.Generator()
                        gp.manual_seed(seed + 2)
                        perm = torch.cat([torch.arange(n_pi), n_pi + torch.randperm(N - n_pi, generator=gp)])
                        perm = perm.to(cand.device)
                        cp = cand[:, perm]
                        qm, quant.mode = quant.mode, None      # unquantized value of the permuted set (slot table kept)
                        torch.manual_seed(seed)
                        v0p = est(z, cp, None).squeeze(-1).nan_to_num(0)
                        quant.mode = qm
                        torch.manual_seed(seed)
                        vqp = est(z, cp, None).squeeze(-1).nan_to_num(0)
                        ebk = torch.empty_like(v0p)
                        ebk[perm] = v0p - vqp                   # back to the original candidate order
                        e3c = ebk[n_pi:] - ebk[n_pi:].mean()
                        row["rho_iter_perm"] = float((e1c * e3c).sum() / torch.sqrt((e1c ** 2).sum() *
                                                                                    (e3c ** 2).sum()).clamp(min=1e-12))
                quant.off()
                rows.append(row)
        if j % max(1, len(sel) // plan_states) != 0:
            continue
        obs = torch.as_tensor(D["dec_obs"][i], device=L.DEV).unsqueeze(0)
        prev = torch.as_tensor(D["dec_prev"][i], device=L.DEV)
        t0 = bool(D["dec_t0"][i])

        def plan(s, v):
            quant.set(v, k, seed=s)
            agent._prev_mean.copy_(prev)
            torch.manual_seed(s)
            a = agent._plan(obs, t0=t0, eval_mode=True).flatten().clone()
            mu = agent._prev_mean[0].clone()
            quant.off()
            return a, mu

        a0, m0 = plan(seed, "none")
        a1, m1 = plan(seed + 1, "none")
        pr = {**base, "sn_mean": L.rms(m1, m0), "sn_act": L.rms(a1, a0)}
        for v in variants:
            a, mu = plan(seed, v)
            pr[f"dev_mean_{v}"] = L.rms(mu, m0)
            pr[f"dev_act_{v}"] = L.rms(a, a0)
        plan_rows.append(pr)
    agent._prev_mean.zero_()
    summ = {}
    for v in variants:
        sv = {}
        for set_name in ("first", "last"):
            rr = [r for r in rows if r["variant"] == v and r["set"] == set_name]
            for m in rr[0]:
                if m in ("state", "ep", "t", "set", "variant"):
                    continue
                sv[f"{set_name}_{m}"] = float(np.mean([r[m] for r in rr if m in r]))
        sn = float(np.mean([p["sn_mean"] for p in plan_rows]))
        sv["dev_mean"] = float(np.mean([p[f"dev_mean_{v}"] for p in plan_rows]))
        sv["dev_mean_rel"] = sv["dev_mean"] / max(sn, 1e-12)
        sv["dev_act_rel"] = float(np.mean([p[f"dev_act_{v}"] for p in plan_rows])) / max(
            float(np.mean([p["sn_act"] for p in plan_rows])), 1e-12)
        summ[v] = sv
    from runner import columnar
    return {"task": task, "ckpt": ckpt, "k": k, "n_states": int(len(sel)), "n_plan_states": len(plan_rows),
            "seed_floor_mean": float(np.mean([p["sn_mean"] for p in plan_rows])), "summary": summ,
            "rows": columnar(rows), "plan_rows": columnar(plan_rows)}
