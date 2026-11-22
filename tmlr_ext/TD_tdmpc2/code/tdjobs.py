"""TMLR extension (tmlr_ext/DESIGN_tmlr.md) job types for the TD-MPC2 5M group td, called from runner.py.

prep    per (task, ckpt), DESIGN_tmlr 0.6. Inputs: the pod's re-recorded calibration episodes 0-4 (seeds 11000-11004,
        E0 convention, calib_lib.record_calib_episode) and the 12,000-row subsample of their planning-time inputs used
        by E0 (calib_lib.quant_ranges_from_calib: numpy default_rng(1), sorted, x 3 horizon steps, reward then next).
          - re-derived per-tensor range [min, 99.99th pct] (E0 rule) and its difference to the stored E0 range file
          - PC: per-unit [min_j, 99.99th pct_j] over the 36,000 activation rows (E5 perchan: 4th largest of 36,000,
            hi_j >= lo_j + 1e-6)
        Output: <out>/prep/<task>_s<ckpt>.npz (pc_lo_/pc_hi_<layer>, re-derived and E0 ranges).
bias    per (task, ckpt), after prep; gated by the driver until the coordinator fixed the iteration rule (decision
        D2, tmlr_ext/LOG.md and DESIGN_tmlr 11: the 3-sweep procedure of 0.6 did not converge in any group). Bias vectors
        b for range kind PT (the E0 file = E1 grid) and PC, k = 1..8, from the 12,000 range rows x 3 steps: stages
        alpha = 1, 1/2, 1/4 (rule file), each from b = 0; every pass rolls the b-corrected det model (all 4 layers) and
        the unquantized model from the same start latents with the same action sequences, measures m (per-unit mean of
        corrected quantized - unquantized output) and r = mean_j |m_j| / delta_j per layer; the first pass with r < 1%
        in every layer gives the final b, otherwise b <- b + alpha m (at most 60 updates per stage). No stage below 1%:
        the stage-final b with the smallest worst-layer r, flagged bc_unconverged. Recorded per cell: stage, updates,
        converged, rows, per pass and layer mean / max |m_j| / delta_j (PC also mean|m| / mean delta), mean_j |b_j| /
        delta_j, per layer MSE (raw, / mean square, / delta^2) and the propagated latent error at b = 0 and final b.
        Output: <out>/prep/<task>_s<ckpt>_bias.npz (b_ and m0_<PT|PC>_k<k>_<layer>: final b, pass-0 residual).
e13     held-state diagnostics, DESIGN_tmlr 3.2. Unquantized episodes on the held seeds (default planner, no probe);
        a pass-through recorder keeps the first and the last MPPI iteration's candidate set on every OEI step (every
        4th MW / 10th DMC control step: E1 density) and 8 evenly spaced kept states are used (E1 rule). Every cell
        (variant, range kind, k) of the spec gets, on both sets:
          - latent errors err1 / err3 / err5 / err1h and E1's errr: qlib.pred_errors on qlib.plan_inputs (K = 5)
          - rew_med: median over (state, horizon step j < H, candidate, dither seed) of |r_q - r_0| / sd_cand(r_0), the
            1-step reward from the unquantized rollout's latent z_0,j (E1 errr inputs), sd over the set's candidates
          - candidate return error e(a) = J_0(a) - J_q(a) (value estimates under the same torch seed, i.e. the same
            Q pair and policy noise): common part mean(e), common_frac = mean(e)^2 / mean(e^2), sigma_e = sd(e),
            sigma_J = sd(J_0), ratio = sigma_e / sigma_J (sampled candidates, the policy-prior slots excluded, as E1)
          - rho_p: correlation of e(a) between two evaluations of the same set in a row (quantizer state continues as
            in consecutive MPPI iterations: fresh draws advance, tables and det stay; E1 rho_iter), sampled candidates
          - rank preservation: Spearman(J_q, J_0) over all candidates, elite overlap |top64(J_q) & top64(J_0)| / 64
        Stochastic variants: mean over the 3 E1 dither seeds 9_000_001 + 1000 seed + j. Per-state values are averaged
        over the 8 states of the episode (cell value = mean over the episode means).
timing  E15 descriptive: wall time per control decision (agent.act, CUDA synchronized) of the default planner at
        iterations 6 / 12 / 24 for none, det, sh-iter, det+shift at the task's primary k, one process; the driver runs
        it alone at the end of the queue (GPU otherwise idle).
"""
import json
import os
import time

import numpy as np
import torch

Q_HI = 0.9999
EPS_RANGE = 1e-6
KS_BIAS = tuple(range(1, 9))
N_SWEEPS = 3


def _atomic_npz(path, **arrs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}.npz"
    np.savez(tmp, **arrs)
    os.replace(tmp, path)


def calib_rows(big, task, ckpt, cal_eps, dev, max_rows=12000):
    """The E0 range rows: start latents (R, L) and candidate action sequences (R, H, A) (quant_ranges_from_calib)."""
    from calib_lib import load_split
    Zu, idx, A = load_split(big, task, ckpt, cal_eps)
    Zu, idx, A = (torch.as_tensor(x, device=dev) for x in (Zu, idx, A))
    rng = np.random.default_rng(1)
    sel = torch.as_tensor(np.sort(rng.choice(len(idx), size=min(max_rows, len(idx)), replace=False)), device=dev)
    return Zu[idx[sel]], A[sel]


@torch.no_grad()
def record_acts(L, model, quant, z, a):
    """Unquantized rollout over the H steps of a (R, H, A): per layer the outputs, concatenated in step order (R*H, W)."""
    quant.set("record")
    acts = {n: [] for n in L.LAYERS}
    for h in range(a.shape[1]):
        quant.last.clear()
        model.reward(z, a[:, h], None)
        z_next = model.next(z, a[:, h], None)
        for n in L.LAYERS:
            acts[n].append(quant.last[n].float())
        z = z_next
    quant.off()
    return {n: torch.cat(v) for n, v in acts.items()}


RULE_FILE = os.path.join(os.environ.get("ATM", "/workspace/atm"), "bias_rule.json")


def load_rule(spec):
    if spec.get("rule"):
        return dict(spec["rule"])
    return json.load(open(RULE_FILE))


def _detail(L, sq, cnt, ref_ms, delta, zq, zs0):
    """MSE of (corrected quantized - unquantized output) per layer: raw, / mean square of the unquantized output,
    / delta^2 (per unit for PC: mean_j MSE_j / delta_j^2); propagated latent error per step (E1 err definition)."""
    out = {}
    for n in L.LAYERS:
        mse_j = sq[n] / cnt[n]                                   # (W,) per-unit MSE
        d2 = (delta[n] ** 2) if torch.is_tensor(delta[n]) else float(delta[n]) ** 2
        out[n] = {"mse": float(mse_j.mean()), "mse_rel_ms": float(mse_j.mean()) / ref_ms[n],
                  "mse_over_delta2": float((mse_j / d2).mean()) if torch.is_tensor(d2) else float(mse_j.mean()) / d2}
    lat = [float(((zq[h] - zs0[h + 1]) ** 2).sum(-1).mean()) /
           max(float(((zs0[h + 1] - zs0[h + 1].mean(0)) ** 2).sum(-1).mean()), 1e-12) for h in range(len(zq))]
    return {"layers": out, "latent_err_by_step": lat}


@torch.no_grad()
def bias_stage(L, model, quant, z0, a, qrange, k, ref_mean, ref_ms, zs0, delta, eta, until, cap):
    """One stage of the D2 rule (tmlr_ext/LOG.md D2, runpod_tmlr/bias_rule_D2.json): b = 0; at every pass s = 0..cap
    the b-corrected det model (all 4 layers) and the unquantized model are rolled H steps from the same start latents
    with the same action sequences; m_j = per-unit mean over rows and steps of (corrected quantized - unquantized output)
    and r = mean_j |m_j| / delta_j per layer; the first pass with r < until in every layer ends the stage (converged,
    that pass's b is final); otherwise b <- b + eta * m (all layers at once); after cap updates the stage ends
    unconverged with the b measured at pass cap."""
    W = {n: quant.mods[n].out_features for n in L.LAYERS}
    b = {n: torch.zeros(W[n], device=z0.device) for n in L.LAYERS}
    passes, m0, d0 = [], None, None
    for s in range(int(cap) + 1):
        sums = {n: torch.zeros(W[n], dtype=torch.float64, device=z0.device) for n in L.LAYERS}
        sq = {n: torch.zeros(W[n], dtype=torch.float64, device=z0.device) for n in L.LAYERS}
        cnt = {n: 0 for n in L.LAYERS}
        step = {n: 0 for n in L.LAYERS}

        def cap_hook(n):
            def f(mod, inp, out):     # registered after the quantizer's hook: sees the corrected quantized output
                h = step[n]
                step[n] += 1
                sums[n] += out.double().sum(0)
                sq[n] += ((out - ref_by_step[n][h]) ** 2).double().sum(0)
                cnt[n] += int(out.shape[0])
            return f

        hs = [quant.mods[n].register_forward_hook(cap_hook(n)) for n in L.LAYERS]
        zq = []
        try:
            quant.qrange, quant.bias = qrange, b
            quant.set("detbc", k)
            z = z0
            for h in range(a.shape[1]):
                model.reward(z, a[:, h], None)
                z = model.next(z, a[:, h], None)
                zq.append(z)
        finally:
            quant.off()
            for h_ in hs:
                h_.remove()
        m = {n: (sums[n] / cnt[n]).float() - ref_mean[n] for n in L.LAYERS}
        rec = {}
        for n in L.LAYERS:
            if torch.is_tensor(delta[n]):
                rel = m[n].abs() / delta[n]
                rec[n] = {"mean_rel": float(rel.mean()), "max_rel": float(rel.max()),
                          "mean_abs_over_mean_delta": float(m[n].abs().mean() / delta[n].mean())}
            else:
                rel = m[n].abs() / float(delta[n])
                rec[n] = {"mean_rel": float(rel.mean()), "max_rel": float(rel.max())}
        worst = max(rec[n]["mean_rel"] for n in L.LAYERS)
        passes.append({"pass": s, "worst": worst, "layers": rec})
        if s == 0:
            m0 = {n: m[n].clone() for n in L.LAYERS}
            d0 = _detail(L, sq, cnt, ref_ms, delta, zq, zs0)
        if worst < float(until) or s == int(cap):
            det = _detail(L, sq, cnt, ref_ms, delta, zq, zs0)
            quant.bias = {}
            return {"b": b, "m0": m0, "passes": passes, "n_updates": s, "converged": bool(worst < float(until)),
                    "worst_final": worst, "at_b0": d0, "at_final": det}
        b = {n: b[n] + float(eta) * m[n] for n in L.LAYERS}


ref_by_step = {}   # layer -> list of (R, W) unquantized outputs per step (set by job_bias)


@torch.no_grad()
def job_bias(spec, out, A, big):
    """Bias vectors under the coordinator's rule (D2: stages alpha = 1, 1/2, 1/4, each from b = 0, until every layer's
    r < 1% or 60 updates; if no stage converges, the stage-final b with the smallest worst-layer r, flagged
    bc_unconverged). Writes b and the pass-0 residual m0 of every (range kind, k) to <out>/prep/<task>_s<ckpt>_bias.npz."""
    import qlib as L
    import runner as R
    agent, quant = A["agent"], A["quant"]
    task, ckpt = spec["task"], int(spec["ckpt"])
    rule = load_rule(spec)
    stages = rule["stages"] if "stages" in rule else [rule]
    t0 = time.time()
    L.set_planner(agent, None)
    quant.off()
    model = agent.model
    z0, a = calib_rows(big, task, ckpt, spec.get("cal_eps", [0, 1, 2, 3, 4]), L.DEV)
    ref = record_acts(L, model, quant, z0, a)
    R_ = z0.shape[0]
    ref_by_step.clear()
    for n in L.LAYERS:
        ref_by_step[n] = [ref[n][h * R_:(h + 1) * R_] for h in range(a.shape[1])]
    ref_mean = {n: ref[n].double().mean(0).float() for n in L.LAYERS}
    ref_ms = {n: float((ref[n].double() ** 2).mean()) for n in L.LAYERS}
    del ref
    zs0, _ = L._rollout(model, z0, a.transpose(0, 1))
    pt = R.load_ranges(out, task, ckpt, "e1")
    pc = R.load_ranges(out, task, ckpt, "pc")
    arrs, info, unconv = {}, {"PT": {}, "PC": {}}, []
    for rk, qr in (("PT", {n: (float(pt[n][0]), float(pt[n][1])) for n in L.LAYERS}), ("PC", pc)):
        for k in KS_BIAS:
            nl = (1 << k) - 1
            delta = {n: (qr[n][1] - qr[n][0]) / nl for n in L.LAYERS}
            res = []
            chosen = None
            for si, st in enumerate(stages):
                r = bias_stage(L, model, quant, z0, a, qr, k, ref_mean, ref_ms, zs0, delta, st["eta"],
                               st.get("until_below", 0.01), st.get("max_sweeps", 60))
                res.append(r)
                if r["converged"]:
                    chosen = si
                    break
            flag = chosen is None
            if flag:
                chosen = min(range(len(res)), key=lambda i: res[i]["worst_final"])
                unconv.append(f"{rk}k{k}")
            fin = res[chosen]
            for n in L.LAYERS:
                arrs[f"b_{rk}_k{k}_{n}"] = fin["b"][n].cpu().numpy()
                arrs[f"m0_{rk}_k{k}_{n}"] = res[0]["m0"][n].cpu().numpy()
            info[rk][str(k)] = {
                "converged": not flag, "bc_unconverged": flag, "stage_index": chosen,
                "alpha": float(stages[chosen]["eta"]), "n_updates": fin["n_updates"],
                "resid_mean_abs_rel_delta_max_layer": fin["worst_final"], "check_lt_0.01": bool(fin["worst_final"] < 0.01),
                "rows": int(R_), "rows_x_steps": int(R_ * a.shape[1]),
                "b_mean_abs_rel_delta": {n: float((fin["b"][n].abs() / delta[n]).mean()) if torch.is_tensor(delta[n])
                                         else float(fin["b"][n].abs().mean()) / float(delta[n]) for n in L.LAYERS},
                "at_b0": res[0]["at_b0"], "at_final": fin["at_final"],
                "stages": [{"alpha": float(stages[i]["eta"]), "converged": x["converged"], "n_updates": x["n_updates"],
                            "worst_final": x["worst_final"], "passes": x["passes"]} for i, x in enumerate(res)]}
    ref_by_step.clear()
    path = R.bias_path(out, task, ckpt)
    _atomic_npz(path, **arrs)
    quant.off()
    worst = max(x["resid_mean_abs_rel_delta_max_layer"] for rk in info for x in info[rk].values())
    by_stage = {}
    for rk in info:
        for x in info[rk].values():
            key = "unconverged" if x["bc_unconverged"] else f"alpha={x['alpha']:g}"
            by_stage[key] = by_stage.get(key, 0) + 1
    return {"path": path, "rule": rule, "n_rows": int(R_), "bias": info, "worst_resid": worst,
            "unconverged": unconv, "n_cells_unconverged": len(unconv), "cells_by_stage": by_stage,
            "max_updates": max(x["n_updates"] for rk in info for x in info[rk].values()),
            "secs_compute": time.time() - t0,
            "brief": f"by stage {by_stage}, unconverged {unconv}, worst final resid/delta {worst:.2e}"}


@torch.no_grad()
def job_prep(spec, out, A, big):
    import qlib as L
    import runner as R
    agent, quant = A["agent"], A["quant"]
    task, ckpt = spec["task"], int(spec["ckpt"])
    t0 = time.time()
    L.set_planner(agent, None)
    quant.off()
    model = agent.model
    z0, a = calib_rows(big, task, ckpt, spec.get("cal_eps", [0, 1, 2, 3, 4]), L.DEV)
    ref = record_acts(L, model, quant, z0, a)
    pt = R.load_ranges(out, task, ckpt, "e1")
    arrs, rng_info = {}, {}
    pc = {}
    for n in L.LAYERS:
        v = ref[n]
        flat = v.reshape(-1)
        kk = max(1, int(round((1 - Q_HI) * flat.numel())))
        hi_rd, lo_rd = float(torch.topk(flat, kk).values[-1]), float(flat.min())
        lo_c = v.min(0).values
        kc = max(1, int(round((1 - Q_HI) * v.shape[0])))
        hi_c = torch.maximum(torch.topk(v, kc, dim=0).values[-1], lo_c + EPS_RANGE)
        pc[n] = (lo_c, hi_c)
        lo0, hi0 = pt[n]
        rng_info[n] = {"e0_lo": lo0, "e0_hi": hi0, "rederived_lo": lo_rd, "rederived_hi": hi_rd,
                       "lo_minus_e0": lo_rd - lo0, "hi_minus_e0": hi_rd - hi0,
                       "hi_minus_e0_rel_width": (hi_rd - hi0) / (hi0 - lo0), "n_values": int(flat.numel()),
                       "pc_kth": kc, "pt_kth": kk, "pc_lo_mean": float(lo_c.mean()), "pc_hi_mean": float(hi_c.mean()),
                       "pc_hi_min": float(hi_c.min()), "pc_hi_max": float(hi_c.max()),
                       "pc_width_mean_rel_pt": float((hi_c - lo_c).mean()) / (hi0 - lo0)}
        arrs[f"pc_lo_{n}"] = lo_c.cpu().numpy()
        arrs[f"pc_hi_{n}"] = hi_c.cpu().numpy()
        arrs[f"rd_{n}"] = np.array([lo_rd, hi_rd], np.float64)
        arrs[f"e0_{n}"] = np.array([lo0, hi0], np.float64)
    del ref
    path = R.prep_path(out, task, ckpt)
    _atomic_npz(path, **arrs)
    quant.off()
    return {"path": path, "n_rows": int(z0.shape[0]), "horizon": int(a.shape[1]), "ranges": rng_info,
            "secs_compute": time.time() - t0,
            "brief": "rows %d PT hi - E0 (rel width) %s" % (int(z0.shape[0]), {n: round(v["hi_minus_e0_rel_width"], 4)
                                                                             for n, v in rng_info.items()})}


# ---------------------------------------------------------------------------------------------------------------- E13
class SetRecorder:
    """Pass-through wrapper of agent._estimate_value (like calib_lib.CandRecorder; no random numbers drawn): keeps
    (t, z_root, first-iteration actions, last-iteration actions) of every control step t % stride == 0. Step and
    iteration come from the call count (TDMPC2._plan calls _estimate_value once per MPPI iteration)."""

    def __init__(self, agent, stride, n_it):
        self.agent = agent
        self.prev = agent.__dict__.get("_estimate_value")
        self.inner = agent._estimate_value
        agent._estimate_value = self
        self.stride, self.n_it, self.c, self.kept, self.cur = int(stride), int(n_it), 0, [], None

    def __call__(self, z, actions, task):
        t, i = divmod(self.c, self.n_it)
        self.c += 1
        if t % self.stride == 0:
            if i == 0:
                self.cur = [t, z[0].clone(), actions.clone(), None]
            if i == self.n_it - 1:
                self.cur[3] = actions.clone()
                self.kept.append(tuple(self.cur))
        return self.inner(z, actions, task)

    def detach(self):
        if self.prev is None:
            del self.agent._estimate_value
        else:
            self.agent._estimate_value = self.prev


def _corr(x, y):
    xc, yc = x - x.mean(), y - y.mean()
    return float((xc * yc).sum() / torch.sqrt((xc ** 2).sum() * (yc ** 2).sum()).clamp(min=1e-12))


@torch.no_grad()
def value_metrics(L, est, quant, v, k, z_root, cand, vseed, dseeds, n_pi, n_el):
    """Candidate-return-error metrics of one candidate set (see module docstring), mean over the dither seeds."""
    N = cand.shape[1]
    z = z_root.unsqueeze(0).repeat(N, 1)
    quant.off()
    torch.manual_seed(vseed)
    v0 = est(z, cand, None).squeeze(-1).nan_to_num(0)
    J = v0[n_pi:]
    top0 = set(torch.topk(v0, n_el).indices.tolist())
    acc = []
    for sd in dseeds:
        quant.set(v, k, seed=sd)
        torch.manual_seed(vseed)
        vq1 = est(z, cand, None).squeeze(-1).nan_to_num(0)
        torch.manual_seed(vseed)
        vq2 = est(z, cand, None).squeeze(-1).nan_to_num(0)
        quant.off()
        e1, e2 = (v0 - vq1)[n_pi:], (v0 - vq2)[n_pi:]
        se = float(e1.std())
        sj = float(J.std())
        acc.append({"e_mean": float(e1.mean()), "common_frac": float(e1.mean() ** 2 / (e1 ** 2).mean().clamp(min=1e-30)),
                    "sigma_e": se, "sigma_J": sj, "ratio": se / max(sj, 1e-12), "rho_p": _corr(e1, e2),
                    "spear": L.spearman_t(vq1, v0),
                    "elite_overlap": len(top0 & set(torch.topk(vq1, n_el).indices.tolist())) / n_el})
    out = {m: float(np.mean([r[m] for r in acc])) for m in acc[0]}
    out["prod"] = out["rho_p"] * out["ratio"]          # per-state product (auxiliary; Dg1 uses the cell means)
    return out


@torch.no_grad()
def reward_med(L, model, cfg, quant, v, k, Z0, A, N, dseeds):
    """rew_med pieces: normalized absolute 1-step reward errors |r_q - r_0| / sd_cand(r_0) on the unquantized latents
    z_0,j, j < H (all candidates of every state), for every dither seed; returns the pooled values (1-d tensor)."""
    H = int(cfg.horizon)
    quant.off()
    zs, rs = L._rollout(model, Z0, A[:H])
    r0 = [L.two_hot_inv(r, cfg) for r in rs]                        # (S*N,) per step
    vals = []
    for sd in dseeds:
        quant.set(v, k, seed=sd, table_rows=max(H, A.shape[0]))
        for j in range(H):
            quant.hforce = j if quant.qvar in L.TABLE_VARIANTS else None
            rq = L.two_hot_inv(model.reward(zs[j], A[j], None), cfg)
            d = (rq - r0[j]).abs().reshape(-1, N)
            sd_c = r0[j].reshape(-1, N).std(1, keepdim=True).clamp(min=1e-12)
            vals.append((d / sd_c).reshape(-1))
        quant.off()
    return torch.cat(vals)


def cell_key(v, rk, k):
    return "none" if v == "none" else f"{rk}k{int(k)}_{v}"


@torch.no_grad()
def job_e13(spec, out, A):
    import qlib as L
    import runner as R
    agent, cfg, quant, probe = A["agent"], A["cfg"], A["quant"], A["probe"]
    task, ckpt = spec["task"], int(spec["ckpt"])
    L.set_planner(agent, None)
    quant.H = int(agent.cfg.horizon)
    H, n_it = int(agent.cfg.horizon), int(agent.cfg.iterations)
    n_pi, n_el = int(agent.cfg.num_pi_trajs), int(agent.cfg.num_elites)
    stride, n_states, nseeds = int(spec["stride"]), int(spec.get("n_states", 8)), int(spec.get("nseeds", 3))
    est = probe.inner if probe is not None else agent._estimate_value
    if probe is not None:
        probe.start(False)
    ranges = {"PT": R.load_ranges(out, task, ckpt, "e1"), "PC": R.load_ranges(out, task, ckpt, "pc")}
    biases = {}
    eps = []
    for ep in spec["eps"]:
        seed = int(spec["seed0"]) + int(ep)
        t0 = time.time()
        rec = SetRecorder(agent, stride, n_it)
        try:
            r = L.run_episode(agent, cfg, quant, "none", 0, seed)
        finally:
            rec.detach()
        kept = rec.kept
        idx = np.linspace(0, len(kept) - 1, min(n_states, len(kept))).round().astype(int)
        st = [kept[i] for i in idx]
        erec = {"ep": int(ep), "seed": seed, "ret": r["ret"], "success": r["success"],
                "success_any": r["success_any"], "steps": r["steps"], "n_kept": len(kept),
                "state_t": [int(s[0]) for s in st], "secs_loop": time.time() - t0, "cells": {}, "sets": {}}
        dseeds_all = [9_000_001 + 1000 * seed + j for j in range(nseeds)]
        for si, sname in enumerate(("first", "last")):
            cands = [(s[1], s[2] if sname == "first" else s[3]) for s in st]
            N = cands[0][1].shape[1]
            quant.off()
            Z0, Ap = L.plan_inputs(agent, cands, K=5)
            # unquantized context of the set: candidate value sd and top1-top2 gap / sd (near-tie density)
            ctx = []
            for j, (zr, cand) in enumerate(cands):
                torch.manual_seed(70_000_000 + 100 * seed + 10 * j + si)
                v0 = est(zr.unsqueeze(0).repeat(N, 1), cand, None).squeeze(-1).nan_to_num(0)
                o = torch.argsort(v0, descending=True)
                sdv = float(v0[n_pi:].std())
                ctx.append({"sigma_J": sdv, "gap": float(v0[o[0]] - v0[o[1]]) / max(sdv, 1e-12)})
            erec["sets"][sname] = {m: float(np.mean([c[m] for c in ctx])) for m in ctx[0]}
            for cell in spec["cells"]:
                v, rk, k = L.canon_variant(cell[0]), cell[1], int(cell[2])
                key = cell_key(v, rk, k)
                if v == "none":
                    continue
                quant.qrange = ranges[rk]
                if v in L.BIAS_VARIANTS:
                    if (rk, k) not in biases:
                        biases[(rk, k)] = R.load_bias(out, task, ckpt, {"PT": "e1", "PC": "pc"}[rk], k)
                    quant.bias = biases[(rk, k)]
                else:
                    quant.bias = {}
                dseeds = dseeds_all if v in L.STOCHASTIC else dseeds_all[:1]
                pe = L.pred_errors(agent, cfg, quant, v, k, Z0, Ap, dseeds_all, h1=H)
                m = {kk: pe[kk] for kk in ("err1", "err3", "err5", "err1h", "errr")}
                rv = reward_med(L, agent.model, cfg, quant, v, k, Z0, Ap, N, dseeds)
                m["rew_med"] = float(rv.median())
                m["rew_mean"] = float(rv.mean())
                vm = [value_metrics(L, est, quant, v, k, zr, cand, 70_000_000 + 100 * seed + 10 * j + si, dseeds,
                                    n_pi, n_el) for j, (zr, cand) in enumerate(cands)]
                for mm in vm[0]:
                    m[mm] = float(np.mean([x[mm] for x in vm]))
                m["nseeds"] = len(dseeds)
                erec["cells"].setdefault(key, {})[sname] = m
        quant.off()
        quant.bias = {}
        erec["secs"] = time.time() - t0
        eps.append(erec)
        torch.cuda.empty_cache()
    quant.off()
    quant.bias = {}
    n_c = len({cell_key(L.canon_variant(c[0]), c[1], c[2]) for c in spec["cells"] if c[0] != "none"})
    return {"episodes": eps, "brief": f"{len(eps)} held eps, {n_c} cells, ret "
                                       f"{np.mean([e['ret'] for e in eps]):.1f} succ {sum(e['success'] for e in eps):.0f}"}


# ------------------------------------------------------------------------------------------------------------- timing
@torch.no_grad()
def job_timing(spec, out, get_agent):
    import copy
    import qlib as L
    import runner as R
    A = get_agent(spec["task"], spec["ckpt"])
    agent, cfg, quant = A["agent"], A["cfg"], A["quant"]
    k = int(spec["k"])
    quant.qrange = R.load_ranges(out, spec["task"], spec["ckpt"], "e1")
    quant.bias = {}
    res = {}
    for it in spec["iterations"]:
        for v in spec["variants"]:
            L.set_planner(agent, {"iterations": it})
            c = copy.deepcopy(cfg)
            c.seed = int(spec["seed"])
            env = L.make_env(c)
            L.seed_all(int(spec["seed"]))
            quant.set(v, k, seed=7 * int(spec["seed"]) + 13)
            obs, dts = env.reset(), []
            for t in range(int(spec["warmup"]) + int(spec["steps"])):
                quant.begin_step()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                a = agent.act(obs, t0=(t == 0), eval_mode=True)
                torch.cuda.synchronize()
                if t >= int(spec["warmup"]):
                    dts.append(time.perf_counter() - t0)
                obs, _, done, _ = env.step(a)
                if done:
                    break
            quant.off()
            res[f"it{it}|{v}"] = {"iterations": it, "variant": v, "n": len(dts),
                                  "ms_median": 1000 * float(np.median(dts)), "ms_mean": 1000 * float(np.mean(dts)),
                                  "ms_p90": 1000 * float(np.percentile(dts, 90))}
    L.set_planner(agent, None)
    return {"timing": res, "brief": " ".join(f"{kk}={x['ms_median']:.1f}ms" for kk, x in res.items())}
