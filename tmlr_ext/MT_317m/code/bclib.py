"""E14 det+BC bias vector b (DESIGN_tmlr 0.5-0.6) for the mt80-317M checkpoint.

One (width,) vector per hooked layer (dyn0, dyn1, rew0, rew1) for one (task, k): the task's E10 per-tensor range and
E10 primary k (DESIGN_tmlr 0.4 R4, 0.6 "317M").
Data: the task's held none episodes (seeds 103000-103009) recorded with mtlib.HeldRecorder (E10/E0 rows): every control
step's root latent and 16 random candidates of every MPPI iteration. ALL recorded rows are used
(10 episodes x 100 steps x 6 iterations x 16 candidates = 96,000 start latents with their 3-step action sequences).
Procedure (DESIGN_tmlr 0.6):
  1. b = 0.
  2. With the current b, roll the det+BC model (all four layers quantized: det rounding minus b) and the unquantized
     model from the same start latents with the same action sequences for H = 3 steps (planning horizon). Every step
     calls the reward head and then the dynamics on the same latent, as TDMPC2._estimate_value does.
  3. Per layer, the per-unit mean over all rows and all steps of (corrected quantized output - unquantized output) is
     added to b.
  4. Steps 2-3 are done three times; the mean of every sweep is recorded. A fourth pass with the final b (nothing
     added) measures the residual after the third sweep: check 0.7-2 = mean over units of |residual| < 1% of delta.
The corrected quantized output is read by a forward hook registered AFTER the Quantizer's hooks on the same modules
(OutTap; PyTorch runs forward hooks in registration order and each one sees the previous hook's return value), so the
values are exactly what the next layer receives in the closed loop. Nothing here draws random numbers.
"""
import numpy as np
import torch

import mtlib as M
import qlib as L

N_SWEEPS = 3
BC_TOL = 0.01          # check 0.7-2: residual mean |m| / delta after the third sweep must be < 1%
TOL = 1e-9


class OutTap:
    """Forward hooks after the Quantizer's hooks: while on, keep each module's final output (post-quantizer)."""

    def __init__(self, quant):
        self.on, self.last = False, {}
        self.handles = [mod.register_forward_hook(self._hook(n)) for n, mod in quant.mods.items()]

    def _hook(self, name):
        def hook(module, inputs, out):
            if self.on:
                self.last[name] = out
            return None
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def deltas(qrange, k):
    nl = (1 << int(k)) - 1
    return {n: (float(qrange[n][1]) - float(qrange[n][0])) / nl for n in L.LAYERS}


@torch.no_grad()
def bias_sweeps(agent, quant, files, task, k, n_sweeps=N_SWEEPS, chunk=8000, alpha=1.0, stop_tol=None):
    """Returns (b {layer: (W,) float32 cuda tensor}, report dict, arrays {name: numpy} for the npz file).
    quant.qrange must hold the task's ranges. The quantizer is left off with an empty bias.
    Defaults = DESIGN_tmlr 0.6 (alpha 1, 3 sweeps, then the residual pass). alpha < 1 (b <- b + alpha m) and stop_tol
    (stop at the first pass whose residual is below stop_tol x delta in every layer, at most n_sweeps updates) are the
    damped variant, used only when the coordinator sets it in code/inputs/bias_rule.json."""
    model = agent.model
    tt = M.task_tensor(task)
    Zu, idx, A = M.load_rows(files)          # (T_total, L) root latents, (rows,) root index, (rows, H, A_dim)
    n_rows = int(len(idx))
    Zu = torch.as_tensor(Zu, device=L.DEV)
    idx = torch.as_tensor(idx, device=L.DEV)
    A = torch.as_tensor(A, device=L.DEV)
    H = int(A.shape[1])
    W = {n: int(quant.mods[n].out_features) for n in L.LAYERS}
    dl = deltas(quant.qrange, k)
    b = {n: torch.zeros(W[n], device=L.DEV, dtype=torch.float32) for n in L.LAYERS}
    tap = OutTap(quant)
    sweeps, arrays = [], {}
    try:
        for s in range(n_sweeps + 1):
            S1 = {n: torch.zeros(W[n], device=L.DEV, dtype=torch.float64) for n in L.LAYERS}
            S2 = {n: 0.0 for n in L.LAYERS}
            cnt = 0
            for c0 in range(0, n_rows, chunk):
                sl = slice(c0, min(c0 + chunk, n_rows))
                z0, a = Zu[idx[sl]], A[sl]
                # unquantized reference rollout (record mode keeps each layer's output, values unchanged)
                quant.set("record")
                U, z = [], z0
                for h in range(H):
                    quant.last.clear()
                    model.reward(z, a[:, h], tt)
                    zn = model.next(z, a[:, h], tt)
                    U.append({n: quant.last[n] for n in L.LAYERS})
                    z = zn
                quant.off()
                # det+BC rollout with the current b (all four layers quantized)
                quant.bias = b
                quant.set("detbc", k)
                tap.on, z = True, z0
                for h in range(H):
                    tap.last.clear()
                    model.reward(z, a[:, h], tt)
                    zn = model.next(z, a[:, h], tt)
                    for n in L.LAYERS:
                        dlt = tap.last[n] - U[h][n]
                        S1[n] += dlt.sum(0, dtype=torch.float64)
                        S2[n] += float((dlt.double() ** 2).sum())
                    z = zn
                tap.on = False
                quant.off()
                cnt += (sl.stop - sl.start) * H
                del U
            m = {n: (S1[n] / cnt).float() for n in L.LAYERS}
            conv = stop_tol is not None and all(float(m[n].abs().mean()) / dl[n] < stop_tol - TOL for n in L.LAYERS)
            last_pass = s == n_sweeps or conv
            rec = {"sweep": "residual_after_last" if last_pass else s + 1, "pass": s, "n_samples_per_unit": cnt,
                   "layers": {}}
            for n in L.LAYERS:
                mn = m[n]
                rec["layers"][n] = {"delta": dl[n],
                                    "mean_abs_over_delta": float(mn.abs().mean()) / dl[n],
                                    "max_abs_over_delta": float(mn.abs().max()) / dl[n],
                                    "mean_signed_over_delta": float(mn.mean()) / dl[n],
                                    "rms_over_delta": float(mn.pow(2).mean().sqrt()) / dl[n],
                                    "mse_over_delta2": S2[n] / (cnt * W[n]) / dl[n] ** 2,
                                    "b_mean_abs_over_delta": float(b[n].abs().mean()) / dl[n]}
                arrays[f"m{s + 1}_{n}"] = mn.cpu().numpy()
            sweeps.append(rec)
            if last_pass:
                break
            for n in L.LAYERS:
                b[n] = b[n] + (m[n] if alpha == 1.0 else alpha * m[n])
    finally:
        tap.remove()
        quant.off()
        quant.bias = {}
    last = sweeps[-1]["layers"]
    ok = {n: bool(last[n]["mean_abs_over_delta"] < BC_TOL - TOL) for n in L.LAYERS}
    for n in L.LAYERS:
        arrays[f"b_{n}"] = b[n].cpu().numpy()
    report = {"task": task, "k": int(k), "n_rows": n_rows, "horizon": H, "n_files": len(files),
              "n_sweeps": n_sweeps, "n_updates": len(sweeps) - 1, "alpha": alpha, "stop_tol": stop_tol,
              "sweeps": sweeps, "residual_ok_by_layer": ok, "residual_ok": all(ok.values()),
              "criterion": f"mean over units |residual| / delta < {BC_TOL} after the last sweep",
              "ranges": {n: [float(quant.qrange[n][0]), float(quant.qrange[n][1])] for n in L.LAYERS}}
    return b, report, arrays


# ----------------------------------------------------------------------------------------------------------------------
# DESIGN_tmlr deviation D2 (2026-11-21T19:01:09Z, judge agent; DESIGN_tmlr 11, LOG.md): staged iteration, replaces the
# 3-sweep rule above for the main run (bias_sweeps stays as the record of the old rule).
#   stages alpha = 1, 1/2, 1/4 in this order; every stage starts from b = 0. Pass s measures m(b_s) and per layer
#   r = mean_j |m_j| / delta; the first pass with r < until_below in EVERY layer fixes b = b_s (converged). Otherwise
#   b <- b + alpha m (all layers at once), at most max_sweeps updates; a stage that does not get there hands over to the
#   next stage. No stage converged: the stage whose last b has the smallest worst-layer r is used, flagged
#   bc_unconverged. Data (all recorded held rows, 3 steps) and the target are those of DESIGN_tmlr 0.6.
@torch.no_grad()
def _measure(model, quant, tap, Zu, idx, A, tt, b, k, W, chunk):
    """One pass: per-unit mean m of (det+BC output with b - unquantized output) over all rows and steps, the
    per-layer MSE of that difference, and the propagated latent error after every step (sum ||z_q - z_0||^2 over
    the rows divided by the rows' variance of z_0, E1 err definition)."""
    n_rows, H = int(len(idx)), int(A.shape[1])
    S1 = {n: torch.zeros(W[n], device=L.DEV, dtype=torch.float64) for n in L.LAYERS}
    S2 = {n: 0.0 for n in L.LAYERS}
    ez = [0.0] * H
    sz = [torch.zeros(Zu.shape[1], device=L.DEV, dtype=torch.float64) for _ in range(H)]
    szz = [0.0] * H
    cnt = 0
    for c0 in range(0, n_rows, chunk):
        sl = slice(c0, min(c0 + chunk, n_rows))
        z0, a = Zu[idx[sl]], A[sl]
        quant.set("record")
        U, Z0, z = [], [], z0
        for h in range(H):
            quant.last.clear()
            model.reward(z, a[:, h], tt)
            zn = model.next(z, a[:, h], tt)
            U.append({n: quant.last[n] for n in L.LAYERS})
            Z0.append(zn)
            z = zn
        quant.off()
        quant.bias = b
        quant.set("detbc", k)
        tap.on, z = True, z0
        for h in range(H):
            tap.last.clear()
            model.reward(z, a[:, h], tt)
            zn = model.next(z, a[:, h], tt)
            for n in L.LAYERS:
                dlt = tap.last[n] - U[h][n]
                S1[n] += dlt.sum(0, dtype=torch.float64)
                S2[n] += float((dlt.double() ** 2).sum())
            ez[h] += float(((zn - Z0[h]).double() ** 2).sum())
            sz[h] += Z0[h].sum(0, dtype=torch.float64)
            szz[h] += float((Z0[h].double() ** 2).sum())
            z = zn
        tap.on = False
        quant.off()
        cnt += sl.stop - sl.start
        del U, Z0
    m = {n: (S1[n] / (cnt * H)).float() for n in L.LAYERS}
    mse = {n: S2[n] / (cnt * H * W[n]) for n in L.LAYERS}
    zerr = []
    for h in range(H):
        var = szz[h] / cnt - float((sz[h] / cnt).pow(2).sum())
        zerr.append(ez[h] / cnt / max(var, 1e-12))
    return m, mse, zerr


@torch.no_grad()
def bias_staged(agent, quant, files, task, k, stages, chunk=8000):
    """D2 rule. Returns (b, report, arrays): b {layer: (W,) float32 cuda}, arrays = final b, m0 (= m at b = 0) and
    the final residual m for the npz file. quant.qrange must hold the task's ranges."""
    model = agent.model
    tt = M.task_tensor(task)
    Zu, idx, A = M.load_rows(files)
    n_rows = int(len(idx))
    Zu, idx, A = (torch.as_tensor(x, device=L.DEV) for x in (Zu, idx, A))
    W = {n: int(quant.mods[n].out_features) for n in L.LAYERS}
    dl = deltas(quant.qrange, k)
    tap = OutTap(quant)
    res, m0 = [], None
    try:
        for si, st in enumerate(stages):
            alpha, tol, max_up = float(st["eta"]), float(st["until_below"]), int(st["max_sweeps"])
            b = {n: torch.zeros(W[n], device=L.DEV, dtype=torch.float32) for n in L.LAYERS}
            passes, converged, s = [], False, 0
            for s in range(max_up + 1):
                m, mse, zerr = _measure(model, quant, tap, Zu, idx, A, tt, b, k, W, chunk)
                if m0 is None:
                    m0 = {n: m[n].clone() for n in L.LAYERS}
                r = {n: float(m[n].abs().mean()) / dl[n] for n in L.LAYERS}
                passes.append({"pass": s, "r_mean": r, "r_max": {n: float(m[n].abs().max()) / dl[n] for n in L.LAYERS},
                               "mse_over_delta2": {n: mse[n] / dl[n] ** 2 for n in L.LAYERS},
                               "b_mean_abs_over_delta": {n: float(b[n].abs().mean()) / dl[n] for n in L.LAYERS},
                               "latent_err": zerr})
                if all(r[n] < tol - TOL for n in L.LAYERS):
                    converged = True
                    break
                if s < max_up:
                    for n in L.LAYERS:
                        b[n] = b[n] + (m[n] if alpha == 1.0 else alpha * m[n])
            res.append({"eta": alpha, "until_below": tol, "max_sweeps": max_up, "converged": converged,
                        "n_updates": s, "passes": passes, "worst_r_last": max(passes[-1]["r_mean"].values()),
                        "_b": {n: b[n].clone() for n in L.LAYERS}, "_m": {n: m[n].clone() for n in L.LAYERS}})
            if converged:
                break
    finally:
        tap.remove()
        quant.off()
        quant.bias = {}
    conv = [i for i, x in enumerate(res) if x["converged"]]
    pick = conv[0] if conv else min(range(len(res)), key=lambda i: res[i]["worst_r_last"])
    fin = res[pick]
    b = fin["_b"]
    last = fin["passes"][-1]
    arrays = {}
    for n in L.LAYERS:
        arrays[f"b_{n}"] = b[n].cpu().numpy()
        arrays[f"m0_{n}"] = m0[n].cpu().numpy()
        arrays[f"mfinal_{n}"] = fin["_m"][n].cpu().numpy()
    first = res[0]["passes"][0]
    report = {"task": task, "k": int(k), "n_rows": n_rows, "horizon": int(A.shape[1]), "n_files": len(files),
              "rule_id": "D2", "stages_rule": stages,
              "converged": bool(conv), "bc_unconverged": not conv, "final_stage": pick, "final_eta": fin["eta"],
              "final_n_updates": fin["n_updates"],
              "residual_final": last["r_mean"], "residual_final_max": last["r_max"],
              "residual_ok_by_layer": {n: bool(last["r_mean"][n] < BC_TOL - TOL) for n in L.LAYERS},
              "residual_ok": all(last["r_mean"][n] < BC_TOL - TOL for n in L.LAYERS),
              "b_mean_abs_over_delta": {n: float(b[n].abs().mean()) / dl[n] for n in L.LAYERS},
              "mse_over_delta2_b0": first["mse_over_delta2"], "mse_over_delta2_final": last["mse_over_delta2"],
              "latent_err_b0": first["latent_err"], "latent_err_final": last["latent_err"],
              "stages": [{k_: v for k_, v in x.items() if not k_.startswith("_")} for x in res],
              "deltas": dl, "ranges": {n: [float(quant.qrange[n][0]), float(quant.qrange[n][1])] for n in L.LAYERS},
              "criterion": f"every layer mean_j |m_j| / delta_j < {BC_TOL} at the final b (DESIGN_tmlr D2)"}
    return b, report, arrays
