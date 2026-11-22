"""Diagnostic for the DESIGN_tmlr 0.6 bias-vector procedure (phase A, smoke calibration data only; no episode outcome
is used). The procedure as written (b = 0; per sweep one b-corrected det rollout vs the unquantized rollout from the
same 12,000 start latents and action sequences, 3 steps; every layer's b += per-unit mean of (corrected quantized -
unquantized output); 3 sweeps) did not converge in the smoke prep (the residual grows from sweep to sweep). This script
measures, on the same rows, the residual after every sweep for:
  jacobi       the procedure as written (all 4 layers updated from one rollout per sweep), 6 sweeps
  gs           Gauss-Seidel order: within a sweep the layers are updated one at a time in forward order (dyn0, dyn1,
               rew0, rew1), each from a new rollout with the already updated b, 6 sweeps
  damped       jacobi with b += 0.5 * mean, 6 sweeps
  local        same-input rounding bias (no propagation): b = per-unit mean of det(a) - a on the quantized-path inputs of
               the layer itself, iterated 3 times (the inputs move with b) -- the classical empirical bias correction
Residual = mean over units of |per-unit mean of (corrected quantized - unquantized output)| / delta_j, worst layer.
Usage: python bias_diag.py --out <json> --prep_out <smoke out> --big <smoke big>
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

import qlib as L  # noqa: E402
import runner as R  # noqa: E402
import tdjobs as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--prep_out", required=True)
ap.add_argument("--big", required=True)
ap.add_argument("--tasks", default="mw-push,mw-pick-place,cheetah-run,walker-walk")
ap.add_argument("--ks", default="2,3,4,5")
a = ap.parse_args()
torch.set_num_threads(1)


@torch.no_grad()
def rollout_means(model, quant, z0, acts, qr, b, k, local=False):
    """Per-layer per-unit mean of the corrected quantized outputs over the b-corrected det rollout; local=True also
    the per-unit mean of the same-input rounding error det(a) - a (a = the layer's own unquantized output on the
    quantized path, from a direct forward call that bypasses the hooks)."""
    W = {n: quant.mods[n].out_features for n in L.LAYERS}
    sums = {n: torch.zeros(W[n], dtype=torch.float64, device=z0.device) for n in L.LAYERS}
    lsum = {n: torch.zeros(W[n], dtype=torch.float64, device=z0.device) for n in L.LAYERS}
    cnt = {n: 0 for n in L.LAYERS}

    def cap(n):
        def f(mod, inp, out):
            sums[n] += out.double().sum(0)
            cnt[n] += int(out.shape[0])
            if local:
                raw = mod.forward(inp[0])
                lsum[n] += (out.double() + b[n].double() - raw.double()).sum(0)
        return f

    hs = [quant.mods[n].register_forward_hook(cap(n)) for n in L.LAYERS]
    try:
        quant.qrange, quant.bias = qr, b
        quant.set("detbc", k)
        z = z0
        for h in range(acts.shape[1]):
            model.reward(z, acts[:, h], None)
            z = model.next(z, acts[:, h], None)
    finally:
        quant.off()
        for h_ in hs:
            h_.remove()
    return ({n: (sums[n] / cnt[n]).float() for n in L.LAYERS},
            {n: (lsum[n] / cnt[n]).float() for n in L.LAYERS} if local else None)


def resid(m, ref_mean, delta):
    return max(float(((m[n] - ref_mean[n]).abs() / delta[n]).mean()) for n in L.LAYERS)


out = {}
for task in a.tasks.split(","):
    agent, cfg = L.build_agent(task, 1)
    quant = L.Quantizer(agent.model, horizon=int(cfg.horizon))
    model = agent.model
    z0, acts = T.calib_rows(a.big, task, 1, [0, 1, 2, 3, 4], L.DEV)
    ref = T.record_acts(L, model, quant, z0, acts)
    ref_mean = {n: ref[n].double().mean(0).float() for n in L.LAYERS}
    del ref
    pt = R.load_ranges(a.prep_out, task, 1, "e1")
    pc = R.load_ranges(a.prep_out, task, 1, "pc")
    for rk, qr0 in (("PT", pt), ("PC", pc)):
        for k in [int(x) for x in a.ks.split(",")]:
            nl = (1 << k) - 1
            if rk == "PT":
                qr = {n: (float(qr0[n][0]), float(qr0[n][1])) for n in L.LAYERS}
                delta = {n: torch.full((quant.mods[n].out_features,), (qr[n][1] - qr[n][0]) / nl, device=L.DEV)
                         for n in L.LAYERS}
            else:
                qr = qr0
                delta = {n: (qr[n][1] - qr[n][0]) / nl for n in L.LAYERS}
            zero = {n: torch.zeros(quant.mods[n].out_features, device=L.DEV) for n in L.LAYERS}
            res = {}
            for meth in ("jacobi", "gs", "damped", "local"):
                b = dict(zero)
                hist = []
                for s in range(6 if meth != "local" else 3):
                    m, lm = rollout_means(model, quant, z0, acts, qr, b, k, local=(meth == "local"))
                    hist.append(resid(m, ref_mean, delta))
                    if meth == "jacobi":
                        b = {n: b[n] + (m[n] - ref_mean[n]) for n in L.LAYERS}
                    elif meth == "damped":
                        b = {n: b[n] + 0.5 * (m[n] - ref_mean[n]) for n in L.LAYERS}
                    elif meth == "local":
                        b = {n: lm[n] for n in L.LAYERS}     # the same-input rounding bias itself
                    else:   # Gauss-Seidel over the layers in forward order
                        for n in L.LAYERS:
                            m2, _ = rollout_means(model, quant, z0, acts, qr, b, k)
                            b = dict(b)
                            b[n] = b[n] + (m2[n] - ref_mean[n])
                m, _ = rollout_means(model, quant, z0, acts, qr, b, k)
                hist.append(resid(m, ref_mean, delta))
                res[meth] = [round(x, 4) for x in hist]
            out[f"{task}|{rk}|k{k}"] = res
            print(task, rk, k, json.dumps(res), flush=True)
    quant.remove()
json.dump(out, open(a.out, "w"), indent=1)
print("BIAS_DIAG_DONE")
