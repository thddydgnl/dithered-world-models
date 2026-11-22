"""DIAGNOSTICS ONLY (not a main-run job, never used for b): convergence of the det+BC bias iteration of DESIGN_tmlr 0.6
under other update rules, on the recorded held rows of one task. Written after the design procedure failed check 0.7-2
on mw-door-open k=3 (residual after sweep 3: 22-126% of delta in the dynamics layers).
Rules (pass s measures m_s = per-unit mean of (corrected quantized output - unquantized output) with the current b over
all rows and the 3 steps, exactly as bclib.bias_sweeps; then b <- b + alpha m_s):
  design      alpha 1, the design's 3 sweeps continued to N sweeps (does it settle later?)
  damped<a>   alpha = a
  local       b <- b + mean of (corrected output - the layer's own pre-quantization output in the same quantized
              rollout): only the layer's own rounding bias, without the shift propagated from earlier layers / steps
              (the classic empirical bias correction); the design residual is still measured every pass
usage: python diag_bias.py --task <task> --k <k> --out <json> --rules design:10,damped0.5:12,local:6
"""
import argparse
import json
import time

import numpy as np
import torch

import bclib as B
import mt14_jobs as J
import mtlib as M
import qlib as L


class PreTap:
    """Forward hook placed BEFORE the Quantizer's hook (prepend=True): the layer's own pre-quantization output."""

    def __init__(self, quant):
        self.on, self.last = False, {}
        self.handles = [mod.register_forward_hook(self._hook(n), prepend=True) for n, mod in quant.mods.items()]

    def _hook(self, name):
        def hook(module, inputs, out):
            if self.on:
                self.last[name] = out
            return None
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def run_rule(agent, quant, files, task, k, rule, n, chunk=8000):
    model = agent.model
    tt = M.task_tensor(task)
    Zu, idx, A = M.load_rows(files)
    n_rows = int(len(idx))
    Zu, idx, A = (torch.as_tensor(x, device=L.DEV) for x in (Zu, idx, A))
    H = int(A.shape[1])
    W = {nm: int(quant.mods[nm].out_features) for nm in L.LAYERS}
    dl = B.deltas(quant.qrange, k)
    alpha = 1.0 if rule in ("design", "local") else float(rule.replace("damped", ""))
    b = {nm: torch.zeros(W[nm], device=L.DEV) for nm in L.LAYERS}
    post, pre = B.OutTap(quant), PreTap(quant)
    hist = []
    try:
        for s in range(n + 1):
            S1 = {nm: torch.zeros(W[nm], device=L.DEV, dtype=torch.float64) for nm in L.LAYERS}
            SL = {nm: torch.zeros(W[nm], device=L.DEV, dtype=torch.float64) for nm in L.LAYERS}
            S2 = {nm: 0.0 for nm in L.LAYERS}
            cnt = 0
            for c0 in range(0, n_rows, chunk):
                sl = slice(c0, min(c0 + chunk, n_rows))
                z0, a = Zu[idx[sl]], A[sl]
                quant.set("record")
                U, z = [], z0
                for h in range(H):
                    quant.last.clear()
                    model.reward(z, a[:, h], tt)
                    zn = model.next(z, a[:, h], tt)
                    U.append({nm: quant.last[nm] for nm in L.LAYERS})
                    z = zn
                quant.off()
                quant.bias = b
                quant.set("detbc", k)
                post.on = pre.on = True
                z = z0
                for h in range(H):
                    post.last.clear()
                    pre.last.clear()
                    model.reward(z, a[:, h], tt)
                    zn = model.next(z, a[:, h], tt)
                    for nm in L.LAYERS:
                        dlt = post.last[nm] - U[h][nm]
                        S1[nm] += dlt.sum(0, dtype=torch.float64)
                        S2[nm] += float((dlt.double() ** 2).sum())
                        SL[nm] += (post.last[nm] - pre.last[nm]).sum(0, dtype=torch.float64)
                    z = zn
                post.on = pre.on = False
                quant.off()
                cnt += (sl.stop - sl.start) * H
                del U
            m = {nm: (S1[nm] / cnt).float() for nm in L.LAYERS}
            ml = {nm: (SL[nm] / cnt).float() for nm in L.LAYERS}
            hist.append({"pass": s, "design_resid_mean_abs_over_delta": {nm: float(m[nm].abs().mean()) / dl[nm]
                                                                          for nm in L.LAYERS},
                         "local_resid_mean_abs_over_delta": {nm: float(ml[nm].abs().mean()) / dl[nm] for nm in L.LAYERS},
                         "mse_over_delta2": {nm: S2[nm] / (cnt * W[nm]) / dl[nm] ** 2 for nm in L.LAYERS}})
            if s < n:
                step = ml if rule == "local" else m
                for nm in L.LAYERS:
                    b[nm] = b[nm] + alpha * step[nm]
    finally:
        post.remove()
        pre.remove()
        quant.off()
        quant.bias = {}
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rules", default="design:10,damped0.5:12,damped0.25:16,local:6")
    a = ap.parse_args()
    torch.set_num_threads(1)
    agent, cfg = M.build_agent_mt()
    quant = L.Quantizer(agent.model, horizon=int(cfg.horizon))
    quant.qrange = J.load_ranges(a.task)
    files = [J.calib_path(a.task, s) for s in J.HELD_SEEDS]
    res = {"task": a.task, "k": a.k, "deltas": B.deltas(quant.qrange, a.k), "rules": {}}
    for item in a.rules.split(","):
        rule, n = item.split(":")
        t0 = time.time()
        res["rules"][item] = run_rule(agent, quant, files, a.task, a.k, rule, int(n))
        last = res["rules"][item][-1]
        print(f"DIAG {a.task} k{a.k} {item} ({time.time() - t0:.0f}s) design resid "
              + " ".join(f"{nm}:{last['design_resid_mean_abs_over_delta'][nm]:.4f}" for nm in L.LAYERS)
              + " | min over passes " + " ".join(
                  f"{nm}:{min(h['design_resid_mean_abs_over_delta'][nm] for h in res['rules'][item]):.4f}"
                  for nm in L.LAYERS), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    print("DIAG_DONE", flush=True)


if __name__ == "__main__":
    main()
