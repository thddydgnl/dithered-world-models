"""Diagnostic 2 for the DESIGN_tmlr 0.6 bias vectors (phase A, smoke calibration rows only): the damped form of the
same fixed-point iteration, b += eta * (per-unit mean of corrected quantized - unquantized output), all 4 layers from
one rollout per sweep (as written, but eta < 1), until the residual (worst layer, mean_j |m_j| / delta_j) is below 1%
of delta or 40 sweeps. Reports, per (task, range kind, k), the residual history and the first sweep meeting < 0.01.
Usage: python bias_diag2.py --out <json> --prep_out <smoke out> --big <smoke big> [--eta 0.5]
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
from bias_diag import rollout_means, resid  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--prep_out", required=True)
ap.add_argument("--big", required=True)
ap.add_argument("--tasks", default="mw-push,mw-pick-place,cheetah-run,walker-walk")
ap.add_argument("--eta", type=float, default=0.5)
ap.add_argument("--max_sweeps", type=int, default=40)
a = ap.parse_args()
torch.set_num_threads(1)
out = {}
with torch.no_grad():
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
            for k in range(1, 9):
                nl = (1 << k) - 1
                if rk == "PT":
                    qr = {n: (float(qr0[n][0]), float(qr0[n][1])) for n in L.LAYERS}
                    delta = {n: torch.full((quant.mods[n].out_features,), (qr[n][1] - qr[n][0]) / nl, device=L.DEV)
                             for n in L.LAYERS}
                else:
                    qr = qr0
                    delta = {n: (qr[n][1] - qr[n][0]) / nl for n in L.LAYERS}
                b = {n: torch.zeros(quant.mods[n].out_features, device=L.DEV) for n in L.LAYERS}
                hist, first = [], None
                for s in range(a.max_sweeps + 1):
                    m, _ = rollout_means(model, quant, z0, acts, qr, b, k)
                    r = resid(m, ref_mean, delta)
                    hist.append(round(r, 5))
                    if r < 0.01 and first is None:
                        first = s
                        break
                    b = {n: b[n] + a.eta * (m[n] - ref_mean[n]) for n in L.LAYERS}
                out[f"{task}|{rk}|k{k}"] = {"first_sweep_below_1pct": first, "hist": hist}
                print(task, rk, k, "first<1%:", first, "hist", hist[:4], "...", hist[-3:], flush=True)
        quant.remove()
json.dump(out, open(a.out, "w"), indent=1)
print("BIAS_DIAG2_DONE", "max_first", max((v["first_sweep_below_1pct"] or 999) for v in out.values()))
