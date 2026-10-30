"""Phase B (R2-8b, per task): quantizer ranges and held-out prediction errors on this pod.

Quantizer: R2-8's per-layer ranges (inputs/r28/prep_<task>.json 'quant_range': lo = minimum activation, hi = 99.99th
percentile of the layer's planning-time activations), so the k-bit grid is identical to R2-8. The ranges recomputed
from this pod's calibration split (episodes 0-4 of the calibration rollouts, same procedure as R2-8) are stored for
comparison only.
Held-out split (60k planning-time candidate sequences of calibration episodes 5-9): normalized err1 (1-step latent),
err3 (3-step open-loop latent rollout) and errr (1-step reward) of every variant at every k in --kbits, averaged over
--nseeds dither seeds. The horizon index of each forward call is set explicitly (ab.hforce), so shstep / shep use
table row h for horizon step h as in planning (one table per error evaluation: within one forward the three shared
variants are identical in distribution).
Usage: python prep_r28b.py --task mw-push --kbits 3,4 --out /workspace/out_main --inputs /workspace/inputs
Writes <out>/results/prep_<task>.json and <out>/data/scales_<task>.pt ({'quant': ranges})
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import nt_lib as L
from prep import CHUNK, MATCH_SEED, N_HELD, calib_eps, load_split, reference


@torch.no_grad()
def quant_ranges(model, ab, Zu, idx, A, q_hi=0.9999, max_rows=12000):
    """Per-layer (lo, hi) from a row subsample of the planning-time activations (copied from R2-8 prep_r28.py)."""
    rng = np.random.default_rng(1)
    sel = torch.as_tensor(np.sort(rng.choice(len(idx), size=min(max_rows, len(idx)), replace=False)), device=L.DEV)
    acts = {n: [] for n in L.LAYERS}
    ab.set("record")
    z = Zu[idx[sel]]
    a = A[sel]
    for h in range(A.shape[1]):
        ab.last.clear()
        model.reward(z, a[:, h], None)
        z_next = model.next(z, a[:, h], None)
        for n in L.LAYERS:
            acts[n].append(ab.last[n].flatten())
        z = z_next
    ab.set(None)
    out = {}
    for n in L.LAYERS:
        v = torch.cat(acts[n])
        k = max(1, int(round((1 - q_hi) * v.numel())))
        hi = float(torch.topk(v, k).values[-1])
        out[n] = (float(v.min()), hi)
    return out


@torch.no_grad()
def errors_h(model, cfg, ab, ref, setting, seed):
    """Normalized 1-step latent / reward errors and 3-step rollout error vs the unquantized reference."""
    H, N, A = ref["H"], ref["N"], ref["A"]
    ab.set(**setting, seed=seed)
    s1 = sr = 0.0
    for h in range(H):
        ab.hforce = h
        for s in range(0, N, CHUNK):
            zin, a = ref["zin"][h][s:s + CHUNK], A[s:s + CHUNK, h]
            s1 += float(((model.next(zin, a, None) - ref["d"][h][s:s + CHUNK]) ** 2).sum())
            sr += float(((L.two_hot_inv(model.reward(zin, a, None), cfg) - ref["r"][h][s:s + CHUNK]) ** 2).sum())
    ab.set(**setting, seed=seed + 1)
    s3 = 0.0
    for s in range(0, N, CHUNK):
        z = ref["zin"][0][s:s + CHUNK]
        for h in range(H):
            ab.hforce = h
            z = model.next(z, A[s:s + CHUNK, h], None)
        s3 += float(((z - ref["d"][-1][s:s + CHUNK]) ** 2).sum())
    ab.set(None)
    return {"err1": s1 / (H * N) / ref["var_d"], "errr": sr / (H * N) / ref["var_r"], "err3": s3 / N / ref["var_z3"]}


def run(task, out, inputs, kbits, nseeds):
    t0 = time.time()
    agent, cfg = L.build_agent(task)
    model = agent.model
    ab = L.Ablator(model, horizon=int(cfg.horizon))
    eps = calib_eps(out, task)
    half = max(1, len(eps) // 2)
    cal_eps, held_eps = eps[:half], (eps[half:] if len(eps) > 1 else eps)
    Zu, idx, A = load_split(out, task, cal_eps)
    Zu_t, idx_t, A_t = (torch.as_tensor(x, device=L.DEV) for x in (Zu, idx, A))
    qr_pod = quant_ranges(model, ab, Zu_t, idx_t, A_t)
    del Zu_t, idx_t, A_t
    qr = {k: (float(v[0]), float(v[1])) for k, v in json.load(open(f"{inputs}/r28/prep_{task}.json"))["quant_range"].items()}
    ab.qrange = qr
    Zu, idx, A = load_split(out, task, held_eps)
    sel = np.random.default_rng(0).choice(len(idx), size=min(N_HELD, len(idx)), replace=False)
    Z0 = torch.as_tensor(Zu, device=L.DEV)[torch.as_tensor(idx[sel], device=L.DEV)]
    ref = reference(model, cfg, Z0, torch.as_tensor(A[sel], device=L.DEV))
    res = {"task": task, "cal_eps": cal_eps, "held_eps": held_eps, "n_held_seq": ref["N"], "quant_range": qr,
           "quant_range_this_pod": qr_pod, "nseeds": nseeds, "pred": {}}
    for k in kbits:
        for v in L.QVARS:
            cond = f"Q{v}{k}"
            runs = [errors_h(model, cfg, ab, ref, L.setting_for(cond, float(k)), MATCH_SEED + 101 * j)
                    for j in range(nseeds)]
            e = {m: float(np.mean([r[m] for r in runs])) for m in ("err1", "err3", "errr")}
            e["sd"] = {m: float(np.std([r[m] for r in runs])) for m in ("err1", "err3", "errr")}
            res["pred"][f"{cond}@{k}"] = e
        print(f"{task} k={k}: " + " | ".join(
            f"{v}: e1 {res['pred'][f'Q{v}{k}@{k}']['err1']:.4f} e3 {res['pred'][f'Q{v}{k}@{k}']['err3']:.4f} "
            f"er {res['pred'][f'Q{v}{k}@{k}']['errr']:.4f}" for v in L.QVARS), flush=True)
    sp = f"{out}/data/scales_{task}.pt"
    if not os.path.exists(sp):  # never rewrite while grid workers may be loading it (content is R2-8's ranges anyway)
        torch.save({"quant": qr}, sp)
    res["secs"] = time.time() - t0
    json.dump(res, open(f"{out}/results/prep_{task}.json", "w"), indent=1, default=float)
    print(f"{task} ranges R2-8 vs this pod: " + ", ".join(
        f"{n} hi {qr[n][1]:.3f}/{qr_pod[n][1]:.3f}" for n in L.LAYERS), flush=True)
    print(f"PREP {task} done in {res['secs']:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--kbits", default="3,4")
    ap.add_argument("--nseeds", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", default="/workspace/inputs")
    a = ap.parse_args()
    torch.set_num_threads(1)
    run(a.task, a.out, a.inputs, [int(x) for x in a.kbits.split(",")], a.nseeds)
