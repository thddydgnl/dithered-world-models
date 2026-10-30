"""Phase B (R2-8, per task): every parameter needed on this pod.

From the calibration split (episodes 0-4 of the unablated calibration rollouts, planning-time inputs as in R1-1/R2-3):
  - m_j, B statistics (R2-3) and the matched random controls D1z/D3z/D1m/D3m (R2-3) for alpha in {0.1, 0.2, 0.3}
  - D-fix1/D-fix3/D-dir1 (R2-3b) for alpha in {0.1, 0.2} and fixed-subset seeds {0, 1, 2}
    (these old conditions are only re-parameterized here for the S1 metric analysis; their planning results come from
    the R2-3 / R2-3b pods)
  - R2-8 quantization ranges per hidden layer: lo = minimum activation, hi = 99.99th percentile of all activations of
    the layer (pooled over units), on the unablated planning-time activations
Held-out split: normalized err1 / err3 / errr of every condition, including k-bit quantization with deterministic
rounding (det), per-candidate stochastic rounding (ind), shared stochastic rounding (sh) and subtractive dither (sub)
for k in --kbits.
Usage: python prep_r28.py --task mw-push --kbits 2,3,4,5,6,7,8 --out /workspace/out_main
"""
import argparse
import json
import time

import numpy as np
import torch

import nt_lib as L
from prep import (CHUNK, MATCH_SEED, N_HELD, calib_eps, compute_m, compute_stats, errors, load_split, match_scale,
                  reference)

R23_ALPHAS = (0.1, 0.2, 0.3)
R3B_ALPHAS = (0.1, 0.2)
SUB_SEEDS = (0, 1, 2)
QVARS = ("det", "ind", "sh", "sub")


@torch.no_grad()
def quant_ranges(model, ab, Zu, idx, A, q_hi=0.9999, max_rows=12000):
    """Per-layer (lo, hi) from a row subsample of the planning-time activations (all horizon steps)."""
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


def r23_controls(model, cfg, ab, ref, alpha, res_r):
    """R2-3 matched random controls (zero fill matched to A, mean fill matched to B)."""
    tA, tB = res_r["A"], res_r["B"]
    out = {}
    for cond, fill, tgt in (("D1z", "zero", tA), ("D3z", "zero", tA), ("D1m", "mean", tB), ("D3m", "mean", tB)):
        base = {"mode": "rand", "alpha": alpha, "fill": fill}
        if cond.startswith("D1"):
            fd = lambda x: errors(model, cfg, ab, ref, {**base, "r_dyn": x, "r_rew": 0.0}, MATCH_SEED, what=("dyn",))["err1"]
            r_dyn, _, _ = match_scale(fd, tgt["err1"])
            fr = lambda x: errors(model, cfg, ab, ref, {**base, "r_dyn": 0.0, "r_rew": x}, MATCH_SEED, what=("rew",))["errr"]
            r_rew, _, _ = match_scale(fr, tgt["errr"])
        else:
            f3 = lambda x: errors(model, cfg, ab, ref, {**base, "r_dyn": x, "r_rew": 0.0}, MATCH_SEED, what=("roll",))["err3"]
            r_dyn, _, _ = match_scale(f3, tgt["err3"])
            r_rew = out["D1" + cond[2]]["rew"]
        out[cond] = {"dyn": r_dyn, "rew": r_rew}
    return out


def r3b_controls(model, cfg, ab, ref, alpha, tA):
    out = {}
    for s_ in SUB_SEEDS:
        fix = {"mode": "fix", "alpha": alpha, "sub_seed": s_}
        fd1 = lambda x: errors(model, cfg, ab, ref, {**fix, "f_dyn": x, "f_rew": 0.0}, MATCH_SEED, what=("dyn",))["err1"]
        fd3 = lambda x: errors(model, cfg, ab, ref, {**fix, "f_dyn": x, "f_rew": 0.0}, MATCH_SEED, what=("roll",))["err3"]
        frw = lambda x: errors(model, cfg, ab, ref, {**fix, "f_dyn": 0.0, "f_rew": x}, MATCH_SEED, what=("rew",))["errr"]
        f1, _, _ = match_scale(fd1, tA["err1"], hi_cap=1.0)
        f3, _, _ = match_scale(fd3, tA["err3"], hi_cap=1.0)
        fr, _, _ = match_scale(frw, tA["errr"], hi_cap=1.0)
        out[f"Dfix1_s{s_}"] = {"dyn": f1, "rew": fr}
        out[f"Dfix3_s{s_}"] = {"dyn": f3, "rew": fr}
        dr = {"mode": "dir", "alpha": alpha, "sub_seed": s_}
        gd = lambda x: errors(model, cfg, ab, ref, {**dr, "lam_dyn": x, "lam_rew": 0.0}, MATCH_SEED, what=("dyn",))["err1"]
        gr = lambda x: errors(model, cfg, ab, ref, {**dr, "lam_dyn": 0.0, "lam_rew": x}, MATCH_SEED, what=("rew",))["errr"]
        l1, _, _ = match_scale(gd, tA["err1"], lo=1e-4, hi=1.0, hi_cap=4096.0)
        lr, _, _ = match_scale(gr, tA["errr"], lo=1e-4, hi=1.0, hi_cap=4096.0)
        out[f"Ddir1_s{s_}"] = {"dyn": l1, "rew": lr}
    return out


def run(task, out, kbits, skip_old=False):
    t0 = time.time()
    agent, cfg = L.build_agent(task)
    model = agent.model
    ab = L.Ablator(model)
    eps = calib_eps(out, task)
    half = max(1, len(eps) // 2)
    cal_eps, held_eps = eps[:half], eps[half:] if len(eps) > 1 else eps
    Zu, idx, A = load_split(out, task, cal_eps)
    Zu_t, idx_t, A_t = (torch.as_tensor(x, device=L.DEV) for x in (Zu, idx, A))
    m, m_info, n_cal_rows = compute_m(model, ab, Zu_t, idx_t, A_t)
    ab.m = m
    stats, abar = compute_stats(model, ab, Zu_t, idx_t, A_t, list(R23_ALPHAS))
    ab.stats, ab.abar = stats, abar
    qr = quant_ranges(model, ab, Zu_t, idx_t, A_t)
    ab.qrange = qr
    del Zu_t, idx_t, A_t
    Zu, idx, A = load_split(out, task, held_eps)
    sel = np.random.default_rng(0).choice(len(idx), size=min(N_HELD, len(idx)), replace=False)
    Z0 = torch.as_tensor(Zu, device=L.DEV)[torch.as_tensor(idx[sel], device=L.DEV)]
    ref = reference(model, cfg, Z0, torch.as_tensor(A[sel], device=L.DEV))
    res = {"task": task, "n_cal_rows": n_cal_rows, "quant_range": qr, "pred": {}}
    match, match3b = {}, {}
    sc_tmp = {"match": match, "match3b": match3b}
    # R2-8 quantization conditions (alpha field = k)
    for k in kbits:
        for v in QVARS:
            cond = f"Q{v}{k}"
            res["pred"][f"{cond}@{k}"] = errors(model, cfg, ab, ref, L.setting_for(cond, float(k)), MATCH_SEED)
        print(f"{task} k={k}: " + " | ".join(
            f"{v}: e1 {res['pred'][f'Q{v}{k}@{k}']['err1']:.4f} e3 {res['pred'][f'Q{v}{k}@{k}']['err3']:.4f} "
            f"er {res['pred'][f'Q{v}{k}@{k}']['errr']:.4f}" for v in QVARS), flush=True)
    if not skip_old:
        for alpha in R23_ALPHAS:
            ka = L.akey(alpha)
            r = {c: errors(model, cfg, ab, ref, L.setting_for(c, alpha), MATCH_SEED) for c in ("A", "B", "C")}
            match[ka] = r23_controls(model, cfg, ab, ref, alpha, r)
            for c in ("D1z", "D3z", "D1m", "D3m"):
                r[c] = errors(model, cfg, ab, ref, L.setting_for(c, alpha, sc_tmp), MATCH_SEED + 17)
            if alpha in R3B_ALPHAS:
                match3b[ka] = r3b_controls(model, cfg, ab, ref, alpha, r["A"])
                for c in match3b[ka]:
                    r[c] = errors(model, cfg, ab, ref, L.setting_for(c, alpha, sc_tmp), MATCH_SEED)
            for c, v in r.items():
                res["pred"][f"{c}@{ka}"] = v
            print(f"{task} old conditions alpha {ka} done ({time.time() - t0:.0f}s)", flush=True)
    torch.save({"m": {k: v.cpu() for k, v in m.items()}, "abar": {k: v.cpu() for k, v in abar.items()},
                "stats": {a: {s_: {k: v.cpu() for k, v in d.items()} for s_, d in st.items()} for a, st in stats.items()},
                "match": match, "match3b": match3b, "quant": qr}, f"{out}/data/scales_{task}.pt")
    res["match"], res["match3b"], res["secs"] = match, match3b, time.time() - t0
    json.dump(res, open(f"{out}/results/prep_{task}.json", "w"), indent=1, default=float)
    print(f"PREP {task} done in {res['secs']:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--kbits", default="2,3,4,5,6,7,8")
    ap.add_argument("--skip_old", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    run(a.task, a.out, [int(x) for x in a.kbits.split(",")], a.skip_old)
