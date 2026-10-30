"""Phase B (per task) for R2-3: reference scales, A/B/C statistics, prediction metrics and matched random controls.

Calibration split (first half of the calibration episodes), planning-time inputs (z_h, a_h) rolled out from the
recorded root latents and MPPI candidate sequences with the unablated model (as in R1-1):
  m_j   = 99.9th percentile of |a_j|                     (weak set: |a_j| < alpha * m_j)
  mu_j  = E[a_j | weak]          (condition B fill)       delta_j = -E[a_j 1(weak)]   (condition C shift)
  abar_j = E[a_j]                (fill of the mean-corrected random controls)
Held-out split (second half), relative to the unablated model: err1 (1-step latent), err3 (3-step latent rollout),
errr (1-step reward), each normalized by the variance of the unablated output.
Matched random controls (geometric bisection on the per-row drop scale r, fixed mask seed => nested masks):
  D1z: zero fill,  r_dyn: err1 = err1(A), r_rew: errr = errr(A)      (= R1-1 rand_pred)
  D3z: zero fill,  r_dyn: err3 = err3(A), r_rew as D1z
  D1m: mean fill,  r_dyn: err1 = err1(B), r_rew: errr = errr(B)
  D3m: mean fill,  r_dyn: err3 = err3(B), r_rew as D1m
Usage: python prep.py --tasks mw-push --alphas 0.1,0.2,0.3 --out /workspace/out_main
Writes <out>/results/prep_<task>.json and <out>/data/scales_<task>.pt
"""
import argparse
import glob
import json
import os
import re
import time

import numpy as np
import torch

import nt_lib as L

N_HELD = 60000
CHUNK = 20000
MATCH_SEED = 4242


def calib_eps(out, task):
    files = glob.glob(f"{out}/data/calib_{task}_ep*.npz")
    return sorted(int(re.search(r"_ep(\d+)\.npz$", f).group(1)) for f in files)


def load_split(out, task, eps):
    zs, idx, acts, base = [], [], [], 0
    for ep in eps:
        d = np.load(f"{out}/data/calib_{task}_ep{ep}.npz")
        z, c = d["z_root"], d["cand"]  # (T, Lz), (T, I, K, H, A)
        T, I, K, H, A = c.shape
        zs.append(z)
        idx.append(base + np.repeat(np.arange(T), I * K))
        acts.append(c.reshape(T * I * K, H, A))
        base += T
    return np.concatenate(zs), np.concatenate(idx), np.concatenate(acts)


@torch.no_grad()
def compute_m(model, ab, Zu, idx, A, q=0.999):
    """Streaming 99.9th percentile of |a| per unit (k-th largest with k = ceil((1-q) n) + 1)."""
    H = A.shape[1]
    n_rows = len(idx) * H
    k = int(np.ceil((1 - q) * n_rows)) + 1
    top, stats = {}, {n: [0.0, 0.0, 0.0] for n in L.LAYERS}  # sum, count, count_negative
    ab.set("record")
    for s in range(0, len(idx), CHUNK):
        z = Zu[idx[s:s + CHUNK]]
        a = A[s:s + CHUNK]
        for h in range(H):
            ab.last.clear()
            model.reward(z, a[:, h], None)
            z_next = model.next(z, a[:, h], None)
            for name in L.LAYERS:
                v = ab.last[name]
                stats[name][0] += float(v.abs().sum())
                stats[name][1] += v.numel()
                stats[name][2] += float((v < 0).sum())
                cat = v.abs() if name not in top else torch.cat([top[name], v.abs()], 0)
                top[name] = torch.topk(cat, min(k, cat.shape[0]), dim=0).values
            z = z_next
    ab.set(None)
    m = {name: top[name][-1].clamp(min=1e-6).contiguous() for name in L.LAYERS}
    info = {name: {"m_median": float(m[name].median()), "m_min": float(m[name].min()), "m_max": float(m[name].max()),
                   "mean_abs": stats[name][0] / stats[name][1], "frac_negative": stats[name][2] / stats[name][1]}
            for name in L.LAYERS}
    return m, info, n_rows


@torch.no_grad()
def reference(model, cfg, Z0, A):
    """Unablated planning-time pairs: zin[h] (input latent), d[h] (next latent), r[h] (reward)."""
    H = A.shape[1]
    zin, d, r = [Z0], [], []
    for h in range(H):
        dd, rr = [], []
        for s in range(0, len(Z0), CHUNK):
            dd.append(model.next(zin[h][s:s + CHUNK], A[s:s + CHUNK, h], None))
            rr.append(L.two_hot_inv(model.reward(zin[h][s:s + CHUNK], A[s:s + CHUNK, h], None), cfg))
        d.append(torch.cat(dd))
        r.append(torch.cat(rr))
        if h + 1 < H:
            zin.append(d[-1])
    D = torch.cat(d)
    R = torch.cat(r)
    return {"zin": zin, "d": d, "r": r, "A": A, "N": len(Z0), "H": H,
            "var_d": float(((D - D.mean(0)) ** 2).sum(-1).mean()),
            "var_r": float(R.var()),
            "var_z3": float(((d[-1] - d[-1].mean(0)) ** 2).sum(-1).mean())}


@torch.no_grad()
def errors(model, cfg, ab, ref, setting, seed, what=("dyn", "rew", "roll")):
    """Normalized errors of the ablated model vs the unablated reference on the held-out planning-time set."""
    ab.set(**setting, seed=seed)
    H, N, A = ref["H"], ref["N"], ref["A"]
    s1 = sr = s3 = 0.0
    if "dyn" in what or "rew" in what:
        for h in range(H):
            for s in range(0, N, CHUNK):
                zin, a = ref["zin"][h][s:s + CHUNK], A[s:s + CHUNK, h]
                if "dyn" in what:
                    s1 += float(((model.next(zin, a, None) - ref["d"][h][s:s + CHUNK]) ** 2).sum())
                if "rew" in what:
                    sr += float(((L.two_hot_inv(model.reward(zin, a, None), cfg) - ref["r"][h][s:s + CHUNK]) ** 2).sum())
    frac = ab.fractions()
    if "roll" in what:
        ab.set(**setting, seed=seed + 1)
        for s in range(0, N, CHUNK):
            z = ref["zin"][0][s:s + CHUNK]
            for h in range(H):
                z = model.next(z, A[s:s + CHUNK, h], None)
            s3 += float(((z - ref["d"][-1][s:s + CHUNK]) ** 2).sum())
    ab.set(None)
    out = {"frac": frac}
    if "dyn" in what:
        out["err1"] = s1 / (H * N) / ref["var_d"]
    if "rew" in what:
        out["errr"] = sr / (H * N) / ref["var_r"]
    if "roll" in what:
        out["err3"] = s3 / N / ref["var_z3"]
    return out


@torch.no_grad()
def true_target_errors(model, cfg, ab, held, setting, seed):
    """Latent consistency error vs the encoder of the true next obs, and reward error vs the env reward."""
    ab.set(**setting, seed=seed)
    z = model.encode(held["obs"], None)
    zt = model.encode(held["obs_next"], None)
    pred = model.next(z, held["act"], None)
    rhat = L.two_hot_inv(model.reward(z, held["act"], None), cfg)
    ab.set(None)
    cons = float(((pred - zt) ** 2).sum(-1).mean() / ((zt - zt.mean(0)) ** 2).sum(-1).mean())
    rerr = float(((rhat - held["rew"]) ** 2).mean() / held["rew"].var().clamp(min=1e-8))
    return {"cons_err": cons, "rew_err": rerr}


def match_scale(fn, target, lo=1e-5, hi=1.0, iters=22):
    """Geometric bisection for r with fn(r) = target (fn increasing in r)."""
    e_hi = fn(hi)
    while e_hi < target and hi < 64:
        hi *= 2
        e_hi = fn(hi)
    if e_hi < target:
        return hi, e_hi, "hi_capped"
    e_lo = fn(lo)
    if e_lo > target:
        return lo, e_lo, "lo_capped"
    for _ in range(iters):
        mid = float(np.sqrt(lo * hi))
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
    r = float(np.sqrt(lo * hi))
    return r, fn(r), "ok"


@torch.no_grad()
def compute_stats(model, ab, Zu, idx, A, alphas):
    """Per-unit statistics on the calibration split: mu = E[a | weak], delta = -E[a 1(weak)], p = P(weak), abar = E[a]."""
    H = A.shape[1]
    acc = {name: {"sum": 0.0, "n": 0, **{a: {"sw": 0.0, "cw": 0.0} for a in alphas}} for name in L.LAYERS}
    ab.set("record")
    for s in range(0, len(idx), CHUNK):
        z = Zu[idx[s:s + CHUNK]]
        a_ = A[s:s + CHUNK]
        for h in range(H):
            ab.last.clear()
            model.reward(z, a_[:, h], None)
            z_next = model.next(z, a_[:, h], None)
            for name in L.LAYERS:
                v = ab.last[name].double()
                acc[name]["sum"] = acc[name]["sum"] + v.sum(0)
                acc[name]["n"] += v.shape[0]
                sc = v.abs() * ab.inv_m[name].double()
                for a in alphas:
                    w = (sc < a).double()
                    acc[name][a]["sw"] = acc[name][a]["sw"] + (v * w).sum(0)
                    acc[name][a]["cw"] = acc[name][a]["cw"] + w.sum(0)
            z = z_next
    ab.set(None)
    stats, abar = {}, {}
    for name in L.LAYERS:
        n = acc[name]["n"]
        abar[name] = (acc[name]["sum"] / n).float()
        for a in alphas:
            sw, cw = acc[name][a]["sw"], acc[name][a]["cw"]
            st = stats.setdefault(L.akey(a), {"mu": {}, "delta": {}, "p": {}})
            st["mu"][name] = torch.where(cw > 0, sw / cw.clamp(min=1), torch.zeros_like(sw)).float()
            st["delta"][name] = (-sw / n).float()
            st["p"][name] = (cw / n).float()
    return stats, abar


def summarize_stats(stats, abar):
    out = {}
    for a, st in stats.items():
        out[a] = {name: {"p_mean": float(st["p"][name].mean()), "mu_mean": float(st["mu"][name].mean()),
                         "mu_min": float(st["mu"][name].min()), "mu_max": float(st["mu"][name].max()),
                         "delta_mean": float(st["delta"][name].mean()), "delta_absmax": float(st["delta"][name].abs().max()),
                         "abar_mean": float(abar[name].mean())} for name in st["mu"]}
    return out


def run_task(task, out, alphas):
    t0 = time.time()
    agent, cfg = L.build_agent(task)
    model = agent.model
    ab = L.Ablator(model)
    eps = calib_eps(out, task)
    half = max(1, len(eps) // 2)
    cal_eps, held_eps = eps[:half], eps[half:] if len(eps) > 1 else eps
    # m_j and A/B/C statistics on the calibration split
    Zu, idx, A = load_split(out, task, cal_eps)
    Zu_t = torch.as_tensor(Zu, device=L.DEV)
    A_t = torch.as_tensor(A, device=L.DEV)
    idx_t = torch.as_tensor(idx, device=L.DEV)
    m, m_info, n_cal_rows = compute_m(model, ab, Zu_t, idx_t, A_t)
    ab.m = m
    stats, abar = compute_stats(model, ab, Zu_t, idx_t, A_t, alphas)
    ab.stats, ab.abar = stats, abar
    del Zu_t, A_t, idx_t
    # held-out planning-time set
    Zu, idx, A = load_split(out, task, held_eps)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(idx), size=min(N_HELD, len(idx)), replace=False)
    Z0 = torch.as_tensor(Zu, device=L.DEV)[torch.as_tensor(idx[sel], device=L.DEV)]
    ref = reference(model, cfg, Z0, torch.as_tensor(A[sel], device=L.DEV))
    o, o2, ac, rw = [], [], [], []
    for ep in held_eps:
        d = np.load(f"{out}/data/calib_{task}_ep{ep}.npz")
        o.append(d["obs_seq"][:-1]); o2.append(d["obs_seq"][1:]); ac.append(d["act_seq"]); rw.append(d["rew_seq"])
    held = {k: torch.as_tensor(np.concatenate(v), device=L.DEV) for k, v in
            (("obs", o), ("obs_next", o2), ("act", ac), ("rew", rw))}
    res = {"task": task, "cal_eps": cal_eps, "held_eps": held_eps, "n_cal_rows": n_cal_rows, "n_held_seq": ref["N"],
           "horizon": ref["H"], "var_d": ref["var_d"], "var_r": ref["var_r"], "var_z3": ref["var_z3"],
           "m_info": m_info, "stats_summary": summarize_stats(stats, abar),
           "true_target": {"none": true_target_errors(model, cfg, ab, held, {"mode": None}, 0)}, "alphas": {}}
    match = {}
    for alpha in alphas:
        k = L.akey(alpha)
        r = {c: errors(model, cfg, ab, ref, L.setting_for(c, alpha), MATCH_SEED) for c in ("A", "B", "C")}
        mk = {}
        for cond, fill, tgt in (("D1z", "zero", "A"), ("D3z", "zero", "A"), ("D1m", "mean", "B"), ("D3m", "mean", "B")):
            base = {"mode": "rand", "alpha": alpha, "fill": fill}
            if cond.startswith("D1"):
                fd = lambda x: errors(model, cfg, ab, ref, {**base, "r_dyn": x, "r_rew": 0.0}, MATCH_SEED, what=("dyn",))["err1"]
                r_dyn, e_dyn, st_dyn = match_scale(fd, r[tgt]["err1"])
                fr = lambda x: errors(model, cfg, ab, ref, {**base, "r_dyn": 0.0, "r_rew": x}, MATCH_SEED, what=("rew",))["errr"]
                r_rew, e_rew, st_rew = match_scale(fr, r[tgt]["errr"])
            else:
                f3 = lambda x: errors(model, cfg, ab, ref, {**base, "r_dyn": x, "r_rew": 0.0}, MATCH_SEED, what=("roll",))["err3"]
                r_dyn, e_dyn, st_dyn = match_scale(f3, r[tgt]["err3"])
                r_rew, e_rew, st_rew = mk["D1" + cond[2]]["rew"], None, "as_D1"
            mk[cond] = {"dyn": r_dyn, "rew": r_rew}
            match.setdefault(k, {})[cond] = mk[cond]
            # fresh masks (different seed) for the reported errors: checks that the match generalizes
            r[cond] = errors(model, cfg, ab, ref, L.setting_for(cond, alpha, {"match": match}), MATCH_SEED + 17)
            r[cond].update(r_dyn=r_dyn, r_rew=r_rew, search_err=e_dyn, search_errr=e_rew, status_dyn=st_dyn,
                           status_rew=st_rew, matched_to=tgt)
        for cond in L.CONDS:
            res["true_target"][f"{cond}_{k}"] = true_target_errors(model, cfg, ab, held,
                                                                   L.setting_for(cond, alpha, {"match": match}), MATCH_SEED)
        res["alphas"][k] = r
        fmt = lambda c: (f"{c}: frac {r[c]['frac']['all'] if r[c]['frac']['all'] is None else round(r[c]['frac']['all'], 3)} "
                         f"e1 {r[c]['err1']:.4f} e3 {r[c]['err3']:.4f} er {r[c]['errr']:.4f}")
        print(f"{task} alpha {k}: " + " | ".join(fmt(c) for c in L.CONDS), flush=True)
    torch.save({"m": {k: v.cpu() for k, v in m.items()}, "abar": {k: v.cpu() for k, v in abar.items()},
                "stats": {a: {s_: {k: v.cpu() for k, v in d.items()} for s_, d in st.items()} for a, st in stats.items()},
                "match": match}, f"{out}/data/scales_{task}.pt")
    res["match"] = match
    res["secs"] = time.time() - t0
    json.dump(res, open(f"{out}/results/prep_{task}.json", "w"), indent=1)
    print(f"PREP {task} done in {res['secs']:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--alphas", default="0.1,0.2,0.3")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(2)
    for task in a.tasks.split(","):
        run_task(task, a.out, [float(x) for x in a.alphas.split(",")])
    print("PREP_DONE", flush=True)
