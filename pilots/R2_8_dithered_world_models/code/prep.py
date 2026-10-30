"""Phase B (per task) for R2-3b: reference scales, A's prediction errors and the deterministic matched controls.

Same calibration / held-out split and planning-time inputs as R2-3 (m_j = 99.9th pct of |a_j| on the calibration
split; held-out: 60k candidate sequences of episodes 5-9). A, B and the independent random zero controls D1z / D3z of
R2-3 are recomputed here (the pod GPU differs, so they are rerun for exact pairing). For each alpha and each
fixed-subset seed s in {0,1,2}:
  Dfix1_s: fixed random unit subset (fraction f per module family, nested in f), zeroed for every candidate;
           f_dyn: err1 = err1(A); f_rew: errr = errr(A)
  Dfix3_s: as Dfix1_s but f_dyn: err3 = err3(A); f_rew as Dfix1_s
  Ddir1_s: fixed random unit-norm direction added to every activation, magnitude lam per module family;
           lam_dyn: err1 = err1(A); lam_rew: errr = errr(A)
Usage: python prep.py --tasks mw-push --alphas 0.1,0.2 --out /workspace/out_main
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


def match_scale(fn, target, lo=1e-5, hi=1.0, iters=22, hi_cap=64.0):
    """Geometric bisection for r with fn(r) = target (fn increasing in r)."""
    e_hi = fn(hi)
    while e_hi < target and hi < hi_cap:
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


SUB_SEEDS = (0, 1, 2)


def run_task(task, out, alphas):
    t0 = time.time()
    agent, cfg = L.build_agent(task)
    model = agent.model
    ab = L.Ablator(model)
    eps = calib_eps(out, task)
    half = max(1, len(eps) // 2)
    cal_eps, held_eps = eps[:half], eps[half:] if len(eps) > 1 else eps
    Zu, idx, A = load_split(out, task, cal_eps)
    m, m_info, n_cal_rows = compute_m(model, ab, torch.as_tensor(Zu, device=L.DEV),
                                      torch.as_tensor(idx, device=L.DEV), torch.as_tensor(A, device=L.DEV))
    ab.m = m
    stats, abar = compute_stats(model, ab, torch.as_tensor(Zu, device=L.DEV), torch.as_tensor(idx, device=L.DEV),
                                torch.as_tensor(A, device=L.DEV), alphas)
    ab.stats, ab.abar = stats, abar
    Zu, idx, A = load_split(out, task, held_eps)
    sel = np.random.default_rng(0).choice(len(idx), size=min(N_HELD, len(idx)), replace=False)
    Z0 = torch.as_tensor(Zu, device=L.DEV)[torch.as_tensor(idx[sel], device=L.DEV)]
    ref = reference(model, cfg, Z0, torch.as_tensor(A[sel], device=L.DEV))
    res = {"task": task, "cal_eps": cal_eps, "held_eps": held_eps, "n_cal_rows": n_cal_rows, "n_held_seq": ref["N"],
           "m_info": m_info, "alphas": {}}
    match = {}
    for alpha in alphas:
        k = L.akey(alpha)
        r = {c: errors(model, cfg, ab, ref, L.setting_for(c, alpha), MATCH_SEED) for c in ("A", "B")}
        tA = r["A"]
        mk = match.setdefault(k, {})
        # R2-3 independent random zero controls (rerun here so that every condition is paired on the same GPU)
        base = {"mode": "rand", "alpha": alpha, "fill": "zero"}
        fz1 = lambda x: errors(model, cfg, ab, ref, {**base, "r_dyn": x, "r_rew": 0.0}, MATCH_SEED, what=("dyn",))["err1"]
        fz3 = lambda x: errors(model, cfg, ab, ref, {**base, "r_dyn": x, "r_rew": 0.0}, MATCH_SEED, what=("roll",))["err3"]
        fzr = lambda x: errors(model, cfg, ab, ref, {**base, "r_dyn": 0.0, "r_rew": x}, MATCH_SEED, what=("rew",))["errr"]
        z1, _, _ = match_scale(fz1, tA["err1"])
        z3, _, _ = match_scale(fz3, tA["err3"])
        zr, _, _ = match_scale(fzr, tA["errr"])
        rmatch = {"D1z": {"dyn": z1, "rew": zr}, "D3z": {"dyn": z3, "rew": zr}}
        for cond, prm in rmatch.items():
            r[cond] = errors(model, cfg, ab, ref, L.setting_for(cond, alpha, {"match": {k: rmatch}}), MATCH_SEED + 17)
            r[cond].update(r_dyn=prm["dyn"], r_rew=prm["rew"])
        mk.update(rmatch)
        for s_ in SUB_SEEDS:
            fix = {"mode": "fix", "alpha": alpha, "sub_seed": s_}
            fd1 = lambda x: errors(model, cfg, ab, ref, {**fix, "f_dyn": x, "f_rew": 0.0}, MATCH_SEED, what=("dyn",))["err1"]
            fd3 = lambda x: errors(model, cfg, ab, ref, {**fix, "f_dyn": x, "f_rew": 0.0}, MATCH_SEED, what=("roll",))["err3"]
            frw = lambda x: errors(model, cfg, ab, ref, {**fix, "f_dyn": 0.0, "f_rew": x}, MATCH_SEED, what=("rew",))["errr"]
            f1, e1, st1 = match_scale(fd1, tA["err1"], hi_cap=1.0)
            f3, e3, st3 = match_scale(fd3, tA["err3"], hi_cap=1.0)
            fr, er, str_ = match_scale(frw, tA["errr"], hi_cap=1.0)
            mk[f"Dfix1_s{s_}"] = {"dyn": f1, "rew": fr}
            mk[f"Dfix3_s{s_}"] = {"dyn": f3, "rew": fr}
            dr = {"mode": "dir", "alpha": alpha, "sub_seed": s_}
            gd = lambda x: errors(model, cfg, ab, ref, {**dr, "lam_dyn": x, "lam_rew": 0.0}, MATCH_SEED, what=("dyn",))["err1"]
            gr = lambda x: errors(model, cfg, ab, ref, {**dr, "lam_dyn": 0.0, "lam_rew": x}, MATCH_SEED, what=("rew",))["errr"]
            l1, el1, sl1 = match_scale(gd, tA["err1"], lo=1e-4, hi=1.0, hi_cap=4096.0)
            lr, elr, slr = match_scale(gr, tA["errr"], lo=1e-4, hi=1.0, hi_cap=4096.0)
            mk[f"Ddir1_s{s_}"] = {"dyn": l1, "rew": lr}
            for cond, st in ((f"Dfix1_s{s_}", (st1, str_)), (f"Dfix3_s{s_}", (st3, str_)), (f"Ddir1_s{s_}", (sl1, slr))):
                r[cond] = errors(model, cfg, ab, ref, L.setting_for(cond, alpha, {"match3b": match}), MATCH_SEED)
                r[cond].update(param_dyn=mk[cond]["dyn"], param_rew=mk[cond]["rew"], status=list(st))
        res["alphas"][k] = r
        fmt = lambda c: f"{c}: e1 {r[c]['err1']:.4f} e3 {r[c]['err3']:.4f} er {r[c]['errr']:.4f}"
        print(f"{task} alpha {k}: " + " | ".join(fmt(c) for c in r), flush=True)
        print("   params: " + ", ".join(f"{c} ({mk[c]['dyn']:.4g}, {mk[c]['rew']:.4g})" for c in mk), flush=True)
    torch.save({"m": {k: v.cpu() for k, v in m.items()}, "abar": {k: v.cpu() for k, v in abar.items()},
                "stats": {a: {s_: {k: v.cpu() for k, v in d.items()} for s_, d in st.items()} for a, st in stats.items()},
                "match": {a: {c: v for c, v in d.items() if c in ("D1z", "D3z")} for a, d in match.items()},
                "match3b": match}, f"{out}/data/scales_{task}.pt")
    res["match3b"] = match
    res["secs"] = time.time() - t0
    json.dump(res, open(f"{out}/results/prep_{task}.json", "w"), indent=1)
    print(f"PREP {task} done in {res['secs']:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--alphas", default="0.1,0.2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(2)
    for task in a.tasks.split(","):
        run_task(task, a.out, [float(x) for x in a.alphas.split(",")])
    print("PREP_DONE", flush=True)
