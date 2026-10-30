"""Stages: stats (m_j, mu_j, delta_j, abar_j on calibration windows), match (random controls calibrated to the
normalized prediction change of A / B on calibration windows), eval (window-major: every condition on every window).

Normalized prediction change of a condition vs the unablated model (same windows, same candidates):
  delta_k = sum |zh'_k - zh_k| / sum |zh_k|  (sums over windows, candidates, tokens, dims), delta = (delta_1 + delta_2) / 2
Usage: python run.py <prep_dir> <out_dir> [--prec tf32] [--stages stats,match,eval] [--n_eval N] [--nmatch 6]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r25lib as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("prep")
ap.add_argument("out")
ap.add_argument("--prec", default="tf32")
ap.add_argument("--stages", default="stats,match,eval")
ap.add_argument("--n_eval", type=int, default=10 ** 9)
ap.add_argument("--nmatch", type=int, default=6)
ap.add_argument("--fix_seeds", default="0,1,2")
ap.add_argument("--indep_seed", type=int, default=7)
ap.add_argument("--num_floor", default="bf16:1000000,fp32:1000000")  # precision:max_windows for the numerical-floor conditions
ap.add_argument("--save_every", type=int, default=5)
ap.add_argument("--rows", type=int, default=1024)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
stages = args.stages.split(",")
FIX = [int(s) for s in args.fix_seeds.split(",") if s != ""]
NUMF = {s.split(":")[0]: int(s.split(":")[1]) for s in args.num_floor.split(",") if s != ""}

P = torch.load(f"{args.prep}/prep.pt", weights_only=False)
meta = json.load(open(f"{args.prep}/prep.json"))
wins = meta["windows"]
cal_idx = [w["wi"] for w in wins if w["split"] == "cal"]
ev_idx = [w["wi"] for w in wins if w["split"] == "eval"][:args.n_eval]
lat, cands, states = P["lat"], P["cands"].numpy(), P["states"].numpy()

_, pred, info = L.load_models(need_encoder=False)
L.log("loaded", info)
iv = L.Intervention(pred)
NB, H = iv.NB, iv.H
LMAX = 2 * (L.NTOK + L.COND)
torch.backends.cuda.matmul.allow_tf32 = False


def roll(wi, prec=None, seed=0):
    z = lat[wi].cuda()
    zh1, zh2 = L.rollout(pred, z[0], states[wi], cands[wi], prec or args.prec, iv=iv, seed=seed)
    return z, zh1, zh2


def indep_seed(wi):
    return args.indep_seed * 100003 + wi


# ------------------------------------------------------------------------------------------------------------ stats
if "stats" in stages:
    t0 = time.time()
    p1 = L.Pass1(NB, H, rows=args.rows, seed=0)
    iv.set(None)
    iv.collector = p1
    for wi in cal_idx:
        roll(wi)
    m = p1.quantile()
    abar = (p1.sum / p1.n[:, None]).float()
    n_rows = int(sum(x.shape[0] for x in p1.samples[0]))
    del p1
    thr = torch.stack([a * m for a in L.ALPHAS])  # [nA, NB, H]
    p2 = L.Pass2(thr)
    iv.collector = p2
    for wi in cal_idx:
        roll(wi)
    iv.collector = None
    mu = torch.where(p2.nw > 0, p2.sw / p2.nw.clamp(min=1), torch.zeros_like(p2.sw)).float()
    delta = (-p2.sw / p2.n[None, :, None]).float()
    wf = (p2.nw.sum(-1) / (p2.n[None] * H))                 # [nA, NB] weak fraction per block
    wf_cond = p2.nw_cond / (p2.n_cond[None] * H)
    wf_patch = (p2.nw.sum(-1) - p2.nw_cond) / ((p2.n - p2.n_cond)[None] * H)
    torch.save(dict(m=m.cpu(), abar=abar.cpu(), thr=thr.cpu(), mu=mu.cpu(), delta=delta.cpu()), f"{args.out}/stats.pt")
    qs = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], device=m.device)
    summ = dict(n_cal=len(cal_idx), rows_per_block_for_quantile=n_rows, tokens_per_block=float(p2.n[0]),
                m_quantiles_over_units=torch.quantile(m.flatten().float(), qs).tolist(),
                m_dead_units=int((m == 0).sum()),
                weak_frac_all={str(a): float(wf[i].mean()) for i, a in enumerate(L.ALPHAS)},
                weak_frac_cond_tokens={str(a): float(wf_cond[i].mean()) for i, a in enumerate(L.ALPHAS)},
                weak_frac_patch_tokens={str(a): float(wf_patch[i].mean()) for i, a in enumerate(L.ALPHAS)},
                weak_frac_per_block={str(a): wf[i].tolist() for i, a in enumerate(L.ALPHAS)},
                mean_abs_mu={str(a): float(mu[i].abs().mean()) for i, a in enumerate(L.ALPHAS)},
                mean_mu={str(a): float(mu[i].mean()) for i, a in enumerate(L.ALPHAS)},
                mean_delta={str(a): float(delta[i].mean()) for i, a in enumerate(L.ALPHAS)},
                mean_abs_delta={str(a): float(delta[i].abs().mean()) for i, a in enumerate(L.ALPHAS)},
                mean_abar=float(abar.mean()), seconds=round(time.time() - t0, 1), prec=args.prec)
    json.dump(summ, open(f"{args.out}/stats.json", "w"), indent=1)
    L.log("STATS_DONE", {k: v for k, v in summ.items() if k != "weak_frac_per_block"})

S = {k: v.cuda() for k, v in torch.load(f"{args.out}/stats.pt").items()}
U = {s: L.shared_uniform(NB, LMAX, H, seed=1000 + s) for s in FIX}


def setc(mode, ai=None, seed=None, p=0.0):
    if mode is None:
        iv.set(None)
    elif mode == "A":
        iv.set("A", thr=S["thr"][ai])
    elif mode == "B":
        iv.set("B", thr=S["thr"][ai], fill=S["mu"][ai])
    elif mode == "C":
        iv.set("C", delta=S["delta"][ai])
    elif mode in ("F1", "F2"):
        iv.set(mode, u=U[seed], p=p, fill=S["abar"])
    elif mode in ("D1", "D2"):
        iv.set(mode, p=p, fill=S["abar"])
    else:
        raise ValueError(mode)


# ------------------------------------------------------------------------------------------------------------ match
if "match" in stages:
    t0 = time.time()
    mw = cal_idx[:args.nmatch]
    iv.set(None)
    base = {wi: roll(wi)[1:] for wi in mw}

    def pred_change(mode, ai=None, seed=None, p=0.0):
        num, den = [0.0, 0.0], [0.0, 0.0]
        for wi in mw:
            setc(mode, ai, seed, p)
            _, zh1, zh2 = roll(wi, seed=indep_seed(wi))
            b1, b2 = base[wi]
            num[0] += float((zh1 - b1).abs().sum())
            den[0] += float(b1.abs().sum())
            num[1] += float((zh2 - b2).abs().sum())
            den[1] += float(b2.abs().sum())
        iv.set(None)
        d = [num[0] / den[0], num[1] / den[1]]
        return 0.5 * (d[0] + d[1]), d

    def search(fn, target, p0=0.01, tol=0.03, maxit=14, lo_lim=1e-6, hi_lim=0.95):
        hist = []

        def ev(p):
            d = fn(p)
            hist.append((p, d))
            return d

        lo = hi = None
        p = p0
        d = ev(p)
        if d < target:
            lo = (p, d)
        else:
            hi = (p, d)
        while (lo is None or hi is None) and len(hist) < maxit:
            if lo is None:
                p = max(hi[0] / 4, lo_lim)
                d = ev(p)
                if d < target:
                    lo = (p, d)
                else:
                    hi = (p, d)
                    if p <= lo_lim:
                        break
            else:
                p = min(lo[0] * 4, hi_lim)
                d = ev(p)
                if d >= target:
                    hi = (p, d)
                else:
                    lo = (p, d)
                    if p >= hi_lim:
                        break
        while lo is not None and hi is not None and len(hist) < maxit:
            best = min(hist, key=lambda x: abs(x[1] / target - 1))
            if abs(best[1] / target - 1) < tol:
                break
            if lo[1] > 0 and hi[1] > lo[1]:
                frac = (math.log(target) - math.log(lo[1])) / (math.log(hi[1]) - math.log(lo[1]))
            else:
                frac = 0.5
            frac = min(max(frac, 0.1), 0.9)
            p = math.exp(math.log(lo[0]) + frac * (math.log(hi[0]) - math.log(lo[0])))
            d = ev(p)
            if d < target:
                lo = (p, d)
            else:
                hi = (p, d)
        best = min(hist, key=lambda x: abs(x[1] / target - 1))
        return best[0], best[1], hist

    res = dict(match_windows=mw, prec=args.prec, targets={}, p={}, achieved={}, hist={})
    for ai, a in enumerate(L.ALPHAS):
        for mode in ["A", "B", "C"]:
            d, dk = pred_change(mode, ai)
            res["targets"][f"{mode}@{a}"] = dict(delta=d, delta_k=dk)
        for mode, tgt in [("F2", "B"), ("F1", "A"), ("D2", "B"), ("D1", "A")]:
            target = res["targets"][f"{tgt}@{a}"]["delta"]
            fn = (lambda p, mode=mode, ai=ai: pred_change(mode, ai, FIX[0] if mode[0] == "F" else None, p)[0])
            p, d, hist = search(fn, target)
            res["p"][f"{mode}@{a}"] = p
            res["achieved"][f"{mode}@{a}"] = dict(delta=d, target=target, ratio=d / target, n_eval=len(hist))
            res["hist"][f"{mode}@{a}"] = hist
            L.log(f"match {mode}@{a}: target {target:.4f} -> p {p:.3e} delta {d:.4f} ({len(hist)} evals)")
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(f"{args.out}/match.json", "w"), indent=1)
    L.log("MATCH_DONE", res["seconds"], "s")
    del base

# ------------------------------------------------------------------------------------------------------------- eval
if "eval" in stages:
    M = json.load(open(f"{args.out}/match.json"))
    conds = [dict(name="base", mode=None)]
    for ai, a in enumerate(L.ALPHAS):
        for mode in ["A", "B", "C"]:
            conds.append(dict(name=f"{mode}@{a}", mode=mode, ai=ai))
        for mode in ["F2", "F1"]:
            for s in FIX:
                conds.append(dict(name=f"{mode}s{s}@{a}", mode=mode, ai=ai, seed=s, p=M["p"][f"{mode}@{a}"]))
        for mode in ["D2", "D1"]:
            conds.append(dict(name=f"{mode}@{a}", mode=mode, ai=ai, p=M["p"][f"{mode}@{a}"]))
    for pm, nmax in NUMF.items():
        conds.append(dict(name=f"NUM_{pm}", mode=None, prec=pm, nmax=nmax))
    nC, nW = len(conds), len(ev_idx)
    E = np.full((nW, nC, 2, L.NCAND), np.nan)
    DN = np.full((nW, nC, 2, L.NCAND), np.nan)
    MF = np.full((nW, nC), np.nan)
    DEN = np.zeros((nW, 2, L.NCAND))
    SPR = np.zeros((nW, 2, L.NCAND))
    t0 = time.time()
    tcond = np.zeros(nC)
    n_nonfinite = []
    for n, wi in enumerate(ev_idx):
        for ci, c in enumerate(conds):
            if n >= c.get("nmax", 10 ** 9):
                continue
            tc = time.time()
            setc(c["mode"], c.get("ai"), c.get("seed"), c.get("p", 0.0))
            iv.count = True
            iv.reset_count()
            z, zh1, zh2 = roll(wi, prec=c.get("prec"), seed=indep_seed(wi))
            iv.count = False
            mf = iv.modfrac()
            if not (torch.isfinite(zh1).all() and torch.isfinite(zh2).all()):  # fp16 overflow guard: redo in tf32
                n_nonfinite.append([n, c["name"]])
                iv.reset_count()
                iv.count = True
                z, zh1, zh2 = roll(wi, prec="tf32", seed=indep_seed(wi))
                iv.count = False
                mf = iv.modfrac()
            iv.set(None)
            if ci == 0:
                zb1, zb2 = zh1, zh2
                DEN[n, 0] = zb1.abs().sum((1, 2)).double().cpu().numpy()
                DEN[n, 1] = zb2.abs().sum((1, 2)).double().cpu().numpy()
                SPR[n, 0] = (zb1 - zb1.mean(0, keepdim=True)).abs().sum((1, 2)).double().cpu().numpy()
                SPR[n, 1] = (zb2 - zb2.mean(0, keepdim=True)).abs().sum((1, 2)).double().cpu().numpy()
            E[n, ci, 0] = L.energy(zh1, z[1]).double().cpu().numpy()
            E[n, ci, 1] = L.energy(zh2, z[2]).double().cpu().numpy()
            DN[n, ci, 0] = (zh1 - zb1).abs().sum((1, 2)).double().cpu().numpy()
            DN[n, ci, 1] = (zh2 - zb2).abs().sum((1, 2)).double().cpu().numpy()
            MF[n, ci] = mf
            torch.cuda.synchronize()
            tcond[ci] += time.time() - tc
        if (n + 1) % args.save_every == 0 or n + 1 == nW:
            np.savez(f"{args.out}/eval_raw.npz", E=E[:n + 1], DN=DN[:n + 1], MF=MF[:n + 1], DEN=DEN[:n + 1], SPR=SPR[:n + 1],
                     ev_idx=np.array(ev_idx[:n + 1]), cond_names=np.array([c["name"] for c in conds]))
            json.dump(dict(done=n + 1, total=nW, conds=conds, prec=args.prec, elapsed_s=round(time.time() - t0, 1),
                           s_per_window=round((time.time() - t0) / (n + 1), 2),
                           s_per_cond_window=dict(zip([c["name"] for c in conds], np.round(tcond / (n + 1), 3).tolist())),
                           peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), nonfinite_redone_tf32=n_nonfinite),
                      open(f"{args.out}/eval_progress.json", "w"), indent=1)
            L.log(f"eval {n + 1}/{nW} windows, {time.time() - t0:.0f}s, {(time.time() - t0) / (n + 1):.1f}s/window")
    L.log("EVAL_DONE")
