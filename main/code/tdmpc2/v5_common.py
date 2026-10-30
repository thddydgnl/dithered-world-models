"""Shared helpers of the v5 analyses (E2, E3, E5, E7) of the TD-MPC2 part; the statistics are those of v4 6.1 / v5 7.0
(analyze_e1.py): paired bootstrap 10,000 resamples with RandomState(0) over the sorted pairs (checkpoint, episode),
percentile 95% CI, one-sided p = (#{D* <= 0} + 1) / (B + 1), Meta-World drop = success-rate difference, DMC drop =
relative return drop recomputed as a ratio inside every resample."""
import glob
import gzip
import json
import os

import numpy as np

import analyze_e1 as A1

NB = A1.NB
OEI_FIELDS = ("oei", "oei_indep", "sel_last", "sel_indep", "drift", "oei_fresh", "sel_fresh", "e_chosen", "e_indep",
              "chosen_pi", "elite_pi", "elite_pi_w", "f_e_rand", "l_e_rand", "f_sd_v0", "f_var_cand", "l_var_cand",
              "drift_ref", "drift_minus_ref", "ref_f_e_rand", "ref_l_e_rand")


def ptag_of(job_id):
    return job_id.split("_")[-2]


def load_episodes(res, prefix=None):
    """-> E[(ptag, task, k, variant, ckpt, ep)] = {ret, success, oei: {..}, extra: {...}} from <res>/jobs/episodes."""
    E = {}
    pat = f"{res}/jobs/episodes/{prefix or ''}*.json.gz"
    for f in sorted(glob.glob(pat)):
        d = json.load(gzip.open(f, "rt"))
        s = d["spec"]
        pt = ptag_of(s["id"])
        extra = {kk: d.get(kk) for kk in ("ptq_resolved", "layer_k", "qat_path", "e7") if d.get(kk) is not None}
        for e in d["episodes"]:
            om = e.get("oei_mean") or {}
            E[(pt, s["task"], int(s["k"]), s["variant"], int(s["ckpt"]), int(e["ep"]))] = {
                "ret": float(e["ret"]), "success": float(e["success"]),
                "oei": {kk: om[kk] for kk in OEI_FIELDS if kk in om and om[kk] is not None}, "extra": extra}
    return E


def is_mw(task):
    return task.startswith("mw-")


def drop(m0, mv, mw):
    return (m0 - mv) if mw else (m0 - mv) / m0


def metric(rec, mw):
    return rec["success"] if mw else rec["ret"]


class Pairs:
    """The sorted pair set of one comparison and its resample indices (RandomState(0))."""

    def __init__(self, pairs):
        self.pairs = sorted(pairs)
        self.n = len(self.pairs)
        self.idx = A1.boot_idx(self.n) if self.n else None

    def arr(self, E, key_fn, field=None, mw=None):
        out = []
        for c, e in self.pairs:
            r = E[key_fn(c, e)]
            if field is None:
                out.append(metric(r, mw))
            else:
                out.append(r["oei"].get(field, np.nan))
        return np.array(out, float)

    def mean_ci(self, x):
        if not np.isfinite(x).all():
            return None
        return {"mean": float(x.mean()), "ci": A1.ci95(x[self.idx].mean(1))}

    def drop_ci(self, y0, yv, mw):
        b0, bv = y0[self.idx].mean(1), yv[self.idx].mean(1)
        D = drop(float(y0.mean()), float(yv.mean()), mw)
        Db = drop(b0, bv, mw)
        return {"D": float(D), "D_ci": A1.ci95(Db), "p_le0": A1.p_le0(Db)}, Db


def pairs_with(E, sel):
    """(ckpt, ep) pairs of the records matching sel(key) -> bool."""
    return {(k[4], k[5]) for k in E if sel(k)}


def spearman(x, y):
    return A1.spearman(x, y)


# Threshold comparisons of rank correlations (E2/E3 analysis fix, 2026-10-23, before any E2/E3 outcome was read): a
# Spearman on <= 12 points takes values on a coarse rational grid, but scipy returns e.g. -0.7999999999999999 for one
# adjacent swap of 4 points (exactly -0.8, which DESIGN_main v5 7.2 H4a explicitly allows) and 0.7999999999999999 for
# sum d^2 = 4 on 5 points (exactly 0.8, H2c ">= 0.8"). The rules are written with ">=" / "<=", so the comparisons use
# a tolerance far below the grid spacing (no value that is truly on the wrong side can pass).
EPS_RHO = 1e-9


def ge(x, thr):
    return bool(x is not None and np.isfinite(x) and x >= thr - EPS_RHO)


def le(x, thr):
    return bool(x is not None and np.isfinite(x) and x <= thr + EPS_RHO)


def _avg_ranks(X):
    """Average ranks along the last axis (ties share their mean rank, as in scipy.stats.spearmanr)."""
    from scipy.stats import rankdata
    return rankdata(X, axis=-1).astype(float)


def spearman_rows(X, y):
    """Spearman of every row of X (B, L) with the fixed vector y (L,). Ties get average ranks (scipy convention; the
    pre-E2 version broke ties by position, which biased resampled success-rate rows with ties); a constant row -> NaN."""
    return spearman_rows2(X, np.broadcast_to(np.asarray(y, float), np.shape(X)))


def spearman_rows2(X, Y):
    """Row-wise Spearman of two (B, L) matrices (average ranks; a row with a constant side -> NaN)."""
    rx, ry = _avg_ranks(np.asarray(X, float)), _avg_ranks(np.asarray(Y, float))
    rx -= rx.mean(1, keepdims=True)
    ry -= ry.mean(1, keepdims=True)
    den = np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, (rx * ry).sum(1) / np.where(den > 0, den, 1.0), np.nan)


def boot_rho_summary(rb, thr=None, side="ge"):
    """Percentile 95% interval of a bootstrap rho distribution (NaN rows dropped) and, with thr, the share of resamples
    on the rule's side of the threshold (same tolerance as the decision)."""
    rb = np.asarray(rb, float)
    ok = rb[np.isfinite(rb)]
    out = {"ci": [float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))] if len(ok) else [float("nan")] * 2,
           "median": float(np.median(ok)) if len(ok) else float("nan"), "n_defined": int(len(ok))}
    if thr is not None and len(ok):
        out[f"share_{side}_{thr}"] = float(np.mean(ok >= thr - EPS_RHO) if side == "ge" else np.mean(ok <= thr + EPS_RHO))
    return out


def write_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1, default=float)


def f2(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{nd}f}"


def ci_txt(ci, scale=1.0, nd=2):
    if not ci or ci[0] is None or not np.isfinite(ci[0]):
        return ""
    return f"[{ci[0] * scale:+.{nd}f}, {ci[1] * scale:+.{nd}f}]"
