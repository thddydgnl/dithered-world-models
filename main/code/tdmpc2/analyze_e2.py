"""E2 analysis of the TD-MPC2 family: DESIGN_main v5 7.1 layer (a) with the v5.1 8.2 fixed-count N sweep, and the
v5.2 9.1 layer (b) (added after E1, E2 seeds only: b1 H2a_indep, b2 H2a' with its positive control, b3 region-effect
control; no H2c'). Written before any E2 outcome existed; checked on synthetic data only (test_v5_analyses.py).

Cells: setting s (16 OFAT settings) x task (mw-push k=3, mw-pick-place k=3, cheetah-run k=5) x variant; pairs =
(checkpoint, episode) of the E2 seeds, the same pair set in every setting of a task (so one resample matrix serves all
settings, and settings are paired with each other); D_v(s) against none(s) of the same setting.
  - H2a replication (base, frozen `oei`): det OEI 95% CI upper < 0 and sh-iter OEI CI contains 0 (per task; the 4th
    task, PushT, is the DINO-WM family's E1 value)
  - H2c (confirmatory reading (i), on the v5.1 fixed-count sweep N64k / N128k / N256k / base / N1024k: elite 64 and
    policy prior 24 fixed): per task rho_s(N, |OEI_det|) >= 0.8 and rho_s(N, D_det) >= 0.8; the family decision
    needs 3 of 4 tasks with PushT; the OEI decomposition (sel_last, drift) is reported along N; the v5 7.1
    ratio-fixed sweep gets the same statistics as an auxiliary (H2c_aux_ratio_sweep)
    auxiliary: rho_s(D_det, |OEI_det|) over N (reading (ii)), rho_s(N, |OEI_sh-iter|), rho_s(N, D_sh-iter), and the
    bootstrap distribution of every rho (pairs resampled jointly over the N levels)
  - descriptive: every setting's D and the OEI versions / components with CIs (oei, oei_indep, sel_last, sel_indep,
    drift, oei_fresh, sel_fresh, policy-slot shares); iteration curves with and without warm start
usage (server): OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python analyze_e2.py \
                    --res ~/pmx/A/tdmpc2_main/out_e2 --outdir ~/pmx/A/tdmpc2_main/analysis_e2

Changes of 2026-10-23 (E2/E3 analysis worker, after the runs ended but before any E2 outcome was read; rules unchanged,
see worklogs/E2E3_tdmpc2.md):
  - rho threshold decisions (H2c >= 0.8) use v5_common.ge (tolerance 1e-9: scipy returns 0.7999999999999999 for an
    exact 0.8 on 5 points)
  - bootstrap rho distributions use average ranks for ties (v5_common.spearman_rows; resampled success-rate drops tie),
    and are reported for every rho of H2c (also the auxiliary D-|OEI| and sh-iter flatness rhos)
  - per-level CIs of OEI_det and D_det in the H2c output; family_points_tdmpc2.json (per-task values the coordinator
    needs to combine the families)
"""
import argparse

import numpy as np

import v5_common as C

SETTINGS = ["base", "N64k", "N128k", "N256k", "N1024k", "N64", "N128", "N256", "N1024", "el2", "el25", "T01", "T2",
            "it1", "it3", "it12", "nows1", "nows3", "nows6", "nows12"]
# H2c confirmatory sweep (v5.1 8.2): elite count 64 and policy prior 24 fixed; auxiliary: the ratio-fixed sweep (v5 7.1)
N_LEVELS = [("N64k", 64), ("N128k", 128), ("N256k", 256), ("base", 512), ("N1024k", 1024)]
N_LEVELS_RATIO = [("N64", 64), ("N128", 128), ("N256", 256), ("base", 512), ("N1024", 1024)]
ITER_CURVES = {"warm": [("it1", 1), ("it3", 3), ("base", 6), ("it12", 12)],
               "no_warm": [("nows1", 1), ("nows3", 3), ("nows6", 6), ("nows12", 12)]}
TASKS = {"mw-push": 3, "mw-pick-place": 3, "cheetah-run": 5}
FIELDS = ("oei", "oei_indep", "sel_last", "sel_indep", "drift", "oei_fresh", "sel_fresh", "chosen_pi", "elite_pi",
          "elite_pi_w", "drift_ref", "drift_minus_ref", "ref_f_e_rand", "ref_l_e_rand")


def n_sweep(out, boot, t, levels):
    cells_n = [(out["cells"].get(f"{t}|{s}"), n) for s, n in levels]
    if not all(c is not None and "det" in c["variants"] and "oei" in c["variants"]["det"] for c, _ in cells_n):
        return None
    Ns = np.array([n for _, n in levels], float)
    det = [c["variants"]["det"] for c, _ in cells_n]
    absO = np.array([abs(x["oei"]) for x in det])
    Dd = np.array([x["D"] for x in det])
    r1, r2, r3 = C.spearman(Ns, absO), C.spearman(Ns, Dd), C.spearman(Dd, absO)
    h = {"levels": [s for s, _ in levels], "N": Ns.tolist(), "abs_oei_det": absO.tolist(), "D_det": Dd.tolist(),
         "oei_det": [x["oei"] for x in det], "oei_det_ci": [x.get("oei_ci") for x in det],
         "D_det_ci": [x.get("D_ci") for x in det], "oei_indep_det": [x.get("oei_indep") for x in det],
         "rho_N_absOEI_det": r1, "rho_N_D_det": r2, "holds": bool(C.ge(r1, 0.8) and C.ge(r2, 0.8)),
         "aux_rho_D_absOEI_det": r3,
         "decomposition": {v: [{"N": n, "sel_last": c["variants"].get(v, {}).get("sel_last"),
                                "sel_last_ci": c["variants"].get(v, {}).get("sel_last_ci"),
                                "drift": c["variants"].get(v, {}).get("drift"),
                                "drift_ci": c["variants"].get(v, {}).get("drift_ci"),
                                "oei": c["variants"].get(v, {}).get("oei"),
                                "oei_indep": c["variants"].get(v, {}).get("oei_indep"),
                                "sel_indep": c["variants"].get(v, {}).get("sel_indep"),
                                "D": c["variants"].get(v, {}).get("D"),
                                "D_ci": c["variants"].get(v, {}).get("D_ci")} for c, n in cells_n]
                           for v in ("det", "shiter", "shstep")}}
    # bootstrap distributions of the rhos: the pairs are resampled jointly over the N levels (one index matrix per task)
    Xo = np.stack([np.abs(boot[(s, "det", "oei")]) for s, _ in levels], 1)
    Xd = np.stack([boot[(s, "det", "D")] for s, _ in levels], 1)
    h["rho_N_absOEI_det_boot"] = C.boot_rho_summary(C.spearman_rows(Xo, Ns), 0.8, "ge")
    h["rho_N_D_det_boot"] = C.boot_rho_summary(C.spearman_rows(Xd, Ns), 0.8, "ge")
    h["aux_rho_D_absOEI_det_boot"] = C.boot_rho_summary(C.spearman_rows2(Xd, Xo))
    sh = [c["variants"].get("shiter", {}) for c, _ in cells_n]
    if all("oei" in x for x in sh) and all((s, "shiter", "oei") in boot for s, _ in levels):
        h["abs_oei_shiter"] = [abs(x["oei"]) for x in sh]
        h["D_shiter"] = [x["D"] for x in sh]
        h["aux_rho_N_absOEI_shiter"] = C.spearman(Ns, np.abs([x["oei"] for x in sh]))
        h["aux_rho_N_D_shiter"] = C.spearman(Ns, [x["D"] for x in sh])
        Xso = np.stack([np.abs(boot[(s, "shiter", "oei")]) for s, _ in levels], 1)
        Xsd = np.stack([boot[(s, "shiter", "D")] for s, _ in levels], 1)
        h["aux_rho_N_absOEI_shiter_boot"] = C.boot_rho_summary(C.spearman_rows(Xso, Ns))
        h["aux_rho_N_D_shiter_boot"] = C.boot_rho_summary(C.spearman_rows(Xsd, Ns))
    return h


def ratio_ci(num, den, idx):
    """r = mean(num) / mean(den) with the ratio recomputed per resample; resamples with mean(den)* >= 0 dropped."""
    nb, db = num[idx].mean(1), den[idx].mean(1)
    keep = db < 0
    rb = nb[keep] / db[keep]
    r = float(num.mean() / den.mean()) if den.mean() != 0 else float("nan")
    ci = [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))] if keep.any() else [float("nan")] * 2
    return {"r": r, "ci": ci, "resamples_kept": int(keep.sum())}


def layer_b(out, arrs, P, t, k):
    """DESIGN_main v5.2 9.1 E2 layer (b), added after E1, E2 seeds only (base setting):
    (b1) H2a_indep: det oei_indep 95% CI upper < 0 and sh-iter oei_indep CI contains 0
    (b2) H2a': drift_det CI upper < 0 and r = drift_sh-iter / drift_det CI upper < 0.5 (ratio per resample, resamples
         with drift_det* >= 0 dropped); positive control outside the rule: sh-step and sh-ep r CI lower > 0.5;
         descriptive: r of ind and det+shift, same-draw vs independent-draw sel_last of every variant
    (b3) region-effect control (descriptive): drift_ref and drift - drift_ref of every variant"""
    c = out["cells"].get(f"{t}|base")
    if not c:
        return None
    V = c["variants"]
    res = {"label": "layer (b): pre-registered after E1 (v5.2), E2 seeds only"}
    det, sh = V.get("det", {}), V.get("shiter", {})
    if "oei_indep_ci" in det and "oei_indep_ci" in sh:
        a_ = det["oei_indep_ci"][1] < 0
        b_ = sh["oei_indep_ci"][0] <= 0 <= sh["oei_indep_ci"][1]
        res["b1_H2a_indep"] = {"det_oei_indep": det["oei_indep"], "det_ci": det["oei_indep_ci"],
                               "shiter_oei_indep": sh["oei_indep"], "shiter_ci": sh["oei_indep_ci"],
                               "det_ci_upper_lt_0": bool(a_), "shiter_ci_contains_0": bool(b_), "holds": bool(a_ and b_)}
    if ("base", "det", "drift") in arrs:
        dd = arrs[("base", "det", "drift")]
        b2 = {"drift_det": float(dd.mean()), "drift_det_ci": C.A1.ci95(dd[P.idx].mean(1))}
        b2["cond1_drift_det_ci_upper_lt_0"] = bool(b2["drift_det_ci"][1] < 0)
        for v in ("shiter", "shstep", "shep", "ind", "detshift"):
            if ("base", v, "drift") in arrs:
                b2[f"r_{v}"] = ratio_ci(arrs[("base", v, "drift")], dd, P.idx)
        if "r_shiter" in b2:
            b2["cond2_r_shiter_ci_upper_lt_0.5"] = bool(b2["r_shiter"]["ci"][1] < 0.5)
            b2["holds"] = bool(b2["cond1_drift_det_ci_upper_lt_0"] and b2["cond2_r_shiter_ci_upper_lt_0.5"])
        b2["positive_control_outside_rule"] = {v: bool(b2[f"r_{v}"]["ci"][0] > 0.5) for v in ("shstep", "shep")
                                               if f"r_{v}" in b2}
        res["b2_H2a_prime"] = b2
    res["sel_last_same_vs_indep"] = {v: {"sel_last": r.get("sel_last"), "sel_last_ci": r.get("sel_last_ci"),
                                         "sel_indep": r.get("sel_indep"), "sel_indep_ci": r.get("sel_indep_ci")}
                                     for v, r in V.items()}
    res["b3_region"] = {v: {"drift": r.get("drift"), "drift_ref": r.get("drift_ref"), "drift_ref_ci": r.get("drift_ref_ci"),
                            "drift_minus_ref": r.get("drift_minus_ref"), "drift_minus_ref_ci": r.get("drift_minus_ref_ci")}
                        for v, r in V.items() if "drift_ref" in r}
    return res


def analyze(E):
    out = {"cells": {}, "H2a_replication": {}, "H2c": {}, "H2c_aux_ratio_sweep": {}, "iteration_curves": {},
           "layer_b": {}, "incomplete": []}
    for t, k in TASKS.items():
        mw = C.is_mw(t)
        keys = [key for key in E if key[1] == t]
        groups = {}
        for key in keys:
            groups.setdefault((key[0], key[3]), set()).add((key[4], key[5]))
        if not groups:
            continue
        common = set.intersection(*groups.values())
        for g, prs in groups.items():
            if prs != common:
                out["incomplete"].append([t, g[0], g[1], len(prs), len(common)])
        P = C.Pairs(common)
        boot, arrs = {}, {}
        for s in SETTINGS:
            vs = sorted({key[3] for key in keys if key[0] == s})
            if "none" not in vs:
                continue
            y0 = P.arr(E, lambda c, e, s=s: (s, t, 0, "none", c, e), mw=mw)
            cell = {"n_pairs": P.n, "none_mean": float(y0.mean()), "variants": {}}
            for v in vs:
                if v == "none":
                    continue
                kf = lambda c, e, s=s, v=v: (s, t, k, v, c, e)
                yv = P.arr(E, kf, mw=mw)
                st, Db = P.drop_ci(y0, yv, mw)
                row = dict(st, mean=float(yv.mean()))
                for fld in FIELDS:
                    x = P.arr(E, kf, field=fld)
                    mc = P.mean_ci(x)
                    if mc:
                        row[fld], row[fld + "_ci"] = mc["mean"], mc["ci"]
                        arrs[(s, v, fld)] = x
                        if fld == "oei":
                            boot[(s, v, "oei")] = x[P.idx].mean(1)
                boot[(s, v, "D")] = Db
                cell["variants"][v] = row
            out["cells"][f"{t}|{s}"] = cell
        b = out["cells"].get(f"{t}|base")
        if b and "det" in b["variants"] and "shiter" in b["variants"]:
            det, sh = b["variants"]["det"], b["variants"]["shiter"]
            if "oei_ci" in det and "oei_ci" in sh:
                ok_det = det["oei_ci"][1] < 0
                ok_sh = sh["oei_ci"][0] <= 0 <= sh["oei_ci"][1]
                out["H2a_replication"][t] = {"k": k, "det_oei": det["oei"], "det_oei_ci": det["oei_ci"],
                                             "shiter_oei": sh["oei"], "shiter_oei_ci": sh["oei_ci"],
                                             "det_ci_upper_lt_0": bool(ok_det), "shiter_ci_contains_0": bool(ok_sh),
                                             "holds": bool(ok_det and ok_sh)}
        lb = layer_b(out, arrs, P, t, k)
        if lb:
            out["layer_b"][t] = lb
        # H2c over the N levels: confirmatory fixed-count sweep, then the ratio-fixed sweep as auxiliary
        for lab, levels in (("H2c", N_LEVELS), ("H2c_aux_ratio_sweep", N_LEVELS_RATIO)):
            h = n_sweep(out, boot, t, levels)
            if h:
                out.setdefault(lab, {})[t] = h
        # iteration curves
        cur = {}
        for nm, lv in ITER_CURVES.items():
            for v in ("det", "shiter", "shstep"):
                pts = []
                for s, n in lv:
                    c = out["cells"].get(f"{t}|{s}")
                    if c and v in c["variants"]:
                        r = c["variants"][v]
                        pts.append({"setting": s, "iterations": n, "D": r["D"], "D_ci": r["D_ci"],
                                    "oei": r.get("oei"), "oei_ci": r.get("oei_ci"), "oei_indep": r.get("oei_indep"),
                                    "oei_indep_ci": r.get("oei_indep_ci"), "drift": r.get("drift"),
                                    "drift_ci": r.get("drift_ci"), "sel_last": r.get("sel_last"),
                                    "sel_last_ci": r.get("sel_last_ci")})
                cur[f"{nm}|{v}"] = pts
        out["iteration_curves"][t] = cur
    out["H2c_tasks_holding_tdmpc2"] = sum(h["holds"] for h in out["H2c"].values())
    lb = out["layer_b"].values()
    out["layer_b_summary"] = {"b1_tasks_holding_tdmpc2": sum(x.get("b1_H2a_indep", {}).get("holds", False) for x in lb),
                              "b2_tasks_holding_tdmpc2": sum(x.get("b2_H2a_prime", {}).get("holds", False) for x in lb),
                              "rule": "each needs 3 of 4 tasks with PushT (DINO-WM: b1 = oei_ref_mf of E1, b2 = "
                                      "last_pop_dJ - it0_rand_dJ)"}
    out["H2a_replication_tasks_holding_tdmpc2"] = sum(h["holds"] for h in out["H2a_replication"].values())
    return out


def family_points(S):
    """Per-task TD-MPC2 values that the coordinator combines with the DINO-WM family (PushT is the 4th task of H2a
    replication, H2c, (b1), (b2)); the TD-MPC2 share is judged here, the family decision is not."""
    P = {"source": "analyze_e2.py (TD-MPC2, E2 seeds 40000-40029, 90 pairs per setting)",
         "rules": {"H2a_replication": "v5 7.1 layer (a): base, frozen oei; det CI upper < 0 and sh-iter CI contains 0",
                   "H2c": "v5 7.1 + v5.1 8.2: fixed-count sweep N64k/N128k/N256k/base/N1024k; rho_s(N,|OEI_det|) >= 0.8 "
                          "and rho_s(N, D_det) >= 0.8",
                   "b1_H2a_indep": "v5.2 9.1 layer (b, pre-registered after E1): det oei_indep CI upper < 0 and sh-iter "
                                   "oei_indep CI contains 0",
                   "b2_H2a_prime": "v5.2 9.1 layer (b): drift_det CI upper < 0 and r = drift_shiter/drift_det CI upper "
                                   "< 0.5 (ratio per resample, drift_det* >= 0 resamples dropped)",
                   "family_rule": "each: 3 of 4 tasks (mw-push, mw-pick-place, cheetah-run, PushT)"},
         "units": "OEI / drift in value (return) units of the task; D: Meta-World success-rate difference (fraction), "
                  "DMC relative return drop (fraction)", "tasks": {}}
    for t, k in TASKS.items():
        e = {"k": k}
        if t in S["H2a_replication"]:
            e["H2a_replication"] = S["H2a_replication"][t]
        if t in S["H2c"]:
            h = S["H2c"][t]
            e["H2c"] = {kk: h.get(kk) for kk in ("holds", "levels", "N", "oei_det", "oei_det_ci", "abs_oei_det",
                                                  "D_det", "D_det_ci", "rho_N_absOEI_det", "rho_N_D_det",
                                                  "rho_N_absOEI_det_boot", "rho_N_D_det_boot", "aux_rho_D_absOEI_det",
                                                  "aux_rho_N_absOEI_shiter", "aux_rho_N_D_shiter")}
        lb = S["layer_b"].get(t, {})
        if "b1_H2a_indep" in lb:
            e["b1_H2a_indep"] = lb["b1_H2a_indep"]
        if "b2_H2a_prime" in lb:
            b2 = lb["b2_H2a_prime"]
            e["b2_H2a_prime"] = {kk: b2.get(kk) for kk in ("holds", "drift_det", "drift_det_ci",
                                                            "cond1_drift_det_ci_upper_lt_0", "r_shiter",
                                                            "cond2_r_shiter_ci_upper_lt_0.5", "r_shstep", "r_shep",
                                                            "positive_control_outside_rule", "r_ind", "r_detshift")}
        P["tasks"][t] = e
    P["tdmpc2_tasks_holding"] = {"H2a_replication": S["H2a_replication_tasks_holding_tdmpc2"],
                                 "H2c": S["H2c_tasks_holding_tdmpc2"],
                                 "b1_H2a_indep": S["layer_b_summary"]["b1_tasks_holding_tdmpc2"],
                                 "b2_H2a_prime": S["layer_b_summary"]["b2_tasks_holding_tdmpc2"]}
    return P


def _rho_boot_txt(b):
    return "" if not b else f"[{b['ci'][0]:+.2f}, {b['ci'][1]:+.2f}]"


def write_md(S, path):
    L = ["# E2 TD-MPC2 (v5 7.1 층 (a), v5.1 8.2, v5.2 9.1 층 (b))", "",
         "D: Meta-World 성공률 차(비율), DMC 상대 return 하락(비율). OEI·drift: 가치 단위. CI: 짝 bootstrap 10,000회 95%.", ""]
    L += ["## H2a 재현(base, 동결 oei)", "", "| 과제 | det OEI [CI] | sh-iter OEI [CI] | 성립 |", "|---|---|---|---|"]
    for t, h in S["H2a_replication"].items():
        L.append(f"| {t} k={h['k']} | {C.f2(h['det_oei'])} {C.ci_txt(h['det_oei_ci'])} | {C.f2(h['shiter_oei'])} "
                 f"{C.ci_txt(h['shiter_oei_ci'])} | {h['holds']} |")
    for lab, key in (("H2c 확증(고정 수 sweep: elite 64, 정책 prior 24; N64k, N128k, N256k, base, N1024k)", "H2c"),
                     ("보조: 비율 고정 sweep(N64, N128, N256, base, N1024)", "H2c_aux_ratio_sweep")):
        L += ["", f"## {lab}", "",
              "| 과제 | \\|OEI_det\\|(N 순) | D_det(N 순) | ρ(N, \\|OEI\\|) [boot] | ρ(N, D) [boot] | 성립 | "
              "보조 ρ(D, \\|OEI\\|) | 보조 sh-iter ρ(N, \\|OEI\\|) / ρ(N, D) |", "|---|---|---|---|---|---|---|---|"]
        for t, h in S.get(key, {}).items():
            L.append(f"| {t} | {', '.join(f'{x:.2f}' for x in h['abs_oei_det'])} | "
                     f"{', '.join(f'{x:.3f}' for x in h['D_det'])} | {h['rho_N_absOEI_det']:.2f} "
                     f"{_rho_boot_txt(h.get('rho_N_absOEI_det_boot'))} | {h['rho_N_D_det']:.2f} "
                     f"{_rho_boot_txt(h.get('rho_N_D_det_boot'))} | {h['holds'] if key == 'H2c' else '(보조)'} | "
                     f"{h['aux_rho_D_absOEI_det']:.2f} | {C.f2(h.get('aux_rho_N_absOEI_shiter'))} / "
                     f"{C.f2(h.get('aux_rho_N_D_shiter'))} |")
        L += ["", "| 과제 | 변형 | N | D [CI] | OEI | sel_last [CI] | drift [CI] | OEI_indep | sel_indep |",
              "|---|---|---|---|---|---|---|---|---|"]
        for t, h in S.get(key, {}).items():
            for v, pts in h["decomposition"].items():
                for p in pts:
                    L.append(f"| {t} | {v} | {p['N']} | {C.f2(p.get('D'), 3)} {C.ci_txt(p.get('D_ci'), 1, 3)} | "
                             f"{C.f2(p.get('oei'))} | {C.f2(p.get('sel_last'))} {C.ci_txt(p.get('sel_last_ci'))} | "
                             f"{C.f2(p.get('drift'))} {C.ci_txt(p.get('drift_ci'))} | {C.f2(p.get('oei_indep'))} | "
                             f"{C.f2(p.get('sel_indep'))} |")
    L += ["", "## 층 (b)(v5.2, E1 뒤 사전등록, E2 seed만)", "",
          "| 과제 | (b1) det OEI_indep [CI] | (b1) sh-iter OEI_indep [CI] | (b1) 성립 | (b2) drift_det [CI] | "
          "(b2) r_sh-iter [CI] | (b2) 성립 | 양성 대조 r_sh-step / r_sh-ep [CI] |", "|---|---|---|---|---|---|---|---|"]
    for t, b in S["layer_b"].items():
        b1, b2 = b.get("b1_H2a_indep", {}), b.get("b2_H2a_prime", {})
        rs, rp, re_ = b2.get("r_shiter", {}), b2.get("r_shstep", {}), b2.get("r_shep", {})
        L.append(f"| {t} | {C.f2(b1.get('det_oei_indep'))} {C.ci_txt(b1.get('det_ci'))} | "
                 f"{C.f2(b1.get('shiter_oei_indep'))} {C.ci_txt(b1.get('shiter_ci'))} | {b1.get('holds')} | "
                 f"{C.f2(b2.get('drift_det'))} {C.ci_txt(b2.get('drift_det_ci'))} | {C.f2(rs.get('r'))} "
                 f"{C.ci_txt(rs.get('ci'))} | {b2.get('holds')} | {C.f2(rp.get('r'))} {C.ci_txt(rp.get('ci'))} / "
                 f"{C.f2(re_.get('r'))} {C.ci_txt(re_.get('ci'))} |")
    L += ["", "| 과제 | 기술: r_ind [CI] | 기술: r_det+shift [CI] |", "|---|---|---|"]
    for t, b in S["layer_b"].items():
        b2 = b.get("b2_H2a_prime", {})
        ri, rd = b2.get("r_ind", {}), b2.get("r_detshift", {})
        L.append(f"| {t} | {C.f2(ri.get('r'))} {C.ci_txt(ri.get('ci'))} | {C.f2(rd.get('r'))} {C.ci_txt(rd.get('ci'))} |")
    L += ["", "| 과제 | 변형 | sel_last [CI] | sel_indep [CI] | drift | (b3) drift_ref [CI] | drift − drift_ref [CI] |",
          "|---|---|---|---|---|---|---|"]
    for t, b in S["layer_b"].items():
        for v, r in b["sel_last_same_vs_indep"].items():
            g = b["b3_region"].get(v, {})
            L.append(f"| {t} | {v} | {C.f2(r['sel_last'])} {C.ci_txt(r['sel_last_ci'])} | {C.f2(r['sel_indep'])} "
                     f"{C.ci_txt(r['sel_indep_ci'])} | {C.f2(g.get('drift'))} | {C.f2(g.get('drift_ref'))} "
                     f"{C.ci_txt(g.get('drift_ref_ci'))} | {C.f2(g.get('drift_minus_ref'))} "
                     f"{C.ci_txt(g.get('drift_minus_ref_ci'))} |")
    L += ["", "## 반복 수 곡선(warm start 켬: it1, it3, base, it12 / 끔: nows1, nows3, nows6, nows12)", "",
          "| 과제 | 곡선 | 변형 | 반복 | D [CI] | OEI | drift | sel_last |", "|---|---|---|---|---|---|---|---|"]
    for t, cur in S["iteration_curves"].items():
        for nm, pts in cur.items():
            curve, v = nm.split("|")
            for p in pts:
                L.append(f"| {t} | {curve} | {v} | {p['iterations']} | {C.f2(p['D'], 3)} {C.ci_txt(p['D_ci'], 1, 3)} | "
                         f"{C.f2(p.get('oei'))} | {C.f2(p.get('drift'))} | {C.f2(p.get('sel_last'))} |")
    L += ["", "## 설정별 칸", "",
          "| 과제·설정 | 변형 | D [CI] | OEI [CI] | OEI_indep [CI] | sel_last | sel_indep | drift [CI] | 정책 slot 선택 비율 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for cid, c in S["cells"].items():
        for v, r in c["variants"].items():
            L.append(f"| {cid} | {v} | {C.f2(r['D'], 3)} {C.ci_txt(r['D_ci'], 1, 3)} | {C.f2(r.get('oei'))} "
                     f"{C.ci_txt(r.get('oei_ci'))} | {C.f2(r.get('oei_indep'))} {C.ci_txt(r.get('oei_indep_ci'))} | "
                     f"{C.f2(r.get('sel_last'))} | {C.f2(r.get('sel_indep'))} | {C.f2(r.get('drift'))} "
                     f"{C.ci_txt(r.get('drift_ci'))} | {C.f2(r.get('chosen_pi'), 3)} |")
    open(path, "w").write("\n".join(L) + "\n")


CSV_COLS = ["task", "setting", "variant", "n_pairs", "none_mean", "mean", "D", "D_ci", "p_le0", "oei", "oei_ci",
            "oei_indep", "oei_indep_ci", "sel_last", "sel_last_ci", "sel_indep", "sel_indep_ci", "drift", "drift_ci",
            "oei_fresh", "sel_fresh", "chosen_pi", "elite_pi", "elite_pi_w", "drift_ref", "drift_ref_ci",
            "drift_minus_ref", "drift_minus_ref_ci"]


def write_csv(S, path):
    fmt = lambda x: ("[" + ";".join(fmt(y) for y in x) + "]") if isinstance(x, (list, tuple)) else (
        f"{x:.6g}" if isinstance(x, float) else ("" if x is None else str(x)))
    with open(path, "w") as f:
        f.write(",".join(CSV_COLS) + "\n")
        for cid, c in S["cells"].items():
            t, s = cid.split("|")
            for v, r in c["variants"].items():
                row = dict(r, task=t, setting=s, variant=v, n_pairs=c["n_pairs"], none_mean=c["none_mean"])
                f.write(",".join(fmt(row.get(col)) for col in CSV_COLS) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    E = C.load_episodes(a.res, prefix="e2_")
    S = analyze(E)
    S["n_episodes"] = len(E)
    C.write_json(S, f"{a.outdir}/e2_tdmpc2_summary.json")
    C.write_json(family_points(S), f"{a.outdir}/family_points_tdmpc2.json")
    write_md(S, f"{a.outdir}/e2_tdmpc2_tables.md")
    write_csv(S, f"{a.outdir}/e2_tdmpc2_cells.csv")
    print({"episodes": len(E), "H2a_replication": {t: h["holds"] for t, h in S["H2a_replication"].items()},
           "H2c": {t: h["holds"] for t, h in S["H2c"].items()}, "layer_b": S["layer_b_summary"],
           "incomplete_groups": len(S["incomplete"])})


if __name__ == "__main__":
    main()
