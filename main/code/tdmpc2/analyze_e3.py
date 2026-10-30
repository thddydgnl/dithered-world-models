"""E3 analysis of the TD-MPC2 family, DESIGN_main v5 7.2. Written before any E3 outcome existed; checked on synthetic
data only (test_v5_analyses.py).

  - 2-axis cells (12 tasks, primary k, E3 seeds 50000-50029, 90 pairs): D of none-relative drop and red vs det for
    sh-iter, sh-step, sh-ep, ind, indstep, indep (+ det), with the OEI versions and components
  - table-size sweep (mw-push, mw-pick-place, cheetah-run, walker-walk): D_sh-fixed at the 4 shared plan-fixed table
    levels (N_eff = 4, 128, 2,048, 6,144: shstep_l, shstep_g32, shstep_u, shstep); H4a per eligible task (E1
    primary-k det drop real under Holm): Spearman(N_eff, D) <= -0.8; the family holds with 3 of 4 eligible tasks
    (both families must hold for H4a); auxiliary: the median of the per-task Spearman; indstep (candidate x H x unit)
    reported separately (candidate-axis contrast sh-step vs indstep); the fresh controls shiter_l / shiter_g32 and the
    persistence cost D(shstep_g) - D(shiter_g) with paired CIs
  - with --final only, and only when every E3 episode job is complete (v5 7.0: computed once, after all E3 data):
      H4b: Spearman(near-tie density, CRN gain) >= 0.5 over the 12 tasks; density = -(median top1-top2 gap / std of
        the last-iteration candidate set, E1 decision jobs); CRN gain = D_ind - D_sh-iter (E3); auxiliary: the
        first-iteration density
      DMC explanation: tasks whose det drop is real on the E3 seeds (one-sided bootstrap p < 0.025); S_det =
        drift_det / oei_det (cell means; a task whose oei_det CI contains 0 is excluded); outcome red_sh-iter;
        Spearman(S_det, red) >= 0.5 -> "explained"; else Spearman(density, red) >= 0.5 -> "partially explained";
        else "not explained"
  - v5.1 8.3: var_cand (f_var_cand, across-candidate variance of the first-iteration cost error, in-episode at the
    OEI steps; var_cand in the decision rows) per table level as an auxiliary predictor of D (H4a unchanged)
  - decision jobs of the new variants: rho_iter, rho_iter_perm, common_frac, var_cand (descriptive)
usage (server): python analyze_e3.py --res ~/pmx/A/tdmpc2_main/out_e3 --e1 ~/pmx/A/tdmpc2_main/analysis_e1/\
e1_tdmpc2_summary.json --manifest ../v5/manifests/e3.jsonl --outdir ~/pmx/A/tdmpc2_main/analysis_e3 [--final]

Changes of 2026-10-23 (E2/E3 analysis worker, after the runs ended but before any E3 outcome was read; rules unchanged,
see worklogs/E2E3_tdmpc2.md):
  - rho threshold decisions use v5_common.le / ge with a 1e-9 tolerance: scipy returns -0.7999999999999999 for one
    adjacent swap of 4 points, which v5 7.2 H4a explicitly allows ("4점이라 이웃 자리바꿈 한 번까지 허용"); the same
    for H4b >= 0.5 and the DMC explanation >= 0.5
  - bootstrap rho distribution of H4a with average ranks for ties (v5_common.spearman_rows)
  - --final is one-shot: it refuses to run when e3_tdmpc2_final.json already exists in --outdir (v5 7.0 "once");
    the H4b / DMC results are written to e3_tdmpc2_final.json and h4b_points_tdmpc2.json (per-task values the
    coordinator needs for the 15-task auxiliary), the CRN gain gets its paired bootstrap CI; a later run without
    --final re-reads e3_tdmpc2_final.json into the summary and tables instead of recomputing
  - descriptive additions: paired contrasts of the 2-axis cells (the pre-stated expectation indstep ~ ind ~ indep in D,
    oei_indep and drift; the sharing and persistence axes), per-level CIs in the sweep, decision-job means over the
    checkpoints, CSV of the cells
"""
import argparse
import glob
import gzip
import json
import os
import time

import numpy as np

import v5_common as C

PRIMARY = {"mw-button-press-topdown": 3, "mw-door-open": 4, "mw-drawer-open": 3, "mw-peg-insert-side": 4,
           "mw-pick-place": 3, "mw-push": 3, "mw-sweep-into": 3, "mw-window-open": 4, "cheetah-run": 5,
           "finger-turn-hard": 4, "quadruped-walk": 3, "walker-walk": 3}
AXIS = ["det", "shiter", "shstep", "shep", "ind", "indstep", "indep"]
SWEEP = ["mw-push", "mw-pick-place", "cheetah-run", "walker-walk"]
LEVELS = [("shstep_l", 4), ("shstep_g32", 128), ("shstep_u", 2048), ("shstep", 6144)]
FRESH = [("shstep_l", "shiter_l"), ("shstep_g32", "shiter_g32")]
FIELDS = ("oei", "oei_indep", "sel_last", "sel_indep", "drift", "oei_fresh", "sel_fresh", "chosen_pi", "f_var_cand",
          "f_sd_v0")
# paired contrasts a - b of the 2-axis cells (descriptive): the pre-stated expectation (v5 7.2) indstep ~ ind ~ indep,
# the sharing axis at each persistence level, the persistence axis within the shared and the independent cells
CONTRASTS = [("indstep-ind", "indstep", "ind"), ("indep-ind", "indep", "ind"), ("indep-indstep", "indep", "indstep"),
             ("ind-shiter", "ind", "shiter"), ("indstep-shstep", "indstep", "shstep"), ("indep-shep", "indep", "shep"),
             ("shstep-shiter", "shstep", "shiter"), ("shep-shiter", "shep", "shiter"),
             ("shep-shstep", "shep", "shstep")]
CONTRAST_FIELDS = ("oei_indep", "drift", "oei")


def task_cells(E, t):
    k = PRIMARY[t]
    mw = C.is_mw(t)
    keys = [key for key in E if key[1] == t and key[0] == "base"]
    vs = sorted({key[3] for key in keys})
    groups = {v: {(key[4], key[5]) for key in keys if key[3] == v} for v in vs}
    common = set.intersection(*groups.values()) if groups else set()
    P = C.Pairs(common)
    if "none" not in vs or P.n == 0:
        return None, None, None, None
    y0 = P.arr(E, lambda c, e: ("base", t, 0, "none", c, e), mw=mw)
    cell = {"k": k, "n_pairs": P.n, "none_mean": float(y0.mean()), "variants": {},
            "incomplete_variants": [v for v, g in groups.items() if g != common]}
    Db, X = {}, {}
    for v in vs:
        if v == "none":
            continue
        kf = lambda c, e, v=v: ("base", t, k, v, c, e)
        yv = P.arr(E, kf, mw=mw)
        st, Db[v] = P.drop_ci(y0, yv, mw)
        row = dict(st, mean=float(yv.mean()))
        for fld in FIELDS:
            x = P.arr(E, kf, field=fld)
            mc = P.mean_ci(x)
            if mc:
                row[fld], row[fld + "_ci"] = mc["mean"], mc["ci"]
                X[(v, fld)] = x
        cell["variants"][v] = row
    if "det" in Db:
        dd = Db["det"]
        for v, row in cell["variants"].items():
            if v == "det":
                continue
            row["Dv_minus_Ddet"] = row["D"] - cell["variants"]["det"]["D"]
            row["Dv_minus_Ddet_ci"] = C.A1.ci95(Db[v] - dd)
            with np.errstate(divide="ignore", invalid="ignore"):
                rb = np.where(dd > 0, (dd - Db[v]) / np.where(dd > 0, dd, 1.0), np.nan)
            D0 = cell["variants"]["det"]["D"]
            row["red"] = float((D0 - row["D"]) / D0) if D0 > 0 else None
            row["red_ci"] = C.A1.ci95(rb)
    # paired contrasts (descriptive)
    con = {}
    for nm, va, vb in CONTRASTS:
        if va not in cell["variants"] or vb not in cell["variants"]:
            continue
        r = {"D": cell["variants"][va]["D"] - cell["variants"][vb]["D"], "D_ci": C.A1.ci95(Db[va] - Db[vb])}
        for fld in CONTRAST_FIELDS:
            if (va, fld) in X and (vb, fld) in X:
                dx = X[(va, fld)] - X[(vb, fld)]
                r[fld] = float(dx.mean())
                r[fld + "_ci"] = C.A1.ci95(dx[P.idx].mean(1))
        con[nm] = r
    cell["contrasts"] = con
    return cell, Db, P, X


def ratio_point_ci(num, den, idx):
    """mean(num) / mean(den) and its percentile CI with the ratio recomputed per resample (descriptive)."""
    nb, db = num[idx].mean(1), den[idx].mean(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rb = np.where(db != 0, nb / np.where(db != 0, db, 1.0), np.nan)
    return (float(num.mean() / den.mean()) if den.mean() != 0 else float("nan")), C.A1.ci95(rb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--e1", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    final_path = f"{a.outdir}/e3_tdmpc2_final.json"
    if a.final and os.path.exists(final_path):
        raise SystemExit(f"REFUSED: {final_path} exists; H4b and the DMC explanation are computed once (v5 7.0)")
    E = C.load_episodes(a.res, prefix="e3_")
    e1 = json.load(open(a.e1))
    jobs = [json.loads(l) for l in open(a.manifest) if l.strip()]
    missing = [j["id"] for j in jobs if j["type"] == "episodes" and
               not os.path.exists(f"{a.res}/jobs/episodes/{j['id']}.json.gz")]
    out = {"n_episode_jobs_missing": len(missing), "n_episodes": len(E), "cells": {}, "sweep": {}, "H4a": {},
           "decision": {}, "decision_mean": {}}
    Dboot = {}
    for t in PRIMARY:
        cell, Db, P, X = task_cells(E, t)
        if cell is None:
            continue
        out["cells"][t] = cell
        Dboot[t] = (Db, P, X)
    # sweep / H4a
    elig = {t: bool(e1["cells"].get(f"{t}:{PRIMARY[t]}", {}).get("det_drop_real_holm")) for t in SWEEP}
    rhos = {}
    for t in SWEEP:
        c = out["cells"].get(t)
        if not c or not all(lv in c["variants"] for lv, _ in LEVELS):
            continue
        Db, P, X = Dboot[t]
        ne = np.array([n for _, n in LEVELS], float)
        D = np.array([c["variants"][lv]["D"] for lv, _ in LEVELS])
        r = C.spearman(ne, D)
        rb = C.spearman_rows(np.stack([Db[lv] for lv, _ in LEVELS], 1), ne)
        sw = {"levels": [lv for lv, _ in LEVELS], "N_eff": ne.tolist(), "D_shfixed": D.tolist(),
              "D_shfixed_ci": [c["variants"][lv]["D_ci"] for lv, _ in LEVELS], "spearman": r,
              "eligible_E1_det_drop": elig[t], "holds": C.le(r, -0.8),
              "spearman_boot": C.boot_rho_summary(rb, -0.8, "le")}
        sw["spearman_boot_ci"] = sw["spearman_boot"]["ci"]
        # v5.1 8.3 auxiliary predictor: across-candidate cost-error variance of the first MPPI candidate set
        vc = [c["variants"][lv].get("f_var_cand") for lv, _ in LEVELS]
        if all(x is not None for x in vc):
            sw["var_cand"] = vc
            sw["var_cand_ci"] = [c["variants"][lv].get("f_var_cand_ci") for lv, _ in LEVELS]
            sw["aux_spearman_var_cand_D"] = C.spearman(vc, D)
            sw["aux_spearman_N_eff_var_cand"] = C.spearman(ne, vc)
        if "indstep" in c["variants"]:
            sw["indstep_D"] = c["variants"]["indstep"]["D"]
            sw["indstep_D_ci"] = c["variants"]["indstep"]["D_ci"]
            sw["indstep_var_cand"] = c["variants"]["indstep"].get("f_var_cand")
            sw["indstep_minus_shstep"] = c["variants"]["indstep"]["D"] - c["variants"]["shstep"]["D"]
            sw["indstep_minus_shstep_ci"] = C.A1.ci95(Db["indstep"] - Db["shstep"])
        for fixed, fresh in FRESH:
            if fresh in c["variants"]:
                sw[f"{fresh}_D"] = c["variants"][fresh]["D"]
                sw[f"{fresh}_D_ci"] = c["variants"][fresh]["D_ci"]
                sw[f"{fresh}_var_cand"] = c["variants"][fresh].get("f_var_cand")
                sw[f"persistence_cost_{fixed}"] = c["variants"][fixed]["D"] - c["variants"][fresh]["D"]
                sw[f"persistence_cost_{fixed}_ci"] = C.A1.ci95(Db[fixed] - Db[fresh])
        if "shiter" in c["variants"]:   # the full-granularity counterpart (2-axis cells): D(shstep) - D(shiter)
            sw["persistence_cost_shstep"] = c["variants"]["shstep"]["D"] - c["variants"]["shiter"]["D"]
            sw["persistence_cost_shstep_ci"] = C.A1.ci95(Db["shstep"] - Db["shiter"])
        out["sweep"][t] = sw
        if elig[t]:
            rhos[t] = r
    n_el = sum(elig.values())
    n_hold = sum(C.le(r, -0.8) for r in rhos.values())
    out["H4a"] = {"eligible_tasks": [t for t, ok in elig.items() if ok],
                  "holding": [t for t, r in rhos.items() if C.le(r, -0.8)],
                  "n_eligible": n_el, "n_holding": int(n_hold),
                  "family_holds_tdmpc2": bool(n_el > 0 and n_hold > n_el / 2),
                  "aux_median_spearman": float(np.median(list(rhos.values()))) if rhos else None,
                  "note": "H4a needs both families (DINO-WM: 2 of its 3 eligible tasks)"}
    # decision jobs
    acc = {}
    for f in sorted(glob.glob(f"{a.res}/jobs/decision/decision_e3_*.json.gz")):
        d = json.load(gzip.open(f, "rt"))
        out["decision"][f"{d['task']}|s{d['ckpt']}"] = {
            v: dict({kk: sv.get(f"first_{kk}") for kk in ("rho_iter", "rho_iter_perm", "common_frac", "var_cand")},
                    dev_mean_rel=sv.get("dev_mean_rel"))
            for v, sv in d["summary"].items()}
        for v, sv in d["summary"].items():
            for kk in ("rho_iter", "rho_iter_perm", "common_frac", "var_cand"):
                if sv.get(f"first_{kk}") is not None:
                    acc.setdefault((d["task"], v, kk), []).append(sv[f"first_{kk}"])
            if sv.get("dev_mean_rel") is not None:
                acc.setdefault((d["task"], v, "dev_mean_rel"), []).append(sv["dev_mean_rel"])
    for (t, v, kk), x in acc.items():
        out["decision_mean"].setdefault(t, {}).setdefault(v, {})[kk] = float(np.mean(x))
        out["decision_mean"][t][v]["n_ckpts"] = len(x)
    # H4b and the DMC explanation: once, after all E3 data
    if a.final:
        if missing:
            raise SystemExit(f"REFUSED: --final needs every E3 episode job ({len(missing)} missing)")
        nt = e1["near_tie"]
        dens = {t: -nt[t]["last"]["median"] for t in PRIMARY if t in nt}
        dens1 = {t: -nt[t]["first"]["median"] for t in PRIMARY if t in nt}
        gain, gain_ci = {}, {}
        for t, c in out["cells"].items():
            Db = Dboot[t][0]
            gain[t] = c["variants"]["ind"]["D"] - c["variants"]["shiter"]["D"]
            gain_ci[t] = C.A1.ci95(Db["ind"] - Db["shiter"])
        ts = [t for t in PRIMARY if t in dens and t in gain]
        r4b = C.spearman([dens[t] for t in ts], [gain[t] for t in ts])
        r4b1 = C.spearman([dens1[t] for t in ts], [gain[t] for t in ts])
        fin = {"computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "rule": "v5 7.0: computed once after all E3 data; this run refuses if this file exists"}
        fin["H4b"] = {"tasks": ts, "density_last": [dens[t] for t in ts], "density_first": [dens1[t] for t in ts],
                      "crn_gain": [gain[t] for t in ts], "crn_gain_ci": [gain_ci[t] for t in ts],
                      "spearman": r4b, "holds": C.ge(r4b, 0.5), "aux_spearman_first_set": r4b1}
        rows = []
        for t, c in out["cells"].items():
            det, sh = c["variants"]["det"], c["variants"]["shiter"]
            Db, P, X = Dboot[t]
            real = det["p_le0"] < 0.025
            s_def = "oei_ci" in det and not (det["oei_ci"][0] <= 0 <= det["oei_ci"][1])
            S = det["drift"] / det["oei"] if s_def and det.get("oei") else None
            S_pt, S_ci = ratio_point_ci(X[("det", "drift")], X[("det", "oei")], P.idx) if s_def else (None, None)
            rows.append({"task": t, "D_det": det["D"], "D_det_ci": det["D_ci"], "p_le0_det": det["p_le0"],
                         "det_drop_real_e3": bool(real), "oei_det": det.get("oei"), "oei_det_ci": det.get("oei_ci"),
                         "drift_det": det.get("drift"), "drift_det_ci": det.get("drift_ci"), "S_det": S,
                         "S_det_ci_descriptive": S_ci, "S_defined": bool(s_def), "red_shiter": sh.get("red"),
                         "red_shiter_ci": sh.get("red_ci"), "density": dens.get(t)})
        use = [r for r in rows if r["det_drop_real_e3"] and r["S_defined"] and r["red_shiter"] is not None]
        r1 = C.spearman([r["S_det"] for r in use], [r["red_shiter"] for r in use])
        use2 = [r for r in rows if r["det_drop_real_e3"] and r["red_shiter"] is not None and r["density"] is not None]
        r2 = C.spearman([r["density"] for r in use2], [r["red_shiter"] for r in use2])
        verdict = ("explained" if C.ge(r1, 0.5) else "partially explained" if C.ge(r2, 0.5) else "not explained")
        fin["DMC_explanation"] = {"rows": rows, "spearman_S_red": r1, "n_tasks_S": len(use),
                                  "tasks_S": [r["task"] for r in use], "spearman_density_red": r2,
                                  "n_tasks_density": len(use2), "tasks_density": [r["task"] for r in use2],
                                  "verdict": verdict, "excluded_from_S": [r["task"] for r in rows if r not in use],
                                  "excluded_from_density": [r["task"] for r in rows if r not in use2]}
        out.update(fin)
        out["final_computed_at"] = fin["computed_at"]
        C.write_json(fin, final_path)
        pts = {"source": "analyze_e3.py --final (TD-MPC2; density from the E1 decision jobs, 450 fixed states = 3 "
                         "checkpoints x 150; CRN gain = D_ind - D_sh-iter on the E3 seeds 50000-50029, 90 pairs)",
               "rule": "v5 7.2 H4b: density = -(median of the unquantized top1-top2 gap / candidate std); confirmatory "
                       "TD-MPC2 12 tasks Spearman(density_last, crn_gain) >= 0.5; auxiliary: the first-iteration set "
                       "and the 15-task pool with DINO-WM (coordinator)",
               "units": "crn_gain: Meta-World success-rate difference (fraction), DMC relative return drop (fraction)",
               "tasks": {t: {"k": PRIMARY[t], "density_last": dens[t], "density_first": dens1[t],
                             "gap_median_last": nt[t]["last"]["median"], "gap_q25_last": nt[t]["last"]["q25"],
                             "gap_q75_last": nt[t]["last"]["q75"], "gap_median_first": nt[t]["first"]["median"],
                             "n_states": nt[t]["last"]["n_states"], "crn_gain": gain[t], "crn_gain_ci": gain_ci[t],
                             "D_ind": out["cells"][t]["variants"]["ind"]["D"],
                             "D_ind_ci": out["cells"][t]["variants"]["ind"]["D_ci"],
                             "D_shiter": out["cells"][t]["variants"]["shiter"]["D"],
                             "D_shiter_ci": out["cells"][t]["variants"]["shiter"]["D_ci"]} for t in ts},
               "tdmpc2_12task": {"spearman_last": r4b, "holds": C.ge(r4b, 0.5), "aux_spearman_first": r4b1}}
        C.write_json(pts, f"{a.outdir}/h4b_points_tdmpc2.json")
    elif os.path.exists(final_path):
        # a later non-final run (e.g. regenerating tables) re-reads the one-shot H4b / DMC result, never recomputes it
        fin = json.load(open(final_path))
        out.update({kk: fin[kk] for kk in ("H4b", "DMC_explanation") if kk in fin})
        out["final_computed_at"] = fin.get("computed_at")
        out["final_loaded_from"] = final_path
    C.write_json(out, f"{a.outdir}/e3_tdmpc2_summary.json")
    write_md(out, missing, f"{a.outdir}/e3_tdmpc2_tables.md")
    write_csv(out, f"{a.outdir}/e3_tdmpc2_cells.csv")
    print({"missing": len(missing), "H4a": out["H4a"]["family_holds_tdmpc2"], "final": a.final,
           "H4b": out.get("H4b", {}).get("holds"), "DMC": out.get("DMC_explanation", {}).get("verdict")})


def write_md(out, missing, path):
    f2, ci = C.f2, C.ci_txt
    L = ["# E3 TD-MPC2 (v5 7.2, v5.1 8.3)", "", f"빠진 episode job: {len(missing)}, episode {out['n_episodes']}", "",
         "D: Meta-World 성공률 차(비율), DMC 상대 return 하락(비율). red = (D_det − D_v)/D_det. CI: 짝 bootstrap 10,000회 95%.", "",
         "## 2축 칸(12과제 1차 k)", "",
         "| 과제 | 변형 | D [CI] | red [CI] | D_v − D_det [CI] | OEI | OEI_indep [CI] | drift [CI] | OEI_fresh | var_cand |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for t, c in out["cells"].items():
        for v, r in c["variants"].items():
            L.append(f"| {t} k={c['k']} | {v} | {f2(r['D'], 3)} {ci(r['D_ci'], 1, 3)} | {f2(r.get('red'))} "
                     f"{ci(r.get('red_ci'))} | {f2(r.get('Dv_minus_Ddet'), 3)} {ci(r.get('Dv_minus_Ddet_ci'), 1, 3)} | "
                     f"{f2(r.get('oei'))} | {f2(r.get('oei_indep'))} {ci(r.get('oei_indep_ci'))} | "
                     f"{f2(r.get('drift'))} {ci(r.get('drift_ci'))} | {f2(r.get('oei_fresh'))} | "
                     f"{'' if r.get('f_var_cand') is None else format(r['f_var_cand'], '.4g')} |")
    L += ["", "## 짝 대조(기술): a − b", "", "| 과제 | 대조 | ΔD [CI] | ΔOEI_indep [CI] | Δdrift [CI] |", "|---|---|---|---|---|"]
    for t, c in out["cells"].items():
        for nm, r in c.get("contrasts", {}).items():
            L.append(f"| {t} | {nm} | {f2(r['D'], 3)} {ci(r['D_ci'], 1, 3)} | {f2(r.get('oei_indep'))} "
                     f"{ci(r.get('oei_indep_ci'))} | {f2(r.get('drift'))} {ci(r.get('drift_ci'))} |")
    L += ["", "## 표 크기 sweep(H4a)", "",
          "| 과제 | D(N_eff 4, 128, 2048, 6144) | Spearman [boot] | 자격 | 성립 | var_cand(N_eff 순) | ρ(var_cand, D) | "
          "indstep D | shiter_l D / shiter_g32 D | 지속 비용 l / g32 / 기본 [CI] |", "|---|---|---|---|---|---|---|---|---|---|"]
    for t, sw in out["sweep"].items():
        vc = ", ".join(f"{x:.4g}" for x in sw.get("var_cand", []))
        pc = " / ".join(f"{f2(sw.get(f'persistence_cost_{x}'), 3)} {ci(sw.get(f'persistence_cost_{x}_ci'), 1, 3)}"
                        for x in ("shstep_l", "shstep_g32", "shstep"))
        L.append(f"| {t} | {', '.join(f'{x:.3f}' for x in sw['D_shfixed'])} | {sw['spearman']:.2f} "
                 f"{ci(sw['spearman_boot_ci'])} | {sw['eligible_E1_det_drop']} | {sw['holds']} | {vc} | "
                 f"{f2(sw.get('aux_spearman_var_cand_D'))} | {f2(sw.get('indstep_D'), 3)} | "
                 f"{f2(sw.get('shiter_l_D'), 3)} / {f2(sw.get('shiter_g32_D'), 3)} | {pc} |")
    h = out["H4a"]
    L += ["", f"H4a TD-MPC2 몫: 자격 {h['n_eligible']}과제 중 성립 {h['n_holding']} → {h['family_holds_tdmpc2']} "
              f"(보조 Spearman 중앙값 {f2(h['aux_median_spearman'])})", ""]
    L += ["## decision job(새 변형, 체크포인트 평균)", "",
          "| 과제 | 변형 | rho_iter | rho_iter_perm | common_frac | var_cand | 계획 편차(seed 바닥 배수) |", "|---|---|---|---|---|---|---|"]
    for t, dv in out["decision_mean"].items():
        for v, r in dv.items():
            L.append(f"| {t} | {v} | {f2(r.get('rho_iter'))} | {f2(r.get('rho_iter_perm'))} | {f2(r.get('common_frac'))} | "
                     f"{'' if r.get('var_cand') is None else format(r['var_cand'], '.4g')} | {f2(r.get('dev_mean_rel'))} |")
    if "H4b" in out:
        hb = out["H4b"]
        L += ["", "## H4b(한 번만 계산)", "", "| 과제 | 밀도(마지막) | 밀도(첫) | CRN 이득 D_ind − D_sh-iter [CI] |", "|---|---|---|---|"]
        for i, t in enumerate(hb["tasks"]):
            L.append(f"| {t} | {hb['density_last'][i]:+.3f} | {hb['density_first'][i]:+.4f} | "
                     f"{f2(hb['crn_gain'][i], 3)} {ci(hb['crn_gain_ci'][i], 1, 3)} |")
        L.append(f"\nSpearman(밀도 마지막, 이득) = {f2(hb['spearman'])} → 성립 {hb['holds']}; 보조(첫 반복) "
                 f"{f2(hb['aux_spearman_first_set'])}")
        dm = out["DMC_explanation"]
        L += ["", "## DMC 설명(한 번만 계산)", "",
              "| 과제 | D_det [CI] | p | 실재 | OEI_det [CI] | drift_det | S_det | red_sh-iter [CI] | 밀도 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in dm["rows"]:
            L.append(f"| {r['task']} | {f2(r['D_det'], 3)} {ci(r['D_det_ci'], 1, 3)} | {r['p_le0_det']:.4f} | "
                     f"{r['det_drop_real_e3']} | {f2(r['oei_det'])} {ci(r['oei_det_ci'])} | {f2(r['drift_det'])} | "
                     f"{f2(r['S_det'])} | {f2(r['red_shiter'])} {ci(r['red_shiter_ci'])} | {f2(r['density'], 3)} |")
        L.append(f"\nSpearman(S_det, red) = {f2(dm['spearman_S_red'])} (n = {dm['n_tasks_S']}), Spearman(밀도, red) = "
                 f"{f2(dm['spearman_density_red'])} (n = {dm['n_tasks_density']}) → {dm['verdict']}; S에서 뺀 과제 "
                 f"{dm['excluded_from_S']}, 밀도에서 뺀 과제 {dm['excluded_from_density']}")
    open(path, "w").write("\n".join(L) + "\n")


CSV_COLS = ["task", "k", "variant", "n_pairs", "none_mean", "mean", "D", "D_ci", "p_le0", "red", "red_ci",
            "Dv_minus_Ddet", "Dv_minus_Ddet_ci", "oei", "oei_ci", "oei_indep", "oei_indep_ci", "sel_last", "sel_last_ci",
            "sel_indep", "sel_indep_ci", "drift", "drift_ci", "oei_fresh", "oei_fresh_ci", "sel_fresh", "chosen_pi",
            "f_var_cand", "f_var_cand_ci", "f_sd_v0"]


def write_csv(out, path):
    fmt = lambda x: ("[" + ";".join(fmt(y) for y in x) + "]") if isinstance(x, (list, tuple)) else (
        f"{x:.6g}" if isinstance(x, float) else ("" if x is None else str(x)))
    with open(path, "w") as f:
        f.write(",".join(CSV_COLS) + "\n")
        for t, c in out["cells"].items():
            for v, r in c["variants"].items():
                row = dict(r, task=t, k=c["k"], variant=v, n_pairs=c["n_pairs"], none_mean=c["none_mean"])
                f.write(",".join(fmt(row.get(col)) for col in CSV_COLS) + "\n")


if __name__ == "__main__":
    main()
