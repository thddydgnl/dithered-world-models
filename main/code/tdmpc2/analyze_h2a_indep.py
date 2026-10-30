"""POST-HOC correction of H2a for the TD-MPC2 family (written before the rerun; the rerun needs the user's approval).

The frozen definition (DESIGN v1 0.4 / v4 6.2) values the executed elite with the realized error of the draw that
selected it (same draw: `oei`). The DINO-WM family values its executed plan with one extra forward (`oei_ref_mf`):
persistent error fields kept (det, plan- / episode-fixed tables), fresh-noise variants with a new draw. The TD-MPC2
counterpart is `oei_indep` (qlib.PlanProbe.independent_reeval: the executed elite valued again under the planner's
RNG state with a separate dither generator). This script
  1. checks that every rerun episode reproduces its E1 episode bitwise (compare_runs.compare_job: returns, success,
     refresh counts, every per-step OEI / decision field, plan-rerun records; the rerun does not recompute the
     per-episode prediction errors);
  2. recomputes H2a with oei_indep in place of oei and the E1 rules otherwise unchanged: primary-k cells, pairs =
     (checkpoint, episode) shared by none and every E1 variant, paired bootstrap 10,000 resamples RandomState(0) over
     the sorted pairs (analyze_e1.boot_idx), percentile 95% CI; a task holds if det's OEI CI upper < 0 and sh-iter's
     OEI CI contains 0. det, sh-step and sh-ep have oei_indep == oei by construction (indep_same = 1, checked on the
     smoke split); a variant that is not rerun gets its E1 value only in that case;
  3. reports per variant (descriptive): oei, oei_indep, sel_last, sel_indep, drift, oei_fresh / sel_fresh (tables
     redrawn; table variants only) and the policy-prior shares, with CIs from the same resamples.
The frozen (same-draw) H2a result stays on record unchanged.
usage (server): python analyze_h2a_indep.py --e1 ../out --rerun ../out_h2a_indep --manifest <rerun manifest> \
                    --outdir ../analysis_h2a_indep
       --split smoke: code-path check on the identity run (nothing is interpreted)
"""
import argparse
import glob
import gzip
import json
import os

import numpy as np

import analyze_e1 as A1
from compare_runs import compare_job

H2A = {"mw-push": 3, "mw-pick-place": 3, "cheetah-run": 5}
SAME_BY_CONSTRUCTION = ("det", "shstep", "shep")
KEYS = ["oei", "oei_indep", "sel_last", "sel_indep", "drift", "oei_fresh", "sel_fresh", "e_chosen", "e_indep",
        "indep_same", "chosen_pi", "elite_pi", "elite_pi_w"]
CI_KEYS = ("oei", "oei_indep", "sel_last", "sel_indep", "drift", "oei_fresh", "sel_fresh")


def load_cell_eps(res, split, task, k):
    """E1 episode OEI means of one cell (and the task's none episodes): {(ckpt, variant, ep): oei_mean}."""
    E = {}
    for f in sorted(glob.glob(f"{res}/jobs/episodes/{split}_{task}_s*_k*_*_base_c*.json.gz")):
        m = A1.EP_RE.match(os.path.basename(f)[:-len(".json.gz")])
        if not m or m.group("task") != task or int(m.group("k")) not in (0, k):
            continue
        d = json.load(gzip.open(f, "rt"))
        s = d["spec"]
        if s.get("planner") or s.get("warm_start", True) is False:
            continue
        for e in d["episodes"]:
            E[(int(s["ckpt"]), s["variant"], int(e["ep"]))] = dict(e.get("oei_mean") or {}, ret=e["ret"])
    return E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", required=True)
    ap.add_argument("--rerun", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="eval")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    jobs = [json.loads(l) for l in open(a.manifest) if l.strip() and json.loads(l)["type"] == "episodes"]

    # 1. bitwise identity with the reference (E1) episodes
    ident = {"jobs": 0, "episodes": 0, "differing": [], "missing": []}
    R = {}
    for j in jobs:
        rp, np_ = (f"{d}/jobs/episodes/{j['id']}.json.gz" for d in (a.e1, a.rerun))
        if not os.path.exists(np_):
            ident["missing"].append(j["id"])
            continue
        ign = () if j["instr"].get("perr") else ("perr_traj", "perr_plan")
        c = compare_job(rp, np_, ignore=ign)
        ident["jobs"] += 1
        ident["episodes"] += c["n_eps"]
        if c["diffs"]:
            ident["differing"].append({"id": j["id"], "n_fields": len(c["diffs"]), "first": c["diffs"][:5]})
        if j["planner"] or j.get("warm_start", True) is False:
            continue                       # default-planner cells only (as analyze_e1.load)
        for e in json.load(gzip.open(np_, "rt"))["episodes"]:
            R[(j["task"], int(j["ckpt"]), int(j["k"]), j["variant"], int(e["ep"]))] = e.get("oei_mean") or {}
    ident["all_identical"] = not ident["differing"] and not ident["missing"]
    # v5.2 9.1: a cell with any non-identical (or missing) rerun episode is reported but not judged
    bad_cells = set()
    for j in jobs:
        if j["id"] in ident["missing"] or any(d["id"] == j["id"] for d in ident["differing"]):
            bad_cells.add((j["task"], int(j["k"])))

    # 2./3. cells
    cells = sorted({(j["task"], int(j["k"])) for j in jobs if j["variant"] != "none"})
    out = {"identity": ident, "cells": {}, "H2a_indep": {}}
    for task, k in cells:
        E = load_cell_eps(a.e1, a.split, task, k)
        vs = [v for v in A1.VARS if any(vv == v for (_, vv, _) in E)]
        base = {(c, e) for (c, v, e) in E if v == "none"}
        for v in vs:
            base &= {(c, e) for (c, vv, e) in E if vv == v}
        prs = sorted(base)
        n = len(prs)
        if n == 0:
            continue
        idx = A1.boot_idx(n)
        C = {"n_pairs": n, "variants": {}}
        for v in vs:
            rr = [R.get((task, c, k, v, e)) for c, e in prs]
            rerun = all(r is not None for r in rr)
            row = {"rerun": rerun}
            for kk in KEYS:
                if rerun:
                    x = np.array([r.get(kk, np.nan) for r in rr], float)
                elif kk == "oei_indep" and v in SAME_BY_CONSTRUCTION:
                    x = np.array([E[(c, v, e)].get("oei", np.nan) for c, e in prs], float)
                    row["oei_indep_from"] = "E1 oei (identical by construction)"
                elif kk in ("oei", "sel_last", "drift", "e_chosen"):
                    x = np.array([E[(c, v, e)].get(kk, np.nan) for c, e in prs], float)
                else:
                    continue
                if not np.isfinite(x).all():
                    continue
                row[kk] = float(x.mean())
                if kk in CI_KEYS:
                    row[kk + "_ci"] = A1.ci95(x[idx].mean(1))
            if rerun and v in SAME_BY_CONSTRUCTION:
                row["indep_equals_same_draw"] = bool(all(r.get("oei_indep") == r.get("oei") for r in rr))
            C["variants"][v] = row
        out["cells"][f"{task}|k{k}"] = C
        if H2A.get(task) == k and (task, k) in bad_cells:
            out["H2a_indep"][task] = {"k": k, "judged": False,
                                      "reason": "rerun episodes not bitwise identical to E1 (or missing)"}
            continue
        if H2A.get(task) == k and "det" in C["variants"] and "shiter" in C["variants"]:
            det, sh = C["variants"]["det"], C["variants"]["shiter"]
            if "oei_indep_ci" in det and "oei_indep_ci" in sh:
                ok_det = det["oei_indep_ci"][1] < 0
                ok_sh = sh["oei_indep_ci"][0] <= 0 <= sh["oei_indep_ci"][1]
                out["H2a_indep"][task] = {"k": k, "det_oei_indep": det["oei_indep"], "det_ci": det["oei_indep_ci"],
                                          "shiter_oei_indep": sh["oei_indep"], "shiter_ci": sh["oei_indep_ci"],
                                          "shiter_oei_same_draw": sh.get("oei"), "shiter_same_draw_ci": sh.get("oei_ci"),
                                          "det_ci_upper_lt_0": bool(ok_det), "shiter_ci_contains_0": bool(ok_sh),
                                          "holds": bool(ok_det and ok_sh), "judged": True}
    out["H2a_indep_tasks_holding"] = sum(h.get("holds", False) for h in out["H2a_indep"].values())
    out["label"] = ("E1 H2a re-measured with the independent draw (v5.2 9.1, post-hoc correction shown beside the "
                    "frozen verdict; it does not replace it)")
    json.dump(out, open(f"{a.outdir}/h2a_indep_summary.json", "w"), indent=1, default=float)
    write_md(out, f"{a.outdir}/h2a_indep_tables.md")
    print(json.dumps({"identity": {k: ident[k] for k in ("jobs", "episodes", "all_identical")},
                      "n_differing": len(ident["differing"]), "H2a_indep": out["H2a_indep"]}, default=float))


def write_md(S, path):
    f2 = lambda x: "—" if x is None else f"{x:+.2f}"
    ci = lambda c: "" if not c else f"[{c[0]:+.2f}, {c[1]:+.2f}]"
    f0 = lambda x: "—" if x is None else f"{x:.2f}"
    L = ["# TD-MPC2 E1 H2a의 독립 draw판 재측정 (DESIGN_main v5.2 9.1, 사후 정정: 동결 판정 옆에 싣고 대신하지 않는다)", "",
         f"동일성 검사: job {S['identity']['jobs']}개, episode {S['identity']['episodes']}개, "
         f"다른 job {len(S['identity']['differing'])}개, 빠진 job {len(S['identity']['missing'])}개", ""]
    L += ["| 과제 | k | det OEI_indep [CI] | sh-iter OEI_indep [CI] | sh-iter OEI(같은 draw) [CI] | 성립 |", "|---|---|---|---|---|---|"]
    for t, h in S["H2a_indep"].items():
        if not h.get("judged", True):
            L.append(f"| {t} | {h['k']} | 판정 안 함: {h['reason']} | | | |")
            continue
        L.append(f"| {t} | {h['k']} | {f2(h['det_oei_indep'])} {ci(h['det_ci'])} | {f2(h['shiter_oei_indep'])} "
                 f"{ci(h['shiter_ci'])} | {f2(h['shiter_oei_same_draw'])} {ci(h['shiter_same_draw_ci'])} | {h['holds']} |")
    L += ["", "| 칸 | 변형 | 재실행 | OEI | OEI_indep | sel_last | sel_indep | drift | OEI_fresh | 선택 elite가 정책 slot |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for cid, C in S["cells"].items():
        for v, r in C["variants"].items():
            L.append(f"| {cid} | {v} | {r['rerun']} | {f2(r.get('oei'))} | {f2(r.get('oei_indep'))} | "
                     f"{f2(r.get('sel_last'))} | {f2(r.get('sel_indep'))} | {f2(r.get('drift'))} | "
                     f"{f2(r.get('oei_fresh'))} | {f0(r.get('chosen_pi'))} |")
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
