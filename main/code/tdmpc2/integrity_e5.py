"""E5 TD-MPC2 integrity check (no H3b / QAT / weight outcome is printed): every job of the 7 E5 manifests has one result
whose spec equals the manifest entry; seeds, splits and episode ids; cells against DESIGN_main v5 7.3 (written from
the design text, not from make_manifest); GPU / CVD / host / MPS pipe per stage; error and lock files; the resolved
baseline of every evaluation job against selection.json and the evaluation start times against the selection time;
the mixed baseline's 8-bit layers; QAT paths, data files and training specs; the MSE clip factors against the ends of
the v5.1 8.1 grid; the per-tensor range check of the statistics; and an independent recomputation of the baseline
selection from the held episodes (v5 7.3: argmin_B D_det(B), tie -> earlier in the list).
usage (server, CPU only): CUDA_VISIBLE_DEVICES="" python integrity_e5.py --res <out_e5> --mdir <v5/manifests> --out <json>
"""
import argparse
import glob
import gzip
import json
import math
import os
import time
from collections import Counter, defaultdict

# DESIGN_main v5 7.3: the 19 E1 Holm-qualifying 3-4 bit cells, the 11 QAT cells (primary k among them), sweep tasks
CELLS = {("mw-button-press-topdown", 3), ("mw-door-open", 3), ("mw-door-open", 4), ("mw-drawer-open", 3),
         ("mw-peg-insert-side", 3), ("mw-peg-insert-side", 4), ("mw-pick-place", 3), ("mw-push", 3), ("mw-push", 4),
         ("mw-sweep-into", 3), ("mw-window-open", 3), ("mw-window-open", 4),
         ("finger-turn-hard", 3), ("finger-turn-hard", 4), ("quadruped-walk", 3), ("quadruped-walk", 4),
         ("walker-walk", 3), ("walker-walk", 4), ("cheetah-run", 4)}
QAT_CELLS = {("mw-button-press-topdown", 3), ("mw-door-open", 4), ("mw-drawer-open", 3), ("mw-peg-insert-side", 4),
             ("mw-pick-place", 3), ("mw-push", 3), ("mw-sweep-into", 3), ("mw-window-open", 4),
             ("finger-turn-hard", 4), ("quadruped-walk", 3), ("walker-walk", 3)}
SWEEP = {"mw-push", "mw-pick-place", "cheetah-run", "walker-walk"}
PTQ = ["perchan", "mseclip", "smooth", "mixed"]
WV = ["w8rtn", "w4rtn", "w4awq", "w4sr1", "w4sr2", "w4sr4"]
ALPHA_ENDS = (0.10, 1.1)                       # v5.1 8.1 grid {0.10, 0.15, ..., 0.45} U {0.5, ..., 1.1}
SEED0 = {"held": 12000, "e5": 60000, "e5qat": 65000}
NEPS = {"held": 20, "e5": 30, "e5qat": 20}
MANIFESTS = ["e5_stats", "e5_held", "e5_qatdata", "e5_qattrain", "e5_eval", "e5_w", "e5_qateval"]


def ptag_of(job_id):
    return job_id.split("_")[-2]


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--mdir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.res = os.path.abspath(a.res)          # result files record absolute paths (qat_path, path)
    J = {m: [json.loads(l) for l in open(f"{a.mdir}/{m}.jsonl") if l.strip()] for m in MANIFESTS}
    rep = {"manifest_jobs": {m: len(v) for m, v in J.items()}}
    # the combined manifests the queue ran must be the plain concatenations
    for comb, parts in (("e5_held_qatdata", ["e5_held", "e5_qatdata"]), ("e5_eval_all", ["e5_eval", "e5_w", "e5_qateval"])):
        p = f"{a.mdir}/{comb}.jsonl"
        if os.path.exists(p):
            cj = [json.loads(l) for l in open(p) if l.strip()]
            rep[f"{comb}_equals_concatenation"] = cj == [j for m in parts for j in J[m]]
    bad = defaultdict(list)
    env = defaultdict(Counter)
    eps_cond = defaultdict(set)
    t_eval_min = math.inf
    held = defaultdict(dict)                    # (task, k, B) -> {(ckpt, ep): metric}
    sel = json.load(open(f"{a.res}/e5/selection.json"))
    sel_mtime = os.path.getmtime(f"{a.res}/e5/selection.json")
    ids = set()
    for m, jobs in J.items():
        for j in jobs:
            ids.add(j["id"])
            sub = {"episodes": "episodes", "e5stats": "e5stats", "qat_train": "qat_train"}[j["type"]]
            p = f"{a.res}/jobs/{sub}/{j['id']}.json.gz"
            if not os.path.exists(p):
                bad["missing_result"].append(j["id"])
                continue
            d = json.load(gzip.open(p, "rt"))
            if not d.get("ok"):
                bad["not_ok"].append(j["id"])
            if d.get("spec") != j:
                bad["spec_differs"].append(j["id"])
            e = d.get("env", {})
            env[m][(e.get("host"), e.get("gpu"), e.get("cvd"), e.get("mps"), e.get("torch"))] += 1
            if j["type"] == "e5stats":
                for k, lay in d.get("mse_alpha", {}).items():
                    for n, al in lay.items():
                        if any(abs(al - x) < 1e-9 for x in ALPHA_ENDS):
                            bad["mse_alpha_at_grid_end"].append([j["task"], j["ckpt"], k, n, al])
                        rep.setdefault("mse_alpha_range", {}).setdefault(k, [math.inf, -math.inf])
                        r = rep["mse_alpha_range"][k]
                        r[0], r[1] = min(r[0], al), max(r[1], al)
                for n, c in d.get("check", {}).items():
                    if abs(c["lo_minus_e0"]) > 0 or abs(c["hi_minus_e0"]) > 0:
                        bad["stats_range_differs_from_E0"].append([j["task"], j["ckpt"], n])
                continue
            if j["type"] == "qat_train":
                if (j["task"], int(j["k"])) not in QAT_CELLS:
                    bad["qat_train_cell"].append(j["id"])
                if (j["steps"], j["batch"], j["lr"], j["grad_clip"]) != (5000, 256, 1e-4, 20.0):
                    bad["qat_train_spec"].append(j["id"])
                qp = f"{a.res}/qat/{j['task']}_s{j['ckpt']}_k{j['k']}.pt"
                if d.get("path") != qp or not os.path.exists(qp):
                    bad["qat_weights_file"].append(j["id"])
                h = d.get("hist") or []
                if not h or h[-1].get("step") != 4999 or not all(finite(x.get("loss")) for x in h):
                    bad["qat_hist"].append(j["id"])
                continue
            # ---- episode jobs
            t, k, v, c, split, pt = j["task"], int(j["k"]), j["variant"], int(j["ckpt"]), j["split"], ptag_of(j["id"])
            mw = t.startswith("mw-")
            if int(j["seed0"]) != SEED0.get(split, -1):
                bad["seed0"].append(j["id"])
            if j.get("planner") or j.get("instr") or j.get("warm_start", True) is not True:
                bad["planner_or_instr"].append(j["id"])
            if m == "e5_held":
                ok = (v == "none" and k == 0 and pt == "e5none") or \
                     (v == "det" and (t, k) in CELLS and j.get("ptq") in PTQ and pt == f"e5{j.get('ptq')}")
                if not ok or d.get("ptq_resolved", j.get("ptq")) != j.get("ptq"):
                    bad["held_job"].append(j["id"])
                if j.get("ptq") == "mixed" and d.get("layer_k") != {"dyn1": 8, "rew1": 8}:
                    bad["mixed_layer_k"].append(j["id"])
                if j.get("ptq") in ("perchan", "mseclip", "smooth") and d.get("layer_k") not in ({}, None):
                    bad["layer_k_unexpected"].append(j["id"])
                B = "none" if v == "none" else j["ptq"]
                for ep in d["episodes"]:
                    held[(t, k, B)][(c, int(ep["ep"]))] = float(ep["success"] if mw else ep["ret"])
            elif m == "e5_eval":
                ok = (v == "none" and k == 0 and pt == "base") or \
                     (v in ("det", "shiter") and (t, k) in CELLS and j.get("ptq") == "best" and pt == "e5best")
                if not ok:
                    bad["eval_job"].append(j["id"])
                if v != "none":
                    want = sel["cells"].get(f"{t}|{k}", {}).get("best")
                    if d.get("ptq_resolved") != want:
                        bad["ptq_resolved_differs_from_selection"].append(j["id"])
                    if want == "mixed" and d.get("layer_k") != {"dyn1": 8, "rew1": 8}:
                        bad["mixed_layer_k"].append(j["id"])
                    t_eval_min = min(t_eval_min, d["t_start"])
            elif m == "e5_w":
                if not (v == "none" and k == 0 and t in SWEEP and j.get("wq") in WV and pt == f"e5{j.get('wq')}"):
                    bad["weight_job"].append(j["id"])
            elif m == "e5_qateval":
                ok = v in ("det", "shiter") and (t, k) in QAT_CELLS and j.get("qat") is True and pt == "e5qat"
                qp = f"{a.res}/qat/{t}_s{c}_k{k}.pt"
                if not ok or d.get("qat_path") != qp:
                    bad["qat_eval_job"].append(j["id"])
            elif m == "e5_qatdata":
                if not (v == "none" and k == 0 and j.get("save_traj") and pt == "qatdata" and
                        (t, ) in {(x[0], ) for x in QAT_CELLS}):
                    bad["qatdata_job"].append(j["id"])
                for ep in d["episodes"]:
                    if not os.path.exists(f"{a.res}/qatdata/{t}_s{c}_ep{int(ep['ep'])}.npz"):
                        bad["qatdata_npz_missing"].append([j["id"], ep["ep"]])
            got = [int(x["ep"]) for x in d["episodes"]]
            if got != [int(x) for x in j["eps"]]:
                bad["episode_ids"].append(j["id"])
            for ep in d["episodes"]:
                if int(ep["seed"]) != int(j["seed0"]) + int(ep["ep"]):
                    bad["seed"].append([j["id"], ep["ep"]])
                if not finite(ep.get("ret")) or not finite(ep.get("success")):
                    bad["nonfinite"].append([j["id"], ep["ep"]])
                eps_cond[(m, t, c, k, v, j.get("ptq") or j.get("wq") or ("qat" if j.get("qat") else ""))].add(int(ep["ep"]))
    extra = []
    for sub in ("episodes", "e5stats", "qat_train"):
        for f in glob.glob(f"{a.res}/jobs/{sub}/*.json.gz"):
            if os.path.basename(f)[:-len(".json.gz")] not in ids:
                extra.append(os.path.basename(f))
    nep = {"e5_held": 20, "e5_qatdata": 20, "e5_eval": 30, "e5_w": 30, "e5_qateval": 30}
    incomplete = [list(k) + [len(s)] for k, s in eps_cond.items() if s != set(range(nep[k[0]]))]
    # cell coverage against the design lists
    cov = {"eval_cells": sorted({(k[1], k[3]) for k in eps_cond if k[0] == "e5_eval" and k[4] != "none"}),
           "held_cells": sorted({(k[1], k[3]) for k in eps_cond if k[0] == "e5_held" and k[4] != "none"}),
           "qat_cells": sorted({(k[1], k[3]) for k in eps_cond if k[0] == "e5_qateval"}),
           "weight_tasks": sorted({k[1] for k in eps_cond if k[0] == "e5_w"})}
    rep["cells_match_design"] = {
        "eval_19": set(map(tuple, cov["eval_cells"])) == CELLS, "held_19": set(map(tuple, cov["held_cells"])) == CELLS,
        "qat_11": set(map(tuple, cov["qat_cells"])) == QAT_CELLS, "weights_4": set(cov["weight_tasks"]) == SWEEP,
        "n_mw_dmc": [sum(1 for t, _ in CELLS if t.startswith("mw-")), sum(1 for t, _ in CELLS if not t.startswith("mw-"))]}
    # independent recomputation of the selection (v5 7.3)
    rec, sel_bad = {}, []
    for t, k in sorted(CELLS):
        mw = t.startswith("mw-")
        none = held[(t, 0, "none")]
        prs = sorted(none)
        r0 = sum(none[x] for x in prs) / len(prs)
        D = {}
        for b in PTQ:
            vb = held[(t, k, b)]
            if sorted(vb) != prs:
                sel_bad.append([t, k, b, "pairs"])
                continue
            rv = sum(vb[x] for x in prs) / len(prs)
            D[b] = (r0 - rv) if mw else (r0 - rv) / r0
        best = min(PTQ, key=lambda b: (D.get(b, math.inf), PTQ.index(b)))
        s = sel["cells"].get(f"{t}|{k}", {})
        same_D = all(abs(D[b] - s.get("D_det", {}).get(b, math.nan)) < 1e-12 for b in PTQ if b in D)
        if best != s.get("best") or not same_D or s.get("n_pairs") != len(prs):
            sel_bad.append([t, k, best, s.get("best"), same_D])
        ties = sorted(b for b in PTQ if b in D and abs(D[b] - D[best]) < 1e-12)
        rec[f"{t}|{k}"] = {"best": best, "n_pairs": len(prs), "ties_at_min": ties}
    rep["selection_recomputed_equals_file"] = not sel_bad
    rep["selection_mismatches"] = sel_bad
    rep["selection_ties_at_min"] = {c: r["ties_at_min"] for c, r in rec.items() if len(r["ties_at_min"]) > 1}
    rep["selection_best"] = {c: r["best"] for c, r in rec.items()}
    rep["selection_written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(sel_mtime))
    rep["first_eval_job_start_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_eval_min)) \
        if math.isfinite(t_eval_min) else None
    rep["eval_started_after_selection"] = bool(t_eval_min > sel_mtime)
    rep.update({
        "result_files_extra": extra, "conditions": len(eps_cond), "conditions_incomplete": incomplete,
        "env_by_manifest": {m: [list(k) + [n] for k, n in c.items()] for m, c in env.items()},
        "errors_dir_files": len(glob.glob(f"{a.res}/errors/*")), "lock_files": len(glob.glob(f"{a.res}/locks/*")),
        "problems": {k: (len(v), v[:5]) for k, v in bad.items()}})
    rep["ALL_OK"] = bool(not bad and not extra and not incomplete and not sel_bad and rep["eval_started_after_selection"]
                         and all(v for kk, v in rep["cells_match_design"].items() if kk != "n_mw_dmc")
                         and rep["errors_dir_files"] == 0 and rep["lock_files"] == 0
                         and all(rep.get(f"{c}_equals_concatenation", True) for c in ("e5_held_qatdata", "e5_eval_all")))
    s = json.dumps(rep, indent=1, default=str)
    if a.out:
        open(a.out, "w").write(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
