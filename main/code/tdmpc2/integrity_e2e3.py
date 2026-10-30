"""E2 / E3 TD-MPC2 integrity check (no outcome values are printed): every manifest job has exactly one result whose spec
equals the manifest entry; episode ids and seeds; settings, variants, k and instrumentation against the DESIGN_main
v5 7.1 / v5.1 8.2 / v5 7.2 tables (written from the design text, not from make_manifest); GPU / CVD / host / MPS pipe
of every job; error and lock files; missing or non-finite OEI fields; OEI-step counts; mechanical identities of the
independent re-evaluation (indep_same = 1 for det and the persistent-table variants, 0 for the fresh variants;
fresh_same = 0); job time span.
usage (server, CPU only): CUDA_VISIBLE_DEVICES="" python integrity_e2e3.py --exp e2|e3 --manifest <jsonl> --res <out dir>
"""
import argparse
import glob
import gzip
import json
import math
import os
from collections import Counter, defaultdict

DEFAULT = {"num_samples": 512, "num_elites": 64, "num_pi_trajs": 24, "temperature": 0.5, "iterations": 6}
# DESIGN_main v5 7.1 table + v5.1 8.2 fixed-count sweep: effective planner and warm start per setting
E2_SET = {"base": ({}, True)}
for _n in (64, 128, 256, 1024):
    E2_SET[f"N{_n}k"] = ({"num_samples": _n}, True)
E2_SET.update({
    "N64": ({"num_samples": 64, "num_elites": 8, "num_pi_trajs": 3}, True),
    "N128": ({"num_samples": 128, "num_elites": 16, "num_pi_trajs": 6}, True),
    "N256": ({"num_samples": 256, "num_elites": 32, "num_pi_trajs": 12}, True),
    "N1024": ({"num_samples": 1024, "num_elites": 128, "num_pi_trajs": 48}, True),
    "el2": ({"num_elites": 10}, True), "el25": ({"num_elites": 128}, True),
    "T01": ({"temperature": 0.1}, True), "T2": ({"temperature": 2.0}, True),
    "it1": ({"iterations": 1}, True), "it3": ({"iterations": 3}, True), "it12": ({"iterations": 12}, True),
    "nows1": ({"iterations": 1}, False), "nows3": ({"iterations": 3}, False), "nows6": ({"iterations": 6}, False),
    "nows12": ({"iterations": 12}, False)})
E2_TASKS = {"mw-push": 3, "mw-pick-place": 3, "cheetah-run": 5}
E2_V_BASE = {"none", "det", "shiter", "shstep", "ind", "shep", "detshift"}
E2_V = {"none", "det", "shiter", "shstep"}
# DESIGN_main v1 0.2 primary k; v5 7.2 2-axis cells and table-size sweep
PRIMARY = {"mw-button-press-topdown": 3, "mw-door-open": 4, "mw-drawer-open": 3, "mw-peg-insert-side": 4,
           "mw-pick-place": 3, "mw-push": 3, "mw-sweep-into": 3, "mw-window-open": 4, "cheetah-run": 5,
           "finger-turn-hard": 4, "quadruped-walk": 3, "walker-walk": 3}
E3_V = {"none", "det", "ind", "shiter", "shstep", "shep", "indstep", "indep"}
E3_SWEEP = {"mw-push", "mw-pick-place", "cheetah-run", "walker-walk"}
E3_V_SWEEP = {"shstep_l", "shstep_g32", "shstep_u", "shiter_l", "shiter_g32"}
SEED0 = {"e2": 40000, "e3": 50000}
PERSIST = {"det", "shstep", "shep", "indstep", "indep", "shstep_u", "shstep_g32", "shstep_l"}
TABLES = {"shstep", "shep", "indstep", "indep", "shstep_u", "shstep_g32", "shstep_l"}
REQ = ["oei", "oei_indep", "sel_last", "sel_indep", "drift", "e_chosen", "e_indep", "f_e_rand", "l_e_rand",
       "f_var_cand", "chosen_pi", "indep_same"]
REQ_BASE_E2 = ["drift_ref", "drift_minus_ref", "ref_f_e_rand", "ref_l_e_rand"]


def finite(x):
    return isinstance(x, (int, float)) and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def ptag_of(job_id):
    return job_id.split("_")[-2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["e2", "e3"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    jobs = [json.loads(l) for l in open(a.manifest) if l.strip()]
    rep = {"exp": a.exp, "manifest_jobs": len(jobs), "by_type": dict(Counter(j["type"] for j in jobs))}
    bad = defaultdict(list)
    env_c = Counter()
    span = [math.inf, -math.inf]
    secs_total = 0.0
    eps_per_cond = defaultdict(set)
    n_ep = 0
    nan_by_key, miss_by_key = Counter(), Counter()
    indep_same, fresh_same = defaultdict(list), defaultdict(list)
    n_oei_steps = defaultdict(Counter)
    perr_missing = 0
    variants_by_task_set = defaultdict(set)
    ids = set()
    for j in jobs:
        ids.add(j["id"])
        sub = "episodes" if j["type"] == "episodes" else "decision"
        p = f"{a.res}/jobs/{sub}/{j['id']}.json.gz"
        if not os.path.exists(p):
            bad["missing_result"].append(j["id"])
            continue
        d = json.load(gzip.open(p, "rt"))
        if not d.get("ok", False):
            bad["not_ok"].append(j["id"])
        if d.get("spec") != j:
            bad["spec_differs_from_manifest"].append(j["id"])
        env = d.get("env", {})
        env_c[(env.get("host"), env.get("gpu"), env.get("cvd"), env.get("mps"), env.get("torch"))] += 1
        if "t_start" in d and "secs" in d:
            span[0] = min(span[0], d["t_start"])
            span[1] = max(span[1], d["t_start"] + d["secs"])
            secs_total += d["secs"]
        if j["type"] == "decision":
            if j["task"] not in PRIMARY or int(j["k"]) != PRIMARY[j["task"]]:
                bad["decision_task_or_k"].append(j["id"])
            exp_v = {"indstep", "indep"} | (E3_V_SWEEP if j["task"] in E3_SWEEP else set())
            if set(j["variants"]) != exp_v:
                bad["decision_variants"].append(j["id"])
            if int(d.get("n_states", -1)) != 150:
                bad["decision_n_states"].append([j["id"], d.get("n_states")])
            if not exp_v <= set(d.get("summary", {})):
                bad["decision_summary_variants"].append(j["id"])
            continue
        # ---- episode job: split, seeds, k, setting, variants, instrumentation
        t, v, pt = j["task"], j["variant"], ptag_of(j["id"])
        mw = t.startswith("mw-")
        if j["split"] != a.exp or int(j["seed0"]) != SEED0[a.exp]:
            bad["split_or_seed0"].append(j["id"])
        ins = j.get("instr", {})
        if int(ins.get("oei_stride", -1)) != (4 if mw else 10) or int(ins.get("pd_stride", -1)) != (20 if mw else 50) \
                or not ins.get("indep_reeval"):
            bad["instr"].append(j["id"])
        if a.exp == "e2":
            if t not in E2_TASKS or pt not in E2_SET:
                bad["task_or_setting"].append(j["id"])
                continue
            if int(j["k"]) != (0 if v == "none" else E2_TASKS[t]):
                bad["k"].append(j["id"])
            ov, warm = E2_SET[pt]
            eff = dict(DEFAULT, **{kk: type(DEFAULT[kk])(x) for kk, x in j.get("planner", {}).items()})
            if eff != dict(DEFAULT, **ov) or bool(j.get("warm_start", True)) != warm:
                bad["planner_or_warm_start"].append(j["id"])
            if bool(ins.get("region_ref", False)) != (pt == "base") or bool(ins.get("perr", False)):
                bad["instr_e2"].append(j["id"])
            variants_by_task_set[(t, pt)].add(v)
        else:
            if t not in PRIMARY or pt != "base":
                bad["task_or_setting"].append(j["id"])
                continue
            if int(j["k"]) != (0 if v == "none" else PRIMARY[t]):
                bad["k"].append(j["id"])
            if j.get("planner") or j.get("warm_start", True) is not True:
                bad["planner_or_warm_start"].append(j["id"])
            if not ins.get("perr") or int(ins.get("perr_seeds", -1)) != 3 or ins.get("region_ref"):
                bad["instr_e3"].append(j["id"])
            variants_by_task_set[(t, pt)].add(v)
        got = [int(e["ep"]) for e in d["episodes"]]
        if got != [int(x) for x in j["eps"]]:
            bad["episode_ids"].append(j["id"])
        for e in d["episodes"]:
            n_ep += 1
            eps_per_cond[(t, int(j["ckpt"]), pt, v)].add(int(e["ep"]))
            if int(e["seed"]) != SEED0[a.exp] + int(e["ep"]):
                bad["seed"].append([j["id"], e["ep"]])
            if not finite(e.get("ret")) or not finite(e.get("success")):
                bad["ret_or_success_nonfinite"].append([j["id"], e["ep"]])
            om = e.get("oei_mean") or {}
            st = e.get("oei_steps") or {}
            n_oei_steps[(t, pt)][len(st.get("oei", []))] += 1
            if v == "none":
                req = REQ_BASE_E2 if (a.exp == "e2" and pt == "base") else []
            else:
                req = REQ + (["oei_fresh", "sel_fresh", "fresh_same"] if v in TABLES else []) + \
                    (REQ_BASE_E2 if (a.exp == "e2" and pt == "base") else [])
            for kk in req:
                if kk not in om:
                    miss_by_key[(v, kk)] += 1
                elif not finite(om[kk]):
                    nan_by_key[(v, kk)] += 1
            for kk in ("oei", "e_chosen", "oei_indep"):
                if kk in st and any(x is None or (isinstance(x, float) and math.isnan(x)) for x in st[kk]):
                    nan_by_key[(v, "steps_" + kk)] += 1
            if v != "none" and "indep_same" in om:
                indep_same[v].append(om["indep_same"])
            if v in TABLES and "fresh_same" in om:
                fresh_same[v].append(om["fresh_same"])
            if a.exp == "e3" and v != "none":
                for src in ("perr_plan", "perr_traj"):
                    pe = e.get(src)
                    if not pe or any(not finite(pe.get(kk)) for kk in ("err1", "err1h", "err3", "err5")):
                        perr_missing += 1
    # extra result files (not in the manifest)
    extra = []
    for sub in ("episodes", "decision"):
        for f in glob.glob(f"{a.res}/jobs/{sub}/*.json.gz"):
            if os.path.basename(f)[:-len(".json.gz")] not in ids:
                extra.append(os.path.basename(f))
    # condition completeness: 30 episodes (0..29) per (task, ckpt, setting, variant); expected variant sets
    incomplete = [list(k) + [len(s)] for k, s in eps_per_cond.items() if s != set(range(30))]
    vs_bad = []
    for (t, pt), vs in variants_by_task_set.items():
        if a.exp == "e2":
            exp_v = E2_V_BASE if pt == "base" else E2_V
        else:
            exp_v = E3_V | (E3_V_SWEEP if t in E3_SWEEP else set())
        if vs != exp_v:
            vs_bad.append([t, pt, sorted(vs)])
    n_sets = len(variants_by_task_set)
    exp_sets = len(E2_TASKS) * len(E2_SET) if a.exp == "e2" else len(PRIMARY)
    rep.update({
        "result_files_extra": extra, "episode_records": n_ep, "conditions": len(eps_per_cond),
        "conditions_not_30_eps": incomplete, "task_setting_groups": n_sets, "task_setting_groups_expected": exp_sets,
        "variant_set_mismatch": vs_bad,
        "env_host_gpu_cvd_mps_torch": [list(k) + [n] for k, n in env_c.items()],
        "errors_dir_files": len(glob.glob(f"{a.res}/errors/*")) + len(glob.glob(f"{a.res}/jobs/errors/*")),
        "lock_files": len(glob.glob(f"{a.res}/locks/*")),
        "missing_fields": {f"{v}:{kk}": n for (v, kk), n in sorted(miss_by_key.items())},
        "nonfinite_fields": {f"{v}:{kk}": n for (v, kk), n in sorted(nan_by_key.items())},
        "indep_same_mean_by_variant": {v: sum(x) / len(x) for v, x in sorted(indep_same.items())},
        "fresh_same_mean_by_variant": {v: sum(x) / len(x) for v, x in sorted(fresh_same.items())},
        "oei_steps_per_episode": {f"{t}|{pt}": dict(c) for (t, pt), c in sorted(n_oei_steps.items())},
        "perr_missing_episodes": perr_missing if a.exp == "e3" else None,
        "time_span_utc": [None if not math.isfinite(x) else
                          __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime(x))
                          for x in span],
        "job_secs_total": secs_total,
        "problems": {k: (len(v), v[:5]) for k, v in bad.items()}})
    # mechanical expectations of the independent re-evaluation
    ok_is = all(abs(m - (1.0 if v in PERSIST else 0.0)) < 1e-12 for v, m in rep["indep_same_mean_by_variant"].items())
    ok_fs = all(m == 0.0 for m in rep["fresh_same_mean_by_variant"].values())
    rep["indep_reeval_identities_ok"] = bool(ok_is and ok_fs)
    rep["ALL_OK"] = bool(not bad and not extra and not incomplete and not vs_bad and n_sets == exp_sets
                         and len(env_c) == 1 and rep["errors_dir_files"] == 0 and rep["lock_files"] == 0
                         and not miss_by_key and not nan_by_key and rep["indep_reeval_identities_ok"]
                         and (a.exp != "e3" or perr_missing == 0))
    s = json.dumps(rep, indent=1, default=str)
    if a.out:
        open(a.out, "w").write(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
