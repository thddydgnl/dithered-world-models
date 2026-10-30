"""Spot-check of the v5 manifests against DESIGN_main v5 section 7 (counts, seeds, settings, instrumentation) and the
GPU-hour estimate at 90 pairs. CPU only.
usage (server): python check_v5_manifests.py [--m ../v5/manifests] [--e1 ../analysis_e1/e1_tdmpc2_summary.json]
                    [--timing ../v5/timing_v5.json] [--out ../v5/manifest_check.json]
"""
import argparse
import json
from collections import Counter

PRIMARY = {"mw-button-press-topdown": 3, "mw-door-open": 4, "mw-drawer-open": 3, "mw-peg-insert-side": 4,
           "mw-pick-place": 3, "mw-push": 3, "mw-sweep-into": 3, "mw-window-open": 4, "cheetah-run": 5,
           "finger-turn-hard": 4, "quadruped-walk": 3, "walker-walk": 3}
E2_SET = {  # v5 7.1 table: (num_samples, num_elites, num_pi_trajs, temperature, iterations, warm start)
    "base": (512, 64, 24, 0.5, 6, True), "N64k": (64, 64, 24, 0.5, 6, True), "N128k": (128, 64, 24, 0.5, 6, True),
    "N256k": (256, 64, 24, 0.5, 6, True), "N1024k": (1024, 64, 24, 0.5, 6, True),
    "N64": (64, 8, 3, 0.5, 6, True), "N128": (128, 16, 6, 0.5, 6, True),
    "N256": (256, 32, 12, 0.5, 6, True), "N1024": (1024, 128, 48, 0.5, 6, True), "el2": (512, 10, 24, 0.5, 6, True),
    "el25": (512, 128, 24, 0.5, 6, True), "T01": (512, 64, 24, 0.1, 6, True), "T2": (512, 64, 24, 2.0, 6, True),
    "it1": (512, 64, 24, 0.5, 1, True), "it3": (512, 64, 24, 0.5, 3, True), "it12": (512, 64, 24, 0.5, 12, True),
    "nows1": (512, 64, 24, 0.5, 1, False), "nows3": (512, 64, 24, 0.5, 3, False), "nows6": (512, 64, 24, 0.5, 6, False),
    "nows12": (512, 64, 24, 0.5, 12, False)}
RATE = {"mps16": {"mw": 3716, "dmc": 860}, "mps16_noinstr": {"mw": 4800, "dmc": 929},
        "nomps16": {"mw": 2095, "dmc": 488}}


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def fam(t):
    return "mw" if t.startswith("mw-") else "dmc"


def ptag(j):
    return j["id"].split("_")[-2]


def eps_of(jobs):
    c = Counter()
    for j in jobs:
        if j["type"] == "episodes":
            c[fam(j["task"])] += len(j["eps"])
    return c


def pairs_ok(jobs, n_eps):
    """every (cell, variant, setting, ckpt) has episodes 0..n_eps-1 exactly once"""
    got = {}
    for j in jobs:
        if j["type"] != "episodes":
            continue
        key = (j["task"], j["k"], j["variant"], ptag(j), j["ckpt"], json.dumps(j.get("planner", {}), sort_keys=True))
        got.setdefault(key, []).extend(j["eps"])
    return all(sorted(v) == list(range(n_eps)) for v in got.values()), len(got)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="../v5/manifests")
    ap.add_argument("--e1", default="../analysis_e1/e1_tdmpc2_summary.json")
    ap.add_argument("--timing", default="../v5/timing_v5.json")
    ap.add_argument("--out", default="../v5/manifest_check.json")
    a = ap.parse_args()
    R, errs = {}, []

    def check(cond, msg):
        if not cond:
            errs.append(msg)

    # ---- E2
    e2 = load(f"{a.m}/e2.jsonl")
    check({ptag(j) for j in e2} == set(E2_SET), "E2 settings")
    for j in e2:
        n, el, pi, T, it, ws = E2_SET[ptag(j)]
        pl = {"num_samples": 512, "num_elites": 64, "num_pi_trajs": 24, "temperature": 0.5, "iterations": 6}
        pl.update(j["planner"])
        check((pl["num_samples"], pl["num_elites"], pl["num_pi_trajs"], pl["temperature"], pl["iterations"],
               j.get("warm_start", True)) == (n, el, pi, T, it, ws), f"E2 planner {j['id']}")
        check(j["seed0"] == 40000 and j["split"] == "e2", f"E2 seed {j['id']}")
        exp_instr = {"oei_stride": 4 if fam(j["task"]) == "mw" else 10, "pd_stride": 20 if fam(j["task"]) == "mw" else 50,
                     "perr": False, "perr_seeds": 3, "perr_plan_states": 8, "indep_reeval": True}
        if ptag(j) == "base":
            exp_instr["region_ref"] = True       # v5.2 (b3), E2 base only
        check(j["instr"] == exp_instr, f"E2 instr {j['id']}")
        vs = {"none", "det", "shiter", "shstep"} | ({"ind", "shep", "detshift"} if ptag(j) == "base" else set())
        check(j["variant"] in vs, f"E2 variant {j['id']}")
        check((j["task"], j["k"] if j["variant"] != "none" else None) in
              {("mw-push", 3), ("mw-pick-place", 3), ("cheetah-run", 5), ("mw-push", None), ("mw-pick-place", None),
               ("cheetah-run", None)}, f"E2 cell {j['id']}")
    ok, ncell = pairs_ok(e2, 30)
    check(ok, "E2 pairs")
    R["e2"] = {"jobs": len(e2), "episodes": dict(eps_of(e2)), "cells_x_variants_x_ckpt": ncell}
    check(sum(eps_of(e2).values()) == 3 * 90 * (7 + 19 * 4), "E2 episode total")
    # ---- E3
    e3 = load(f"{a.m}/e3.jsonl")
    sweep = {"mw-push", "mw-pick-place", "cheetah-run", "walker-walk"}
    for j in e3:
        if j["type"] == "decision":
            check(j["k"] == PRIMARY[j["task"]], f"E3 decision k {j['id']}")
            continue
        check(j["seed0"] == 50000 and j["instr"].get("indep_reeval") is True and j["instr"].get("perr") is True,
              f"E3 seed/instr {j['id']}")
        check(j["variant"] == "none" or j["k"] == PRIMARY[j["task"]], f"E3 k {j['id']}")
        base_v = {"none", "det", "ind", "shiter", "shstep", "shep", "indstep", "indep"}
        sw_v = {"shstep_l", "shstep_g32", "shstep_u", "shiter_l", "shiter_g32"}
        check(j["variant"] in base_v or (j["task"] in sweep and j["variant"] in sw_v), f"E3 variant {j['id']}")
    ok, ncell = pairs_ok(e3, 30)
    check(ok, "E3 pairs")
    R["e3"] = {"jobs": len(e3), "types": dict(Counter(j["type"] for j in e3)), "episodes": dict(eps_of(e3))}
    check(sum(eps_of(e3).values()) == 90 * (12 * 8 + 4 * 5), "E3 episode total")
    # ---- E5
    e1 = json.load(open(a.e1))
    q34 = sorted((c["task"], c["k"]) for c in e1["cells"].values() if c.get("det_drop_real_holm") and c["k"] in (3, 4))
    held = load(f"{a.m}/e5_held.jsonl")
    cells = sorted({(j["task"], j["k"]) for j in held if j["variant"] != "none"})
    check(cells == q34, f"E5 cells {cells} vs E1 {q34}")
    check(len(cells) == 19, "E5 19 cells")
    check(all(j["seed0"] == 12000 and j["split"] == "held" and j["instr"] == {} for j in held), "E5 held seeds")
    check(Counter(j.get("ptq", "none") for j in held for _ in j["eps"]) ==
          Counter({"perchan": 1140, "mseclip": 1140, "smooth": 1140, "mixed": 1140, "none": 720}), "E5 held counts")
    ok, _ = pairs_ok(held, 20)
    check(ok, "E5 held pairs")
    ev = load(f"{a.m}/e5_eval.jsonl")
    check(all(j["seed0"] == 60000 for j in ev) and pairs_ok(ev, 30)[0], "E5 eval seeds/pairs")
    check(sum(len(j["eps"]) for j in ev) == 19 * 2 * 90 + 12 * 90, "E5 eval total")
    w = load(f"{a.m}/e5_w.jsonl")
    check(sum(len(j["eps"]) for j in w) == 4 * 6 * 90 and pairs_ok(w, 30)[0], "E5 weight total")
    qd = load(f"{a.m}/e5_qatdata.jsonl")
    check(all(j["seed0"] == 65000 and j.get("save_traj") for j in qd) and sum(len(j["eps"]) for j in qd) == 11 * 3 * 20,
          "E5 QAT data")
    qt = load(f"{a.m}/e5_qattrain.jsonl")
    check(len(qt) == 33 and all((j["steps"], j["batch"], j["lr"], j["grad_clip"]) == (5000, 256, 1e-4, 20.0)
                                for j in qt), "E5 QAT train")
    check({(j["task"], j["k"]) for j in qt} <= set(q34) and
          all(j["k"] == PRIMARY[j["task"]] for j in qt), "E5 QAT cells = qualifying primary-k cells")
    qe = load(f"{a.m}/e5_qateval.jsonl")
    check(sum(len(j["eps"]) for j in qe) == 11 * 2 * 90 and pairs_ok(qe, 30)[0], "E5 QAT eval")
    R["e5"] = {"cells": len(cells), "held": dict(eps_of(held)), "eval": dict(eps_of(ev)), "weights": dict(eps_of(w)),
               "qat_data": dict(eps_of(qd)), "qat_train_jobs": len(qt), "qat_eval": dict(eps_of(qe)),
               "jobs": {"stats": len(load(f"{a.m}/e5_stats.jsonl")), "held": len(held), "eval": len(ev), "w": len(w),
                        "qatdata": len(qd), "qattrain": len(qt), "qateval": len(qe)}}
    # ---- E7
    e7 = load(f"{a.m}/e7.jsonl")
    check(all(j["seed0"] == 70000 for j in e7) and pairs_ok(e7, 30)[0], "E7 seeds/pairs")
    check(Counter((j["e7"], j["variant"], j["planner"]["iterations"]) for j in e7 for _ in j["eps"]) ==
          Counter({(p, v, it): 4 * 90 for p, v in (("fp32", "none"), ("bf16", "none"), ("int8", "det"),
                                                    ("int8", "shiter"), ("int4", "det"), ("int4", "shiter"))
                   for it in (3, 6, 12)}), "E7 grid")
    R["e7"] = {"jobs": len(e7), "episodes": dict(eps_of(e7))}
    h2a = load(f"{a.m}/h2a_indep_min.jsonl")
    R["h2a_fresh3"] = {"jobs": len(h2a), "episodes": dict(eps_of(h2a)),
                       "variants": sorted({j["variant"] for j in h2a})}
    # ---- GPU hours at 90 pairs
    T = json.load(open(a.timing))
    fac = {}
    for r in T:
        if not r["instr"]:
            fac[(fam(r["task"]), r["setting"])] = r["ms_median"]
    for f_ in ("mw", "dmc"):
        b = fac[(f_, "base")]
        for kk in list(fac):
            if kk[0] == f_:
                fac[kk] /= b

    def hours(jobs, rate, factor_of=None):
        h = Counter()
        for j in jobs:
            if j["type"] != "episodes":
                continue
            f_ = fam(j["task"])
            x = factor_of(j) if factor_of else 1.0
            h[f_] += len(j["eps"]) * x / RATE[rate][f_]
        return round(h["mw"] + h["dmc"], 2)

    # fixed-count N settings (v5.1) were not timed: the ratio-fixed factor of the same N is used (the elite count
    # barely changes the step cost)
    # E2 base with the v5.2 (b3) region reference: + 1 unquantized plan (6 evaluations) + 2 evaluations per OEI step
    # (every 4th MW / 10th DMC step) -> x (1 + 8/6/4) MW, x (1 + 8/6/10) DMC on the planning time (upper bound)
    reg = {"mw": 1 + 8 / 6 / 4, "dmc": 1 + 8 / 6 / 10}
    e2f = lambda j: (fac.get((fam(j["task"]), ptag(j).rstrip("k") if ptag(j).endswith("k") else ptag(j)), 1.0) *
                     (reg[fam(j["task"])] if ptag(j) == "base" else 1.0))
    itf = lambda j: fac.get((fam(j["task"]), {3: "it3", 6: "base", 12: "it12"}[j["planner"]["iterations"]]), 1.0)
    R["gpu_hours"] = {
        "h2a_fresh3": {r: hours(h2a, r) for r in ("mps16", "nomps16")},
        "e2": {"mps16": hours(e2, "mps16", e2f), "mps16_noinstr": hours(e2, "mps16_noinstr", e2f),
               "nomps16": hours(e2, "nomps16", e2f)},
        "e3": {r: hours(e3, r) for r in ("mps16", "mps16_noinstr", "nomps16")},
        "e5_episodes": {r: hours(held + ev + w + qd + qe, r) for r in ("mps16_noinstr", "nomps16")},
        "e5_qat_train_single_process": round(33 * 3 / 60, 2),
        "e7_success": {r: hours(e7, r, itf) for r in ("mps16_noinstr", "nomps16")},
        "basis": "busy-time rates (eps/h) MPS16 E1-instr MW 3716 / DMC 860, uninstrumented 4800 / 929, no MPS 16 "
                 "workers (E1-instr) 2095 / 488; per-setting cost factors from timing_v5 (single process); E5 / E7 "
                 "jobs are uninstrumented; QAT training about 3 min per run",
    }
    gh = R["gpu_hours"]
    R["gpu_hours"]["total_mps"] = round(gh["h2a_fresh3"]["mps16"] + gh["e2"]["mps16"] + gh["e3"]["mps16"] +
                                        gh["e5_episodes"]["mps16_noinstr"] + gh["e5_qat_train_single_process"] +
                                        gh["e7_success"]["mps16_noinstr"], 1)
    R["errors"] = errs
    json.dump(R, open(a.out, "w"), indent=1)
    print(json.dumps(R, indent=1))
    print("CHECK_OK" if not errs else f"CHECK_FAILED {len(errs)}")


if __name__ == "__main__":
    main()
