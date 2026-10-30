"""Job manifests for runner.py (TD-MPC2 part of paper A).

Episode seed ranges (env, torch and numpy seeded with the episode seed; the same seeds for every checkpoint seed and
every variant, so cells are paired):
  calib   11000-11009  unquantized calibration rollouts (ep 0-4: quantizer ranges; ep 5-9: held prediction-error set
                       and fixed decision states)
  held    12000-12019  E0 task / bit-width selection (baseline success check, det drop at k = 3..6)
  eval    20000-20029  E1 (judgement); never used before E1
  smoke   19000-19009  smoke tests;  bench 18000-...: throughput;  repro 2000-2039: R2-8b reproduction (pilot seeds)
Subcommands (each writes one JSONL manifest):
  calib  --tasks .. --ckpts 1,2,3 [--eps 10]            calib jobs + ranges jobs (run --types calib, then ranges)
  held   --tasks .. --ckpts 1,2,3 [--ks 3,4,5,6] [--eps 20] [--dmc_held_eps 10] [--chunk 5]
  heldx  --tasks .. --ks 2 --variants det,shiter          supplementary held cells (information only)
  repro  [--eps 40] [--chunk 10]                          R2-8b cells with the R2-8 ranges, checkpoint seed 1
  bench  [--part mw|dmc] [--mw_eps 48] [--dmc_eps 24]     fixed throughput set (instrumented, like E1)
  e1     --ktable ktable.json [--ckpts 1,2,3] [--eps 30] [--chunk 5] [--variants ...]
  smoke  (2-3 tiny instrumented jobs)
  e2 / e3 / smoke2                                      v5 drafts (seeds 40000 / 50000; smoke code paths)
  identity  --src out/manifests/smoke.jsonl --res out   v5 identity check: E0 smoke jobs with the re-evaluations on
  h2a_indep --src out/manifests/e1.jsonl --res out [--variants ind,shiter,detshift]   post-hoc H2a rerun (prepared)
  e5stats / e5held / e5eval / e5w / e5qatdata / e5qattrain / e5qateval / e7     DESIGN_main v5 7.3 / 7.4
Usage: python make_manifest.py e1 --ktable <out>/ktable.json --out <out>/manifests/e1.jsonl
"""
import argparse
import json
import os

SEED0 = {"calib": 11000, "held": 12000, "eval": 20000, "smoke": 19000, "bench": 18000, "repro": 2000,
         # v5 draft: a fresh, never-used seed range per experiment (independent replication of the E1 default)
         "e2": 40000, "e3": 50000, "e5": 60000, "e7": 70000, "e5qat": 65000}
E2_TASKS = {"mw-push": 3, "mw-pick-place": 3, "cheetah-run": 5}     # experiment 2 tasks at their primary k
DEFAULT_PLANNER = {"num_samples": 512, "num_elites": 64, "num_pi_trajs": 24, "temperature": 0.5, "iterations": 6}


def e2_settings():
    """One-factor-at-a-time around the TD-MPC2 default (DESIGN_main v5 7.1 + v5.1 8.2). Confirmatory N sweep (H2c,
    v5.1): only N changes, elite count 64 and policy prior 24 fixed (N64k / N128k / N256k / N1024k; base = N 512; at
    N = 64 all 64 candidates are elites). Auxiliary N sweep (v5 7.1, descriptive): elite RATIO 12.5% and policy-prior
    share 24/512 scaled with N (N64 / N128 / N256 / N1024). Elite ratio and temperature at N = 512; iteration count
    with warm start on and off. Returns [(ptag, planner overrides, warm_start)]; 'base' is the default (rerun on E2
    seeds)."""
    S = [("base", {}, True)]
    for n in (64, 128, 256, 1024):
        S.append((f"N{n}k", {"num_samples": n}, True))
    for n in (64, 128, 256, 1024):
        S.append((f"N{n}", {"num_samples": n, "num_elites": n // 8, "num_pi_trajs": round(24 * n / 512)}, True))
    S += [("el2", {"num_elites": 10}, True), ("el25", {"num_elites": 128}, True),
          ("T01", {"temperature": 0.1}, True), ("T2", {"temperature": 2.0}, True)]
    for it in (1, 3, 12):
        S.append((f"it{it}", {"iterations": it}, True))
    for it in (1, 3, 6, 12):
        S.append((f"nows{it}", {"iterations": it}, False))
    return S
MW8 = ["mw-push", "mw-pick-place", "mw-door-open", "mw-drawer-open", "mw-button-press-topdown",
       "mw-peg-insert-side", "mw-window-open", "mw-sweep-into"]
DMC4 = ["cheetah-run", "walker-walk", "quadruped-walk", "finger-turn-hard"]
VARIANTS = ["none", "det", "ind", "shiter", "shstep", "shep", "detshift"]


def instr(task, on=True):
    """E1 instrumentation: OEI/decision metrics every 4th (MW, 25 states/episode) or 10th (DMC, 50) control step, plan
    reruns every 20th (MW, 5 states/episode) or 50th (DMC, 10) step, per-episode prediction errors with 3 dither seeds
    (trajectory inputs: every window; planning-time inputs: 8 of the OEI states). Densities chosen in E0 from the
    measured cost (bench): 90 episodes per cell give 2,250 / 4,500 OEI states and 450 / 900 plan-deviation states."""
    if not on:
        return {}
    mw = task.startswith("mw-")
    return {"oei_stride": 4 if mw else 10, "pd_stride": 20 if mw else 50, "perr": True, "perr_seeds": 3,
            "perr_plan_states": 8}


def ep_job(split, task, ckpt, k, variant, eps, chunk_id, scales="main", planner=None, ptag="base", instr_on=True,
           extra=None):
    j = {"id": f"{split}_{task}_s{ckpt}_k{k}_{variant}_{ptag}_c{chunk_id:02d}", "type": "episodes", "split": split,
         "task": task, "ckpt": int(ckpt), "k": int(k), "variant": variant, "eps": list(eps), "seed0": SEED0[split],
         "scales": scales, "planner": planner or {}, "instr": instr(task, instr_on)}
    j.update(extra or {})
    return j


def chunks(n, size):
    return [list(range(s, min(n, s + size))) for s in range(0, n, size)]


def dmc_first(tasks):
    return [t for t in tasks if not t.startswith("mw-")] + [t for t in tasks if t.startswith("mw-")]


def m_calib(a):
    jobs = []
    for task in dmc_first(a.tasks):
        for c in a.ckpts:
            for ep in range(a.eps):
                jobs.append({"id": f"calib_{task}_s{c}_ep{ep}", "type": "calib", "task": task, "ckpt": c, "ep": ep,
                             "seed": SEED0["calib"] + ep})
    for task in a.tasks:
        for c in a.ckpts:
            jobs.append({"id": f"ranges_{task}_s{c}", "type": "ranges", "task": task, "ckpt": c,
                         "cal_eps": list(range(min(5, a.eps)))})
    return jobs


def m_held(a):
    """Held split: none and det at every k, 20 episodes per checkpoint seed for Meta-World (binary success) and
    --dmc_held_eps (default 10) per checkpoint seed for DMC (return; see the E0 report for this choice)."""
    jobs = []
    for ch_i, eps in enumerate(chunks(a.eps, a.chunk)):
        for task in dmc_first(a.tasks):
            if not task.startswith("mw-"):
                eps_t = [e for e in eps if e < a.dmc_held_eps]
                if not eps_t:
                    continue
            else:
                eps_t = eps
            for c in a.ckpts:
                jobs.append(ep_job("held", task, c, 0, "none", eps_t, ch_i, instr_on=False))
                for k in a.ks:
                    jobs.append(ep_job("held", task, c, k, "det", eps_t, ch_i, instr_on=False))
    return jobs


def m_heldx(a):
    """Supplementary held-split cells (information for a decision, e.g. the k-1 = 2 cell of a primary k = 3):
    --tasks, --ks, --variants (quantized only), --ckpts, --eps per checkpoint (DMC capped by --dmc_held_eps)."""
    jobs = []
    for ch_i, eps in enumerate(chunks(a.eps, a.chunk)):
        for task in dmc_first(a.tasks):
            eps_t = eps if task.startswith("mw-") else [e for e in eps if e < a.dmc_held_eps]
            if not eps_t:
                continue
            for c in a.ckpts:
                for k in a.ks:
                    for v in a.variants:
                        if v != "none":
                            jobs.append(ep_job("held", task, c, k, v, eps_t, ch_i, instr_on=False))
    return jobs


def e2_instr(task):
    """E2 instrumentation: E1's OEI / plan-rerun densities, no per-episode prediction errors (E2 does not use them),
    plus the independent-draw re-evaluation of the executed elite (e_indep) on every OEI step."""
    d = instr(task, True)
    d.update(perr=False, indep_reeval=True)
    return d


def m_e2(a):
    """E2 (TD-MPC2 part, DESIGN_main v5 7.1). Variants: none, det, sh-iter, sh-step at every setting; ind, sh-ep,
    det+shift at the default only. Every setting 30 episodes per checkpoint (90 pairs, v5 7.0; --eps_other kept for
    the record, default 30)."""
    jobs = []
    sets = e2_settings()
    for ptag, ov, warm in sets:
        n_eps = a.eps if (ptag == "base" or ptag.startswith("N")) else a.eps_other
        vs = ["none", "det", "shiter", "shstep"] + (["ind", "shep", "detshift"] if ptag == "base" else [])
        for ch_i, eps in enumerate(chunks(n_eps, a.chunk)):
            for task, k in E2_TASKS.items():
                for c in a.ckpts:
                    for v in vs:
                        j = ep_job("e2", task, c, 0 if v == "none" else k, v, eps, ch_i, planner=ov, ptag=ptag,
                                   extra={"warm_start": warm})
                        j["instr"] = e2_instr(task)
                        if ptag == "base":      # v5.2 layer (b3): region-effect reference at E2 base only
                            j["instr"]["region_ref"] = True
                        jobs.append(j)
    # DMC first inside every (setting, chunk) block keeps the tail short
    jobs.sort(key=lambda j: (j["eps"][0], 0 if not j["task"].startswith("mw-") else 1))
    return jobs


def m_e3(a):
    """E3 (TD-MPC2 part, v5 draft). (i) 2-axis cells at the primary k of all 12 tasks: none, det, ind, sh-iter,
    sh-step, sh-ep, indstep, indep. (ii) table-size sweep of the plan-fixed shared table on --sweep_tasks: shstep_l
    (1 per layer), shstep_g32, shstep_u (512), [shstep (H x 512) and indstep (cand x H x 512) come from (i)].
    (iii) decision jobs (rho_iter, common_frac, fixed-state plan deviation) for every new variant.
    Episodes use the E1 instrumentation plus the independent-draw re-evaluations (e_indep, e_fresh)."""
    kt = json.load(open(a.ktable))
    jobs = []
    sweep = a.sweep_tasks.split(",") if a.sweep_tasks else []
    for ch_i, eps in enumerate(chunks(a.eps, a.chunk)):
        for task in dmc_first(list(kt)):
            k = kt[task]["primary"]
            vs = ["none", "det", "ind", "shiter", "shstep", "shep", "indstep", "indep"]
            if task in sweep:   # plan-fixed shared tables of 4 sizes + fresh draws of the same coarse granularities
                vs += ["shstep_l", "shstep_g32", "shstep_u", "shiter_l", "shiter_g32"]
            for c in a.ckpts:
                for v in vs:
                    j = ep_job("e3", task, c, 0 if v == "none" else k, v, eps, ch_i)
                    j["instr"]["indep_reeval"] = True     # both OEI versions from E2 on (same as the DINO-WM family)
                    jobs.append(j)
    for task in kt:
        k = kt[task]["primary"]
        qv = ["indstep", "indep"] + (["shstep_l", "shstep_g32", "shstep_u", "shiter_l", "shiter_g32"] if task in sweep
                                      else [])
        for c in a.ckpts:
            jobs.append({"id": f"decision_e3_{task}_s{c}_k{k}", "type": "decision", "task": task, "ckpt": c, "k": k,
                         "variants": qv, "held_eps": [5, 6, 7, 8, 9], "max_states": 150, "plan_states": 150})
    return jobs


def m_smoke2(a):
    """Smoke jobs for the v5 code paths (smoke seeds, never analysed): E2 settings (N64 with 3 policy proposals,
    N1024, elite 2%, T 0.1, 1 and 12 iterations, warm start off) with the independent re-evaluation, and every E3
    variant, on mw-push k=3 (R2-8 ranges) and one cheetah-run job."""
    jobs = []
    for ptag, ov, warm in e2_settings():
        if ptag in ("base", "N64", "N1024", "el2", "T01", "it1", "it12", "nows6"):
            for v in ("det", "shiter"):
                j = ep_job("smoke", "mw-push", 1, 3, v, [3], 0, scales="r28", planner=ov, ptag=f"e2{ptag}",
                           extra={"warm_start": warm, "log_calls": True})
                j["instr"] = e2_instr("mw-push")
                jobs.append(j)
    for v in ("indstep", "indep", "shstep_u", "shstep_g32", "shstep_l", "shep_u", "ind", "shstep", "shiter_l",
              "shiter_g32"):
        j = ep_job("smoke", "mw-push", 1, 3, v, [3], 0, scales="r28", ptag="e3", extra={"log_calls": True})
        j["instr"] = e2_instr("mw-push")
        jobs.append(j)
    j = ep_job("smoke", "cheetah-run", 1, 4, "shiter", [3], 0, scales="r28", ptag="e2base")
    j["instr"] = e2_instr("cheetah-run")
    jobs.append(j)
    jobs.append({"id": "smoke2_decision_mw-push_s1_k3", "type": "decision", "task": "mw-push", "ckpt": 1, "k": 3,
                 "variants": ["indstep", "indep", "shstep_u", "shstep_g32", "shstep_l", "shiter_l", "shiter_g32"],
                 "held_eps": [5, 6, 7, 8, 9], "max_states": 6, "plan_states": 6})
    jobs.append({"id": "smoke2_prederr_mw-push_s1_k3", "type": "prederr", "task": "mw-push", "ckpt": 1, "k": 3,
                 "variants": ["indstep", "indep", "shstep_u", "shstep_g32", "shstep_l", "shiter_l", "shiter_g32"],
                 "held_eps": [5, 6, 7, 8, 9], "nseeds": 3})
    return jobs


def spec_from_result(res, spec):
    """The spec actually used by a finished job (stored in its result file)."""
    import gzip
    return json.load(gzip.open(f"{res}/jobs/{spec['type']}/{spec['id']}.json.gz", "rt"))["spec"]


def m_identity(a):
    """v5 identity check on the smoke split: the E0 smoke episode jobs (--src, e.g. out/manifests/smoke.jsonl, results in
    --res) re-run with the current code and the E2/E3 re-evaluations on. Specs are read from the stored E0 results;
    perr_plan_states = 0 (all OEI states) reproduces the E0-era planning-time prediction errors (the 8-state subset
    came at 17:56Z, before E1). Compare with compare_runs.py; every stored field must be equal."""
    jobs = []
    for j in (json.loads(l) for l in open(a.src) if l.strip()):
        if j["type"] != "episodes" or j.get("split") != "smoke":
            continue
        if not os.path.exists(f"{a.res}/jobs/episodes/{j['id']}.json.gz"):
            continue
        s = spec_from_result(a.res, j)
        s["instr"] = dict(s["instr"], indep_reeval=True)
        if s["instr"].get("perr"):
            s["instr"].setdefault("perr_plan_states", 0)
        jobs.append(s)
    return jobs


H2A_CELLS = {"mw-push": 3, "mw-pick-place": 3, "cheetah-run": 5}


def m_h2a_indep(a):
    """POST-HOC correction (needs the user's approval; prepared, not run): the E1 episode jobs of the three H2a cells
    (mw-push k=3, mw-pick-place k=3, cheetah-run k=5; --variants) re-run with the independent-draw re-evaluation of
    the executed elite. Same job ids and specs as the stored E1 results (seeds 20000-20029, checkpoints 1-3, E0 ranges,
    OEI / plan-rerun strides), per-episode prediction errors off (not needed; they never touch the closed loop).
    Write to a separate --out directory (same ids); every rerun episode must reproduce E1 bitwise (compare_runs.py)."""
    jobs = []
    for j in (json.loads(l) for l in open(a.src) if l.strip()):
        if j["type"] != "episodes" or j.get("split") != "eval" or j["task"] not in H2A_CELLS:
            continue
        if j["variant"] not in a.variants or (j["variant"] != "none" and j["k"] != H2A_CELLS[j["task"]]):
            continue
        s = spec_from_result(a.res, j)
        s["instr"] = dict(s["instr"], indep_reeval=True, perr=False)
        jobs.append(s)
    jobs.sort(key=lambda j: (j["eps"][0], 0 if not j["task"].startswith("mw-") else 1))
    return jobs


# ---------------------------------------------------------------------------------------------- E5 (v5 7.3)
# E1 Holm-qualifying 3-4 bit cells (v5 7.3 list; cross-checked against results/E1_tdmpc2/e1_tdmpc2_summary.json)
E5_CELLS = {"mw-button-press-topdown": [3], "mw-door-open": [3, 4], "mw-drawer-open": [3], "mw-peg-insert-side": [3, 4],
            "mw-pick-place": [3], "mw-push": [3, 4], "mw-sweep-into": [3], "mw-window-open": [3, 4],
            "cheetah-run": [4], "finger-turn-hard": [3, 4], "quadruped-walk": [3, 4], "walker-walk": [3, 4]}
# QAT: the qualifying cells at the task's primary k (11 cells; cheetah's primary k = 5 is not a 3-4 bit cell)
E5_QAT = {"mw-button-press-topdown": 3, "mw-door-open": 4, "mw-drawer-open": 3, "mw-peg-insert-side": 4,
          "mw-pick-place": 3, "mw-push": 3, "mw-sweep-into": 3, "mw-window-open": 4, "finger-turn-hard": 4,
          "quadruped-walk": 3, "walker-walk": 3}
E5_PTQ = ["perchan", "mseclip", "smooth", "mixed"]           # v5 order (a tie goes to the earlier one)
E5_W = ["w8rtn", "w4rtn", "w4awq", "w4sr1", "w4sr2", "w4sr4"]
SWEEP4 = ["mw-push", "mw-pick-place", "cheetah-run", "walker-walk"]


def m_e5stats(a):
    """Per (task, checkpoint): activation statistics of calibration episodes 0-4 (the E0 range rows) for the PTQ
    baselines (per-channel ranges, MSE clip factors for k = 3, 4, SmoothQuant scales) and the AWQ input means."""
    return [{"id": f"e5stats_{t}_s{c}", "type": "e5stats", "task": t, "ckpt": c, "cal_eps": [0, 1, 2, 3, 4],
             "ks": [3, 4]} for t in dmc_first(list(E5_CELLS)) for c in a.ckpts]


def m_e5held(a):
    """Baseline selection on the held split (seeds 12000-12019, 20 per checkpoint = 60 pairs): det under each of the 4
    PTQ baselines in every E5 cell, and none per task (re-run here so the held drop D_det(B) is reported from one
    code version). No instrumentation."""
    jobs = []
    for ch_i, eps in enumerate(chunks(20, 5)):
        for t in dmc_first(list(E5_CELLS)):
            for c in a.ckpts:
                jobs.append(ep_job("held", t, c, 0, "none", eps, ch_i, ptag="e5none", instr_on=False))
                for k in E5_CELLS[t]:
                    for b in E5_PTQ:
                        jobs.append(ep_job("held", t, c, k, "det", eps, ch_i, ptag=f"e5{b}", instr_on=False,
                                           extra={"ptq": b}))
    return jobs


def m_e5eval(a):
    """H3b evaluation (E5 seeds 60000-60029, 90 pairs): B + det and B + sh-iter in every E5 cell with B = the cell's
    selected baseline ('best' is resolved at run time from <out>/e5/selection.json, written by e5_select.py from the
    held jobs, and recorded in the result), none per task. No instrumentation."""
    jobs = []
    for ch_i, eps in enumerate(chunks(30, 5)):
        for t in dmc_first(list(E5_CELLS)):
            for c in a.ckpts:
                jobs.append(ep_job("e5", t, c, 0, "none", eps, ch_i, instr_on=False))
                for k in E5_CELLS[t]:
                    for v in ("det", "shiter"):
                        jobs.append(ep_job("e5", t, c, k, v, eps, ch_i, ptag="e5best", instr_on=False,
                                           extra={"ptq": "best"}))
    return jobs


def m_e5w(a):
    """Weight-only quantization (exploratory; activations in fp32) on the 4 sweep tasks, E5 seeds, 90 pairs; the none
    baseline is the E5 evaluation's none (same seeds)."""
    jobs = []
    for ch_i, eps in enumerate(chunks(30, 5)):
        for t in dmc_first(SWEEP4):
            for c in a.ckpts:
                for w in E5_W:
                    jobs.append(ep_job("e5", t, c, 0, "none", eps, ch_i, ptag=f"e5{w}", instr_on=False,
                                       extra={"wq": w}))
    return jobs


def m_e5qatdata(a):
    """QAT data: 20 unquantized episodes per (task, checkpoint) on seeds 65000-65019, trajectories saved to
    <out>/qatdata/<task>_s<ckpt>_ep<ep>.npz."""
    jobs = []
    for ch_i, eps in enumerate(chunks(20, 5)):
        for t in dmc_first(list(E5_QAT)):
            for c in a.ckpts:
                jobs.append(ep_job("e5qat", t, c, 0, "none", eps, ch_i, ptag="qatdata", instr_on=False,
                                   extra={"save_traj": True}))
    return jobs


def m_e5qattrain(a):
    """QAT (v5 7.3): det fake-quant + STE on the 4 hidden activations at k (E0 ranges), dynamics and reward MLPs only,
    TD-MPC2 consistency + reward losses, 5,000 steps, batch 256, lr 1e-4, grad clip 20, on the 20 QAT data episodes."""
    return [{"id": f"qattrain_{t}_s{c}_k{k}", "type": "qat_train", "task": t, "ckpt": c, "k": k,
             "data_eps": list(range(20)), "steps": 5000, "batch": 256, "lr": 1e-4, "grad_clip": 20.0, "seed": 0}
            for t, k in sorted(E5_QAT.items(), key=lambda x: (x[0].startswith("mw-"), x[0])) for c in a.ckpts]


def m_e5qateval(a):
    """QAT evaluation (E5 seeds, 90 pairs): QAT + det and QAT + sh-iter; none is the E5 evaluation's none."""
    jobs = []
    for ch_i, eps in enumerate(chunks(30, 5)):
        for t in dmc_first(list(E5_QAT)):
            k = E5_QAT[t]
            for c in a.ckpts:
                for v in ("det", "shiter"):
                    jobs.append(ep_job("e5", t, c, k, v, eps, ch_i, ptag="e5qat", instr_on=False,
                                       extra={"qat": True}))
    return jobs


# ---------------------------------------------------------------------------------------------- E7 (v5 7.4)
E7_PRECS = [("fp32", 0, "none"), ("bf16", 0, "none"), ("int8", 8, "det"), ("int8", 8, "shiter"),
            ("int4", 4, "det"), ("int4", 4, "shiter")]


def m_e7(a):
    """E7 success rates with the real kernels (v5 7.4): 6 precisions x MPPI iterations {3, 6, 12} x 4 tasks, seeds
    70000-70029, 90 pairs. int8 / int4 = W8 + 8- / 4-bit activations on the E0 grid (det or sh-iter dither)."""
    jobs = []
    for ch_i, eps in enumerate(chunks(30, 5)):
        for t in dmc_first(SWEEP4):
            for c in a.ckpts:
                for prec, k, v in E7_PRECS:
                    for it in (3, 6, 12):
                        jobs.append(ep_job("e7", t, c, k, v, eps, ch_i, planner={"iterations": it},
                                           ptag=f"e7{prec}it{it}", instr_on=False, extra={"e7": prec}))
    return jobs


def m_repro(a):
    cells = [("cheetah-run", 4, ["none", "det", "shiter"]),
             ("mw-push", 3, ["none", "det", "ind", "shiter", "shstep", "shep"]),
             ("mw-pick-place", 3, ["none", "det", "ind", "shiter", "shstep", "shep"])]
    jobs = []
    for ch_i, eps in enumerate(chunks(a.eps, a.chunk)):
        for task, k, vs in cells:
            for v in vs:
                jobs.append(ep_job("repro", task, 1, 0 if v == "none" else k, v, eps, ch_i, scales="r28",
                                   instr_on=False))
    return jobs


def m_bench(a):
    """Throughput set: E1-like instrumented jobs (det / shiter at the pilot k, R2-8 ranges); --part mw: mw-push and
    mw-pick-place, 2 episodes per job; --part dmc: cheetah-run, 1 episode per job."""
    jobs = []
    on = not a.noinstr
    if a.part in ("dmc", "all"):
        for ch_i, eps in enumerate(chunks(a.dmc_eps, 1)):
            for v in ("det", "shiter"):
                jobs.append(ep_job("bench", "cheetah-run", 1, 4, v, eps, ch_i, scales="r28", instr_on=on))
    if a.part in ("mw", "all"):
        for ch_i, eps in enumerate(chunks(a.mw_eps, 2)):
            for task in ("mw-push", "mw-pick-place"):
                for v in ("det", "shiter"):
                    jobs.append(ep_job("bench", task, 1, 3, v, eps, ch_i, scales="r28", instr_on=on))
    return jobs


def m_e1(a):
    """E1: per-cell held-set prediction errors and fixed-state decision jobs first (fast, early diagnostics), then the
    closed-loop episodes in episode-major chunks (every cell gets its first chunk before any cell gets its second; DMC
    before MW inside a chunk)."""
    kt = json.load(open(a.ktable))          # {"task": {"primary": k, "cells": [k-1, k, k+1]}, ...}
    tasks = dmc_first([t for t in kt if kt[t].get("cells")])
    jobs = []
    for task in tasks:
        for c in a.ckpts:
            for k in kt[task]["cells"]:
                qv = [v for v in a.variants if v != "none"]
                jobs.append({"id": f"prederr_{task}_s{c}_k{k}", "type": "prederr", "task": task, "ckpt": c, "k": k,
                             "variants": qv, "held_eps": [5, 6, 7, 8, 9], "nseeds": 3})
                jobs.append({"id": f"decision_{task}_s{c}_k{k}", "type": "decision", "task": task, "ckpt": c, "k": k,
                             "variants": qv, "held_eps": [5, 6, 7, 8, 9], "max_states": 150, "plan_states": 150})
    for ch_i, eps in enumerate(chunks(a.eps, a.chunk)):
        for task in tasks:
            for c in a.ckpts:
                if "none" in a.variants:
                    jobs.append(ep_job("eval", task, c, 0, "none", eps, ch_i))
                for k in kt[task]["cells"]:
                    for v in a.variants:
                        if v != "none":
                            jobs.append(ep_job("eval", task, c, k, v, eps, ch_i))
    return jobs


def m_smoke(a):
    """Tiny instrumented jobs on smoke seeds (never analysed): every variant on mw-push k=3 (1 episode, R2-8 ranges)
    with call logging, cheetah-run det k=4, two planner-override jobs (E2 code paths), plus prederr / decision on
    mw-push checkpoint 1 (need the calibration files and ranges of that task)."""
    jobs = []
    for v in VARIANTS:
        jobs.append(ep_job("smoke", "mw-push", 1, 0 if v == "none" else 3, v, [0], 0, scales="r28",
                           extra={"log_calls": True}))
    jobs.append(ep_job("smoke", "cheetah-run", 1, 4, "det", [0], 0, scales="r28"))
    jobs.append(ep_job("smoke", "mw-push", 1, 3, "shiter", [1], 0, scales="r28", ptag="N128it3",
                       planner={"num_samples": 128, "iterations": 3}))
    jobs.append(ep_job("smoke", "mw-push", 1, 3, "det", [1], 0, scales="r28", ptag="nows",
                       extra={"warm_start": False}))
    # E1-shaped jobs with the main (E0-calibrated) ranges on other checkpoints, smoke seeds
    jobs.append(ep_job("smoke", "cheetah-run", 2, 5, "shiter", [2], 0))
    jobs.append(ep_job("smoke", "mw-door-open", 3, 4, "detshift", [2, 3], 0))
    jobs.append(ep_job("smoke", "walker-walk", 1, 2, "det", [2], 0))
    jobs.append({"id": "smoke_prederr_mw-push_s1_k3", "type": "prederr", "task": "mw-push", "ckpt": 1, "k": 3,
                 "variants": VARIANTS[1:], "held_eps": [5, 6, 7, 8, 9], "nseeds": 3})
    jobs.append({"id": "smoke_decision_mw-push_s1_k3", "type": "decision", "task": "mw-push", "ckpt": 1, "k": 3,
                 "variants": VARIANTS[1:], "held_eps": [5, 6, 7, 8, 9], "max_states": 10, "plan_states": 10})
    return jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calib", "held", "heldx", "repro", "bench", "e1", "smoke", "e2", "e3", "smoke2",
                                    "identity", "h2a_indep", "e5stats", "e5held", "e5eval", "e5w", "e5qatdata",
                                    "e5qattrain", "e5qateval", "e7"])
    ap.add_argument("--tasks", default=",".join(MW8 + DMC4))
    ap.add_argument("--ckpts", default="1,2,3")
    ap.add_argument("--ks", default="3,4,5,6")
    ap.add_argument("--eps", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--mw_eps", type=int, default=48)
    ap.add_argument("--dmc_eps", type=int, default=24)
    ap.add_argument("--part", default="all", choices=["all", "mw", "dmc"])
    ap.add_argument("--dmc_held_eps", type=int, default=10)
    ap.add_argument("--noinstr", action="store_true")
    ap.add_argument("--eps_other", type=int, default=30)
    ap.add_argument("--sweep_tasks", default="mw-push,mw-pick-place,cheetah-run,walker-walk")
    ap.add_argument("--ktable", default=None)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--src", default=None, help="identity / h2a_indep: source manifest")
    ap.add_argument("--res", default=None, help="identity / h2a_indep: result directory of the source jobs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.tasks = a.tasks.split(",")
    a.ckpts = [int(x) for x in a.ckpts.split(",")]
    a.ks = [int(x) for x in a.ks.split(",")]
    a.variants = a.variants.split(",")
    dflt = {"calib": (10, 1), "held": (20, 5), "heldx": (20, 5), "repro": (40, 10), "bench": (0, 0), "e1": (30, 5),
            "smoke": (1, 1), "e2": (30, 5), "e3": (30, 5), "smoke2": (1, 1), "identity": (0, 0), "h2a_indep": (0, 0),
            "e5stats": (0, 0), "e5held": (20, 5), "e5eval": (30, 5), "e5w": (30, 5), "e5qatdata": (20, 5),
            "e5qattrain": (0, 0), "e5qateval": (30, 5), "e7": (30, 5)}
    a.eps = a.eps if a.eps is not None else dflt[a.cmd][0]
    a.chunk = a.chunk if a.chunk is not None else dflt[a.cmd][1]
    jobs = {"calib": m_calib, "held": m_held, "heldx": m_heldx, "repro": m_repro, "bench": m_bench, "e1": m_e1,
            "smoke": m_smoke, "e2": m_e2, "e3": m_e3, "smoke2": m_smoke2, "identity": m_identity,
            "h2a_indep": m_h2a_indep, "e5stats": m_e5stats, "e5held": m_e5held, "e5eval": m_e5eval, "e5w": m_e5w,
            "e5qatdata": m_e5qatdata, "e5qattrain": m_e5qattrain, "e5qateval": m_e5qateval, "e7": m_e7}[a.cmd](a)
    ids = [j["id"] for j in jobs]
    assert len(ids) == len(set(ids)), "duplicate job ids"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    from collections import Counter
    print(f"{len(jobs)} jobs -> {a.out}: {dict(Counter(j['type'] for j in jobs))}")
