"""Job plan of the td group (TD-MPC2 5M; tmlr_ext/DESIGN_tmlr.md v1.0 sections 0, 1, 2, 3, 5, 6, 8).

Cells and seeds are the design's; this file only fixes the operational layout (job size, ids, order, dependencies).
  seeds   eval 100000-100029 x checkpoints 1-3 (90 pairs; E11, E12, E15, E16), held 101000-101029 (E13),
          calibration 11000-11004 (E0 convention, re-recorded on the pod), smoke 109000+, bench 108000+ (never analysed)
  job     one (task, ckpt, cell, planner tag) x 5 consecutive episodes (chunk c = episodes 5c..5c+4); chunks 0-2 are
          seeds 100000-100014 = the first 45 pairs of the DMC two-stage rule, chunks 0-3 the 60 pairs of iterations 24
  cell    (variant, range kind rk in {PT, PC}, k); none has no range; planner tags: base (TD-MPC2 default), it12 / it24
          (iterations 12 / 24, E15), cem (temperature 0 = equal elite weights, min_std 0.01, num_pi_trajs 0, E16)
  reuse   R1 (E12 PT cells at the primary k are E11's), R2 (E15 iterations 6 = E11 DMC cells), R3 (E16 default =
          E11), R5 (one none per (task, ckpt, planner tag) on the pod, shared by every experiment)
  order   calib -> prep -> E11 none -> E11 variants (chunk-major) -> E12 (extension cells, then grid boundary cells,
          then interior cells; cell-major so the adaptive rules get their inputs early; DMC stage 2 as decided) ->
          E15 -> E16 -> E13 -> timing (alone)
Adaptive rules (DESIGN_tmlr 2.2, applied by the driver through adaptive_jobs()):
  two-stage   DMC E12 cells first run 45 pairs; if the 95% paired-bootstrap CI (10,000, RandomState(0)) of D on those
              pairs contains tau = 0.10, the other 45 pairs are run. Meta-World cells run 90 pairs from the start.
  extension   per (task, range kind, rounding): if the lowest measured k has D <= tau (complete cell), the next lower
              k is added (down to 1); if the highest measured k has D > tau, the next higher k is added (up to 8).
              D of a cell uses all its pairs; comparisons with tolerance 1e-9.
"""
import gzip
import json
import os

import numpy as np

MW8 = ["mw-push", "mw-pick-place", "mw-door-open", "mw-drawer-open", "mw-button-press-topdown",
       "mw-peg-insert-side", "mw-window-open", "mw-sweep-into"]
DMC4 = ["cheetah-run", "walker-walk", "quadruped-walk", "finger-turn-hard"]
TASKS = MW8 + DMC4
K1 = {"mw-push": 3, "mw-pick-place": 3, "mw-door-open": 4, "mw-drawer-open": 3, "mw-button-press-topdown": 3,
      "mw-peg-insert-side": 4, "mw-window-open": 4, "mw-sweep-into": 3, "cheetah-run": 5, "walker-walk": 3,
      "quadruped-walk": 3, "finger-turn-hard": 4}                      # E1 primary k (DESIGN_main 0.2)
POD_TASKS = {"td.1": ["cheetah-run"], "td.2": ["walker-walk"], "td.3": ["quadruped-walk"], "td.4": ["finger-turn-hard"],
             "td.5": ["mw-push", "mw-pick-place", "mw-door-open", "mw-drawer-open"],
             "td.6": ["mw-button-press-topdown", "mw-peg-insert-side", "mw-window-open", "mw-sweep-into"]}
E11_VARS = ["det", "detbc", "shep", "detshiftfix", "shiterb", "shiter", "detshift", "detnoise"]
E12_ROUND = ["det", "detbc", "shiter", "detshift"]
E12_GRID = {True: {"PT": [2, 3, 4, 5], "PC": [1, 2, 3]}, False: {"PT": [3, 4, 5], "PC": [2, 3, 4]}}   # key: is_mw
E15_ITERS = {"it12": 6, "it24": 4}           # chunks per checkpoint: 90 pairs / 60 pairs (seeds 100000-100019)
E15_VARS = ["none", "det", "shiter", "detshift"]
E16_TASKS = ["mw-push", "mw-pick-place", "cheetah-run"]
E16_VARS = ["none", "det", "shiter", "shep", "detshift"]
MECH = ["mw-push", "mw-pick-place", "cheetah-run"]      # E11 mechanism tasks: E2 PlanProbe instrumentation
RKI = {"PT": 0, "PC": 1}
E1_VARS = ["det", "ind", "shiter", "shstep", "shep", "detshift"]   # + none (E1's 7 variants)
PLANNERS = {"base": {}, "it12": {"iterations": 12}, "it24": {"iterations": 24},
            "cem": {"temperature": 0.0, "min_std": 0.01, "num_pi_trajs": 0}}
SEED0 = {"eval": 100000, "held": 101000, "calib": 11000, "smoke": 109000, "bench": 108000, "repro": 12000}
CKPTS = [1, 2, 3]
CHUNK = 5
N_CH = 6                     # 30 episodes per checkpoint
STAGE1_CH = 3                # DMC two-stage: chunks 0-2 = seeds 100000-100014 = 45 pairs
TAU = 0.10
TOL = 1e-9
KMIN, KMAX = 1, 8
NB = 10000
# rough cost (worker-seconds at 8 MPS workers on an A40, E9 PLAN 3) for the ETA only
EP_SECS = {True: 4.5, False: 22.0}
PT_COST = {"base": 1.0, "it12": 1.9, "it24": 3.7, "cem": 0.95}


def is_mw(t):
    return t.startswith("mw-")


def cell_name(v, rk, k):
    return "none" if v == "none" else f"{rk}k{int(k)}_{v}"


def ep_id(task, ckpt, v, rk, k, ptag, ch, split="ev"):
    return f"{split}_{task}_s{ckpt}_{cell_name(v, rk, k)}_{ptag}_c{ch:02d}"


def e11_instr(task):
    """E2's PlanProbe instrumentation (make_manifest.e2_instr without plan reruns / region reference): OEI states every
    4th (MW) / 10th (DMC) control step, decision metrics with drift and sel_last, independent-draw re-evaluation."""
    return {"oei_stride": 4 if is_mw(task) else 10, "indep_reeval": True} if task in MECH else {}


def ep_job(task, ckpt, v, rk, k, ptag, ch, exps, prio, instr=None, split="eval", seed0=None, extra=None):
    deps = []
    if v != "none" and (rk == "PC" or v in ("detbc", "shiterb")):
        deps.append(f"prep_{task}_s{ckpt}")
    if v in ("detbc", "shiterb"):            # bias vectors: the gated bias job (rule pending, see bias_jobs)
        deps.append(f"bias_{task}_s{ckpt}")
    cost = EP_SECS[is_mw(task)] * CHUNK * PT_COST[ptag] * (1.25 if instr else 1.0)
    j = {"id": ep_id(task, ckpt, v, rk, k, ptag, ch, "ev" if split == "eval" else split), "type": "episodes",
         "split": split, "task": task, "ckpt": int(ckpt), "k": 0 if v == "none" else int(k), "variant": v,
         "rk": "-" if v == "none" else rk, "scales": "pc" if (v != "none" and rk == "PC") else "e1",
         "eps": list(range(CHUNK * ch, CHUNK * ch + CHUNK)), "seed0": SEED0[split] if seed0 is None else seed0,
         "ptag": ptag, "planner": dict(PLANNERS[ptag]), "instr": dict(instr or {}), "exps": list(exps),
         "deps": deps, "prio": list(prio), "cost": cost}
    j.update(extra or {})
    return j


def calib_jobs(task, prio=(0,)):
    out = []
    for c in CKPTS:
        for ep in range(5):
            out.append({"id": f"calib_{task}_s{c}_ep{ep}", "type": "calib", "task": task, "ckpt": c, "ep": ep,
                        "seed": SEED0["calib"] + ep, "deps": [], "prio": list(prio) + [c, ep],
                        "cost": EP_SECS[is_mw(task)] * 1.2})
    return out


def prep_jobs(task, prio=(1,)):
    return [{"id": f"prep_{task}_s{c}", "type": "prep", "task": task, "ckpt": c, "cal_eps": [0, 1, 2, 3, 4],
             "deps": [f"calib_{task}_s{c}_ep{ep}" for ep in range(5)], "prio": list(prio) + [c], "cost": 20.0}
            for c in CKPTS]


def bias_jobs(task, prio=(1,)):
    """Bias vectors b (DESIGN_tmlr 0.6) per (task, ckpt). Gated ("bias"): the driver submits them only after the
    coordinator has fixed the iteration rule (<ATM>/bias_rule.json) and opened the gate (<ATM>/GATE_bias); every
    det+BC / sh-iter+b episode job and every E13 job depends on them."""
    return [{"id": f"bias_{task}_s{c}", "type": "bias", "task": task, "ckpt": c, "cal_eps": [0, 1, 2, 3, 4],
             "deps": [f"prep_{task}_s{c}"], "gate": "bias", "prio": list(prio) + [5, c], "cost": 60.0}
            for c in CKPTS]


def e12_grid(task):
    return {rk: list(ks) for rk, ks in E12_GRID[is_mw(task)].items()}


def e12_cell_chunks(task, stage2=False):
    """Chunks of an E12 cell: MW all 6; DMC stage 1 = chunks 0-2, stage 2 = chunks 3-5."""
    if is_mw(task):
        return list(range(N_CH)) if not stage2 else []
    return list(range(STAGE1_CH)) if not stage2 else list(range(STAGE1_CH, N_CH))


def e12_cell_jobs(task, rk, v, k, chunks, prio_head):
    return [ep_job(task, c, v, rk, k, "base", ch, ["E12"], list(prio_head) + [ch, c]) for ch in chunks for c in CKPTS]


def e13_cells(task):
    """DESIGN_tmlr 3.2 targets: every E11 cell, the E12 cells around the primary k (PT k1 +- 1, PC k1 - 1..k1 + 1; the
    4 E12 roundings), E1's 3 cells x 7 variants (PT, k1 - 1..k1 + 1). Duplicates removed, order kept."""
    k1 = K1[task]
    cells = [["none", "-", 0]]
    cells += [[v, "PT", k1] for v in E11_VARS]
    for k in (k1 - 1, k1, k1 + 1):
        cells += [[v, "PT", k] for v in E1_VARS]
    for k in (k1 - 1, k1 + 1):
        cells += [[v, "PT", k] for v in E12_ROUND]
    for k in (k1 - 1, k1, k1 + 1):
        if k >= KMIN:
            cells += [[v, "PC", k] for v in E12_ROUND]
    seen, out = set(), []
    for c in cells:
        key = (c[0], c[1], int(c[2]))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def static_jobs(tasks):
    """Every job of a pod that does not depend on outcomes (the E12 initial grid at its first stage included)."""
    jobs = []
    ti = {t: i for i, t in enumerate(tasks)}
    for t in tasks:
        jobs += calib_jobs(t, (0, ti[t]))
        jobs += prep_jobs(t, (1, ti[t]))
        jobs += bias_jobs(t, (1, ti[t]))
    # E11: none first (it is every experiment's baseline), then the 8 variants chunk-major
    for ch in range(N_CH):
        for t in tasks:
            for c in CKPTS:
                exps = ["E11", "E12"] + ([] if is_mw(t) else ["E15"]) + (["E16"] if t in E16_TASKS else [])
                jobs.append(ep_job(t, c, "none", "-", 0, "base", ch, exps, (2, ch, ti[t], c), instr=e11_instr(t)))
    for ch in range(N_CH):
        for t in tasks:
            for c in CKPTS:
                for vi, v in enumerate(E11_VARS):
                    jobs.append(ep_job(t, c, v, "PT", K1[t], "base", ch, ["E11", "E12"] if v in E12_ROUND else ["E11"],
                                       (3, ch, ti[t], c, vi), instr=e11_instr(t)))
    # E12 initial grid (R1: PT at the primary k is E11's), boundary cells first, cell-major
    for t in tasks:
        g = e12_grid(t)
        for rk in ("PT", "PC"):
            ks = g[rk]
            for k in ks:
                if rk == "PT" and k == K1[t]:
                    continue
                bnd = 1 if k in (ks[0], ks[-1]) else 2
                for vi, v in enumerate(E12_ROUND):
                    jobs += e12_cell_jobs(t, rk, v, k, e12_cell_chunks(t), (4, bnd, ti[t], RKI[rk], k, vi))
    # E15 (DMC): iterations 12 (90 pairs) and 24 (60 pairs), none first
    for t in tasks:
        if is_mw(t):
            continue
        for pi, (ptag, nch) in enumerate(E15_ITERS.items()):
            for ch in range(nch):
                for c in CKPTS:
                    for vi, v in enumerate(E15_VARS):
                        jobs.append(ep_job(t, c, v, "PT", K1[t], ptag, ch, ["E15"], (5, pi, ch, ti[t], c, vi)))
    # E16 (CEM-like MPPI) on the mechanism tasks of this pod
    for t in tasks:
        if t not in E16_TASKS:
            continue
        for ch in range(N_CH):
            for c in CKPTS:
                for vi, v in enumerate(E16_VARS):
                    jobs.append(ep_job(t, c, v, "PT", K1[t], "cem", ch, ["E16"], (6, ch, ti[t], c, vi)))
    # E13: held none episodes + offline metrics on their candidate sets
    for t in tasks:
        cells = e13_cells(t)
        for c in CKPTS:
            for ch in range(N_CH):
                jobs.append({"id": f"e13_{t}_s{c}_c{ch:02d}", "type": "e13", "task": t, "ckpt": c,
                             "eps": list(range(CHUNK * ch, CHUNK * ch + CHUNK)), "seed0": SEED0["held"],
                             "stride": 4 if is_mw(t) else 10, "n_states": 8, "nseeds": 3, "cells": cells,
                             "deps": [f"prep_{t}_s{c}", f"bias_{t}_s{c}"], "prio": [7, ch, ti[t], c],
                             "cost": CHUNK * (EP_SECS[is_mw(t)] + 12.0)})
    # E15 timing (descriptive), alone at the end
    for t in tasks:
        if is_mw(t):
            continue
        jobs.append({"id": f"timing_{t}", "type": "timing", "task": t, "ckpt": 1, "k": K1[t], "seed": SEED0["smoke"] + 50,
                     "iterations": [6, 12, 24], "variants": ["none", "det", "shiter", "detshift"], "warmup": 5,
                     "steps": 60, "exclusive": True, "deps": [], "prio": [9, ti[t]], "cost": 120.0})
    ids = [j["id"] for j in jobs]
    assert len(ids) == len(set(ids)), "duplicate ids"
    return jobs


# ------------------------------------------------------------------------------------------- adaptive E12 rules
def drop(m0, mv, mw):
    return (m0 - mv) if mw else (m0 - mv) / m0


def paired_D(y0, yv, mw, nb=NB):
    """Point D and the paired-bootstrap 95% percentile CI (RandomState(0), resample matrix over the sorted pairs)."""
    n = len(y0)
    idx = np.random.RandomState(0).randint(0, n, size=(nb, n))
    D = drop(float(y0.mean()), float(yv.mean()), mw)
    Db = drop(y0[idx].mean(1), yv[idx].mean(1), mw)
    return D, [float(np.percentile(Db, 2.5)), float(np.percentile(Db, 97.5))]


class Results:
    """Episode outcomes of finished episode jobs: E[(task, ckpt, cell, ptag)][ep] = (ret, success)."""

    def __init__(self, out):
        self.out, self.E, self.loaded, self.memo = out, {}, set(), {}

    def add_file(self, path):
        if path in self.loaded:
            return
        d = json.load(gzip.open(path, "rt"))
        s = d["spec"]
        key = (s["task"], int(s["ckpt"]), cell_name(s["variant"], s.get("rk", "PT"), s["k"]), s.get("ptag", "base"))
        dd = self.E.setdefault(key, {})
        for e in d["episodes"]:
            dd[int(e["ep"])] = (float(e["ret"]), float(e["success"]))
        self.loaded.add(path)

    def add_job(self, spec):
        p = f"{self.out}/jobs/{spec['type']}/{spec['id']}.json.gz"
        if spec["type"] == "episodes" and os.path.exists(p):
            self.add_file(p)

    def arrays(self, task, cell, ptag, eps_by_ckpt):
        """Paired arrays (none, cell) over the pairs (ckpt, ep) with ep in eps_by_ckpt[ckpt]; None if any is missing."""
        mw = is_mw(task)
        y0, yv = [], []
        for c in CKPTS:
            n0, nv = self.E.get((task, c, "none", ptag), {}), self.E.get((task, c, cell, ptag), {})
            for ep in eps_by_ckpt:
                if ep not in n0 or ep not in nv:
                    return None
                y0.append(n0[ep][1] if mw else n0[ep][0])
                yv.append(nv[ep][1] if mw else nv[ep][0])
        return np.array(y0, float), np.array(yv, float)

    def stat(self, task, cell, ptag, eps, label):
        """(D, CI, n) on the pairs of eps for every checkpoint, or None while any pair is missing. Finished results
        never change, so a complete statistic is memoized."""
        key = (task, cell, ptag, label)
        if key in self.memo:
            return self.memo[key]
        a = self.arrays(task, cell, ptag, eps)
        if a is None:
            return None
        D, ci = paired_D(*a, is_mw(task))
        self.memo[key] = (D, ci, len(a[0]))
        return self.memo[key]


def adaptive_jobs(task, res, known_ids):
    """New E12 jobs implied by the finished results of `task` (two-stage stage 2, extension cells), and the decision
    records behind them. Idempotent: jobs already in known_ids are not returned again."""
    mw = is_mw(task)
    g = e12_grid(task)
    new, decisions = [], []
    eps1 = list(range(STAGE1_CH * CHUNK))
    eps_all = list(range(N_CH * CHUNK))
    for rk in ("PT", "PC"):
        for vi, v in enumerate(E12_ROUND):
            ks = set(g[rk])
            # cells added by earlier extension decisions (their jobs are known)
            for k in range(KMIN, KMAX + 1):
                if ep_id(task, 1, v, rk, k, "base", 0) in known_ids:
                    ks.add(k)

            def status(k):
                """State of cell (rk, v, k): complete, D / CI / n on all its finished pairs, and for DMC cells the
                stage-1 values (45 pairs) with the stage-2 decision (CI contains tau)."""
                cell = cell_name(v, rk, k)
                st = {"complete": False, "D": None, "ci": None, "n": 0, "need2": False, "D1": None, "ci1": None}
                if mw or (rk == "PT" and k == K1[task]):
                    r = res.stat(task, cell, "base", eps_all, "all")
                    if r is not None:
                        st.update(complete=True, D=r[0], ci=r[1], n=r[2])
                    return st
                r1 = res.stat(task, cell, "base", eps1, "stage1")
                if r1 is None:
                    return st
                D1, ci1 = r1[0], r1[1]
                need2 = bool(ci1[0] <= TAU + TOL and ci1[1] >= TAU - TOL)
                st.update(D1=D1, ci1=ci1, need2=need2)
                if not need2:
                    st.update(complete=True, D=D1, ci=ci1, n=r1[2])
                    return st
                r = res.stat(task, cell, "base", eps_all, "all")
                if r is not None:
                    st.update(complete=True, D=r[0], ci=r[1], n=r[2])
                return st

            # two-stage: stage-2 jobs of DMC cells whose stage-1 CI contains tau
            for k in sorted(ks):
                if mw or (rk == "PT" and k == K1[task]):
                    continue
                st = status(k)
                if st["need2"]:
                    js = e12_cell_jobs(task, rk, v, k, e12_cell_chunks(task, stage2=True),
                                       (4, 0, 0, RKI[rk], k, vi))
                    js = [j for j in js if j["id"] not in known_ids]
                    if js:
                        new += js
                        decisions.append({"rule": "stage2", "task": task, "rk": rk, "variant": v, "k": k,
                                          "n_pairs_stage1": 45, "D_stage1": st["D1"], "ci_stage1": st["ci1"],
                                          "jobs": len(js)})
            # extension at the low and the high end
            lo, hi = min(ks), max(ks)
            st = status(lo)
            if st["complete"] and st["D"] <= TAU + TOL and lo > KMIN:
                js = e12_cell_jobs(task, rk, v, lo - 1, e12_cell_chunks(task), (4, 0, 0, RKI[rk], lo - 1, vi))
                js = [j for j in js if j["id"] not in known_ids]
                if js:
                    new += js
                    decisions.append({"rule": "extend_down", "task": task, "rk": rk, "variant": v, "from_k": lo,
                                      "D": st["D"], "ci": st["ci"], "n_pairs": st["n"], "add_k": lo - 1,
                                      "jobs": len(js)})
            st = status(hi)
            if st["complete"] and st["D"] > TAU + TOL and hi < KMAX:
                js = e12_cell_jobs(task, rk, v, hi + 1, e12_cell_chunks(task), (4, 0, 0, RKI[rk], hi + 1, vi))
                js = [j for j in js if j["id"] not in known_ids]
                if js:
                    new += js
                    decisions.append({"rule": "extend_up", "task": task, "rk": rk, "variant": v, "from_k": hi,
                                      "D": st["D"], "ci": st["ci"], "n_pairs": st["n"], "add_k": hi + 1,
                                      "jobs": len(js)})
    for j in new:
        j["exps"] = ["E12"]
    return new, decisions


# ------------------------------------------------------------------------------------------- smoke and bench
def smoke_jobs():
    """Phase A smoke (separate out/big folders; smoke seeds; never analysed)."""
    jobs = []
    pairs = [("mw-push", 1), ("mw-pick-place", 1), ("cheetah-run", 1), ("walker-walk", 1)]
    for t, c in pairs:
        for ep in range(5):
            jobs.append({"id": f"calib_{t}_s{c}_ep{ep}", "type": "calib", "task": t, "ckpt": c, "ep": ep,
                         "seed": SEED0["calib"] + ep})
        jobs.append({"id": f"prep_{t}_s{c}", "type": "prep", "task": t, "ckpt": c, "cal_eps": [0, 1, 2, 3, 4]})
    sm = {"split": "smoke", "extra": {"log_calls": True, "eps": [0]}}     # one episode per smoke job
    for t in ("mw-push", "cheetah-run"):
        k = K1[t]
        jobs.append(ep_job(t, 1, "none", "-", 0, "base", 0, [], (0,), instr=e11_instr(t), **sm))
        for v in E11_VARS:
            jobs.append(ep_job(t, 1, v, "PT", k, "base", 0, [], (0,), instr=e11_instr(t), **sm))
        for v in E12_ROUND:
            jobs.append(ep_job(t, 1, v, "PC", k - 1, "base", 0, [], (0,), **sm))
        for v in ("none", "det", "shep"):
            jobs.append(ep_job(t, 1, v, "PT", k, "cem", 0, [], (0,), **sm))
    for v in ("none", "det"):
        jobs.append(ep_job("cheetah-run", 1, v, "PT", 5, "it12", 0, [], (0,), **sm))
        jobs.append(ep_job("cheetah-run", 1, v, "PT", 5, "it24", 0, [], (0,), **sm))
    # job-path reproduction of check_e1_paths.py (same seed 12000, E0 ranges): runner job == direct run_episode
    for v in ("none", "det", "shiter", "shep", "detshift"):
        jobs.append(ep_job("mw-push", 1, v, "PT", 3, "base", 0, [], (0,), split="repro",
                           extra={"eps": [0], "id": f"repro_mw-push_s1_{cell_name(v, 'PT', 3)}"}))
    for t in ("mw-push", "cheetah-run"):
        jobs.append({"id": f"e13smoke_{t}_s1", "type": "e13", "task": t, "ckpt": 1, "eps": [0],
                     "seed0": SEED0["smoke"] + 10, "stride": 4 if is_mw(t) else 10, "n_states": 8, "nseeds": 3,
                     "cells": e13_cells(t)})
    jobs.append({"id": "timingsmoke_cheetah-run", "type": "timing", "task": "cheetah-run", "ckpt": 1, "k": 5,
                 "seed": SEED0["smoke"] + 50, "iterations": [6, 12, 24], "variants": ["none", "det"], "warmup": 3,
                 "steps": 10})
    for j in jobs:
        for kk in ("deps", "prio", "cost", "exps"):
            j.pop(kk, None)
    return jobs


def bench_jobs(part, n=60):
    """Throughput set (bench seeds 108000+, 1-episode jobs, checkpoint 1, needs the smoke prep of these tasks):
    mw: mw-push, mw-pick-place; dmc: cheetah-run, walker-walk. The E11 / E12 variant mix of the main run (PT at the
    primary k: 8 variants + none; PC at k - 1: det, sh-iter), E11 instrumentation on the mechanism tasks; dmc also
    iterations 12 / 24 (none, det)."""
    tasks = ["mw-push", "mw-pick-place"] if part == "mw" else ["cheetah-run", "walker-walk"]
    jobs = []
    for e in range(n):
        for t in tasks:
            k = K1[t]
            cells = [("none", "-", 0, "base")] + [(v, "PT", k, "base") for v in E11_VARS] + \
                    [(v, "PC", k - 1, "base") for v in ("det", "shiter")]
            if part == "dmc":
                cells += [("none", "-", 0, "it12"), ("det", "PT", k, "it12"), ("none", "-", 0, "it24"),
                          ("det", "PT", k, "it24")]
            for v, rk, kk, ptag in cells:
                # instrumented on one task of each pair only (the main run instruments E11 on 3 of 12 tasks)
                instr = e11_instr(t) if ptag == "base" and rk != "PC" and t in ("mw-push", "cheetah-run") else {}
                j = ep_job(t, 1, v, rk, kk, ptag, 0, [], (0,), instr=instr, split="bench",
                           extra={"eps": [e]})
                j["id"] = f"bench_{t}_s1_{cell_name(v, rk, kk)}_{ptag}_e{e:03d}"
                for x in ("deps", "prio", "cost", "exps"):
                    j.pop(x, None)
                jobs.append(j)
    return jobs


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["static", "smoke", "bench", "count", "none"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--part", default="mw")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.cmd == "count":
        from collections import Counter
        for tag, ts in POD_TASKS.items():
            js = static_jobs(ts)
            eps = sum(len(j.get("eps", [])) for j in js if j["type"] in ("episodes", "e13"))
            print(tag, len(js), dict(Counter(j["type"] for j in js)), "episodes", eps,
                  "cost_h_at_8w", round(sum(j["cost"] for j in js) / 8 / 3600, 2))
        raise SystemExit(0)
    if a.cmd == "none":
        # the default-planner none jobs of a pod in exactly the form the driver submits them (check 3 before the
        # driver starts; the driver later skips the existing results)
        js = []
        for j in static_jobs(POD_TASKS[a.tag]):
            if j["type"] == "episodes" and j["variant"] == "none" and j["ptag"] == "base":
                spec = {k: v for k, v in j.items() if k not in ("deps", "prio", "cost", "exps", "exclusive")}
                spec["exps"] = j.get("exps", [])
                js.append(spec)
    else:
        js = {"static": lambda: static_jobs(POD_TASKS[a.tag]), "smoke": smoke_jobs,
              "bench": lambda: bench_jobs(a.part, a.n)}[a.cmd]()
    with open(a.out, "w") as f:
        for j in js:
            f.write(json.dumps(j) + "\n")
    print(f"{a.cmd}: {len(js)} jobs -> {a.out}")
