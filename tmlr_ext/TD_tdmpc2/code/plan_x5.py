"""Job plan of X5 (tmlr_ext/DESIGN_tmlr.md 11.1, D6; post hoc, frozen before data in commit 1d73e10): the bit curve of
det+noise on Meta-World, pod td.7 (A40). Same code paths and job specs as the td main run (plan_td.ep_job), only the
cell list differs:
  none      default planner, 90 pairs (checkpoints 1-3 x seeds 100000-100029), E11's instrumentation on mw-push /
            mw-pick-place (so the jobs are identical to E11's none jobs; PlanProbe never changes the closed loop)
  det+noise PT k = 1..5 and PC k = 1..3, 90 pairs per cell, no instrumentation except the E11 primary-k PT cell of
            mw-push / mw-pick-place (identical to E11's job spec: the reproduction check of D6)
  ranges    PT = E0 files (code/inputs/ranges); PC = the files re-derived on td.5 / td.6, copied unchanged into
            <out>/prep/<task>_s<ckpt>.npz (md5 checked), so no calibration or prep job runs here
  extension E12's rule upwards: if the highest measured k of (task, range kind) has D > tau = 0.10 (all 90 pairs,
            tolerance 1e-9), the next k is added, up to k = 8 (the grids start at k = 1, so there is no downward step)
  order     none first, then the highest grid k of each (task, range kind) (input of the extension rule), then the rest
"""
import plan_td as P
from plan_td import Results  # noqa: F401  (the driver uses P.Results)

PT_KS = [1, 2, 3, 4, 5]
PC_KS = [1, 2, 3]
GRID = {"PT": PT_KS, "PC": PC_KS}


def _cell_jobs(task, rk, k, prio_head):
    instr = P.e11_instr(task) if (rk == "PT" and k == P.K1[task]) else {}
    js = []
    for ch in range(P.N_CH):
        for c in P.CKPTS:
            j = P.ep_job(task, c, "detnoise", rk, k, "base", ch, ["X5"], list(prio_head) + [ch, c], instr=instr)
            j["deps"] = []            # PC ranges are pre-placed copies (no prep job on this pod)
            js.append(j)
    return js


def static_jobs(tasks):
    ti = {t: i for i, t in enumerate(tasks)}
    jobs = []
    for ch in range(P.N_CH):
        for t in tasks:
            for c in P.CKPTS:
                j = P.ep_job(t, c, "none", "-", 0, "base", ch, ["X5"], (2, ch, ti[t], c), instr=P.e11_instr(t))
                j["deps"] = []
                jobs.append(j)
    for t in tasks:
        for rk, ks in GRID.items():
            for k in ks:
                first = 0 if k == ks[-1] else 1
                jobs += _cell_jobs(t, rk, k, (3, first, ti[t], P.RKI[rk], k))
    ids = [j["id"] for j in jobs]
    assert len(ids) == len(set(ids))
    return jobs


def adaptive_jobs(task, res, known_ids):
    """E12's upward extension for det+noise: highest measured k complete with D > tau -> add k + 1 (up to 8)."""
    new, decisions = [], []
    eps_all = list(range(P.N_CH * P.CHUNK))
    for rk, ks0 in GRID.items():
        ks = set(ks0) | {k for k in range(P.KMIN, P.KMAX + 1)
                         if P.ep_id(task, 1, "detnoise", rk, k, "base", 0) in known_ids}
        hi = max(ks)
        r = res.stat(task, P.cell_name("detnoise", rk, hi), "base", eps_all, "all")
        if r is not None and r[0] > P.TAU + P.TOL and hi < P.KMAX:
            js = [j for j in _cell_jobs(task, rk, hi + 1, (3, 0, 0, P.RKI[rk], hi + 1)) if j["id"] not in known_ids]
            if js:
                new += js
                decisions.append({"rule": "extend_up", "task": task, "rk": rk, "variant": "detnoise", "from_k": hi,
                                  "D": r[0], "ci": r[1], "n_pairs": r[2], "add_k": hi + 1, "jobs": len(js)})
    return new, decisions
