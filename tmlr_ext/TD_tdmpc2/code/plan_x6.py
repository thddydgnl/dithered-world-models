"""Job plan of X6 (tmlr_ext/DESIGN_tmlr.md 11.1, D7; post hoc, descriptive; frozen before data in commit 24a59a00):
the planning-free policy baseline of TD-MPC2 5M on pod td.8 (A40).
  none     default MPPI planner, FP32, 90 pairs (checkpoints 1-3 x seeds 100000-100029) per task; the same job specs
           as E11's none (E11 instrumentation on mw-push, mw-pick-place, cheetah-run), checked bit for bit against E11
  policy   TD-MPC2's mpc=False path (runner: "mpc": false): pi's mean action in eval_mode; the world model's dynamics
           and reward heads are never called (checked with hook counts, check_policy_calls.py); same 90 pairs
  tasks    the 12 E11 tasks; job = (task, ckpt, 5 consecutive episodes); no adaptive rule
"""
import plan_td as P
from plan_td import Results  # noqa: F401  (the driver uses P.Results)


def static_jobs(tasks):
    ti = {t: i for i, t in enumerate(tasks)}
    jobs = []
    for ch in range(P.N_CH):
        for t in tasks:
            for c in P.CKPTS:
                j = P.ep_job(t, c, "none", "-", 0, "base", ch, ["X6"], (2, ch, ti[t], c), instr=P.e11_instr(t))
                j["deps"] = []
                jobs.append(j)
                q = P.ep_job(t, c, "none", "-", 0, "base", ch, ["X6"], (3, ch, ti[t], c))
                q.update(id=P.ep_id(t, c, "none", "-", 0, "policy", ch), ptag="policy", mpc=False, deps=[],
                         cost=q["cost"] * 0.15)
                jobs.append(q)
    ids = [j["id"] for j in jobs]
    assert len(ids) == len(set(ids))
    return jobs


def adaptive_jobs(task, res, known_ids):
    return [], []
