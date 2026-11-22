"""Job manifests of E17's closed-loop parts (tmlr_ext/DESIGN_tmlr.md 7, 0.4) for runner.py.

Seeds (env, random, numpy and torch seeded with the episode seed; the integer kernels' Philox draws are seeded per
episode from 7 seed + 13 as in E7; the same seeds for every checkpoint and precision, so cells are paired):
  eval    100000-100029  closed-loop episodes (DESIGN_tmlr 0.4: the TD-MPC2 5M seed block shared with E11/E12), ckpts 1-3
                         -> 90 pairs per cell
  calib   11000-11004    unquantized calibration episodes 0-4 of the E0 convention (per-channel ranges, DESIGN_tmlr 0.6)
  bench   108000-...     throughput bench (never analysed);  smoke 109000-...: e17_checks.py (never analysed)
Cells (MPPI default planner, 6 iterations; E0 per-tensor ranges = scales "e1"):
  part 2  mw-push, mw-pick-place, cheetah-run, walker-walk at their E1 primary k (3, 3, 5, 3):
          intk det, intk shiter, intk detshift                                       (+ the FP32 none)
  part 3  12 tasks: bf16p (pre-cast BF16)                                             (+ the FP32 none)
  part 4  the 4 part-2 tasks x k {2, 3}: intpc det, intpc shiter (per-channel ranges of the pcstats jobs)
  FP32 none (no e7 key: the unchanged model, quantizer off) of every task, shared by parts 2-4
Job = (task, ckpt, precision, k, variant, 5 episodes); id eval_<task>_s<ckpt>_<prec><k>_<variant>_c<chunk>.
Order of the main manifest: chunk-major (every cell gets its chunk 0 before any cell gets chunk 1), DMC tasks first
inside a chunk (E9 convention).
Subcommands: pre (bench + calib + FP32 none), pcstats, main (every eval job incl. FP32 none), all (writes every file)
usage: python make_manifest_e17.py all --outdir /workspace/atm/out/manifests
"""
import argparse
import json
import os

SEED0 = {"eval": 100000, "calib": 11000, "bench": 108000}
MW8 = ["mw-push", "mw-pick-place", "mw-door-open", "mw-drawer-open", "mw-button-press-topdown",
       "mw-peg-insert-side", "mw-window-open", "mw-sweep-into"]
DMC4 = ["cheetah-run", "walker-walk", "quadruped-walk", "finger-turn-hard"]
TASKS12 = MW8 + DMC4
PART2 = {"mw-push": 3, "mw-pick-place": 3, "cheetah-run": 5, "walker-walk": 3}    # E1 primary k (DESIGN_tmlr 1.2)
PART4_K = (2, 3)
CKPTS = (1, 2, 3)


def is_mw(task):
    return task.startswith("mw-")


def dmc_first(tasks):
    return [t for t in tasks if not is_mw(t)] + [t for t in tasks if is_mw(t)]


def chunks(n, size):
    return [list(range(s, min(n, s + size))) for s in range(0, n, size)]


def ep_job(split, task, ckpt, prec, k, variant, eps, ch, part):
    j = {"id": f"{split}_{task}_s{ckpt}_{prec}{k if prec in ('intk', 'intpc') else ''}_{variant}_c{ch:02d}",
         "type": "episodes", "split": split, "task": task, "ckpt": int(ckpt), "k": int(k), "variant": variant,
         "eps": list(eps), "seed0": SEED0[split], "scales": "e1", "planner": {}, "instr": {}, "part": part,
         "prec": prec}
    if prec != "fp32":
        j["e7"] = prec
    return j


def cells(task):
    """[(prec, k, variant, part)] of one task, the FP32 none first."""
    out = [("fp32", 0, "none", "none")]
    if task in PART2:
        k = PART2[task]
        out += [("intk", k, v, "2") for v in ("det", "shiter", "detshift")]
    out.append(("bf16p", 0, "none", "3"))
    if task in PART2:
        out += [("intpc", k, v, "4") for k in PART4_K for v in ("det", "shiter")]
    return out


def m_main(eps=30, chunk=5):
    jobs = []
    for ch, ee in enumerate(chunks(eps, chunk)):
        for task in dmc_first(TASKS12):
            for c in CKPTS:
                for prec, k, v, part in cells(task):
                    jobs.append(ep_job("eval", task, c, prec, k, v, ee, ch, part))
    return jobs


def m_none(eps=30, chunk=5):
    return [j for j in m_main(eps, chunk) if j["prec"] == "fp32"]


def m_calib(n_eps=5):
    jobs = []
    for task in dmc_first(list(PART2)):
        for c in CKPTS:
            for ep in range(n_eps):
                jobs.append({"id": f"calib_{task}_s{c}_ep{ep}", "type": "calib", "task": task, "ckpt": c, "ep": ep,
                             "seed": SEED0["calib"] + ep})
    return jobs


def m_pcstats(n_eps=5):
    return [{"id": f"pcstats_{task}_s{c}", "type": "pcstats", "task": task, "ckpt": c, "cal_eps": list(range(n_eps)),
             "scales": "e1"} for task in dmc_first(list(PART2)) for c in CKPTS]


def m_bench():
    """one-episode jobs of every closed-loop precision type on 2 MW and 2 DMC tasks (checkpoint 1, bench seeds)."""
    jobs = []
    precs = [("fp32", 0, "none"), ("intk", None, "det"), ("intk", None, "shiter"), ("intk", None, "detshift"),
             ("bf16p", 0, "none")]
    for e in range(4):
        for task in ("cheetah-run", "walker-walk", "mw-push", "mw-pick-place"):
            if not is_mw(task) and e >= 2:
                continue
            for prec, k, v in precs:
                kk = PART2[task] if k is None else k
                j = ep_job("bench", task, 1, prec, kk, v, [e], e, "bench")
                jobs.append(j)
    return jobs


def write(path, jobs):
    ids = [j["id"] for j in jobs]
    assert len(ids) == len(set(ids)), "duplicate job ids"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    os.replace(tmp, path)
    print(f"{os.path.basename(path)}: {len(jobs)} jobs, {sum(len(j.get('eps', [1])) for j in jobs)} episodes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all"])
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    write(f"{a.outdir}/pre.jsonl", m_bench() + m_calib() + m_none())
    write(f"{a.outdir}/pcstats.jsonl", m_pcstats())
    write(f"{a.outdir}/calib.jsonl", m_calib())
    write(f"{a.outdir}/main.jsonl", m_main())
