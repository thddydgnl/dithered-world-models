"""Job manifests for E9 (rss_ext/DESIGN_ext.md 2: NVFP4 / MXFP4 fake quantization on the E1 TD-MPC2 setting).

Seeds (env, random, numpy and torch seeded with the episode seed; the same seeds for every checkpoint and variant, so
cells are paired; the dither generator is seeded with 7 seed + 13 as in E1):
  eval   80000-80029  closed-loop episodes of none and the 4 FP4 variants (DESIGN_ext 0.4), checkpoints 1-3 -> 90 pairs
  held   81000-81029  unquantized episodes that provide the planning-time inputs of the 1-step errors (pt_err1h)
  smoke  89000-89099  smoke tests;  bench 88000-...: throughput (never analysed)
  repro  12000        E0 held seeds: reproduction of stored E0 episodes (none, uniform det) with this code on the pod
Subcommands (each writes one JSONL manifest):
  main   --tasks .. [--eps 30] [--chunk 5] [--impl triton]   eval episode jobs (chunk-major, DMC first inside a
                                                              chunk), then the heldperr jobs
  smoke  [--impl_ref]                                         repro + FP4 smoke episodes (both implementations) + heldperr
  bench  --part mw|dmc [--eps 40]                             one-episode jobs of every variant (throughput)
Usage: python make_manifest_e9.py main --tasks mw-push,... --out manifests/main_a.jsonl
"""
import argparse
import json

SEED0 = {"eval": 80000, "held": 81000, "smoke": 89000, "bench": 88000, "repro": 12000}
MW8 = ["mw-push", "mw-pick-place", "mw-door-open", "mw-drawer-open", "mw-button-press-topdown",
       "mw-peg-insert-side", "mw-window-open", "mw-sweep-into"]
DMC4 = ["cheetah-run", "walker-walk", "quadruped-walk", "finger-turn-hard"]
FP4 = ["nvfp4_det", "nvfp4_shiter", "mxfp4_det", "mxfp4_shiter"]
VARS = ["none"] + FP4


def is_mw(task):
    return task.startswith("mw-")


def held_instr(task):
    """E1 instrumentation density for the planning-time inputs: OEI states every 4th (MW) / 10th (DMC) control step,
    8 of them per episode for the planning-time errors, 3 dither seeds (make_manifest.instr of E1)."""
    return {"oei_stride": 4 if is_mw(task) else 10, "perr_plan_states": 8, "perr_seeds": 3}


def ep_job(split, task, ckpt, v, eps, ch, impl="triton", tag="", extra=None):
    j = {"id": f"{split}_{task}_s{ckpt}_{v}{tag}_c{ch:02d}", "type": "episodes", "split": split, "task": task,
         "ckpt": int(ckpt), "k": 0 if v == "none" else 4, "variant": v, "eps": list(eps), "seed0": SEED0[split],
         "scales": "e1", "planner": {}, "instr": {}, "fp4_impl": impl}
    j.update(extra or {})
    return j


def held_job(split, task, ckpt, eps, ch, impl="triton", seed0=None):
    return {"id": f"{split}perr_{task}_s{ckpt}_c{ch:02d}", "type": "heldperr", "split": split, "task": task,
            "ckpt": int(ckpt), "eps": list(eps), "seed0": SEED0["held"] if seed0 is None else seed0, "scales": "e1",
            "variants": FP4, "instr": held_instr(task), "fp4_impl": impl}


def chunks(n, size):
    return [list(range(s, min(n, s + size))) for s in range(0, n, size)]


def dmc_first(tasks):
    return [t for t in tasks if not is_mw(t)] + [t for t in tasks if is_mw(t)]


def m_main(a):
    tasks = dmc_first(a.tasks)
    jobs = []
    for ch, eps in enumerate(chunks(a.eps, a.chunk)):
        for task in tasks:
            for c in a.ckpts:
                for v in VARS:
                    jobs.append(ep_job("eval", task, c, v, eps, ch, a.impl))
    for ch, eps in enumerate(chunks(a.eps, a.chunk)):
        for task in tasks:
            for c in a.ckpts:
                jobs.append(held_job("held", task, c, eps, ch, a.impl))
    return jobs


def m_smoke(a):
    jobs = []
    # E0 reproduction (stored E0 held episodes, A5000): none and uniform det with the E1 ranges, episode seed 12000
    for task, k in (("mw-push", 3), ("cheetah-run", 4)):
        jobs.append({"id": f"repro_{task}_s1_none", "type": "episodes", "split": "repro", "task": task, "ckpt": 1,
                     "k": 0, "variant": "none", "eps": [0], "seed0": SEED0["repro"], "scales": "e1", "planner": {},
                     "instr": {}})
        jobs.append({"id": f"repro_{task}_s1_k{k}_det", "type": "episodes", "split": "repro", "task": task, "ckpt": 1,
                     "k": k, "variant": "det", "eps": [0], "seed0": SEED0["repro"], "scales": "e1", "planner": {},
                     "instr": {}})
    # FP4 smoke: every variant, both implementations (same seed -> the same episode if the two agree bit for bit)
    for task in ("mw-push", "cheetah-run"):
        for v in FP4:
            for impl in ("triton", "ref"):
                jobs.append(ep_job("smoke", task, 1, v, [0], 0, impl, tag=f"_{impl}", extra={"log_calls": True}))
    for task in ("mw-push", "cheetah-run"):
        jobs.append(held_job("smoke", task, 1, [1], 0, "triton", seed0=SEED0["smoke"]))
    return jobs


def m_bench(a):
    tasks = ["mw-push", "mw-pick-place"] if a.part == "mw" else ["cheetah-run", "walker-walk"]
    jobs = []
    for e in range(a.eps):
        for task in tasks:
            for v in VARS:
                jobs.append(ep_job("bench", task, 1, v, [e], e, a.impl))
    return jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["main", "smoke", "bench"])
    ap.add_argument("--tasks", default=",".join(MW8 + DMC4))
    ap.add_argument("--ckpts", default="1,2,3")
    ap.add_argument("--eps", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--impl", default="triton", choices=["triton", "ref"])
    ap.add_argument("--part", default="mw", choices=["mw", "dmc"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.tasks = [t for t in a.tasks.split(",") if t]
    unknown = [t for t in a.tasks if t not in MW8 + DMC4]
    if unknown:
        raise SystemExit(f"unknown tasks {unknown}")
    a.ckpts = [int(c) for c in a.ckpts.split(",")]
    jobs = {"main": m_main, "smoke": m_smoke, "bench": m_bench}[a.cmd](a)
    ids = [j["id"] for j in jobs]
    assert len(ids) == len(set(ids)), "duplicate job ids"
    with open(a.out, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"{a.cmd}: {len(jobs)} jobs -> {a.out}")
