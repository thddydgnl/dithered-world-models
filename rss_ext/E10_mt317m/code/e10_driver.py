"""E10 driver: resumable stage machine of DESIGN_ext section 3 (mt80-317M) for the tasks assigned to one pod.

Per task, in this order (every decision is derived from finished result files only, so a restarted driver - on the same
pod or on another pod holding a copy of <out> - continues where it stopped):
  held    none episodes on the held seeds 91000-91029 (6 jobs x 5 episodes) with mtlib.HeldRecorder
  qualify held none success rate >= 0.7 (- 1e-9)                                                   (design 3.3)
  ranges  per-layer [min, 99.99th pct] of the held planning forwards (E0 method)                   (design 3.2)
  ksel    det at k = 5, 4, 3, 2 in this order on the held seeds (6 jobs x 5 episodes per k); the first k with
          drop = held none rate - held det rate >= 0.20 (- 1e-9) is the primary k and the search stops; no such k
          down to k = 2 -> 'no damage at >= 2 bits' (no eval, out of G2)                           (design 3.4)
  eval    none (as soon as the task qualifies; it does not depend on k), det and sh-iter at the primary k on the eval
          seeds 90000-90089 (9 jobs x 10 episodes per variant)                                      (design 3.5)
  perr    pt_err1h of det and sh-iter at the primary k on the held inputs                          (design 3.5)
Scheduling priority: held / ranges, then ksel, then perr and eval det / sh-iter, then eval none (fills idle slots).
The fallback list of design 3.3 (fewer than 4 qualified candidates) spans both pods; the coordinator of this experiment
(the E10 worker) adds those tasks by hand with --tasks, in list order.
usage: python e10_driver.py --out /workspace/e10 --tasks mw-push,mw-pick-place,... --workers 2 [--dry]
"""
import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

import e10_jobs as J

HELD_SEEDS = list(range(91000, 91030))
EVAL_SEEDS = list(range(90000, 90090))
TOL = 1e-9
QUAL = 0.7
DROP = 0.20
K_ORDER = (5, 4, 3, 2)
HELD_CHUNK = 5
EVAL_CHUNK = 10
STAGE_RANK = {"held": 0, "ranges": 0, "ksel": 1, "perr": 2, "eval_q": 2, "eval_none": 3}


def chunks(seeds, n):
    return [seeds[i:i + n] for i in range(0, len(seeds), n)]


def held_specs(task):
    return [{"id": f"held_{task}_c{i}", "stage": "held", "type": "episodes", "task": task, "variant": "none", "k": 0,
             "seeds": s, "record": True, "chunk": i} for i, s in enumerate(chunks(HELD_SEEDS, HELD_CHUNK))]


def ranges_spec(task):
    return {"id": f"ranges_{task}", "stage": "ranges", "type": "ranges", "task": task, "chunk": 0}


def ksel_specs(task, k):
    return [{"id": f"ksel_{task}_k{k}_c{i}", "stage": "ksel", "type": "episodes", "task": task, "variant": "det",
             "k": k, "seeds": s, "chunk": i} for i, s in enumerate(chunks(HELD_SEEDS, HELD_CHUNK))]


def eval_specs(task, v, k):
    return [{"id": f"eval_{task}_{v}_k{k}_c{i}", "stage": "eval", "type": "episodes", "task": task, "variant": v,
             "k": k, "seeds": s, "chunk": i} for i, s in enumerate(chunks(EVAL_SEEDS, EVAL_CHUNK))]


def perr_spec(task, k):
    return {"id": f"perr_{task}_k{k}", "stage": "perr", "type": "perr", "task": task, "k": k, "chunk": 0}


class Disk:
    def __init__(self, out):
        self.out, self.cache = out, {}

    def done(self, spec):
        return os.path.exists(J.result_path(self.out, spec))

    def load(self, spec):
        p = J.result_path(self.out, spec)
        if p not in self.cache:
            self.cache[p] = J.read_json_gz(p)
        return self.cache[p]

    def successes(self, specs):
        """{seed: success} if every job of the list is finished, else None."""
        if not all(self.done(s) for s in specs):
            return None
        out = {}
        for s in specs:
            for e in self.load(s)["episodes"]:
                out[int(e["seed"])] = float(e["success"])
        return out


def task_plan(dk, task):
    """(ready specs, status) of one task from the finished results."""
    st = {"task": task}
    H = held_specs(task)
    hs = dk.successes(H)
    if hs is None:
        st["phase"] = "held"
        st["held_done"] = sum(dk.done(s) for s in H)
        return [s for s in H if not dk.done(s)], st
    assert sorted(hs) == HELD_SEEDS
    rate = float(np.mean([hs[s] for s in HELD_SEEDS]))
    st["held_none_success"] = rate
    st["qualified"] = bool(rate >= QUAL - TOL)
    if not st["qualified"]:
        st["phase"] = "not qualified"
        return [], st
    ready = [dict(s, prio="eval_none") for s in eval_specs(task, "none", 0) if not dk.done(s)]
    R = ranges_spec(task)
    if not dk.done(R):
        st["phase"] = "ranges"
        return ready + [R], st
    st["ksel"] = {}
    primary = None
    for k in K_ORDER:
        S = ksel_specs(task, k)
        ds = dk.successes(S)
        if ds is None:
            st["phase"] = f"ksel k={k}"
            return ready + [s for s in S if not dk.done(s)], st
        assert sorted(ds) == HELD_SEEDS
        det_rate = float(np.mean([ds[s] for s in HELD_SEEDS]))
        drop = rate - det_rate
        st["ksel"][k] = {"det_success": det_rate, "drop": drop}
        if drop >= DROP - TOL:
            primary = k
            break
    st["primary_k"] = primary
    if primary is None:
        st["phase"] = "no damage at >= 2 bits"
        return ready, st       # eval none (already started) completes; nothing else
    for v in ("det", "shiter"):
        ready += [dict(s, prio="eval_q") for s in eval_specs(task, v, primary) if not dk.done(s)]
    P = perr_spec(task, primary)
    if not dk.done(P):
        ready.append(P)
    st["eval_done"] = {v: sum(dk.done(s) for s in eval_specs(task, v, 0 if v == "none" else primary))
                       for v in ("none", "det", "shiter")}
    st["perr_done"] = dk.done(P)
    st["phase"] = "eval" if ready else "done"
    return ready, st


def prio(spec, order):
    rank = STAGE_RANK[spec.get("prio", spec["stage"])]
    return (rank, order.index(spec["task"]), spec.get("k", 0) * -1, spec["chunk"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    tasks = a.tasks.split(",")
    dk = Disk(a.out)
    t0 = time.time()
    log = lambda msg: print(f"{time.strftime('%H:%M:%S', time.gmtime())} {time.time() - t0:7.0f}s {msg}", flush=True)
    if a.dry:
        for t in tasks:
            r, s = task_plan(dk, t)
            print(t, s, [x["id"] for x in r][:8], len(r))
        return
    ex = cf.ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn"), initializer=J.init_worker)
    inflight, failed, n_ok = {}, set(), 0
    log(f"driver start: tasks {tasks}, workers {a.workers}, out {a.out}")
    while True:
        ready, status = [], {}
        for t in tasks:
            r, s = task_plan(dk, t)
            ready += r
            status[t] = s
        ready = [s for s in ready if s["id"] not in inflight and s["id"] not in failed]
        ready.sort(key=lambda s: prio(s, tasks))
        while len(inflight) < a.workers + 1 and ready:
            s = ready.pop(0)
            inflight[s["id"]] = ex.submit(J.run_job, s, a.out)
        json.dump({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tasks": status,
                   "inflight": sorted(inflight), "failed": sorted(failed), "n_ok": n_ok},
                  open(f"{a.out}/state.json.tmp", "w"), indent=1)
        os.replace(f"{a.out}/state.json.tmp", f"{a.out}/state.json")
        if not inflight:
            break
        done, _ = cf.wait(list(inflight.values()), timeout=900, return_when=cf.FIRST_COMPLETED)
        for jid in [j for j, f in inflight.items() if f in done]:
            try:
                r = inflight.pop(jid).result()
            except cf.process.BrokenProcessPool as e:   # a worker died (e.g. OOM / segfault): restart the driver
                log(f"BROKEN_POOL at {jid}: {e!r}; unfinished jobs stay pending, restart the driver")
                return 2
            if r.get("skipped"):
                continue
            if r["ok"]:
                n_ok += 1
            else:
                failed.add(jid)
            log(f"{'OK ' if r['ok'] else 'ERR'} {jid} ({r['secs']:.0f}s) {r['brief']}")
    ex.shutdown()
    log(f"DRIVER_DONE ok {n_ok} failed {len(failed)} {sorted(failed)}")
    for t in tasks:
        log(f"{t}: {json.dumps(status[t])}")


if __name__ == "__main__":
    sys.exit(main())
