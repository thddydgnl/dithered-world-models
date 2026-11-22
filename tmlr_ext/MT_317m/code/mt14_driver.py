"""E14 driver: resumable job queue (DESIGN_tmlr section 4, checks 0.7; WORKER_RULES section 3) for the tasks of one pod.
Adapted from E10's e10_driver.py. Every decision is derived from finished result files only, so a restarted driver (same
pod, or a continuation pod of the same GPU type holding a copy of <out>) continues where it stopped.

Per task (primary k = E10's, R4; all of a task's episodes stay on this pod, DESIGN_tmlr 0.3):
  held      none episodes on the held seeds 103000-103009 (2 jobs x 5) with mtlib.HeldRecorder -> <big>/calib
  bias      det+BC bias vector b at k (bclib, 3 sweeps of 3-step rollouts, DESIGN_tmlr 0.6)
  eval      none, det, sh-iter, sh-ep, det+shift, det+BC on the eval seeds 102000-102089 (9 jobs x 10 per variant)
Gates (checks of DESIGN_tmlr 0.7 that can only be made with main-run data; a failed gate holds only that task's jobs):
  check 3   a task's variants start once its 90 none episodes are done and |none - E10 none| <= 0.10 (+ 1e-9);
            otherwise they are held until <out>/gate/release_check3_<task> exists (coordinator decision)
  check 2   a task's det+BC episodes start once its bias job is done and the residual after the third sweep is below
            1% of delta in every layer; otherwise held until <out>/gate/release_bc_<task> exists
Scheduling: rank held, bias, none, det, sh-iter, sh-ep, det+shift, det+BC; then task order; then chunk.
Failures: a failed job is re-queued at most twice (attempts = files <out>/errors/<id>.<n>.txt); a broken worker pool is
rebuilt (at most 5 times per run). One driver per pod (flock on <atm>/logs/driver.lock); pid in <atm>/logs/driver.pid.
Status: <out>/status.json (read by status.sh) and <out>/state.json (per-task detail), rewritten at least every 60 s.
usage: python mt14_driver.py --out /workspace/atm/out --tasks mw-push,mw-pick-place --workers 1 --tag mt.1 [--dry]
"""
import argparse
import concurrent.futures as cf
import fcntl
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

import mt14_jobs as J

ATM = os.environ.get("ATM", "/workspace/atm")
TOL = 1e-9
CHECK3_MAX = 0.10
HELD_CHUNK = 5
EVAL_CHUNK = 10
MAX_RETRY = 2
RANK = {"held": 0, "bias": 1, "none": 2, "det": 3, "shiter": 4, "shep": 5, "detshift": 6, "detbc": 7}
QVARS = ("det", "shiter", "shep", "detshift")       # plus detbc (needs the bias)


def utc(t=None):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def chunks(seeds, n):
    return [seeds[i:i + n] for i in range(0, len(seeds), n)]


def held_specs(task):
    return [{"id": f"held_{task}_c{i}", "stage": "held", "type": "episodes", "task": task, "variant": "none", "k": 0,
             "seeds": s, "record": True, "chunk": i, "rank": "held"}
            for i, s in enumerate(chunks(J.HELD_SEEDS, HELD_CHUNK))]


def bias_spec(task, k):
    return {"id": f"bias_{task}_k{k}", "stage": "bias", "type": "bias", "task": task, "k": k, "chunk": 0,
            "rank": "bias"}


def eval_specs(task, v, k):
    kk = 0 if v == "none" else int(k)
    return [{"id": f"eval_{task}_{v}_k{kk}_c{i}", "stage": "eval", "type": "episodes", "task": task, "variant": v,
             "k": kk, "seeds": s, "chunk": i, "rank": v} for i, s in enumerate(chunks(J.EVAL_SEEDS, EVAL_CHUNK))]


def all_specs(task):
    k = J.PRIMARY_K[task]
    out = held_specs(task) + [bias_spec(task, k)]
    for v in J.VARIANTS:
        out += eval_specs(task, v, k)
    return out


def gate_path(out, kind, task):
    return f"{out}/gate/{kind}_{task}.json"


def release_path(out, kind, task):
    return f"{out}/gate/release_{kind}_{task}"


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


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def task_plan(dk, out, task, e10none):
    """(ready specs, held-back notes, status) of one task from the finished results."""
    k = J.PRIMARY_K[task]
    st = {"task": task, "k": k}
    ready, hold = [], []
    H, B = held_specs(task), bias_spec(task, k)
    bias_done = dk.done(B)
    for s in H:
        files_ok = all(os.path.exists(J.calib_path(task, sd)) for sd in s["seeds"])
        if not dk.done(s):
            ready.append(s)
        elif not bias_done and not files_ok:      # continuation pod without the recordings: re-run (same GPU type)
            ready.append(dict(s, redo=True))
    calib_all = all(os.path.exists(J.calib_path(task, sd)) for sd in J.HELD_SEEDS)
    if not bias_done and all(dk.done(s) for s in H) and calib_all:
        ready.append(B)
    st["held_done"] = sum(dk.done(s) for s in H)
    # check 2 (det+BC residual) gate. D2 (2026-11-21T19:01Z): a D2 bias job opens the gate whether it converged or
    # not; an unconverged one (bc_unconverged, best b of the stages) still runs det+BC and is flagged in the status
    # note for the coordinator. A bias result of the frozen 3-sweep rule holds det+BC unless released.
    open_bc = False
    if bias_done:
        rep = dk.load(B)
        rel = os.path.exists(release_path(out, "bc", task))
        if rep.get("rule_id") == "D2":
            st["bias"] = {"rule": "D2", "residual_ok": rep["residual_ok"], "bc_unconverged": rep["bc_unconverged"],
                          "final_eta": rep["final_eta"], "n_updates": rep["final_n_updates"],
                          "resid_over_delta": rep["residual_final"]}
            open_bc = True
            if rep["bc_unconverged"]:
                hold.append(f"FLAG bc_unconverged:{task}")
        else:
            last = rep["sweeps"][-1]["layers"]
            st["bias"] = {"rule": "frozen 0.6", "residual_ok": rep["residual_ok"],
                          "resid_over_delta": {n: last[n]["mean_abs_over_delta"] for n in last}}
            open_bc = bool(rep["residual_ok"]) or rel
            if not rep["residual_ok"] and not rel:
                hold.append(f"detbc:{task}:bias_residual")
    # check 3 (none on the new seeds vs E10 none) gate
    N = eval_specs(task, "none", 0)
    ready += [s for s in N if not dk.done(s)]
    ns = dk.successes(N)
    open3 = False
    if ns is not None:
        assert sorted(ns) == J.EVAL_SEEDS
        rate = float(np.mean([ns[s] for s in J.EVAL_SEEDS]))
        ref = float(e10none[task])
        diff = rate - ref
        passed = bool(abs(diff) <= CHECK3_MAX + TOL)
        rel = os.path.exists(release_path(out, "check3", task))
        g = {"task": task, "none_new": rate, "n": len(ns), "none_e10": ref, "diff": diff, "max_abs_diff": CHECK3_MAX,
             "pass": passed, "released": rel}
        if not os.path.exists(gate_path(out, "check3", task)) or json.load(open(gate_path(out, "check3", task))) != g:
            write_json(gate_path(out, "check3", task), g)
        st["check3"] = g
        open3 = passed or rel
        if not open3:
            hold.append(f"check3:{task}:none={rate:.3f}_vs_E10={ref:.3f}")
    if open3:
        for v in QVARS:
            ready += [s for s in eval_specs(task, v, k) if not dk.done(s)]
        if open_bc:
            ready += [s for s in eval_specs(task, "detbc", k) if not dk.done(s)]
    st["eval_done"] = {v: sum(dk.done(s) for s in eval_specs(task, v, k)) for v in J.VARIANTS}
    st["jobs_done"] = sum(dk.done(s) for s in all_specs(task))
    st["jobs_total"] = len(all_specs(task))
    return ready, hold, st


def prio(spec, order):
    return (RANK[spec["rank"]], order.index(spec["task"]), spec["chunk"])


def attempts(out, jid):
    return len(J.error_files(out, jid))


def make_pool(n):
    return cf.ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("spawn"), initializer=J.init_worker)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default=os.environ.get("ATM_TAG", "?"))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    tasks = a.tasks.split(",")
    for t in tasks:
        assert t in J.TASKS8, t
        assert os.path.exists(J.ranges_path(t)), t
    pk = J.e10_primary_k()
    assert all(pk[t] == J.PRIMARY_K[t] for t in J.TASKS8), (pk, J.PRIMARY_K)
    e10none = J.e10_none()
    assert all(t in e10none for t in J.TASKS8)
    dk = Disk(a.out)
    t0 = time.time()
    log = lambda msg: print(f"{time.strftime('%H:%M:%S', time.gmtime())} {time.time() - t0:7.0f}s {msg}", flush=True)
    if a.dry:
        for t in tasks:
            r, h, s = task_plan(dk, a.out, t, e10none)
            print(t, json.dumps(s), [x["id"] for x in r][:6], len(r), h)
        return 0
    os.makedirs(f"{ATM}/logs", exist_ok=True)
    lock = open(f"{ATM}/logs/driver.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another driver holds the lock; exiting", flush=True)
        return 0
    with open(f"{ATM}/logs/driver.pid", "w") as f:
        f.write(str(os.getpid()))
    if os.path.exists(f"{a.out}/DRIVER_FINISHED"):
        os.remove(f"{a.out}/DRIVER_FINISHED")
    total = sum(len(all_specs(t)) for t in tasks)
    ex, pool_restarts = make_pool(a.workers), 0
    inflight, n_ok, ep_secs, bias_secs = {}, 0, [], []
    log(f"driver start: tag {a.tag}, tasks {tasks}, workers {a.workers}, out {a.out}, jobs {total}")
    while True:
        ready, hold, status = [], [], {}
        for t in tasks:
            r, h, s = task_plan(dk, a.out, t, e10none)
            ready += r
            hold += h
            status[t] = s
        failed = sorted({s["id"] for t in tasks for s in all_specs(t)
                         if not dk.done(s) and attempts(a.out, s["id"]) > MAX_RETRY})
        ready = [s for s in ready if s["id"] not in inflight and s["id"] not in failed]
        ready.sort(key=lambda s: prio(s, tasks))
        while len(inflight) < a.workers + 1 and ready:
            s = ready.pop(0)
            inflight[s["id"]] = (ex.submit(J.run_job, s, a.out), s, time.time())
        n_done = sum(s["jobs_done"] for s in status.values())
        # ETA from this run's per-episode time (remaining episodes of every unfinished job, gated ones included)
        rem_eps = sum(len(s.get("seeds", [])) for t in tasks for s in all_specs(t)
                      if not dk.done(s) and s["id"] not in failed and s["type"] == "episodes")
        rem_bias = sum(1 for t in tasks if not dk.done(bias_spec(t, J.PRIMARY_K[t])))
        eta = "-"
        if ep_secs:
            sec = float(np.mean(ep_secs[-30:]))
            eta = utc(time.time() + rem_eps * sec / max(1, a.workers) + rem_bias * (np.mean(bias_secs) if bias_secs else 90))
        running = sorted(inflight)[:1]
        slow = [j for j, (_, _, ts) in inflight.items() if time.time() - ts > 1800]
        note = ";".join(([h if h.startswith("FLAG ") else f"HOLD {h}" for h in hold]) + ([f"SLOW {j}" for j in slow]) +
                        ([f"failed {len(failed)}"] if failed else []) +
                        ([f"ep {np.mean(ep_secs[-30:]):.1f}s"] if ep_secs else []) +
                        ([f"run {running[0]}"] if running else []))
        finished = not inflight and not ready and not [h for h in hold if not h.startswith("FLAG ")]
        phase = "done" if finished else "main"
        write_json(f"{a.out}/status.json", {"tag": a.tag, "phase": phase, "done": n_done, "total": total,
                                             "failed": len(failed), "eta": eta, "updated": utc(), "note": note or "-",
                                             "workers": a.workers, "hold": hold, "failed_ids": failed})
        write_json(f"{a.out}/state.json", {"time": utc(), "tag": a.tag, "tasks": status, "inflight": sorted(inflight),
                                            "failed": failed, "hold": hold, "n_ok_this_run": n_ok,
                                            "pool_restarts": pool_restarts})
        waiting = [h for h in hold if not h.startswith("FLAG ")]
        if not inflight:
            if waiting and not ready:
                time.sleep(60)          # held jobs wait for a release file (coordinator decision)
                continue
            if not ready:
                break
        futs = {f: jid for jid, (f, _, _) in inflight.items()}
        done, _ = cf.wait(list(futs), timeout=60, return_when=cf.FIRST_COMPLETED)
        broken = False
        for f in done:
            jid = futs[f]
            _, spec, ts = inflight.pop(jid)
            try:
                r = f.result()
            except cf.process.BrokenProcessPool as e:
                broken = True
                os.makedirs(f"{a.out}/errors", exist_ok=True)
                n = attempts(a.out, jid) + 1
                with open(f"{a.out}/errors/{jid}.{n}.txt", "w") as fh:
                    fh.write(f"BrokenProcessPool: {e!r}\n")
                log(f"ERR {jid} BrokenProcessPool (attempt {n})")
                continue
            except Exception as e:                       # e.g. a result that cannot be unpickled
                os.makedirs(f"{a.out}/errors", exist_ok=True)
                n = attempts(a.out, jid) + 1
                with open(f"{a.out}/errors/{jid}.{n}.txt", "w") as fh:
                    fh.write(f"driver-side exception: {e!r}\n")
                log(f"ERR {jid} driver-side exception {e!r} (attempt {n})")
                continue
            if r.get("skipped"):
                continue
            if r["ok"]:
                n_ok += 1
                if spec["type"] == "episodes" and r.get("n_eps"):
                    ep_secs.append(r["secs"] / r["n_eps"])
                elif spec["type"] == "bias":
                    bias_secs.append(r["secs"])
            log(f"{'OK ' if r['ok'] else 'ERR'} {jid} ({r['secs']:.0f}s) {r['brief']}")
        if broken:
            for jid, (f, spec, ts) in list(inflight.items()):
                n = attempts(a.out, jid) + 1
                with open(f"{a.out}/errors/{jid}.{n}.txt", "w") as fh:
                    fh.write("BrokenProcessPool (in flight when the pool broke)\n")
            inflight.clear()
            ex.shutdown(wait=False, cancel_futures=True)
            pool_restarts += 1
            if pool_restarts > 5:
                log("BROKEN_POOL too often; exiting (restart.sh starts a new driver)")
                return 2
            log(f"worker pool rebuilt ({pool_restarts})")
            ex = make_pool(a.workers)
    ex.shutdown()
    n_done = sum(task_plan(dk, a.out, t, e10none)[2]["jobs_done"] for t in tasks)
    write_json(f"{a.out}/DRIVER_FINISHED", {"time": utc(), "tag": a.tag, "done": n_done, "total": total,
                                            "failed": failed})
    write_json(f"{a.out}/status.json", {"tag": a.tag, "phase": "done", "done": n_done, "total": total,
                                         "failed": len(failed), "eta": "-", "updated": utc(),
                                         "note": ";".join(["all jobs finished" if not failed else f"finished with {len(failed)} failed"]
                                                          + [h for h in hold if h.startswith("FLAG ")]),
                                         "workers": a.workers, "hold": hold, "failed_ids": failed})
    log(f"DRIVER_DONE ok {n_ok} done {n_done}/{total} failed {len(failed)} {failed}")
    for t in tasks:
        log(f"{t}: {json.dumps(status[t])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
