"""Queue driver of one td pod (WORKER_RULES 3): runs every job of plan_td for the pod's tasks with W spawned workers
(runner.run_job: one result file per job, atomic write, existing results skipped), applies the adaptive E12 rules
(plan_td.adaptive_jobs: DMC two-stage sampling, k extension) from the finished results, and keeps a one-line status.

- single instance: an exclusive flock on <logs>/driver.lock; restart.sh may be called any number of times
- resumable: finished jobs are the result files; adaptive jobs are re-derived from the results at start
- failures: a failed job is re-queued up to 2 times per driver session (3 attempts); a new session (restart) gives
  the failed jobs 2 more retries; errors are in <out>/errors/<id>.txt and <logs>/driver.log
- a crashed worker pool (BrokenProcessPool) or a job hung beyond its time limit restarts the pool; the jobs in flight
  are re-queued (one attempt counted)
- 'exclusive' jobs (E15 timing) run alone after everything else
- decisions of the adaptive rules are appended to <out>/driver/decisions.jsonl
Usage: python driver.py --conf /workspace/atm/driver_conf.json   (conf: tag, tasks, workers, out, big, logs)
"""
import argparse
import concurrent.futures as cf
import fcntl
from concurrent.futures.process import BrokenProcessPool
import glob
import json
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plan_td as P  # noqa: E402

MAX_ATTEMPTS = 3


def now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json_atomic(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def _run(args):
    import runner
    return runner.run_job(args)


class Driver:
    def __init__(self, conf):
        self.c = conf
        self.out, self.big, self.logs = conf["out"], conf["big"], conf["logs"]
        self.W = int(conf.get("workers", 8))
        for d in (self.out, self.big, self.logs, f"{self.out}/driver"):
            os.makedirs(d, exist_ok=True)
        self.log_f = open(f"{self.logs}/driver.log", "a", buffering=1)
        self.jobs = {}
        for j in P.static_jobs(conf["tasks"]):
            self.jobs[j["id"]] = j
        self.res = P.Results(self.out)
        self.done = set()
        self.attempts, self.failed = {}, set()
        self.inflight = {}
        self.hist = deque()
        self.t0 = time.time()
        self.n_ok = self.n_err = 0
        self.note = ""

    def log(self, msg):
        self.log_f.write(f"{now_utc()} {msg}\n")

    def result_path(self, j):
        return f"{self.out}/jobs/{j['type']}/{j['id']}.json.gz"

    def scan_done(self):
        for jid, j in self.jobs.items():
            if jid not in self.done and os.path.exists(self.result_path(j)):
                self.done.add(jid)
                self.res.add_job(j)

    def adapt(self, tasks=None):
        """Closure of the adaptive rules over the current results (adds jobs until nothing new is implied)."""
        added = 0
        for t in (tasks or self.c["tasks"]):
            while True:
                new, dec = P.adaptive_jobs(t, self.res, set(self.jobs))
                if not new:
                    break
                for j in new:
                    self.jobs[j["id"]] = j
                    if os.path.exists(self.result_path(j)):
                        self.done.add(j["id"])
                        self.res.add_job(j)
                with open(f"{self.out}/driver/decisions.jsonl", "a") as f:
                    for d in dec:
                        d["time"] = now_utc()
                        f.write(json.dumps(d, default=float) + "\n")
                for d in dec:
                    self.log(f"ADAPT {d['rule']} {d['task']} {d['rk']} {d['variant']} k="
                             f"{d.get('k', d.get('add_k'))} jobs={d['jobs']}")
                added += len(new)
        return added

    def gate_open(self, g):
        return os.path.exists(f"{self.c.get('atm', '/workspace/atm')}/GATE_{g}")

    def ready(self):
        out = []
        for jid, j in self.jobs.items():
            if jid in self.done or jid in self.inflight or jid in self.failed:
                continue
            if j.get("gate") and not self.gate_open(j["gate"]):
                continue
            if all(d in self.done for d in j.get("deps", [])):
                out.append(j)
        out.sort(key=lambda j: j["prio"])
        return out

    def gated_waiting(self):
        """Not-done jobs that wait for a closed gate, directly or through their dependencies."""
        closed = {jid for jid, j in self.jobs.items() if jid not in self.done and j.get("gate")
                  and not self.gate_open(j["gate"])}
        wait = set(closed)
        changed = True
        while changed:
            changed = False
            for jid, j in self.jobs.items():
                if jid not in wait and jid not in self.done and any(d in wait for d in j.get("deps", [])):
                    wait.add(jid)
                    changed = True
        return wait

    def blocked(self):
        """Jobs that can never run in this session (a dependency failed)."""
        bad = set(self.failed)
        changed = True
        while changed:
            changed = False
            for jid, j in self.jobs.items():
                if jid not in bad and jid not in self.done and any(d in bad for d in j.get("deps", [])):
                    bad.add(jid)
                    changed = True
        return bad - set(self.failed)

    def state(self, phase="main", driver="alive"):
        rem = [j for jid, j in self.jobs.items() if jid not in self.done and jid not in self.failed]
        cost_rem = sum(j.get("cost", 0.0) for j in rem)
        now = time.time()
        while self.hist and now - self.hist[0][0] > 1800:
            self.hist.popleft()
        rate = None
        if len(self.hist) >= 5:
            span = max(now - self.hist[0][0], 60.0)
            rate = sum(c for _, c in self.hist) / span          # cost units per wall second
        eta = "-"
        if rem and rate:
            eta = time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime(now + cost_rem / rate))
        elif not rem:
            eta = "done"
        gw = len(self.gated_waiting())
        st = {"tag": self.c["tag"], "phase": phase, "done": len(self.done), "total": len(self.jobs), "gated": gw,
              "failed": len(self.failed), "inflight": len(self.inflight), "driver": driver, "eta": eta,
              "updated": now_utc(), "pid": os.getpid(), "errors": self.n_err, "ok_session": self.n_ok,
              "cost_remaining_ws": round(cost_rem, 1), "rate_ws_per_s": rate, "note": self.note,
              "blocked": len(self.blocked()), "workers": self.W}
        write_json_atomic(f"{self.logs}/driver_state.json", st)
        return st

    def new_pool(self):
        return cf.ProcessPoolExecutor(max_workers=self.W, mp_context=mp.get_context("spawn"))

    def kill_pool(self, pool):
        procs = list(getattr(pool, "_processes", {}).values())
        for p in procs:
            try:
                p.kill()
            except Exception:
                pass
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    def run(self):
        self.scan_done()
        n_ad = self.adapt()
        self.log(f"START tag={self.c['tag']} tasks={self.c['tasks']} workers={self.W} jobs={len(self.jobs)} "
                 f"done={len(self.done)} adaptive_rederived={n_ad}")
        with open(f"{self.logs}/phase", "w") as f:
            f.write("main\n")
        pool = self.new_pool()
        last_state = 0.0
        try:
            while True:
                rd = self.ready()
                normal = [j for j in rd if not j.get("exclusive")]
                excl = [j for j in rd if j.get("exclusive")]
                excl_running = any(self.jobs[i].get("exclusive") for i in self.inflight)
                while normal and len(self.inflight) < self.W and not excl_running:
                    j = normal.pop(0)
                    self.submit(pool, j)
                if not normal and not self.inflight and excl:
                    self.submit(pool, excl[0])
                if not self.inflight:
                    if not self.ready():
                        gw = self.gated_waiting()
                        if not gw:
                            break
                        self.note = f"waiting for gate(s) {sorted({self.jobs[i]['gate'] for i in gw if self.jobs[i].get('gate')})}: {len(gw)} jobs"
                        if time.time() - last_state > 10:
                            self.state()
                            last_state = time.time()
                        time.sleep(10)
                        continue
                # wait for a completion
                done_f, _ = cf.wait([v[0] for v in self.inflight.values()], timeout=5.0,
                                    return_when=cf.FIRST_COMPLETED)
                broken = False
                for fut in done_f:
                    jid = next(i for i, v in self.inflight.items() if v[0] is fut)
                    j, t_sub = self.inflight.pop(jid)[1:]
                    try:
                        r = fut.result()
                    except BrokenProcessPool:
                        broken = True
                        self.inflight[jid] = (fut, j, t_sub)   # handled below with the others
                        continue
                    except Exception:
                        r = {"id": jid, "ok": False, "secs": time.time() - t_sub,
                             "brief": traceback.format_exc().strip().splitlines()[-1]}
                    self.finish(j, r)
                # hung jobs
                tnow = time.time()
                hung = [i for i, v in self.inflight.items()
                        if tnow - v[2] > max(1800.0, 8.0 * v[1].get("cost", 100.0))]
                if broken or hung:
                    self.log(f"POOL RESTART broken={broken} hung={hung} inflight={list(self.inflight)}")
                    self.kill_pool(pool)
                    for jid in list(self.inflight):
                        self.count_failure(jid, "pool restart (broken pool or hung job)")
                    self.inflight.clear()
                    pool = self.new_pool()
                if time.time() - last_state > 10:
                    self.state()
                    last_state = time.time()
        finally:
            self.kill_pool(pool)
        bl = self.blocked()
        rem = [jid for jid in self.jobs if jid not in self.done]
        if not rem:
            self.note = "all jobs done"
            st = self.state(phase="done", driver="finished")
            with open(f"{self.logs}/phase", "w") as f:
                f.write("done\n")
            open(f"{self.logs}/DRIVER_FINISHED", "w").write(now_utc() + "\n")
        else:
            self.note = f"stopped: {len(self.failed)} failed, {len(bl)} blocked"
            st = self.state(phase="main", driver="finished")
        self.log(f"END done={len(self.done)}/{len(self.jobs)} failed={len(self.failed)} blocked={len(bl)} "
                 f"ok_session={self.n_ok} errors={self.n_err} secs={time.time() - self.t0:.0f}")
        return st

    def submit(self, pool, j):
        spec = {k: v for k, v in j.items() if k not in ("deps", "prio", "cost", "exps", "exclusive", "gate")}
        spec["exps"] = j.get("exps", [])
        fut = pool.submit(_run, (spec, self.out))
        self.inflight[j["id"]] = (fut, j, time.time())

    def count_failure(self, jid, why):
        self.attempts[jid] = self.attempts.get(jid, 0) + 1
        self.n_err += 1
        if self.attempts[jid] >= MAX_ATTEMPTS:
            self.failed.add(jid)
            self.log(f"FAILED (attempts {self.attempts[jid]}) {jid}: {why}")

    def finish(self, j, r):
        jid = j["id"]
        el = time.time() - self.t0
        if r.get("ok"):
            if os.path.exists(self.result_path(j)):
                self.done.add(jid)
                self.n_ok += 0 if r.get("skipped") else 1
                self.hist.append((time.time(), j.get("cost", 0.0)))
                self.res.add_job(j)
                if j["type"] == "episodes" and j.get("ptag") == "base":
                    self.adapt([j["task"]])
                self.log(f"{el:8.0f}s [{len(self.done)}/{len(self.jobs)}] OK  {jid} ({r.get('secs', 0):.0f}s) "
                         f"{r.get('brief', '')}")
            else:   # claimed by another runner (should not happen with one driver): re-queue later
                self.log(f"SKIP {jid}: {r.get('brief')}")
                self.count_failure(jid, "skipped without a result")
        else:
            self.count_failure(jid, r.get("brief", ""))
            self.log(f"{el:8.0f}s ERR {jid} ({r.get('secs', 0):.0f}s) {r.get('brief', '')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True)
    a = ap.parse_args()
    conf = json.load(open(a.conf))
    os.makedirs(conf["logs"], exist_ok=True)
    lockf = open(f"{conf['logs']}/driver.lock", "w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another driver holds the lock; exiting", flush=True)
        return 0
    with open(f"{conf['logs']}/driver.pid", "w") as f:
        f.write(f"{os.getpid()}\n")
    try:
        os.remove(f"{conf['logs']}/DRIVER_FINISHED")
    except FileNotFoundError:
        pass
    d = Driver(conf)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        st = d.run()
        print(json.dumps(st), flush=True)
    except KeyboardInterrupt:
        d.log("STOPPED by signal")
        d.state(phase="main", driver="dead")
    except Exception:
        d.log("DRIVER CRASH " + traceback.format_exc().replace("\n", " | "))
        d.state(phase="main", driver="dead")
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
