"""Queue state of the E17 pod (standard-library only, fast; used by status.sh, restart.sh and driver.sh).
  status  one line: tag=<tag> phase=<..> done=<a>/<b> failed=<c> driver=<alive|dead|finished> eta=<UTC|-> updated=<UTC>
          note=<..>  (WORKER_RULES 3). done/b = result files of the current driver mode's manifests; failed = jobs with an
          error record and no result (waiting for a retry or given up after 3 attempts); eta from the episode-weighted
          completion rate of the last 30 min (DMC job weight 5 per episode, Meta-World 1)
  left    number of jobs of the current mode without a result that may still run (attempts < 3)
usage: python3 qstat.py status|left [mode]
"""
import glob
import json
import os
import sys
import time

W = "/workspace/atm"
O = f"{W}/out"
TAG = os.environ.get("E17_TAG", "k.1")
MANIFESTS = {"pre": ["pre.jsonl", "pcstats.jsonl"], "main": ["calib.jsonl", "pcstats.jsonl", "main.jsonl"]}
MAX_ATTEMPTS = 3


def read(path, default=""):
    try:
        return open(path).read().strip()
    except OSError:
        return default


def jobs_of(mode):
    jobs = {}
    for m in MANIFESTS.get(mode, []):
        try:
            for line in open(f"{O}/manifests/{m}"):
                if line.strip():
                    j = json.loads(line)
                    jobs[j["id"]] = j
        except OSError:
            pass
    return jobs


def weight(j):
    n = len(j.get("eps", [1])) if j["type"] == "episodes" else 1
    return n * (1 if j.get("task", "mw-").startswith("mw-") else 5)


def state(mode):
    jobs = jobs_of(mode)
    done, failed, gave_up, left = [], [], [], []
    for jid, j in jobs.items():
        res = f"{O}/jobs/{j['type']}/{jid}.json.gz"
        if os.path.exists(res):
            done.append((jid, os.path.getmtime(res)))
            continue
        att = int(read(f"{O}/attempts/{jid}.n", "0") or 0)
        if os.path.exists(f"{O}/errors/{jid}.txt"):
            failed.append(jid)
        if att >= MAX_ATTEMPTS:
            gave_up.append(jid)
        else:
            left.append(jid)
    return jobs, done, failed, gave_up, left


def driver_state():
    pid = read(f"{O}/driver.pid")
    if pid and os.path.exists(f"/proc/{pid}"):
        return "alive"
    if read(f"{O}/DRIVER_STATE") == "finished":
        return "finished"
    return "dead"


def utc(t=None):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def status():
    mode = read(f"{O}/DRIVER_MODE", "main")
    phase = read(f"{O}/PHASE", "setup")
    jobs, done, failed, gave_up, left = state(mode)
    drv = driver_state()
    now = time.time()
    try:   # the rate window starts at the driver start (driver.pid is written then), at most 30 min back
        t_drv = os.path.getmtime(f"{O}/driver.pid")
    except OSError:
        t_drv = now - 1800
    win = max(60.0, min(1800.0, now - t_drv))
    recent = [jobs[j] for j, t in done if t > now - win]
    rate = sum(weight(j) for j in recent) / win
    rem = sum(weight(jobs[j]) for j in left)
    eta = utc(now + rem / rate) if (rate > 0 and rem > 0 and drv == "alive") else "-"
    note = [f"mode={mode}"]
    lat = read(f"{O}/latency/progress.json")
    if lat:
        try:
            p = json.loads(lat)
            if p.get("stage") in ("smoke", "main", "quick"):
                note.append(p.get("note", "latency"))
            elif p.get("stage") == "main_done":
                note.append("latency done")
        except ValueError:
            pass
    if gave_up:
        note.append(f"gave_up={len(gave_up)}")
    if failed:
        note.append(f"last_err={sorted(failed)[-1]}")
    extra = read(f"{O}/NOTE")
    if extra:
        note.append(extra)
    print(f"tag={TAG} phase={phase} done={len(done)}/{len(jobs)} failed={len(failed)} driver={drv} eta={eta} "
          f"updated={utc(now)} note={';'.join(note).replace(' ', '_')}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        status()
    elif cmd == "left":
        mode = sys.argv[2] if len(sys.argv) > 2 else read(f"{O}/DRIVER_MODE", "main")
        print(len(state(mode)[4]))
