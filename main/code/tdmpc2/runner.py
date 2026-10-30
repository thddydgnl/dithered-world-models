"""Manifest-based, resumable job runner for the TD-MPC2 part of paper A (E0, E1; reused by E2/E3/E5).

A manifest is a JSONL file, one job spec per line ({"id": ..., "type": ..., ...}; built by make_manifest.py). Every job
writes one result file <out>/jobs/<type>/<id>.json.gz (written to a temp name, then renamed); jobs whose result file
exists are skipped, so a crashed or stopped run continues where it stopped. Workers are spawned processes on GPU 0
(CUDA_VISIBLE_DEVICES=0; run under the user's own CUDA MPS daemon, see env.sh / mps.sh).

Job types
  calib     one unquantized calibration episode; stores planning-time inputs (root latent + 16 random candidates per
            MPPI iteration), the trajectory and 50 decision states (full first/last candidate sets) in
            <out>/calib/<task>_s<ckpt>_ep<ep>.npz (pilot calib.py)
  ranges    per (task, ckpt): per-layer [lo, hi] = [min, 99.99th pct] of the planning-time activations of calibration
            episodes 0-4 (R2-8 procedure) -> <out>/scales/<task>_s<ckpt>.json
  episodes  closed-loop episodes of one cell chunk (task, ckpt, k, variant, episode list, planner overrides) with
            optional instrumentation (OEI / decision metrics, plan deviation, per-episode prediction errors)
  prederr   per (task, ckpt, k): held-set 1/3/5-step prediction errors of every variant (planning-time inputs and
            trajectory inputs of calibration episodes 5-9), mean of 3 dither seeds
  decision  per (task, ckpt, k): R2-8b decision analysis on 150 fixed decision states of calibration episodes 5-9:
            metrics on the fixed first/last candidate sets and one-step plan reruns (plan deviation / seed floor)
  e5stats   per (task, ckpt): E5 PTQ / AWQ statistics of the calibration activations (e5lib.compute_stats)
  qat_train per (task, ckpt, k): E5 QAT of the dynamics and reward MLPs (e5lib.qat_train)
Usage: python runner.py --manifest m.jsonl --out <dir> --workers 16 [--types episodes] [--limit 0] [--dry]
"""
import argparse
import gzip
import json
import os
import socket
import time
import traceback
from collections import OrderedDict
from multiprocessing import TimeoutError as MPTimeoutError
from multiprocessing import get_context

R2_8_INPUTS = os.environ.get("R28_INPUTS", os.path.expanduser("~/pmx/A/tdmpc2_main/inputs/r28"))
_cache = OrderedDict()
CACHE_SIZE = 2


def result_path(out, spec):
    return f"{out}/jobs/{spec['type']}/{spec['id']}.json.gz"


def write_json_gz(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with gzip.open(tmp, "wt") as f:
        json.dump(obj, f, separators=(",", ":"), default=float)
    os.replace(tmp, path)


def r5(x):
    """Round to 5 significant digits (compact per-step records)."""
    if isinstance(x, float):
        return float(f"{x:.5g}") if x == x else None
    return x


def _keys(recs, drop=()):
    """Union of the record keys in first-seen order (records of one job normally share their keys)."""
    keys = {}
    for r in recs:
        for k in r:
            if k not in drop and not k.startswith("_"):
                keys.setdefault(k, None)
    return list(keys)


def columnar(recs, drop=()):
    if not recs:
        return {}
    return {k: [r5(r.get(k)) for r in recs] for k in _keys(recs, drop)}


def means(recs):
    import numpy as np
    if not recs:
        return {}
    out = {}
    for k in _keys(recs):
        if k == "t":
            continue
        v = [r[k] for r in recs if isinstance(r.get(k), (int, float)) and r[k] == r[k]]
        if v:
            out[k] = float(np.mean(v))
    return out


def get_agent(task, ckpt):
    import qlib as L
    key = (task, int(ckpt))
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    while len(_cache) >= CACHE_SIZE:
        _, old = _cache.popitem(last=False)
        old["probe"].detach()
        old["quant"].remove()
    agent, cfg = L.build_agent(task, ckpt)
    quant = L.Quantizer(agent.model, horizon=int(cfg.horizon))
    probe = L.PlanProbe(agent, quant)
    _cache[key] = {"agent": agent, "cfg": cfg, "quant": quant, "probe": probe}
    return _cache[key]


def load_ranges(out, task, ckpt, scales="main"):
    if scales == "r28":  # pilot R2-8 ranges (ckpt seed 1 only), for the reproduction of R2-8b
        if int(ckpt) != 1:
            raise ValueError("r28 ranges exist for checkpoint seed 1 only")
        qr = json.load(open(f"{R2_8_INPUTS}/prep_{task}.json"))["quant_range"]
    else:
        qr = json.load(open(f"{out}/scales/{task}_s{int(ckpt)}.json"))["quant"]
    return {k: (float(v[0]), float(v[1])) for k, v in qr.items()}


def env_info():
    import torch
    return {"host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "cvd": os.environ.get("CUDA_VISIBLE_DEVICES"), "mps": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"),
            "pid": os.getpid()}


# ----------------------------------------------------------------------------------------------------------------------
def job_episodes(spec, out):
    """Episodes of one cell chunk. E5 / E7 job-level settings (restored at the end, also on errors):
    ptq (perchan / mseclip / smooth / mixed / best), wq (weight-only variants), qat (QAT weights of (task, ckpt, k)),
    save_traj (QAT data), e7 (real-kernel precision)."""
    import qlib as L
    A = get_agent(spec["task"], spec["ckpt"])
    agent, quant = A["agent"], A["quant"]
    extra, restore = {}, []
    try:
        if spec.get("ptq"):
            import e5lib as E5
            ptq = spec["ptq"]
            if ptq == "best":
                ptq = E5.resolve_best(out, spec["task"], spec["k"])
            qr, lk = E5.ptq_config(E5.load_stats(out, spec["task"], spec["ckpt"]), ptq, spec["k"])
            extra.update(ptq_resolved=ptq, layer_k=lk)
            restore.append(lambda: setattr(quant, "layer_k", {}))
            quant.layer_k = lk
            spec_q = qr
        else:
            spec_q = None
        if spec.get("wq"):
            import e5lib as E5
            L.set_planner(agent, spec.get("planner"))
            st = E5.load_stats(out, spec["task"], spec["ckpt"]) if spec["wq"] == "w4awq" else None
            wq = E5.WeightQuant(quant, spec["wq"], agent.cfg.num_samples, agent.cfg.horizon, st)
            restore.append(lambda: (wq.restore(), setattr(quant, "wq", None)))
            quant.wq = wq
        if spec.get("qat"):
            import e5lib as E5
            saved = E5.load_qat(agent, out, spec["task"], spec["ckpt"], spec["k"])
            restore.append(lambda: E5.load_state(agent, saved))
            extra["qat_path"] = E5.qat_path(out, spec["task"], spec["ckpt"], spec["k"])
        if spec.get("e7"):
            import e7lib as E7
            e7 = E7.install(agent, quant, out, spec)
            restore.append(e7.restore)
            extra["e7"] = e7.info()
        res = _job_episodes(spec, out, A, spec_q)
        res.update(extra)
        return res
    finally:
        for f in reversed(restore):
            f()
        quant.layer_k, quant.wq = {}, None


def _job_episodes(spec, out, A, qrange_override=None):
    import numpy as np
    import torch
    import qlib as L
    agent, cfg, quant, probe = A["agent"], A["cfg"], A["quant"], A["probe"]
    v = L.canon_variant(spec["variant"])
    k = int(spec.get("k", 0))
    if v != "none":
        quant.qrange = qrange_override or load_ranges(out, spec["task"], spec["ckpt"], spec.get("scales", "main"))
    L.set_planner(agent, spec.get("planner"))
    quant.H = int(agent.cfg.horizon)   # horizon index of shstep/shep table rows follows a horizon override
    instr = spec.get("instr", {})
    oei_stride, pd_stride = int(instr.get("oei_stride", 0)), int(instr.get("pd_stride", 0))
    perr, nseeds = bool(instr.get("perr", False)), int(instr.get("perr_seeds", 3))
    perr_states = int(instr.get("perr_plan_states", 8))   # planning-time errors on <= this many OEI states / episode
    eps = []
    for ep in spec["eps"]:
        seed = int(spec["seed0"]) + int(ep)
        t0 = time.time()
        r = L.run_episode(agent, cfg, quant, v, k, seed, probe=probe if oei_stride > 0 else None,
                          oei_stride=oei_stride, pd_stride=pd_stride, keep_cands=perr and v != "none",
                          warm_start=bool(spec.get("warm_start", True)), log_calls=bool(spec.get("log_calls", False)),
                          indep_reeval=bool(instr.get("indep_reeval", False)),
                          region_ref=bool(instr.get("region_ref", False)))
        secs_loop = time.time() - t0
        rec = {"ep": int(ep), "seed": seed, "ret": r["ret"], "success": r["success"], "success_any": r["success_any"],
               "steps": r["steps"], "n_refresh": r["n_refresh"], "secs_loop": secs_loop}
        if "calls_patterns" in r:
            rec.update(calls_patterns=r["calls_patterns"], table_sig=r["table_sig"])
        if r["steps_rec"]:
            rec["oei_steps"] = columnar(r["steps_rec"])
            rec["oei_mean"] = means(r["steps_rec"])
        if r["pd_rec"]:
            rec["pd_steps"] = columnar(r["pd_rec"])
            rec["pd_mean"] = means(r["pd_rec"])
        if perr and v != "none":
            seeds = [9_000_001 + 1000 * seed + j for j in range(nseeds)]
            ti = L.traj_inputs(agent, r["_obs"], r["_act"], K=5)
            if ti is not None:
                Z0, Aw, Zt = ti
                rec["perr_traj"] = L.pred_errors(agent, cfg, quant, v, k, Z0, Aw, seeds, Ztrue=Zt, h1=int(cfg.horizon))
            cands = [s["_cand"] for s in r["steps_rec"] if "_cand" in s]
            if len(cands) > perr_states > 0:   # evenly spaced subset of the recorded states (cost control)
                cands = [cands[i] for i in np.linspace(0, len(cands) - 1, perr_states).round().astype(int)]
            if cands:
                Z0, Ap = L.plan_inputs(agent, cands, K=5)
                rec["perr_plan"] = L.pred_errors(agent, cfg, quant, v, k, Z0, Ap, seeds, h1=int(cfg.horizon))
                rec["perr_plan"]["n_states"] = len(cands)
        if spec.get("save_traj"):
            import e5lib as E5
            rec["traj"] = E5.save_traj(out, spec["task"], spec["ckpt"], ep, r)
        rec["secs"] = time.time() - t0
        eps.append(rec)
        torch.cuda.empty_cache() if perr else None
    quant.off()
    L.set_planner(agent, None)
    quant.H = int(agent.cfg.horizon)
    return {"episodes": eps}


def job_calib(spec, out):
    import numpy as np
    import torch
    import qlib as L
    from calib_lib import record_calib_episode
    A = get_agent(spec["task"], spec["ckpt"])
    L.set_planner(A["agent"], None)
    A["quant"].off()
    path = f"{out}/calib/{spec['task']}_s{int(spec['ckpt'])}_ep{int(spec['ep'])}.npz"
    summ = record_calib_episode(A["agent"], A["cfg"], A["quant"], int(spec["seed"]), path, seed_rec=int(spec["ep"]))
    return {"summary": summ, "npz": path}


def job_ranges(spec, out):
    import qlib as L
    from calib_lib import quant_ranges_from_calib
    A = get_agent(spec["task"], spec["ckpt"])
    qr, info = quant_ranges_from_calib(A["agent"], A["quant"], out, spec["task"], spec["ckpt"], spec["cal_eps"])
    os.makedirs(f"{out}/scales", exist_ok=True)
    p = f"{out}/scales/{spec['task']}_s{int(spec['ckpt'])}.json"
    tmp = p + f".tmp{os.getpid()}"
    json.dump({"quant": qr, "info": info, "cal_eps": spec["cal_eps"], "method": "min / 99.99th pct, 12000 rows"},
              open(tmp, "w"), indent=1)
    os.replace(tmp, p)
    return {"quant": qr, "info": info}


def job_prederr(spec, out):
    from calib_lib import held_pred_errors
    A = get_agent(spec["task"], spec["ckpt"])
    A["quant"].qrange = load_ranges(out, spec["task"], spec["ckpt"], spec.get("scales", "main"))
    return held_pred_errors(A["agent"], A["cfg"], A["quant"], out, spec)


def job_decision(spec, out):
    from calib_lib import fixed_state_decision
    A = get_agent(spec["task"], spec["ckpt"])
    A["quant"].qrange = load_ranges(out, spec["task"], spec["ckpt"], spec.get("scales", "main"))
    return fixed_state_decision(A["agent"], A["cfg"], A["quant"], A["probe"], out, spec)


def job_e5stats(spec, out):
    import e5lib as E5
    import qlib as L
    A = get_agent(spec["task"], spec["ckpt"])
    L.set_planner(A["agent"], None)
    return E5.compute_stats(A["agent"], A["quant"], out, spec)


def job_qat_train(spec, out):
    import e5lib as E5
    import qlib as L
    A = get_agent(spec["task"], spec["ckpt"])
    L.set_planner(A["agent"], None)
    return E5.qat_train(A["agent"], A["cfg"], A["quant"], out, spec)


JOBS = {"episodes": job_episodes, "calib": job_calib, "ranges": job_ranges, "prederr": job_prederr,
        "decision": job_decision, "e5stats": job_e5stats, "qat_train": job_qat_train}


def claim(out, spec):
    """Exclusive claim of a job (lock file with host:pid), so several runner instances can share one manifest. A lock
    whose process is gone (same host) is stale and is taken over."""
    os.makedirs(f"{out}/locks", exist_ok=True)
    lp = f"{out}/locks/{spec['id']}.lock"
    me = f"{socket.gethostname()}:{os.getpid()}"
    for _ in range(2):
        try:
            fd = os.open(lp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, me.encode())
            os.close(fd)
            return lp
        except FileExistsError:
            try:
                host, pid = open(lp).read().strip().split(":")
                alive = host != socket.gethostname() or os.path.exists(f"/proc/{int(pid)}")
            except Exception:
                alive = False
            if alive:
                return None
            try:
                os.remove(lp)
            except FileNotFoundError:
                pass
    return None


def run_job(args):
    spec, out = args
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    if os.path.exists(result_path(out, spec)):
        return {"id": spec["id"], "ok": True, "secs": 0.0, "brief": "already done", "skipped": True}
    lp = claim(out, spec)
    if lp is None:
        return {"id": spec["id"], "ok": True, "secs": 0.0, "brief": "claimed by another runner", "skipped": True}
    try:
        with torch.no_grad():
            res = JOBS[spec["type"]](spec, out)
        res.update(spec=spec, env=env_info(), t_start=t0, secs=time.time() - t0, ok=True)
        write_json_gz(result_path(out, spec), res)
        return {"id": spec["id"], "ok": True, "secs": res["secs"], "brief": brief(spec, res)}
    except Exception:
        err = traceback.format_exc()
        os.makedirs(f"{out}/errors", exist_ok=True)
        with open(f"{out}/errors/{spec['id']}.txt", "w") as f:
            f.write(err)
        return {"id": spec["id"], "ok": False, "secs": time.time() - t0, "brief": err.strip().splitlines()[-1]}
    finally:
        try:
            os.remove(lp)
        except FileNotFoundError:
            pass


def brief(spec, res):
    if spec["type"] == "episodes":
        e = res["episodes"]
        return f"ret {sum(x['ret'] for x in e) / len(e):8.1f} succ {sum(x['success'] for x in e):.0f}/{len(e)}"
    return ""


def load_manifest(path):
    return [json.loads(l) for l in open(path) if l.strip()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--types", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--max_seconds", type=float, default=0, help="stop (terminate the pool) after this many seconds "
                    "(throughput benchmark); unfinished jobs simply stay pending")
    a = ap.parse_args()
    jobs = load_manifest(a.manifest)
    if a.types:
        jobs = [j for j in jobs if j["type"] in a.types.split(",")]
    todo = [j for j in jobs if not os.path.exists(result_path(a.out, j))]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(jobs)} jobs in manifest, {len(jobs) - len([j for j in jobs if not os.path.exists(result_path(a.out, j))])}"
          f" done, running {len(todo)} with {a.workers} workers", flush=True)
    if a.dry or not todo:
        print("RUNNER_DONE", flush=True)
        raise SystemExit(0)
    t0 = time.time()
    n_ok = n_fail = 0
    stopped = False
    with get_context("spawn").Pool(min(a.workers, len(todo))) as pool:
        it = pool.imap_unordered(run_job, [(j, a.out) for j in todo], chunksize=1)
        while True:
            try:
                if a.max_seconds > 0:
                    left = a.max_seconds - (time.time() - t0)
                    if left <= 0:
                        raise TimeoutError
                    r = it.next(timeout=left)
                else:
                    r = it.next()
            except StopIteration:
                break
            except (TimeoutError, MPTimeoutError):
                pool.terminate()
                stopped = True
                break
            if r.get("skipped"):
                continue
            n_ok += r["ok"]
            n_fail += not r["ok"]
            print(f"{time.time() - t0:8.0f}s [{n_ok + n_fail}/{len(todo)}] {'OK ' if r['ok'] else 'ERR'} {r['id']} "
                  f"({r['secs']:.0f}s) {r['brief']}", flush=True)
    print(f"RUNNER_{'STOPPED' if stopped else 'DONE'} ok {n_ok} failed {n_fail} in {time.time() - t0:.0f}s", flush=True)
