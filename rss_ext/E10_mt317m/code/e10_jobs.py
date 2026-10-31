"""E10 worker side: jobs run in spawned worker processes that each hold ONE mt80-317M agent (the same model serves every
task; only the task index changes). Every job writes one result file <out>/jobs/<stage>/<id>.json.gz (temp name, then
rename); a job whose result file exists is skipped. Failures write <out>/errors/<id>.txt.

Job types
  episodes  closed-loop episodes of (task, variant, k, seed list); optional HeldRecorder (held none episodes) saving
            <out>/calib/<task>/held_<seed>.npz (root latents + 16 random candidates per MPPI iteration for the ranges,
            full first-iteration candidate sets of the 8 E1 perr states for pt_err1h)
  ranges    per-layer [min, 99.99th pct] of the held planning forwards (E0 method) -> <out>/scales/<task>.json
  perr      planning-time prediction errors of det and sh-iter at k on the held inputs: per held episode
            plan_inputs_mt (8 states x 512 candidates, K = 5) + pred_errors_mt (h1 = horizon 3; dither seeds
            9_000_001 + 1000 seed + j, j < 3, as E1's per-episode perr_plan); pt_err1h = mean over the 30 held episodes
"""
import gzip
import json
import os
import socket
import time
import traceback

import numpy as np

_W = {}


def init_worker():
    import torch
    import mtlib as M
    import qlib as L
    torch.set_num_threads(1)
    agent, cfg = M.build_agent_mt()
    quant = L.Quantizer(agent.model, horizon=int(cfg.horizon))
    _W.update(agent=agent, cfg=cfg, quant=quant, obs_dim=int(cfg.obs_shape["state"][0]), t_init=time.time())


def result_path(out, spec):
    return f"{out}/jobs/{spec['stage']}/{spec['id']}.json.gz"


def write_json_gz(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with gzip.open(tmp, "wt") as f:
        json.dump(obj, f, separators=(",", ":"), default=float)
    os.replace(tmp, path)


def read_json_gz(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def scales_path(out, task):
    return f"{out}/scales/{task}.json"


def calib_path(out, task, seed):
    return f"{out}/calib/{task}/held_{int(seed)}.npz"


def load_ranges(out, task):
    qr = json.load(open(scales_path(out, task)))["quant"]
    return {k: (float(v[0]), float(v[1])) for k, v in qr.items()}


def env_info():
    import torch
    return {"host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "pid": os.getpid(), "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32)}


def job_episodes(spec, out):
    import mtlib as M
    import qlib as L
    agent, quant, obs_dim = _W["agent"], _W["quant"], _W["obs_dim"]
    task, v, k = spec["task"], L.canon_variant(spec["variant"]), int(spec.get("k", 0))
    if v != "none":
        quant.qrange = load_ranges(out, task)
    L.set_planner(agent, None)          # planner defaults (E1)
    eps = []
    for seed in spec["seeds"]:
        seed = int(seed)
        rec = M.HeldRecorder(agent, seed_rec=seed - M.HELD_SEEDS[0]) if spec.get("record") else None
        t0 = time.time()
        try:
            r = M.run_episode_mt(agent, quant, task, v, k, seed, obs_dim, recorder=rec)
        finally:
            if rec is not None:
                rec.detach()
        e = {"seed": seed, "ret": r["ret"], "success": r["success"], "success_any": r["success_any"],
             "steps": r["steps"], "nq": r["nq"], "act_ms_mean": r["act_ms_mean"], "act_ms_median": r["act_ms_median"],
             "secs": time.time() - t0}
        if rec is not None:
            arr = rec.arrays()
            p = calib_path(out, task, seed)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            tmp = p + f".tmp{os.getpid()}.npz"
            np.savez_compressed(tmp, obs_seq=r["_obs"].numpy(), act_seq=r["_act"].numpy(),
                                summary=json.dumps({k_: e[k_] for k_ in ("seed", "ret", "success", "steps")}), **arr)
            os.replace(tmp, p)
            e["calib_file"] = p
            e["n_pt_states"] = int(len(arr["pt_t"]))
        eps.append(e)
    quant.off()
    return {"episodes": eps}


def job_ranges(spec, out):
    import mtlib as M
    agent, quant = _W["agent"], _W["quant"]
    task = spec["task"]
    files = [calib_path(out, task, s) for s in M.HELD_SEEDS]
    qr, info = M.quant_ranges_mt(agent, quant, files, task)
    os.makedirs(f"{out}/scales", exist_ok=True)
    p = scales_path(out, task)
    tmp = p + f".tmp{os.getpid()}"
    json.dump({"quant": qr, "info": info, "held_seeds": M.HELD_SEEDS,
               "method": "min / 99.99th pct of 12000 rows (16 random candidates per MPPI iteration, every step of the "
                         "30 held none episodes), E0 calib_lib procedure with the mt80 task index"},
              open(tmp, "w"), indent=1)
    os.replace(tmp, p)
    return {"quant": qr, "info": info}


def job_perr(spec, out):
    import torch
    import mtlib as M
    import qlib as L
    agent, cfg, quant = _W["agent"], _W["cfg"], _W["quant"]
    task, k = spec["task"], int(spec["k"])
    quant.qrange = load_ranges(out, task)
    L.set_planner(agent, None)
    tt = M.task_tensor(task)
    torch.manual_seed(12345)    # nothing below samples (policy-prior mean); keep any RNG use fixed (calib_lib)
    rows = []
    for s in M.HELD_SEEDS:
        d = np.load(calib_path(out, task, s))
        cands = [(torch.as_tensor(d["pt_z"][i], device=L.DEV), torch.as_tensor(d["pt_cand"][i], device=L.DEV))
                 for i in range(len(d["pt_t"]))]
        Z0, Ap = M.plan_inputs_mt(agent, cands, tt, K=5)
        seeds = [9_000_001 + 1000 * int(s) + j for j in range(3)]
        row = {"seed": int(s), "n_states": len(cands), "n_seq": int(Z0.shape[0])}
        for v in ("det", "shiter"):
            row[v] = M.pred_errors_mt(agent, cfg, quant, v, k, Z0, Ap, seeds, tt, h1=int(cfg.horizon))
        rows.append(row)
        torch.cuda.empty_cache()
    quant.off()
    summ = {v: {m: float(np.mean([r[v][m] for r in rows])) for m in ("err1", "err3", "err5", "err1h", "errr")}
            for v in ("det", "shiter")}
    return {"task": task, "k": k, "episodes": rows, "summary": summ}


JOBS = {"episodes": job_episodes, "ranges": job_ranges, "perr": job_perr}


def brief(spec, res):
    if spec["type"] == "episodes":
        e = res["episodes"]
        return (f"succ {sum(x['success'] for x in e):.0f}/{len(e)} act {np.median([x['act_ms_median'] for x in e]):.1f}ms "
                f"ep {np.mean([x['secs'] for x in e]):.1f}s")
    if spec["type"] == "perr":
        s = res["summary"]
        return f"err1h det {s['det']['err1h']:.4g} shiter {s['shiter']['err1h']:.4g}"
    if spec["type"] == "ranges":
        return " ".join(f"{n}:{v[1]:.3g}" for n, v in res["quant"].items())
    return ""


def run_job(spec, out):
    import torch
    t0 = time.time()
    p = result_path(out, spec)
    if os.path.exists(p):
        return {"id": spec["id"], "ok": True, "secs": 0.0, "brief": "already done", "skipped": True}
    try:
        with torch.no_grad():
            res = JOBS[spec["type"]](spec, out)
        res.update(spec=spec, env=env_info(), t_start=t0, secs=time.time() - t0, ok=True)
        write_json_gz(p, res)
        return {"id": spec["id"], "ok": True, "secs": res["secs"], "brief": brief(spec, res)}
    except Exception:
        err = traceback.format_exc()
        os.makedirs(f"{out}/errors", exist_ok=True)
        with open(f"{out}/errors/{spec['id']}.txt", "w") as f:
            f.write(err)
        return {"id": spec["id"], "ok": False, "secs": time.time() - t0, "brief": err.strip().splitlines()[-1]}
