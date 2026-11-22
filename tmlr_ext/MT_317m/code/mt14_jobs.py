"""E14 (DESIGN_tmlr section 4) worker side: jobs run in spawned worker processes that each hold ONE mt80-317M agent (the
same model serves every task; only the task index changes). Adapted from E10's e10_jobs.py.

Every job writes one result file <out>/jobs/<stage>/<id>.json.gz (temp name, then rename); a job whose result file
exists is skipped. A failure writes <out>/errors/<id>.<attempt>.txt (the driver re-queues a job at most twice).

Job types
  episodes  closed-loop episodes of (task, variant, k, seed list) with E10's mtlib.run_episode_mt (E1 conventions: fresh
            env with cfg.seed = seed, random/numpy/torch seeded with seed, dither generator 7 seed + 13,
            eval_mode=True, success = last step's info['success']). Held none episodes (seeds 103000-103009) also run
            mtlib.HeldRecorder and save <big>/calib/<task>/held_<seed>.npz (root latents + 16 random candidates per MPPI
            iteration), the input of the det+BC bias.
  bias      det+BC bias vector b of (task, primary k) from the 10 held recordings (bclib.bias_sweeps, DESIGN_tmlr 0.6)
            -> <out>/bias/bias_<task>_k<k>.npz (b and the per-sweep means) + the job JSON (sweep report).
Inputs (code/inputs, copied from rss_ext/E10_mt317m/results): E10 per-tensor ranges ranges_<task>.json.gz (R4), the
E10 primary k (e10_ksel.csv; PRIMARY_K below is asserted against it) and the E10 eval none rates (e10_eval.csv; check
0.7-3).
"""
import csv
import gzip
import hashlib
import json
import os
import socket
import time
import traceback

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
INPUTS = os.path.join(CODE, "inputs")
BIG = os.environ.get("ATM_BIG", "/workspace/atm/big")

TASKS8 = ["mw-push", "mw-pick-place", "mw-door-open", "mw-drawer-open", "mw-button-press-topdown",
          "mw-peg-insert-side", "mw-window-open", "mw-sweep-into"]
PRIMARY_K = {"mw-push": 5, "mw-pick-place": 3, "mw-door-open": 3, "mw-drawer-open": 4,
             "mw-button-press-topdown": 3, "mw-peg-insert-side": 3, "mw-window-open": 3, "mw-sweep-into": 3}
EVAL_SEEDS = list(range(102000, 102090))     # DESIGN_tmlr 0.4: 317M eval, one checkpoint -> 90 pairs
HELD_SEEDS = list(range(103000, 103010))     # DESIGN_tmlr 0.4: 317M held (det+BC bias)
SMOKE_SEEDS = list(range(108000, 108010))    # phase-A checks only (never analysed)
VARIANTS = ("none", "det", "shiter", "shep", "detshift", "detbc")   # DESIGN_tmlr 4 (sh-iter, sh-ep, det+shift, det+BC)

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


def ranges_path(task):
    return f"{INPUTS}/ranges/ranges_{task}.json.gz"


def load_ranges(task):
    qr = read_json_gz(ranges_path(task))["quant"]
    return {k: (float(v[0]), float(v[1])) for k, v in qr.items()}


def e10_primary_k():
    """{task: primary k} from E10's e10_ksel.csv."""
    out = {}
    with open(f"{INPUTS}/e10_ksel.csv") as f:
        for r in csv.DictReader(f):
            out[r["task"]] = int(r["primary_k"])
    return out


def e10_none():
    """{task: E10 eval none success rate (90 episodes, seeds 90000-90089)}: full precision from E10's e10_summary.json
    (inputs/e10_eval_none_exact.json); e10_eval.csv holds 4 significant digits only (fallback)."""
    p = f"{INPUTS}/e10_eval_none_exact.json"
    if os.path.exists(p):
        d = json.load(open(p))
        return {t: float(v["none"]) for t, v in d.items() if not t.startswith("_")}
    out = {}
    with open(f"{INPUTS}/e10_eval.csv") as f:
        for r in csv.DictReader(f):
            out[r["task"]] = float(r["none"])
    return out


def bias_rule():
    """det+BC rule, read at every bias job: code/inputs/bias_rule.json = runpod_tmlr/bias_rule_D2.json (deviation D2,
    staged alpha 1 -> 1/2 -> 1/4 from b = 0 until every layer is below 1% of delta, at most 60 updates per stage).
    Without the file: DESIGN_tmlr 0.6 as frozen (alpha 1, 3 sweeps; superseded by D2)."""
    rule = {"alpha": 1.0, "n_sweeps": 3, "stop_tol": None, "source": "DESIGN_tmlr 0.6"}
    p = f"{INPUTS}/bias_rule.json"
    if os.path.exists(p):
        rule.update(json.load(open(p)))
        rule["source"] = p
    return rule


def calib_path(task, seed):
    return f"{BIG}/calib/{task}/held_{int(seed)}.npz"


def bias_path(out, task, k):
    return f"{out}/bias/bias_{task}_k{int(k)}.npz"


def load_bias(out, task, k):
    import torch
    import qlib as L
    d = np.load(bias_path(out, task, k))
    return {n: torch.as_tensor(d[f"b_{n}"], device=L.DEV, dtype=torch.float32) for n in L.LAYERS}


def sha1(b):
    return hashlib.sha1(b).hexdigest()


def rng_sha1():
    """Hash of the global torch CPU + CUDA RNG states (the planner's stream; the dither generator is private)."""
    import torch
    return sha1(torch.get_rng_state().numpy().tobytes() + torch.cuda.get_rng_state().cpu().numpy().tobytes())


def env_info():
    import torch
    return {"host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "pid": os.getpid(), "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "pod_tag": os.environ.get("ATM_TAG", "")}


def job_episodes(spec, out):
    import mtlib as M
    import qlib as L
    agent, quant, obs_dim = _W["agent"], _W["quant"], _W["obs_dim"]
    task, v, k = spec["task"], L.canon_variant(spec["variant"]), int(spec.get("k", 0))
    if v != "none":
        quant.qrange = load_ranges(task)
    if v == "detbc":
        quant.bias = load_bias(out, task, k)
    L.set_planner(agent, None)          # planner defaults (E1 / E10)
    eps = []
    try:
        for seed in spec["seeds"]:
            seed = int(seed)
            rec = M.HeldRecorder(agent, seed_rec=seed - HELD_SEEDS[0]) if spec.get("record") else None
            t0 = time.time()
            try:
                r = M.run_episode_mt(agent, quant, task, v, k, seed, obs_dim, recorder=rec)
            finally:
                if rec is not None:
                    rec.detach()
            e = {"seed": seed, "ret": r["ret"], "success": r["success"], "success_any": r["success_any"],
                 "steps": r["steps"], "nq": r["nq"], "n_refresh": int(quant.n_refresh),
                 "act_ms_mean": r["act_ms_mean"], "act_ms_median": r["act_ms_median"],
                 "act_sha1": sha1(r["_act"].numpy().tobytes()), "rng_sha1": rng_sha1(), "secs": time.time() - t0}
            if rec is not None:
                arr = rec.arrays()
                p = calib_path(task, seed)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                tmp = p + f".tmp{os.getpid()}.npz"
                np.savez_compressed(tmp, obs_seq=r["_obs"].numpy(), act_seq=r["_act"].numpy(),
                                    summary=json.dumps({k_: e[k_] for k_ in ("seed", "ret", "success", "steps")}),
                                    **arr)
                os.replace(tmp, p)
                e["calib_file"] = p
                e["n_rows"] = int(np.prod(arr["cand"].shape[:3]))
            eps.append(e)
    finally:
        quant.off()
        quant.bias = {}
    return {"episodes": eps}


def job_bias(spec, out):
    import bclib as B
    agent, quant = _W["agent"], _W["quant"]
    task, k = spec["task"], int(spec["k"])
    files = [calib_path(task, s) for s in HELD_SEEDS]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"held recordings missing: {missing}")
    quant.qrange = load_ranges(task)
    rule = bias_rule()
    if "stages" in rule:                # D2 (the main-run rule since 2026-11-21T19:01Z)
        b, rep, arrays = B.bias_staged(agent, quant, files, task, k, rule["stages"])
    else:
        b, rep, arrays = B.bias_sweeps(agent, quant, files, task, k, n_sweeps=rule["n_sweeps"], alpha=rule["alpha"],
                                       stop_tol=rule["stop_tol"])
    rep["rule"] = rule
    p = bias_path(out, task, k)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + f".tmp{os.getpid()}.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, p)                 # written before the job JSON: a finished bias job always has its npz
    rep["bias_file"] = p
    rep["bias_sha1"] = sha1(b"".join(arrays[f"b_{n}"].tobytes() for n in ("dyn0", "dyn1", "rew0", "rew1")))
    return rep


JOBS = {"episodes": job_episodes, "bias": job_bias}


def brief(spec, res):
    if spec["type"] == "episodes":
        e = res["episodes"]
        return (f"succ {sum(x['success'] for x in e):.0f}/{len(e)} act {np.median([x['act_ms_median'] for x in e]):.1f}ms "
                f"ep {np.mean([x['secs'] for x in e]):.1f}s")
    if spec["type"] == "bias" and res.get("rule_id") == "D2":
        st = " ".join(f"eta{x['eta']}:{'conv' if x['converged'] else 'no'}@{x['n_updates']}" for x in res["stages"])
        return (f"D2 {st} | final eta {res['final_eta']} resid/delta " +
                " ".join(f"{n}:{v:.2e}" for n, v in res["residual_final"].items()) +
                f" ok={res['residual_ok']} unconverged={res['bc_unconverged']}")
    if spec["type"] == "bias":
        last = res["sweeps"][-1]["layers"]
        first = res["sweeps"][0]["layers"]
        return ("resid/delta " + " ".join(f"{n}:{last[n]['mean_abs_over_delta']:.2e}" for n in last) +
                " | sweep1 " + " ".join(f"{n}:{first[n]['mean_abs_over_delta']:.3f}" for n in first) +
                f" ok={res['residual_ok']}")
    return ""


def error_files(out, jid):
    d = f"{out}/errors"
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.startswith(jid + ".") and f.endswith(".txt"))


def run_job(spec, out):
    import torch
    t0 = time.time()
    p = result_path(out, spec)
    if os.path.exists(p) and not spec.get("redo"):
        return {"id": spec["id"], "ok": True, "secs": 0.0, "brief": "already done", "skipped": True}
    try:
        with torch.no_grad():
            res = JOBS[spec["type"]](spec, out)
        res.update(spec=spec, env=env_info(), t_start=t0, secs=time.time() - t0, ok=True)
        write_json_gz(p, res)
        return {"id": spec["id"], "ok": True, "secs": res["secs"], "brief": brief(spec, res),
                "n_eps": len(res.get("episodes", []))}
    except Exception:
        err = traceback.format_exc()
        os.makedirs(f"{out}/errors", exist_ok=True)
        n = len(error_files(out, spec["id"])) + 1
        with open(f"{out}/errors/{spec['id']}.{n}.txt", "w") as f:
            f.write(err)
        return {"id": spec["id"], "ok": False, "secs": time.time() - t0, "brief": err.strip().splitlines()[-1]}
