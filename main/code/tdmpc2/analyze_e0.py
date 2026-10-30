"""E0 analyses for the TD-MPC2 part (run on the Mac on rsynced job results, or on the server).
  repro   R2-8b reproduction: drops vs the pilot table (paired bootstrap 90% CI as in R2-8b, plus 95%)
  bench   throughput: episodes/hour per worker count (from the bench run logs written by run_bench.sh)
  held    task and bit-width selection (design 6.0): MW baseline success >= 80%; largest k in {3,4,5,6} with a det drop
          >= 20pp (MW success) / >= 20% (DMC return) on the held episodes -> primary k and cells {k-1, k, k+1}
Inputs: <res>/jobs/episodes/*.json.gz. Usage: python analyze_e0.py repro --res <dir> --out <dir>
"""
import argparse
import glob
import gzip
import json
import os
from collections import defaultdict

import numpy as np

NBOOT, BOOT_SEED = 10000, 3
PILOT_R28B = {  # R2-8b REPORT 2.1 (20 episodes, seeds 2000-2019, RTX 3090): drop (MW %p of success, DMC % of return)
    ("mw-push", 3): {"det": 65, "ind": 15, "shiter": 5, "shstep": 65, "shep": 85},
    ("mw-pick-place", 3): {"det": 55, "ind": 35, "shiter": 10, "shstep": 40, "shep": 75},
    ("cheetah-run", 4): {"det": 52.4, "ind": 50.0, "shiter": 47.1, "shstep": 77.2, "shep": 80.7},
}
PILOT_RED = {("mw-push", 3): 0.92, ("mw-pick-place", 3): 0.82, ("cheetah-run", 4): 0.10}
PILOT_BASE = {"mw-push": (1.0, 1786), "mw-pick-place": (1.0, 1682), "cheetah-run": (None, 864.6)}


def load_eps(res, prefix=None):
    rows = []
    for f in sorted(glob.glob(f"{res}/jobs/episodes/*.json.gz")):
        if prefix and not os.path.basename(f).startswith(prefix):
            continue
        d = json.load(gzip.open(f, "rt"))
        s = d["spec"]
        for e in d["episodes"]:
            rows.append({"split": s["split"], "task": s["task"], "ckpt": s["ckpt"], "k": s["k"],
                         "variant": s["variant"], "ep": e["ep"], "seed": e["seed"], "ret": e["ret"],
                         "success": e["success"], "secs": e.get("secs"), "gpu": d["env"]["gpu"],
                         "mps": d["env"].get("mps")})
    return rows


def metric_of(task):
    return "success" if task.startswith("mw-") else "ret"


def pct(b, lo, hi):
    b = b[np.isfinite(b)]
    return [float(np.percentile(b, lo)), float(np.percentile(b, hi))] if len(b) else [float("nan")] * 2


def paired(rows, task, ckpts, k, variants, metric, seeds=None):
    """Per-episode paired drops on (ckpt, seed) pairs that every condition has; bootstrap over pairs.
    MW success (design 4.3: success-rate drop in %p): d_v,i = base_i - x_v,i (absolute).
    DMC / return: d_v,i = (base_i - x_v,i) / R0 with R0 = mean baseline (relative drop)."""
    base = {(r["ckpt"], r["seed"]): r[metric] for r in rows
            if r["task"] == task and r["variant"] == "none" and r["ckpt"] in ckpts}
    vals = {v: {(r["ckpt"], r["seed"]): r[metric] for r in rows
                if r["task"] == task and r["variant"] == v and r["k"] == k and r["ckpt"] in ckpts} for v in variants}
    keys = sorted(set(base).intersection(*[set(x) for x in vals.values()]))
    if seeds is not None:
        keys = [kk for kk in keys if kk[1] in seeds]
    n = len(keys)
    if n == 0:
        return None
    b = np.array([base[kk] for kk in keys], float)
    R0 = float(b.mean())
    scale = 1.0 if metric == "success" else R0
    d = {v: (b - np.array([vals[v][kk] for kk in keys], float)) / scale for v in variants}
    idx = np.random.default_rng(BOOT_SEED).integers(0, n, size=(NBOOT, n))
    out = {"n": n, "R0": R0, "drop_type": "absolute" if metric == "success" else "relative",
           "mean": {v: float(np.mean([vals[v][kk] for kk in keys])) for v in variants},
           "drop": {}, "ci90": {}, "ci95": {}, "red": {}, "red_ci90": {}}
    for v in variants:
        bs = d[v][idx].mean(1)
        out["drop"][v] = float(d[v].mean())
        out["ci90"][v] = pct(bs, 5, 95)
        out["ci95"][v] = pct(bs, 2.5, 97.5)
    if "det" in d:
        bd = d["det"][idx].mean(1)
        for v in variants:
            if v in ("det",):
                continue
            Dd = out["drop"]["det"]
            out["red"][v] = float((Dd - out["drop"][v]) / Dd) if Dd > 0 else float("nan")
            bv = d[v][idx].mean(1)
            r = np.where(bd > 0, (bd - bv) / np.where(bd > 0, bd, 1.0), np.nan)
            out["red_ci90"][v] = pct(r, 5, 95)
    return out


def cmd_repro(a):
    rows = load_eps(a.res, "repro_")
    out = {"cells": {}, "gpu": sorted({r["gpu"] for r in rows}), "n_episodes": len(rows)}
    lines = ["| 칸 | 변형 | 하락 seed 2000~2019 (90% CI) | 하락 seed 2000~2039 (90% CI) | 파일럿 R2-8b |",
             "|---|---|---|---|---|"]
    for (task, k), pil in PILOT_R28B.items():
        vs = [v for v in ("det", "ind", "shiter", "shstep", "shep")
              if any(r["task"] == task and r["variant"] == v for r in rows)]
        m = metric_of(task)
        c20 = paired(rows, task, [1], k, vs, m, seeds=set(range(2000, 2020)))
        c40 = paired(rows, task, [1], k, vs, m)
        sec = paired(rows, task, [1], k, vs, "ret") if m == "success" else None
        out["cells"][f"{task}:{k}"] = {"seeds2000_2019": c20, "seeds2000_2039": c40, "return_drop_all": sec,
                                       "pilot_drop": pil, "pilot_red_shiter": PILOT_RED[(task, k)]}
        for v in vs:
            f = lambda c: (f"{100 * c['drop'][v]:.1f} [{100 * c['ci90'][v][0]:.1f}, {100 * c['ci90'][v][1]:.1f}]"
                           if c else "-")
            lines.append(f"| {task} k={k} | {v} | {f(c20)} | {f(c40)} | {pil.get(v, '-')} |")
        for lab, c in (("2000~2019", c20), ("2000~2039", c40)):
            if c and "shiter" in c["red"]:
                lines.append(f"| {task} k={k} | red(sh-iter) {lab} | {100 * c['red']['shiter']:.0f}% "
                             f"[{100 * c['red_ci90']['shiter'][0]:.0f}, {100 * c['red_ci90']['shiter'][1]:.0f}] | | "
                             f"{100 * PILOT_RED[(task, k)]:.0f}% |")
        base = [r for r in rows if r["task"] == task and r["variant"] == "none"]
        out["cells"][f"{task}:{k}"]["baseline"] = {"n": len(base), "success": float(np.mean([r["success"] for r in base])),
                                                   "ret": float(np.mean([r["ret"] for r in base]))}
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/repro_summary.json", "w"), indent=1)
    open(f"{a.out}/repro_table.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    for c, v in out["cells"].items():
        print(c, "baseline", v["baseline"])


def cmd_held(a):
    rows = [r for r in load_eps(a.res, "held_") if r["split"] == "held"]
    tasks = sorted({r["task"] for r in rows}, key=lambda t: (not t.startswith("mw-"), t))
    res, ktable = {}, {}
    lines = ["| 과제 | 기준(성공률 또는 return) | det k=3 | k=4 | k=5 | k=6 | 1차 k | 칸 | 1차 k(20 ep 문자 그대로 / ckpt1만) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for t in tasks:
        m = metric_of(t)
        ckpts = sorted({r["ckpt"] for r in rows if r["task"] == t})
        base = [r for r in rows if r["task"] == t and r["variant"] == "none"]
        info = {"metric": m, "n_base": len(base), "base_success": float(np.mean([r["success"] for r in base])),
                "base_ret": float(np.mean([r["ret"] for r in base])),
                "base_success_by_ckpt": {c: float(np.mean([r["success"] for r in base if r["ckpt"] == c]))
                                         for c in ckpts},
                "base_ret_by_ckpt": {c: float(np.mean([r["ret"] for r in base if r["ckpt"] == c])) for c in ckpts},
                "det": {}, "det_by_ckpt": {}}
        for k in (2, 3, 4, 5, 6, 7):
            p = paired(rows, t, ckpts, k, ["det"], m)
            if p is None:
                continue
            info["det"][k] = {"drop": p["drop"]["det"], "ci90": p["ci90"]["det"], "ci95": p["ci95"]["det"], "n": p["n"],
                              "mean": p["mean"]["det"]}
            info["det_by_ckpt"][k] = {c: (paired(rows, t, [c], k, ["det"], m) or {"drop": {"det": None}})["drop"]["det"]
                                      for c in ckpts}
            p1 = paired(rows, t, [1], k, ["det"], m)
            info["det"][k]["drop_ckpt1_only"] = p1["drop"]["det"] if p1 else None
        qualifies = (info["base_success"] >= 0.8) if m == "success" else True
        info["qualifies"] = bool(qualifies)
        ks = [k for k in (3, 4, 5, 6) if k in info["det"] and info["det"][k]["drop"] >= 0.20]
        prim = max(ks) if ks else None
        rule = "drop>=20"
        if prim is None:   # R2-13 fallback (design: 'R2-13 rule'): >= 15pp, else the smallest tested k
            ks15 = [k for k in (3, 4, 5, 6) if k in info["det"] and info["det"][k]["drop"] >= 0.15]
            prim, rule = (max(ks15), "fallback>=15") if ks15 else (3, "fallback_smallest")
        info["primary_k"], info["rule"] = prim, rule
        # supplementary held cells (information only, e.g. k-1 = 2 when the primary k is 3): every quantized variant
        # present at k in {2, 7}
        sup = {}
        for k in (2, 7):
            vs = sorted({r["variant"] for r in rows if r["task"] == t and r["k"] == k and r["variant"] != "none"})
            if vs:
                p = paired(rows, t, ckpts, k, vs, m)
                if p:
                    sup[k] = {v: {"drop": p["drop"][v], "ci90": p["ci90"][v], "mean": p["mean"][v]} for v in vs}
                    sup[k]["n"] = p["n"]
        info["supplementary"] = sup
        ks1 = [k for k in (3, 4, 5, 6) if k in info["det"] and info["det"][k].get("drop_ckpt1_only") is not None
               and info["det"][k]["drop_ckpt1_only"] >= 0.20]
        info["primary_k_ckpt1_only"] = max(ks1) if ks1 else None
        # literal reading "20 held episodes per task": episodes 0-6 of ckpt 1, 0-6 of ckpt 2, 0-5 of ckpt 3 (7+7+6)
        lit = [r for r in rows if r["task"] == t and r["ep"] < {1: 7, 2: 7, 3: 6}.get(r["ckpt"], 0)]
        info["det_20lit"] = {}
        for k in (3, 4, 5, 6):
            p20 = paired(lit, t, ckpts, k, ["det"], m)
            if p20 is not None:
                info["det_20lit"][k] = {"drop": p20["drop"]["det"], "n": p20["n"]}
        b20 = [r for r in lit if r["variant"] == "none"]
        info["base_success_20lit"] = float(np.mean([r["success"] for r in b20])) if b20 else None
        ks20 = [k for k in (3, 4, 5, 6) if k in info["det_20lit"] and info["det_20lit"][k]["drop"] >= 0.20]
        info["primary_k_20lit"] = max(ks20) if ks20 else None
        res[t] = info
        if qualifies:
            ktable[t] = {"primary": prim, "cells": [prim - 1, prim, prim + 1], "rule": rule}
        bs = (f"{100 * info['base_success']:.0f}% (" + "/".join(f"{100 * v:.0f}" for v in info['base_success_by_ckpt'].values())
              + ")") if m == "success" else f"{info['base_ret']:.1f} (" + "/".join(
            f"{v:.0f}" for v in info['base_ret_by_ckpt'].values()) + ")"
        dk = lambda k: (f"{100 * info['det'][k]['drop']:.0f} [{100 * info['det'][k]['ci90'][0]:.0f}, "
                        f"{100 * info['det'][k]['ci90'][1]:.0f}]" if k in info["det"] else "-")
        lines.append(f"| {t} | {bs} | {dk(3)} | {dk(4)} | {dk(5)} | {dk(6)} | "
                     f"{prim if qualifies else '제외'} | {ktable[t]['cells'] if qualifies else '-'} | "
                     f"{info['primary_k_20lit']} / {info['primary_k_ckpt1_only']} |")
    sup_lines = []
    for t in tasks:
        for k, d in res[t].get("supplementary", {}).items():
            sup_lines.append(f"| {t} | k={k} | n={d['n']} | " + " | ".join(
                f"{v} {100 * d[v]['drop']:.0f} [{100 * d[v]['ci90'][0]:.0f}, {100 * d[v]['ci90'][1]:.0f}]"
                for v in d if v != "n") + " |")
    if sup_lines:
        lines += ["", "보조 held 칸(정보용, 하락 %p 또는 %, 90% CI):", "", "| 과제 | k | 짝 수 | 변형별 하락 |",
                  "|---|---|---|---|"] + sup_lines
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/held_selection.json", "w"), indent=1, default=float)
    json.dump(ktable, open(f"{a.out}/ktable.json", "w"), indent=1)
    open(f"{a.out}/held_table.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def cmd_bench(a):
    """Throughput from the runner logs of run_bench.sh (fixed-duration runs; job finish times in s since start).
    eps_per_h_busy = W * 3600 / (median per-episode wall-clock of a job): all W workers busy (steady state);
    eps_per_h_window = episodes finished after the first job-duration (warm-up) / that part of the run.
    steps/h use 100 (MW) and 500 (DMC) control steps per episode."""
    import re
    pat = re.compile(r"^\s*(\d+)s \[(\d+)/(\d+)\] OK\s+(\S+) \((\d+)s\)")
    res = {}
    for f in sorted(glob.glob(f"{a.res}/E0_bench_*_*.log")):
        name = os.path.basename(f)[len("E0_bench_"):-4]
        if name.count("_") != 1:
            continue
        cfg, part = name.split("_")
        W = int(re.sub(r"[a-z]", "", cfg))
        ts, secs, end = [], [], None
        for line in open(f):
            m = pat.match(line)
            if m:
                ts.append(int(m.group(1)))
                secs.append(int(m.group(5)))
            mm = re.match(r"RUNNER_(STOPPED|DONE) .* in (\d+)s", line)
            if mm:
                end = int(mm.group(2))
        if not ts:
            continue
        ts.sort()
        epj = 2 if part == "mw" else 1
        steps = 100 if part == "mw" else 500
        per_ep = float(np.median(secs)) / epj
        end = end or ts[-1]
        warm = float(np.median(secs))
        win = [t for t in ts if t > warm]
        rate_win = len(win) * epj / max(end - warm, 1) * 3600 if win else float("nan")
        res[f"{cfg}_{part}"] = {"cfg": cfg, "part": part, "workers": W, "mps": not cfg.startswith("nomps"),
                                "instr": not cfg.endswith("n"), "jobs_done": len(ts), "episodes_done": len(ts) * epj,
                                "run_s": end, "sec_per_episode_per_worker": per_ep,
                                "eps_per_h_busy": W * 3600 / per_ep, "eps_per_h_window": rate_win,
                                "steps_per_h_busy": W * 3600 / per_ep * steps}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/throughput.json", "w"), indent=1)
    for k, v in sorted(res.items(), key=lambda kv: (kv[1]["part"], not kv[1]["mps"], not kv[1]["instr"], kv[1]["workers"])):
        print(f"{k:14s} W={v['workers']:2d} mps={v['mps']!s:5s} instr={v['instr']!s:5s} s/ep/worker "
              f"{v['sec_per_episode_per_worker']:6.1f}  eps/h busy {v['eps_per_h_busy']:7.0f} window "
              f"{v['eps_per_h_window']:7.0f}  done {v['episodes_done']} in {v['run_s']}s")


def cmd_e1est(a):
    """E1 load from <res>/ktable.json (or --ktable) and the measured throughput (<res>/throughput.json, config --cfg):
    episodes per family (30 eval episodes x 3 checkpoints per cell; none once per task and checkpoint) and the
    wall-clock hours on GPU 0 = MW episodes / MW rate + DMC episodes / DMC rate (busy-worker rates)."""
    kt = json.load(open(a.ktable or f"{a.res}/ktable.json"))
    thr = json.load(open(f"{a.res}/throughput.json"))
    n = {"mw": 0, "dmc": 0}
    cells = 0
    for t, v in kt.items():
        fam = "mw" if t.startswith("mw-") else "dmc"
        n[fam] += 90 * (1 + 6 * len(v["cells"]))
        cells += 3 * len(v["cells"])
    out = {"episodes": n, "cells_task_k_ckpt": cells, "by_cfg": {}}
    for cfg in sorted({v["cfg"] for v in thr.values()}):
        r = {p: thr.get(f"{cfg}_{p}", {}).get("eps_per_h_busy") for p in ("mw", "dmc")}
        if not all(r.values()):
            continue
        h = n["mw"] / r["mw"] + n["dmc"] / r["dmc"]
        out["by_cfg"][cfg] = {"rate_mw": r["mw"], "rate_dmc": r["dmc"], "hours_episodes": h,
                              "hours_mw": n["mw"] / r["mw"], "hours_dmc": n["dmc"] / r["dmc"]}
        print(f"{cfg:8s} MW {n['mw']} ep / {r['mw']:.0f} per h = {n['mw'] / r['mw']:.1f} h;  DMC {n['dmc']} ep / "
              f"{r['dmc']:.0f} per h = {n['dmc'] / r['dmc']:.1f} h;  total {h:.1f} h (+ per-cell prederr/decision jobs)")
    json.dump(out, open(f"{a.out}/e1_estimate.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["repro", "held", "bench", "e1est"])
    ap.add_argument("--ktable", default=None)
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"repro": cmd_repro, "held": cmd_held, "bench": cmd_bench, "e1est": cmd_e1est}[a.cmd](a)
