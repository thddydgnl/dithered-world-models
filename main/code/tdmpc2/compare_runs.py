"""Bitwise comparison of re-run episode jobs with stored reference results (same job ids).

Every field of every reference episode (returns, success, step counts, dither-table refresh count, call log, per-step
OEI / decision records, plan-rerun records, prediction errors) must be equal in the re-run; the re-run may only add
fields (E2/E3 re-evaluations). Wall-clock fields are skipped; --ignore drops whole reference fields that the re-run
does not compute (e.g. perr_traj, perr_plan for the H2a rerun). Used for
  (1) the v5 identity check on the smoke split (reference: the E0 smoke results, E1-era code), and
  (2) the H2a rerun (reference: the E1 results); analyze_h2a_indep.py calls compare_job().
usage: python compare_runs.py --ref <out dir> --new <out dir> --manifest <rerun manifest> [--ignore perr_traj,perr_plan]
"""
import argparse
import gzip
import json
import math
import os

SKIP = {"secs", "secs_loop"}


def _eq(a, b, path, diffs):
    if isinstance(a, dict):
        if not isinstance(b, dict):
            diffs.append((path, "type"))
            return
        for k in a:
            if k in SKIP:
                continue
            if k not in b:
                diffs.append((f"{path}.{k}", "missing"))
                continue
            _eq(a[k], b[k], f"{path}.{k}", diffs)
        return
    if isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            diffs.append((path, "length"))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _eq(x, y, f"{path}[{i}]", diffs)
        return
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return
    if a != b:
        diffs.append((path, f"{a!r} != {b!r}"))


def load(path):
    return json.load(gzip.open(path, "rt"))


def compare_job(ref_path, new_path, ignore=()):
    """-> dict(n_eps, diffs [(path, what)], new_oei_keys)."""
    A, B = load(ref_path), load(new_path)
    ea, eb = A["episodes"], B["episodes"]
    diffs = []
    if [e["ep"] for e in ea] != [e["ep"] for e in eb]:
        diffs.append(("episodes", "episode lists differ"))
    for x, y in zip(ea, eb):
        xa = {k: v for k, v in x.items() if k not in ignore}
        _eq(xa, y, f"ep{x['ep']}", diffs)
    new_keys = sorted(set((eb[0].get("oei_mean") or {})) - set((ea[0].get("oei_mean") or {}))) if ea and eb else []
    # the spec may differ only in the instrumentation switches of the re-run
    sa, sb = dict(A["spec"]), dict(B["spec"])
    ia, ib = sa.pop("instr", {}), sb.pop("instr", {})
    if sa != sb:
        diffs.append(("spec", "job spec differs outside instr"))
    for k in ("oei_stride", "pd_stride"):
        if ia.get(k) != ib.get(k):
            diffs.append((f"spec.instr.{k}", f"{ia.get(k)} != {ib.get(k)}"))
    return {"n_eps": len(eb), "diffs": diffs, "new_oei_keys": new_keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ignore", default="")
    a = ap.parse_args()
    ignore = tuple(x for x in a.ignore.split(",") if x)
    jobs = [json.loads(l) for l in open(a.manifest) if l.strip()]
    n_ok = n_bad = n_missing = n_eps = 0
    for j in jobs:
        rp, np_ = (f"{d}/jobs/{j['type']}/{j['id']}.json.gz" for d in (a.ref, a.new))
        if not (os.path.exists(rp) and os.path.exists(np_)):
            n_missing += 1
            continue
        r = compare_job(rp, np_, ignore)
        n_eps += r["n_eps"]
        if r["diffs"]:
            n_bad += 1
            print(f"DIFF {j['id']}: {len(r['diffs'])} fields, first: {r['diffs'][:4]}", flush=True)
        else:
            n_ok += 1
            print(f"SAME {j['id']} ({r['n_eps']} ep; new oei fields: {' '.join(r['new_oei_keys'])})", flush=True)
    print(f"COMPARE_DONE identical {n_ok} differing {n_bad} missing {n_missing} episodes {n_eps}", flush=True)


if __name__ == "__main__":
    main()
