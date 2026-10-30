"""v5.2 (b3) region-effect reference: code-path checks on smoke seeds (nothing here is analysed).
For every E2 base variant: the closed loop (return, action-sequence hash, refresh count, call log), every existing
per-step field and the plan-rerun records are bitwise the same with region_ref off and on (E2 instrumentation: OEI
every 4th step, plan reruns every 20th, independent re-evaluation on); the new fields exist; none has drift_ref = 0.
usage (server): python test_b3_paths.py
"""
import hashlib
import json
import math
import os

import torch

import qlib as L

R28 = os.path.expanduser("~/pmx/A/tdmpc2_main/inputs/r28")


def ahash(acts):
    return hashlib.sha1(torch.stack([torch.as_tensor(a) for a in acts]).float().numpy().tobytes()).hexdigest()[:16]


def same(a, b):
    return (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)) or a == b


def main():
    ok_all = True
    for task, k, seed, variants in (("mw-push", 3, 19004, ("none", "det", "shiter", "ind", "shstep", "shep",
                                                              "detshift")),
                                    ("cheetah-run", 4, 19004, ("det", "shiter"))):
        agent, cfg = L.build_agent(task, 1)
        quant = L.Quantizer(agent.model, horizon=int(cfg.horizon))
        qr = json.load(open(f"{R28}/prep_{task}.json"))["quant_range"]
        quant.qrange = {kk: (float(v[0]), float(v[1])) for kk, v in qr.items()}
        probe = L.PlanProbe(agent, quant)
        mw = task.startswith("mw-")
        for v in variants:
            R = {}
            for ref in (False, True):
                R[ref] = L.run_episode(agent, cfg, quant, v, 0 if v == "none" else k, seed, probe=probe,
                                       oei_stride=4 if mw else 10, pd_stride=20 if mw else 50, log_calls=True,
                                       indep_reeval=True, region_ref=ref)
            a, b = R[False], R[True]
            closed = (a["ret"] == b["ret"] and ahash(list(a["_act"])) == ahash(list(b["_act"])) and
                      a["n_refresh"] == b["n_refresh"] and a["calls_patterns"] == b["calls_patterns"])
            nd = sum(0 if same(x[kk], y.get(kk)) else 1 for x, y in zip(a["steps_rec"], b["steps_rec"]) for kk in x)
            fields = len(a["steps_rec"]) == len(b["steps_rec"]) and nd == 0 and a["pd_rec"] == b["pd_rec"]
            sb = b["steps_rec"]
            new = sorted(set(sb[0]) - set(a["steps_rec"][0]))
            dref = [x["drift_ref"] for x in sb]
            rec = {"task": task, "variant": v, "ret": a["ret"], "closed_loop_identical": closed,
                   "existing_fields_identical": fields, "n_field_diffs": nd, "new_keys": new,
                   "drift_ref_mean": sum(dref) / len(dref), "drift_mean": sum(x["drift"] for x in sb) / len(sb)}
            ok = closed and fields and set(new) == {"ref_f_e_rand", "ref_l_e_rand", "drift_ref", "drift_minus_ref"}
            if v == "none":
                ok &= all(x == 0.0 for x in dref)
            rec["ok"] = bool(ok)
            ok_all &= ok
            print(json.dumps(rec), flush=True)
        probe.detach()
        quant.remove()
    print("ALL_OK" if ok_all else "SOME_FAILED", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
