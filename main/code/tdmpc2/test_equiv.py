"""E0 implementation checks (run on the server, smoke seeds only; nothing here is analysed as a result).
  python test_equiv.py pilot   episodes of the pilot library (pilot_r28b/nt_lib.py) for the 5 pilot variants
  python test_equiv.py qlib    the same episodes with qlib (must be bit-identical), every variant with and without the
                               E1 instrumentation (must be bit-identical), and sanity values of 'none' (OEI = 0,
                               plan deviation = 0)
Both print one JSON line per episode: task, variant, seed, instr flag, return, success, hash of the action sequence.
"""
import hashlib
import json
import os
import sys

import torch

TASK, K, SEEDS = "mw-push", 3, (19000, 19001)
R28 = os.path.expanduser("~/pmx/A/tdmpc2_main/inputs/r28")


def ranges(task):
    qr = json.load(open(f"{R28}/prep_{task}.json"))["quant_range"]
    return {k: (float(v[0]), float(v[1])) for k, v in qr.items()}


def ahash(acts):
    return hashlib.sha1(torch.stack([torch.as_tensor(a) for a in acts]).float().numpy().tobytes()).hexdigest()[:16]


def run_pilot():
    sys.path.insert(0, os.path.expanduser("~/pmx/A/tdmpc2_main/pilot_r28b"))
    import nt_lib as P
    agent, cfg = P.build_agent(TASK)
    ab = P.Ablator(agent.model, horizon=int(cfg.horizon))
    ab.qrange = ranges(TASK)
    for v in ("det", "ind", "shfwd", "shstep", "shep"):
        for seed in SEEDS:
            ab.set(**P.setting_for(f"Q{v}{K}", float(K)), seed=seed * 7 + 13)
            acts = []
            r = P.run_episode(agent, cfg, seed, before_act=lambda t, o: ab.begin_step(),
                              after_act=lambda t, o, a: acts.append(a.clone()))
            ab.set(None)
            print(json.dumps({"lib": "pilot", "variant": {"shfwd": "shiter"}.get(v, v), "seed": seed, "instr": False,
                              "ret": r["ret"], "success": r["success"], "ahash": ahash(acts)}), flush=True)


def run_qlib():
    import qlib as L
    agent, cfg = L.build_agent(TASK, 1)
    quant = L.Quantizer(agent.model, horizon=int(cfg.horizon))
    quant.qrange = ranges(TASK)
    probe = L.PlanProbe(agent, quant)
    for v in L.VARIANTS:
        for seed in SEEDS:
            for on in (False, True):
                r = L.run_episode(agent, cfg, quant, v, K, seed, probe=probe if on else None,
                                  oei_stride=4 if on else 0, pd_stride=10 if on else 0, keep_cands=on)
                rec = {"lib": "qlib", "variant": v, "seed": seed, "instr": on, "ret": r["ret"],
                       "success": r["success"], "ahash": ahash(list(r["_act"]))}
                if on:
                    s = r["steps_rec"]
                    rec.update(oei=sum(x["oei"] for x in s) / len(s), f_e_rand=sum(x["f_e_rand"] for x in s) / len(s),
                               f_spear=sum(x["f_spear"] for x in s) / len(s),
                               f_sn_spear=sum(x["f_sn_spear"] for x in s) / len(s),
                               dev_mean=sum(x["dev_mean"] for x in r["pd_rec"]) / len(r["pd_rec"]),
                               sn_mean=sum(x["sn_mean"] for x in r["pd_rec"]) / len(r["pd_rec"]),
                               n_oei=len(s), n_pd=len(r["pd_rec"]))
                print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        run_pilot() if sys.argv[1] == "pilot" else run_qlib()
