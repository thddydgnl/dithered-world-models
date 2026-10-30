"""v5 code-path checks (smoke seeds only; nothing here is analysed):
  1. the E2 independent-draw re-evaluations leave the closed loop bit-identical (returns, action hashes, dither-table
     refresh count, quantizer call log) for E1 and E3 variants, probe off / probe on / probe on + re-evaluations;
  2. every per-step field of the probe-on run (all E1 fields and the plan-rerun records) is bitwise the same when the
     re-evaluations are switched on (the new fields are only added);
  3. e_indep == e_chosen exactly for det and the table variants (indep_same = 1) and differs for fresh-noise variants;
     e_fresh (tables redrawn) exists only for the table variants and differs from e_chosen (fresh_same = 0);
  4. sel_last + drift == oei and sel_indep + drift == oei_indep per plan;
  5. E1 variants still give the episodes of the E0 smoke (mw-push k=3, R2-8 ranges, seed 19000: det 116.36395911313593).
Usage (server): python test_v5_paths.py
"""
import hashlib
import json
import math
import os

import torch

import qlib as L

R28 = os.path.expanduser("~/pmx/A/tdmpc2_main/inputs/r28")
E0_DET_19000 = 116.36395911313593


def ahash(acts):
    return hashlib.sha1(torch.stack([torch.as_tensor(a) for a in acts]).float().numpy().tobytes()).hexdigest()[:16]


def same_val(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def main():
    agent, cfg = L.build_agent("mw-push", 1)
    quant = L.Quantizer(agent.model, horizon=int(cfg.horizon))
    qr = json.load(open(f"{R28}/prep_mw-push.json"))["quant_range"]
    quant.qrange = {k: (float(v[0]), float(v[1])) for k, v in qr.items()}
    probe = L.PlanProbe(agent, quant)
    ok_all = True
    for v, seed in (("det", 19000), ("shiter", 19000), ("ind", 19001), ("shstep", 19001), ("detshift", 19001),
                    ("shep", 19001), ("indstep", 19002), ("indep", 19002), ("shstep_u", 19002), ("shstep_g32", 19002),
                    ("shstep_l", 19002), ("shiter_l", 19002)):
        R = {}
        for mode in ("off", "oei", "oei+indep"):
            R[mode] = L.run_episode(agent, cfg, quant, v, 3, seed, probe=probe if mode != "off" else None,
                                    oei_stride=4 if mode != "off" else 0, pd_stride=20, log_calls=True,
                                    indep_reeval=(mode == "oei+indep"))
        h = {m: ahash(list(R[m]["_act"])) for m in R}
        closed = (R["off"]["ret"] == R["oei"]["ret"] == R["oei+indep"]["ret"] and h["off"] == h["oei"] == h["oei+indep"]
                  and R["off"]["n_refresh"] == R["oei"]["n_refresh"] == R["oei+indep"]["n_refresh"]
                  and R["off"]["calls_patterns"] == R["oei"]["calls_patterns"] == R["oei+indep"]["calls_patterns"])
        a, b = R["oei"]["steps_rec"], R["oei+indep"]["steps_rec"]
        n_diff = sum(0 if same_val(x[kk], y.get(kk)) else 1 for x, y in zip(a, b) for kk in x)
        fields_same = len(a) == len(b) and n_diff == 0 and R["oei"]["pd_rec"] == R["oei+indep"]["pd_rec"] \
            and R["off"]["pd_rec"] == R["oei"]["pd_rec"]
        new_keys = sorted(set(b[0]) - set(a[0]))
        s = b
        rec = {"variant": v, "seed": seed, "ret": R["off"]["ret"], "closed_loop_identical": closed,
               "existing_fields_identical": fields_same, "n_field_diffs": n_diff, "new_keys": new_keys,
               "indep_same_frac": sum(x["indep_same"] for x in s) / len(s),
               "max_abs_decomp_err": max(max(abs(x["sel_last"] + x["drift"] - x["oei"]),
                                             abs(x["sel_indep"] + x["drift"] - x["oei_indep"])) for x in s),
               "chosen_pi_frac": sum(x["chosen_pi"] for x in s) / len(s)}
        if "e_fresh" in s[0]:
            rec["fresh_same_frac"] = sum(x["fresh_same"] for x in s) / len(s)
        exp_same = 1.0 if (v == "det" or v in L.PERSIST_TABLES) else None
        ok = closed and fields_same and rec["max_abs_decomp_err"] < 1e-9
        if exp_same is not None:
            ok &= rec["indep_same_frac"] == exp_same
        else:
            ok &= rec["indep_same_frac"] < 1.0
        ok &= ("e_fresh" in s[0]) == (v in L.PERSIST_TABLES)
        if "fresh_same_frac" in rec:
            ok &= rec["fresh_same_frac"] == 0.0
        if v == "det" and seed == 19000:
            rec["e0_smoke_ret_match"] = R["off"]["ret"] == E0_DET_19000
            ok &= rec["e0_smoke_ret_match"]
        rec["ok"] = bool(ok)
        ok_all &= ok
        print(json.dumps(rec), flush=True)
    print("ALL_OK" if ok_all else "SOME_FAILED", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
