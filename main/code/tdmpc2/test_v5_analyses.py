"""Synthetic result files for dry runs of the v5 analyses (analyze_e2 / e3 / e5 / e7): one fake result per episode
job of a manifest, written to a scratch output directory (never the real ones). Numbers are random and meaningless.
usage: python test_v5_analyses.py --manifest <m.jsonl> --out <scratch dir> [--seed 0]
"""
import argparse
import gzip
import json
import os

import numpy as np

TABLE = ("shstep", "shep", "indstep", "indep", "shstep_u", "shstep_g32", "shstep_l", "shep_u", "shep_g32", "shep_l")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    n = 0
    for l in open(a.manifest):
        if not l.strip():
            continue
        j = json.loads(l)
        if j["type"] != "episodes":
            continue
        mw = j["task"].startswith("mw-")
        v = j["variant"]
        p_succ = {"none": 0.97, "det": 0.6}.get(v, 0.9)
        eps = []
        for ep in j["eps"]:
            succ = float(rng.random() < p_succ)
            ret = float(rng.normal(1500 if mw else 800, 200 if mw else 60) * (0.7 if v == "det" else 1.0))
            om = {}
            if v != "none":
                drift = float(rng.normal(-20 if v in ("det",) + TABLE else -2, 5))
                sel = float(rng.normal(-15, 5))
                om = {"oei": drift + sel, "sel_last": sel, "drift": drift, "oei_indep": drift + sel * 0.2,
                      "sel_indep": sel * 0.2, "chosen_pi": float(rng.random() < 0.02), "elite_pi": 0.01,
                      "elite_pi_w": 0.01, "e_chosen": 0.0, "e_indep": 0.0, "f_e_rand": 1.0, "l_e_rand": 1.0,
                      "f_sd_v0": 30.0, "f_var_cand": float(rng.gamma(2.0, 100.0))}
                if v in TABLE:
                    om.update(oei_fresh=drift * 0.5 + sel * 0.2, sel_fresh=sel * 0.2)
                if j["instr"].get("region_ref"):
                    dref = float(rng.normal(5, 3))
                    om.update(drift_ref=dref, drift_minus_ref=drift - dref, ref_f_e_rand=1.0, ref_l_e_rand=1.0 + dref)
            elif j["instr"].get("region_ref"):
                om = {"drift_ref": 0.0, "drift_minus_ref": 0.0, "drift": 0.0}
            eps.append({"ep": ep, "seed": j["seed0"] + ep, "ret": ret if not mw else ret, "success": succ,
                        "success_any": succ, "steps": 100 if mw else 500, "n_refresh": 0, "secs_loop": 1.0,
                        "secs": 1.0, "oei_mean": om})
        res = {"episodes": eps, "spec": j, "ok": True}
        if j.get("ptq"):
            res["ptq_resolved"] = j["ptq"] if j["ptq"] != "best" else "perchan"
        if j.get("e7"):
            res["e7"] = {"prec": j["e7"], "variant": v, "k": j["k"]}
        p = f"{a.out}/jobs/episodes/{j['id']}.json.gz"
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with gzip.open(p, "wt") as f:
            json.dump(res, f)
        n += 1
    print(f"{n} fake job results -> {a.out}")


if __name__ == "__main__":
    main()
