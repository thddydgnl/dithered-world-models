"""POST-HOC (decided after the one-shot E3 final result was read; not a pre-registered statistic, never a verdict):
the DMC-explanation inputs (S_det, near-tie density, red_sh-iter on the E3 seeds) re-read from e3_tdmpc2_final.json and
correlated within each TD-MPC2 sub-family (Meta-World 8 tasks, DMC 4 tasks) beside the pre-registered 12-task value,
to show whether the pooled Spearman only separates the two sub-families. Reads the stored rows; recomputes nothing
of the final result.
usage: python posthoc_e3_dmc_within_family.py --final <analysis_e3/e3_tdmpc2_final.json> --out <json>
"""
import argparse
import json

import analyze_e1 as A1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = json.load(open(a.final))["DMC_explanation"]["rows"]
    out = {"label": "post-hoc, descriptive (after the one-shot E3 final result); not used for any verdict", "groups": {}}
    for lab, sel in (("all_12_preregistered_value", lambda r: True), ("meta_world_8", lambda r: r["task"].startswith("mw-")),
                     ("dmc_4", lambda r: not r["task"].startswith("mw-"))):
        R = [r for r in rows if sel(r) and r["det_drop_real_e3"] and r["S_defined"] and r["red_shiter"] is not None]
        out["groups"][lab] = {"tasks": [r["task"] for r in R],
                              "spearman_S_red": A1.spearman([r["S_det"] for r in R], [r["red_shiter"] for r in R]),
                              "spearman_density_red": A1.spearman([r["density"] for r in R], [r["red_shiter"] for r in R])}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: {kk: v[kk] for kk in ("spearman_S_red", "spearman_density_red")} for k, v in out["groups"].items()}))


if __name__ == "__main__":
    main()
