"""Analysis of the POST-HOC TD-MPC2 QAT+none check (DESIGN_main v5.3 10.2; descriptive, no threshold, no verdict).
Written before the check ran.
  - per QAT cell (11 cells, E5 seeds 60000-60029, 90 pairs): D of QAT+none (QAT weights, no activation quantization)
    against the E5 none of the ORIGINAL weights (same pairs), with the paired bootstrap 95% CI (v4 6.1: RandomState(0),
    10,000 resamples of the sorted (checkpoint, episode) pairs, DMC relative return drop recomputed per resample)
  - in the same table and on the same pairs: D(QAT+det), D(QAT+sh-iter) (E5) and, for reference, D(B+det),
    D(B+sh-iter) of the best PTQ baseline (E5 H3b cells)
  - D(QAT+det) - D(QAT+none) with its paired CI (the activation-quantization damage on top of the QAT weights)
  - completeness: missing QAT+none jobs of the manifest, and that every QAT+none result read the E5 QAT file
usage (server, CPU only): python analyze_posthoc_qatnone.py --e5 ../out_e5 --res ../out_posthoc_qatnone \
                              --manifest ../posthoc/manifests/qatnone.jsonl --outdir ../analysis_posthoc_qatnone
"""
import argparse
import glob
import gzip
import json
import os

import v5_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e5", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    e5, res = os.path.abspath(a.e5), os.path.abspath(a.res)
    jobs = [json.loads(l) for l in open(a.manifest) if l.strip()]
    missing = [j["id"] for j in jobs if not os.path.exists(f"{res}/jobs/episodes/{j['id']}.json.gz")]
    bad_path = []
    for f in sorted(glob.glob(f"{res}/jobs/episodes/e5_*_e5qatnone_c*.json.gz")):
        d = json.load(gzip.open(f, "rt"))
        s = d["spec"]
        want = os.path.realpath(f"{e5}/qat/{s['task']}_s{s['ckpt']}_k{s['k']}.pt")
        if os.path.realpath(d.get("qat_path", "")) != want or s["variant"] != "none":
            bad_path.append(s["id"])
    E = C.load_episodes(e5, prefix="e5_")
    E.update(C.load_episodes(res, prefix="e5_"))
    out = {"label": "post-hoc (DESIGN_main v5.3 10.2), descriptive", "n_jobs_missing": len(missing),
           "results_not_reading_the_e5_qat_file": bad_path, "cells": {}}
    cells = sorted({(k[1], k[2]) for k in E if k[0] == "e5qatnone"})
    for t, k in cells:
        mw = C.is_mw(t)
        kv = {"qat_none": ("e5qatnone", t, k, "none"), "qat_det": ("e5qat", t, k, "det"),
              "qat_shiter": ("e5qat", t, k, "shiter"), "B_det": ("e5best", t, k, "det"),
              "B_shiter": ("e5best", t, k, "shiter")}
        kv = {n: x for n, x in kv.items() if any(key[:4] == x for key in E)}
        prs = {(key[4], key[5]) for key in E if key[:4] == ("base", t, 0, "none")}
        for x in kv.values():
            prs &= {(key[4], key[5]) for key in E if key[:4] == x}
        P = C.Pairs(prs)
        if P.n == 0:
            continue
        y0 = P.arr(E, lambda c, e: ("base", t, 0, "none", c, e), mw=mw)
        row = {"n_pairs": P.n, "none_mean": float(y0.mean()), "family": "mw" if mw else "dmc"}
        Db = {}
        for n, x in kv.items():
            y = P.arr(E, lambda c, e, x=x: x + (c, e), mw=mw)
            st, Db[n] = P.drop_ci(y0, y, mw)
            row[n] = dict(st, mean=float(y.mean()))
        if "qat_none" in Db and "qat_det" in Db:
            row["qat_det_minus_qat_none"] = row["qat_det"]["D"] - row["qat_none"]["D"]
            row["qat_det_minus_qat_none_ci"] = C.A1.ci95(Db["qat_det"] - Db["qat_none"])
        if "qat_none" in Db and "B_det" in Db:
            row["qat_none_minus_B_det"] = row["qat_none"]["D"] - row["B_det"]["D"]
            row["qat_none_minus_B_det_ci"] = C.A1.ci95(Db["qat_none"] - Db["B_det"])
        out["cells"][f"{t}|{k}"] = row
    os.makedirs(a.outdir, exist_ok=True)
    C.write_json(out, f"{a.outdir}/posthoc_qatnone_summary.json")
    f2, ci = C.f2, C.ci_txt
    L = ["# TD-MPC2 QAT+none (사후, v5.3 10.2, 기술)", "",
         f"빠진 job {len(missing)}, E5 QAT 파일을 읽지 않은 결과 {len(bad_path)}", "",
         "D는 원래 가중치 E5 none 대비(같은 90짝). CI: 짝 bootstrap 95%.", "",
         "| 칸 | D(QAT+none) [CI] | D(QAT+det) [CI] | D(QAT+sh-iter) [CI] | D(QAT+det) − D(QAT+none) [CI] | 참고 D(B+det) | 참고 D(B+sh-iter) |",
         "|---|---|---|---|---|---|---|"]
    for c, r in out["cells"].items():
        g = lambda n: f"{f2(r[n]['D'], 3)} {ci(r[n]['D_ci'], 1, 3)}" if n in r else "—"
        L.append(f"| {c} | {g('qat_none')} | {g('qat_det')} | {g('qat_shiter')} | "
                 f"{f2(r.get('qat_det_minus_qat_none'), 3)} {ci(r.get('qat_det_minus_qat_none_ci'), 1, 3)} | "
                 f"{f2(r.get('B_det', {}).get('D'), 3)} | {f2(r.get('B_shiter', {}).get('D'), 3)} |")
    open(f"{a.outdir}/posthoc_qatnone_tables.md", "w").write("\n".join(L) + "\n")
    print(json.dumps({"missing": len(missing), "bad_path": len(bad_path), "cells": len(out["cells"])}))


if __name__ == "__main__":
    main()
