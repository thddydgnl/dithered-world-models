"""Smoke of the post-hoc QAT+none check (DESIGN_main v5.3 10.2) on one cell, WITHOUT MPS (GPU 0 is shared with a non-MPS
DINO-WM chain, so only identities and code paths are checked, no timing):
  1. the QAT weights of (task, ckpt, k) differ from the original dynamics / reward weights; load_state restores them
  2. the post-hoc job (runner.job_episodes with the qatnone spec, episode 0) loads the QAT file (recorded qat_path ->
     out_e5/qat) and leaves the original weights in place afterwards
  3. QAT weights + variant none does not quantize: Quantizer.mode is None and nq (quantized layer calls) = 0 in a direct
     run_episode; the same weights + det gives nq > 0
  4. identities with the stored E5 results of the same seed (E5 ran on GPU 0 with MPS, which does not change numbers):
     original weights + none = the E5 none episode; QAT weights + det = the E5 QAT+det episode; the runner path equals
     the direct QAT+none run; QAT+none differs from both stored episodes
usage (server, GPU 0, MPS pipe unset): python smoke_posthoc_qatnone.py --out <smoke out dir> --e5 <out_e5>
                                          --manifest <qatnone.jsonl> [--task mw-push --ckpt 1]
"""
import argparse
import gzip
import json
import os

import e5lib as E5
import qlib as L
import runner as R


def stored(e5, jid, ep):
    d = json.load(gzip.open(f"{e5}/jobs/episodes/{jid}.json.gz", "rt"))
    return next(e for e in d["episodes"] if int(e["ep"]) == ep)


def maxdiff(a, b):
    return max(float((a[m][n].float() - b[m][n].float()).abs().max()) for m in a for n in a[m])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--e5", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--task", default="mw-push")
    ap.add_argument("--ckpt", type=int, default=1)
    a = ap.parse_args()
    if os.environ.get("CUDA_MPS_PIPE_DIRECTORY"):
        raise SystemExit("REFUSED: unset the MPS pipe (GPU 0 is shared with a non-MPS process)")
    out, e5 = os.path.abspath(a.out), os.path.abspath(a.e5)
    jobs = [json.loads(l) for l in open(a.manifest) if l.strip()]
    j = next(x for x in jobs if x["task"] == a.task and int(x["ckpt"]) == a.ckpt and 0 in x["eps"])
    k, seed = int(j["k"]), int(j["seed0"])
    spec = dict(j, eps=[0], id=j["id"] + "_smoke")
    A = R.get_agent(a.task, a.ckpt)
    agent, cfg, quant = A["agent"], A["cfg"], A["quant"]
    orig = E5.model_state(agent)
    rep = {"task": a.task, "ckpt": a.ckpt, "k": k, "seed": seed}
    # 1. QAT weights differ, restore works
    saved = E5.load_qat(agent, out, a.task, a.ckpt, k)
    rep["qat_vs_orig_maxabs"] = maxdiff(E5.model_state(agent), orig)
    E5.load_state(agent, saved)
    rep["restored_vs_orig_maxabs"] = maxdiff(E5.model_state(agent), orig)
    # 2. the real job path (QAT + none, episode 0)
    res = R.job_episodes(spec, out)
    rep["job_qat_path"] = res.get("qat_path")
    rep["job_qat_path_real"] = os.path.realpath(res.get("qat_path", ""))
    rep["job_ret"] = res["episodes"][0]["ret"]
    rep["after_job_vs_orig_maxabs"] = maxdiff(E5.model_state(agent), orig)
    # 3. direct runs with the QAT weights: none (no quantization) and det
    saved = E5.load_qat(agent, out, a.task, a.ckpt, k)
    try:
        L.set_planner(agent, None)
        r_none = L.run_episode(agent, cfg, quant, "none", k, seed)
        rep["qat_none"] = {"mode": quant.mode, "nq": int(quant.nq), "ret": r_none["ret"], "success": r_none["success"]}
        quant.qrange = R.load_ranges(out, a.task, a.ckpt, "main")
        r_det = L.run_episode(agent, cfg, quant, "det", k, seed)
        rep["qat_det"] = {"mode": quant.mode, "nq": int(quant.nq), "ret": r_det["ret"], "success": r_det["success"]}
    finally:
        quant.off()
        E5.load_state(agent, saved)
    r_orig = L.run_episode(agent, cfg, quant, "none", 0, seed)
    rep["orig_none"] = {"nq": int(quant.nq), "ret": r_orig["ret"], "success": r_orig["success"]}
    quant.off()
    # 4. stored E5 episodes of the same seed
    s_none = stored(e5, f"e5_{a.task}_s{a.ckpt}_k0_none_base_c00", 0)
    s_qdet = stored(e5, f"e5_{a.task}_s{a.ckpt}_k{k}_det_e5qat_c00", 0)
    rep["stored_e5_none_ret"], rep["stored_e5_qatdet_ret"] = s_none["ret"], s_qdet["ret"]
    want_qat = os.path.realpath(f"{e5}/qat/{a.task}_s{a.ckpt}_k{k}.pt")
    rep["checks"] = {
        "qat_weights_differ": rep["qat_vs_orig_maxabs"] > 0,
        "restore_exact": rep["restored_vs_orig_maxabs"] == 0 and rep["after_job_vs_orig_maxabs"] == 0,
        "job_reads_e5_qat_file": rep["job_qat_path_real"] == want_qat,
        "none_does_not_quantize": rep["qat_none"]["mode"] is None and rep["qat_none"]["nq"] == 0,
        "det_quantizes": rep["qat_det"]["nq"] > 0,
        "orig_none_equals_stored_e5_none": rep["orig_none"]["ret"] == s_none["ret"],
        "qat_det_equals_stored_e5_qatdet": rep["qat_det"]["ret"] == s_qdet["ret"],
        "job_equals_direct_qat_none": rep["job_ret"] == rep["qat_none"]["ret"],
        "qat_none_differs_from_both": rep["qat_none"]["ret"] not in (s_none["ret"], s_qdet["ret"])}
    rep["ALL_OK"] = bool(all(rep["checks"].values()))
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
