"""E7 kernel correctness tests (single process; smoke seeds; nothing here is analysed).
  1. kernels against PyTorch references of the same math: fused LN/Mish/quantize (det codes; dither: one u per column
     shared by all rows of a call, unbiased codes), fused int8 GEMM + LN/Mish/quantize, int8 GEMM + dequantize;
  2. the integer dynamics / reward MLPs against the E1 fake-quant path with the same W8 weights (det, k = 8 and 4) and
     against fp32, on planning-time inputs of a held calibration episode;
  3. closed-loop smoke episodes for every precision (mw-push, checkpoint 1) and the restore check (fp32 episode
     before and after the integer jobs bitwise equal); rough per-step times (shared GPU: indicative only).
usage (server): python e7_test.py [--quick]
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

import qlib as L
import e7lib as E7


def ref_ln_mish_q(y, g, b, lo, hi, d, nl, zp):
    z = F.layer_norm(y, (y.shape[-1],), g, b, E7.LN_EPS)
    m = F.mish(z)
    x = (m.clamp(lo, hi) - lo) * (1.0 / d)
    return (torch.round(x).clamp(0, nl) - zp).to(torch.int8), x


def kernel_tests(dev="cuda"):
    out = {}
    g_ = torch.Generator(device=dev)
    g_.manual_seed(0)
    M, N = 512, 512
    y = torch.randn((M, N), generator=g_, device=dev) * 2.0
    g = 1.0 + 0.1 * torch.randn(N, generator=g_, device=dev)
    b = 0.1 * torch.randn(N, generator=g_, device=dev)
    for k in (8, 4):
        nl, zp = (1 << k) - 1, (128 if k == 8 else 0)
        lo, hi = -0.31, 3.0
        d = (hi - lo) / nl
        q = E7.ln_mish_quant(y, g, b, lo, hi, d, nl, zp, 123, False)
        qr, x = ref_ln_mish_q(y, g, b, lo, hi, d, nl, zp)
        diff = (q.int() - qr.int()).abs()
        out[f"ln_mish_quant_det_k{k}"] = {"mismatch_frac": float((diff > 0).float().mean()), "max_abs": int(diff.max())}
        # dither: identical rows -> identical codes within a call; different calls differ; unbiased
        yr = y[:1].repeat(M, 1)
        q1 = E7.ln_mish_quant(yr, g, b, lo, hi, d, nl, zp, 7, True)
        q2 = E7.ln_mish_quant(yr, g, b, lo, hi, d, nl, zp, 8, True)
        acc = torch.zeros(N, device=dev)
        _, x1 = ref_ln_mish_q(y[:1], g, b, lo, hi, d, nl, zp)
        S = 400
        for s in range(S):
            acc += E7.ln_mish_quant(y[:1].contiguous(), g, b, lo, hi, d, nl, zp, 1000 + s, True)[0].float() + zp
        interior = (x1[0] > 0.5) & (x1[0] < nl - 0.5)
        out[f"ln_mish_quant_dither_k{k}"] = {
            "rows_identical_within_call": bool((q1 == q1[:1]).all()), "calls_differ": bool((q1 != q2).any()),
            "mean_code_minus_x_interior": float((acc / S - x1[0])[interior].abs().mean()),
            "se_bound": float(0.5 / np.sqrt(S))}
        # int8 GEMM + LN/Mish/quantize
        a = torch.randint(-zp, nl - zp + 1, (M, 512), generator=g_, device=dev, dtype=torch.int8)
        W = torch.randn((N, 512), generator=g_, device=dev) * 0.05
        Wq, s = E7.w8_per_channel(W)
        alpha = (d * s).float()
        c = (0.1 * torch.randn(N, generator=g_, device=dev)).float()
        WqT = Wq.t().contiguous().to(torch.int8)
        qk = E7.i8gemm_ln_mish_quant(a, WqT, alpha, c, g, b, lo, hi, d, nl, zp, 5, False)
        yref = (a.float() @ Wq.t()) * alpha + c
        qref, _ = ref_ln_mish_q(yref, g, b, lo, hi, d, nl, zp)
        diff = (qk.int() - qref.int()).abs()
        out[f"i8gemm_ln_mish_quant_det_k{k}"] = {"mismatch_frac": float((diff > 0).float().mean()),
                                                 "max_abs": int(diff.max())}
        # int8 GEMM + dequantize (head, N = 101 padded)
        Wh = torch.randn((101, 512), generator=g_, device=dev) * 0.05
        Whq, sh = E7.w8_per_channel(Wh)
        al = (d * sh).float()
        ch = (0.1 * torch.randn(101, generator=g_, device=dev)).float()
        pad = 128 - 101
        WhT = torch.cat([Whq.t().to(torch.int8), torch.zeros((512, pad), device=dev, dtype=torch.int8)], 1).contiguous()
        alp = torch.cat([al, torch.zeros(pad, device=dev)]).contiguous()
        chp = torch.cat([ch, torch.zeros(pad, device=dev)]).contiguous()
        yk = E7.i8gemm_dequant(a, WhT, alp, chp, 101)
        yr2 = (a.float() @ Whq.t()) * al + ch
        out[f"i8gemm_dequant_k{k}"] = {"max_abs_diff": float((yk - yr2).abs().max()),
                                       "max_abs_ref": float(yr2.abs().max())}
    return out


@torch.no_grad()
def model_tests(task="mw-push", ckpt=1):
    from calib_lib import load_split
    agent, cfg = L.build_agent(task, ckpt)
    quant = L.Quantizer(agent.model, horizon=int(cfg.horizon))
    out_dir = L.os.path.expanduser("~/pmx/A/tdmpc2_main/out")
    qr = json.load(open(f"{out_dir}/scales/{task}_s{ckpt}.json"))["quant"]
    quant.qrange = {k: (float(v[0]), float(v[1])) for k, v in qr.items()}
    Zu, idx, A = load_split(out_dir, task, ckpt, [5])
    sel = np.arange(0, len(idx), max(1, len(idx) // 512))[:512]
    z = torch.as_tensor(Zu[idx[sel]], device=L.DEV)
    a = torch.as_tensor(A[sel][:, 0], device=L.DEV)
    m = agent.model
    res = {}
    f_dyn, f_rew = m._dynamics(torch.cat([z, a], -1)), m._reward(torch.cat([z, a], -1))
    for k in (8, 4):
        spec = {"e7": f"int{k}", "variant": "det", "k": k, "task": task, "ckpt": ckpt}
        e7 = E7.install(agent, quant, out_dir, spec)
        i_dyn, i_rew = m.next(z, a, None), m.reward(z, a, None)
        e7.restore()
        # fake-quant reference with the same W8 weights in layers 1, 2
        saved = {}
        for seq in (m._dynamics, m._reward):
            for li in (1, 2):
                W = seq[li].weight
                saved[(id(seq), li)] = W.detach().clone()
                Wq, s = E7.w8_per_channel(W.detach().float())
                W.data.copy_(Wq * s[:, None])
        quant.set("det", k, seed=0)
        r_dyn, r_rew = m._dynamics(torch.cat([z, a], -1)), m._reward(torch.cat([z, a], -1))
        quant.off()
        for seq in (m._dynamics, m._reward):
            for li in (1, 2):
                seq[li].weight.data.copy_(saved[(id(seq), li)])
        rel = lambda x, y: float((x - y).norm() / y.norm().clamp(min=1e-12))
        rv = lambda lg: L.two_hot_inv(lg, cfg)
        res[f"k{k}"] = {"dyn_int_vs_fakequant": rel(i_dyn, r_dyn), "rew_int_vs_fakequant": rel(i_rew, r_rew),
                        "reward_value_int_vs_fakequant_maxabs": float((rv(i_rew) - rv(r_rew)).abs().max()),
                        "dyn_fakequant_vs_fp32": rel(r_dyn, f_dyn), "rew_fakequant_vs_fp32": rel(r_rew, f_rew)}
    # packed int4 = int4 in int8 containers, bitwise (same codes, same seeds)
    for v in ("det", "shiter"):
        outs = {}
        for prec in ("int4", "int4p"):
            e7 = E7.install(agent, quant, out_dir, {"e7": prec, "variant": v, "k": 4, "task": task, "ckpt": ckpt})
            e7.new_episode(12345)
            outs[prec] = (m.next(z, a, None), m.reward(z, a, None), m.next(z[:24], a[:24], None))
            e7.restore()
        res[f"int4_packed_equal_{v}"] = all(torch.equal(x, y) for x, y in zip(outs["int4"], outs["int4p"]))
    # bf16
    e7 = E7.install(agent, quant, out_dir, {"e7": "bf16", "variant": "none", "k": 0, "task": task, "ckpt": ckpt})
    b_dyn, b_rew = m.next(z, a, None), m.reward(z, a, None)
    e7.restore()
    res["bf16"] = {"dyn_vs_fp32": float((b_dyn - f_dyn).norm() / f_dyn.norm()),
                   "rew_vs_fp32": float((b_rew - f_rew).norm() / f_rew.norm())}
    res["restored"] = ("next" not in m.__dict__) and ("reward" not in m.__dict__)
    return res, agent, cfg, quant, out_dir


@torch.no_grad()
def closed_loop(agent, cfg, quant, out_dir, task="mw-push", ckpt=1, seed=19007):
    res = {}

    def ep(spec):
        e7 = E7.install(agent, quant, out_dir, spec) if spec else None
        try:
            t0 = time.time()
            v = L.canon_variant(spec["variant"]) if spec else "none"
            r = L.run_episode(agent, cfg, quant, v, int(spec["k"]) if spec else 0, seed)
            return r["ret"], (time.time() - t0) / r["steps"] * 1000.0
        finally:
            if e7 is not None:
                e7.restore()
            quant.layer_k, quant.wq = {}, None

    base_ret, base_ms = ep(None)
    res["fp32_unpatched"] = {"ret": base_ret, "ms_per_step": base_ms}
    for prec, k, v in (("fp32", 0, "none"), ("bf16", 0, "none"), ("int8", 8, "det"), ("int8", 8, "shiter"),
                       ("int4", 4, "det"), ("int4", 4, "shiter")):
        ret, ms = ep({"e7": prec, "variant": v, "k": k, "task": task, "ckpt": ckpt})
        res[f"{prec}_{v}"] = {"ret": ret, "ms_per_step": ms}
    after_ret, _ = ep(None)
    res["restore_check_fp32_equal"] = after_ret == base_ret
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = {"kernels": kernel_tests()}
    print(json.dumps(out["kernels"]), flush=True)
    mres, agent, cfg, quant, out_dir = model_tests()
    out["model"] = mres
    print(json.dumps(mres), flush=True)
    if not a.quick:
        out["closed_loop"] = closed_loop(agent, cfg, quant, out_dir)
        print(json.dumps(out["closed_loop"]), flush=True)
    json.dump(out, open(L.os.path.expanduser("~/pmx/A/tdmpc2_main/v5/e7_test.json"), "w"), indent=1)
    print("E7_TEST_DONE", flush=True)


if __name__ == "__main__":
    main()
