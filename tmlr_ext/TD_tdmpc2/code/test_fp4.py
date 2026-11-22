"""E9 unit checks of the FP4 fake quantization (pod, GPU). Prints PASS / FAIL per check and writes a JSON summary.

1 E2M1 RNE: grid points, midpoints, +-1 ulp around midpoints, values above 6 and random values, fed through the full
  fake quant with block scale 1 (MXFP4 blocks whose amax is 7.9 -> s = 2^(2-2) = 1), vs a float64 brute force
  (nearest grid magnitude, ties to the even mantissa = even grid index); sh-iter: q in {lo, hi} and q = lo + (hi - lo)
  1[u < p] exactly
2 E4M3 RNE: fp4lib.e4m3_rne vs torch's float8_e4m3fn cast and vs a brute force over all E4M3 values
3 block scales: NVFP4 vs an fp32 brute force (numpy), MXFP4 vs 2^(floor(log2 amax) - 2) from math.frexp, amax = 0
4 stochastic rounding is unbiased: per-entry mean over 20,000 per-row draws vs x (z-scores), saturated entries exact
5 Triton kernel == PyTorch reference bit for bit (random inputs at several scales, constructed ties, zero blocks,
  outliers, recorded activations of the 4 layers of a real checkpoint)
6 Quantizer hook: identical candidates -> identical outputs (shared u), a fresh u every forward call, re-seeding
  reproduces, the generator advances by exactly one (1, W) draw per hooked layer call, the planner's global RNG is
  untouched, encoder / Q / policy are not quantized
7 block boundaries: changing one entry changes only its own block of its own row
Usage: python test_fp4.py --out /workspace/results/smoke/test_fp4.json
"""
import argparse
import gzip
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp4lib as F  # noqa: E402

DEV = "cuda"
RES = {}


def check(name, ok, **info):
    RES[name] = {"ok": bool(ok), **{k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in info.items()}}
    print(f"{'PASS' if ok else 'FAIL'} {name} {json.dumps(RES[name], default=str)[:400]}", flush=True)


G = np.array(F.E2M1_GRID)


def e2m1_brute(v):
    """float64 brute force: nearest grid magnitude, ties to the even grid index (= even mantissa), sign restored."""
    a = abs(v)
    d = np.abs(G - a)
    best = np.flatnonzero(d == d.min())
    i = best[0] if len(best) == 1 else [b for b in best if b % 2 == 0][0]
    return math.copysign(G[i], v) if G[i] != 0 else 0.0


def e4m3_values():
    vals = [0.0] + [m / 8 * 2.0 ** -6 for m in range(1, 8)]
    for e in range(1, 16):
        for m in range(8):
            if e == 15 and m == 7:
                continue
            vals.append((1 + m / 8) * 2.0 ** (e - 7))
    return np.array(sorted(vals))


E4 = e4m3_values()


def e4m3_brute(r):
    d = np.abs(E4 - r)
    best = np.flatnonzero(d == d.min())
    i = best[0] if len(best) == 1 else [b for b in best if b % 2 == 0][0]
    return E4[i]


def both(x, fmt, u=None, hi=None):
    return F.fake_quant_ref(x, fmt, u, hi), F.fake_quant_triton(x, fmt, u, hi)


def bit_equal(a, b):
    """Bit-identical after canonicalising signed zeros (x + 0.0 turns -0.0 into +0.0): the Triton build returns +0.0
    where the reference returns -0.0 (a negative entry rounded to 0); the sign of a zero changes no later value."""
    return bool(torch.equal((a + 0.0).contiguous().view(torch.int32), (b + 0.0).contiguous().view(torch.int32)))


def bit_equal_raw(a, b):
    return bool(torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)))


# -------------------------------------------------------------------------------------------------------- 1
def t_e2m1():
    f32 = np.float32
    mids = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    vals = list(G) + mids + [6.5, 7.0, 7.5, 7.9]
    for mpt in mids + list(G[1:]):
        vals += [float(np.nextafter(f32(mpt), f32(0))), float(np.nextafter(f32(mpt), f32(10)))]
    rng = np.random.default_rng(0)
    vals += list(rng.uniform(0, 7.9, 3000).astype(np.float32))
    vals = np.array(vals, np.float32)
    vals = np.concatenate([vals, -vals])
    per = 31                                   # 31 test values + the 7.9 amax entry per 32-block
    nblk = int(math.ceil(len(vals) / per))
    nblk += (-nblk) % 16                       # 16 blocks per 512-row
    X = np.zeros((nblk, 32), np.float32)
    X[:, 0] = 7.9
    flat = np.zeros(nblk * per, np.float32)
    flat[:len(vals)] = vals
    X[:, 1:] = flat.reshape(nblk, per)
    x = torch.tensor(X.reshape(-1, 512), device=DEV)
    yr, yt = both(x, "mxfp4")
    Y = yr.cpu().numpy().reshape(nblk, 32)[:, 1:].reshape(-1)[:len(vals)]
    exp = np.array([e2m1_brute(float(v)) for v in vals])
    bad = np.flatnonzero(Y != exp)
    check("1a_e2m1_rne_det_vs_bruteforce", len(bad) == 0 and bit_equal(yr, yt), n=len(vals), n_bad=len(bad),
          examples=[(float(vals[i]), float(Y[i]), float(exp[i])) for i in bad[:5]],
          triton_equal=bit_equal(yr, yt))
    table = {f"{m:g}": [e2m1_brute(m), float(Y[list(vals).index(np.float32(m))])] for m in mids}
    RES["1a_e2m1_rne_det_vs_bruteforce"]["midpoint_table"] = table
    # sh-iter on the same values with one shared u row
    gen = torch.Generator(device=DEV)
    gen.manual_seed(1)
    u = torch.rand((1, 512), generator=gen, device=DEV)
    yr, yt = both(x, "mxfp4", u)
    Y = yr.cpu().numpy().reshape(nblk, 32)[:, 1:].reshape(-1)[:len(vals)]
    U = u.cpu().numpy().reshape(16, 32)
    Ub = np.tile(U, (x.shape[0], 1))[:, 1:].reshape(-1)[:len(vals)]   # u of each test value's unit
    nbad = 0
    for v, y, uu in zip(vals, Y, Ub):
        a = min(abs(float(v)), 6.0)
        if a >= 6.0:
            q = 6.0
        else:
            j = int(np.searchsorted(G, a, side="right") - 1)
            lo, hi = G[j], G[j + 1]
            q = lo + (hi - lo) * (uu < (a - lo) / (hi - lo))
        q = math.copysign(q, float(v)) if q != 0 else 0.0
        nbad += (q != y)
    check("1b_e2m1_shiter_formula", nbad == 0 and bit_equal(yr, yt), n=len(vals), n_bad=int(nbad),
          triton_equal=bit_equal(yr, yt))


# -------------------------------------------------------------------------------------------------------- 2
def t_e4m3():
    f32 = np.float32
    pos = E4[1:]
    vals = list(pos)
    mids = (pos[:-1] + pos[1:]) / 2
    vals += list(mids)
    vals += [float(np.nextafter(f32(m), f32(0))) for m in mids] + [float(np.nextafter(f32(m), f32(1e9))) for m in mids]
    rng = np.random.default_rng(1)
    vals += list(np.exp(rng.uniform(np.log(2.0 ** -9), np.log(448.0), 20000)))
    r = torch.tensor(np.array(vals, np.float32), device=DEV)
    q = F.e4m3_rne(r)
    q8 = r.to(torch.float8_e4m3fn).to(torch.float32)
    qb = torch.tensor([e4m3_brute(float(v)) for v in r.cpu().numpy()], device=DEV, dtype=torch.float32)
    n1 = int((q != q8).sum())
    n2 = int((q != qb).sum())
    check("2_e4m3_rne", n1 == 0 and n2 == 0, n=len(vals), n_bad_vs_float8_cast=n1, n_bad_vs_bruteforce=n2)


# -------------------------------------------------------------------------------------------------------- 3
def t_scales():
    rng = np.random.default_rng(2)
    X = (rng.standard_normal((256, 512)) * np.exp(rng.uniform(-6, 3, (256, 1)))).astype(np.float32)
    X[0, :16] = 0.0                                      # an all-zero NVFP4 block
    X[1, :32] = 0.0                                      # an all-zero MXFP4 block
    x = torch.tensor(X, device=DEV)
    hi = 8.544163703918457
    g, sixg = F.nv_global(hi)
    s, amax = F.block_scales(x.view(256, 32, 16), "nvfp4", hi)
    s = s.cpu().numpy().reshape(-1)
    am = amax.cpu().numpy().reshape(-1)
    exp = []
    for a in am:
        if a == 0:
            exp.append(np.float32(g))
            continue
        rr = np.float32(a) / np.float32(sixg)
        rr = min(max(float(rr), 2.0 ** -9), 448.0)
        exp.append(np.float32(np.float32(e4m3_brute(rr)) * np.float32(g)))
    exp = np.array(exp, np.float32)
    n_nv = int((s != exp).sum())
    s2, amax2 = F.block_scales(x.view(256, 16, 32), "mxfp4")
    s2 = s2.cpu().numpy().reshape(-1)
    am2 = amax2.cpu().numpy().reshape(-1)
    exp2 = np.array([1.0 if a == 0 else 2.0 ** (math.frexp(float(a))[1] - 1 - 2) for a in am2], np.float32)
    n_mx = int((s2 != exp2).sum())
    check("3_block_scales", n_nv == 0 and n_mx == 0, n_blocks_nv=len(s), n_bad_nv=n_nv, n_blocks_mx=len(s2),
          n_bad_mx=n_mx, g=g, sixg=sixg, zero_block_scale_nv=float(s[0]), zero_block_scale_mx=float(s2[16]))


# -------------------------------------------------------------------------------------------------------- 4
def t_unbiased():
    T = 20000
    rng = np.random.default_rng(3)
    x0 = rng.standard_normal(512).astype(np.float32) * 2.0
    x0[5] = 40.0                                        # NVFP4 block 0 / MXFP4 block 0 outlier -> saturation in it
    x0[100] = -0.2
    out = {}
    for fmt, hi in (("nvfp4", 8.0), ("mxfp4", None)):
        x = torch.tensor(x0, device=DEV).expand(T, 512).contiguous()
        gen = torch.Generator(device=DEV)
        gen.manual_seed(4)
        u = torch.rand((T, 512), generator=gen, device=DEV)
        y = F.fake_quant_ref(x, fmt, u, hi)
        BS = F.BLOCK[fmt]
        s, _ = F.block_scales(x[:1].view(1, 512 // BS, BS), fmt, hi)
        sx = s.expand(1, 512 // BS, BS).reshape(512)
        xs = x[0] / sx
        a = xs.abs().clamp(max=6.0)
        sp = F.e2m1_spacing(a)
        lo = torch.floor(a / sp) * sp
        p = (a - lo) / sp
        sat = xs.abs() >= 6.0
        mean = y.double().mean(0)
        x64 = x[0].double()
        sd = (sx * sp).double() * torch.sqrt(p.double() * (1 - p.double()) / T)
        unsat = ~sat & (p > 0)
        z = ((mean - x64) / sd)[unsat]
        on = ~sat & (p == 0)                 # x_s on the grid: every draw gives the same value
        exact_on_grid = bool(((mean[on] - x64[on]).abs() <= 1e-6 * x64[on].abs() + 1e-30).all())
        sat_ok = bool(torch.equal(y[:, sat], (torch.sign(xs[sat]) * 6.0 * sx[sat]).expand(T, -1)))
        out[fmt] = {"max_abs_z": float(z.abs().max()), "mean_abs_z": float(z.abs().mean()), "n_unsat": int(unsat.sum()),
                    "n_sat": int(sat.sum()), "on_grid_exact": exact_on_grid, "saturated_exact": sat_ok}
    ok = all(v["max_abs_z"] < 6.0 and v["mean_abs_z"] < 1.0 and v["saturated_exact"] and v["on_grid_exact"]
             for v in out.values())
    check("4_shiter_unbiased", ok, **out)


# -------------------------------------------------------------------------------------------------------- 5
def t_triton_equal(real_acts):
    rng = np.random.default_rng(5)
    cases = {}
    for sc in (1e-3, 0.1, 1.0, 10.0, 100.0):
        cases[f"normal_x{sc:g}"] = rng.standard_normal((512, 512)).astype(np.float32) * sc
    z = rng.standard_normal((512, 512)).astype(np.float32) * 2
    cases["mish"] = z * np.tanh(np.log1p(np.exp(z)))
    t = cases["mish"].copy()
    t[:, ::7] = 0.0
    t[3, :] = 0.0
    t[4, :64] = 1e3
    cases["mish_zeros_outliers"] = t
    cases["odd_rows_24"] = cases["mish"][:24]
    cases["one_row"] = cases["mish"][:1]
    cases["rows_4096"] = np.tile(cases["mish"], (8, 1))
    gen = torch.Generator(device=DEV)
    gen.manual_seed(6)
    res, ok = {}, True
    items = [(k, torch.tensor(v, device=DEV)) for k, v in cases.items()] + list(real_acts.items())
    for name, x in items:
        for fmt in ("nvfp4", "mxfp4"):
            hi = float(x.max()) if fmt == "nvfp4" else None
            if name.startswith("real_"):
                hi = real_acts_hi[name] if fmt == "nvfp4" else None
            u = torch.rand((1, x.shape[-1]), generator=gen, device=DEV)
            for lab, uu in (("det", None), ("shiter", u)):
                yr, yt = both(x, fmt, uu, hi)
                be = bit_equal(yr, yt)
                ne = bool(torch.equal(yr, yt))
                nz = int(((yr.view(torch.int32) != yt.view(torch.int32)) & (yr != 0)).sum())   # nonzero bit diffs
                fin = bool(torch.isfinite(yt).all())
                res[f"{name}/{fmt}/{lab}"] = {"bit_equal": be, "bit_equal_raw": bit_equal_raw(yr, yt), "num_equal": ne,
                                              "n_bitdiff_nonzero": nz, "finite": fin,
                                              "max_abs_diff": float((yr - yt).abs().max())}
                ok &= be and fin
    n_bad = sum(1 for v in res.values() if not v["bit_equal"])
    check("5_triton_equals_reference", ok, n_cases=len(res), n_not_bit_equal=n_bad,
          bad=[k for k, v in res.items() if not v["bit_equal"]][:10])
    RES["5_triton_equals_reference"]["cases"] = res


real_acts_hi = {}


def record_real(task="mw-push", ckpt=1):
    """Planning-batch activations of the 4 layers (512 random-action candidates from 8 encoded random observations)."""
    import qlib as L
    import runner as R
    agent, cfg = L.build_agent(task, ckpt)
    quant = L.Quantizer(agent.model, horizon=int(cfg.horizon))
    qr = R.load_ranges(None, task, ckpt, "e1")
    torch.manual_seed(0)
    z = agent.model.encode(torch.randn(8, cfg.obs_shape["state"][0], device=DEV), None)
    z = z.repeat_interleave(64, 0)
    a = torch.rand(512, cfg.action_dim, device=DEV) * 2 - 1
    quant.set("record")
    agent.model.next(z, a, None)
    agent.model.reward(z, a, None)
    acts = {f"real_{task}_{n}": quant.last[n].detach().clone() for n in L.LAYERS}
    for n in L.LAYERS:
        real_acts_hi[f"real_{task}_{n}"] = qr[n][1]
    quant.off()
    return agent, cfg, quant, qr, acts


# -------------------------------------------------------------------------------------------------------- 6
def t_hook(agent, cfg, quant, qr):
    import qlib as L
    quant.qrange = qr
    m = agent.model
    torch.manual_seed(1)
    z1 = m.encode(torch.randn(1, cfg.obs_shape["state"][0], device=DEV), None)
    z = z1.expand(512, -1).contiguous()
    a = (torch.rand(1, cfg.action_dim, device=DEV) * 2 - 1).expand(512, -1).contiguous()
    out = {}
    for v in L.FP4_VARIANTS:
        for impl in ("triton", "ref"):
            quant.fp4_impl = impl
            cpu0, cuda0 = torch.get_rng_state(), torch.cuda.get_rng_state()
            quant.set(v, 4, seed=123)
            y1 = m.next(z, a, None)
            nq1 = quant.nq
            y2 = m.next(z, a, None)
            r1 = m.reward(z, a, None)
            nq3 = quant.nq
            quant.set(v, 4, seed=123)
            y1b = m.next(z, a, None)
            rng_ok = bool(torch.equal(cpu0, torch.get_rng_state())) and bool(torch.equal(cuda0, torch.cuda.get_rng_state()))
            # generator advance: one (1, 512) draw per hooked layer call for sh-iter, none for det
            g2 = torch.Generator(device=DEV)
            g2.manual_seed(123)
            if v.endswith("shiter"):
                torch.rand((1, 512), generator=g2, device=DEV)
                torch.rand((1, 512), generator=g2, device=DEV)
            gen_ok = bool(torch.equal(quant.gen.get_state(), g2.get_state()))
            nq_before = quant.nq
            m.pi(z, None)                       # (uses the global RNG itself; checked above, before this)
            m.Q(z, a, None, return_type="avg")
            m.encode(torch.randn(2, cfg.obs_shape["state"][0], device=DEV), None)
            untouched = quant.nq == nq_before
            quant.off()
            y0 = m.next(z, a, None)
            same_rows = bool((y1 == y1[:1]).all()) and bool((r1 == r1[:1]).all())
            fresh = not torch.equal(y1, y2)
            out[f"{v}/{impl}"] = {"rows_identical": same_rows, "second_call_differs": fresh, "reseed_reproduces":
                                  bool(torch.equal(y1, y1b)), "gen_advance_ok": gen_ok, "global_rng_untouched": rng_ok,
                                  "nq_next": nq1, "nq_after_next_next_reward": nq3, "other_modules_untouched": untouched,
                                  "changes_output": not torch.equal(y0, y1)}
    ok = True
    for k, d in out.items():
        stoch = "shiter" in k
        ok &= d["rows_identical"] and d["reseed_reproduces"] and d["gen_advance_ok"] and d["global_rng_untouched"]
        ok &= d["nq_next"] == 2 and d["nq_after_next_next_reward"] == 6 and d["other_modules_untouched"]
        ok &= d["changes_output"] and (d["second_call_differs"] == stoch)
    # the two implementations give the same hooked outputs
    check("6_quantizer_hook", ok, **out)


# -------------------------------------------------------------------------------------------------------- 7
def t_blocks():
    rng = np.random.default_rng(7)
    x = torch.tensor(rng.standard_normal((4, 512)).astype(np.float32), device=DEV)
    gen = torch.Generator(device=DEV)
    gen.manual_seed(8)
    u = torch.rand((1, 512), generator=gen, device=DEV)
    out, ok = {}, True
    for fmt in ("nvfp4", "mxfp4"):
        BS = F.BLOCK[fmt]
        for uu, lab in ((None, "det"), (u, "shiter")):
            y = F.fake_quant_triton(x, fmt, uu, 3.0)
            for j in (0, 15, 16, 31, 32, 255, 256, 511):
                x2 = x.clone()
                x2[1, j] = 50.0
                y2 = F.fake_quant_triton(x2, fmt, uu, 3.0)
                diff = (y2 != y)
                rows = sorted(set(torch.nonzero(diff)[:, 0].tolist()))
                cols = torch.nonzero(diff[1]).flatten().tolist()
                blocks = sorted(set(c // BS for c in cols))
                good = rows == [1] and blocks == [j // BS]
                ok &= good
                out[f"{fmt}/{lab}/j{j}"] = {"rows_changed": rows, "blocks_changed": blocks, "ok": good}
    check("7_block_boundaries", ok, n=len(out), bad=[k for k, v in out.items() if not v["ok"]])


def diag_real(acts):
    """Saturation and scale-clamp shares on the recorded activations (information)."""
    d = {}
    for name, x in acts.items():
        hi = real_acts_hi[name]
        for fmt in ("nvfp4", "mxfp4"):
            BS = F.BLOCK[fmt]
            xb = x.reshape(-1, 512 // BS, BS)
            s, amax = F.block_scales(xb, fmt, hi)
            xs = xb / s
            row = {"sat_share": float((xs.abs() > 6).float().mean()), "zero_blocks": int((amax == 0).sum())}
            if fmt == "nvfp4":
                g, sixg = F.nv_global(hi)
                r = amax / torch.full_like(amax, sixg)
                row.update(r_below_min=int((r < F.E4M3_MIN).sum()), r_above_max=int((r > F.E4M3_MAX).sum()))
            y = F.fake_quant_ref(x, fmt, None, hi)
            row["rel_mse_det"] = float(((y - x) ** 2).mean() / x.var())
            d[f"{name}/{fmt}"] = row
    RES["diag_real_activations"] = d
    print("DIAG", json.dumps(d)[:1500], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t_e2m1()
    t_e4m3()
    t_scales()
    t_unbiased()
    t_blocks()
    agent, cfg, quant, qr, acts = record_real()
    t_triton_equal(acts)
    t_hook(agent, cfg, quant, qr)
    diag_real(acts)
    RES["env"] = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    import triton
    RES["env"]["triton"] = triton.__version__
    n_fail = sum(1 for k, v in RES.items() if isinstance(v, dict) and v.get("ok") is False)
    RES["n_fail"] = n_fail
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(RES, open(a.out, "w"), indent=1, default=str)
    print(f"TEST_FP4_DONE n_fail={n_fail}", flush=True)
