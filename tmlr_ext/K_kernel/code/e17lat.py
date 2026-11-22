"""E17 latency (tmlr_ext/DESIGN_tmlr.md 7, K1): integer det+shift on E8's general integer path.

e8lib.py is E8's file, unchanged (md5 7dd7b30d...): the int det and int sh-iter paths (GEMM path 'a' = Triton tiled
int8 GEMM with E8's per-shape tile configs, row kernels, device-counter dither, W8 weights, 4-bit codes in int8
containers, layer 0 = fp32 cuBLAS GEMM) are E8's, so int det / int sh-iter here are exactly E8's latency conditions.

int det+shift (this file): the same layers, weights, GEMMs and launch configuration as int det; only the two quantizing
row kernels (layer 0 and layer 1, MODE 0) are replaced by k_row_s:
  - offsets: o = u - 1/2 with u the row E8's sh-iter would draw at the same call (u = Philox(seed(f, l), column), drawn
    once per (layer, forward) one kernel ahead into the device buffer U by E8's scheme: the layer-0 row kernel draws
    U[1], the head kernel draws U[0] of the next forward, the layer-1 row kernel bumps the forward counter). Every row
    (candidate) of the call shares o; every forward gets new offsets.
  - codes: q = clamp(rint(x - o), 0, nl) on the same 2^k-level grid (det: rint(x)); dequantized value lo + delta (q + o)
  - offset correction "W (delta o) added to the next layer's bias" as one matrix-vector product per layer call, fused
    into the quantizing row kernel (no extra launch): programs split the outputs n of the next layer, 8 outputs per
    program (GCH = 8, K blocks of <= 1024), and write c_eff[n] = c[n] + alpha[n] * sum_k Wq[n, k] o[k] (alpha =
    delta s, the next GEMM's dequant scale; Wq the next layer's int8 weight). The launch grid is max(row programs,
    ceil(GN / 8)), so the GEMV keeps its parallelism at the policy-rollout batch (24 rows); programs beyond the row
    tiles skip the row part. The next GEMM (E8's gemm_a, unchanged) reads c_eff as its bias. The GEMV reads the next
    layer's int8 weight once per call; it does not depend on the number of candidates.
    (v1, smoke only: the GEMV used the row kernel's own programs with GCH = next_pow2(GN / programs), i.e. 2-3 programs
    at 5M x 24 rows and 1-output programs at 8,192 rows; replaced by v2 before the main measurement, PLAN.md.)
  Hazards: no kernel writes a location it (or a concurrent kernel) reads: layer 0 reads U[0], writes U[1] and c1_eff;
  the layer-1 GEMM reads c1_eff; layer 1 reads U[1], writes c2_eff and the counter; the head reads c2_eff, writes U[0].
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

import e8lib as E

VARIANTS = ("det", "shiter", "detshift")


@triton.jit
def _draw_chunk_s(S, UG, WG, gsalt, pid, GEN_BLOCK: tl.constexpr):
    """E8's _draw_chunk (copy): this program's chunk of u = Philox(seed(f, gsalt), column) (columns < WG)."""
    offs_g = pid * GEN_BLOCK + tl.arange(0, GEN_BLOCK)
    seed = (tl.load(S) * 1000003 + tl.load(S + 1) * 8 + gsalt) & 0x7FFFFFFF
    tl.store(UG + offs_g, tl.rand(seed, offs_g), mask=offs_g < WG)


@triton.jit
def _gemv_chunk(U, GW, GA, GC, GOUT, GN, GK, pid, GCH: tl.constexpr, GKB: tl.constexpr):
    """GOUT[n] = GC[n] + GA[n] * sum_k GW[n, k] (U[k] - 1/2) for this program's outputs n in [pid GCH, (pid+1) GCH)."""
    n0 = pid * GCH
    if n0 < GN:
        offs_n = n0 + tl.arange(0, GCH)
        nmask = offs_n < GN
        acc = tl.zeros((GCH,), dtype=tl.float32)
        for k0 in range(0, GK, GKB):
            offs_k = k0 + tl.arange(0, GKB)
            kmask = offs_k < GK
            o = tl.load(U + offs_k, mask=kmask, other=0.5) - 0.5
            w = tl.load(GW + offs_n[:, None] * GK + offs_k[None, :], mask=nmask[:, None] & kmask[None, :], other=0)
            acc += tl.sum(w.to(tl.float32) * o[None, :], axis=1)
        ga = tl.load(GA + offs_n, mask=nmask, other=0.0)
        gc = tl.load(GC + offs_n, mask=nmask, other=0.0)
        tl.store(GOUT + offs_n, gc + ga * acc, mask=nmask)


@triton.jit
def k_row_s(X, G, B, OUT, S, U, UG, GW, GA, GC, GOUT, M, W, WG, GN, stride_x, stride_o, lo, hi, inv_d, nl, zp,
            gsalt, eps, BLOCK_M: tl.constexpr, BLOCK_W: tl.constexpr, GEN: tl.constexpr, GEN_BLOCK: tl.constexpr,
            BUMP: tl.constexpr, GCH: tl.constexpr, GKB: tl.constexpr, DOGEMV: tl.constexpr):
    """det+shift quantizing row kernel (E8's k_row MODE 0 with TILES = 1, DEQ off): LayerNorm -> Mish -> codes
    clamp(rint(x - o), 0, nl) with o = U[0:W] - 1/2; then this program's chunk of the next layer's c_eff (DOGEMV);
    GEN: its chunk of the next u vector (E8 scheme); BUMP: program 0 increments the forward counter at its end."""
    tl.static_assert(not (BUMP and GEN))
    pid = tl.program_id(0)
    if BUMP:
        cnt = tl.load(S + 1)
    if pid * BLOCK_M < M:            # programs beyond the row tiles only do their GEMV chunk (v2 grid)
        offs_w = tl.arange(0, BLOCK_W)
        wmask = offs_w < W
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = (offs_m[:, None] < M) & wmask[None, :]
        y = tl.load(X + offs_m[:, None] * stride_x + offs_w[None, :], mask=mask, other=0)
        o = tl.load(U + offs_w, mask=wmask, other=0.5) - 0.5
        y = y.to(tl.float32)
        g = tl.load(G + offs_w, mask=wmask, other=0.0)
        b = tl.load(B + offs_w, mask=wmask, other=0.0)
        y = tl.where(mask, y, 0.0)
        mean = tl.sum(y, axis=1) / W
        yc = tl.where(mask, y - mean[:, None], 0.0)
        var = tl.sum(yc * yc, axis=1) / W
        rstd = 1.0 / tl.sqrt(var + eps)
        z = yc * rstd[:, None] * g[None, :] + b[None, :]
        m = z * libdevice.tanh(libdevice.log1p(tl.exp(z)))
        v = (tl.minimum(tl.maximum(m, lo), hi) - lo) * inv_d
        q = libdevice.rint(v - o[None, :])
        q = tl.minimum(tl.maximum(q, 0.0), nl)
        tl.store(OUT + offs_m[:, None] * stride_o + offs_w[None, :], (q - zp).to(tl.int8), mask=mask)
    if DOGEMV:
        _gemv_chunk(U, GW, GA, GC, GOUT, GN, W, pid, GCH, GKB)
    if GEN:
        _draw_chunk_s(S, UG, WG, gsalt, pid, GEN_BLOCK)
    if BUMP:
        if pid == 0:
            tl.store(S + 1, cnt + 1)


GCH = 8                              # GEMV outputs per program (v2)


def gemv_cfg(k):
    """(GCH, GKB): 8 outputs per program and the K block (next power of two of K, at most 1024)."""
    return GCH, min(triton.next_power_of_2(k), max(64, 8192 // GCH))


def row_launch_s(x, g, b, out, state, M, W, lo, hi, d, nl, zp, u, ug=None, wg=0, gsalt=0, bump=False, gw=None,
                 ga=None, gc=None, gout=None, gn=0):
    bw, bm, tiles, grid_rows = E.row_cfg(M, W)
    assert tiles == 1
    dogemv = gw is not None and gn > 0
    gch, gkb = gemv_cfg(W) if dogemv else (1, 64)
    P = max(grid_rows[0], triton.cdiv(gn, gch)) if dogemv else grid_rows[0]
    grid = (P,)
    gen = ug is not None
    k_row_s[grid](x, g, b, out, state, u, ug if gen else u, gw if dogemv else x, ga if dogemv else u,
                  gc if dogemv else u, gout if dogemv else u, M, W, wg, gn, x.stride(0), out.stride(0), lo, hi,
                  1.0 / d, nl, zp, gsalt, E.LN_EPS, BLOCK_M=bm, BLOCK_W=bw, GEN=gen,
                  GEN_BLOCK=E.gen_block(wg, P) if gen else 16, BUMP=bump, GCH=gch, GKB=gkb, DOGEMV=dogemv,
                  num_warps=8)


class GenIntMLPShift(E.GenIntMLP):
    """E8's GenIntMLP (path 'a') with det+shift rounding (module docstring). The dither machinery of the parent (state
    [base, f], u buffers, new_episode drawing U[0]) is used as is with dither=True; only the two quantizing row
    kernels differ (k_row_s) and the next GEMMs read the c_eff buffers."""

    def __init__(self, seq, qr0, qr1, salt, dyn_head, impl, gemm_cfgs=None, k=E.KBITS, simd=8, gemv=True):
        assert impl == "a", impl
        super().__init__(seq, qr0, qr1, True, salt, dyn_head, impl, gemm_cfgs, k, simd)
        self.gemv = bool(gemv)            # False only in the descriptive row-kernel bench (biased output)
        self.c1eff = torch.empty_like(self.c1)
        self.c2eff = torch.empty_like(self.c2)

    def _forward(self, x):
        x2 = x.reshape(-1, x.shape[-1])
        M = x2.shape[0]
        dev, Wd, st = x2.device, self.width, self.state
        u0, u1 = self.ubuf[0], self.ubuf[1]
        y0 = F.linear(x2, self.W0, self.b0)
        q0 = torch.empty((M, Wd), device=dev, dtype=torch.int8)
        g1 = dict(gw=self.W1q, ga=self.alpha1, gc=self.c1, gout=self.c1eff, gn=Wd) if self.gemv else {}
        # layer 0: reads u0 (offsets o0 = u0 - 1/2), draws u1, writes c1_eff = c1 + alpha1 (W1q o0)
        row_launch_s(y0, self.g0, self.be0, q0, st, M, Wd, self.lo0, self.hi0, self.d0, self.nl, self.zp, u0,
                     ug=u1, wg=Wd, gsalt=self.salt + 1, **g1)
        y1 = E.gemm_a(q0, self.W1q, self.alpha1, self.c1eff if self.gemv else self.c1, st, self.gemm_cfgs)
        q1 = torch.empty((M, Wd), device=dev, dtype=torch.int8)
        g2 = dict(gw=self.W2q, ga=self.alpha2, gc=self.c2, gout=self.c2eff, gn=self.n_out) if self.gemv else {}
        # layer 1: reads u1 (o1), increments f, writes c2_eff = c2 + alpha2 (W2q o1)
        row_launch_s(y1, self.g1, self.be1, q1, st, M, Wd, self.lo1, self.hi1, self.d1, self.nl, self.zp, u1,
                     bump=True, **g2)
        c2 = self.c2eff if self.gemv else self.c2
        head_gen = dict(ug=u0, wg=Wd, gsalt=self.salt)
        if self.dyn_head:
            y2 = E.gemm_a(q1, self.W2q, self.alpha2, c2, st, self.gemm_cfgs)
            out = torch.empty((M, self.n_out), device=dev, dtype=torch.float32)
            # head: draws u0 = Philox(seed(f + 1, 0)) for the next forward (E8)
            E.row_launch(y2, None, None, self.g2, self.be2, out, st, M, self.n_out, 0.0, 1.0, 1.0, 1.0, 0.0, 1,
                         False, False, simd=self.simd, **head_gen)
        else:
            out = E.gemm_a(q1, self.W2q, self.alpha2, c2, st, self.gemm_cfgs, **head_gen)
        return q0, q1, out


class WMshift(E.WMint):
    """int det+shift WM path (E8's WMint interface)."""

    def __init__(self, model, qr, impl, gemm_cfgs=None, gemv=True):
        E.WMfp32.__init__(self, model)
        self.name, self.impl = "int-detshift", impl
        self.dyn = GenIntMLPShift(model._dynamics, qr["dyn0"], qr["dyn1"], E.SALT["dyn"], True, impl, gemm_cfgs,
                                  gemv=gemv)
        self.rew = GenIntMLPShift(model._reward, qr["rew0"], qr["rew1"], E.SALT["rew"], False, impl, gemm_cfgs,
                                  gemv=gemv)


def make_wm_s(model, prec, qr, impl, gemm_cfgs):
    """prec in int-det, int-shiter (E8, unchanged) and int-detshift (this file)."""
    if prec == "int-detshift":
        return WMshift(model, qr, impl, gemm_cfgs)
    return E.make_wm(model, prec, qr, impl, gemm_cfgs)
