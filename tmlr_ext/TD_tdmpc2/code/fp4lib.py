"""E9 (rss_ext/DESIGN_ext.md 2.3): block-scaled 4-bit floating-point fake quantization of hidden activations.

Row = one candidate (one row of the (N, W) Mish output of a quantized layer), block = BS consecutive hidden units.
Element format E2M1: magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6}; |x_s| > 6 saturates to 6.
  NVFP4 (BS = 16): per-layer global scale g = hi / (6 * 448) (hi = the layer's E1 range upper bound, 99.99th pct);
                   block scale s_b = E4M3_RNE(amax_b / (6 g)) * g, the E4M3 code clipped to the E4M3 range
                   [2^-9, 448] before rounding; s_b = g if amax_b = 0.
  MXFP4 (BS = 32, OCP MX v1.0): s_b = 2^(floor(log2 amax_b) - 2) (E8M0); s_b = 1 if amax_b = 0.
  x_s = x / s_b, output q * s_b.
  det:    q = E2M1 value nearest to x_s, ties to the even mantissa (RNE).
  shiter: lo <= |x_s| < hi neighbouring grid magnitudes, q = lo + (hi - lo) 1[u < (|x_s| - lo) / (hi - lo)], sign
          restored; |x_s| >= 6 -> +-6. u in [0, 1) is one value per (layer, unit), shared by all rows (candidates) of a
          forward call; the caller draws it (qlib.Quantizer, private generator, fresh every forward call).
  The block scale is deterministic in both variants; only the element rounding differs.

Floating-point conventions (identical in the two implementations, checked bit for bit by test_fp4.py):
  g = fp32(hi / 2688) (one rounding of the float64 quotient), 6g = fp32(6 * g); r = fl(amax_b / 6g) (IEEE division);
  E4M3 rounding of r in [2^-9, 448]: spacing 2^(max(e, -6) - 3), e = floor(log2 r) read from the fp32 exponent bits,
  q = rint(r / spacing) * spacing (exact); s_b = fl(q_e4m3 * g); x_s = fl(x / s_b) (IEEE division); E2M1 rounding in
  units of the spacing 0.5 / 1 / 2 of the binade of |x_s| (exact); output fl(q * s_b).
  MXFP4: floor(log2 amax_b) from the exponent bits (amax_b < 2^-124 never occurs for Mish outputs; it would be
  clamped to a scale of 2^-126).

Implementations: fake_quant_ref (PyTorch ops, the reference) and fake_quant_triton (one fused Triton kernel per call,
the same arithmetic). fake_quant(..., impl=...) dispatches.
"""
import numpy as np
import torch

E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MAX = 6.0
E4M3_MAX = 448.0
E4M3_MIN = 2.0 ** -9          # smallest positive E4M3 (subnormal) value
BLOCK = {"nvfp4": 16, "mxfp4": 32}
_NV_CACHE = {}


def nv_global(hi):
    """(g, 6g) of NVFP4 as Python floats that are exact fp32 numbers: g = fp32(hi / (6 * 448)), 6g = fp32(6 * g)."""
    hi = float(hi)
    v = _NV_CACHE.get(hi)
    if v is None:
        g = np.float32(hi / (6.0 * 448.0))
        v = (float(g), float(np.float32(np.float32(6.0) * g)))
        _NV_CACHE[hi] = v
    return v


# ---------------------------------------------------------------------------------------------------- reference
def _exp_bits(x):
    """Biased fp32 exponent of positive finite x (int32): floor(log2 x) + 127 for normal x."""
    return (x.view(torch.int32) >> 23) & 0xFF


def _pow2_biased(eb):
    """2^(eb - 127) for int32 eb in [1, 254], built from the exponent bits (exact)."""
    return (eb.to(torch.int32) << 23).view(torch.float32)


def e4m3_rne(r):
    """r (fp32, every value in [2^-9, 448]) -> nearest E4M3 value, ties to even mantissa (exact)."""
    eb = torch.clamp(_exp_bits(r), min=127 - 6)            # subnormal binade shares the spacing of 2^-6
    sp = _pow2_biased(eb - 3)                              # 3 mantissa bits
    return torch.round(r / sp) * sp                        # r / sp exact (power of two); round = half to even


def e2m1_spacing(a):
    """Grid spacing of the E2M1 binade holding a in [0, 6]: 0.5 below 2 (subnormal 0.5 and 1, 1.5), 1 in [2, 4),
    2 in [4, 6]."""
    return torch.where(a < 2.0, 0.5, torch.where(a < 4.0, 1.0, 2.0)).to(a.dtype)


def e2m1_rne(a):
    """a = |x_s| clipped to [0, 6] -> nearest E2M1 magnitude, ties to even mantissa (= even multiple of the spacing)."""
    sp = e2m1_spacing(a)
    return torch.round(a / sp) * sp


def block_scales(xb, fmt, hi=None):
    """xb (R, NB, BS) -> block scales s (R, NB, 1) and amax (R, NB, 1)."""
    amax = xb.abs().amax(-1, keepdim=True)
    if fmt == "nvfp4":
        g, sixg = nv_global(hi)
        r = amax / torch.full_like(amax, sixg)            # IEEE division (a Python scalar divisor would be a reciprocal)
        rq = e4m3_rne(r.clamp(E4M3_MIN, E4M3_MAX))
        s = torch.where(amax > 0, rq * g, torch.full_like(amax, g))
    elif fmt == "mxfp4":
        eb = torch.clamp(_exp_bits(amax), min=3)           # floor(log2 amax) + 127
        s = torch.where(amax > 0, _pow2_biased(eb - 2), torch.ones_like(amax))
    else:
        raise ValueError(fmt)
    return s, amax


def fake_quant_ref(x, fmt, u=None, hi=None):
    """Reference implementation. x (..., W) fp32; u None (det) or (1, W) / broadcastable uniform draws (shiter)."""
    shp = x.shape
    W = shp[-1]
    BS = BLOCK[fmt]
    xb = x.reshape(-1, W // BS, BS)
    s, _ = block_scales(xb, fmt, hi)
    xs = xb / s
    a = xs.abs().clamp(max=E2M1_MAX)
    sp = e2m1_spacing(a)
    if u is None:
        q = torch.round(a / sp) * sp
    else:
        lo = torch.floor(a / sp) * sp                      # lower neighbour (exact)
        p = (a - lo) / sp                                  # exact (Sterbenz; division by a power of two)
        q = lo + sp * (u.reshape(-1, W // BS, BS) < p).to(a.dtype)
    q = torch.where(xs < 0, -q, q)
    return (q * s).reshape(shp)


# ---------------------------------------------------------------------------------------------------- Triton
try:     # module-level imports: the Triton JIT resolves tl / libdevice from this module's globals
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except Exception:   # no Triton (e.g. the MacBook): only the reference implementation is available
    triton = None

if triton is not None:
    @triton.jit
    def _fp4_kernel(X, Y, U, M, g, sixg, stride_x, stride_y,
                    W: tl.constexpr, BS: tl.constexpr, NV: tl.constexpr, STOCH: tl.constexpr,
                    BLOCK_M: tl.constexpr):
        NB: tl.constexpr = W // BS
        R: tl.constexpr = BLOCK_M * NB
        pid = tl.program_id(0)
        rr = tl.arange(0, R)
        m = pid * BLOCK_M + rr // NB                       # row (candidate)
        cols = (rr % NB)[:, None] * BS + tl.arange(0, BS)[None, :]   # (R, BS): the block's units
        mask = (m < M)[:, None]
        x = tl.load(X + m[:, None] * stride_x + cols, mask=mask, other=0.0)
        amax = tl.max(tl.abs(x), axis=1)                   # (R,)
        if NV:
            r = libdevice.div_rn(amax, sixg)
            r = tl.minimum(tl.maximum(r, 0.001953125), 448.0)
            eb = tl.maximum((r.to(tl.int32, bitcast=True) >> 23) & 0xFF, 121)
            sp = ((eb - 3) << 23).to(tl.float32, bitcast=True)
            isp = ((257 - eb) << 23).to(tl.float32, bitcast=True)    # 1 / sp (exact)
            rq = libdevice.rint(r * isp) * sp              # E4M3 RNE (exact)
            s = tl.where(amax > 0.0, rq * g, g)
        else:
            eb = tl.maximum((amax.to(tl.int32, bitcast=True) >> 23) & 0xFF, 3)
            s = ((eb - 2) << 23).to(tl.float32, bitcast=True)
            s = tl.where(amax > 0.0, s, 1.0)
        xs = libdevice.div_rn(x, tl.broadcast_to(s[:, None], (R, BS)))
        a = tl.minimum(tl.abs(xs), 6.0)
        sp2 = tl.where(a < 2.0, 0.5, tl.where(a < 4.0, 1.0, 2.0))
        isp2 = tl.where(a < 2.0, 2.0, tl.where(a < 4.0, 1.0, 0.5))
        if STOCH:
            u = tl.load(U + cols)                          # one u per unit, the same for every row
            lo = tl.floor(a * isp2) * sp2
            p = (a - lo) * isp2
            q = lo + tl.where(u < p, sp2, 0.0)
        else:
            q = libdevice.rint(a * isp2) * sp2
        q = tl.where(xs < 0.0, -q, q)
        y = q * s[:, None]
        tl.store(Y + m[:, None] * stride_y + cols, y, mask=mask)


def fake_quant_triton(x, fmt, u=None, hi=None, block_m=8, num_warps=4):
    kern = _fp4_kernel
    shp = x.shape
    W = shp[-1]
    x2 = x.reshape(-1, W)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    M = x2.shape[0]
    y = torch.empty_like(x2)
    if fmt == "nvfp4":
        g, sixg = nv_global(hi)
    else:
        g, sixg = 1.0, 1.0
    if u is not None:
        u = u.reshape(-1)
        assert u.numel() == W and u.is_contiguous() and u.dtype == x2.dtype
    grid = (triton.cdiv(M, block_m),)
    kern[grid](x2, y, u if u is not None else x2, M, g, sixg, x2.stride(0), y.stride(0),
               W=W, BS=BLOCK[fmt], NV=(fmt == "nvfp4"), STOCH=(u is not None), BLOCK_M=block_m, num_warps=num_warps)
    return y.reshape(shp)


def fake_quant(x, fmt, u=None, hi=None, impl="triton"):
    if impl == "triton":
        return fake_quant_triton(x, fmt, u, hi)
    if impl == "ref":
        return fake_quant_ref(x, fmt, u, hi)
    raise ValueError(impl)
