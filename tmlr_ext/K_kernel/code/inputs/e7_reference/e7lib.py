"""E7 (DESIGN_main v5 7.4) for the TD-MPC2 part of paper A: real integer kernels for the dynamics and reward MLPs.

Precisions (one job = one precision for the whole job; `install` patches the WorldModel's next / reward methods of the
cached agent and `restore` removes the patch):
  fp32      the unchanged model (TD-MPC2 default)
  bf16      the dynamics and reward MLPs under torch.autocast(bfloat16), outputs cast back to fp32
  int8 / int4 (det or sh-iter dither, from the job's variant)
            layer 0 (input [z, a], not quantized in the E1 scheme): fp32 GEMM (cuBLAS), then one fused Triton kernel
              LayerNorm -> Mish -> quantize to the layer's E0 grid (det: round half to even; dither: floor(x + u) with
              one u vector per layer and forward shared by all rows = the sh-iter schedule) -> int8 codes
            layer 1: one fused Triton kernel: int8 x int8 dot (int32 accumulation) -> dequantize (per-column step x
              weight scale, zero-point correction folded into a per-column constant) -> bias -> LayerNorm -> Mish ->
              re-quantize to the layer's grid -> int8 codes
            layer 2 (head): Triton int8 dot + dequantize + bias -> fp32; the dynamics head then applies its
              LayerNorm and SimNorm (PyTorch), the reward head returns the 101 logits
            weights of layers 1 and 2: symmetric per-output-channel int8 round-to-nearest (W8) for int8 and int4
            activations; the int4 codes (0..15) are stored in int8 containers ('int4', used for the success rates)
            or packed two per byte ('int4p': low nibble = even column, unpacked in the consumer kernel before the
            int8 dot; latency only). Packed and container kernels are separate compilations, so their LayerNorm /
            Mish arithmetic can differ in floating-point order: same codes except rare boundary cases (1 of 262,144
            layer-0 codes differed by one level in the model-level check)
            codes are stored as q - zp (zp = 128 for 8 bits, 0 otherwise) so that they fit int8
The quantization grids are the E1 ones (E0 per-layer [lo, hi] of the task and checkpoint, 2^k levels), so int8 / int4
det match E1's det at k = 8 / 4 up to the W8 weights and floating-point order. Dither draws come from Philox
(tl.rand) seeded per episode (Quantizer.set seed) and forward call; they are not the E1 draws.
"""
import json
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

import qlib as L

LN_EPS = 1e-5


# ---------------------------------------------------------------------------------------------------- kernels
@triton.jit
def _ln_mish_q(y, g, b, lo, hi, inv_d, nl, zp, seed, offs_n, eps, N: tl.constexpr, DITHER: tl.constexpr):
    mean = tl.sum(y, axis=1) / N
    yc = y - mean[:, None]
    var = tl.sum(yc * yc, axis=1) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    z = yc * rstd[:, None] * g[None, :] + b[None, :]
    m = z * libdevice.tanh(libdevice.log1p(tl.exp(z)))
    x = (tl.minimum(tl.maximum(m, lo), hi) - lo) * inv_d
    if DITHER:
        u = tl.rand(seed, offs_n)
        q = tl.floor(x + u[None, :])
    else:
        q = libdevice.rint(x)
    q = tl.minimum(tl.maximum(q, 0.0), nl)
    return (q - zp).to(tl.int8)


@triton.jit
def _store_codes(Q, q, offs_m, offs_n, stride_qm, mask, BLOCK_M: tl.constexpr, N: tl.constexpr,
                 PACK_OUT: tl.constexpr):
    if PACK_OUT:   # packed int4: column 2j in the low nibble, 2j + 1 in the high nibble (codes 0..15, zp = 0)
        lo4, hi4 = tl.split(tl.reshape(q.to(tl.uint8), (BLOCK_M, N // 2, 2)))
        offs_h = tl.arange(0, N // 2)
        tl.store(Q + offs_m[:, None] * stride_qm + offs_h[None, :], lo4 | (hi4 << 4), mask=mask)
    else:
        tl.store(Q + offs_m[:, None] * stride_qm + offs_n[None, :], q, mask=mask)


@triton.jit
def _load_codes(A, offs_m, k0, stride_am, rmask, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
                PACK_IN: tl.constexpr):
    if PACK_IN:
        offs_kh = tl.arange(0, BLOCK_K // 2)
        ap = tl.load(A + offs_m[:, None] * stride_am + (k0 // 2 + offs_kh)[None, :], mask=rmask, other=0)
        a = tl.reshape(tl.join((ap & 15).to(tl.int8), (ap >> 4).to(tl.int8)), (BLOCK_M, BLOCK_K))
    else:
        offs_k = tl.arange(0, BLOCK_K)
        a = tl.load(A + offs_m[:, None] * stride_am + (k0 + offs_k)[None, :], mask=rmask, other=0)
    return a


@triton.jit
def k_ln_mish_quant(Y, G, B, Q, M, lo, hi, inv_d, nl, zp, seed, eps, stride_ym, stride_qm,
                    N: tl.constexpr, BLOCK_M: tl.constexpr, DITHER: tl.constexpr, PACK_OUT: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, N)
    mask = offs_m[:, None] < M
    y = tl.load(Y + offs_m[:, None] * stride_ym + offs_n[None, :], mask=mask, other=0.0)
    q = _ln_mish_q(y, tl.load(G + offs_n), tl.load(B + offs_n), lo, hi, inv_d, nl, zp, seed, offs_n, eps, N, DITHER)
    _store_codes(Q, q, offs_m, offs_n, stride_qm, mask, BLOCK_M, N, PACK_OUT)


@triton.jit
def k_i8gemm_ln_mish_quant(A, Bw, ALPHA, C, G, Bt, Q, M, K, lo, hi, inv_d, nl, zp, seed, eps,
                           stride_am, stride_bk, stride_qm,
                           N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, DITHER: tl.constexpr,
                           PACK_IN: tl.constexpr, PACK_OUT: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, BLOCK_K)
    rmask = offs_m[:, None] < M
    acc = tl.zeros((BLOCK_M, N), dtype=tl.int32)
    for k0 in range(0, K, BLOCK_K):
        a = _load_codes(A, offs_m, k0, stride_am, rmask, BLOCK_M, BLOCK_K, PACK_IN)
        b = tl.load(Bw + (k0 + offs_k)[:, None] * stride_bk + offs_n[None, :])
        acc += tl.dot(a, b, out_dtype=tl.int32)
    y = acc.to(tl.float32) * tl.load(ALPHA + offs_n)[None, :] + tl.load(C + offs_n)[None, :]
    q = _ln_mish_q(y, tl.load(G + offs_n), tl.load(Bt + offs_n), lo, hi, inv_d, nl, zp, seed, offs_n, eps, N,
                   DITHER)
    _store_codes(Q, q, offs_m, offs_n, stride_qm, rmask, BLOCK_M, N, PACK_OUT)


@triton.jit
def k_i8gemm_dequant(A, Bw, ALPHA, C, Y, M, N, K, stride_am, stride_bk, stride_ym,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, PACK_IN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    rmask = offs_m[:, None] < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k0 in range(0, K, BLOCK_K):
        a = _load_codes(A, offs_m, k0, stride_am, rmask, BLOCK_M, BLOCK_K, PACK_IN)
        b = tl.load(Bw + (k0 + offs_k)[:, None] * stride_bk + offs_n[None, :])
        acc += tl.dot(a, b, out_dtype=tl.int32)
    y = acc.to(tl.float32) * tl.load(ALPHA + offs_n)[None, :] + tl.load(C + offs_n)[None, :]
    tl.store(Y + offs_m[:, None] * stride_ym + offs_n[None, :], y, mask=rmask & (offs_n[None, :] < N))


BM1, BM2, BK, BM3, BN3 = 16, 16, 64, 32, 128
W1, W2, W3 = 8, 8, 4


def _codes(M, N, dev, packed):
    if packed:
        return torch.empty((M, N // 2), device=dev, dtype=torch.uint8)
    return torch.empty((M, N), device=dev, dtype=torch.int8)


def ln_mish_quant(y, g, b, lo, hi, d, nl, zp, seed, dither, packed=False):
    M, N = y.shape
    q = _codes(M, N, y.device, packed)
    k_ln_mish_quant[(triton.cdiv(M, BM1),)](y, g, b, q, M, lo, hi, 1.0 / d, float(nl), float(zp), seed, LN_EPS,
                                           y.stride(0), q.stride(0), N=N, BLOCK_M=BM1, DITHER=dither,
                                           PACK_OUT=packed, num_warps=W1)
    return q


def i8gemm_ln_mish_quant(a, bw, alpha, c, g, b, lo, hi, d, nl, zp, seed, dither, packed=False):
    M = a.shape[0]
    K, N = bw.shape
    q = _codes(M, N, a.device, packed)
    k_i8gemm_ln_mish_quant[(triton.cdiv(M, BM2),)](a, bw, alpha, c, g, b, q, M, K, lo, hi, 1.0 / d, float(nl),
                                                  float(zp), seed, LN_EPS, a.stride(0), bw.stride(0), q.stride(0),
                                                  N=N, BLOCK_M=BM2, BLOCK_K=BK, DITHER=dither, PACK_IN=packed,
                                                  PACK_OUT=packed, num_warps=W2)
    return q


def i8gemm_dequant(a, bw, alpha, c, n_out, packed=False):
    M = a.shape[0]
    K = bw.shape[0]
    y = torch.empty((M, n_out), device=a.device, dtype=torch.float32)
    k_i8gemm_dequant[(triton.cdiv(M, BM3), triton.cdiv(n_out, BN3))](a, bw, alpha, c, y, M, n_out, K, a.stride(0),
                                                                    bw.stride(0), y.stride(0), BLOCK_M=BM3,
                                                                    BLOCK_N=BN3, BLOCK_K=BK, PACK_IN=packed,
                                                                    num_warps=W3)
    return y


# ---------------------------------------------------------------------------------------------------- MLPs
def w8_per_channel(W):
    """symmetric per-output-channel int8 RTN: W ~ s[:, None] * Wq."""
    s = W.abs().amax(dim=1).clamp(min=1e-12) / 127.0
    Wq = torch.round(W / s[:, None]).clamp(-128, 127)
    return Wq, s


class IntMLP:
    """layers.mlp(in, [512, 512], out) of TD-MPC2 (NormedLinear, NormedLinear, head) on integer kernels."""

    def __init__(self, seq, names, qrange, k, dither, owner, salt, dyn_head, packed=False):
        L0, L1, L2 = seq[0], seq[1], seq[2]
        dev = L0.weight.device
        self.k, self.nl, self.zp = int(k), (1 << int(k)) - 1, (128 if int(k) == 8 else 0)
        self.packed = bool(packed)
        assert not self.packed or self.k <= 4
        self.dither, self.owner, self.salt, self.dyn_head = bool(dither), owner, int(salt), bool(dyn_head)
        self.W0, self.b0 = L0.weight.detach(), L0.bias.detach()
        self.g0, self.be0 = L0.ln.weight.detach().contiguous(), L0.ln.bias.detach().contiguous()
        self.g1, self.be1 = L1.ln.weight.detach().contiguous(), L1.ln.bias.detach().contiguous()
        self.lo0, self.hi0 = (float(x) for x in qrange[names[0]])
        self.lo1, self.hi1 = (float(x) for x in qrange[names[1]])
        self.d0, self.d1 = (self.hi0 - self.lo0) / self.nl, (self.hi1 - self.lo1) / self.nl
        self.W1qT, self.alpha1, self.c1 = self._pack(L1, self.lo0, self.d0, None)
        self.n_out = L2.out_features
        self.W2qT, self.alpha2, self.c2 = self._pack(L2, self.lo1, self.d1, BN3)
        self.L2 = L2
        self.calls = 0
        self.dev = dev

    def _pack(self, lin, lo_in, d_in, pad_to):
        W = lin.weight.detach().float()
        Wq, s = w8_per_channel(W)
        alpha = d_in * s
        c = (lo_in + self.zp * d_in) * s * Wq.sum(dim=1) + lin.bias.detach().float()
        WqT = Wq.t().contiguous().to(torch.int8)                    # (K, N)
        if pad_to is not None and WqT.shape[1] % pad_to:
            pad = pad_to - WqT.shape[1] % pad_to
            WqT = torch.cat([WqT, torch.zeros((WqT.shape[0], pad), device=WqT.device, dtype=torch.int8)], 1)
            alpha = torch.cat([alpha, torch.zeros(pad, device=alpha.device)])
            c = torch.cat([c, torch.zeros(pad, device=c.device)])
        return WqT.contiguous(), alpha.contiguous().float(), c.contiguous().float()

    def _seed(self, layer):
        self.calls += 1
        return (int(self.owner.base) * 1_000_003 + self.calls * 8 + self.salt + layer) & 0x7FFFFFFF

    def __call__(self, x):
        shp = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1]).float()
        y0 = F.linear(x2, self.W0, self.b0).contiguous()
        q0 = ln_mish_quant(y0, self.g0, self.be0, self.lo0, self.hi0, self.d0, self.nl, self.zp, self._seed(0),
                           self.dither, self.packed)
        q1 = i8gemm_ln_mish_quant(q0, self.W1qT, self.alpha1, self.c1, self.g1, self.be1, self.lo1, self.hi1,
                                  self.d1, self.nl, self.zp, self._seed(1), self.dither, self.packed)
        y2 = i8gemm_dequant(q1, self.W2qT, self.alpha2, self.c2, self.n_out, self.packed)
        if self.dyn_head:
            y2 = self.L2.act(self.L2.ln(y2))
        return y2.reshape(*shp, self.n_out)


class E7:
    """One E7 precision installed on a cached agent for one job (see the module docstring)."""

    def __init__(self, agent, quant, out, spec):
        self.agent, self.quant, self.model = agent, quant, agent.model
        self.prec = spec["e7"]
        self.variant = L.canon_variant(spec["variant"])
        self.k = int(spec.get("k", 0))
        self.base = 0
        m = self.model
        if self.prec == "fp32":
            pass
        elif self.prec == "bf16":
            m.next = lambda z, a, task: self._bf16(m._dynamics, z, a)
            m.reward = lambda z, a, task: self._bf16(m._reward, z, a)
        elif self.prec in ("int8", "int4", "int4p"):
            assert self.k == (8 if self.prec == "int8" else 4), (self.prec, self.k)
            assert self.variant in ("det", "shiter"), self.variant
            qr = json.load(open(f"{out}/scales/{spec['task']}_s{int(spec['ckpt'])}.json"))["quant"]
            dith, pk = self.variant == "shiter", self.prec == "int4p"
            self.dyn = IntMLP(m._dynamics, ("dyn0", "dyn1"), qr, self.k, dith, self, 0, True, pk)
            self.rew = IntMLP(m._reward, ("rew0", "rew1"), qr, self.k, dith, self, 4, False, pk)
            m.next = lambda z, a, task: self.dyn(torch.cat([z, a], dim=-1))
            m.reward = lambda z, a, task: self.rew(torch.cat([z, a], dim=-1))
        else:
            raise ValueError(self.prec)
        quant.wq = self          # per-episode seed via Quantizer.set, begin_step via Quantizer.begin_step

    @staticmethod
    def _bf16(seq, z, a):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = seq(torch.cat([z, a], dim=-1))
        return out.float()

    def new_episode(self, seed):
        self.base = int(seed)
        for mlp in (getattr(self, "dyn", None), getattr(self, "rew", None)):
            if mlp is not None:
                mlp.calls = 0

    def begin_step(self):
        pass

    def restore(self):
        for name in ("next", "reward"):
            if name in self.model.__dict__:
                delattr(self.model, name)
        if self.quant.wq is self:
            self.quant.wq = None

    def info(self):
        d = {"prec": self.prec, "variant": self.variant, "k": self.k}
        if self.prec in ("int8", "int4", "int4p"):
            d.update(weights="W8 per-output-channel RTN (layers 1, 2)",
                     act_container="packed int4 (2 per byte)" if self.prec == "int4p" else "int8",
                     layer0="fp32 cuBLAS GEMM + fused LN/Mish/quant kernel")
        return d


def install(agent, quant, out, spec):
    return E7(agent, quant, out, spec)
