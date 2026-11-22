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

E17 additions (tmlr_ext/DESIGN_tmlr.md 7; tmlr_ext/K_kernel). Everything above this note and every kernel, class and
precision branch of E7 is unchanged; the new precisions only add code paths:
  intk   (det | shiter | detshift) k-bit activation codes (k = 2..7, codes 0..2^k - 1 in int8 containers, zp = 0) on the
         E0 per-tensor grids, W8 weights as E7.
           det, shiter   E7's IntMLP and kernels unchanged (they always took k as a runtime parameter; E7's class only
                         restricted its precisions to k = 8 / 4)
           detshift      IntMLP2 on the v2 kernels: per layer and forward one offset row o ~ U[-1/2, 1/2) shared by all
                         rows, o = Philox(seed, column) - 1/2 with E7's seed schedule (the same uniforms as sh-iter's u
                         at the same call), codes q = clamp(rint(x - o), 0, nl) (2^k levels), and the next layer adds
                         W (delta o) to its bias: c_eff = c + alpha * (Wq o) (alpha = delta s), one matrix-vector product
                         per layer call (torch.addmv on the dequantized-scale copy alpha[:, None] Wq). The quantize kernel
                         reads the same o buffer, so the codes and the correction use the identical offsets. Dequantized
                         value = lo + delta (q + o), i.e. the fake-quant det+shift of qlib with W8 weights.
  intpc  (det | shiter) per-channel activation grids [lo_j, hi_j] (e5lib perchan, re-derived on the pod) folded into the
         next layer: W' = W diag(delta), bias' = b + W lo (fp32 W), W' quantized per output channel to int8 (W8 RTN),
         codes q_j = clamp(rint | floor(x_j + u_j)) on the unit's own grid (v2 kernels with per-column lo / hi / 1/delta)
  bf16p  pre-cast BF16 (E8c's bf16cast): copies of the dynamics and reward MLPs with every parameter cast to bf16 once
         per job, bf16 input, fp32 outputs, no autocast (LayerNorm / Mish / SimNorm in bf16, as E8c)
"""
import copy
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

    def __init__(self, agent, quant, out, spec, qr=None, pcr=None):
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
        elif self.prec == "intk":       # E17: k-bit codes, per-tensor E0 grids (qr given by the caller)
            assert 2 <= self.k <= 7, self.k
            assert self.variant in ("det", "shiter", "detshift"), self.variant
            assert qr is not None
            if self.variant == "detshift":
                self.dyn = IntMLP2(m._dynamics, ("dyn0", "dyn1"), qr, self.k, "detshift", self, 0, True)
                self.rew = IntMLP2(m._reward, ("rew0", "rew1"), qr, self.k, "detshift", self, 4, False)
            else:
                dith = self.variant == "shiter"
                self.dyn = IntMLP(m._dynamics, ("dyn0", "dyn1"), qr, self.k, dith, self, 0, True, False)
                self.rew = IntMLP(m._reward, ("rew0", "rew1"), qr, self.k, dith, self, 4, False, False)
            m.next = lambda z, a, task: self.dyn(torch.cat([z, a], dim=-1))
            m.reward = lambda z, a, task: self.rew(torch.cat([z, a], dim=-1))
        elif self.prec == "intpc":      # E17: per-channel grids folded into the next layer (pcr given by the caller)
            assert 2 <= self.k <= 7, self.k
            assert self.variant in ("det", "shiter"), self.variant
            assert pcr is not None
            self.dyn = IntMLP2(m._dynamics, ("dyn0", "dyn1"), None, self.k, self.variant, self, 0, True, pcr)
            self.rew = IntMLP2(m._reward, ("rew0", "rew1"), None, self.k, self.variant, self, 4, False, pcr)
            m.next = lambda z, a, task: self.dyn(torch.cat([z, a], dim=-1))
            m.reward = lambda z, a, task: self.rew(torch.cat([z, a], dim=-1))
        elif self.prec == "bf16p":      # E17: pre-cast BF16 (E8c bf16cast)
            assert self.variant == "none", self.variant
            self.dyn16, self.rew16 = bf16_copy(m._dynamics), bf16_copy(m._reward)
            m.next = lambda z, a, task: self.dyn16(torch.cat([z, a], dim=-1).to(torch.bfloat16)).float()
            m.reward = lambda z, a, task: self.rew16(torch.cat([z, a], dim=-1).to(torch.bfloat16)).float()
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
        elif self.prec in ("intk", "intpc"):
            d.update(weights="W8 per-output-channel RTN (layers 1, 2)" + (
                " of the folded W diag(delta)" if self.prec == "intpc" else ""),
                act_container="int8 (codes 0..2^k-1)", layer0="fp32 cuBLAS GEMM + fused LN/Mish/quant kernel",
                grid="per-channel folded" if self.prec == "intpc" else "per-tensor E0",
                kernels=("v2 (E17)" if (self.prec == "intpc" or self.variant == "detshift") else "E7 unchanged"),
                offset_correction="torch.addmv per layer call" if self.variant == "detshift" else None)
        elif self.prec == "bf16p":
            d.update(weights="bf16 copies of the dynamics / reward MLPs (all parameters), no autocast")
        return d


def install(agent, quant, out, spec, qr=None, pcr=None):
    return E7(agent, quant, out, spec, qr=qr, pcr=pcr)


# ---------------------------------------------------------------------------------------------------- E17 additions
@triton.jit
def _ln_mish_q2(y, g, b, LO, HI, INVD, O, lo, hi, inv_d, nl, zp, seed, offs_n, eps, N: tl.constexpr,
                ROUND: tl.constexpr, PC: tl.constexpr):
    """E7's LN -> Mish -> quantize with ROUND 0 det rint(x), 1 sh-iter floor(x + u) (u = Philox(seed, column)),
    2 det+shift rint(x - o) (o read from O, one row for all rows); PC: per-column lo / hi / 1/delta vectors."""
    mean = tl.sum(y, axis=1) / N
    yc = y - mean[:, None]
    var = tl.sum(yc * yc, axis=1) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    z = yc * rstd[:, None] * g[None, :] + b[None, :]
    m = z * libdevice.tanh(libdevice.log1p(tl.exp(z)))
    if PC:
        lov = tl.load(LO + offs_n)
        hiv = tl.load(HI + offs_n)
        idv = tl.load(INVD + offs_n)
        x = (tl.minimum(tl.maximum(m, lov[None, :]), hiv[None, :]) - lov[None, :]) * idv[None, :]
    else:
        x = (tl.minimum(tl.maximum(m, lo), hi) - lo) * inv_d
    if ROUND == 1:
        u = tl.rand(seed, offs_n)
        q = tl.floor(x + u[None, :])
    elif ROUND == 2:
        o = tl.load(O + offs_n)
        q = libdevice.rint(x - o[None, :])
    else:
        q = libdevice.rint(x)
    q = tl.minimum(tl.maximum(q, 0.0), nl)
    return (q - zp).to(tl.int8)


@triton.jit
def k_ln_mish_quant2(Y, G, B, Q, LO, HI, INVD, O, M, lo, hi, inv_d, nl, zp, seed, eps, stride_ym, stride_qm,
                     N: tl.constexpr, BLOCK_M: tl.constexpr, ROUND: tl.constexpr, PC: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, N)
    mask = offs_m[:, None] < M
    y = tl.load(Y + offs_m[:, None] * stride_ym + offs_n[None, :], mask=mask, other=0.0)
    q = _ln_mish_q2(y, tl.load(G + offs_n), tl.load(B + offs_n), LO, HI, INVD, O, lo, hi, inv_d, nl, zp, seed,
                    offs_n, eps, N, ROUND, PC)
    tl.store(Q + offs_m[:, None] * stride_qm + offs_n[None, :], q, mask=mask)


@triton.jit
def k_i8gemm_ln_mish_quant2(A, Bw, ALPHA, C, G, Bt, Q, LO, HI, INVD, O, M, K, lo, hi, inv_d, nl, zp, seed, eps,
                            stride_am, stride_bk, stride_qm, N: tl.constexpr, BLOCK_M: tl.constexpr,
                            BLOCK_K: tl.constexpr, ROUND: tl.constexpr, PC: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, BLOCK_K)
    rmask = offs_m[:, None] < M
    acc = tl.zeros((BLOCK_M, N), dtype=tl.int32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(A + offs_m[:, None] * stride_am + (k0 + offs_k)[None, :], mask=rmask, other=0)
        b = tl.load(Bw + (k0 + offs_k)[:, None] * stride_bk + offs_n[None, :])
        acc += tl.dot(a, b, out_dtype=tl.int32)
    y = acc.to(tl.float32) * tl.load(ALPHA + offs_n)[None, :] + tl.load(C + offs_n)[None, :]
    q = _ln_mish_q2(y, tl.load(G + offs_n), tl.load(Bt + offs_n), LO, HI, INVD, O, lo, hi, inv_d, nl, zp, seed,
                    offs_n, eps, N, ROUND, PC)
    tl.store(Q + offs_m[:, None] * stride_qm + offs_n[None, :], q, mask=rmask)


@triton.jit
def k_offset(O, seed, N: tl.constexpr):
    """o = Philox(seed, column) - 1/2: the det+shift offset row of one layer call (same uniforms as sh-iter's u)."""
    offs = tl.arange(0, N)
    tl.store(O + offs, tl.rand(seed, offs) - 0.5)


ROUNDS = {"det": 0, "shiter": 1, "detshift": 2}


def offsets(seed, n, dev):
    o = torch.empty(n, device=dev, dtype=torch.float32)
    k_offset[(1,)](o, seed, N=n)
    return o


def ln_mish_quant2(y, g, b, lo, hi, d, nl, zp, seed, rnd, o=None, pcv=None):
    """v2 layer-0 kernel: rnd in ROUNDS; o (N,) offsets for detshift; pcv = (lo_vec, hi_vec, invd_vec) for PC."""
    M, N = y.shape
    q = torch.empty((M, N), device=y.device, dtype=torch.int8)
    pc = pcv is not None
    LO, HI, INVD = pcv if pc else (y, y, y)
    k_ln_mish_quant2[(triton.cdiv(M, BM1),)](y, g, b, q, LO, HI, INVD, o if o is not None else y, M,
                                            0.0 if pc else lo, 0.0 if pc else hi, 0.0 if pc else 1.0 / d, float(nl),
                                            float(zp), seed, LN_EPS, y.stride(0), q.stride(0), N=N, BLOCK_M=BM1,
                                            ROUND=ROUNDS[rnd], PC=pc, num_warps=W1)
    return q


def i8gemm_ln_mish_quant2(a, bw, alpha, c, g, b, lo, hi, d, nl, zp, seed, rnd, o=None, pcv=None):
    M = a.shape[0]
    K, N = bw.shape
    q = torch.empty((M, N), device=a.device, dtype=torch.int8)
    pc = pcv is not None
    LO, HI, INVD = pcv if pc else (alpha, alpha, alpha)
    k_i8gemm_ln_mish_quant2[(triton.cdiv(M, BM2),)](a, bw, alpha, c, g, b, q, LO, HI, INVD,
                                                   o if o is not None else alpha, M, K, 0.0 if pc else lo,
                                                   0.0 if pc else hi, 0.0 if pc else 1.0 / d, float(nl), float(zp),
                                                   seed, LN_EPS, a.stride(0), bw.stride(0), q.stride(0), N=N,
                                                   BLOCK_M=BM2, BLOCK_K=BK, ROUND=ROUNDS[rnd], PC=pc, num_warps=W2)
    return q


def bf16_copy(seq):
    """pre-cast BF16 copy of an MLP (E8c bf16cast); forward hooks of the original modules (Quantizer) are dropped."""
    c = copy.deepcopy(seq)
    for mod in c.modules():
        mod._forward_hooks.clear()
        mod._forward_pre_hooks.clear()
    return c.to(torch.bfloat16).eval()


def _pad_cols(WqT, alpha, c, pad_to):
    if pad_to is not None and WqT.shape[1] % pad_to:
        pad = pad_to - WqT.shape[1] % pad_to
        WqT = torch.cat([WqT, torch.zeros((WqT.shape[0], pad), device=WqT.device, dtype=torch.int8)], 1)
        alpha = torch.cat([alpha, torch.zeros(pad, device=alpha.device)])
        c = torch.cat([c, torch.zeros(pad, device=c.device)])
    return WqT.contiguous(), alpha.contiguous().float(), c.contiguous().float()


class IntMLP2(IntMLP):
    """E17 MLP on the v2 kernels: det+shift on the per-tensor E0 grids (pc None), or det / sh-iter on per-channel grids
    folded into the next layer (pc = {layer: (lo (W,), hi (W,))}). Layer layout, W8 weights, head kernel, seed schedule
    and episode handling are E7's IntMLP."""

    def __init__(self, seq, names, qrange, k, rounding, owner, salt, dyn_head, pc=None):
        assert rounding in ROUNDS, rounding
        self.rounding, self.pc = rounding, pc is not None
        if not self.pc:
            assert rounding == "detshift"
            super().__init__(seq, names, qrange, k, False, owner, salt, dyn_head, False)
            # GEMV copies alpha[:, None] * Wq (N_pad, K) for the offset correction c_eff = c + alpha (Wq o)
            self.Wa1 = (self.W1qT.t().float() * self.alpha1[:, None]).contiguous()
            self.Wa2 = (self.W2qT.t().float() * self.alpha2[:, None]).contiguous()
            self.pcv0 = self.pcv1 = None
            return
        assert rounding in ("det", "shiter")
        L0, L1, L2 = seq[0], seq[1], seq[2]
        dev = L0.weight.device
        self.k, self.nl, self.zp = int(k), (1 << int(k)) - 1, 0
        self.packed, self.dither = False, rounding == "shiter"
        self.owner, self.salt, self.dyn_head = owner, int(salt), bool(dyn_head)
        self.W0, self.b0 = L0.weight.detach(), L0.bias.detach()
        self.g0, self.be0 = L0.ln.weight.detach().contiguous(), L0.ln.bias.detach().contiguous()
        self.g1, self.be1 = L1.ln.weight.detach().contiguous(), L1.ln.bias.detach().contiguous()
        lo0, hi0 = (torch.as_tensor(v, device=dev, dtype=torch.float32).contiguous() for v in pc[names[0]])
        lo1, hi1 = (torch.as_tensor(v, device=dev, dtype=torch.float32).contiguous() for v in pc[names[1]])
        d0, d1 = (hi0 - lo0) / self.nl, (hi1 - lo1) / self.nl
        self.lo0v, self.hi0v, self.d0v, self.lo1v, self.hi1v, self.d1v = lo0, hi0, d0, lo1, hi1, d1
        self.pcv0 = (lo0, hi0, (1.0 / d0).contiguous())
        self.pcv1 = (lo1, hi1, (1.0 / d1).contiguous())
        self.lo0 = self.hi0 = self.lo1 = self.hi1 = self.d0 = self.d1 = 0.0
        self.W1qT, self.alpha1, self.c1, self.W1fold = self._pack_fold(L1, lo0, d0, None)
        self.n_out = L2.out_features
        self.W2qT, self.alpha2, self.c2, self.W2fold = self._pack_fold(L2, lo1, d1, BN3)
        self.L2 = L2
        self.calls = 0
        self.dev = dev

    def _pack_fold(self, lin, lo_in, d_in, pad_to):
        """per-channel fold: W' = W diag(d_in), bias' = b + W lo_in (fp32 W), W' -> W8 per output channel."""
        W = lin.weight.detach().float()
        Wf = W * d_in[None, :]
        Wq, s = w8_per_channel(Wf)
        c = lin.bias.detach().float() + W @ lo_in
        WqT = Wq.t().contiguous().to(torch.int8)
        WqT, alpha, c = _pad_cols(WqT, s, c, pad_to)
        return WqT, alpha, c, (Wq, s)

    def __call__(self, x):
        shp = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1]).float()
        y0 = F.linear(x2, self.W0, self.b0).contiguous()
        s0, s1 = self._seed(0), self._seed(1)
        if self.rounding == "detshift":
            Wd = self.W1qT.shape[0]
            o0 = offsets(s0, Wd, x2.device)
            o1 = offsets(s1, self.W1qT.shape[1], x2.device)
            q0 = ln_mish_quant2(y0, self.g0, self.be0, self.lo0, self.hi0, self.d0, self.nl, self.zp, s0, "detshift",
                                o=o0)
            c1 = torch.addmv(self.c1, self.Wa1, o0)
            q1 = i8gemm_ln_mish_quant2(q0, self.W1qT, self.alpha1, c1, self.g1, self.be1, self.lo1, self.hi1,
                                       self.d1, self.nl, self.zp, s1, "detshift", o=o1)
            c2 = torch.addmv(self.c2, self.Wa2, o1)
            self.last_offsets = (o0, o1)
        else:
            q0 = ln_mish_quant2(y0, self.g0, self.be0, 0.0, 0.0, 1.0, self.nl, self.zp, s0, self.rounding,
                                pcv=self.pcv0)
            q1 = i8gemm_ln_mish_quant2(q0, self.W1qT, self.alpha1, self.c1, self.g1, self.be1, 0.0, 0.0, 1.0,
                                       self.nl, self.zp, s1, self.rounding, pcv=self.pcv1)
            c2 = self.c2
        self.last_codes = (q0, q1)
        y2 = i8gemm_dequant(q1, self.W2qT, self.alpha2, c2, self.n_out, False)
        if self.dyn_head:
            y2 = self.L2.act(self.L2.ln(y2))
        return y2.reshape(*shp, self.n_out)
