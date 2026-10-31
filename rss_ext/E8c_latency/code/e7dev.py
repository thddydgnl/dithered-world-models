"""E8 (descriptive, 5M only): E7's fused integer kernels with the dither seed read from device memory.

Copy of A_dithered_world_models/main/code/tdmpc2/e7lib.py (int8-container path only; the packed-int4 variants and the
E7 install class are dropped). Changes, all needed for CUDA-graph replay:
  - the scalar host seed argument of the two LN/Mish/quant kernels is replaced by a pointer S to the MLP's device
    state [base, f] and a salt; seed = (base * 1000003 + f * 8 + salt + layer) & 0x7FFFFFFF (same Philox tl.rand on
    columns, so one u vector per layer and forward shared by every row = the sh-iter schedule)
  - the head kernel (k_i8gemm_dequant) increments f on the device (program (0, 0) at its start, dither only), so every
    forward, eager or replayed, draws a new u. E7 incremented a host counter twice per forward (once per layer); the
    seeds differ from E7's but the schedule is the same. The seed loads are placed so that their latency overlaps
    other loads (after the Y load in the layer-0 kernel, before the K loop in the fused layer-1 kernel).
Everything else (block sizes BM1 = BM2 = 16, BK = 64, BM3 = 32, BN3 = 128, warps 8 / 8 / 4, the fused layer-1
kernel holding the full 512-wide row, K x N weights, the dynamics head LN and SimNorm in PyTorch) is E7's.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

LN_EPS = 1e-5


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
def k_ln_mish_quant(Y, G, B, Q, M, lo, hi, inv_d, nl, zp, S, salt, eps, stride_ym, stride_qm,
                    N: tl.constexpr, BLOCK_M: tl.constexpr, DITHER: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, N)
    mask = offs_m[:, None] < M
    y = tl.load(Y + offs_m[:, None] * stride_ym + offs_n[None, :], mask=mask, other=0.0)
    if DITHER:
        seed = (tl.load(S) * 1000003 + tl.load(S + 1) * 8 + salt) & 0x7FFFFFFF
    else:
        seed = 0
    q = _ln_mish_q(y, tl.load(G + offs_n), tl.load(B + offs_n), lo, hi, inv_d, nl, zp, seed, offs_n, eps, N, DITHER)
    tl.store(Q + offs_m[:, None] * stride_qm + offs_n[None, :], q, mask=mask)


@triton.jit
def k_i8gemm_ln_mish_quant(A, Bw, ALPHA, C, G, Bt, Q, M, K, lo, hi, inv_d, nl, zp, S, salt, eps,
                           stride_am, stride_bk, stride_qm,
                           N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, DITHER: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, BLOCK_K)
    rmask = offs_m[:, None] < M
    if DITHER:   # device seed loaded before the K loop (its latency overlaps the GEMM)
        seed = (tl.load(S) * 1000003 + tl.load(S + 1) * 8 + salt) & 0x7FFFFFFF
    else:
        seed = 0
    acc = tl.zeros((BLOCK_M, N), dtype=tl.int32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(A + offs_m[:, None] * stride_am + (k0 + offs_k)[None, :], mask=rmask, other=0)
        b = tl.load(Bw + (k0 + offs_k)[:, None] * stride_bk + offs_n[None, :])
        acc += tl.dot(a, b, out_dtype=tl.int32)
    y = acc.to(tl.float32) * tl.load(ALPHA + offs_n)[None, :] + tl.load(C + offs_n)[None, :]
    q = _ln_mish_q(y, tl.load(G + offs_n), tl.load(Bt + offs_n), lo, hi, inv_d, nl, zp, seed, offs_n, eps, N,
                   DITHER)
    tl.store(Q + offs_m[:, None] * stride_qm + offs_n[None, :], q, mask=rmask)


@triton.jit
def k_i8gemm_dequant(A, Bw, ALPHA, C, Y, S, M, N, K, stride_am, stride_bk, stride_ym,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BUMP: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if BUMP:   # head of a dither forward: program (0, 0) increments the forward counter first (nothing here reads it)
        if (pid_m == 0) & (pid_n == 0):
            tl.store(S + 1, tl.load(S + 1) + 1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    rmask = offs_m[:, None] < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(A + offs_m[:, None] * stride_am + (k0 + offs_k)[None, :], mask=rmask, other=0)
        b = tl.load(Bw + (k0 + offs_k)[:, None] * stride_bk + offs_n[None, :])
        acc += tl.dot(a, b, out_dtype=tl.int32)
    y = acc.to(tl.float32) * tl.load(ALPHA + offs_n)[None, :] + tl.load(C + offs_n)[None, :]
    tl.store(Y + offs_m[:, None] * stride_ym + offs_n[None, :], y, mask=rmask & (offs_n[None, :] < N))


BM1, BM2, BK, BM3, BN3 = 16, 16, 64, 32, 128
W1, W2, W3 = 8, 8, 4


def w8_per_channel(W):
    s = W.abs().amax(dim=1).clamp(min=1e-12) / 127.0
    Wq = torch.round(W / s[:, None]).clamp(-128, 127)
    return Wq, s


class E7IntMLP:
    """E7's IntMLP (int8-container codes) with the device-side dither state. _forward returns (q0, q1, out)."""

    def __init__(self, seq, qr0, qr1, dither, salt, dyn_head, k=4):
        L0, L1, L2 = seq[0], seq[1], seq[2]
        dev = L0.weight.device
        self.k, self.nl, self.zp = int(k), float((1 << int(k)) - 1), float(128 if int(k) == 8 else 0)
        self.dither, self.salt, self.dyn_head = bool(dither), int(salt), bool(dyn_head)
        self.W0, self.b0 = L0.weight.detach(), L0.bias.detach()
        self.g0, self.be0 = L0.ln.weight.detach().contiguous(), L0.ln.bias.detach().contiguous()
        self.g1, self.be1 = L1.ln.weight.detach().contiguous(), L1.ln.bias.detach().contiguous()
        self.lo0, self.hi0 = float(qr0[0]), float(qr0[1])
        self.lo1, self.hi1 = float(qr1[0]), float(qr1[1])
        self.d0, self.d1 = (self.hi0 - self.lo0) / self.nl, (self.hi1 - self.lo1) / self.nl
        self.width = L0.out_features
        assert self.width == triton.next_power_of_2(self.width), "E7 fused kernels need a power-of-2 width"
        self.W1qT, self.alpha1, self.c1, self.s1 = self._pack(L1, self.lo0, self.d0, None)
        self.n_out = L2.out_features
        self.W2qT, self.alpha2, self.c2, self.s2 = self._pack(L2, self.lo1, self.d1, BN3)
        self.L2 = L2
        self.state = torch.zeros(2, dtype=torch.int64, device=dev)
        self.seq = seq

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
        return WqT.contiguous(), alpha.contiguous().float(), c.contiguous().float(), s.float()

    @property
    def W1q(self):   # (N, K) view for the smoke reference
        return self.W1qT.t()

    @property
    def W2q(self):
        return self.W2qT.t()

    def new_episode(self, base):
        self.state[0] = int(base)
        self.state[1] = 0

    def _forward(self, x):
        x2 = x.reshape(-1, x.shape[-1]).float()
        M = x2.shape[0]
        N = self.width
        y0 = F.linear(x2, self.W0, self.b0).contiguous()
        q0 = torch.empty((M, N), device=x2.device, dtype=torch.int8)
        k_ln_mish_quant[(triton.cdiv(M, BM1),)](y0, self.g0, self.be0, q0, M, self.lo0, self.hi0, 1.0 / self.d0,
                                               self.nl, self.zp, self.state, self.salt, LN_EPS, y0.stride(0),
                                               q0.stride(0), N=N, BLOCK_M=BM1, DITHER=self.dither, num_warps=W1)
        K, N1 = self.W1qT.shape
        q1 = torch.empty((M, N1), device=x2.device, dtype=torch.int8)
        k_i8gemm_ln_mish_quant[(triton.cdiv(M, BM2),)](q0, self.W1qT, self.alpha1, self.c1, self.g1, self.be1, q1, M,
                                                      K, self.lo1, self.hi1, 1.0 / self.d1, self.nl, self.zp,
                                                      self.state, self.salt + 1, LN_EPS, q0.stride(0),
                                                      self.W1qT.stride(0), q1.stride(0), N=N1, BLOCK_M=BM2,
                                                      BLOCK_K=BK, DITHER=self.dither, num_warps=W2)
        K2 = self.W2qT.shape[0]
        y2 = torch.empty((M, self.n_out), device=x2.device, dtype=torch.float32)
        k_i8gemm_dequant[(triton.cdiv(M, BM3), triton.cdiv(self.n_out, BN3))](
            q1, self.W2qT, self.alpha2, self.c2, y2, self.state, M, self.n_out, K2, q1.stride(0), self.W2qT.stride(0),
            y2.stride(0), BLOCK_M=BM3, BLOCK_N=BN3, BLOCK_K=BK, BUMP=self.dither, num_warps=W3)
        if self.dyn_head:
            y2 = self.L2.act(self.L2.ln(y2))
        return q0, q1, y2

    def __call__(self, x):
        shp = x.shape[:-1]
        return self._forward(x)[2].reshape(*shp, self.n_out)
