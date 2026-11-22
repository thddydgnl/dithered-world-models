"""E8 (rss_ext/DESIGN_ext.md 1): shared library for the latency map of low-bit integer kernels in TD-MPC2 world models.

Models: the TD-MPC2 WorldModel of the official code (commit 332f19e) at model_size 5 / 19 / 48 / 317, single-task config
(no task embedding), observation 39, action 4 (mw-push dims), random init with torch.manual_seed(0), no checkpoint.
The official init zeroes the reward head weight; it is kept for the latency runs (latency does not depend on values).

Precision paths of the dynamics and reward MLPs (encoder, Q-ensemble and policy prior always fp32 with TF32 off):
  fp32   the unchanged modules, TF32 off (PyTorch / TD-MPC2 eval default = E7's fp32)
  tf32   the same modules with TF32 matmuls allowed inside the dynamics / reward calls only
  bf16   the modules under torch.autocast(bfloat16, cache_enabled=False), output cast to fp32 (E7's bf16; the weight
         cast cache is off for CUDA-graph capture; it has no effect because every MLP call is its own autocast region)
  int    (det | shiter): E7's numeric path (W8 symmetric per-output-channel RTN weights in layers 1 and 2, 4-bit
         activation codes in int8 containers on per-layer [lo, hi] grids, layer 0 = fp32 GEMM on the unquantized
         input [z, a]) in a size-independent implementation:
           layer 0  F.linear (cuBLAS fp32)         -> row kernel: LN -> Mish -> 4-bit code
           layer 1  int8 GEMM, dequant (+bias)     -> row kernel: LN -> Mish -> 4-bit code
           layer 2  int8 GEMM, dequant (+bias)     -> dynamics: row kernel LN -> SimNorm; reward: the 101 logits
         GEMM 'a': Triton tiled int8 GEMM (int32 accumulation) with a dequant epilogue (fp32 out); per-shape tile
                   config from the tuning table (tune_gemm_a)
         GEMM 'b': torch._int_mm (cuBLASLt, int32 out); the dequant is the first step of the row kernel
  e7     (5M only, descriptive): E7's fused kernels, e7dev.py (= main/code/tdmpc2/e7lib.py with a device-side seed)
Dither (shiter): q = floor(x + u) with u = Philox(seed, column) shared by every row (candidate) of the forward call;
  seed(f, l) = (base * 1000003 + f * 8 + salt + l) & 0x7FFFFFFF, where [base, f] is a device int64 tensor of the MLP
  (salt 0 dynamics, 4 reward; layer l = 0 / 1) and f counts that MLP's forward calls; u vectors live in a device
  buffer U (2, width). Each u vector is drawn once (not once per program), one kernel ahead, chunked over the programs
  of the generating kernel:
    layer-0 row kernel   reads U[0] (u of layer 0, forward f); its programs draw U[1] = Philox(seed(f, 1))
    layer-1 row kernel   reads U[1]; program 0 increments f (no program of this kernel reads f)
    head kernel          (dynamics LN/SimNorm row kernel, reward dequant row kernel on path 'b', or the reward head
                         GEMM on path 'a') its programs draw U[0] = Philox(seed(f + 1, 0)) for the next forward
    new_episode(base)    sets [base, 0] and draws U[0] = Philox(seed(0, 0))
  No kernel both writes and reads the same location (stream order separates writer and reader), so the eager and the
  CUDA-graph forwards draw a fresh u at every forward. det: q = rint(x) (round half to even), no counter, no buffer.
  The smoke check compares every forward with the reference u = Philox(seed(f, l)) of the pre-forward counter.
Row kernel launch config (fixed, the same for det and shiter): BLOCK_W = next_pow2(width), one tile of 4096 elements
  per program (BLOCK_M = max(1, 4096 // BLOCK_W) rows, TILES = 1; chosen by the det row-kernel time, dev_rowtiles.py),
  8 warps; chunk of a drawn u vector per program: max(16, next_pow2(ceil(width / programs))).
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

TDMPC2_DIR = os.environ.get("TDMPC2_DIR", "/root/tdmpc2/tdmpc2")
if TDMPC2_DIR not in sys.path:
    sys.path.insert(0, TDMPC2_DIR)

import hydra.utils  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

hydra.utils.get_original_cwd = lambda: os.getcwd()  # parse_cfg expects a running hydra app

from common.parser import parse_cfg  # noqa: E402
from tdmpc2 import TDMPC2  # noqa: E402

LN_EPS = 1e-5
SIZES = (5, 19, 48, 317)
NS = (512, 2048, 8192)
OBS_DIM, ACT_DIM = 39, 4
KBITS = 4
ITERS, HORIZON, N_PI = 6, 3, 24          # MPPI defaults (config.yaml); elites 64, temperature 0.5 are used by eager-step
SALT = {"dyn": 0, "rew": 4}
FLOAT_PRECS = ("fp32", "tf32", "bf16")


# ---------------------------------------------------------------------------------------------------- kernels
@triton.jit
def _row_tile(y, offs_m, offs_w, wmask, u, ALPHA, C, G, B, OUT, M, W, stride_o, lo, hi, inv_d, nl, zp, eps,
              BLOCK_M: tl.constexpr, BLOCK_W: tl.constexpr, DEQ: tl.constexpr, MODE: tl.constexpr,
              DITHER: tl.constexpr, SIMD: tl.constexpr):
    mask = (offs_m[:, None] < M) & wmask[None, :]
    y = y.to(tl.float32)
    if DEQ:
        al = tl.load(ALPHA + offs_w, mask=wmask, other=0.0)
        cc = tl.load(C + offs_w, mask=wmask, other=0.0)
        y = y * al[None, :] + cc[None, :]
    optr = OUT + offs_m[:, None] * stride_o + offs_w[None, :]
    if MODE == 2:
        tl.store(optr, y, mask=mask)
    else:
        g = tl.load(G + offs_w, mask=wmask, other=0.0)
        b = tl.load(B + offs_w, mask=wmask, other=0.0)
        y = tl.where(mask, y, 0.0)
        mean = tl.sum(y, axis=1) / W
        yc = tl.where(mask, y - mean[:, None], 0.0)
        var = tl.sum(yc * yc, axis=1) / W
        rstd = 1.0 / tl.sqrt(var + eps)
        z = yc * rstd[:, None] * g[None, :] + b[None, :]
        if MODE == 0:
            m = z * libdevice.tanh(libdevice.log1p(tl.exp(z)))
            v = (tl.minimum(tl.maximum(m, lo), hi) - lo) * inv_d
            if DITHER:
                q = tl.floor(v + u[None, :])
            else:
                q = libdevice.rint(v)
            q = tl.minimum(tl.maximum(q, 0.0), nl)
            tl.store(optr, (q - zp).to(tl.int8), mask=mask)
        else:
            z3 = tl.reshape(z, (BLOCK_M, BLOCK_W // SIMD, SIMD))
            mx = tl.max(z3, axis=2)
            e = tl.exp(z3 - mx[:, :, None])
            den = tl.sum(e, axis=2)
            o = tl.reshape(e / den[:, :, None], (BLOCK_M, BLOCK_W))
            tl.store(optr, o, mask=mask)


@triton.jit
def _draw_chunk(S, UG, WG, gsalt, pid, GEN_BLOCK: tl.constexpr):
    """this program's chunk of u = Philox(seed(f, gsalt), column) for the current counter f (columns < WG)."""
    offs_g = pid * GEN_BLOCK + tl.arange(0, GEN_BLOCK)
    seed = (tl.load(S) * 1000003 + tl.load(S + 1) * 8 + gsalt) & 0x7FFFFFFF
    tl.store(UG + offs_g, tl.rand(seed, offs_g), mask=offs_g < WG)


@triton.jit
def k_row(X, ALPHA, C, G, B, OUT, S, U, UG, M, W, WG, stride_x, stride_o, lo, hi, inv_d, nl, zp, gsalt, eps,
          BLOCK_M: tl.constexpr, BLOCK_W: tl.constexpr, TILES: tl.constexpr, DEQ: tl.constexpr,
          MODE: tl.constexpr, DITHER: tl.constexpr, GEN: tl.constexpr, GEN_BLOCK: tl.constexpr,
          BUMP: tl.constexpr, SIMD: tl.constexpr):
    """Row kernel over a (M, W) matrix X (fp32, or int32 GEMM accumulators when DEQ); each program handles TILES
    consecutive tiles of BLOCK_M rows.
    MODE 0: [dequant] -> LayerNorm -> Mish -> k-bit code (int8 out, code - zp)
    MODE 1: [dequant] -> LayerNorm -> SimNorm (fp32 out; dynamics head)
    MODE 2: [dequant] only (fp32 out; reward head logits)
    DITHER (MODE 0): floor(x + u) with u = U[0:W] (drawn one kernel ahead; the same vector for every row)
    GEN: after its tiles, every program draws its chunk of UG = Philox(seed(f, gsalt)) (columns < WG)
    BUMP: program 0 increments the forward counter S[1] at its end (never together with GEN)"""
    tl.static_assert(not (BUMP and GEN))
    pid = tl.program_id(0)
    offs_w = tl.arange(0, BLOCK_W)
    wmask = offs_w < W
    row0 = pid * (TILES * BLOCK_M)
    offs_m = row0 + tl.arange(0, BLOCK_M)
    y = tl.load(X + offs_m[:, None] * stride_x + offs_w[None, :], mask=(offs_m[:, None] < M) & wmask[None, :],
                other=0)
    if DITHER:
        u = tl.load(U + offs_w, mask=wmask, other=0.0)
    else:
        u = tl.zeros((BLOCK_W,), dtype=tl.float32)
    if BUMP:
        cnt = tl.load(S + 1)
    _row_tile(y, offs_m, offs_w, wmask, u, ALPHA, C, G, B, OUT, M, W, stride_o, lo, hi, inv_d, nl, zp, eps,
              BLOCK_M, BLOCK_W, DEQ, MODE, DITHER, SIMD)
    for t in range(1, TILES):
        offs_m = row0 + t * BLOCK_M + tl.arange(0, BLOCK_M)
        y = tl.load(X + offs_m[:, None] * stride_x + offs_w[None, :], mask=(offs_m[:, None] < M) & wmask[None, :],
                    other=0)
        _row_tile(y, offs_m, offs_w, wmask, u, ALPHA, C, G, B, OUT, M, W, stride_o, lo, hi, inv_d, nl, zp, eps,
                  BLOCK_M, BLOCK_W, DEQ, MODE, DITHER, SIMD)
    if GEN:
        _draw_chunk(S, UG, WG, gsalt, pid, GEN_BLOCK)
    if BUMP:
        if pid == 0:
            tl.store(S + 1, cnt + 1)


@triton.jit
def k_gemm_a(A, Wt, ALPHA, C, Y, S, UG, M, N, K, WG, gsalt, stride_am, stride_wn, stride_ym,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
             GEN: tl.constexpr, GEN_BLOCK: tl.constexpr):
    """Y (M, N) fp32 = (A (M, K) int8 @ Wt (N, K)^T int8) * ALPHA[n] + C[n]; int32 accumulation; K % BLOCK_K == 0.
    Grouped program order for L2 reuse; out-of-range rows / columns are wrapped on load and masked on store.
    GEN (reward head of a dither forward on path 'a'): every program draws its chunk of UG after its tile."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = offs_m % M
    rn = offs_n % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + rm[:, None] * stride_am + offs_k[None, :]
    b_ptrs = Wt + rn[None, :] * stride_wn + offs_k[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b, out_dtype=tl.int32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
    alpha = tl.load(ALPHA + rn)
    c = tl.load(C + rn)
    y = acc.to(tl.float32) * alpha[None, :] + c[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Y + offs_m[:, None] * stride_ym + offs_n[None, :], y, mask=mask)
    if GEN:
        _draw_chunk(S, UG, WG, gsalt, pid, GEN_BLOCK)


@triton.jit
def k_u(S, OUT, W, salt, BLOCK_W: tl.constexpr):
    """Reference dither vector of the current counter state (smoke checks only)."""
    offs = tl.arange(0, BLOCK_W)
    base = tl.load(S)
    cnt = tl.load(S + 1)
    seed = (base * 1000003 + cnt * 8 + salt) & 0x7FFFFFFF
    tl.store(OUT + offs, tl.rand(seed, offs), mask=offs < W)


# ---------------------------------------------------------------------------------------------------- launchers
_ROWCFG = {}


def row_cfg(M, W):
    """(BLOCK_W, BLOCK_M, TILES, grid): one tile of max(1, 4096 // BLOCK_W) rows per program (TILES = 1, fixed by the
    det row-kernel time in dev_rowtiles.py; the kernel keeps the generic tile loop)."""
    c = _ROWCFG.get((M, W))
    if c is None:
        bw = triton.next_power_of_2(W)
        bm = max(1, 4096 // bw)
        tiles = 1
        c = (bw, bm, tiles, (triton.cdiv(M, bm * tiles),))
        _ROWCFG[(M, W)] = c
    return c


def gen_block(WG, programs):
    return max(16, triton.next_power_of_2(triton.cdiv(WG, programs)))


def row_launch(x, alpha, c, g, b, out, state, M, W, lo, hi, d, nl, zp, mode, deq, dither, u=None, ug=None, wg=0,
               gsalt=0, bump=False, simd=8):
    """u: the (W,) dither vector read by a DITHER kernel; ug (wg,): the u vector this launch draws (GEN) for seed
    salt gsalt at the current counter; bump: increment the counter at the end."""
    bw, bm, tiles, grid = row_cfg(M, W)
    gen = ug is not None
    k_row[grid](x, alpha if deq else x, c if deq else x, g if mode != 2 else x, b if mode != 2 else x, out, state,
                u if dither else x, ug if gen else x, M, W, wg, x.stride(0), out.stride(0), lo, hi, 1.0 / d, nl, zp,
                gsalt, LN_EPS, BLOCK_M=bm, BLOCK_W=bw, TILES=tiles, DEQ=deq, MODE=mode, DITHER=dither, GEN=gen,
                GEN_BLOCK=gen_block(wg, grid[0]) if gen else 16, BUMP=bump, SIMD=simd, num_warps=8)


GEMM_DEFAULT = (64, 128, 128, 4, 4)
# (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps); all fit the 99 KB shared memory of GA102 (int8 operands)
GEMM_CONFIGS = [
    (16, 64, 128, 4, 4), (16, 128, 128, 3, 4), (32, 32, 128, 4, 2), (32, 64, 128, 4, 4), (32, 128, 128, 4, 4),
    (64, 64, 64, 5, 4), (64, 64, 128, 4, 4), (64, 128, 64, 4, 4), (64, 128, 128, 4, 4), (64, 256, 64, 4, 4),
    (128, 64, 64, 4, 4), (128, 64, 128, 4, 4), (128, 128, 64, 3, 4), (128, 128, 128, 3, 8), (128, 256, 64, 3, 8),
    (256, 128, 64, 3, 8), (128, 256, 128, 2, 8),
]


def gemm_a(a, w, alpha, c, state, cfgs, cfg=None, ug=None, wg=0, gsalt=0):
    M, K = a.shape
    N = w.shape[0]
    bm, bn, bk, ns, nw = cfg if cfg is not None else cfgs.get((M, N, K), GEMM_DEFAULT)
    y = torch.empty((M, N), device=a.device, dtype=torch.float32)
    P = triton.cdiv(M, bm) * triton.cdiv(N, bn)
    gen = ug is not None
    k_gemm_a[(P,)](a, w, alpha, c, y, state, ug if gen else alpha, M, N, K, wg, gsalt, a.stride(0), w.stride(0),
                   y.stride(0), BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8, GEN=gen,
                   GEN_BLOCK=gen_block(wg, P) if gen else 16, num_stages=ns, num_warps=nw)
    return y


def u_vector(state, W, salt):
    out = torch.empty(W, device=state.device, dtype=torch.float32)
    k_u[(1,)](state, out, W, salt, BLOCK_W=triton.next_power_of_2(W))
    return out


def gemm_shapes(size):
    """(M, N, K) of the int8 GEMMs (layers 1 and 2) of one model size for every row count of the plan."""
    m = model_dims(size)
    W, lat = m["mlp_dim"], m["latent_dim"]
    shapes = []
    for M in (N_PI,) + NS:
        shapes += [(M, W, W), (M, lat, W)]          # dynamics layer 1, head (the reward layer 1 has the same shape)
        if M != N_PI:
            shapes.append((M, 101, W))               # reward head (never called with the policy-rollout batch)
    return shapes


def tune_gemm_a(shapes, log=print):
    """Best tile config per (M, N, K) by triton.testing.do_bench (int-det GEMM only, no dither)."""
    best, table = {}, {}
    for (M, N, K) in shapes:
        a = torch.randint(0, 16, (M, K), dtype=torch.int8, device="cuda")
        w = torch.randint(-127, 128, (N, K), dtype=torch.int8, device="cuda")
        alpha = torch.rand(N, device="cuda")
        c = torch.rand(N, device="cuda")
        st = torch.zeros(2, dtype=torch.int64, device="cuda")
        res = {}
        for cfg in GEMM_CONFIGS:
            if K % cfg[2]:
                continue
            try:
                fn = (lambda cf=cfg: gemm_a(a, w, alpha, c, st, {}, cf))
                fn()
                torch.cuda.synchronize()
                ms = float(triton.testing.do_bench(fn, warmup=10, rep=40))
            except Exception as ex:  # noqa: BLE001  (out of shared memory / registers)
                res[str(list(cfg))] = repr(ex)[:160]
                continue
            res[str(list(cfg))] = ms
            if (M, N, K) not in best or ms < best[(M, N, K)][1]:
                best[(M, N, K)] = (list(cfg), ms)
        table[f"{M},{N},{K}"] = res
        log(f"tune {M},{N},{K}: {best[(M, N, K)]}")
    return best, table


# ---------------------------------------------------------------------------------------------------- int MLPs
def w8_per_channel(W):
    """symmetric per-output-channel int8 RTN (E7): W ~ s[:, None] * Wq."""
    s = W.abs().amax(dim=1).clamp(min=1e-12) / 127.0
    Wq = torch.round(W / s[:, None]).clamp(-128, 127)
    return Wq, s


def pack_w8(lin, lo_in, d_in, zp, pad_to=1):
    """Wq (N, K) int8 row-major (K contiguous), alpha = d_in * s, c = (lo_in + zp d_in) s sum_k Wq + bias (E7)."""
    W = lin.weight.detach().float()
    Wq, s = w8_per_channel(W)
    alpha = d_in * s
    c = (lo_in + zp * d_in) * s * Wq.sum(dim=1) + lin.bias.detach().float()
    Wq = Wq.to(torch.int8)
    n = Wq.shape[0]
    if n % pad_to:
        pad = pad_to - n % pad_to
        Wq = torch.cat([Wq, torch.zeros((pad, Wq.shape[1]), device=Wq.device, dtype=torch.int8)], 0)
        alpha = torch.cat([alpha, torch.zeros(pad, device=alpha.device)])
        c = torch.cat([c, torch.zeros(pad, device=c.device)])
    return Wq.contiguous(), alpha.float().contiguous(), c.float().contiguous(), s.float().contiguous()


class GenIntMLP:
    """layers.mlp(in, [W, W], out) of TD-MPC2 (NormedLinear, NormedLinear, head) on the general integer path."""

    def __init__(self, seq, qr0, qr1, dither, salt, dyn_head, impl, gemm_cfgs=None, k=KBITS, simd=8):
        L0, L1, L2 = seq[0], seq[1], seq[2]
        assert impl in ("a", "b"), impl
        self.k, self.nl, self.zp = int(k), float((1 << int(k)) - 1), float(128 if int(k) == 8 else 0)
        self.dither, self.salt, self.dyn_head, self.impl, self.simd = bool(dither), int(salt), bool(dyn_head), impl, simd
        self.W0, self.b0 = L0.weight.detach().contiguous(), L0.bias.detach().contiguous()
        self.g0, self.be0 = L0.ln.weight.detach().contiguous(), L0.ln.bias.detach().contiguous()
        self.g1, self.be1 = L1.ln.weight.detach().contiguous(), L1.ln.bias.detach().contiguous()
        if self.dyn_head:
            self.g2, self.be2 = L2.ln.weight.detach().contiguous(), L2.ln.bias.detach().contiguous()
        else:
            self.g2 = self.be2 = None
        self.width, self.n_out = L0.out_features, L2.out_features
        self.lo0, self.hi0 = float(qr0[0]), float(qr0[1])
        self.lo1, self.hi1 = float(qr1[0]), float(qr1[1])
        self.d0, self.d1 = (self.hi0 - self.lo0) / self.nl, (self.hi1 - self.lo1) / self.nl
        self.W1q, self.alpha1, self.c1, self.s1 = pack_w8(L1, self.lo0, self.d0, self.zp)
        self.W2q, self.alpha2, self.c2, self.s2 = pack_w8(L2, self.lo1, self.d1, self.zp,
                                                         pad_to=8 if impl == "b" else 1)
        self.state = torch.zeros(2, dtype=torch.int64, device=L0.weight.device)   # [base, forward counter]
        self.ubuf = torch.zeros((2, self.width), dtype=torch.float32, device=L0.weight.device)   # u of layer 0, 1
        self.gemm_cfgs = gemm_cfgs or {}
        self.seq = seq
        self.new_episode(0)

    def new_episode(self, base):
        self.state[0] = int(base)
        self.state[1] = 0
        if self.dither:
            k_u[(1,)](self.state, self.ubuf[0], self.width, self.salt, BLOCK_W=triton.next_power_of_2(self.width))

    def _forward(self, x):
        x2 = x.reshape(-1, x.shape[-1])
        M = x2.shape[0]
        dev, Wd, st, dith = x2.device, self.width, self.state, self.dither
        u0, u1 = self.ubuf[0], self.ubuf[1]
        y0 = F.linear(x2, self.W0, self.b0)
        q0 = torch.empty((M, Wd), device=dev, dtype=torch.int8)
        # layer 0: reads u0 (forward f); draws u1 = Philox(seed(f, 1))
        row_launch(y0, None, None, self.g0, self.be0, q0, st, M, Wd, self.lo0, self.hi0, self.d0, self.nl, self.zp,
                   0, False, dith, u=u0, ug=u1 if dith else None, wg=Wd, gsalt=self.salt + 1)
        q1 = torch.empty((M, Wd), device=dev, dtype=torch.int8)
        head_gen = dict(ug=u0, wg=Wd, gsalt=self.salt) if dith else {}
        if self.impl == "a":
            y1 = gemm_a(q0, self.W1q, self.alpha1, self.c1, st, self.gemm_cfgs)
            # layer 1: reads u1; increments f
            row_launch(y1, None, None, self.g1, self.be1, q1, st, M, Wd, self.lo1, self.hi1, self.d1, self.nl, self.zp,
                       0, False, dith, u=u1, bump=dith)
            if self.dyn_head:
                y2 = gemm_a(q1, self.W2q, self.alpha2, self.c2, st, self.gemm_cfgs)
                out = torch.empty((M, self.n_out), device=dev, dtype=torch.float32)
                # head: draws u0 = Philox(seed(f + 1, 0)) for the next forward
                row_launch(y2, None, None, self.g2, self.be2, out, st, M, self.n_out, 0.0, 1.0, 1.0, 1.0, 0.0, 1,
                           False, False, simd=self.simd, **head_gen)
            else:
                out = gemm_a(q1, self.W2q, self.alpha2, self.c2, st, self.gemm_cfgs, **head_gen)
        else:
            a1 = torch._int_mm(q0, self.W1q.t())
            row_launch(a1, self.alpha1, self.c1, self.g1, self.be1, q1, st, M, Wd, self.lo1, self.hi1, self.d1,
                       self.nl, self.zp, 0, True, dith, u=u1, bump=dith)
            a2 = torch._int_mm(q1, self.W2q.t())
            out = torch.empty((M, self.n_out), device=dev, dtype=torch.float32)
            row_launch(a2, self.alpha2, self.c2, self.g2, self.be2, out, st, M, self.n_out, 0.0, 1.0, 1.0, 1.0, 0.0,
                       1 if self.dyn_head else 2, True, False, simd=self.simd, **head_gen)
        return q0, q1, out

    def __call__(self, x):
        shp = x.shape[:-1]
        return self._forward(x)[2].reshape(*shp, self.n_out)


# ---------------------------------------------------------------------------------------------------- WM paths
class WMfp32:
    """dynamics / reward calls of WorldModel.next / .reward (single task: cat([z, a]) -> MLP)."""
    name = "fp32"

    def __init__(self, model):
        self.m = model

    def _call(self, seq, z, a):
        return seq(torch.cat([z, a], dim=-1))

    def next(self, z, a):
        return self._call(self.m._dynamics, z, a)

    def reward(self, z, a):
        return self._call(self.m._reward, z, a)

    def new_episode(self, base):
        pass


class WMtf32(WMfp32):
    name = "tf32"

    def _call(self, seq, z, a):
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            return seq(torch.cat([z, a], dim=-1))
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev


class WMbf16(WMfp32):
    name = "bf16"

    def _call(self, seq, z, a):
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            out = seq(torch.cat([z, a], dim=-1))
        return out.float()


class WMint(WMfp32):
    """int det / shiter on the general path (impl 'a' or 'b'), or E7's fused kernels (impl 'e7', 5M only)."""

    def __init__(self, model, qr, dither, impl, gemm_cfgs=None):
        super().__init__(model)
        self.name = f"int-{'shiter' if dither else 'det'}"
        self.impl = impl
        if impl == "e7":
            import e7dev
            self.dyn = e7dev.E7IntMLP(model._dynamics, qr["dyn0"], qr["dyn1"], dither, SALT["dyn"], True)
            self.rew = e7dev.E7IntMLP(model._reward, qr["rew0"], qr["rew1"], dither, SALT["rew"], False)
        else:
            self.dyn = GenIntMLP(model._dynamics, qr["dyn0"], qr["dyn1"], dither, SALT["dyn"], True, impl, gemm_cfgs)
            self.rew = GenIntMLP(model._reward, qr["rew0"], qr["rew1"], dither, SALT["rew"], False, impl, gemm_cfgs)

    def next(self, z, a):
        return self.dyn(torch.cat([z, a], dim=-1))

    def reward(self, z, a):
        return self.rew(torch.cat([z, a], dim=-1))

    def new_episode(self, base):
        self.dyn.new_episode(base)
        self.rew.new_episode(base)


def make_wm(model, prec, qr=None, impl=None, gemm_cfgs=None):
    if prec == "fp32":
        return WMfp32(model)
    if prec == "tf32":
        return WMtf32(model)
    if prec == "bf16":
        return WMbf16(model)
    if prec in ("int-det", "int-shiter", "e7-det", "e7-shiter"):
        return WMint(model, qr, prec.endswith("shiter"), "e7" if prec.startswith("e7") else impl, gemm_cfgs)
    raise ValueError(prec)


def install(model, wm):
    """patch WorldModel.next / .reward of an agent's model (eager-step); fp32 keeps the original methods."""
    if isinstance(wm, WMfp32) and type(wm) is WMfp32:
        return
    model.next = lambda z, a, task: wm.next(z, a)
    model.reward = lambda z, a, task: wm.reward(z, a)


def restore(model):
    for name in ("next", "reward"):
        if name in model.__dict__:
            delattr(model, name)


# ---------------------------------------------------------------------------------------------------- models
def model_dims(size):
    from common import MODEL_SIZE
    d = dict(MODEL_SIZE[size])
    d.setdefault("num_q", 5)
    return d


def make_cfg(size, num_samples=512):
    cfg = OmegaConf.load(os.path.join(TDMPC2_DIR, "config.yaml"))
    cfg.task = "mw-push"
    cfg.model_size = size
    cfg.checkpoint = "none"
    cfg.compile = False
    cfg.mpc = True
    cfg.seed = 0
    cfg.obs = "state"
    cfg.enable_wandb = False
    cfg.save_video = False
    cfg.data_dir = "none"
    cfg.wandb_project = "none"
    cfg.wandb_entity = "none"
    cfg = parse_cfg(cfg)
    cfg.obs_shape = {"state": (OBS_DIM,)}
    cfg.action_dim = ACT_DIM
    cfg.episode_length = 100      # only sets the discount factor (no effect on latency)
    cfg.seed_steps = 1000
    cfg.num_samples = int(num_samples)
    return cfg


def build_agent(size, num_samples=512):
    cfg = make_cfg(size, num_samples)
    torch.manual_seed(0)
    agent = TDMPC2(cfg)
    agent.model.eval()
    return agent, cfg


@torch.no_grad()
def calib_ranges(model, n=2048, seed=0):
    """[min, 99.99th percentile] of the Mish outputs of dyn0, dyn1, rew0, rew1 on random inputs:
    z = encoder(obs ~ N(0, 1)), a ~ U[-1, 1] (fixed generator seed)."""
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    obs = torch.randn(n, OBS_DIM, device="cuda", generator=g)
    a = torch.rand(n, ACT_DIM, device="cuda", generator=g) * 2 - 1
    x = torch.cat([model.encode(obs, None), a], dim=-1)
    out = {}
    for name, seq in (("dyn", model._dynamics), ("rew", model._reward)):
        h0 = seq[0](x)
        h1 = seq[1](h0)
        for i, h in enumerate((h0, h1)):
            hf = h.flatten().float()
            out[f"{name}{i}"] = (float(hf.min()), float(torch.quantile(hf, 0.9999)))
    return out


@torch.no_grad()
def wm_inputs(model, N, seed=0):
    """static rollout inputs: z0 = encoder(one random obs) repeated (as TDMPC2._plan), random actions in [-1, 1]."""
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    obs = torch.randn(1, OBS_DIM, device="cuda", generator=g)
    z = model.encode(obs, None)
    Z0 = z.repeat(N, 1).contiguous()
    A = (torch.rand(ITERS, HORIZON, N, ACT_DIM, device="cuda", generator=g) * 2 - 1).contiguous()
    Zpi = z.repeat(N_PI, 1).contiguous()
    Api = (torch.rand(HORIZON - 1, N_PI, ACT_DIM, device="cuda", generator=g) * 2 - 1).contiguous()
    return Z0, A, Zpi, Api


@torch.no_grad()
def wm_rollout(wm, Z0, A, Zpi, Api):
    """the world-model part of one control step: policy-prior rollout (batch 24, dynamics x 2), then per MPPI iteration
    (6) and horizon step (3): reward(z, a_t), z = next(z, a_t) at batch N (order of TDMPC2._estimate_value)."""
    z = Zpi
    for t in range(HORIZON - 1):
        z = wm.next(z, Api[t])
    r = None
    for it in range(ITERS):
        z = Z0
        for t in range(HORIZON):
            r = wm.reward(z, A[it, t])
            z = wm.next(z, A[it, t])
    return z, r


class GraphRunner:
    """one CUDA graph of fn (warmup on a side stream, then capture into a private memory pool)."""

    def __init__(self, fn, warmup=3):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            self.out = fn()
        torch.cuda.synchronize()

    def __call__(self):
        self.g.replay()


class StepRunner:
    """eager-step: agent.act(obs, t0=(first call of the block), eval_mode=True) on a fixed observation stream."""

    def __init__(self, agent, obs):
        self.agent, self.obs, self.s = agent, obs, 0

    def reset(self):
        self.s = 0

    def __call__(self):
        self.agent.act(self.obs[self.s % len(self.obs)], t0=(self.s == 0), eval_mode=True)
        self.s += 1


def obs_stream(n=200, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return [o for o in torch.randn(n, OBS_DIM, generator=g)]


# ---------------------------------------------------------------------------------------------------- timing / state
class EventPool:
    def __init__(self, n):
        self.ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n)]

    def ensure(self, n):
        while len(self.ev) < n:
            self.ev.append((torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)))


def time_block(fn, n_warm, n_rep, pool, mid=None):
    """n_warm untimed calls, then n_rep calls each timed with a CUDA event pair and followed by a device sync.
    mid(): called once after rep n_rep // 2 (outside the timed window). Returns per-rep milliseconds."""
    for _ in range(n_warm):
        fn()
        torch.cuda.synchronize()
    pool.ensure(n_rep)
    ev = pool.ev
    half = n_rep // 2
    mid_out = None
    for i in range(n_rep):
        e0, e1 = ev[i]
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        if mid is not None and i == half:
            mid_out = mid()
    return [ev[i][0].elapsed_time(ev[i][1]) for i in range(n_rep)], mid_out


class GpuState:
    def __init__(self):
        self.h = None
        try:
            import pynvml as nv
            nv.nvmlInit()
            self.nv = nv
            self.h = nv.nvmlDeviceGetHandleByIndex(int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]))
        except Exception as ex:  # noqa: BLE001
            self.err = repr(ex)

    def snap(self):
        if self.h is None:
            return {"err": getattr(self, "err", "no nvml")}
        nv, h = self.nv, self.h
        d = {}
        try:
            d["sm_mhz"] = nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)
            d["mem_mhz"] = nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM)
            d["temp_c"] = nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)
            d["power_w"] = nv.nvmlDeviceGetPowerUsage(h) / 1000.0
            try:
                d["throttle"] = int(nv.nvmlDeviceGetCurrentClocksEventReasons(h))
            except Exception:  # noqa: BLE001
                d["throttle"] = int(nv.nvmlDeviceGetCurrentClocksThrottleReasons(h))
            try:
                d["procs"] = len(nv.nvmlDeviceGetComputeRunningProcesses(h))
            except Exception:  # noqa: BLE001
                d["procs"] = -1
        except Exception as ex:  # noqa: BLE001
            d["err"] = repr(ex)
        return d


def cpu_times():
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    return sum(v), v[3] + v[4]


def summarize(ms):
    x = np.asarray(ms, float)
    return {"median_ms": float(np.median(x)), "mean_ms": float(x.mean()), "min_ms": float(x.min()),
            "p10_ms": float(np.percentile(x, 10)), "p90_ms": float(np.percentile(x, 90)), "max_ms": float(x.max()),
            "n": int(len(x))}


# ---------------------------------------------------------------------------------------------------- conditions
def cond_key(mode, size, N, prec, impl):
    return f"{mode}|{size}|{N}|{prec}|{impl}"


def units(selection):
    """measurement units of one round: single blocks (fp32, tf32, bf16) and det/shiter pairs (int on the selected
    GEMM path; E7 fused kernels at 5M). modes: graph (graph-WM, primary), eager (eager-WM), step (eager-step, N 512)."""
    out = []
    for mode in ("graph", "eager", "step"):
        for size in SIZES:
            for N in (NS if mode != "step" else (512,)):
                for p in FLOAT_PRECS:
                    out.append({"kind": "single", "mode": mode, "size": size, "N": N, "prec": p, "impl": "-"})
                out.append({"kind": "pair", "mode": mode, "size": size, "N": N, "prec": "int",
                            "impl": selection[str(size)]})
                if size == 5:
                    out.append({"kind": "pair", "mode": mode, "size": size, "N": N, "prec": "e7", "impl": "e7"})
    return out


def unit_conds(u):
    if u["kind"] == "single":
        return [(u["mode"], u["size"], u["N"], u["prec"], u["impl"])]
    return [(u["mode"], u["size"], u["N"], f"{u['prec']}-{v}", u["impl"]) for v in ("det", "shiter")]


def now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
