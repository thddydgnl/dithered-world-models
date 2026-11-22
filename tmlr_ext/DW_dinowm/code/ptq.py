"""E5 (design 6.5) PTQ baselines for the DINO-WM predictor (paper A).

Activation baselines (same site as E1: GELU outputs of the 6 predictor MLPs); each writes a ranges file
results/ranges/ranges_<id>_<task>.json used by any job through job['ranges']:
  p999 / p99999   per-tensor [min, 99.9th / 99.999th percentile]          (E1 'main' = 99.99th, 64 windows)
  pc              per-channel (per unit) [min_j, 99.99th percentile_j]
  mse_k<k>        per-tensor [min, hi*] with hi* minimizing the det (round-to-nearest) quantization MSE at k bits
  smooth          SmoothQuant-style migration (alpha 0.5): s_j = max|x_j|^a / max|W2[:, j]|^(1-a) with W2 the next
                  Linear (net.4, 2048 -> 404); the grid is per-tensor on x/s, i.e. per-channel lo_j = lo_s s_j and
                  step d_j = d_s s_j (activation-only quantization, so folding s into W2 is exact and omitted)
  mixed precision: job['qopts'] = {'last_bits': 8} keeps the last block's MLP hidden at 8 bits (with any ranges)
Calibration data: train-split dataset windows rendered by the simulator, window rng seed 11 (the first 64 windows are
the E1 calibration windows), never the eval pairs.

Weight baselines (activations unquantized; variant names in qlib.WVARS), all 24 predictor Linear layers:
  w8rtn / w4rtn   symmetric per-output-channel round-to-nearest
  w4awq           AWQ-style: input-channel scales s_i = mean|x_i|^a (a from a grid, per layer, minimizing the output
                  MSE on calibration inputs), W_eff = Q(W diag(s)) diag(1/s)
  w4sr<K>         K independent stochastic roundings of W (per episode), copy (CEM iteration mod K) in use
usage: python ptq.py <task> [--nwin 256] [--kbits 3,4] [--alpha 0.5]   (writes ranges files and awq stats)
"""
import os, sys, json, time, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wmlib as L
import qlib as QL

WEIGHT_LAYERS = ('0.to_qkv', '0.to_out.0', '1.net.1', '1.net.4')    # per transformer block


def predictor_linears(predictor):
    out = []
    for bi, blk in enumerate(predictor.transformer.layers):
        out += [(f'{bi}.attn.to_qkv', blk[0].to_qkv), (f'{bi}.attn.to_out', blk[0].to_out[0]),
                (f'{bi}.ff.fc1', blk[1].net[1]), (f'{bi}.ff.fc2', blk[1].net[4])]
    return out


# ------------------------------------------------------------------------------------------------ calibration data
@torch.no_grad()
def record(T, env, m, nwin, seed=11, linear_inputs=False, B=16, max_rows_per_layer=400000):
    """GELU outputs per layer (float16, CPU; at most max_rows_per_layer rows each, evenly subsampled) and, with
    linear_inputs, the mean |input| per input channel of every predictor Linear (for AWQ) plus a subsample of inputs."""
    nh = m.num_hist
    cal = QL.dataset_windows(T, env, 'train', nwin, seed=seed, n_frames=nh)
    mods = [blk[1].net[2] for blk in m.predictor.transformer.layers]
    acts = [[] for _ in mods]
    hs = [mod.register_forward_hook(lambda mo, i, o, li=li: acts[li].append(o.detach().reshape(-1, o.shape[-1]).half().cpu()))
          for li, mod in enumerate(mods)]
    lin = predictor_linears(m.predictor)
    absmean = {n: None for n, _ in lin}; cnt = {n: 0 for n, _ in lin}; xs = {n: [] for n, _ in lin}
    hl = []
    if linear_inputs:
        def mk(n):
            def f(mo, inp):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                s = x.abs().sum(0)
                absmean[n] = s if absmean[n] is None else absmean[n] + s
                cnt[n] += x.shape[0]
                if sum(t.shape[0] for t in xs[n]) < 8192:
                    xs[n].append(x[torch.randperm(x.shape[0], device=x.device)[:1024]].cpu())
            return f
        hl = [mod.register_forward_pre_hook(mk(n)) for n, mod in lin]
    for s in range(0, nwin, B):
        idx = np.arange(s, min(s + B, nwin))
        zv, zp = QL.encode_windows(m, pre_of(T), cal, idx)
        QL.window_predict(m, zv, zp, torch.as_tensor(cal['act'][idx, :nh]).cuda())
    for h in hs + hl:
        h.remove()
    A = []
    for li in range(len(mods)):
        a = torch.cat(acts[li])
        if a.shape[0] > max_rows_per_layer:
            a = a[torch.linspace(0, a.shape[0] - 1, max_rows_per_layer).long()]
        A.append(a)
    stats = None
    if linear_inputs:
        stats = {n: dict(absmean=(absmean[n] / cnt[n]).cpu(), x=torch.cat(xs[n])[:8192]) for n, _ in lin}
    return A, stats, cal['traj'][:8]


def pre_of(T):
    return T['pre']


def kth_quantile(v, q):
    """q-quantile of a 1-d float tensor by top-k (no size limit)."""
    k = max(1, int(round((1 - q) * v.numel())))
    return float(torch.topk(v, k).values[-1])


def ranges_percentile(A, q):
    return [dict(lo=float(a.float().min()), hi=kth_quantile(a.float().reshape(-1), q)) for a in A]


def ranges_perchannel(A, q=0.9999):
    out = []
    for a in A:
        a = a.float()
        n = a.shape[0]
        k = max(1, int(round((1 - q) * n)))
        hi = torch.topk(a, k, dim=0).values[-1]
        out.append(dict(lo=a.min(0).values.tolist(), hi=hi.tolist()))
    return out


def det_mse(v, lo, hi, nl):
    d = (hi - lo) / nl
    y = ((v - lo) / d).clamp(0, nl)
    return float(((lo + d * torch.round(y) - v) ** 2).mean())


def ranges_mse(A, k, n_sub=4_000_000, n_grid=40, seed=0):
    """per-tensor hi minimizing the det quantization MSE (lo = min), searched on a log grid between the 99th
    percentile and the max (subsample of n_sub values per layer)."""
    nl = (1 << k) - 1
    g = torch.Generator().manual_seed(seed)
    out = []
    for a in A:
        v = a.float().reshape(-1)
        if v.numel() > n_sub:
            v = v[torch.randint(0, v.numel(), (n_sub,), generator=g)]
        lo = float(a.float().min()); vmax = float(v.max()); p99 = kth_quantile(v, 0.99)
        grid = np.exp(np.linspace(np.log(max(p99, 1e-3)), np.log(vmax), n_grid))
        errs = [det_mse(v, lo, float(h), nl) for h in grid]
        j = int(np.argmin(errs))
        out.append(dict(lo=lo, hi=float(grid[j]), mse=float(errs[j]), grid_lo=float(grid[0]), grid_hi=float(grid[-1])))
    return out


def ranges_smooth(A, m, alpha=0.5, q=0.9999):
    out = []
    for li, (a, blk) in enumerate(zip(A, m.predictor.transformer.layers)):
        a = a.float()
        W2 = blk[1].net[4].weight.detach().float().cpu()            # (404, 2048)
        amax = a.abs().max(0).values.clamp(min=1e-5)
        wmax = W2.abs().max(0).values.clamp(min=1e-5)
        s = (amax ** alpha) / (wmax ** (1 - alpha))
        xs = a / s
        lo_s = float(xs.min()); hi_s = kth_quantile(xs.reshape(-1), q)
        out.append(dict(lo=(lo_s * s).tolist(), hi=(hi_s * s).tolist(), alpha=alpha, lo_s=lo_s, hi_s=hi_s))
    return out


# ------------------------------------------------------------------------------------------------ weights
def rtn_per_channel(W, bits):
    qmax = (1 << (bits - 1)) - 1
    s = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return (torch.round(W / s).clamp(-qmax - 1, qmax) * s).to(W.dtype)


def sr_per_channel(W, bits, gen):
    qmax = (1 << (bits - 1)) - 1
    s = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    u = torch.rand(W.shape, generator=gen, device=W.device, dtype=torch.float32)
    return (torch.floor(W.float() / s + u).clamp(-qmax - 1, qmax) * s).to(W.dtype)


def awq_weight(W, absmean, X, bits, alphas=tuple(np.linspace(0, 1, 11))):
    """W (out, in); absmean (in,) mean |x_i|; X (n, in) calibration inputs. Returns W_eff and the chosen alpha."""
    Wf = W.float(); Xf = X.float().to(W.device); am = absmean.float().to(W.device).clamp(min=1e-6)
    ref = Xf @ Wf.T
    best = None
    for a in alphas:
        s = am ** float(a)
        Weff = rtn_per_channel(Wf * s[None], bits) / s[None]
        err = float(((Xf @ Weff.T - ref) ** 2).mean())
        if best is None or err < best[0]:
            best = (err, float(a), Weff)
    return best[2].to(W.dtype), best[1]


class WeightQuant:
    """Holds the predictor's fp weights and puts quantized weights in place for the weight variants."""

    def __init__(self, predictor, awq_stats=None):
        self.lin = predictor_linears(predictor)
        self.orig = {n: mod.weight.detach().clone() for n, mod in self.lin}
        self.awq_stats = awq_stats
        self.active, self.copies, self.loaded, self.cache = None, None, None, {}

    def _put(self, ws):
        for n, mod in self.lin:
            mod.weight.data.copy_(ws[n])

    def restore(self):
        if self.active is not None:
            self._put(self.orig)
            self.active, self.loaded = None, None

    def apply(self, var, seed=0):
        """var in qlib.WVARS. SR variants draw K copies from `seed` (per episode)."""
        if var in ('w8rtn', 'w4rtn'):
            bits = 8 if var == 'w8rtn' else 4
            if var not in self.cache:
                self.cache[var] = {n: rtn_per_channel(self.orig[n], bits) for n, _ in self.lin}
            self._put(self.cache[var]); self.active = var
        elif var == 'w4awq':
            assert self.awq_stats is not None, 'w4awq needs awq stats (ptq.py)'
            if var not in self.cache:
                ws, al = {}, {}
                for n, _ in self.lin:
                    st = self.awq_stats[n]
                    ws[n], al[n] = awq_weight(self.orig[n], st['absmean'], st['x'], 4)
                self.cache[var] = ws; self.awq_alpha = al
            self._put(self.cache[var]); self.active = var
        elif var.startswith('w4sr'):
            K = int(var[4:])
            g = torch.Generator(device=self.orig[self.lin[0][0]].device); g.manual_seed(int(seed))
            self.copies = [{n: sr_per_channel(self.orig[n], 4, g) for n, _ in self.lin} for _ in range(K)]
            self._put(self.copies[0]); self.active, self.loaded = var, 0
        else:
            raise ValueError(var)

    def at_iteration(self, it):
        """SR rotation: CEM iteration `it` uses copy it mod K."""
        if self.active is not None and self.active.startswith('w4sr') and len(self.copies) > 1:
            idx = int(it) % len(self.copies)
            if idx != self.loaded:
                self._put(self.copies[idx]); self.loaded = idx


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task'); ap.add_argument('--nwin', type=int, default=256); ap.add_argument('--kbits', default='3,4')
    ap.add_argument('--alpha', type=float, default=0.5); ap.add_argument('--nocompile', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    L.patch_attention(True)
    T = L.load_task(a.task)
    m = T['model']
    env = L.make_env(a.task)
    A, stats, traj = record(T, env, m, a.nwin, linear_inputs=True)
    print(f'recorded {[tuple(x.shape) for x in A]} in {time.time() - t0:.0f}s', flush=True)
    base = dict(task=a.task, nwin=a.nwin, window_seed=11, traj=traj, gpu=torch.cuda.get_device_name(0),
                time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    rdir = f'{L.MAIN}/results/ranges'
    os.makedirs(rdir, exist_ok=True)
    outs = {'p999': ranges_percentile(A, 0.999), 'p99999': ranges_percentile(A, 0.99999), 'p9999w256': ranges_percentile(A, 0.9999),
            'pc': ranges_perchannel(A, 0.9999), 'smooth': ranges_smooth(A, m, a.alpha)}
    for k in [int(x) for x in a.kbits.split(',')]:
        outs[f'mse_k{k}'] = ranges_mse(A, k)
    for rid, layers in outs.items():
        L.jdump(dict(base, method=rid, layers=layers), f'{rdir}/ranges_{rid}_{a.task}.json')
        his = [l['hi'] if isinstance(l['hi'], float) else float(np.median(l['hi'])) for l in layers]
        print(rid, 'hi (per-channel: median)', [round(h, 3) for h in his], flush=True)
    torch.save(stats, f'{rdir}/awq_stats_{a.task}.pt')
    print(f'PTQ_DONE {a.task} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
