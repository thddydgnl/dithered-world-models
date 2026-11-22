"""Calibration artifacts for the DINO-WM part of paper A's TMLR extension (DESIGN_tmlr.md v1.0 sections 0.6, 2.2).

All functions take a runner.TaskCtx (model, preprocessor, env, QuantizerT) and write small JSON files into the ranges
directory runner.RANGES (pod: /workspace/atm/out/calib):
  record_calib(ctx)        256 train-split calibration windows (window rng seed 11, rendered by the simulator, the E5
                           `pc` data): GELU outputs of the 6 predictor MLPs via one predictor call per window (E5
                           ptq.record computation). Writes
                             ranges_pc_<task>.json       per-channel [min_j, 99.99th percentile_j] (E5 `pc` definition)
                             pctgrid_<task>.json         per-tensor {p99, p99.9, p99.99, p99.999, max} and min (RC grid)
                             ptcheck_<task>.json         per-tensor [min, 99.99%] of the first 64 windows (= E0 'main'
                                                         procedure) and of all 256, against the stored E0 ranges
  compute_bias(ctx, rid, k)  b vector for det+BC / sh-iter+b (DESIGN 0.6): 5-step rollouts of candidate action
                           sequences from the calibration windows' first frames, det-quantized model corrected by the
                           current b vs the unquantized model (same starts, same actions); per layer and unit, the mean
                           over all rows, tokens and steps of (corrected quantized - unquantized) GELU output is added
                           to b; three sweeps, then a fourth measurement-only pass gives the residual after the third
                           sweep (check 0.7-2: mean over units of |residual| < 1% of the grid step). bias_<rid>_k<k>_<task>.json
  select_rc(ctx, k)        RC ranges (DESIGN 2.2): per-layer upper bound from {p99, p99.9, p99.99, p99.999, max}
                           (lo = min), two coordinate-descent sweeps over the 6 layers minimizing the held pairs'
                           planning-time 5-step latent relative error under det at k (E1 definition: iteration-0
                           candidates, 300 N(0,1) sequences, planner seed 1000 + e, row 0 = 0). ranges_rc_k<k>_<task>.json
Candidate sequences for b: per window j, 32 N(0, 1) sequences (normalized actions, horizon 5) from a CPU generator
seeded by qlib.seed_of('bcal', task, j), row 0 = 0 (the CEM iteration-0 distribution with its mu row).
"""
import os, sys, json, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wmlib as L
import qlib as QL
import runner as RN

NWIN, WIN_SEED, NWIN_E0 = 256, 11, 64
B_NCAND, B_GROUP = 32, 8                  # candidates per calibration start, starts per rollout call (256 rows)
B_SWEEPS = 3
PCT = (('p99', 0.99), ('p99.9', 0.999), ('p99.99', 0.9999), ('p99.999', 0.99999), ('max', None))
RC_INIT = 'p99.99'
RC_SWEEPS = 2


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _kq(v, q):
    """q-quantile by top-k, the E0/E5 formula (calib.finish_record, ptq.kth_quantile)."""
    k = max(1, int(round((1 - q) * v.numel())))
    return float(torch.topk(v, k).values[-1])


# ------------------------------------------------------------------------------------------------ calibration windows
_WIN = {}


def windows(ctx):
    """The 256 calibration windows (num_hist frames each, train split, seed 11), rendered once per process."""
    if ctx.task not in _WIN:
        t0 = time.time()
        _WIN[ctx.task] = QL.dataset_windows(ctx.T, ctx.env, 'train', NWIN, seed=WIN_SEED, n_frames=ctx.m.num_hist)
        print(f'{now()} calib windows {ctx.task}: {_WIN[ctx.task]["visual"].shape} in {time.time() - t0:.0f}s', flush=True)
    return _WIN[ctx.task]


@torch.no_grad()
def record_acts(ctx, W, B=16):
    """GELU outputs of the 6 predictor MLPs, one predictor call per window (num_hist frames, dataset actions), as
    ptq.record: float16 on the CPU, rows ordered (window, token). Returns list of (rows, C) tensors and rows/window."""
    m, pre, nh = ctx.m, ctx.pre, ctx.m.num_hist
    acts = [[] for _ in ctx.Q.mods]
    hs = [mod.register_forward_hook(lambda mo, i, o, li=li: acts[li].append(o.detach().reshape(-1, o.shape[-1]).half().cpu()))
          for li, mod in enumerate(ctx.Q.mods)]
    ctx.Q.configure('none')
    try:
        for s in range(0, NWIN, B):
            idx = np.arange(s, min(s + B, NWIN))
            zv, zp = QL.encode_windows(m, pre, W, idx)
            QL.window_predict(m, zv, zp, torch.as_tensor(W['act'][idx, :nh]).cuda())
    finally:
        for h in hs:
            h.remove()
    A = [torch.cat(a) for a in acts]
    assert all(a.shape[0] % NWIN == 0 for a in A)
    return A, A[0].shape[0] // NWIN


def record_calib(ctx):
    task, rdir = ctx.task, RN.RANGES
    os.makedirs(rdir, exist_ok=True)
    t0 = time.time()
    W = windows(ctx)
    A, rpw = record_acts(ctx, W)
    base = dict(task=task, nwin=NWIN, window_seed=WIN_SEED, rows_per_window=rpw, gpu=torch.cuda.get_device_name(0),
                time=now(), traj=[list(x) for x in W['traj'][:8]])
    # per-channel ranges: E5 ptq.ranges_perchannel (q = 0.9999), guarded only if a unit has hi <= lo
    pc, guarded = [], []
    for li, a in enumerate(A):
        a = a.float()
        k = max(1, int(round((1 - 0.9999) * a.shape[0])))
        hi = torch.topk(a, k, dim=0).values[-1]
        lo = a.min(0).values
        bad = (hi - lo) <= 0
        if bool(bad.any()):                  # degenerate unit: give it a tiny grid instead of a zero step
            guarded.append(dict(layer=li, units=torch.nonzero(bad).flatten().tolist()))
            hi = torch.where(bad, lo + 1e-6, hi)
        pc.append(dict(lo=lo.tolist(), hi=hi.tolist()))
    L.jdump(dict(base, method='pc', q_hi=0.9999, layers=pc, guarded=guarded,
                 note='E5 ptq.ranges_perchannel on 256 windows (ptq.record computation)'), f'{rdir}/ranges_pc_{task}.json')
    # per-tensor percentile grid (RC) and the PT re-measure check
    grid, check = [], []
    e0 = json.load(open(f'{RN.CODE}/inputs/ranges_main_{task}.json'))['layers']
    for li, a in enumerate(A):
        v = a.float().reshape(-1)
        g = dict(lo=float(v.min()))
        for name, q in PCT:
            g[name] = float(v.max()) if q is None else _kq(v, q)
        grid.append(g)
        v64 = a[:NWIN_E0 * rpw].float().reshape(-1)
        lo64, hi64 = float(v64.min()), _kq(v64, 0.9999)
        check.append(dict(layer=li, e0_lo=e0[li]['lo'], e0_hi=e0[li]['hi'], lo_64=lo64, hi_64=hi64,
                          d_lo_64=lo64 - e0[li]['lo'], d_hi_64=hi64 - e0[li]['hi'],
                          lo_256=g['lo'], hi_256=g['p99.99'], d_hi_256=g['p99.99'] - e0[li]['hi']))
    L.jdump(dict(base, layers=grid), f'{rdir}/pctgrid_{task}.json')
    exact = all(c['d_lo_64'] == 0 and c['d_hi_64'] == 0 for c in check)
    L.jdump(dict(base, layers=check, exact_match_64=exact,
                 note='DESIGN 0.6 check: per-tensor ranges re-measured on the same data (first 64 of the 256 windows = '
                      'E0 calibration windows) vs the stored E0 ranges_main file'), f'{rdir}/ptcheck_{task}.json')
    print(f'{now()} record_calib {task}: rows/window {rpw}, PT re-measure exact {exact}, '
          f'max |d_hi_64| {max(abs(c["d_hi_64"]) for c in check):.3g}, guarded {guarded} ({time.time() - t0:.0f}s)',
          flush=True)
    return dict(exact=exact, guarded=guarded)


# ------------------------------------------------------------------------------------------------ bias vectors
class OutSum:
    """Forward hooks registered AFTER the quantizer's: per layer and unit, the float64 sum of the module output that the
    model uses (quantized and corrected, or unquantized) over rows and tokens, and the element count per unit."""

    def __init__(self, mods):
        self.on = False
        self.s = [torch.zeros(1, dtype=torch.float64, device='cuda') for _ in mods]
        self.n = [0 for _ in mods]
        self.handles = [mod.register_forward_hook(self._hook(li)) for li, mod in enumerate(mods)]

    def reset(self):
        self.s = [torch.zeros(1, dtype=torch.float64, device='cuda') for _ in self.s]
        self.n = [0 for _ in self.n]

    def _hook(self, li):
        def h(module, inp, out):
            if self.on:
                C = out.shape[-1]
                x = out.reshape(-1, C)
                acc = torch.zeros(C, dtype=torch.float64, device=out.device)
                for s in range(0, x.shape[0], 16384):
                    acc += x[s:s + 16384].float().sum(0).double()
                self.s[li] = self.s[li] + acc
                self.n[li] += x.shape[0]
            return None
        return h

    def means(self):
        return [(s / n).float() for s, n in zip(self.s, self.n)]

    def remove(self):
        for h in self.handles:
            h.remove()


_BIN = {}


def bias_inputs(ctx):
    """Start latents (first frame of each calibration window, bf16 autocast) and candidate sequences (CPU, seeded)."""
    if ctx.task not in _BIN:
        W = windows(ctx)
        zv, zp = [], []
        for s in range(0, NWIN, 16):
            idx = np.arange(s, min(s + 16, NWIN))
            v, p = QL.encode_windows(ctx.m, ctx.pre, W, idx)
            zv.append(v[:, :1]); zp.append(p[:, :1])
        A = ctx.m.action_encoder.patch_embed.in_channels
        acts = []
        for j in range(NWIN):
            a = torch.randn(B_NCAND, L.GOAL_H, A, generator=torch.Generator().manual_seed(QL.seed_of('bcal', ctx.task, j)))
            a[0] = 0.0
            acts.append(a)
        _BIN[ctx.task] = (torch.cat(zv), torch.cat(zp), torch.stack(acts))
    return _BIN[ctx.task]


@torch.no_grad()
def _pass(ctx, stats, z0v, z0p, acts):
    stats.reset(); stats.on = True
    try:
        for s in range(0, NWIN, B_GROUP):
            sl = slice(s, min(s + B_GROUP, NWIN))
            g = sl.stop - sl.start
            ctx.Q.begin_plan(list(range(g)), 'bcal')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                QL.rollout_q(ctx.m, ctx.Q, z0v[sl], z0p[sl], acts[sl].cuda())
    finally:
        stats.on = False
    return stats.means(), list(stats.n)


_UNQ = {}
DESIGN_RULE = dict(damping=1.0, sweeps=3, stop_tol=None, name='design')   # DESIGN 0.6: b <- b + m, three sweeps
PAIRED_STARTS = 16                                                          # starts of the paired (propagated) error


class Capture:
    """Paired per-element comparison of the quantized (corrected) and the unquantized GELU outputs along the same
    rollout: mode 'store' keeps the unquantized outputs of every (layer, call), mode 'cmp' accumulates the squared
    difference of the quantized outputs from them (hooks registered after the quantizer's)."""

    def __init__(self, mods):
        self.mode = None
        self.store = [[] for _ in mods]
        self.cnt = [0 for _ in mods]
        self.sq = [0.0 for _ in mods]
        self.n = [0 for _ in mods]
        self.handles = [mod.register_forward_hook(self._hook(li)) for li, mod in enumerate(mods)]

    def _hook(self, li):
        def h(module, inp, out):
            if self.mode == 'store':
                self.store[li].append(out.detach().clone())
            elif self.mode == 'cmp':
                ref = self.store[li][self.cnt[li]]
                self.cnt[li] += 1
                d = out.float() - ref.float()
                self.sq[li] += float((d * d).sum())
                self.n[li] += d.numel()
            return None
        return h

    def new_start(self):
        self.store = [[] for _ in self.store]
        self.cnt = [0 for _ in self.cnt]

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def paired_error(ctx, z0v, z0p, acts, k, delta):
    """Propagated error of the current det+BC (quantizer already configured with its b): per layer the elementwise mean
    squared (corrected quantized - unquantized) GELU output over rows, tokens and steps, in units of Delta^2, and the
    latent prediction errors at horizon steps 1/3/5 (E1 definition) on the first PAIRED_STARTS calibration starts."""
    cap = Capture(ctx.Q.mods)
    rel = {s: [] for s in QL.ERR_STEPS}
    try:
        for j in range(PAIRED_STARTS):
            cap.new_start()
            a = acts[j:j + 1].cuda()
            ctx.Q.configure('none'); cap.mode = 'store'
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, _, rva, rpa = QL.rollout_q(ctx.m, ctx.Q, z0v[j:j + 1], z0p[j:j + 1], a, all_frames=True)
            ctx.Q.configure('detbc', k); ctx.Q.begin_plan([0], 'bcal'); cap.mode = 'cmp'
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, _, qva, qpa = QL.rollout_q(ctx.m, ctx.Q, z0v[j:j + 1], z0p[j:j + 1], a, all_frames=True)
            cap.mode = None
            er, pv = QL._step_errors(qva, qpa, rva, rpa, ctx.alpha), QL._pred_var(rva, rpa, ctx.alpha)
            for s in QL.ERR_STEPS:
                rel[s].append(float(er[s][0]) / float(pv[s][0]))
            del rva, rpa, qva, qpa
    finally:
        cap.remove()
    d2 = [float((x.double() ** 2).mean()) for x in delta]
    return dict(mse_over_delta2=[cap.sq[li] / cap.n[li] / d2[li] for li in range(len(delta))],
                latent_rel={str(s): float(np.mean(v)) for s, v in rel.items()})


def rule_name(rule):
    if rule.get('name'):
        return rule['name']
    return f"damp{rule['damping']:g}_s{rule['sweeps']}" + (f"_tol{rule['stop_tol']:g}" if rule.get('stop_tol') else '')


def stages_of(rule):
    """Staged form of a rule: D2 gives 'stages' [{eta, until_below, max_sweeps}]; the design rule is one stage of
    3 updates with damping 1 and no early stop."""
    if rule.get('stages'):
        return [dict(st) for st in rule['stages']]
    return [dict(eta=float(rule['damping']), until_below=rule.get('stop_tol'), max_sweeps=int(rule['sweeps']))]


def compute_bias(ctx, rid, k, rule=None):
    """b for (ranges rid, k); writes bias_<rid>_k<k>_<task>.json and returns its dict.
    Staged rule (deviation D2, runpod_tmlr/bias_rule_D2.json): stages eta = 1, 1/2, 1/4; each stage starts from b = 0;
    every pass measures the residual m (mean over rows, tokens and steps of corrected quantized - unquantized, per unit)
    and r_l = mean_j |m_j| / Delta_j per layer; the first pass with r_l < until_below in every layer fixes b; otherwise
    b <- b + eta m, at most max_sweeps updates per stage, then the next stage. If no stage converges: the last b of the
    stage with the smallest worst-layer r, flagged bc_unconverged. All layers are updated together; data and the 5-step
    rollouts are those of DESIGN 0.6."""
    rule = dict(rule or DESIGN_RULE)
    task, out_path = ctx.task, RN.bias_path(ctx.task, f'{rid}_k{k}')
    t0 = time.time()
    z0v, z0p, acts = bias_inputs(ctx)
    ctx.set_ranges(rid)
    nl = (1 << int(k)) - 1
    rng = RN.load_ranges(task, rid)
    delta = [((torch.tensor(hi, dtype=torch.float64) - torch.tensor(lo, dtype=torch.float64)) / nl).float().cuda()
             for lo, hi in rng]                                  # scalar (PT/RC) or (C,) (PC)
    per_channel = delta[0].numel() > 1
    stats = OutSum(ctx.Q.mods)
    nL = len(delta)

    def set_b(b, tag):
        ctx.Q.set_bias([x.tolist() for x in b], tag=tag)
        ctx.Q.configure('detbc', k)

    try:
        if task not in _UNQ:
            ctx.Q.configure('none')
            _UNQ[task] = _pass(ctx, stats, z0v, z0p, acts)
        mu0, n0 = _UNQ[task]
        C = mu0[0].numel()
        zero = [torch.zeros(C, dtype=torch.float32, device='cuda') for _ in range(nL)]
        set_b(zero, f'{rid}_k{k}_b0')
        pe0 = paired_error(ctx, z0v, z0p, acts, k, delta)       # b = 0 (= det): propagated error
        stage_res, m0 = [], None
        for si, st in enumerate(stages_of(rule)):
            b = [x.clone() for x in zero]
            passes, upd, conv = [], 0, False
            while True:
                set_b(b, f'{rid}_k{k}_s{si}_u{upd}')
                mq, nq = _pass(ctx, stats, z0v, z0p, acts)
                assert nq == n0, (nq, n0)
                m = [mq[li] - mu0[li] for li in range(nL)]
                if m0 is None:
                    m0 = [x.clone() for x in m]
                r = [(m[li].abs() / delta[li]).mean().item() for li in range(nL)]
                rec = dict(updates=upd, mean_abs_m_over_delta=r,
                           max_abs_m_over_delta=[(m[li].abs() / delta[li]).max().item() for li in range(nL)])
                if per_channel:
                    rec['mean_abs_m_over_mean_delta'] = [(m[li].abs().mean() / delta[li].mean()).item() for li in range(nL)]
                passes.append(rec)
                if st.get('until_below') is not None and max(r) < float(st['until_below']):
                    conv = True
                    break
                if upd >= int(st['max_sweeps']):
                    break
                b = [b[li] + float(st['eta']) * m[li] for li in range(nL)]
                upd += 1
            stage_res.append(dict(eta=float(st['eta']), until_below=st.get('until_below'), max_sweeps=int(st['max_sweeps']),
                                  updates=upd, converged=conv, worst_final=max(passes[-1]['mean_abs_m_over_delta']),
                                  passes=passes, b=b))
            print(f"{now()} bias {task} {rid} k={k} [{rule_name(rule)}] stage eta={st['eta']:g}: "
                  f"{'converged' if conv else 'NOT converged'} after {upd} updates, worst r {stage_res[-1]['worst_final']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if conv:
                break
        conv_st = [s_ for s_ in stage_res if s_['converged']]
        chosen = conv_st[0] if conv_st else min(stage_res, key=lambda s_: s_['worst_final'])
        b = chosen['b']
        set_b(b, f'{rid}_k{k}_final')
        pe1 = paired_error(ctx, z0v, z0p, acts, k, delta)
    finally:
        stats.remove()
        ctx.Q.configure('none')
        ctx.ranges_id = None; ctx.bias_id = None                 # force reload by the runner
    final = chosen['passes'][-1]['mean_abs_m_over_delta']
    out = dict(task=task, ranges=rid, k=int(k), b=[x.tolist() for x in b], rule=rule, rule_name=rule_name(rule),
               converged=bool(chosen['converged']), bc_unconverged=not chosen['converged'], stage_eta=chosen['eta'],
               updates=chosen['updates'], final_mean_abs_resid_over_delta=final,
               final_max_abs_resid_over_delta=chosen['passes'][-1]['max_abs_m_over_delta'],
               check_resid_lt_1pct=bool(max(final) < 0.01),
               stages=[{kk: v for kk, v in s_.items() if kk != 'b'} for s_ in stage_res],
               m0=[x.tolist() for x in m0], b_mean_abs_over_delta=[(b[li].abs() / delta[li]).mean().item() for li in range(nL)],
               m0_mean_abs_over_delta=[(m0[li].abs() / delta[li]).mean().item() for li in range(nL)],
               mse_over_delta2_b0=pe0['mse_over_delta2'], mse_over_delta2_final=pe1['mse_over_delta2'],
               latent_rel_b0=pe0['latent_rel'], latent_rel_final=pe1['latent_rel'],
               n_starts=NWIN, n_cand=B_NCAND, rows=NWIN * B_NCAND, elements_per_unit=n0, window_seed=WIN_SEED,
               paired_starts=PAIRED_STARTS, cand_seed="seed_of('bcal', task, j)", gpu=torch.cuda.get_device_name(0),
               time=now(), secs=time.time() - t0)
    if os.path.exists(out_path):                                 # keep the earlier rule's vector for the record
        old = json.load(open(out_path))
        if old.get('rule_name') != out['rule_name']:
            os.replace(out_path, out_path.replace('.json', f".{old.get('rule_name', 'old')}.json"))
    L.jdump(out, out_path)
    return out


# ------------------------------------------------------------------------------------------------ RC ranges
def held_iter0(ctx, n=20):
    """Held pairs' episode inputs and iteration-0 populations (planner seed 1000 + e, row 0 = 0) as run_cem draws them."""
    pairs = ctx.get_pairs('held')
    A = ctx.m.action_encoder.patch_embed.in_channels
    out = []
    for e in range(n):
        inp = QL.episode_inputs(ctx.m, ctx.pre, pairs, e)
        g = torch.Generator().manual_seed(QL.plan_seed('held', 'none', e))
        a0 = torch.randn(L.N_SAMPLES, L.GOAL_H, A, generator=g)
        a0[0] = 0.0
        out.append((inp, a0))
    return out


@torch.no_grad()
def select_rc(ctx, k):
    task = ctx.task
    t0 = time.time()
    grid = json.load(open(f'{RN.RANGES}/pctgrid_{task}.json'))['layers']
    names = [n for n, _ in PCT]
    H = held_iter0(ctx)
    alpha = ctx.alpha
    ctx.Q.configure('none')
    refs = []
    for inp, a0 in H:                                           # unquantized step-5 predictions and variance
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, _, rva, rpa = QL.rollout_q(ctx.m, ctx.Q, inp[0], inp[1], a0[None].cuda(), all_frames=True)
        pv = QL._pred_var(rva, rpa, alpha)[5]
        refs.append((rva[:, :, 4:5].clone(), rpa[:, :, 4:5].clone(), float(pv[0])))
        del rva, rpa
    cache = {}

    def objective(choice):
        key = tuple(choice)
        if key in cache:
            return cache[key]
        ctx.Q.set_ranges([(grid[li]['lo'], grid[li][c]) for li, c in enumerate(choice)])
        ctx.Q.configure('det', k)
        rel = []
        for (inp, a0), (rv5, rp5, pv) in zip(H, refs):
            ctx.Q.begin_plan([0], 'rc')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, _, qva, qpa = QL.rollout_q(ctx.m, ctx.Q, inp[0], inp[1], a0[None].cuda(), all_frames=True)
            ev = ((qva[:, :, 4].float() - rv5[:, :, 0].float()) ** 2).mean((-1, -2))
            ep = ((qpa[:, :, 4].float() - rp5[:, :, 0].float()) ** 2).mean(-1)
            rel.append(float((ev + alpha * ep).mean(1)[0]) / pv)
            del qva, qpa
        ctx.Q.configure('none')
        cache[key] = float(np.mean(rel))
        return cache[key]

    choice = [RC_INIT] * len(grid)
    log = [dict(step='init', choice=list(choice), obj=objective(choice))]
    for sweep in range(1, RC_SWEEPS + 1):
        for li in range(len(grid)):
            cur = objective(choice)
            vals = {}
            for c in names:
                trial = list(choice); trial[li] = c
                vals[c] = objective(trial)
            best = min(names, key=lambda c: (vals[c], names.index(c)))
            if vals[best] < cur - 1e-12:                         # move only on a strict improvement
                choice[li] = best
            log.append(dict(step=f'sweep{sweep}_layer{li}', candidates=vals, choice=list(choice), obj=objective(choice)))
            print(f'{now()} rc {task} k={k} sweep {sweep} layer {li}: ' +
                  ' '.join(f'{c}:{vals[c]:.5f}' for c in names) + f' -> {choice[li]} (obj {objective(choice):.5f})',
                  flush=True)
    ctx.ranges_id = None; ctx.bias_id = None
    layers = [dict(lo=grid[li]['lo'], hi=grid[li][c], choice=c) for li, c in enumerate(choice)]
    out = dict(task=task, method=f'rc_k{k}', k=int(k), layers=layers, objective=objective(choice),
               objective_init=log[0]['obj'], log=log, n_evals=len(cache), held_pairs=len(H),
               rule='per-layer hi from {p99, p99.9, p99.99, p99.999, max} of the 256 calibration windows (lo = min), '
                    'init p99.99, 2 coordinate-descent sweeps over layers 0-5, moving only on strict improvement, '
                    'objective = mean over the 20 held pairs of the iteration-0 5-step latent relative error under det',
               gpu=torch.cuda.get_device_name(0), time=now(), secs=time.time() - t0)
    L.jdump(out, f'{RN.RANGES}/ranges_rc_k{k}_{task}.json')
    return out
