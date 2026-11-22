"""DESIGN_tmlr.md 0.7-2: unit checks of the new variants (det+BC, det+shift-fixed, sh-iter+b, det+noise).

A. kernel level (no model; stand-in module at the GELU hook site, float32 inputs uniform on the grid interior, 2 groups x
   64 rows x 49 tokens x 256 units, per-tensor grid lo -0.17, hi 2.0, k 3):
   - det+BC: output = det - b exactly
   - det+shift-fixed: table constant across the CEM iterations of a plan, different per horizon step, per episode and per
     plan; rows of a group share it; elementwise error mean and variance equal det+shift's (relative difference < 2%,
     mean compared in units of the error std)
   - sh-iter+b: per-unit mean error equals b (|mean - b| < 5 standard errors for every unit), variance of (error - b)
     equals sh-iter's (< 2%); fresh at every forward call
   - det+noise: the det code is bitwise det's, the added noise has variance D^2/12 (< 2%) and mean 0; fresh every call
   - no new variant touches the default CPU / CUDA generators
B. planner random stream (real model, smoke split episode 0, CEM with diagnostics): for none and every new variant the
   iteration-0 population is identical (run_cem's assert), the per-episode CPU generator ends in the same state as for
   none, and the default CPU / CUDA generator states and numpy / random states are unchanged by the run
C. det+BC residual (bias files of this pod's calibration): mean over units of |residual| after the third sweep < 1% of
   the grid step for every layer
D. descriptive: elementwise error moments of every variant on real GELU outputs (one rollout of the smoke episode's
   iteration-0 candidates)
usage: python check_variants.py <task> [--out JSON]"""
import os, sys, json, glob, time, random, argparse
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wmlib as L
import qlib as QL
import runner as RN
import podplan as PP

TOL = 0.02


class FakeMod(torch.nn.Module):
    def forward(self, x):
        return x


class FakeModel:
    def __init__(self, n=6):
        layers = []
        for _ in range(n):
            ff = torch.nn.Module(); ff.net = [None, None, FakeMod()]
            layers.append([None, ff])
        self.predictor = type('P', (), {})(); self.predictor.transformer = type('T', (), {})()
        self.predictor.transformer.layers = layers


def part_a():
    res = {}
    torch.manual_seed(0)
    lo, hi, k = -0.17, 2.0, 3
    nl = (1 << k) - 1; d = (hi - lo) / nl
    G, N, Lt, C = 2, 64, 49, 256
    x = lo + (hi - lo) * torch.rand(G * N, Lt, C, device='cuda')          # float32, interior of the grid
    m = FakeModel()
    Q = QL.QuantizerT(m, use_compile=True)
    Q.set_ranges([(lo, hi)] * 6)
    mod = m.predictor.transformer.layers[0][1].net[2]
    b = ((torch.rand(C, device='cuda') - 0.5) * 0.4 * d)
    Q.set_bias([b.tolist()] * 6, tag='test')
    cpu0, cuda0 = torch.get_rng_state(), torch.cuda.get_rng_state()

    def call(var, ep=(11, 12), salt='test', it=0, h=0, plan_idx=0, begin=True):
        if begin:
            Q.configure(var, k); Q.begin_plan(list(ep), salt, plan_idx)
        Q.it, Q.h = it, h
        return mod(x)

    # det+BC
    det = call('det'); dbc = call('detbc')
    res['detbc_max_abs_diff_from_det_minus_b'] = float(((det - dbc) - b).abs().max())   # fp32 rounding only
    res['detbc_exact'] = bool(res['detbc_max_abs_diff_from_det_minus_b'] < 1e-5)
    # det+shift-fixed vs det+shift
    f0 = call('detshiftfix', it=0)
    same_it = all(torch.equal(f0, call('detshiftfix', it=i, begin=False)) for i in range(1, 6))
    diff_h = not torch.equal(f0, call('detshiftfix', it=0, h=1))
    diff_plan = not torch.equal(f0, call('detshiftfix', it=0, plan_idx=1))
    diff_ep = not torch.equal(f0.reshape(G, N, Lt, C)[0], call('detshiftfix', ep=(13, 12)).reshape(G, N, Lt, C)[0])
    xr = x.reshape(G, N, Lt, C)[:, :1].expand(G, N, Lt, C).reshape(G * N, Lt, C)
    Q.configure('detshiftfix', k); Q.begin_plan([11, 12], 'test'); Q.it, Q.h = 0, 0
    yr = mod(xr).reshape(G, N, Lt, C)
    shared = bool((yr == yr[:, :1]).all())
    ef = torch.cat([((call('detshiftfix', ep=(100 + 2 * j, 101 + 2 * j)) - x) / d).reshape(-1) for j in range(20)])
    es = torch.cat([((call('detshift', it=j) - x) / d).reshape(-1) for j in range(20)])
    mf, vf, ms, vs = float(ef.mean()), float(ef.var()), float(es.mean()), float(es.var())
    res['detshiftfix'] = dict(constant_within_plan=same_it, differs_by_step=diff_h, differs_by_plan=diff_plan,
                              differs_by_episode=diff_ep, shared_by_rows=shared, mean=mf, var=vf, detshift_mean=ms,
                              detshift_var=vs, var_rel_diff=abs(vf / vs - 1), mean_diff_over_std=abs(mf - ms) / np.sqrt(vs))
    res['detshiftfix']['passed'] = bool(same_it and diff_h and diff_plan and diff_ep and shared and
                                        res['detshiftfix']['var_rel_diff'] < TOL and res['detshiftfix']['mean_diff_over_std'] < TOL)
    # sh-iter+b vs sh-iter
    eb = torch.stack([((call('shiterb', it=j) - x)) for j in range(40)])            # (40, B, Lt, C), activation units
    eh = torch.stack([((call('shiter', it=j) - x)) for j in range(40)])
    gm = eb.reshape(40, G, N, Lt, C).mean(2)                # rows of a group share u: one value per (call, group, token)
    per_unit = gm.mean((0, 1, 2)); se = gm.std((0, 1, 2)) / np.sqrt(gm[..., 0].numel())
    z = ((per_unit - b).abs() / se)
    vb, vh = float((eb - b).var()), float(eh.var())
    fresh_b = not torch.equal(eb[0], eb[1])
    res['shiterb'] = dict(max_z_mean_minus_b=float(z.max()), var_minus_b=vb, shiter_var=vh, var_rel_diff=abs(vb / vh - 1),
                          fresh=fresh_b)
    res['shiterb']['passed'] = bool(float(z.max()) < 5 and res['shiterb']['var_rel_diff'] < TOL and fresh_b)
    del eb, eh
    # det+noise
    qd = torch.round((call('det') - lo) / d)                  # det code from the compiled det kernel
    ns, codes_ok, fresh_n = [], True, None
    for j in range(20):
        y = call('detnoise', it=j)
        Q.configure('detnoise', k); Q.begin_plan([11, 12], 'test'); Q.it, Q.h = j, 0
        u = Q._shared_u(0, Lt, C)                                                    # the call's table (layer 0)
        n = (u - 0.5).expand(G, N, Lt, C).reshape(G * N, Lt, C)
        codes_ok &= bool(torch.equal(torch.round((y - lo) / d - n), qd))
        ns.append(((y - (lo + d * qd)) / d).reshape(-1))
        if j == 1:
            fresh_n = not torch.equal(ns[0], ns[1])
    nn_ = torch.cat(ns)
    res['detnoise'] = dict(det_code_bitwise=codes_ok, noise_var=float(nn_.var()), target=1 / 12,
                           var_rel_diff=abs(float(nn_.var()) * 12 - 1), noise_mean=float(nn_.mean()), fresh=fresh_n)
    res['detnoise']['passed'] = bool(codes_ok and res['detnoise']['var_rel_diff'] < TOL and abs(float(nn_.mean())) < 0.01
                                     and fresh_n)
    res['global_rng_untouched'] = bool(torch.equal(cpu0, torch.get_rng_state()) and torch.equal(cuda0, torch.cuda.get_rng_state()))
    res['detbc_passed'] = bool(res['detbc_exact'])
    Q.remove()
    return res


def part_b(task, bias_id):
    torch.backends.cuda.matmul.allow_tf32 = True
    L.patch_attention(True)
    ctx = RN.TaskCtx(task, True, 'check')
    m, pre, Q = ctx.m, ctx.pre, ctx.Q
    pairs = ctx.get_pairs('smoke')
    ctx.set_ranges('main')
    k = PP.K1[task]
    out = {}
    ref_gen = None
    for v in ('none',) + QL.TMLR_QVARS:
        if v in QL.BIAS_VARS:
            ctx.set_bias(bias_id, 'main', k)
        E = PP.BATCH_E[task]
        ee = list(range(E))
        inputs = [QL.episode_inputs(m, pre, pairs, e) for e in ee]
        gens = [torch.Generator().manual_seed(QL.plan_seed('smoke', v, e)) for e in ee]
        s_cpu, s_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
        s_np, s_py = np.random.get_state()[1].copy(), random.getstate()
        o = QL.run_cem(m, Q, inputs, gens, ctx.alpha, ctx.opt_steps, ee, v, k, 'smoke', diag=True)
        g_end = [g.get_state() for g in gens]
        if ref_gen is None:
            ref_gen = g_end
        out[v] = dict(it0_population_identical_to_preview=True,
                      gen_end_equal_none=all(torch.equal(a, b) for a, b in zip(g_end, ref_gen)),
                      default_cpu_unchanged=bool(torch.equal(s_cpu, torch.get_rng_state())),
                      default_cuda_unchanged=bool(torch.equal(s_cuda, torch.cuda.get_rng_state())),
                      numpy_unchanged=bool(np.array_equal(s_np, np.random.get_state()[1])),
                      random_unchanged=random.getstate() == s_py, best_last=float(o[0]['best'][-1]))
        out[v]['it0_actions'] = o[0]['pops'][0]['actions']
    a0 = out['none'].pop('it0_actions')
    for v in QL.TMLR_QVARS:
        out[v]['it0_population_identical_to_none'] = bool(np.array_equal(out[v].pop('it0_actions'), a0))
    out['passed'] = all(all(x[kk] for kk in ('gen_end_equal_none', 'default_cpu_unchanged', 'default_cuda_unchanged',
                                              'numpy_unchanged', 'random_unchanged'))
                        and x.get('it0_population_identical_to_none', True) for x in out.values() if isinstance(x, dict))
    return out, ctx


def part_c(task):
    rows = []
    for f in sorted(glob.glob(f'{RN.RANGES}/bias_*_{task}.json')):
        b = json.load(open(f))
        if '.' in os.path.basename(f)[:-5]:               # archived vectors of an earlier rule (bias_x.design.json)
            continue
        rows.append(dict(file=os.path.basename(f), ranges=b['ranges'], k=b['k'], rule=b.get('rule_name'),
                         final=b['final_mean_abs_resid_over_delta'], passed=b['check_resid_lt_1pct'],
                         converged=b.get('converged'), stage_eta=b.get('stage_eta'), updates=b.get('updates')))
    return dict(rows=rows, passed=bool(rows) and all(r['passed'] for r in rows))


@torch.no_grad()
def part_d(ctx, bias_id):
    task, m, Q = ctx.task, ctx.m, ctx.Q
    k = PP.K1[task]
    pairs = ctx.get_pairs('smoke')
    inp = QL.episode_inputs(m, ctx.pre, pairs, 0)
    A = m.action_encoder.patch_embed.in_channels
    a0 = torch.randn(L.N_SAMPLES, L.GOAL_H, A, generator=torch.Generator().manual_seed(QL.plan_seed('smoke', 'none', 0)))
    a0[0] = 0.0
    ctx.set_ranges('main')
    rng = RN.load_ranges(task, 'main')
    nl = (1 << k) - 1
    delta = [(hi - lo) / nl for lo, hi in rng]
    mom = {}

    def hook(li):
        def h(module, inp_, out):
            if cur['v'] is None:
                return None
            st = mom.setdefault(cur['v'], [0.0, 0.0, 0])
            for s0 in range(0, out.shape[0], 50):
                xx = F.gelu(inp_[0][s0:s0 + 50]).float(); yy = out[s0:s0 + 50].float()
                e = (yy - xx) / delta[li]
                st[0] += float(e.sum()); st[1] += float((e * e).sum()); st[2] += e.numel()
            return None
        return h
    cur = {'v': None}
    hs = [mod.register_forward_hook(hook(li)) for li, mod in enumerate(Q.mods)]
    try:
        for v in ('det', 'detbc', 'shiter', 'shiterb', 'detshift', 'detshiftfix', 'shstep', 'detnoise'):
            if v in QL.BIAS_VARS:
                ctx.set_bias(bias_id, 'main', k)
            Q.configure(v, k); Q.begin_plan([0], 'smoke')
            cur['v'] = v
            for it in range(3):
                Q.it = it
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    QL.rollout_q(m, Q, inp[0], inp[1], a0[None].cuda())
            cur['v'] = None
            Q.configure('none')
    finally:
        for h in hs:
            h.remove()
    out = {v: dict(mean=s / n, var=s2 / n - (s / n) ** 2, n=n) for v, (s, s2, n) in mom.items()}
    out['detshiftfix_vs_detshift_var_rel_diff'] = abs(out['detshiftfix']['var'] / out['detshift']['var'] - 1)
    out['detshiftfix_vs_detshift_mean_diff_over_std'] = abs(out['detshiftfix']['mean'] - out['detshift']['mean']) / np.sqrt(out['detshift']['var'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task'); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    t0 = time.time()
    res = dict(task=a.task, gpu=torch.cuda.get_device_name(0), time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    res['A_kernel'] = part_a()
    bias_id = f'main_k{PP.K1[a.task]}'
    res['B_planner_rng'], ctx = part_b(a.task, bias_id)
    res['C_detbc_resid'] = part_c(a.task)
    res['D_real_activations'] = part_d(ctx, bias_id)
    A_ = res['A_kernel']
    res['passed'] = bool(A_['detbc_passed'] and A_['detshiftfix']['passed'] and A_['shiterb']['passed'] and
                         A_['detnoise']['passed'] and A_['global_rng_untouched'] and res['B_planner_rng']['passed'] and
                         res['C_detbc_resid']['passed'])
    res['secs'] = time.time() - t0
    out = a.out or f'/workspace/atm/out/checks/check2_variants_{a.task}.json'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    L.jdump(res, out)
    print('CHECK2', a.task, 'PASSED' if res['passed'] else 'FAILED', json.dumps(dict(
        detbc=A_['detbc_passed'], detshiftfix=A_['detshiftfix']['passed'], shiterb=A_['shiterb']['passed'],
        detnoise=A_['detnoise']['passed'], rng=A_['global_rng_untouched'], planner=res['B_planner_rng']['passed'],
        resid=res['C_detbc_resid']['passed']), default=str))


if __name__ == '__main__':
    main()
