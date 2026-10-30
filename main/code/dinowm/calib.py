"""Range calibration and dataset/planning-time prediction errors for the DINO-WM part of paper A (server version of
the pilot calib13.py; same procedure and seeds).
1. Per-layer quantizer ranges from dataset windows of the TRAIN split (rendered by the simulator from dataset states,
   64 windows, window rng seed 11): GELU outputs of the 6 predictor MLPs under bf16 autocast (as in planning);
   lo = min, hi = 99.99th percentile. Never uses the eval start/goal pairs.
2. Prediction errors of every variant and k vs the unquantized model, in planning-cost units (visual MSE +
   alpha * proprio-embedding MSE between quantized and unquantized prediction), also relative to the variance of
   the unquantized predictions (rel_*):
   - ds_err1: one predictor call on dataset windows of the VALID split (num_hist frames, window rng seed 12)
   - ds_err3 / ds_err5: open-loop rollout from the first window frame with the dataset actions (steps 3 and 5)
   - plan_err1/3/5: held-split start frames (first --nplan pairs) with 300 N(0,1) candidate action sequences
     (the CEM iteration-0 distribution, planner seed 1000+e as in the pilot), horizon steps 1/3/5
   Randomized variants (stochastic rounding and detshift): mean over --nseeds dither seeds.
usage: python calib.py <task> [--ranges_id main] [--kbits 3,4,5,6] [--vars det,ind,...] [--nwin 64] [--nseeds 3]
       [--nplan 8] [--tag T] [--nocompile] [--ranges_only] [--redo_ranges]"""
import sys, os, time, json, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wmlib as L
import qlib as QL
from runner import ranges_path


def cost_err(av, ap, bv, bp, alpha):
    """per-sample error in cost units between predictions a and b: visual (b,...,P,C), proprio (b,...,Dp)."""
    return ((av - bv) ** 2).mean((-1, -2)) + alpha * ((ap - bp) ** 2).mean(-1)


def pred_var(v, p, alpha):
    """total variance of the reference predictions across samples in cost units (normalizer for rel errors)."""
    v = v.reshape(-1, *v.shape[-2:]); p = p.reshape(-1, p.shape[-1])
    return float(((v - v.mean(0, keepdim=True)) ** 2).mean() + alpha * ((p - p.mean(0, keepdim=True)) ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task'); ap.add_argument('--kbits', default='3,4,5,6'); ap.add_argument('--ranges_id', default='main')
    ap.add_argument('--nwin', type=int, default=64); ap.add_argument('--nseeds', type=int, default=3)
    ap.add_argument('--nocompile', action='store_true'); ap.add_argument('--ranges_only', action='store_true')
    ap.add_argument('--nplan', type=int, default=8)
    ap.add_argument('--vars', default=','.join(QL.VARIANTS[1:])); ap.add_argument('--tag', default='E0_dinowm')
    ap.add_argument('--redo_ranges', action='store_true')
    a = ap.parse_args()
    task = a.task
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    L.patch_attention(True)
    T = L.load_task(task)
    m, pre = T['model'], T['pre']
    alpha = L.TASKS[task]['alpha']
    env = L.make_env(task)
    nh = m.num_hist
    Q = QL.Quantizer(m, use_compile=not a.nocompile)
    B = 16
    rpath = ranges_path(task, a.ranges_id)
    os.makedirs(os.path.dirname(rpath), exist_ok=True)
    if os.path.exists(rpath) and not a.redo_ranges:
        layers = json.load(open(rpath))['layers']
        print('loaded ranges', rpath, [(round(r['lo'], 4), round(r['hi'], 3)) for r in layers], flush=True)
    else:
        assert a.ranges_id != 'pilot'
        layers = calibrate(a, T, env, m, pre, Q, nh, B, t0, rpath)
    if a.ranges_only:
        return
    run_errors(a, T, env, m, pre, Q, nh, B, t0, alpha, layers)


def calibrate(a, T, env, m, pre, Q, nh, B, t0, rpath):
    task = T['task']
    cal = QL.dataset_windows(T, env, 'train', a.nwin, seed=11, n_frames=nh)
    print(f'calibration windows: {cal["visual"].shape} rendered in {time.time() - t0:.0f}s', flush=True)
    Q.start_record()
    for s in range(0, a.nwin, B):
        idx = np.arange(s, min(s + B, a.nwin))
        zv, zp = QL.encode_windows(m, pre, cal, idx)
        QL.window_predict(m, zv, zp, torch.as_tensor(cal['act'][idx, :nh]).cuda())
    layers = Q.finish_record()
    rng = dict(task=task, source='train-split dataset windows rendered by the simulator (window rng seed 11)',
               nwin=a.nwin, num_hist=nh, q_hi=0.9999, layers=layers, traj=cal['traj'][:8],
               gpu=torch.cuda.get_device_name(0), time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    L.jdump(rng, rpath)
    for li, r in enumerate(layers):
        print(f"layer {li}: lo {r['lo']:.4f} hi(99.99%) {r['hi']:.3f} max {r['max']:.3f} p50 {r['p50']:.4f} "
              f"p99 {r['p99']:.3f} frac<0 {r['frac_below_0']:.3f}", flush=True)
    return layers


def run_errors(a, T, env, m, pre, Q, nh, B, t0, alpha, layers):
    task = T['task']
    Q.set_ranges([(r['lo'], r['hi']) for r in layers])
    kbits = [int(x) for x in a.kbits.split(',')]
    res = dict(task=task, alpha=alpha, kbits=kbits, nseeds=a.nseeds, ranges_id=a.ranges_id, ds={}, plan={},
               gpu=torch.cuda.get_device_name(0))
    # ---------------------------------------------------------------- dataset windows (valid split)
    hel = QL.dataset_windows(T, env, 'valid', a.nwin, seed=12, n_frames=nh)
    Zv, Zp = [], []
    for s in range(0, a.nwin, B):
        idx = np.arange(s, min(s + B, a.nwin))
        zv, zp = QL.encode_windows(m, pre, hel, idx)
        Zv.append(zv); Zp.append(zp)
    Zv, Zp = torch.cat(Zv), torch.cat(Zp)
    Aw = torch.as_tensor(hel['act']).cuda()

    def ds_preds():
        p1v, p1p, pav, pap = [], [], [], []
        for s in range(0, a.nwin, B):
            sl = slice(s, min(s + B, a.nwin))
            v, p = QL.window_predict(m, Zv[sl], Zp[sl], Aw[sl, :nh])
            p1v.append(v); p1p.append(p)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, _, fva, fpa = QL.rollout_q(m, Q, Zv[sl, :1], Zp[sl, :1], Aw[sl][:, None], all_frames=True)
            pav.append(fva[:, 0].float()); pap.append(fpa[:, 0].float())          # (b, H, P, C), (b, H, Dp)
        return torch.cat(p1v), torch.cat(p1p), torch.cat(pav), torch.cat(pap)

    Q.configure('none'); Q.begin_plan([0], 'calib')
    r1v, r1p, rav, rap = ds_preds()
    var1 = pred_var(r1v, r1p, alpha)
    varh = {s: pred_var(rav[:, s - 1], rap[:, s - 1], alpha) for s in (3, 5)}
    res['ds']['var1'] = var1
    res['ds'].update({f'var{s}': v for s, v in varh.items()})
    # ---------------------------------------------------------------- planning-time inputs (held split starts)
    sp = QL.SPLITS['held']
    pairs = QL.get_pairs(T, env, 'held', f'{L.MAIN}/pairs')
    H, N, A = L.GOAL_H, L.N_SAMPLES, m.action_encoder.patch_embed.in_channels
    pin = [QL.episode_inputs(m, pre, pairs, e) for e in range(a.nplan)]
    pacts = [torch.randn(N, H, A, generator=torch.Generator().manual_seed(sp['plan_seed'] + e)).cuda()
             for e in range(a.nplan)]

    def plan_preds():
        o = []
        for e in range(a.nplan):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, _, av, ap_ = QL.rollout_q(m, Q, pin[e][0], pin[e][1], pacts[e][None], all_frames=True)
            o.append((av[0].float(), ap_[0].float()))           # (N, H, P, C), (N, H, Dp)
        return o

    Q.configure('none'); Q.begin_plan([0], 'calibplan')
    pref = plan_preds()
    pvar = {s: float(np.mean([pred_var(x[0][:, s - 1], x[1][:, s - 1], alpha) for x in pref])) for s in QL.ERR_STEPS}
    res['plan'].update({f'var{s}': v for s, v in pvar.items()})
    out_path = f'{L.MAIN}/results/{a.tag}/err_{task}_{a.ranges_id}.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    for k in kbits:
        for var in a.vars.split(','):
            ds_runs, pl_runs = [], []
            for s in range(a.nseeds if var in QL.RAND else 1):
                Q.configure(var, k); Q.begin_plan([0], f'calib{s}')
                q1v, q1p, qav, qap = ds_preds()
                ds_runs.append({1: float(cost_err(q1v, q1p, r1v, r1p, alpha).mean()),
                                3: float(cost_err(qav[:, 2], qap[:, 2], rav[:, 2], rap[:, 2], alpha).mean()),
                                5: float(cost_err(qav[:, 4], qap[:, 4], rav[:, 4], rap[:, 4], alpha).mean())})
                ee = {st: [] for st in QL.ERR_STEPS}
                for e in range(a.nplan):
                    Q.configure(var, k); Q.begin_plan([e], f'calibplan{s}')
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        _, _, av, ap_ = QL.rollout_q(m, Q, pin[e][0], pin[e][1], pacts[e][None], all_frames=True)
                    for st in QL.ERR_STEPS:
                        ee[st].append(float(cost_err(av[0, :, st - 1].float(), ap_[0, :, st - 1].float(),
                                                     pref[e][0][:, st - 1], pref[e][1][:, st - 1], alpha).mean()))
                pl_runs.append({st: float(np.mean(v)) for st, v in ee.items()})
            Q.configure('none')
            dsm = {st: float(np.mean([r[st] for r in ds_runs])) for st in (1, 3, 5)}
            plm = {st: float(np.mean([r[st] for r in pl_runs])) for st in QL.ERR_STEPS}
            dvar = {1: var1, 3: varh[3], 5: varh[5]}
            res['ds'][f'{var}{k}'] = {**{f'err{st}': dsm[st] for st in dsm}, **{f'rel_err{st}': dsm[st] / dvar[st] for st in dsm},
                                      'sd_err1': float(np.std([r[1] for r in ds_runs]))}
            res['plan'][f'{var}{k}'] = {**{f'err{st}': plm[st] for st in plm}, **{f'rel_err{st}': plm[st] / pvar[st] for st in plm},
                                        'sd_err1': float(np.std([r[1] for r in pl_runs]))}
            print(f"{task} k={k} {var:8s} ds rel_err1 {dsm[1] / var1:.4f} rel_err3 {dsm[3] / varh[3]:.4f} "
                  f"rel_err5 {dsm[5] / varh[5]:.4f} | plan rel_err1 {plm[1] / pvar[1]:.4f} rel_err3 {plm[3] / pvar[3]:.4f} "
                  f"rel_err5 {plm[5] / pvar[5]:.4f}", flush=True)
            L.jdump(res, out_path)
    res['secs'] = time.time() - t0
    L.jdump(res, out_path)
    print(f'CALIB_DONE {task} in {res["secs"]:.0f}s -> {out_path}', flush=True)


if __name__ == '__main__':
    main()
