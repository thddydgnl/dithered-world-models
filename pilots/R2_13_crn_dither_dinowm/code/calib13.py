"""R2-13 calibration and prediction errors.
1. Per-layer quantizer ranges from dataset windows (train split, rendered by the simulator from dataset states):
   GELU outputs of the 6 predictor MLPs under bf16 autocast (as in planning); lo = min, hi = 99.99th percentile.
2. Prediction errors of every variant and k vs the unquantized model, in planning-cost units
   (visual MSE + alpha * proprio-embedding MSE between the quantized and the unquantized prediction):
   - ds_err1: one predictor call on held-out dataset windows (valid split) of num_hist frames (next-frame prediction)
   - ds_err5: 5-step open-loop rollout from the first window frame with the dataset actions (final frame)
   - plan_err1 / plan_err5: planning-time inputs = start frames of the held split with 300 CEM iteration-0 candidates
     (N(0,1) actions); horizon step 1 and 5 of the rollout
   Each error is also given relative to the variance of the unquantized predictions (rel_*).
usage: python calib13.py <task> [--kbits 2,3,4,5,6,8] [--vars det,ind,shiter,shfixed] [--nwin 64] [--nseeds 3]
       [--nplan 8] [--tag T] [--nocompile] [--ranges_only] [--redo_ranges]
Ranges are computed once (results/ranges_<task>.json) and reused unless --redo_ranges."""
import sys, os, time, json, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r21lib as L
import qlib13 as Q13


def cost_err(av, ap, bv, bp, alpha):
    """per-sample error in cost units between predictions a and b: visual (b,...,P,C), proprio (b,...,Dp)."""
    return ((av - bv) ** 2).mean((-1, -2)) + alpha * ((ap - bp) ** 2).mean(-1)


def pred_var(v, p, alpha):
    """total variance of the reference predictions across samples in cost units (normalizer for rel errors)."""
    v = v.reshape(-1, *v.shape[-2:]); p = p.reshape(-1, p.shape[-1])
    return float(((v - v.mean(0, keepdim=True)) ** 2).mean() + alpha * ((p - p.mean(0, keepdim=True)) ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task'); ap.add_argument('--kbits', default='2,3,4,5,6,8')
    ap.add_argument('--nwin', type=int, default=64); ap.add_argument('--nseeds', type=int, default=3)
    ap.add_argument('--nocompile', action='store_true'); ap.add_argument('--ranges_only', action='store_true')
    ap.add_argument('--nplan', type=int, default=8)
    ap.add_argument('--vars', default=','.join(Q13.QVARS)); ap.add_argument('--tag', default='')
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
    Q = Q13.Quantizer(m, use_compile=not a.nocompile)
    B = 16
    # ---------------------------------------------------------------- 1. ranges (train split windows)
    rpath = f'{Q13.RES}/ranges_{task}.json'
    if os.path.exists(rpath) and not a.redo_ranges and not a.ranges_only:
        layers = json.load(open(rpath))['layers']
        print('loaded ranges', [(round(r['lo'], 4), round(r['hi'], 3)) for r in layers], flush=True)
    else:
        layers = calibrate(a, T, env, m, pre, Q, nh, B, t0, rpath)
    if a.ranges_only:
        return
    run_errors(a, T, env, m, pre, Q, nh, B, t0, alpha, layers)


def calibrate(a, T, env, m, pre, Q, nh, B, t0, rpath):
    task = T['task']
    cal = Q13.dataset_windows(T, env, 'train', a.nwin, seed=11, n_frames=nh)
    print(f'calibration windows: {cal["visual"].shape} rendered in {time.time() - t0:.0f}s', flush=True)
    Q.start_record()
    for s in range(0, a.nwin, B):
        idx = np.arange(s, min(s + B, a.nwin))
        zv, zp = Q13.encode_windows(m, pre, cal, idx)
        Q13.window_predict(m, zv, zp, torch.as_tensor(cal['act'][idx, :nh]).cuda())
    layers = Q.finish_record()
    rng = dict(task=task, source='train-split dataset windows rendered by the simulator', nwin=a.nwin, num_hist=nh,
               q_hi=0.9999, layers=layers, traj=cal['traj'][:8])
    L.jdump(rng, rpath)
    for li, r in enumerate(layers):
        print(f"layer {li}: lo {r['lo']:.4f} hi(99.99%) {r['hi']:.3f} max {r['max']:.3f} p50 {r['p50']:.4f} "
              f"p99 {r['p99']:.3f} frac<0 {r['frac_below_0']:.3f}", flush=True)
    return layers


def run_errors(a, T, env, m, pre, Q, nh, B, t0, alpha, layers):
    task = T['task']
    Q.set_ranges([(r['lo'], r['hi']) for r in layers])
    kbits = [int(x) for x in a.kbits.split(',')]
    res = dict(task=task, alpha=alpha, kbits=kbits, nseeds=a.nseeds, ds={}, plan={})
    # ---------------------------------------------------------------- 2a. dataset windows (valid split)
    hel = Q13.dataset_windows(T, env, 'valid', a.nwin, seed=12, n_frames=nh)
    Zv, Zp = [], []
    for s in range(0, a.nwin, B):
        idx = np.arange(s, min(s + B, a.nwin))
        zv, zp = Q13.encode_windows(m, pre, hel, idx)
        Zv.append(zv); Zp.append(zp)
    Zv, Zp = torch.cat(Zv), torch.cat(Zp)
    Aw = torch.as_tensor(hel['act']).cuda()

    def ds_preds():
        p1v, p1p, p5v, p5p = [], [], [], []
        for s in range(0, a.nwin, B):
            sl = slice(s, min(s + B, a.nwin))
            v, p = Q13.window_predict(m, Zv[sl], Zp[sl], Aw[sl, :nh])
            p1v.append(v); p1p.append(p)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                fv, fp = Q13.rollout_q(m, Q, Zv[sl, :1], Zp[sl, :1], Aw[sl][:, None])
            p5v.append(fv[:, 0].float()); p5p.append(fp[:, 0].float())
        return torch.cat(p1v), torch.cat(p1p), torch.cat(p5v), torch.cat(p5p)

    Q.configure('none'); Q.begin_plan([0], 'calib')
    r1v, r1p, r5v, r5p = ds_preds()
    var1, var5 = pred_var(r1v, r1p, alpha), pred_var(r5v, r5p, alpha)
    res['ds']['var1'], res['ds']['var5'] = var1, var5
    # ---------------------------------------------------------------- 2b. planning-time inputs (held split starts)
    sp = Q13.SPLITS['held']
    pairs = L.sample_pairs(T, env, a.nplan, seed=sp['pair_seed'])
    H, N, A = L.GOAL_H, L.N_SAMPLES, m.action_encoder.patch_embed.in_channels
    pin = [Q13.episode_inputs(m, pre, pairs, e) for e in range(a.nplan)]
    pacts = [torch.randn(N, H, A, generator=torch.Generator().manual_seed(sp['plan_seed'] + e)).cuda()
             for e in range(a.nplan)]

    def plan_preds():
        o = []
        for e in range(a.nplan):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, _, av, ap_ = Q13.rollout_q(m, Q, pin[e][0], pin[e][1], pacts[e][None], all_frames=True)
            o.append((av[0, :, 0].float(), ap_[0, :, 0].float(), av[0, :, -1].float(), ap_[0, :, -1].float()))
        return o

    Q.configure('none'); Q.begin_plan([0], 'calibplan')
    pref = plan_preds()
    pv1 = float(np.mean([pred_var(x[0], x[1], alpha) for x in pref]))
    pv5 = float(np.mean([pred_var(x[2], x[3], alpha) for x in pref]))
    res['plan']['var1'], res['plan']['var5'] = pv1, pv5
    out_path = f'{Q13.RES}/err_{task}{a.tag}.json'
    for k in kbits:
        for var in a.vars.split(','):
            ds_runs, pl_runs = [], []
            for s in range(a.nseeds if var != 'det' else 1):
                Q.configure(var, k); Q.begin_plan([0], f'calib{s}')
                q1v, q1p, q5v, q5p = ds_preds()
                ds_runs.append((float(cost_err(q1v, q1p, r1v, r1p, alpha).mean()),
                                float(cost_err(q5v, q5p, r5v, r5p, alpha).mean())))
                e1, e5 = [], []
                for e in range(a.nplan):
                    Q.configure(var, k); Q.begin_plan([e], f'calibplan{s}')
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        _, _, av, ap_ = Q13.rollout_q(m, Q, pin[e][0], pin[e][1], pacts[e][None], all_frames=True)
                    e1.append(float(cost_err(av[0, :, 0].float(), ap_[0, :, 0].float(), pref[e][0], pref[e][1], alpha).mean()))
                    e5.append(float(cost_err(av[0, :, -1].float(), ap_[0, :, -1].float(), pref[e][2], pref[e][3], alpha).mean()))
                pl_runs.append((float(np.mean(e1)), float(np.mean(e5))))
            Q.configure('none')
            d1, d5 = np.mean([r[0] for r in ds_runs]), np.mean([r[1] for r in ds_runs])
            p1, p5 = np.mean([r[0] for r in pl_runs]), np.mean([r[1] for r in pl_runs])
            res['ds'][f'{var}{k}'] = dict(err1=float(d1), err5=float(d5), rel_err1=float(d1 / var1), rel_err5=float(d5 / var5),
                                          sd_err1=float(np.std([r[0] for r in ds_runs])))
            res['plan'][f'{var}{k}'] = dict(err1=float(p1), err5=float(p5), rel_err1=float(p1 / pv1), rel_err5=float(p5 / pv5),
                                            sd_err1=float(np.std([r[0] for r in pl_runs])))
            print(f"{task} k={k} {var:8s} ds rel_err1 {d1 / var1:.4f} rel_err5 {d5 / var5:.4f} | plan rel_err1 "
                  f"{p1 / pv1:.4f} rel_err5 {p5 / pv5:.4f}", flush=True)
            L.jdump(res, out_path)
    res['secs'] = time.time() - t0
    L.jdump(res, out_path)
    print(f'CALIB13_DONE {task} in {res["secs"]:.0f}s', flush=True)


if __name__ == '__main__':
    main()
