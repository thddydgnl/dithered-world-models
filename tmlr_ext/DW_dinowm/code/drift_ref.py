"""DESIGN_main v5.2 section 9.1 (b3), DINO-WM: region-effect control (descriptive, outside the rules).

The unquantized (none) planner's E1 populations of each episode are evaluated with the det-quantized model:
  f_ref(e) = mean over the iteration-0 random candidates (rows 1.., row 0 = mu = 0) of J_det - J_none
             (every variant sees the same iteration-0 population, so this equals the det record's it0_rand_dJ;
             it is recomputed here and compared as a check)
  l_ref(e) = mean over the none planner's LAST population (all rows, as last_pop_dJ) of J_det - J_none
  drift_ref(e) = l_ref(e) - f_ref(e)
drift_det - drift_ref is then the part of det's drift not explained by the region the unquantized planner moves to.
J_none of both populations is taken from the stored E1 arrays and recomputed as a check (max |diff| reported);
J_det of the iteration-0 population is compared with the det variant's stored it0_J.

Numerics as E1 (TF32, SDPA, bf16 autocast rollouts, float32 cost, compiled quantizer kernels, E = 1 rows of 300).

--mode mf_same (descriptive sel_last of the executed plan under the SAME draw, v5.2 (b2) description): for the fresh
shared variants sh-iter and det+shift, whose tables are seeded by (salt, variant, episode, iteration, horizon step,
layer), the executed plan mu_final (exact, from pops/) is evaluated under the last CEM iteration's draw
(it = opt_steps - 1): e_same_mf = J_v(mf | last draw) - J_none_mf. As a check the draw it = opt_steps must give the
stored J_v_mf exactly; episodes are batched with the task's E1 batch size E (single-row forwards depend on E).
ind has no same draw for mf (its noise is per candidate row and mf was never a row).

usage (server): DINOWM_GPU=1 bash run.sh drift_ref.py --man E1_dinowm_pusht --task pusht --ks 3,4,5 [--split eval6]
                [--eps 0-99] [--ranges main] [--mode pop|mf_same] [--out JSON]
default out: results/drift_ref/<man>_<task>.json (pop) or results/mf_same/<man>_<task>.json (mf_same).
Resumable: episodes already in the output JSON are skipped."""
import os, sys, json, glob, time, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wmlib as L
import qlib as QL
from runner import load_ranges, MAIN


def one(path_glob):
    f = [p for p in sorted(glob.glob(path_glob)) if not p.endswith('_truth.npz')]
    return f[0] if len(f) == 1 else None


@torch.no_grad()
def costs(m, Q, inp, actions, alpha):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        zv, zp = QL.rollout_chunks(m, Q, inp[0], inp[1], actions[None])
    J, _, _ = QL.cost(zv[0].float(), zp[0].float(), inp[2], inp[3], alpha)
    return J.cpu().numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--man', required=True); ap.add_argument('--task', required=True); ap.add_argument('--ks', required=True)
    ap.add_argument('--split', default=None); ap.add_argument('--eps', default='0-99'); ap.add_argument('--ranges', default='main')
    ap.add_argument('--out', default=None); ap.add_argument('--mode', default='pop', choices=['pop', 'mf_same'])
    ap.add_argument('--E', type=int, default=None, help='mf_same batch size (default: the E1 batch size of the task)')
    a = ap.parse_args()
    split = a.split or ('eval6' if a.task == 'pusht' else 'eval')
    ks = [int(x) for x in a.ks.split(',')]
    e0, e1 = (int(x) for x in a.eps.split('-'))
    out = a.out or f"{MAIN}/results/{'drift_ref' if a.mode == 'pop' else 'mf_same'}/{a.man}_{a.task}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    res = json.load(open(out)) if os.path.exists(out) else dict(man=a.man, task=a.task, split=split, ranges=a.ranges,
                                                                  episodes={})
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(4)
    L.patch_attention(True)
    T = L.load_task(a.task)
    m, pre = T['model'], T['pre']
    env = L.make_env(a.task)
    pairs = QL.get_pairs(T, env, split, f'{MAIN}/pairs')
    Q = QL.Quantizer(m, use_compile=True)
    Q.set_ranges(load_ranges(a.task, a.ranges))
    alpha = L.TASKS[a.task]['alpha']
    salt = QL.SPLITS[split]['salt']
    P = f'{MAIN}/pops/{a.man}/{a.task}'
    t0 = time.time()
    if a.mode == 'mf_same':
        mf_same(a, res, out, m, pre, pairs, Q, alpha, salt, split, ks, e0, e1, P, t0)
        return
    for e in range(e0, e1 + 1):
        if str(e) in res['episodes'] and all(str(k) in res['episodes'][str(e)]['k'] for k in ks):
            continue
        pn, pi = one(f'{P}/{split}_none_e???-???_e{e}.npz'), f'{P}/{split}_it0_e{e}.npz'
        if pn is None or not os.path.exists(pi):
            print(f'ep {e}: none or it0 population missing, skipped', flush=True)
            continue
        zn, zi = np.load(pn), np.load(pi)
        A0 = torch.as_tensor(zi['actions']).cuda(); AL = torch.as_tensor(zn['last_actions']).cuda()
        inp = QL.episode_inputs(m, pre, pairs, e)
        Q.configure('none')
        Jn0, JnL = costs(m, Q, inp, A0, alpha), costs(m, Q, inp, AL, alpha)
        Jn0s, JnLs = zi['J_none'].astype(np.float64), zn['last_J_none'].astype(np.float64)
        rec = res['episodes'].get(str(e), dict(check_none_it0=float(np.abs(Jn0 - Jn0s).max()),
                                                 check_none_last=float(np.abs(JnL - JnLs).max()), k={}))
        for k in ks:
            Q.configure('det', k); Q.begin_plan([e], salt); Q.it = 0
            Jd0, JdL = costs(m, Q, inp, A0, alpha), costs(m, Q, inp, AL, alpha)
            pd = one(f'{P}/{split}_det_k{k}_e???-???_e{e}.npz')
            chk = float(np.abs(Jd0 - np.load(pd)['it0_J']).max()) if pd else None
            f_ref = float((Jd0[1:] - Jn0s[1:]).mean()); l_ref = float((JdL - JnLs).mean())
            rec['k'][str(k)] = dict(f_ref=f_ref, l_ref=l_ref, drift_ref=l_ref - f_ref, check_det_it0=chk)
        Q.configure('none')
        res['episodes'][str(e)] = rec
        print(f"ep {e} " + ' '.join(f"k{k} drift_ref {rec['k'][str(k)]['drift_ref']:+.4f}" for k in ks) +
              f" checks none {rec['check_none_it0']:.2e}/{rec['check_none_last']:.2e} det "
              + ','.join(f"{rec['k'][str(k)]['check_det_it0']}" for k in ks) + f" ({time.time() - t0:.0f}s)", flush=True)
        if e % 10 == 9 or e == e1:
            tmp = out + '.tmp'; json.dump(res, open(tmp, 'w'), indent=1); os.replace(tmp, out)
    res['meta'] = dict(gpu=torch.cuda.get_device_name(0), cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
                       time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), secs=time.time() - t0)
    tmp = out + '.tmp'; json.dump(res, open(tmp, 'w'), indent=1); os.replace(tmp, out)
    print(f'DRIFT_REF_DONE {a.man} {a.task} k={ks} -> {out} ({time.time() - t0:.0f}s)', flush=True)


@torch.no_grad()
def mf_same(a, res, out, m, pre, pairs, Q, alpha, salt, split, ks, e0, e1, P, t0):
    E = a.E or (2 if a.task == 'wall' else 1)               # E1 batch sizes (DESIGN_main v2 section 4.3)
    opt_steps = L.TASKS[a.task]['opt_steps']
    recs = {}
    for f in glob.glob(f'{MAIN}/results/{a.man}/{a.task}/{split}_*.json'):
        for r in json.load(open(f))['records']:
            recs[(r['var'], int(r['k']), int(r['ep']))] = r
    for var in ('shiter', 'detshift'):
        for k in ks:
            key = f'{var}{k}'
            for g0 in range(e0, e1 + 1, E):
                ee = [e for e in range(g0, min(g0 + E, e1 + 1))]
                if all(key in res['episodes'].get(str(e), {}) for e in ee):
                    continue
                if not all((var, k, e) in recs for e in ee):
                    print(f'{key} eps {ee}: E1 record missing, skipped', flush=True)
                    continue
                pz = [one(f'{P}/{split}_{var}_k{k}_e???-???_e{e}.npz') for e in ee]
                if any(p is None for p in pz):
                    print(f'{key} eps {ee}: population file missing, skipped', flush=True)
                    continue
                inputs = [QL.episode_inputs(m, pre, pairs, e) for e in ee]
                z0v = torch.cat([x[0] for x in inputs]); z0p = torch.cat([x[1] for x in inputs])
                mf = torch.stack([torch.as_tensor(np.load(p)['mu_final']) for p in pz]).cuda()[:, None]
                Js = {}
                for it in (opt_steps - 1, opt_steps):
                    Q.configure(var, k); Q.begin_plan(ee, salt); Q.it = it
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        fv, fp = QL.rollout_q(m, Q, z0v, z0p, mf)
                    Js[it] = [float(QL.cost(fv[kk].float(), fp[kk].float(), inputs[kk][2], inputs[kk][3], alpha)[0][0])
                              for kk in range(len(ee))]
                Q.configure('none')
                for kk, e in enumerate(ee):
                    r = recs[(var, k, e)]
                    res['episodes'].setdefault(str(e), {})[key] = dict(
                        e_same_mf=Js[opt_steps - 1][kk] - r['J_none_mf'], J_v_same_mf=Js[opt_steps - 1][kk],
                        check_v_mf=abs(Js[opt_steps][kk] - r['J_v_mf']))
                print(f"{key} eps {ee} e_same_mf " + ' '.join(f"{res['episodes'][str(e)][key]['e_same_mf']:+.4f}" for e in ee)
                      + ' check ' + ' '.join(f"{res['episodes'][str(e)][key]['check_v_mf']:.1e}" for e in ee)
                      + f' ({time.time() - t0:.0f}s)', flush=True)
            tmp = out + '.tmp'; json.dump(res, open(tmp, 'w'), indent=1); os.replace(tmp, out)
    res['meta'] = dict(mode='mf_same', E=E, gpu=torch.cuda.get_device_name(0), cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
                       time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), secs=time.time() - t0)
    tmp = out + '.tmp'; json.dump(res, open(tmp, 'w'), indent=1); os.replace(tmp, out)
    print(f'MF_SAME_DONE {a.man} {a.task} k={ks} -> {out} ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
