"""S4: mean-replaced weak-activation ablation in the predictor MLPs vs a candidate-consistent fixed random ablation
(D-fix: the same random units for every candidate/token), ranking damage at the first vs the last CEM step.
Calibration and D-fix size matching on held-split populations; measurement on eval-split populations.
usage: python s4_7.py <task> [alpha]"""
import sys, os, glob, json, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r21lib as L
import r27lib as R7
from cem7 import SPLITS


def load_pop(task, split, e):
    z = np.load(f'{R7.POPS}/{task}_{split}_e{e}.npz')
    return z


@torch.no_grad()
def model_costs(m, inp, actions, alpha):
    z0v, z0p, gv, gp = inp
    a = torch.tensor(actions).cuda()[None]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        zv, zp = R7.rollout_batched(m, z0v, z0p, a)
    zv, zp = zv[0].float(), zp[0].float()
    V, Q, _ = R7.cost_parts(zv, zp, gv, gp)
    return (V + alpha * Q), zv, zp


def main():
    task = sys.argv[1]
    alpha_w = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
    torch.backends.cuda.matmul.allow_tf32 = True
    L.patch_attention(True)
    T = L.load_task(task)
    m, pre = T['model'], T['pre']
    alpha = L.TASKS[task]['alpha']
    env = L.make_env(task)
    steps = R7.STEPS_REC[task]
    first, last = steps[0], steps[-1]
    t0 = time.time()
    held = L.sample_pairs(T, env, SPLITS['held']['n'], seed=SPLITS['held']['pair_seed'])
    ev = L.sample_pairs(T, env, SPLITS['eval']['n'], seed=SPLITS['eval']['pair_seed'])
    held_eps = sorted(int(f.split('_e')[-1].split('.')[0]) for f in glob.glob(f'{R7.POPS}/{task}_held_e*.npz')
                      if not f.endswith('_truth.npz'))
    eval_eps = sorted(int(f.split('_e')[-1].split('_')[0]) for f in glob.glob(f'{R7.POPS}/{task}_eval_e*_truth.npz'))
    ab = R7.MLPAblator(m)
    # ---- calibration (held populations at first / middle / last step)
    calib_eps = held_eps[:8]
    ab.start_stats()
    for e in calib_eps:
        inp = R7.episode_inputs(m, pre, held, e)
        z = load_pop(task, 'held', e)
        for s in (first, R7.MID[task], last):
            model_costs(m, inp, z[f's{s}_actions'], alpha)
    ab.finish_stats(alpha_w)
    print('calibrated; weak fraction per layer', [round(x, 3) for x in ab.weak_frac], f'{time.time()-t0:.0f}s', flush=True)

    # ---- perturbation size of weak-meanrep on held populations, then D-fix fraction by bisection
    match_eps = held_eps[8:14] if len(held_eps) >= 14 else held_eps[:6]
    cache = []
    for e in match_eps:
        inp = R7.episode_inputs(m, pre, held, e)
        z = load_pop(task, 'held', e)
        for s in (first, last):
            ab.mode = None
            _, zv0, zp0 = model_costs(m, inp, z[f's{s}_actions'], alpha)
            cache.append((inp, z[f's{s}_actions'], zv0, zp0))

    def perturb(mode, frac=None):
        tot = 0.0
        for inp, acts, zv0, zp0 in cache:
            if mode == 'meanrep':
                ab.mode = 'meanrep'
            else:
                ab.set_dfix(frac)
            _, zv1, zp1 = model_costs(m, inp, acts, alpha)
            tot += float(((zv1 - zv0) ** 2).mean() + ((zp1 - zp0) ** 2).mean())
        ab.mode = None
        return tot / len(cache)

    target = perturb('meanrep')
    lo, hi = 1e-4, 0.6
    for _ in range(12):
        mid = float(np.sqrt(lo * hi))
        if perturb('dfix', mid) < target:
            lo = mid
        else:
            hi = mid
    frac = float(np.sqrt(lo * hi))
    p_dfix = perturb('dfix', frac)
    print(f'target perturbation {target:.4g}; D-fix fraction {frac:.4f} -> {p_dfix:.4g} ({time.time()-t0:.0f}s)', flush=True)

    # ---- measurement on eval populations (first vs last step)
    from scipy.stats import spearmanr
    rows = []
    for e in eval_eps:
        inp = R7.episode_inputs(m, pre, ev, e)
        z = load_pop(task, 'eval', e)
        y = np.load(f'{R7.POPS}/{task}_eval_e{e}_truth.npz')
        row = dict(ep=e)
        for s in (first, last):
            acts = z[f's{s}_actions']; ys = y[f's{s}_margin']
            ab.mode = None
            Jb = model_costs(m, inp, acts, alpha)[0].cpu().numpy()
            ab.mode = 'meanrep'
            Jw = model_costs(m, inp, acts, alpha)[0].cpu().numpy()
            ab.set_dfix(frac)
            Jd = model_costs(m, inp, acts, alpha)[0].cpu().numpy()
            ab.mode = None
            rb, rw, rd = (spearmanr(J, ys).correlation for J in (Jb, Jw, Jd))
            row[f's{s}'] = dict(rho_base=float(rb), rho_weak=float(rw), rho_dfix=float(rd),
                                perturb_weak=float(1 - spearmanr(Jw, Jb).correlation),
                                perturb_dfix=float(1 - spearmanr(Jd, Jb).correlation))
        rows.append(row)
    ab.remove()

    def ratio(key):
        early = np.mean([r[f's{first}']['rho_base'] - r[f's{first}'][key] for r in rows])
        late = np.mean([r[f's{last}']['rho_base'] - r[f's{last}'][key] for r in rows])
        return float(early), float(late), (float(late / early) if early > 0 else None)

    ew, lw, rw = ratio('rho_weak'); ed, ld, rd = ratio('rho_dfix')
    out = dict(task=task, alpha_weak=alpha_w, weak_frac=ab.weak_frac, dfix_frac=frac, target_perturb=target,
               dfix_perturb=p_dfix, n_eval=len(rows), first=first, last=last,
               weak=dict(damage_early=ew, damage_late=lw, ratio=rw),
               dfix=dict(damage_early=ed, damage_late=ld, ratio=rd), rows=rows,
               S4_pass=bool(rw is not None and rw >= 2 and (rd is None or rw > rd)), time_s=time.time() - t0)
    L.jdump(out, f'{R7.RES}/s4_{task}_a{alpha_w:g}.json')
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
