"""Analysis of runner.py results (DINO-WM part of paper A). Runs on the Mac on the synced job JSONs (pure python +
numpy/scipy) or on the server.

usage:
  python analyze.py cells <results/<manifest name>> [--out F]        per (task, k, variant) table: success, drop vs
        none with paired bootstrap CI and McNemar, red_v, errors, plan deviation (seed-noise floor from nonealt), OEI
  python analyze.py repro <results/E0_repro> <pilot results dir> [--out F]   reproduction vs the R2-13 pilot
  python analyze.py ksel <results/E0_ksel> [--out F]                 k selection rule (design 6.0)
Statistics as in the pilots: paired bootstrap over episodes (10,000 resamples, seed 0), exact McNemar, Wilson CI.
"""
import os, sys, json, glob, argparse
import numpy as np

NB = 10000
QORDER = ['det', 'ind', 'shiter', 'shstep', 'shep', 'detshift', 'shfixed']


def wilson(s, n, z=1.96):
    if n == 0:
        return [float('nan')] * 2
    p = s / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


def mcnemar(a, b):
    """a, b paired bool arrays -> (#a only, #b only, exact two-sided p)."""
    from scipy.stats import binomtest
    n10 = int(np.sum(a & ~b)); n01 = int(np.sum(~a & b))
    p = 1.0 if n10 + n01 == 0 else float(binomtest(n10, n10 + n01, 0.5).pvalue)
    return n10, n01, p


def boot(fn, n, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(NB, n))
    v = np.array([fn(i) for i in idx], float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return [float('nan'), float('nan')]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def load(res_dir):
    """-> dict (task, var, k, ep) -> record, plus job metas."""
    R, metas = {}, []
    for f in sorted(glob.glob(f'{res_dir}/*/*.json')):
        d = json.load(open(f))
        metas.append(dict(job=d['job'], meta=d['meta']))
        for r in d['records']:
            R[(r['task'], r['var'], int(r['k']), int(r['ep']))] = r
    return R, metas


def rms(x):
    x = np.asarray(x, float)
    return float(np.sqrt((x ** 2).mean()))


def cell_stats(R, task, var, k, eps_ref=None):
    """Statistics of one (task, var, k) cell against none (k-independent) on the common episodes."""
    eps = sorted(e for (t, v, kk, e) in R if t == task and v == var and kk == k)
    eps_none = {e for (t, v, kk, e) in R if t == task and v == 'none'}
    eps = [e for e in eps if e in eps_none and (eps_ref is None or e in eps_ref)]
    n = len(eps)
    if n == 0:
        return None
    rv = [R[(task, var, k, e)] for e in eps]
    rn = [R[(task, 'none', 0, e)] for e in eps]
    S = np.array([r['success'] for r in rv], bool); Sn = np.array([r['success'] for r in rn], bool)
    out = dict(task=task, var=var, k=k, n=n, eps=[eps[0], eps[-1]], success=float(S.mean()), s=int(S.sum()),
               wilson=wilson(int(S.sum()), n), margin_median=float(np.median([r['margin'] for r in rv])),
               state_dist_mean=float(np.mean([r['state_dist'] for r in rv])),
               time_s=float(np.mean([r['time_s'] for r in rv])))
    if var != 'none':
        dv = Sn.astype(float) - S.astype(float)
        n10, n01, p = mcnemar(Sn, S)
        out.update(D=float(dv.mean()), D_ci=boot(lambda i: dv[i].mean(), n), none_only=n10, var_only=n01, mcnemar_p=p)
        # plan deviation: first action of the final plan mean, RMS over its 10 normalized dims, vs none (same seed)
        dev = np.array([rms(np.array(a['mu_final'])[0] - np.array(b['mu_final'])[0]) for a, b in zip(rv, rn)])
        devf = np.array([rms(np.array(a['mu_final']) - np.array(b['mu_final'])) for a, b in zip(rv, rn)])
        out.update(plan_dev_first=float(dev.mean()), plan_dev_full=float(devf.mean()))
    if 'err' in rv[0]:
        for s in ('1', '3', '5'):
            out[f'rel_err{s}'] = float(np.mean([r['err'][s]['rel'] for r in rv]))
            out[f'abs_err{s}'] = float(np.mean([r['err'][s]['abs'] for r in rv]))
    if 'oei' in rv[0]:
        for key in ('it0_rand_dJ', 'last_pop_dJ', 'last_argmin_dJ', 'last_elite_dJ', 'mf_dJ', 'oei_ref_argmin',
                    'oei_ref_elite', 'oei_ref_mf', 'oei_true_argmin', 'oei_true_elite', 'oei_true_mf', 'rho_last',
                    'rho_last_Jnone', 'rho_it0', 'spec_it0', 'spec_last', 'rho_J_Jnone_it0', 'last_pop_success'):
            vals = np.array([r['oei'].get(key, np.nan) for r in rv], float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                out[key] = float(vals.mean())
                if key.startswith('oei_') or key in ('it0_rand_dJ', 'last_argmin_dJ'):
                    out[key + '_ci'] = boot(lambda i: vals[i].mean(), len(vals))
        if 'rho_last' in out:
            d = np.array([r['oei'].get('rho_last', np.nan) - r['oei'].get('rho_last_Jnone', np.nan) for r in rv])
            d = d[np.isfinite(d)]
            out['d_rho_last'] = float(d.mean()); out['d_rho_last_ci'] = boot(lambda i: d[i].mean(), len(d))
    return out


def seed_floor(R, task):
    eps = sorted(e for (t, v, kk, e) in R if t == task and v == 'nonealt' and (t, 'none', 0, e) in R)
    if not eps:
        return None
    a = [R[(task, 'nonealt', 0, e)] for e in eps]; b = [R[(task, 'none', 0, e)] for e in eps]
    f1 = np.mean([rms(np.array(x['mu_final'])[0] - np.array(y['mu_final'])[0]) for x, y in zip(a, b)])
    ff = np.mean([rms(np.array(x['mu_final']) - np.array(y['mu_final'])) for x, y in zip(a, b)])
    Sa = np.array([x['success'] for x in a]); Sb = np.array([y['success'] for y in b])
    return dict(n=len(eps), floor_first=float(f1), floor_full=float(ff), success_nonealt=float(Sa.mean()),
                success_none=float(Sb.mean()), mcnemar=mcnemar(Sb, Sa))


def cells(R):
    tasks = sorted({t for (t, v, k, e) in R})
    out = {}
    for t in tasks:
        T = dict(none=cell_stats(R, t, 'none', 0), floor=seed_floor(R, t), cells={})
        ks = sorted({k for (tt, v, k, e) in R if tt == t and v not in ('none', 'nonealt')})
        for k in ks:
            C = {}
            for v in QORDER:
                c = cell_stats(R, t, v, k)
                if c is not None:
                    C[v] = c
            if 'det' in C:
                # red_v = (D_det - D_v)/D_det, episode-paired bootstrap on the common episodes of det and v
                for v, c in C.items():
                    if v == 'det':
                        continue
                    eps = sorted(set(range(c['eps'][0], c['eps'][1] + 1)) & set(range(C['det']['eps'][0], C['det']['eps'][1] + 1)))
                    eps = [e for e in eps if (t, v, k, e) in R and (t, 'det', k, e) in R and (t, 'none', 0, e) in R]
                    Sn = np.array([R[(t, 'none', 0, e)]['success'] for e in eps], float)
                    Sd = np.array([R[(t, 'det', k, e)]['success'] for e in eps], float)
                    Sv = np.array([R[(t, v, k, e)]['success'] for e in eps], float)

                    def red(i):
                        dd = (Sn - Sd)[i].mean(); dv = (Sn - Sv)[i].mean()
                        return (dd - dv) / dd if dd > 0 else np.nan
                    r0 = red(np.arange(len(eps)))
                    c['red'] = None if not np.isfinite(r0) else float(r0)
                    c['red_ci'] = boot(red, len(eps))
                    n10, n01, p = mcnemar(Sv.astype(bool), Sd.astype(bool))
                    c['vs_det'] = dict(var_only=n10, det_only=n01, mcnemar_p=p, D_v_minus_D_det=float((Sd - Sv).mean()),
                                       ci=boot(lambda i: (Sd - Sv)[i].mean(), len(eps)))
            if T['floor']:
                for c in C.values():
                    if 'plan_dev_first' in c:
                        c['plan_dev_first_norm'] = c['plan_dev_first'] / T['floor']['floor_first']
                        c['plan_dev_full_norm'] = c['plan_dev_full'] / T['floor']['floor_full']
            T['cells'][k] = C
        out[t] = T
    return out


def print_cells(A):
    for t, T in A.items():
        nn = T['none']
        print(f"== {t}: none {nn['s']}/{nn['n']} ({100 * nn['success']:.0f}%), {nn['time_s']:.1f}s/ep")
        if T['floor']:
            f = T['floor']
            print(f"   seed floor (nonealt vs none, n {f['n']}): first-action RMS {f['floor_first']:.4f}, full {f['floor_full']:.4f}; "
                  f"nonealt success {100 * f['success_nonealt']:.0f}%")
        for k, C in T['cells'].items():
            for v, c in C.items():
                s = (f"  k={k} {v:8s} {c['s']:3d}/{c['n']} ({100 * c['success']:.0f}%) D {100 * c['D']:+.0f}pp "
                     f"[{100 * c['D_ci'][0]:.0f},{100 * c['D_ci'][1]:.0f}] McN {c['none_only']}/{c['var_only']} p={c['mcnemar_p']:.3g}")
                if 'red' in c:
                    s += f" red {c['red'] if c['red'] is None else round(c['red'], 2)}"
                if 'rel_err1' in c:
                    s += f" | err1/3/5 {c['rel_err1']:.4f}/{c['rel_err3']:.4f}/{c['rel_err5']:.4f}"
                if 'oei_ref_argmin' in c:
                    s += (f" | dJ it0 {c['it0_rand_dJ']:+.3f} argmin {c['last_argmin_dJ']:+.3f} OEI(ref,argmin) {c['oei_ref_argmin']:+.3f}"
                          f" OEI(true,argmin) {c.get('oei_true_argmin', float('nan')):+.3f}")
                if 'rho_last' in c:
                    s += f" | rho_last {c['rho_last']:.3f} (Jnone {c['rho_last_Jnone']:.3f})"
                if 'plan_dev_first_norm' in c:
                    s += f" | plandev {c['plan_dev_first_norm']:.2f}"
                s += f" | {c['time_s']:.1f}s/ep"
                print(s)


# ------------------------------------------------------------------------------------------------ reproduction
PILOT_K = {'pusht': 4, 'wall': 3}
PILOT_RATES = {'pusht': dict(none=0.78, det=0.52, ind=0.82, shiter=0.80, shfixed=0.82),
               'wall': dict(none=0.48, det=0.08, ind=0.52, shiter=0.58, shfixed=0.56)}


def repro(res_dir, pilot_dir):
    R, metas = load(res_dir)
    A = cells(R)
    cmp = {}
    for t in A:
        P = {}
        for ln in open(f'{pilot_dir}/cem13_{t}_eval.jsonl'):
            r = json.loads(ln)
            P[(r['var'], int(r['ep']))] = r
        C = {}
        for v in ['none', 'det', 'ind', 'shiter', 'shfixed']:
            k = 0 if v == 'none' else PILOT_K[t]
            eps = sorted(e for (tt, vv, kk, e) in R if tt == t and vv == v and kk == k and (v, e) in P)
            if not eps:
                continue
            s_srv = np.array([R[(t, v, k, e)]['success'] for e in eps]); s_pil = np.array([P[(v, e)]['success'] for e in eps])
            m_srv = np.array([R[(t, v, k, e)]['margin'] for e in eps]); m_pil = np.array([P[(v, e)]['margin'] for e in eps])
            from scipy.stats import spearmanr
            n10, n01, p = mcnemar(s_pil, s_srv)
            C[v] = dict(n=len(eps), server=float(s_srv.mean()), pilot=float(s_pil.mean()), agree=float((s_srv == s_pil).mean()),
                        pilot_only=n10, server_only=n01, mcnemar_p=p, margin_spearman=float(spearmanr(m_srv, m_pil).correlation),
                        margin_identical=float(np.mean(np.abs(m_srv - m_pil) < 1e-6)))
        # pilot-style drops (paired with the pilot's none) for the same episodes
        cmp[t] = C
    gpu = sorted({m['meta']['gpu'] for m in metas})
    return dict(cells=A, vs_pilot=cmp, gpu=gpu, pilot_reported=PILOT_RATES)


# ------------------------------------------------------------------------------------------------ k selection
def ksel(res_dir):
    R, metas = load(res_dir)
    A = cells(R)
    out = {}
    for t, T in A.items():
        rows = {}
        for k, C in T['cells'].items():
            if 'det' in C:
                c = C['det']
                rows[k] = dict(det_success=c['success'], s=c['s'], n=c['n'], drop=c['D'], D_ci=c['D_ci'],
                               mcnemar_p=c['mcnemar_p'], rel_err1=c.get('rel_err1'), rel_err5=c.get('rel_err5'))
        ok = [k for k in sorted(rows) if k in (3, 4, 5, 6) and rows[k]['drop'] >= 0.20 - 1e-9]
        primary = max(ok) if ok else None
        out[t] = dict(none_success=T['none']['success'], none_s=T['none']['s'], n=T['none']['n'], det=rows,
                      primary_k=primary, cells=[primary - 1, primary, primary + 1] if primary else None,
                      rule='largest k in {3,4,5,6} with det success drop >= 20pp (held, 20 episodes)')
    return out


def throughput(res_dir):
    """Episodes per hour from the job metadata: GPU seconds per episode by (task, variant) and the wall-clock rate of
    the whole manifest run (first job start to last job finalized, i.e. including variant interleaving, model loading,
    executed-plan replays and waiting for the true-cost replays)."""
    from datetime import datetime
    _, metas = load(res_dir)
    out = {}
    for t in sorted({m['job']['task'] for m in metas}):
        ms = [m for m in metas if m['job']['task'] == t]
        n_ep = sum(m['job']['ep1'] - m['job']['ep0'] for m in ms)
        ts = [datetime.strptime(m['meta']['t_start'], '%Y-%m-%dT%H:%M:%SZ') for m in ms]
        tf = [datetime.strptime(m['meta']['t_finalized'], '%Y-%m-%dT%H:%M:%SZ') for m in ms]
        wall_h = (max(tf) - min(ts)).total_seconds() / 3600
        gpu_h = sum(m['meta']['gpu_s'] for m in ms) / 3600
        by_var = {}
        for m in ms:
            v = m['job']['var']
            d = by_var.setdefault(v, [0.0, 0])
            d[0] += m['meta']['gpu_s']; d[1] += m['job']['ep1'] - m['job']['ep0']
        out[t] = dict(episodes=n_ep, wall_h=wall_h, ep_per_h_wall=n_ep / wall_h if wall_h > 0 else float('nan'),
                      gpu_h=gpu_h, ep_per_h_gpu=n_ep / gpu_h if gpu_h > 0 else float('nan'),
                      s_per_ep_by_var={v: d[0] / d[1] for v, d in by_var.items()},
                      E=sorted({m['job']['E'] for m in ms}), peak_mem_gb=max(m['meta']['peak_mem_gb'] for m in ms),
                      replay_wait_s_total=sum(m['meta'].get('replay_wait_s', 0) for m in ms),
                      first_start=min(ts).isoformat(), last_end=max(tf).isoformat())
    return out


CSV_KEYS = ['task', 'k', 'var', 'n', 's', 'success', 'wilson', 'D', 'D_ci', 'none_only', 'var_only', 'mcnemar_p', 'red',
            'red_ci', 'margin_median', 'state_dist_mean', 'rel_err1', 'rel_err3', 'rel_err5', 'plan_dev_first',
            'plan_dev_first_norm', 'it0_rand_dJ', 'last_argmin_dJ', 'oei_ref_argmin', 'oei_ref_argmin_ci', 'oei_ref_elite',
            'oei_ref_mf', 'oei_true_argmin', 'rho_last', 'rho_last_Jnone', 'd_rho_last', 'rho_it0', 'time_s']


def write_csv(A, path):
    """One row per (task, k, variant) of a cells() result (none rows have k = 0)."""
    def fmt(x):
        if isinstance(x, (list, tuple)):
            return '[' + ';'.join(fmt(y) for y in x) + ']'
        if isinstance(x, float):
            return f'{x:.5g}'
        return '' if x is None else str(x)
    with open(path, 'w') as f:
        f.write(','.join(CSV_KEYS) + '\n')
        for t, T in A.items():
            rows = [T['none']] + [c for k, C in T['cells'].items() for c in C.values()]
            for c in rows:
                f.write(','.join(fmt(c.get(key)) for key in CSV_KEYS) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode'); ap.add_argument('res'); ap.add_argument('pilot', nargs='?'); ap.add_argument('--out', default=None)
    ap.add_argument('--csv', default=None)
    a = ap.parse_args()
    if a.mode == 'throughput':
        A = throughput(a.res)
        for t, d in A.items():
            print(f"== {t}: {d['episodes']} episode-runs, wall {d['wall_h']:.2f} h -> {d['ep_per_h_wall']:.0f} ep/h "
                  f"(GPU part {d['ep_per_h_gpu']:.0f} ep/h), E {d['E']}, peak {d['peak_mem_gb']:.1f} GB, "
                  f"replay wait {d['replay_wait_s_total']:.0f} s")
            print('   s/ep by variant', {v: round(x, 1) for v, x in d['s_per_ep_by_var'].items()})
    elif a.mode == 'cells':
        R, _ = load(a.res)
        A = cells(R); print_cells(A)
    elif a.mode == 'repro':
        A = repro(a.res, a.pilot)
        print_cells(A['cells'])
        print('GPU', A['gpu'])
        for t, C in A['vs_pilot'].items():
            for v, c in C.items():
                print(f"  {t} {v:8s} n {c['n']} server {100 * c['server']:.0f}% pilot {100 * c['pilot']:.0f}% agree {100 * c['agree']:.0f}% "
                      f"(pilot only {c['pilot_only']}, server only {c['server_only']}, McN p {c['mcnemar_p']:.3g}) "
                      f"margin rho {c['margin_spearman']:.3f} identical {100 * c['margin_identical']:.0f}%")
    elif a.mode == 'ksel':
        A = ksel(a.res)
        for t, T in A.items():
            print(f"== {t}: none {T['none_s']}/{T['n']}; primary k = {T['primary_k']} -> cells {T['cells']}")
            for k, r in sorted(T['det'].items()):
                print(f"   det k={k}: {r['s']}/{r['n']} drop {100 * r['drop']:+.0f}pp [{100 * r['D_ci'][0]:.0f},{100 * r['D_ci'][1]:.0f}] "
                      f"p={r['mcnemar_p']:.3g} rel_err1 {r['rel_err1']} rel_err5 {r['rel_err5']}")
    else:
        raise SystemExit(a.mode)
    if a.csv and a.mode in ('cells', 'repro'):
        write_csv(A['cells'] if a.mode == 'repro' else A, a.csv)
        print('wrote', a.csv)
    if a.out:
        json.dump(A, open(a.out, 'w'), indent=1, default=float)
        print('wrote', a.out)


if __name__ == '__main__':
    main()
