"""E1 analysis for the DINO-WM part of paper A (DESIGN_main v1-v3: section 6.1 H1/H3a, v2 cells, v3 OEI and PushT split).
Runs on the Mac on the synced job JSONs (pure python + numpy/scipy).

usage: python analyze_e1.py <results dir holding E1_dinowm_wp/ and E1_dinowm_pusht/> [--out JSON] [--csv CSV]

Cells (DESIGN_main v2 section 4.1): confirmatory PushT k {3,4,5}, Wall k {4,5,6}, PointMaze k {2,3,4};
exploratory Wall k=3 (reported separately, not in H1/H3a). Variants: none (once per task), det, ind, sh-iter,
sh-plan (code `shstep`; stands for both sh-step and sh-ep, v2 section 4.2), det+shift; nonealt = seed-noise floor.

Operational choices (stated in the E1 report):
- "det drop real" (H1, qualifying cells of H3a): paired bootstrap (10,000, episode resampling) one-sided p for
  D_det <= 0, p = (#{D* <= 0} + 1)/(B + 1); per-task Holm-Bonferroni over the task's confirmatory k cells at
  alpha = 0.025 (one-sided, matching a two-sided 95% CI). The unadjusted criterion (95% CI lower bound > 0) and the
  McNemar exact p are reported too.
- H1 (DINO-WM family): at least one confirmatory cell with a real det drop in which det's planning-time 1-step error
  (mean over the cell's 100 episodes of the per-episode relative error on the iteration-0 candidates, dither seed
  3-draw mean) is <= sh-iter's.
- H3a (DINO-WM family): among qualifying cells, share with red_sh-iter >= 0.5 is >= 75%; and in no confirmatory cell
  does the paired bootstrap 95% CI lower bound of D_sh-iter - D_det exceed +5pp.
- OEI (v3): confirmatory = unquantized-model-cost reference `oei_ref_mf`: cost error e = J_v - J_none of the executed
  plan (the final CEM elite mean) minus the mean e of the iteration-0 random candidates; auxiliary `oei_ref_argmin`
  (last-iteration argmin, the pilot's "chosen candidate"), `oei_ref_elite` (last-iteration top-30 mean) and the
  simulator-true-cost version `oei_true_*`.
- H2a and H2b from E1 data (DESIGN_main v4 section 6):
  H2a: PushT primary cell (k=4): det OEI (oei_ref_mf) 95% CI upper bound < 0 and sh-iter's OEI CI contains 0.
  H2b (DINO-WM family, confirmatory cells, points = (task, k, non-none variant)): primary = task average of
  [Spearman(-OEI, D) - max(Spearman(1-step, D), Spearman(3-step, D))], each Spearman within a task (planning-time
  errors); holds if >= 0.15. Auxiliary: pooled raw Spearman over all points; pooled with OEI divided by the task's
  none-variant iteration-0 candidate cost std (--it0std JSON from it0_cost_std.py); dataset-type errors
  (results/E0_dinowm/err_<task>_main.json, --errdir) in place of planning-time errors.
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A

CONF = {'pusht': [3, 4, 5], 'wall': [4, 5, 6], 'pointmaze': [2, 3, 4]}
EXPL = {'wall': [3]}
VARS = ['det', 'ind', 'shiter', 'shstep', 'detshift']
NB = 10000


def boot_p_le0(d, seed=0):
    """one-sided bootstrap p for mean(d) <= 0 (episode resampling)."""
    rng = np.random.RandomState(seed)
    n = len(d)
    idx = rng.randint(0, n, size=(NB, n))
    m = d[idx].mean(1)
    return float((np.sum(m <= 0) + 1) / (NB + 1))


def holm(pvals, alpha):
    """Holm step-down; returns list of reject flags in the input order."""
    order = np.argsort(pvals)
    m = len(pvals)
    rej = [False] * m
    for r, i in enumerate(order):
        if pvals[i] <= alpha / (m - r):
            rej[i] = True
        else:
            break
    return rej


def paired(R, task, var, k, key_fn):
    eps = sorted(e for (t, v, kk, e) in R if t == task and v == var and kk == k and (t, 'none', 0, e) in R)
    return eps, np.array([key_fn(R[(task, var, k, e)]) for e in eps], float)


def task_tables(R, task, ks):
    cells = {}
    for k in ks:
        C = {}
        for v in VARS:
            c = A.cell_stats(R, task, v, k)
            if c is not None:
                C[v] = c
        if 'det' not in C:
            continue
        # red_v and D_v - D_det (paired on the common episodes), as analyze.cells
        for v, c in C.items():
            if v == 'det':
                continue
            eps = [e for e in range(c['eps'][0], c['eps'][1] + 1)
                   if (task, v, k, e) in R and (task, 'det', k, e) in R and (task, 'none', 0, e) in R]
            Sn = np.array([R[(task, 'none', 0, e)]['success'] for e in eps], float)
            Sd = np.array([R[(task, 'det', k, e)]['success'] for e in eps], float)
            Sv = np.array([R[(task, v, k, e)]['success'] for e in eps], float)

            def red(i):
                dd = (Sn - Sd)[i].mean(); dv = (Sn - Sv)[i].mean()
                return (dd - dv) / dd if dd > 0 else np.nan
            r0 = red(np.arange(len(eps)))
            c['red'] = None if not np.isfinite(r0) else float(r0)
            c['red_ci'] = A.boot(red, len(eps))
            dvd = (Sd - Sv)            # = D_v - D_det
            c['Dv_minus_Ddet'] = float(dvd.mean()); c['Dv_minus_Ddet_ci'] = A.boot(lambda i: dvd[i].mean(), len(eps))
            n10, n01, p = A.mcnemar(Sv.astype(bool), Sd.astype(bool))
            c['vs_det_mcnemar'] = dict(var_only=n10, det_only=n01, p=p)
        # det drop: one-sided bootstrap p
        eps = sorted(e for (t, v, kk, e) in R if t == task and v == 'det' and kk == k and (t, 'none', 0, e) in R)
        d = np.array([float(R[(task, 'none', 0, e)]['success']) - float(R[(task, 'det', k, e)]['success']) for e in eps])
        C['det']['p_boot_drop'] = boot_p_le0(d)
        # planning-time 1-step error det vs sh-iter (paired per episode)
        if 'shiter' in C and 'rel_err1' in C['det']:
            common = [e for e in eps if (task, 'shiter', k, e) in R]
            dd = np.array([R[(task, 'det', k, e)]['err']['1']['rel'] - R[(task, 'shiter', k, e)]['err']['1']['rel'] for e in common])
            C['det']['err1_det_minus_shiter'] = float(dd.mean())
            C['det']['err1_det_minus_shiter_ci'] = A.boot(lambda i: dd[i].mean(), len(dd))
        cells[k] = C
    return cells


def spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(x, y).correlation
    return float(r) if np.isfinite(r) else float('nan')


def h2(out, it0std=None, errdir=None):
    """H2a and H2b of DESIGN_main v4 from the E1 cells (see module docstring)."""
    res = {}
    T = out['tasks'].get('pusht')
    if T and 4 in T['confirmatory']:
        C = T['confirmatory'][4]
        det, sh = C.get('det', {}), C.get('shiter', {})
        if 'oei_ref_mf_ci' in det and 'oei_ref_mf_ci' in sh:
            res['H2a'] = dict(cell='pusht k=4', det_oei=det['oei_ref_mf'], det_oei_ci=det['oei_ref_mf_ci'],
                              shiter_oei=sh['oei_ref_mf'], shiter_oei_ci=sh['oei_ref_mf_ci'],
                              det_upper_lt0=det['oei_ref_mf_ci'][1] < 0,
                              shiter_ci_contains0=sh['oei_ref_mf_ci'][0] <= 0 <= sh['oei_ref_mf_ci'][1])
            res['H2a']['holds'] = bool(res['H2a']['det_upper_lt0'] and res['H2a']['shiter_ci_contains0'])
    pts = []
    for t, T in out['tasks'].items():
        dserr = {}
        if errdir and os.path.exists(f'{errdir}/err_{t}_main.json'):
            dserr = json.load(open(f'{errdir}/err_{t}_main.json')).get('ds', {})
        for k, C in T['confirmatory'].items():
            for v, c in C.items():
                if 'oei_ref_mf' not in c or 'rel_err1' not in c:
                    continue
                d = dserr.get(f'{v}{k}', {})
                pts.append(dict(task=t, k=int(k), var=v, D=c['D'], oei=c['oei_ref_mf'], err1=c['rel_err1'], err3=c['rel_err3'],
                                ds_err1=d.get('rel_err1'), ds_err3=d.get('rel_err3'),
                                oei_norm=(c['oei_ref_mf'] / it0std[t]) if it0std and t in it0std else None))
    res['H2b_points'] = pts

    def contrast(P, e1='err1', e3='err3', o='oei'):
        D = [p['D'] for p in P]
        r_o = spearman([-p[o] for p in P], D)
        r_1 = spearman([p[e1] for p in P], D); r_3 = spearman([p[e3] for p in P], D)
        return dict(rho_negOEI_D=r_o, rho_err1_D=r_1, rho_err3_D=r_3, diff=r_o - max(r_1, r_3), n=len(P))
    per_task = {t: contrast([p for p in pts if p['task'] == t]) for t in sorted({p['task'] for p in pts})}
    if per_task:
        prim = float(np.mean([v['diff'] for v in per_task.values()]))
        res['H2b'] = dict(per_task=per_task, primary_task_average=prim, holds=bool(prim >= 0.15),
                          aux_pooled_raw=contrast(pts))
        if all(p['oei_norm'] is not None for p in pts):
            res['H2b']['aux_pooled_oei_norm'] = contrast(pts, o='oei_norm')
        if all(p['ds_err1'] is not None and p['ds_err3'] is not None for p in pts):
            res['H2b']['aux_dataset_errors_per_task'] = {
                t: contrast([p for p in pts if p['task'] == t], e1='ds_err1', e3='ds_err3') for t in per_task}
            res['H2b']['aux_dataset_errors_task_average'] = float(np.mean(
                [v['diff'] for v in res['H2b']['aux_dataset_errors_per_task'].values()]))
            res['H2b']['aux_dataset_errors_pooled'] = contrast(pts, e1='ds_err1', e3='ds_err3')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('res'); ap.add_argument('--out', default=None); ap.add_argument('--csv', default=None)
    ap.add_argument('--it0std', default=None, help='JSON {task: none iteration-0 candidate cost std} (it0_cost_std.py)')
    ap.add_argument('--errdir', default=None, help='dir with err_<task>_main.json (dataset-type errors)')
    a = ap.parse_args()
    R = {}
    for sub in ('E1_dinowm_wp', 'E1_dinowm_pusht'):
        if os.path.isdir(f'{a.res}/{sub}'):
            r, _ = A.load(f'{a.res}/{sub}')
            R.update(r)
    out = dict(tasks={}, families={})
    qualifying, conf_cells = [], []
    for task in ('pusht', 'wall', 'pointmaze'):
        if not any(t == task for (t, v, k, e) in R):
            continue
        T = dict(none=A.cell_stats(R, task, 'none', 0), floor=A.seed_floor(R, task),
                 confirmatory=task_tables(R, task, CONF[task]), exploratory=task_tables(R, task, EXPL.get(task, [])))
        for C in list(T['confirmatory'].values()) + list(T['exploratory'].values()):
            if T['floor']:
                for c in C.values():
                    if 'plan_dev_first' in c:
                        c['plan_dev_first_norm'] = c['plan_dev_first'] / T['floor']['floor_first']
        ks = sorted(T['confirmatory'])
        ps = [T['confirmatory'][k]['det']['p_boot_drop'] for k in ks]
        rej = holm(ps, 0.025) if ps else []
        for k, rj in zip(ks, rej):
            C = T['confirmatory'][k]
            det = C['det']
            det['drop_real_holm'] = bool(rj and det['D'] > 0)
            det['drop_real_ci95'] = bool(det['D_ci'][0] > 0)
            cell = dict(task=task, k=k, n=det['n'], D_det=det['D'], D_det_ci=det['D_ci'], p_boot=det['p_boot_drop'],
                        drop_real_holm=det['drop_real_holm'], drop_real_ci95=det['drop_real_ci95'],
                        err1_det=det.get('rel_err1'), err1_shiter=C.get('shiter', {}).get('rel_err1'),
                        red_shiter=C.get('shiter', {}).get('red'),
                        Dsh_minus_Ddet_ci=C.get('shiter', {}).get('Dv_minus_Ddet_ci'))
            conf_cells.append(cell)
            if cell['drop_real_holm']:
                qualifying.append(cell)
        out['tasks'][task] = T
    # family verdicts (DINO-WM)
    h1_cells = [c for c in qualifying if c['err1_det'] is not None and c['err1_shiter'] is not None and c['err1_det'] <= c['err1_shiter']]
    red_ok = [c for c in qualifying if c['red_shiter'] is not None and c['red_shiter'] >= 0.5]
    harm = [c for c in conf_cells if c['Dsh_minus_Ddet_ci'] is not None and c['Dsh_minus_Ddet_ci'][0] > 0.05]
    out['families']['dinowm'] = dict(
        confirmatory_cells=conf_cells, qualifying_cells=[(c['task'], c['k']) for c in qualifying],
        H1=dict(holds=len(h1_cells) > 0, cells_meeting=[(c['task'], c['k']) for c in h1_cells],
                qualifying_any=len(qualifying) > 0),
        H3a=dict(holds=(len(qualifying) > 0 and len(red_ok) / len(qualifying) >= 0.75 and not harm),
                 share_red_ge_half=(len(red_ok) / len(qualifying)) if qualifying else None,
                 n_qualifying=len(qualifying), cells_red_ge_half=[(c['task'], c['k']) for c in red_ok],
                 harm_cells=[(c['task'], c['k']) for c in harm]),
        n_confirmatory_complete=sum(1 for c in conf_cells if c['n'] == 100))
    it0std = json.load(open(a.it0std)) if a.it0std else None
    out['families']['dinowm'].update(h2(out, it0std=it0std, errdir=a.errdir))
    print_summary(out)
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1, default=float)
        print('wrote', a.out)
    if a.csv:
        write_csv(out, a.csv)
        print('wrote', a.csv)


def write_csv(out, path):
    keys = ['task', 'role', 'k', 'var', 'n', 's', 'success', 'D', 'D_ci', 'none_only', 'var_only', 'mcnemar_p', 'p_boot_drop',
            'drop_real_holm', 'red', 'red_ci', 'Dv_minus_Ddet', 'Dv_minus_Ddet_ci', 'rel_err1', 'rel_err3', 'rel_err5',
            'plan_dev_first_norm', 'oei_ref_mf', 'oei_ref_mf_ci', 'oei_ref_argmin', 'oei_ref_argmin_ci', 'oei_ref_elite',
            'oei_true_mf', 'oei_true_argmin', 'it0_rand_dJ', 'last_argmin_dJ', 'mf_dJ', 'rho_last', 'rho_last_Jnone',
            'd_rho_last', 'margin_median', 'state_dist_mean', 'time_s']

    def fmt(x):
        if isinstance(x, (list, tuple)):
            return '[' + ';'.join(fmt(y) for y in x) + ']'
        if isinstance(x, float):
            return f'{x:.5g}'
        return '' if x is None else str(x)
    with open(path, 'w') as f:
        f.write(','.join(keys) + '\n')
        for t, T in out['tasks'].items():
            rows = [dict(T['none'], role='baseline')]
            for role in ('confirmatory', 'exploratory'):
                for k, C in T[role].items():
                    for v, c in C.items():
                        rows.append(dict(c, role=role))
            for c in rows:
                f.write(','.join(fmt(c.get(kk)) for kk in keys) + '\n')


def print_summary(out):
    for t, T in out['tasks'].items():
        nn = T['none']
        fl = T['floor']
        print(f"== {t}: none {nn['s']}/{nn['n']} ({100 * nn['success']:.0f}%)" +
              (f"; nonealt {100 * fl['success_nonealt']:.0f}%, seed floor {fl['floor_first']:.3f}" if fl else ''))
        for role in ('confirmatory', 'exploratory'):
            for k, C in T[role].items():
                for v, c in C.items():
                    s = (f"  [{role[:4]}] k={k} {v:8s} {c['s']:3d}/{c['n']} D {100 * c['D']:+.0f}pp "
                         f"[{100 * c['D_ci'][0]:.0f},{100 * c['D_ci'][1]:.0f}] McN p={c['mcnemar_p']:.3g}")
                    if v == 'det':
                        s += f" p_boot {c['p_boot_drop']:.4f} real(Holm) {c.get('drop_real_holm')}"
                    else:
                        s += f" red {c['red'] if c.get('red') is None else round(c['red'], 2)} D_v-D_det {100 * c['Dv_minus_Ddet']:+.0f} " \
                             f"[{100 * c['Dv_minus_Ddet_ci'][0]:.0f},{100 * c['Dv_minus_Ddet_ci'][1]:.0f}]"
                    if 'rel_err1' in c:
                        s += f" | err1 {c['rel_err1']:.4f} err5 {c['rel_err5']:.4f}"
                    if 'oei_ref_mf' in c:
                        s += f" | OEI_ref(mf) {c['oei_ref_mf']:+.3f} (argmin {c.get('oei_ref_argmin', float('nan')):+.3f})"
                    if 'plan_dev_first_norm' in c:
                        s += f" | plandev {c['plan_dev_first_norm']:.2f}"
                    print(s)
    F = out['families'].get('dinowm', {})
    print('DINO-WM H1:', F.get('H1'))
    print('DINO-WM H3a:', F.get('H3a'))
    print('DINO-WM H2a:', F.get('H2a'))
    if 'H2b' in F:
        print('DINO-WM H2b primary (task average):', round(F['H2b']['primary_task_average'], 3), 'holds', F['H2b']['holds'],
              {t: round(v['diff'], 3) for t, v in F['H2b']['per_task'].items()})


if __name__ == '__main__':
    main()
