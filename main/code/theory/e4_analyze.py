"""E4 analysis (paper A): pool the job outputs of e4_jobs.py, run the pre-declared theory-vs-simulation comparisons
C1-C12 (worklogs/E4_theory.md, 07:24Z), write tidy tables, a summary JSON and small figures. Server only.

usage: python e4_analyze.py <res_dir> <out_dir>
"""
import collections
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e4_formulas as F  # noqa: E402

PERSISTENT_BASE = {'M1', 'M1j', 'Mstep', 'R-det', 'R-fixed', 'E-det', 'RV-det', 'RV-fixed', 'NL-det', 'NL-fixed'}
FRESH_BASE = {'M2', 'M2marg', 'M3', 'M3rho', 'R-shiter', 'R-detshift', 'R-ind', 'E-ind', 'E-shiter', 'E-detshift', 'RV-ind',
              'RV-shiter', 'RV-detshift', 'NL-ind', 'NL-shiter', 'NL-detshift'}


def family(model):
    """'persistent' (error fixed within a plan: det, fixed tables, fixed fields) or 'fresh' (new draw per evaluation)."""
    if model == 'none':
        return 'none'
    base = model.split('@')[0]
    if base in PERSISTENT_BASE or base.startswith('E-fixed'):
        return 'persistent'
    if base in FRESH_BASE:
        return 'fresh'
    return 'other'


def pool(stats):
    """Pool chunk statistics {mean, sd, n} (scalars or equal-length vectors)."""
    stats = [s for s in stats if s and s.get('n', 0) > 0]
    if not stats:
        return None
    n = np.array([s['n'] for s in stats], float)
    m = [np.asarray(s['mean'], float) for s in stats]
    sd = [np.asarray(s['sd'], float) for s in stats]
    N = n.sum()
    M = sum(ni * mi for ni, mi in zip(n, m)) / N
    ss = sum((ni - 1) * si ** 2 + ni * (mi - M) ** 2 for ni, mi, si in zip(n, m, sd))
    V = ss / max(N - 1, 1)
    out = dict(mean=M.tolist() if M.ndim else float(M), sd=np.sqrt(V).tolist() if M.ndim else float(np.sqrt(V)),
               n=int(N))
    if not M.ndim:
        out['se'] = float(np.sqrt(V / N))
    return out


def load_jobs(res):
    jobs = {}
    for f in sorted(glob.glob(os.path.join(res, 'jobs', '*.json'))):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        jobs[j['job']['id']] = j
    return jobs


def planner_rows(jobs):
    groups = collections.defaultdict(list)
    for jid, j in jobs.items():
        jb = j['job']
        if jb['block'] in ('A', 'W', 'EV', 'JV'):
            continue
        key = (jb['block'], jb['planner'], jb['regime'], jb['d'], jb['N'], jb['rule'], jb['k'], jb.get('tag', 'x'))
        groups[key].append(j)
    rows = []
    for key, js in groups.items():
        cks = set()
        for j in js:
            cks |= set(j['cells'].keys())
        for ck in cks:
            model, scale, n = ck.split('|')
            names = set()
            for j in js:
                names |= set(j['cells'].get(ck, {}).keys())
            met = {nm: pool([j['cells'][ck].get(nm) for j in js if ck in j['cells']]) for nm in names}
            rows.append(dict(block=key[0], planner=key[1], regime=key[2], d=key[3], N=key[4], rule=key[5], k=key[6],
                             tag=key[7], model=model, scale=float(scale), n=int(n), m=met, chunks=len(js),
                             family=family(model)))
    return rows


def sdiv(a, b):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return np.nan
    return a / b if (np.isfinite(a) and np.isfinite(b) and b != 0) else np.nan


def g(row, name, field='mean'):
    s = row['m'].get(name)
    if s is None:
        return np.nan
    v = s.get(field)
    return np.nan if v is None else v


def index_rows(rows):
    ix = {}
    for r in rows:
        ix[(r['block'], r['planner'], r['regime'], r['d'], r['N'], r['rule'], r['k'], r['tag'], r['model'], r['scale'],
            r['n'])] = r
    return ix


def gkey(r):
    return (r['block'], r['planner'], r['regime'], r['d'], r['N'], r['rule'], r['k'], r['tag'])


def add_derived(rows):
    ix = index_rows(rows)
    nmax = collections.defaultdict(int)
    for r in rows:
        nmax[gkey(r) + (r['model'], r['scale'])] = max(nmax[gkey(r) + (r['model'], r['scale'])], r['n'])
    for r in rows:
        r['is_last'] = r['n'] == nmax[gkey(r) + (r['model'], r['scale'])]
    for r in rows:
        if r['model'] == 'none' or r['n'] == 0:
            continue
        nb = ix.get(gkey(r) + ('none', 0.0, r['n']))
        fl = g(nb, 'floor_a1') if nb else np.nan
        flm = g(nb, 'floor_mean') if nb else np.nan
        r['dev_ratio'] = math.sqrt(max(sdiv(g(r, 'dev_a1'), fl), 0)) if np.isfinite(sdiv(g(r, 'dev_a1'), fl)) else np.nan
        r['spread_ratio'] = math.sqrt(max(sdiv(g(r, 'spread_a1'), fl), 0)) if np.isfinite(sdiv(g(r, 'spread_a1'), fl)) else np.nan
        r['devmean_ratio'] = math.sqrt(max(sdiv(g(r, 'dev_mean'), flm), 0)) if np.isfinite(sdiv(g(r, 'dev_mean'), flm)) else np.nan
        r['persistence'] = sdiv(g(r, 'pers_dot'), math.sqrt(max(g(r, 'pers_n0'), 0) * max(g(r, 'pers_n1'), 0)))
        if 'saa_dot' in r['m']:
            r['saa_slope'] = sdiv(g(r, 'saa_dot'), g(r, 'saa_pp'))
            r['saa_r2'] = 1 - sdiv(g(r, 'saa_res'), g(r, 'saa_vv'))
        it0 = ix.get(gkey(r) + (r['model'], r['scale'], 0))
        if it0:
            r['se0'] = g(it0, 'se')
            r['sJ0'] = g(it0, 'sJ')
            r['gap0'] = g(it0, 'gap')


# ------------------------------------------------------------------------------------------------ fit statistics
def fit_stats(meas, pred, se=None):
    """Fit of measured cell means to predictions. POST-HOC numerical-validity filter (added 08:58Z after the full run):
    cells whose prediction is non-finite or degenerate (|pred| > 1e3 (|meas| + 2 se + 1e-12): a CEM population collapsed
    onto the SAA fixed point where rho -> -1 and the model cost is flat to first order, or tied candidates in the Stein
    k-NN density) are excluded and counted in n_degenerate."""
    meas, pred = np.asarray(meas, float), np.asarray(pred, float)
    ok = np.isfinite(meas) & np.isfinite(pred)
    if se is not None:
        se = np.asarray(se, float)
        ok &= np.isfinite(se)
        bound = 1e3 * (np.abs(meas) + 2 * np.where(np.isfinite(se), se, 0) + 1e-12)
    else:
        bound = 1e3 * (np.abs(meas) + 1e-12)
    degen = ok & (np.abs(pred) > bound)
    n_degen = int(degen.sum())
    ok &= ~degen
    meas, pred = meas[ok], pred[ok]
    if len(meas) < 2:
        return dict(n=int(len(meas)), n_degenerate=n_degen)
    slope = float((meas * pred).sum() / max((pred * pred).sum(), 1e-300))
    ss_res = float(((meas - pred) ** 2).sum())
    ss_tot = float(((meas - meas.mean()) ** 2).sum())
    rel = np.abs(meas - pred) / np.maximum(np.abs(meas), 1e-12)
    out = dict(n=int(len(meas)), n_degenerate=n_degen, slope_through_origin=slope, R2=1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
               median_rel_err=float(np.median(rel)), q90_rel_err=float(np.quantile(rel, 0.9)),
               corr=float(np.corrcoef(meas, pred)[0, 1]) if meas.std() > 0 and pred.std() > 0 else np.nan)
    if se is not None:
        se = se[ok] if len(se) == len(ok) else se
        z = np.abs(meas - pred) / np.maximum(se, 1e-300)
        out['frac_within_2se'] = float(np.mean(z < 2))
        sig = np.abs(meas) > 2 * se
        if sig.any():
            out['median_rel_err_signif'] = float(np.median(rel[sig]))
            out['n_signif'] = int(sig.sum())
    return out


def spearman(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    r = spearmanr(x[ok], y[ok]).correlation
    return float(r) if np.isfinite(r) else np.nan


# ------------------------------------------------------------------------------------------------ comparisons
def c1_singleshot(jobs):
    A = jobs.get('A_singleshot')
    if not A:
        return None
    rows = A['rows']
    out = {}
    nr = [r for r in rows if r['dist'] == 'normal']
    out['normal'] = fit_stats([r['meas'] for r in nr], [r['pred'] for r in nr], [r['meas_se'] for r in nr])
    by_op = collections.defaultdict(list)
    for r in nr:
        by_op[r['op'].split('_')[0]].append(r)
    out['normal_by_op'] = {k: fit_stats([r['meas'] for r in v], [r['pred'] for r in v], [r['meas_se'] for r in v])
                           for k, v in by_op.items()}
    qr = [r for r in rows if r['dist'].startswith('quad')]
    out['quad_bivariate'] = {op: fit_stats([r['meas'] for r in qr if r['op'] == op], [r['pred'] for r in qr if r['op'] == op],
                                           [r['meas_se'] for r in qr if r['op'] == op]) for op in ('argmin', 'top0.125')}
    qa = [r for r in qr if r['op'] == 'argmin' and 'pred_cond_exact' in r]
    out['quad_argmin_refined'] = dict(
        cond_exact=fit_stats([r['meas_sub'] for r in qa], [r['pred_cond_exact'] for r in qa], [r['meas_sub_se'] for r in qa]),
        near_tie=fit_stats([r['meas_sub'] for r in qa], [r['pred_near_tie'] for r in qa], [r['meas_sub_se'] for r in qa]),
        bivariate=fit_stats([r['meas'] for r in qa], [r['pred'] for r in qa], [r['meas_se'] for r in qa]))
    out['quad_table'] = [dict(dist=r['dist'], ratio=r['ratio'], op=r['op'], meas=r['meas'], pred=r['pred'],
                              cond_exact=r.get('pred_cond_exact'), near_tie=r.get('pred_near_tie'),
                              meas_sub=r.get('meas_sub')) for r in qr]
    return out


def c2_selection(rows):
    out = {}
    sel = collections.defaultdict(lambda: ([], [], [], []))
    for r in rows:
        if r['model'] == 'none' or r['n'] == 0:
            continue
        fam = r['family']
        if r['planner'] == 'mppi':
            pairs = (('mppi_same_draw', 'sel_last', 'wc_pred'), ('mppi_stein', 'sel_last', 'stein_pred'))
        else:
            pairs = (('cem_elite', 'sel_last_elite', 'wc_pred_elite'), ('cem_elite_stein', 'sel_last_elite', 'stein_pred_elite'),
                     ('cem_argmin', 'sel_last_argmin', 'wc_pred_argmin'), ('cem_argmin_ntc', 'sel_last_argmin', 'ntc_pred_argmin'))
        stage = 'first' if r['n'] in (1, 5) else ('last' if r['is_last'] else 'mid')
        for name, mk, pk in pairs:
            m, p, s = g(r, mk), g(r, pk), g(r, mk, 'se')
            for key in (f'{name}|all', f'{name}|{fam}', f'{name}|{fam}|{stage}'):
                L = sel[key]
                L[0].append(m)
                L[1].append(p)
                L[2].append(s)
    for key, (m, p, s, _) in sel.items():
        out[key] = fit_stats(m, p, s)
    return out


def c3_decomposition(rows):
    out = {}
    agg = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        if r['model'] == 'none' or r['n'] == 0 or r['block'] == 'J':
            continue
        fam = r['family']
        pl = r['planner']
        dm, dse = g(r, 'drift'), g(r, 'drift', 'se')
        if not np.isfinite(dm):
            continue
        c = agg[f'{pl}|{fam}']
        c['cells'] += 1
        c['drift_ci_covers_0'] += int(abs(dm) <= 1.96 * dse)
        c['drift_ci_below_0'] += int(dm + 1.96 * dse < 0)
        rd = g(r, 'sel_last_rd') if pl == 'mppi' else g(r, 'sel_last_mf')
        rdse = g(r, 'sel_last_rd', 'se') if pl == 'mppi' else g(r, 'sel_last_mf', 'se')
        if np.isfinite(rd):
            c['redraw_ci_covers_0'] += int(abs(rd) <= 1.96 * rdse)
            c['redraw_cells'] += 1
        sd = g(r, 'sel_last') if pl == 'mppi' else g(r, 'sel_last_elite')
        sdse = g(r, 'sel_last', 'se') if pl == 'mppi' else g(r, 'sel_last_elite', 'se')
        if np.isfinite(sd):
            c['samedraw_ci_below_0'] += int(sd + 1.96 * sdse < 0)
        if fam == 'persistent' and pl == 'mppi':
            c['redraw_equals_samedraw'] += int(abs(g(r, 'sel_last_rd') - g(r, 'sel_last')) < 1e-9 * (1 + abs(g(r, 'sel_last'))))
    for k, c in agg.items():
        n = max(c['cells'], 1)
        out[k] = dict(cells=c['cells'], frac_drift_ci_covers_0=c['drift_ci_covers_0'] / n,
                      frac_drift_ci_below_0=c['drift_ci_below_0'] / n,
                      frac_redraw_sel_ci_covers_0=c['redraw_ci_covers_0'] / max(c['redraw_cells'], 1),
                      frac_samedraw_sel_ci_below_0=c['samedraw_ci_below_0'] / n,
                      frac_redraw_equals_samedraw=(c['redraw_equals_samedraw'] / n) if 'mppi|persistent' == k else None)
    return out


def c4_saa(rows):
    out = {}
    for pl in ('mppi', 'cem'):
        rr = [r for r in rows if r['planner'] == pl and r['model'].startswith('M1') and r['model'] not in ('M1j',)
              and r['n'] > 0 and 'saa_slope' in r]
        if not rr:
            continue
        for sub, cond in (('all', lambda r: True), ('last_n', lambda r: r['is_last'])):
            sel = [r for r in rr if cond(r)]
            o = dict(n_cells=len(sel),
                     median_shift_slope=float(np.nanmedian([r['saa_slope'] for r in sel])),
                     median_shift_R2=float(np.nanmedian([r['saa_r2'] for r in sel])))
            o['drift_vs_pred'] = fit_stats([g(r, 'drift') for r in sel], [g(r, 'saa_pred_drift') for r in sel],
                                           [g(r, 'drift', 'se') for r in sel])
            if pl == 'cem':
                o['oei_mf_vs_pred'] = fit_stats([g(r, 'oei_mf') for r in sel], [g(r, 'saa_pred_oei_mf') for r in sel],
                                                [g(r, 'oei_mf', 'se') for r in sel])
            # by error scale (weak vs strong field)
            o['by_scale'] = {}
            for sc in sorted({r['scale'] for r in sel}):
                s2 = [r for r in sel if r['scale'] == sc]
                o['by_scale'][f'{sc:g}'] = dict(shift_slope=float(np.nanmedian([r['saa_slope'] for r in s2])),
                                                shift_R2=float(np.nanmedian([r['saa_r2'] for r in s2])), n=len(s2))
            out[f'{pl}|{sub}'] = o
    return out


def c5_stationary(rows, tabs):
    out = []
    for r in rows:
        if r['block'] != 'V' or r['planner'] != 'mppi' or r['model'] != 'M2' or r['n'] != 100:
            continue
        se = g(r, 'se')
        S = tabs.mppi(r['N'], r['k'], g(r, 'theta'))
        keff = g(r, 'keff')
        pred = 6 * se / (4 * S * keff)
        out.append(dict(k=r['k'], scale=r['scale'], dmg_mean=g(r, 'dmg_mean'), dmg_mean_se=g(r, 'dmg_mean', 'se'),
                        J_mean=g(r, 'J_mean'), pred_regret_mean=pred, keff=keff, S=S, sigma_eps=se))
    res = dict(rows=out)
    if out:
        res['fit'] = fit_stats([o['dmg_mean'] for o in out], [o['pred_regret_mean'] for o in out],
                               [o['dmg_mean_se'] for o in out])
        for sc in sorted({o['scale'] for o in out}):
            oo = sorted([o for o in out if o['scale'] == sc], key=lambda o: o['keff'])
            x = np.log([o['keff'] for o in oo])
            y = np.log(np.maximum([o['dmg_mean'] for o in oo], 1e-12))
            res[f'loglog_slope_dmg_vs_keff|{sc:g}'] = float(np.polyfit(x, y, 1)[0]) if len(oo) > 1 else np.nan
    return res


def c6_iterations(rows, jobs):
    out = {}
    grp = collections.defaultdict(dict)
    for r in rows:
        if r['model'] == 'none' or r['n'] == 0 or r['block'] not in ('M', 'D'):
            continue
        key = gkey(r) + (r['model'], r['scale'])
        dm = g(r, 'dmg_exec') if r['planner'] == 'mppi' else g(r, 'dmg')
        grp[key][r['n']] = dm
    cnt = collections.defaultdict(collections.Counter)
    for key, d in grp.items():
        ns = sorted(d)
        if len(ns) < 2:
            continue
        model = key[8]
        fam = family(model)
        pl = key[1]
        first, last = d[ns[0]], d[ns[-1]]
        c = cnt[f'{pl}|{fam}']
        c['groups'] += 1
        c['dmg_decreases'] += int(last < first)
        if first > 0:
            c['ratio_sum'] += last / first
            c['ratio_n'] += 1
    for k, c in cnt.items():
        out[k] = dict(groups=c['groups'], frac_dmg_decreases_first_to_last=c['dmg_decreases'] / max(c['groups'], 1),
                      mean_ratio_last_over_first=c['ratio_sum'] / max(c['ratio_n'], 1))
    # closed loop
    W = {}
    for jid, j in jobs.items():
        if j['job']['block'] != 'W':
            continue
        ws, n_it = j['job']['ws'], j['job']['n_it']
        rets = j.get('rets', {})
        base = np.array(rets.get('none|0', []), float)
        for key, v in rets.items():
            if key == 'none|0' or len(v) != len(base):
                continue
            v = np.array(v, float)
            rel = (v - base) / base
            W[f'ws{ws}|n{n_it}|{key}'] = dict(rel_damage=float(rel.mean()), se=float(rel.std() / math.sqrt(len(rel))),
                                              n=len(rel))
        for sc in j['job']['scales']:
            a, b = rets.get(f'Mstep|{sc:g}'), rets.get(f'M3|{sc:g}')
            if a and b and len(a) == len(base):
                dd = (np.array(a) - np.array(b)) / base
                W[f'ws{ws}|n{n_it}|Mstep_minus_M3|{sc:g}'] = dict(rel_diff=float(dd.mean()), se=float(dd.std() / math.sqrt(len(dd))))
    out['closed_loop'] = W
    curves = {}
    for r in rows:
        if r['block'] != 'M' or r['rule'] != 'q125' or r['model'] == 'none' or r['n'] == 0:
            continue
        if r['N'] != (512 if r['planner'] == 'mppi' else 256):
            continue
        key = f"{r['planner']}|{r['regime']}|{r['model']}|{r['scale']:g}"
        curves.setdefault(key, {})[r['n']] = (g(r, 'dmg_exec' if r['planner'] == 'mppi' else 'dmg'),
                                              g(r, 'oei' if r['planner'] == 'mppi' else 'oei_mf'), g(r, 'drift'),
                                              g(r, 'sel_last' if r['planner'] == 'mppi' else 'sel_last_elite'))
    out_curves = {k: dict(n=sorted(v), dmg=[v[n][0] for n in sorted(v)], oei=[v[n][1] for n in sorted(v)],
                          drift=[v[n][2] for n in sorted(v)], sel=[v[n][3] for n in sorted(v)]) for k, v in curves.items()}
    return out, out_curves


def c7_dose(rows, tabs):
    """H2c: |OEI| and damage vs N at fixed elite rule (block M), last recorded iteration."""
    out = {}
    curves = {}
    grp = collections.defaultdict(dict)
    for r in rows:
        if r['block'] != 'M' or r['model'] == 'none':
            continue
        nlast = 12 if r['planner'] == 'mppi' else 30
        if r['n'] != nlast:
            continue
        key = (r['planner'], r['regime'], r['rule'], r['model'], r['scale'])
        oei = g(r, 'oei') if r['planner'] == 'mppi' else g(r, 'oei_mf')
        dmg = g(r, 'dmg_exec') if r['planner'] == 'mppi' else g(r, 'dmg')
        grp[key][r['N']] = dict(oei=oei, dmg=dmg, sel=g(r, 'sel_last') if r['planner'] == 'mppi' else g(r, 'sel_last_elite'),
                                drift=g(r, 'drift'), k=r['k'], theta=g(r, 'theta'))
    summ = collections.defaultdict(list)
    for key, d in grp.items():
        Ns = sorted(d)
        if len(Ns) < 4:
            continue
        absoei = [abs(d[N]['oei']) for N in Ns]
        dmg = [d[N]['dmg'] for N in Ns]
        rec = dict(N=Ns, abs_oei=absoei, dmg=dmg, sel=[d[N]['sel'] for N in Ns], drift=[d[N]['drift'] for N in Ns],
                   k=[d[N]['k'] for N in Ns], spearman_N_absoei=spearman(Ns, absoei), spearman_N_dmg=spearman(Ns, dmg),
                   spearman_dmg_absoei=spearman(dmg, absoei),
                   ratio_absoei_1024_64=(absoei[-1] / absoei[0]) if absoei[0] > 0 else np.nan)
        # selection-intensity prediction of the N dependence (operator only)
        pl, rule = key[0], key[2]
        if pl == 'mppi':
            th = [d[N]['theta'] for N in Ns]
            rec['S_ratio_pred'] = [tabs.mppi(N, d[N]['k'], t) / tabs.mppi(Ns[0], d[Ns[0]]['k'], th[0]) for N, t in zip(Ns, th)]
        else:
            rec['S_ratio_pred'] = [tabs.topk(N, d[N]['k']) / tabs.topk(Ns[0], d[Ns[0]]['k']) for N in Ns]
        curves['|'.join(map(str, key))] = rec
        summ[(pl, rule, family(key[3]))].append(rec)
    for (pl, rule, fam), recs in summ.items():
        sp = [r['spearman_N_absoei'] for r in recs]
        sd = [r['spearman_N_dmg'] for r in recs]
        sda = [r['spearman_dmg_absoei'] for r in recs]
        out[f'{pl}|{rule}|{fam}'] = dict(groups=len(recs), median_spearman_N_absoei=float(np.nanmedian(sp)),
                                         frac_spearman_N_absoei_ge_0p8=float(np.nanmean(np.array(sp) >= 0.8)),
                                         median_spearman_N_dmg=float(np.nanmedian(sd)),
                                         frac_spearman_dmg_absoei_ge_0p8=float(np.nanmean(np.array(sda) >= 0.8)),
                                         median_ratio_absoei_1024_64=float(np.nanmedian([r['ratio_absoei_1024_64'] for r in recs])))
    return out, curves


def c8_crn(rows):
    out = {}
    for r in rows:
        if r['block'] != 'C' or r['n'] == 0 or r['model'] == 'none':
            continue
        nlast = 12 if r['planner'] == 'mppi' else 30
        if r['n'] != nlast:
            continue
        key = f"{r['planner']}|{r['regime']}|{r['scale']:g}"
        rc = 1.0 if r['model'] == 'M2marg' else float(r['model'].split('@')[1])
        sel = g(r, 'sel_last') if r['planner'] == 'mppi' else g(r, 'sel_last_elite')
        dmg = g(r, 'dmg_exec') if r['planner'] == 'mppi' else g(r, 'dmg')
        out.setdefault(key, []).append(dict(model=r['model'], rho_c=rc, sel=sel, dmg=dmg, oei=g(r, 'oei') if r['planner'] == 'mppi' else g(r, 'oei_mf'),
                                            se_last=g(r, 'se'), se0=r.get('se0')))
    return out


def c9_neff(jobs, rows):
    out = {}
    ev_rows = []
    for coh in ('coh', 'incoh'):
        j = jobs.get(f'EV_{coh}')
        if not j:
            continue
        ev_rows += j['rows']
    if ev_rows:
        meas = np.array([r['var_meas'] for r in ev_rows])
        ex = np.array([r['var_exact'] for r in ev_rows])
        se = np.array([r['var_meas_se'] for r in ev_rows])
        out['exact_vs_measured'] = fit_stats(meas, ex, se)
        out['exact_vs_measured']['median_ratio_meas_over_exact'] = float(np.median(meas / ex))
        # A/N + B fits of the exact variance, per (coherence, spread), pooled over structures
        fits = {}
        by = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in ev_rows:
            by[(r['coh'], r['spread'])][r['n_eff']].append(r)
        for (coh, sp), d in by.items():
            ne = np.array(sorted(d), float)
            v = np.array([np.mean([x['var_exact'] for x in d[n]]) for n in sorted(d)])
            vm = np.array([np.mean([x['var_meas'] for x in d[n]]) for n in sorted(d)])
            Xm = np.stack([1 / ne, np.ones_like(ne)], 1)
            coef, *_ = np.linalg.lstsq(Xm, v, rcond=None)
            pred = Xm @ coef
            r2 = 1 - ((v - pred) ** 2).sum() / max(((v - v.mean()) ** 2).sum(), 1e-300)
            sl = np.polyfit(np.log(ne), np.log(np.maximum(v, 1e-300)), 1)[0]
            first = [x for x in ev_rows if x['coh'] == coh and x['spread'] == sp]
            fits[f'{coh}|{sp:g}'] = dict(n_eff=ne.tolist(), var_exact=v.tolist(), var_meas=vm.tolist(), A=float(coef[0]),
                                         B=float(coef[1]), R2=float(r2), loglog_slope=float(sl),
                                         var_det=float(np.mean([x['var_det'] for x in first])),
                                         var_ind=float(np.mean([x['var_ind'] for x in first])),
                                         var_shiter=float(np.mean([x['var_shiter'] for x in first])),
                                         var_detshift=float(np.mean([x['var_detshift'] for x in first])),
                                         sigma_y=first[0]['sigma_y'])
        out['fits'] = fits
    # planner metrics vs N_eff (block E)
    pm = {}
    for r in rows:
        if r['block'] != 'E' or not r['model'].startswith('E-fixed') or r['n'] == 0:
            continue
        nlast = 12 if r['planner'] == 'mppi' else 30
        if r['n'] != nlast:
            continue
        ne = int(r['model'].split('@')[0].replace('E-fixed', ''))
        key = f"{r['planner']}|{r['regime']}|{r['tag']}|{r['scale']:g}"
        pm.setdefault(key, []).append(dict(n_eff=ne, oei=g(r, 'oei') if r['planner'] == 'mppi' else g(r, 'oei_mf'),
                                           dmg=g(r, 'dmg_exec') if r['planner'] == 'mppi' else g(r, 'dmg'),
                                           drift=g(r, 'drift'), se0=r.get('se0'),
                                           sel=g(r, 'sel_last') if r['planner'] == 'mppi' else g(r, 'sel_last_elite')))
    sp = {}
    for key, L in pm.items():
        L.sort(key=lambda x: x['n_eff'])
        ne = [x['n_eff'] for x in L]
        # post-hoc (added 07:55Z, after the pre-run declaration): damage vs the measured iteration-0 variance of the table
        v0 = np.array([x['se0'] for x in L], float) ** 2
        dm = np.array([x['dmg'] for x in L], float)
        okv = (v0 > 0) & (dm > 0)
        slope_v = float(np.polyfit(np.log(v0[okv]), np.log(dm[okv]), 1)[0]) if okv.sum() > 2 else np.nan
        sp[key] = dict(points=L, posthoc_loglog_slope_dmg_vs_var0=slope_v, spearman_neff_dmg=spearman(ne, [x['dmg'] for x in L]),
                       spearman_neff_absoei=spearman(ne, [abs(x['oei']) for x in L]),
                       spearman_neff_drift=spearman(ne, [x['drift'] for x in L]),
                       spearman_se0_dmg=spearman([x['se0'] for x in L], [x['dmg'] for x in L]))
        # reference variants of the same toy
        for v in ('det', 'ind', 'shiter', 'detshift'):
            pl, reg, coh, sc = key.split('|')
            rr = [r for r in rows if r['block'] == 'E' and r['planner'] == pl and r['regime'] == reg and r['tag'] == coh
                  and abs(r['scale'] - float(sc)) < 1e-9 and r['model'] == f'E-{v}@{coh}' and r['n'] == (12 if pl == 'mppi' else 30)]
            if rr:
                r = rr[0]
                sp[key][f'ref_{v}'] = dict(oei=g(r, 'oei') if pl == 'mppi' else g(r, 'oei_mf'),
                                           dmg=g(r, 'dmg_exec') if pl == 'mppi' else g(r, 'dmg'), drift=g(r, 'drift'), se0=r.get('se0'))
    out['planner_vs_neff'] = sp
    return out


def c10_detshift(rows):
    out = {}
    for r in rows:
        if r['n'] == 0:
            continue
        nlast = 12 if r['planner'] == 'mppi' else 30
        if r['n'] != nlast or r['block'] not in ('M', 'D', 'E', 'R'):
            continue
        m = r['model']
        if 'detshift' not in m:
            continue
        partner = m.replace('detshift', 'shiter')
        ix = [x for x in rows if gkey(x) == gkey(r) and x['model'] == partner and x['scale'] == r['scale'] and x['n'] == r['n']]
        if not ix:
            continue
        p = ix[0]
        dm = 'dmg_exec' if r['planner'] == 'mppi' else 'dmg'
        oe = 'oei' if r['planner'] == 'mppi' else 'oei_mf'
        key = f"{r['block']}|{r['planner']}"
        out.setdefault(key, []).append(dict(regime=r['regime'], N=r['N'], rule=r['rule'], tag=r['tag'], scale=r['scale'],
                                            dmg_ds=g(r, dm), dmg_sh=g(p, dm), dmg_diff_se=math.hypot(g(r, dm, 'se'), g(p, dm, 'se')),
                                            oei_ds=g(r, oe), oei_sh=g(p, oe), drift_ds=g(r, 'drift'), drift_sh=g(p, 'drift'),
                                            se0_ds=r.get('se0'), se0_sh=p.get('se0')))
    summ = {}
    for key, L in out.items():
        d = np.array([x['dmg_ds'] - x['dmg_sh'] for x in L])
        s = np.array([x['dmg_diff_se'] for x in L])
        summ[key] = dict(cells=len(L), frac_ds_le_sh=float(np.mean(d <= 0)), frac_ds_worse_signif=float(np.mean(d > 2 * s)),
                         median_rel_dmg=float(np.median([x['dmg_ds'] / x['dmg_sh'] for x in L if x['dmg_sh'] > 0])) if any(x['dmg_sh'] > 0 for x in L) else np.nan)
    return dict(summary=summ, cells=out)


def c11_jensen(jobs, rows):
    out = {}
    jv = jobs.get('JV')
    if jv:
        R = jv['rows']
        tab = collections.defaultdict(list)
        for r in R:
            tab[(r['spread'], r['kappa'], r['bits'])].append(r)
        rr = {}
        for (sp, kap, b), L in sorted(tab.items()):
            rr[f'{sp:g}|{kap:g}|{b}'] = dict(sd_det=float(np.mean([x['sd_det'] for x in L])),
                                             sd_bias_SR=float(np.mean([x['sd_bias_shiter'] for x in L])),
                                             sd_bias_detshift=float(np.mean([x['sd_bias_detshift'] for x in L])),
                                             sd_draw_shiter=float(np.mean([x['sd_draw_shiter'] for x in L])),
                                             ratio_bias_det=float(np.mean([x['sd_bias_shiter'] / x['sd_det'] for x in L])))
        out['direct'] = rr
        slopes = {}
        for sp in sorted({k[0] for k in tab}):
            for kap in sorted({k[1] for k in tab}):
                bs = sorted({k[2] for k in tab if k[0] == sp and k[1] == kap})
                yb = [np.log2(rr[f'{sp:g}|{kap:g}|{b}']['sd_bias_SR']) for b in bs]
                yd = [np.log2(rr[f'{sp:g}|{kap:g}|{b}']['sd_det']) for b in bs]
                slopes[f'{sp:g}|{kap:g}'] = dict(log2_sd_bias_per_bit=float(np.polyfit(bs, yb, 1)[0]),
                                                 log2_sd_det_per_bit=float(np.polyfit(bs, yd, 1)[0]))
        out['bit_slopes'] = slopes
    # planner drift by bits / kappa / variant (block J)
    pj = {}
    for r in rows:
        if r['block'] != 'J' or r['n'] == 0 or r['model'] == 'none':
            continue
        nlast = 12 if r['planner'] == 'mppi' else 30
        if r['n'] != nlast:
            continue
        parts = r['model'].split('@')          # NL-<v>@k<b>@kap<k>
        v = parts[0][3:]
        b = int(parts[1][1:])
        kap = float(parts[2][3:])
        key = f"{r['planner']}|{r['regime']}|{kap:g}"
        pj.setdefault(key, {}).setdefault(v, {})[b] = dict(
            drift=g(r, 'drift'), drift_se=g(r, 'drift', 'se'),
            oei=g(r, 'oei') if r['planner'] == 'mppi' else g(r, 'oei_mf'),
            oei_se=g(r, 'oei', 'se') if r['planner'] == 'mppi' else g(r, 'oei_mf', 'se'),
            dmg=g(r, 'dmg_exec') if r['planner'] == 'mppi' else g(r, 'dmg'),
            sel_rd=g(r, 'sel_last_rd') if r['planner'] == 'mppi' else g(r, 'sel_last_mf'))
    out['planner'] = pj
    # predicted vs measured drift ratio (SR minus detshift, over det)
    cmp_ = {}
    if jv:
        for key, d in pj.items():
            kap = float(key.split('|')[2])
            for b in NL_BITS_LIST:
                if all(v in d and b in d[v] for v in ('det', 'shiter', 'ind', 'detshift')):
                    det_d = d['det'][b]['drift']
                    sr = 0.5 * (d['shiter'][b]['drift'] + d['ind'][b]['drift'])
                    ds = d['detshift'][b]['drift']
                    meas_ratio = (sr - ds) / det_d if det_d != 0 else np.nan
                    pr = out['direct'].get(f'1|{kap:g}|{b}', {}).get('ratio_bias_det')
                    cmp_[f'{key}|k{b}'] = dict(meas_ratio=meas_ratio, pred_ratio_var=(pr ** 2) if pr is not None else None,
                                               drift_det=det_d, drift_SR=sr, drift_detshift=ds,
                                               drift_SR_se=0.5 * math.hypot(d['shiter'][b]['drift_se'], d['ind'][b]['drift_se']))
    out['ratio'] = cmp_
    return out


NL_BITS_LIST = (3, 4, 5, 6)


def c12_persistence(rows):
    out = collections.defaultdict(list)
    for r in rows:
        if r['model'] == 'none' or r['n'] == 0 or r['block'] not in ('M', 'D', 'L', 'R', 'E'):
            continue
        nlast = 12 if r['planner'] == 'mppi' else 30
        if r['n'] != nlast:
            continue
        out[f"{r['planner']}|{r['family']}"].append((r['persistence'], r['dev_ratio'], r['spread_ratio']))
    res = {}
    for k, L in out.items():
        a = np.array(L, float)
        res[k] = dict(cells=len(L), median_persistence=float(np.nanmedian(a[:, 0])),
                      median_dev_ratio=float(np.nanmedian(a[:, 1])), median_spread_ratio=float(np.nanmedian(a[:, 2])))
    return res


def c13_posthoc_damage_predictors(rows):
    """POST-HOC (added 07:56Z, after the pre-run declaration): within each block-M group (planner, regime, N, rule),
    across the 7 error models x 5 scales at the last iteration, Spearman of damage with -OEI, -drift and -sel_last
    (same draw). Theory (2.4, 7.1): drift is the damage-relevant persistent part; the same-draw selection term is
    present for fresh noise too."""
    grp = collections.defaultdict(list)
    for r in rows:
        if r['block'] != 'M' or r['model'] == 'none' or not r.get('is_last'):
            continue
        pl = r['planner']
        grp[(pl, r['regime'], r['N'], r['rule'])].append(dict(
            oei=g(r, 'oei' if pl == 'mppi' else 'oei_mf'), drift=g(r, 'drift'),
            sel=g(r, 'sel_last' if pl == 'mppi' else 'sel_last_elite'),
            dmg=g(r, 'dmg_exec' if pl == 'mppi' else 'dmg'), fam=r['family']))
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for (pl, reg, N, rule), L in grp.items():
        if len(L) < 6:
            continue
        d = [x['dmg'] for x in L]
        for name in ('oei', 'drift', 'sel'):
            out[pl][f'spearman_dmg_neg_{name}'].append(spearman([-x[name] for x in L], d))
    res = {}
    for pl, dd in out.items():
        res[pl] = {k: dict(median=float(np.nanmedian(v)), q25=float(np.nanquantile(v, 0.25)), q75=float(np.nanquantile(v, 0.75)),
                           groups=len(v)) for k, v in dd.items()}
    return res


# ------------------------------------------------------------------------------------------------ tables
TABLE_METRICS = {
    'mppi': ['oei', 'oei_rd', 'sel_last', 'sel_last_rd', 'drift', 'wc_pred', 'stein_pred', 'oei_w', 'oei_top1',
             'dmg_exec', 'dmg_mean', 'J_exec', 'J_mean', 'sJ', 'se', 'rho', 'theta', 'keff', 'dist_mean_astar',
             'saa_pred_drift'],
    'cem': ['oei_mf', 'sel_last_mf', 'sel_last_mf_sd', 'drift', 'oei_elite', 'sel_last_elite', 'wc_pred_elite',
            'stein_pred_elite', 'oei_argmin', 'sel_last_argmin', 'wc_pred_argmin', 'ntc_pred_argmin', 'dmg', 'J_mf', 'sJ',
            'se', 'rho', 'dist_mean_astar', 'saa_pred_oei_mf', 'saa_pred_drift']}


def write_tables(rows, out_dir):
    for pl in ('mppi', 'cem'):
        mets = TABLE_METRICS[pl]
        path = os.path.join(out_dir, f'cells_{pl}.csv')
        with open(path, 'w') as f:
            hdr = ['block', 'regime', 'd', 'N', 'rule', 'k', 'tag', 'model', 'family', 'scale', 'n', 'n_runs']
            hdr += [f'{m}' for m in mets] + [f'{m}_se' for m in mets]
            hdr += ['dev_ratio', 'spread_ratio', 'persistence', 'saa_slope', 'saa_r2', 'se0', 'sJ0', 'gap0']
            f.write(','.join(hdr) + '\n')
            for r in sorted(rows, key=lambda r: (r['block'], r['regime'], r['d'], r['N'], r['rule'], r['tag'], r['model'],
                                                   r['scale'], r['n'])):
                if r['planner'] != pl or r['n'] == 0:
                    continue
                nr = r['m'].get('sJ', {}).get('n', 0) if r['m'].get('sJ') else 0
                vals = [r['block'], r['regime'], r['d'], r['N'], r['rule'], r['k'], r['tag'], r['model'], r['family'],
                        r['scale'], r['n'], nr]
                vals += [g(r, m) for m in mets] + [g(r, m, 'se') for m in mets]
                vals += [r.get(x, np.nan) for x in ('dev_ratio', 'spread_ratio', 'persistence', 'saa_slope', 'saa_r2',
                                                    'se0', 'sJ0', 'gap0')]
                f.write(','.join(f'{v:.6g}' if isinstance(v, float) else str(v) for v in vals) + '\n')


# ------------------------------------------------------------------------------------------------ figures
def figures(out_dir, res, rows, C):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'figure.dpi': 110})
    fams = {'persistent': '#c0392b', 'fresh': '#2471a3', 'other': '#7d7d7d'}

    # F1 single shot
    A = C.get('C1')
    if A:
        fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
        jobsA = res['A']
        nr = [r for r in jobsA if r['dist'] == 'normal']
        ax[0].scatter([-r['pred'] for r in nr], [-r['meas'] for r in nr], s=6, c='#2471a3')
        lim = [1e-3, 40]
        ax[0].plot(lim, lim, 'k-', lw=0.6)
        ax[0].set_xscale('log'); ax[0].set_yscale('log')
        ax[0].set_xlabel('predicted -(E e_sel - E e)'); ax[0].set_ylabel('Monte Carlo')
        ax[0].set_title('normal (J, e): bivariate formula')
        qa = [r for r in jobsA if r['dist'].startswith('quad') and r['op'] == 'argmin']
        for key, mk, col in (('pred', 'o', '#7d7d7d'), ('pred_cond_exact', 's', '#c0392b'), ('pred_near_tie', '^', '#27ae60')):
            xs = [-r[key] for r in qa if r.get(key) is not None]
            ys = [-(r['meas_sub'] if key != 'pred' else r['meas']) for r in qa if r.get(key) is not None]
            ax[1].scatter(xs, ys, s=10, marker=mk, c=col, label={'pred': 'bivariate', 'pred_cond_exact': 'fixed-J exact',
                                                                     'pred_near_tie': 'near-tie count'}[key])
        ax[1].plot(lim, lim, 'k-', lw=0.6)
        ax[1].set_xscale('log'); ax[1].set_yscale('log')
        ax[1].set_xlabel('predicted (units of sigma_J)'); ax[1].set_title('quadratic it-0 population, argmin')
        ax[1].legend(fontsize=6)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'fig1_singleshot.png')); plt.close(fig)

    # F2 sel_last vs prediction
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for i, (pl, mk, pk) in enumerate((('mppi', 'sel_last', 'wc_pred'), ('cem', 'sel_last_elite', 'wc_pred_elite'))):
        for fam, col in fams.items():
            rr = [r for r in rows if r['planner'] == pl and r['family'] == fam and r['n'] > 0]
            x = np.array([-g(r, pk) for r in rr]); y = np.array([-g(r, mk) for r in rr])
            ok = (x > 0) & (y > 0)
            ax[i].scatter(x[ok], y[ok], s=3, c=col, alpha=0.4, label=fam)
        ax[i].plot([1e-5, 1e3], [1e-5, 1e3], 'k-', lw=0.6)
        ax[i].set_xscale('log'); ax[i].set_yscale('log')
        ax[i].set_xlabel('bivariate winner\'s curse (pred)'); ax[i].set_ylabel('measured (same draw)')
        ax[i].set_title(f'{pl}: selection inside an evaluation'); ax[i].legend(fontsize=6, markerscale=3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'fig2_selection_vs_pred.png')); plt.close(fig)

    # F3 decomposition bars (block M, default-like cell)
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
    models = ['M1', 'R-det', 'R-fixed', 'M2', 'M3', 'R-shiter', 'R-detshift']
    for i, pl in enumerate(('mppi', 'cem')):
        N0, rule, nlast = (512, 'q125', 12) if pl == 'mppi' else (256, 'q125', 30)
        for sc_i, sc in enumerate((8.0, 32.0)):
            vals = []
            for m in models:
                rr = [r for r in rows if r['block'] == 'M' and r['planner'] == pl and r['regime'] == 'sep' and r['N'] == N0
                      and r['rule'] == rule and r['model'] == m and r['scale'] == sc and r['n'] == nlast]
                if not rr:
                    vals.append((np.nan,) * 3)
                    continue
                r = rr[0]
                if pl == 'mppi':
                    vals.append((g(r, 'sel_last'), g(r, 'sel_last_rd'), g(r, 'drift')))
                else:
                    vals.append((g(r, 'sel_last_elite'), g(r, 'sel_last_mf'), g(r, 'drift')))
            vals = np.array(vals, float)
            x = np.arange(len(models)) + 0.4 * sc_i
            ax[i].bar(x - 0.1, vals[:, 0], 0.13, color='#e67e22', label='sel_last same draw' if sc_i == 0 else None)
            ax[i].bar(x + 0.03, vals[:, 1], 0.13, color='#95a5a6', label='sel_last re-evaluated' if sc_i == 0 else None)
            ax[i].bar(x + 0.16, vals[:, 2], 0.13, color='#8e44ad', label='drift' if sc_i == 0 else None)
        ax[i].set_xticks(np.arange(len(models)) + 0.2); ax[i].set_xticklabels(models, rotation=40, fontsize=6)
        ax[i].axhline(0, color='k', lw=0.5)
        ax[i].set_title(f'{pl} OEI = sel_last + drift (r = 8 | 32)')
        ax[i].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'fig3_decomposition.png')); plt.close(fig)

    # F4 dose-response N
    curves = C.get('C7_curves', {})
    fig, ax = plt.subplots(2, 2, figsize=(7.4, 5.0))
    for i, pl in enumerate(('mppi', 'cem')):
        for j, rule in enumerate(('q125', 'kfix')):
            a = ax[i, j]
            for m, col in (('M1', '#c0392b'), ('R-det', '#e74c3c'), ('R-fixed', '#f1948a'), ('M2', '#2471a3'),
                           ('M3', '#5dade2'), ('R-shiter', '#1abc9c'), ('R-detshift', '#27ae60')):
                key = f'{pl}|sep|{rule}|{m}|32.0'
                if key in curves:
                    c = curves[key]
                    a.plot(c['N'], c['abs_oei'], 'o-', ms=2.5, lw=1, color=col, label=m)
            a.set_xscale('log'); a.set_yscale('log')
            a.set_title(f'{pl}, {rule}, sep, r=32'); a.set_xlabel('N'); a.set_ylabel('|OEI|')
            if i == 0 and j == 0:
                a.legend(fontsize=5, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'fig4_dose_N.png')); plt.close(fig)

    # F5 iterations
    fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.8))
    for i, pl in enumerate(('mppi', 'cem')):
        N0, rule = (512, 'q125') if pl == 'mppi' else (256, 'q125')
        for m, col in (('M1', '#c0392b'), ('R-det', '#e74c3c'), ('R-fixed', '#f1948a'), ('M2', '#2471a3'), ('M3', '#5dade2'),
                       ('R-shiter', '#1abc9c'), ('R-detshift', '#27ae60')):
            rr = sorted([r for r in rows if r['block'] == 'M' and r['planner'] == pl and r['regime'] == 'sep' and r['N'] == N0
                         and r['rule'] == rule and r['model'] == m and r['scale'] == 32.0 and r['n'] > 0], key=lambda r: r['n'])
            if rr:
                ax[i].plot([r['n'] for r in rr], [g(r, 'dmg_exec' if pl == 'mppi' else 'dmg') for r in rr], 'o-', ms=2.5,
                           lw=1, color=col, label=m)
        ax[i].set_xscale('log'); ax[i].set_yscale('log'); ax[i].set_xlabel('iterations n'); ax[i].set_ylabel('damage (J increase)')
        ax[i].set_title(f'{pl}, N={N0}, q12.5%, sep, r=32')
    ax[0].legend(fontsize=5)
    W = C.get('C6', {}).get('closed_loop', {})
    for ws, ls in ((1, '-'), (0, '--')):
        for m, col in (('M1j', '#c0392b'), ('Mstep', '#e67e22'), ('M3', '#5dade2'), ('M2', '#2471a3')):
            xs, ys = [], []
            for n_it in (1, 3, 6, 12):
                v = W.get(f'ws{ws}|n{n_it}|{m}|32')
                if v:
                    xs.append(n_it); ys.append(v['rel_damage'])
            if xs:
                ax[2].plot(xs, ys, ls, marker='o', ms=2.5, lw=1, color=col, label=f'{m} ws{ws}')
    ax[2].set_xscale('log'); ax[2].set_xlabel('iterations n'); ax[2].set_ylabel('closed-loop relative damage')
    ax[2].set_title('MPPI closed loop, r=32'); ax[2].legend(fontsize=5)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'fig5_iterations.png')); plt.close(fig)

    # F6 N_eff
    C9 = C.get('C9', {})
    fits = C9.get('fits', {})
    fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.8))
    for i, coh in enumerate(('coh', 'incoh')):
        for sp, col in ((0.003, '#1b4f72'), (0.03, '#2e86c1'), (0.3, '#85c1e9'), (1.0, '#d6eaf8')):
            f_ = fits.get(f'{coh}|{sp:g}')
            if not f_:
                continue
            ax[i].plot(f_['n_eff'], f_['var_exact'], '-', color=col, lw=1, label=f'spread {sp:g} exact')
            ax[i].plot(f_['n_eff'], f_['var_meas'], 'o', color=col, ms=2.5)
            ax[i].axhline(f_['var_det'], color=col, ls=':', lw=0.8)
        ax[i].set_xscale('log'); ax[i].set_yscale('log'); ax[i].set_xlabel('N_eff (independent offsets)')
        ax[i].set_ylabel('E_u Var_cand (amp=1)'); ax[i].set_title(f'{coh} toy: exact (line), meas. (dot), det (dotted)', fontsize=7)
    ax[0].legend(fontsize=5)
    pv = C9.get('planner_vs_neff', {})
    for key, col in (('mppi|sep|coh|32', '#c0392b'), ('mppi|sep|incoh|32', '#e59866'), ('cem|sep|coh|32', '#2471a3'),
                     ('cem|sep|incoh|32', '#7fb3d5')):
        if key in pv:
            L = pv[key]['points']
            ax[2].plot([x['n_eff'] for x in L], [x['dmg'] for x in L], 'o-', ms=2.5, lw=1, color=col, label=key)
    ax[2].set_xscale('log'); ax[2].set_yscale('log'); ax[2].set_xlabel('N_eff'); ax[2].set_ylabel('damage')
    ax[2].set_title('planner damage, fixed table', fontsize=7); ax[2].legend(fontsize=5)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'fig6_neff.png')); plt.close(fig)

    # F7 Jensen
    C11 = C.get('C11', {})
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    dr = C11.get('direct', {})
    for kap, ls in ((1.0, '-'), (10.0, '--')):
        for sp, col in ((1.0, '#1b4f72'), (0.1, '#5dade2')):
            bs = [b for b in NL_BITS_LIST if f'{sp:g}|{kap:g}|{b}' in dr]
            if bs:
                ax[0].plot(bs, [dr[f'{sp:g}|{kap:g}|{b}']['ratio_bias_det'] for b in bs], ls, marker='o', ms=2.5, color=col,
                           label=f'kappa {kap:g}, spread {sp:g}')
    ax[0].set_yscale('log'); ax[0].set_xlabel('bits k'); ax[0].set_ylabel('sd(Jensen field) / sd(det error)')
    ax[0].set_title('Jensen field vs det (direct)'); ax[0].legend(fontsize=5)
    pj = C11.get('planner', {})
    for key, col in (('mppi|sep|10', '#c0392b'), ('cem|sep|10', '#2471a3')):
        if key not in pj:
            continue
        for v, mk in (('det', 'o'), ('shiter', 's'), ('ind', '^'), ('detshift', 'x')):
            if v in pj[key]:
                bs = sorted(pj[key][v])
                ax[1].plot(bs, [-pj[key][v][b]['drift'] for b in bs], marker=mk, ms=3, lw=0.8, color=col,
                           label=f'{key.split("|")[0]} {v}')
    ax[1].set_yscale('symlog', linthresh=1e-3); ax[1].set_xlabel('bits k'); ax[1].set_ylabel('-drift')
    ax[1].set_title('planner drift, kappa = 10'); ax[1].legend(fontsize=5, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'fig7_jensen.png')); plt.close(fig)


# ------------------------------------------------------------------------------------------------ main
def main():
    res, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    jobs = load_jobs(res)
    tabs = F.Tables(json.load(open(os.path.join(res, 's_tables.json'))))
    rows = planner_rows(jobs)
    add_derived(rows)
    C = {}
    C['C1'] = c1_singleshot(jobs)
    C['C2'] = c2_selection(rows)
    C['C3'] = c3_decomposition(rows)
    C['C4'] = c4_saa(rows)
    C['C5'] = c5_stationary(rows, tabs)
    C['C6'], C['C6_curves'] = c6_iterations(rows, jobs)
    C['C7'], C['C7_curves'] = c7_dose(rows, tabs)
    C['C8'] = c8_crn(rows)
    C['C9'] = c9_neff(jobs, rows)
    C['C10'] = c10_detshift(rows)
    C['C11'] = c11_jensen(jobs, rows)
    C['C12'] = c12_persistence(rows)
    C['C13_posthoc'] = c13_posthoc_damage_predictors(rows)
    C['meta'] = dict(n_jobs=len(jobs), n_rows=len(rows),
                     cpu_s=float(sum(j.get('cpu_s', j.get('wall_s', 0)) for j in jobs.values())),
                     wall_s_sum=float(sum(j.get('wall_s', 0) for j in jobs.values())),
                     n_planner_runs=int(sum(j.get('n_runs', 0) for j in jobs.values())))
    json.dump(C, open(os.path.join(out_dir, 'e4_comparisons.json'), 'w'), indent=1,
              default=lambda o: o.tolist() if hasattr(o, 'tolist') else (None if o is None else str(o)))
    write_tables(rows, out_dir)
    try:
        figures(out_dir, dict(A=jobs['A_singleshot']['rows'] if 'A_singleshot' in jobs else []), rows, C)
    except Exception as ex:
        import traceback
        traceback.print_exc()
        print('figures failed', ex)
    print('wrote', out_dir, C['meta'])


if __name__ == '__main__':
    main()
