"""DESIGN_main v5.2 section 9.1, DINO-WM part (pre-registered after E1; written before any DINO-WM decomposition was
computed): (b1) H2a_indep for PushT, (b2) H2a' for PushT k=4, the descriptive decomposition of every E1 cell, and
(b3) the region-effect control drift_ref when its JSON exists. Pure python + numpy, on the synced E1 job JSONs.

usage: python analyze_e1_v52.py <results dir with E1_dinowm_wp/, E1_dinowm_pusht/> [--drift-ref DIR] [--mf-same DIR]
                                [--out JSON] [--md MD]

Per episode (fields of the E1 record's `oei`; e = J_v - J_none, the unquantized-model reference of oei_ref_*):
  drift          = last_pop_dJ - it0_rand_dJ       (mean e of the last population minus that of the iteration-0 random
                                                    candidates; v5.2 definition)
  sel_indep      = mf_dJ - last_pop_dJ             (executed plan = final elite mean; for the fresh variants ind,
                                                    sh-iter, det+shift its e uses a NEW draw (it = opt_steps); for det
                                                    and sh-plan it is the same error field)
  sel_same_elite = last_elite_dJ - last_pop_dJ     (top-30 of the last population under the draw that selected them)
  sel_same_mf    = e(mf under the last iteration's draw) - last_pop_dJ   (sh-iter, det+shift: recomputed offline by
                                                    drift_ref.py --mode mf_same; det, sh-plan: = sel_indep; ind: undefined)
  oei_ref_mf     = sel_indep + drift               (identity, checked)
Cell values: means over the cell's episodes; CIs: paired bootstrap over episodes, 10,000 resamples, RandomState(0),
percentile 95%. r_v = mean(drift_v) / mean(drift_det) on the common episodes, recomputed per resample; resamples with
mean drift_det* >= 0 are dropped (count reported).
(b1) PushT: det oei_ref_mf CI upper < 0 and sh-iter oei_ref_mf CI contains 0 (= the v4 H2a with oei_ref_mf).
(b2) PushT k=4 passes if drift_det CI upper < 0 and r_sh-iter CI upper < 0.5. Positive control (outside the rule):
     r_sh-plan CI lower > 0.5. Descriptive: r_ind, r_det+shift, sel_same_*, sel_indep.
(b3) drift_ref(e) = l_ref - f_ref (drift_ref.py); reported with drift_det - drift_ref (paired)."""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A

CELLS = {'pusht': [3, 4, 5], 'wall': [3, 4, 5, 6], 'pointmaze': [2, 3, 4]}
ROLE = {('wall', 3): 'exploratory'}
VARS = ['det', 'ind', 'shiter', 'shstep', 'detshift']
VN = dict(det='det', ind='ind', shiter='sh-iter', shstep='sh-plan', detshift='det+shift')
TN = dict(pusht='PushT', wall='Wall', pointmaze='PointMaze')
NB = 10000


def idx(n):
    return np.random.RandomState(0).randint(0, n, size=(NB, n))


def ci_mean(x):
    x = np.asarray(x, float)
    m = x[idx(len(x))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def ratio(num, den):
    """mean(num)/mean(den) with the paired bootstrap; resamples with mean(den*) >= 0 dropped."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    I = idx(len(num))
    n_, d_ = num[I].mean(1), den[I].mean(1)
    keep = d_ < 0
    r = n_[keep] / d_[keep]
    return dict(r=float(num.mean() / den.mean()) if den.mean() != 0 else None,
                ci=[float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))] if keep.any() else [None, None],
                dropped=int((~keep).sum()))


def per_episode(rec):
    o = rec['oei']
    d = dict(drift=o['last_pop_dJ'] - o['it0_rand_dJ'], sel_indep=o['mf_dJ'] - o['last_pop_dJ'],
             sel_same_elite=o['last_elite_dJ'] - o['last_pop_dJ'], sel_same_argmin=o['last_argmin_dJ'] - o['last_pop_dJ'],
             oei=o['oei_ref_mf'])
    d['identity_err'] = abs(d['sel_indep'] + d['drift'] - d['oei'])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('res'); ap.add_argument('--drift-ref', default=None); ap.add_argument('--mf-same', default=None)
    ap.add_argument('--out', default=None); ap.add_argument('--md', default=None)
    a = ap.parse_args()
    R = {}
    for sub in ('E1_dinowm_wp', 'E1_dinowm_pusht'):
        if os.path.isdir(f'{a.res}/{sub}'):
            R.update(A.load(f'{a.res}/{sub}')[0])
    DR, MS = {}, {}
    for t in CELLS:
        man = 'E1_dinowm_pusht' if t == 'pusht' else 'E1_dinowm_wp'
        for dd, store in ((a.drift_ref, DR), (a.mf_same, MS)):
            p = f'{dd}/{man}_{t}.json' if dd else None
            if p and os.path.exists(p):
                store[t] = json.load(open(p))['episodes']
    out = dict(cells={}, checks={})
    max_id = 0.0
    for t, ks in CELLS.items():
        for k in ks:
            eps = sorted(e for (tt, v, kk, e) in R if tt == t and v == 'det' and kk == k)
            if not eps:
                continue
            C = {}
            ep_det = {e: per_episode(R[(t, 'det', k, e)]) for e in eps}
            for v in VARS:
                ev = sorted(e for (tt, vv, kk, e) in R if tt == t and vv == v and kk == k and e in ep_det)
                if not ev:
                    continue
                P = {e: per_episode(R[(t, v, k, e)]) for e in ev}
                max_id = max(max_id, max(p['identity_err'] for p in P.values()))
                c = dict(n=len(ev))
                for key in ('drift', 'sel_indep', 'sel_same_elite', 'sel_same_argmin', 'oei'):
                    x = [P[e][key] for e in ev]
                    c[key] = float(np.mean(x)); c[key + '_ci'] = ci_mean(x)
                if v in ('det', 'shstep'):
                    c['sel_same_mf'] = c['sel_indep']; c['sel_same_mf_ci'] = c['sel_indep_ci']
                elif v in ('shiter', 'detshift') and t in MS:
                    ms = MS[t]
                    em = [e for e in ev if str(e) in ms and f'{v}{k}' in ms[str(e)]]
                    if em:
                        x = [ms[str(e)][f'{v}{k}']['e_same_mf'] - R[(t, v, k, e)]['oei']['last_pop_dJ'] for e in em]
                        c['sel_same_mf'] = float(np.mean(x)); c['sel_same_mf_ci'] = ci_mean(x); c['sel_same_mf_n'] = len(em)
                        c['mf_check_max'] = max(ms[str(e)][f'{v}{k}']['check_v_mf'] for e in em)
                if v != 'det':
                    c['r'] = ratio([P[e]['drift'] for e in ev], [ep_det[e]['drift'] for e in ev])
                C[v] = c
            if t in DR:
                dr = DR[t]
                ed = [e for e in eps if str(e) in dr and str(k) in dr[str(e)]['k']]
                if ed:
                    x = [dr[str(e)]['k'][str(k)]['drift_ref'] for e in ed]
                    y = [ep_det[e]['drift'] - dr[str(e)]['k'][str(k)]['drift_ref'] for e in ed]
                    C['drift_ref'] = dict(n=len(ed), drift_ref=float(np.mean(x)), drift_ref_ci=ci_mean(x),
                                          det_minus_ref=float(np.mean(y)), det_minus_ref_ci=ci_mean(y),
                                          check_det_it0_max=max((dr[str(e)]['k'][str(k)]['check_det_it0'] or 0) for e in ed),
                                          check_none_max=max(max(dr[str(e)]['check_none_it0'], dr[str(e)]['check_none_last']) for e in ed))
            out['cells'][f'{t}_k{k}'] = dict(task=t, k=k, role=ROLE.get((t, k), 'confirmatory'), vars=C)
    out['checks']['identity_max_abs_err'] = max_id
    P4 = out['cells'].get('pusht_k4', {}).get('vars', {})
    if 'det' in P4 and 'shiter' in P4:
        d, s = P4['det'], P4['shiter']
        b1 = dict(det_oei=d['oei'], det_oei_ci=d['oei_ci'], shiter_oei=s['oei'], shiter_oei_ci=s['oei_ci'],
                  det_upper_lt0=d['oei_ci'][1] < 0, shiter_ci_contains0=s['oei_ci'][0] <= 0 <= s['oei_ci'][1])
        b1['holds'] = bool(b1['det_upper_lt0'] and b1['shiter_ci_contains0'])
        rs = s['r']
        b2 = dict(drift_det=d['drift'], drift_det_ci=d['drift_ci'], cond1_drift_det_upper_lt0=d['drift_ci'][1] < 0,
                  r_shiter=rs['r'], r_shiter_ci=rs['ci'], r_shiter_dropped=rs['dropped'],
                  cond2_r_shiter_upper_lt_half=rs['ci'][1] is not None and rs['ci'][1] < 0.5)
        b2['holds'] = bool(b2['cond1_drift_det_upper_lt0'] and b2['cond2_r_shiter_upper_lt_half'])
        if 'shstep' in P4:
            rp = P4['shstep']['r']
            b2['positive_control_r_shplan'] = rp['r']; b2['positive_control_r_shplan_ci'] = rp['ci']
            b2['positive_control_lower_gt_half'] = rp['ci'][0] is not None and rp['ci'][0] > 0.5
        for v in ('ind', 'detshift'):
            if v in P4:
                b2[f'r_{v}'] = P4[v]['r']['r']; b2[f'r_{v}_ci'] = P4[v]['r']['ci']
        out['b1_H2a_indep_pusht'] = b1
        out['b2_H2a_prime_pusht_k4'] = b2
    print(json.dumps({k: out[k] for k in out if k != 'cells'}, indent=1, default=float))
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1, default=float)
    if a.md:
        open(a.md, 'w').write(tables(out))


def tables(out):
    def f(x, nd=3):
        return '—' if x is None else f'{x:+.{nd}f}'

    def c(v, nd=3):
        return '—' if v is None or v[0] is None else f'[{v[0]:+.{nd}f}, {v[1]:+.{nd}f}]'
    L = ['| 과제 | k | 역할 | 변형 | n | OEI(oei_ref_mf) | drift [95% CI] | sel_indep [95% CI] | sel_same(elite) [95% CI] | '
         'sel_same(mf) | r = drift_v/drift_det [95% CI] (버린 재표본) |', '|' + '---|' * 11]
    for key, cell in out['cells'].items():
        for v, x in cell['vars'].items():
            if v == 'drift_ref':
                continue
            r = x.get('r')
            L.append(f"| {TN[cell['task']]} | {cell['k']} | {'탐색' if cell['role'] == 'exploratory' else '확증'} | {VN[v]} | {x['n']} | "
                     f"{f(x['oei'])} | {f(x['drift'])} {c(x['drift_ci'])} | {f(x['sel_indep'])} {c(x['sel_indep_ci'])} | "
                     f"{f(x['sel_same_elite'])} {c(x['sel_same_elite_ci'])} | {f(x.get('sel_same_mf'))} | "
                     + ('—' if r is None else f"{f(r['r'], 2)} {c(r['ci'], 2)} ({r['dropped']})") + ' |')
    L += ['', '| 과제 | k | n | drift_ref [95% CI] | drift_det − drift_ref [95% CI] | 점검(det it0, none) |', '|---|---|---|---|---|---|']
    for key, cell in out['cells'].items():
        d = cell['vars'].get('drift_ref')
        if d:
            L.append(f"| {TN[cell['task']]} | {cell['k']} | {d['n']} | {f(d['drift_ref'])} {c(d['drift_ref_ci'])} | "
                     f"{f(d['det_minus_ref'])} {c(d['det_minus_ref_ci'])} | {d['check_det_it0_max']:.1e}, {d['check_none_max']:.1e} |")
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()
