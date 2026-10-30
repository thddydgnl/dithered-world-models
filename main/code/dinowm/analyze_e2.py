"""E2 analysis, DINO-WM part (DESIGN_main v5 section 7.1, v5.1 section 8.2, v5.2 section 9.1). Written before any E2
data existed (2026-10-22T12:00Z); the confirmatory computation is unchanged since. Pure python + numpy/scipy on the
job JSONs (server CPU or the synced copies).

usage: python analyze_e2.py <results dir holding E1_dinowm_pusht/, E2_dinowm_N/, E2_dinowm_EO/> [--out JSON] [--md MD]
                            [--csv CSV] [--family JSON]

Points (PushT k=4, eval6 pairs 0-99, variants none / det / sh-iter):
  N sweep (confirmatory, top-30 fixed): N = 64, 128, 256, 512, 1024 (E2_dinowm_N) and N = 300 (E1_dinowm_pusht)
  elite ratio at N = 300 (descriptive): top-6, top-38, top-75 (E2_dinowm_EO) and top-30 (E1)
  opt_steps (descriptive): 5, 10 (E2_dinowm_EO) and 30 (E1)
Per point: D_v = success(none at that point) - success(v at that point), paired over the same pairs; OEI = oei_ref_mf
(the frozen definition = the independent-draw version, v5.2 (b1)); decomposition drift = last_pop_dJ - it0_rand_dJ,
sel_indep = mf_dJ - last_pop_dJ, sel_same(elite) = last_elite_dJ - last_pop_dJ (v5.1 section 8.2 last bullet).
CIs: paired bootstrap over pairs (10,000, RandomState(0), percentile 95%).
H2c (task PushT): Spearman(N, |OEI_det|) >= 0.8 and Spearman(N, D_det) >= 0.8 over the 6 N levels; the H2c verdict
needs 3 of the 4 tasks (TD-MPC2 3 + PushT), combined by the coordinator.
Auxiliary: Spearman(D_det, |OEI_det|); sh-iter flatness Spearman(N, |OEI_sh-iter|), Spearman(N, D_sh-iter); the
bootstrap distribution of each rho with the pairs resampled jointly across the N levels (same index for every level).

Changes 2026-10-23 (E2 analysis worker, before any E2 number was computed; the H2c rule computation is unchanged):
  - completeness: every point reports n per variant and `complete` (100 paired episodes); the H2c block reports
    `complete` (all 6 points complete) and is marked provisional otherwise
  - sections without data (E2_dinowm_EO before it runs) are reported as pending ('실행 대기') in the JSON and the tables
  - the joint bootstrap also covers the two sh-iter flatness rhos (auxiliary) and counts undefined (constant) resamples
  - descriptive extras per point: sel_same(elite) of sh-iter in the table, the auxiliary OEI versions oei_ref_argmin
    and oei_true_mf (v3: the true-cost reference is auxiliary), none success
  - --csv (one row per point and variant) and --family (the per-family summary file for the coordinator)
  - sel_same(mf) of sh-iter (re-evaluation of the executed plan under the last iteration's draw, drift_ref.py
    --mode mf_same) needs the GPU and is not computed for E2 (E1 has it); for det it equals sel_indep by construction
Changes 2026-10-24T16:32Z (E2 elite / opt_steps part, after E2_dinowm_EO ended and BEFORE any EO number was computed;
no rule computation changed, the N-sweep numbers stay the same):
  - the second OEI version of v5 7.1 / 7.0 ("OEI 두 판") is added as a descriptive column: oei_elite = oei_ref_elite =
    last_elite_dJ - it0_rand_dJ = sel_same(elite) + drift (the same-draw version: the error of the selected top-k under
    the draw that selected them); oei = oei_ref_mf stays the frozen, independent-draw version
  - per point the sh-iter gain D_det - D_sh-iter = S(sh-iter) - S(det) with its paired CI (descriptive)
  - --family also lists the elite-ratio and opt_steps points
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A

NPTS = [(64, 'E2_dinowm_N', 'N64t30'), (128, 'E2_dinowm_N', 'N128t30'), (256, 'E2_dinowm_N', 'N256t30'),
        (300, 'E1_dinowm_pusht', None), (512, 'E2_dinowm_N', 'N512t30'), (1024, 'E2_dinowm_N', 'N1024t30')]
EPTS = [('top6 (2%)', 'E2_dinowm_EO', 'top6'), ('top30 (10%, E1)', 'E1_dinowm_pusht', None),
        ('top38 (12.5%)', 'E2_dinowm_EO', 'top38'), ('top75 (25%)', 'E2_dinowm_EO', 'top75')]
OPTS = [('opt_steps 5', 'E2_dinowm_EO', 'it5'), ('opt_steps 10', 'E2_dinowm_EO', 'it10'), ('opt_steps 30 (E1)', 'E1_dinowm_pusht', None)]
VARS = ['none', 'det', 'shiter']
COMPS = ('oei', 'oei_elite', 'drift', 'sel_indep', 'sel_same', 'oei_argmin', 'oei_true_mf')
NB = 10000
N_EP = 100
PENDING = '실행 대기'


def load_point(res, man, tag):
    """-> {var: {ep: record}} for PushT k=4 (E1: the eval6 none / det_k4 / shiter_k4 records)."""
    import glob
    out = {v: {} for v in VARS}
    for f in glob.glob(f'{res}/{man}/pusht/*.json'):
        d = json.load(open(f))
        j = d['job']
        if tag is None and (j.get('planner') or j.get('ckpt') or j.get('qopts')):
            continue
        if tag is not None and not j['id'].endswith('_' + tag):
            continue
        if j['var'] not in VARS or (j['var'] != 'none' and int(j['k']) != 4):
            continue
        if j['split'] != 'eval6':
            raise SystemExit(f"{f}: split {j['split']} (expected eval6)")
        for r in d['records']:
            out[j['var']][int(r['ep'])] = r
    return out


def comp(r):
    o = r['oei']
    return dict(oei=o['oei_ref_mf'], drift=o['last_pop_dJ'] - o['it0_rand_dJ'], sel_indep=o['mf_dJ'] - o['last_pop_dJ'],
                sel_same=o['last_elite_dJ'] - o['last_pop_dJ'], oei_argmin=o['oei_ref_argmin'], oei_elite=o['oei_ref_elite'],
                oei_true_mf=o.get('oei_true_mf', np.nan))


def point_stats(P, I=None):
    eps = sorted(set(P['none']) & set(P['det']) & set(P['shiter']))
    n = len(eps)
    if n == 0:
        return None
    S = {v: np.array([float(P[v][e]['success']) for e in eps]) for v in VARS}
    C = {v: {k: np.array([comp(P[v][e])[k] for e in eps]) for k in COMPS} for v in ('det', 'shiter')}
    n_var = {v: len(P[v]) for v in VARS}
    st = dict(n=n, n_var=n_var, complete=bool(n == N_EP and all(x == N_EP for x in n_var.values())), eps=eps,
              success={v: float(S[v].mean()) for v in VARS})
    for v in ('det', 'shiter'):
        d = S['none'] - S[v]
        st[f'D_{v}'] = float(d.mean()); st[f'D_{v}_ci'] = A.boot(lambda i: d[i].mean(), n)
        for k, x in C[v].items():
            st[f'{k}_{v}'] = float(x.mean()); st[f'{k}_{v}_ci'] = A.boot(lambda i: x[i].mean(), n)
    g = S['shiter'] - S['det']                    # D_det - D_sh-iter (descriptive)
    st['gain_shiter'] = float(g.mean()); st['gain_shiter_ci'] = A.boot(lambda i: g[i].mean(), n)
    st['_S'], st['_C'] = S, C
    return st


def spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(x, y).correlation
    return float(r) if np.isfinite(r) else float('nan')


def sweep(pts):
    """pts: list of (N, stats). Spearman tests of H2c and their joint bootstrap (same pair index at every level)."""
    Ns = [N for N, s in pts]
    D = [s['D_det'] for N, s in pts]; O = [abs(s['oei_det']) for N, s in pts]
    Ds = [s['D_shiter'] for N, s in pts]; Os = [abs(s['oei_shiter']) for N, s in pts]
    out = dict(N=Ns, D_det=D, absOEI_det=O, D_shiter=Ds, absOEI_shiter=Os,
               rho_N_absOEI_det=spearman(Ns, O), rho_N_D_det=spearman(Ns, D), rho_D_absOEI_det=spearman(D, O),
               rho_N_absOEI_shiter=spearman(Ns, Os), rho_N_D_shiter=spearman(Ns, Ds))
    out['H2c_pusht'] = bool(out['rho_N_absOEI_det'] >= 0.8 and out['rho_N_D_det'] >= 0.8)
    out['complete'] = bool(len(pts) == len(NPTS) and all(s['complete'] for N, s in pts))
    common = sorted(set.intersection(*[set(s['eps']) for N, s in pts]))
    pos = [{e: i for i, e in enumerate(s['eps'])} for N, s in pts]
    idx = np.random.RandomState(0).randint(0, len(common), size=(NB, len(common)))
    keys = ('rho_N_absOEI_det', 'rho_N_D_det', 'rho_D_absOEI_det', 'rho_N_absOEI_shiter', 'rho_N_D_shiter')
    bs = {k: [] for k in keys}
    for row in idx:
        ee = [common[i] for i in row]
        Db, Ob, Dsb, Osb = [], [], [], []
        for (N, s), p in zip(pts, pos):
            ii = [p[e] for e in ee]
            Db.append(float((s['_S']['none'][ii] - s['_S']['det'][ii]).mean())); Ob.append(abs(float(s['_C']['det']['oei'][ii].mean())))
            Dsb.append(float((s['_S']['none'][ii] - s['_S']['shiter'][ii]).mean()))
            Osb.append(abs(float(s['_C']['shiter']['oei'][ii].mean())))
        bs['rho_N_absOEI_det'].append(spearman(Ns, Ob)); bs['rho_N_D_det'].append(spearman(Ns, Db)); bs['rho_D_absOEI_det'].append(spearman(Db, Ob))
        bs['rho_N_absOEI_shiter'].append(spearman(Ns, Osb)); bs['rho_N_D_shiter'].append(spearman(Ns, Dsb))
    for k, v in bs.items():
        v = np.array(v, float); fin = v[np.isfinite(v)]
        out[k + '_boot'] = dict(ci=[float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))],
                                p_ge_0_8=float((fin >= 0.8).mean()), median=float(np.median(fin)), n=len(fin),
                                n_undefined=int(len(v) - len(fin)))
    out['n_common_pairs'] = len(common)
    return out


def clean(st):
    return {k: v for k, v in st.items() if not k.startswith('_') and k != 'eps'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('res'); ap.add_argument('--out', default=None); ap.add_argument('--md', default=None)
    ap.add_argument('--csv', default=None); ap.add_argument('--family', default=None)
    a = ap.parse_args()
    out = dict(N_sweep={}, elite={}, opt_steps={}, status={})
    pts = []
    for N, man, tag in NPTS:
        st = point_stats(load_point(a.res, man, tag))
        if st:
            pts.append((N, st)); out['N_sweep'][str(N)] = clean(st)
    if len(pts) == len(NPTS):
        out['H2c'] = sweep(pts)
    for name, rows in (('elite', EPTS), ('opt_steps', OPTS)):
        for lab, man, tag in rows:
            st = point_stats(load_point(a.res, man, tag))
            if st:
                out[name][lab] = clean(st)
    for name, rows in (('N_sweep', NPTS), ('elite', EPTS), ('opt_steps', OPTS)):
        keys = [str(r[0]) for r in rows]
        new = [str(r[0]) for r in rows if r[1] != 'E1_dinowm_pusht']    # the E1 reference point exists already
        if all(k in out[name] and out[name][k]['complete'] for k in keys):
            out['status'][name] = 'complete'
        elif not any(k in out[name] for k in new):
            out['status'][name] = PENDING
        else:
            out['status'][name] = 'partial'
    print(json.dumps(dict(status=out['status'], H2c={k: v for k, v in out.get('H2c', {}).items() if not k.endswith('_boot')}),
                     indent=1, default=float, ensure_ascii=False))
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1, default=float, ensure_ascii=False)
    if a.md:
        open(a.md, 'w').write(tables(out))
    if a.csv:
        write_csv(out, a.csv)
    if a.family:
        json.dump(family(out), open(a.family, 'w'), indent=1, default=float, ensure_ascii=False)


def write_csv(out, path):
    keys = ['section', 'point', 'n', 'complete', 'success_none', 'var', 'success', 'D', 'D_lo', 'D_hi'] + \
           [f'{k}{s}' for k in COMPS for s in ('', '_lo', '_hi')]
    with open(path, 'w') as f:
        f.write(','.join(keys) + '\n')
        for name in ('N_sweep', 'elite', 'opt_steps'):
            for lab, s in out[name].items():
                for v in ('det', 'shiter'):
                    row = [name, lab, s['n'], s['complete'], s['success']['none'], v, s['success'][v], s[f'D_{v}'],
                           *s[f'D_{v}_ci']]
                    for k in COMPS:
                        row += [s[f'{k}_{v}'], *s[f'{k}_{v}_ci']]
                    f.write(','.join(f'{x:.6g}' if isinstance(x, float) else str(x) for x in row) + '\n')


def family(out):
    """Per-family summary for the coordinator's H2c combination (PushT is the DINO-WM task of the 4)."""
    h = out.get('H2c') or {}
    pts = []
    for N, s in out['N_sweep'].items():
        p = dict(N=int(N), source='E1_dinowm_pusht (E1 PushT k=4 cell)' if N == '300' else 'E2_dinowm_N', n=s['n'],
                 complete=s['complete'], success_none=s['success']['none'])
        for v, vn in (('det', 'det'), ('shiter', 'sh-iter')):
            p[f'D_{vn}'] = s[f'D_{v}']; p[f'D_{vn}_ci'] = s[f'D_{v}_ci']
            p[f'OEI_{vn}'] = s[f'oei_{v}']; p[f'OEI_{vn}_ci'] = s[f'oei_{v}_ci']
            for k in ('drift', 'sel_indep', 'sel_same'):
                p[f'{k}_{vn}'] = s[f'{k}_{v}']; p[f'{k}_{vn}_ci'] = s[f'{k}_{v}_ci']
        pts.append(p)
    pts.sort(key=lambda p: p['N'])
    return dict(
        family='DINO-WM', task='pusht', k=4, split='eval6 pairs 0-99 (100 paired episodes per point)',
        rule='DESIGN_main v5 7.1 H2c on the v5.1 8.2 top-30 sweep: task holds iff Spearman(N, |OEI_det|) >= 0.8 and '
             'Spearman(N, D_det) >= 0.8 over the 6 N levels (64, 128, 256, 300 = E1, 512, 1024); H2c holds with 3 of '
             'the 4 tasks (mw-push, mw-pick-place, cheetah-run, PushT). OEI = oei_ref_mf (negative = exploitation); '
             'D = success-rate drop vs the same-N none (fraction, x100 = %p)',
        H2c_pusht=h.get('H2c_pusht'), complete=h.get('complete'),
        rho=dict(N_absOEI_det=h.get('rho_N_absOEI_det'), N_D_det=h.get('rho_N_D_det'),
                 aux_D_absOEI_det=h.get('rho_D_absOEI_det'), aux_N_absOEI_shiter=h.get('rho_N_absOEI_shiter'),
                 aux_N_D_shiter=h.get('rho_N_D_shiter')),
        rho_joint_bootstrap={k: h.get(k + '_boot') for k in ('rho_N_absOEI_det', 'rho_N_D_det', 'rho_D_absOEI_det',
                                                             'rho_N_absOEI_shiter', 'rho_N_D_shiter')},
        points=pts, status=dict(N_sweep=out['status'].get('N_sweep'), elite=out['status'].get('elite'),
                                opt_steps=out['status'].get('opt_steps')),
        elite_points=[eo_point(lab, s) for lab, s in out['elite'].items()],
        opt_steps_points=[eo_point(lab, s) for lab, s in out['opt_steps'].items()])


def eo_point(lab, s):
    """one elite-ratio / opt_steps point for the family file (descriptive, no threshold)."""
    p = dict(point=lab, n=s['n'], complete=s['complete'], success_none=s['success']['none'],
             gain_shiter=s['gain_shiter'], gain_shiter_ci=s['gain_shiter_ci'])
    for v, vn in (('det', 'det'), ('shiter', 'sh-iter')):
        p[f'D_{vn}'] = s[f'D_{v}']; p[f'D_{vn}_ci'] = s[f'D_{v}_ci']
        for k in ('oei', 'oei_elite', 'drift', 'sel_indep', 'sel_same'):
            p[f'{k}_{vn}'] = s[f'{k}_{v}']; p[f'{k}_{vn}_ci'] = s[f'{k}_{v}_ci']
    return p


def tables(out):
    def pp(x, c):
        return f'{100 * x:+.0f} [{100 * c[0]:.0f}, {100 * c[1]:.0f}]'

    def f3(x, c):
        return f'{x:+.3f} [{c[0]:+.3f}, {c[1]:+.3f}]'
    L = []
    for name, title in (('N_sweep', 'N (top-30 고정)'), ('elite', 'elite (N = 300)'), ('opt_steps', 'opt_steps (N = 300, top-30)')):
        L += [f'### {title}', '']
        if out['status'][name] == PENDING:
            L += [f'{PENDING} (E2_dinowm_EO 자료 없음; E1 기준점만 있다)', '']
        L += ['| 점 | n | none | D_det %p | D_sh-iter %p | OEI det | OEI sh-iter | drift det | sel_indep det | '
              'sel_same det | drift sh-iter | sel_indep sh-iter | sel_same sh-iter |', '|' + '---|' * 13]
        for lab, s in out[name].items():
            L.append(f"| {lab} | {s['n']} | {100 * s['success']['none']:.0f}% | {pp(s['D_det'], s['D_det_ci'])} | "
                     f"{pp(s['D_shiter'], s['D_shiter_ci'])} | {f3(s['oei_det'], s['oei_det_ci'])} | {f3(s['oei_shiter'], s['oei_shiter_ci'])} | "
                     f"{f3(s['drift_det'], s['drift_det_ci'])} | {f3(s['sel_indep_det'], s['sel_indep_det_ci'])} | "
                     f"{f3(s['sel_same_det'], s['sel_same_det_ci'])} | {f3(s['drift_shiter'], s['drift_shiter_ci'])} | "
                     f"{f3(s['sel_indep_shiter'], s['sel_indep_shiter_ci'])} | {f3(s['sel_same_shiter'], s['sel_same_shiter_ci'])} |")
        L.append('')
        if out[name]:
            L += ['| 점 | D_det − D_sh-iter %p | OEI det 독립 draw(oei_ref_mf) | OEI det 같은 draw(oei_ref_elite) | '
                  'OEI sh-iter 독립 draw | OEI sh-iter 같은 draw |', '|' + '---|' * 6]
            for lab, s in out[name].items():
                L.append(f"| {lab} | {pp(s['gain_shiter'], s['gain_shiter_ci'])} | {f3(s['oei_det'], s['oei_det_ci'])} | "
                         f"{f3(s['oei_elite_det'], s['oei_elite_det_ci'])} | {f3(s['oei_shiter'], s['oei_shiter_ci'])} | "
                         f"{f3(s['oei_elite_shiter'], s['oei_elite_shiter_ci'])} |")
            L.append('')
    if out['N_sweep']:
        L += ['### 보조 OEI (N sweep)', '', '| N | oei_ref_argmin det | oei_ref_argmin sh-iter | oei_true_mf det | oei_true_mf sh-iter |',
              '|' + '---|' * 5]
        for lab, s in out['N_sweep'].items():
            L.append(f"| {lab} | {f3(s['oei_argmin_det'], s['oei_argmin_det_ci'])} | {f3(s['oei_argmin_shiter'], s['oei_argmin_shiter_ci'])} | "
                     f"{f3(s['oei_true_mf_det'], s['oei_true_mf_det_ci'])} | {f3(s['oei_true_mf_shiter'], s['oei_true_mf_shiter_ci'])} |")
        L.append('')
    h = out.get('H2c')
    if h:
        L += ['### H2c (PushT)', '', f"- ρ_s(N, |OEI_det|) = {h['rho_N_absOEI_det']:+.3f}, ρ_s(N, D_det) = {h['rho_N_D_det']:+.3f} → "
              f"PushT {'성립' if h['H2c_pusht'] else '불성립'}" + ('' if h['complete'] else ' (자료 불완전: 잠정)'),
              f"- 보조: ρ_s(D_det, |OEI_det|) = {h['rho_D_absOEI_det']:+.3f}; sh-iter ρ_s(N, |OEI|) = {h['rho_N_absOEI_shiter']:+.3f}, "
              f"ρ_s(N, D) = {h['rho_N_D_shiter']:+.3f}"]
        for k in ('rho_N_absOEI_det', 'rho_N_D_det', 'rho_D_absOEI_det', 'rho_N_absOEI_shiter', 'rho_N_D_shiter'):
            b = h[k + '_boot']
            L.append(f"- bootstrap {k}: 95% [{b['ci'][0]:+.2f}, {b['ci'][1]:+.2f}], 중앙값 {b['median']:+.2f}, "
                     f"P(ρ ≥ 0.8) = {b['p_ge_0_8']:.2f} (정의 안 됨 {b['n_undefined']})")
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()
