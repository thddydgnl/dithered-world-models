"""POST-HOC description of the DINO-WM E2 N sweep (written 2026-10-23 after the analyze_e2.py tables were seen; not part
of any pre-registered rule, no verdict). It quantifies the N dependence that v5.1 8.2 / v5.2 9.1 ask to describe
(sel_last and drift along N) with paired contrasts, because the per-point CIs mostly carry the between-pair spread:
for det and sh-iter and q in (D, OEI = oei_ref_mf, drift, sel_indep, sel_same(elite)):
  Spearman(N, q) over the 6 levels, and the paired differences q(N=1024) - q(N=64) and q(N=1024) - q(N=300) with a
  paired bootstrap over the same eval6 pairs (10,000, RandomState(0), percentile 95%).
EO part (added 2026-10-24T16:34Z after the elite / opt_steps tables were seen; post-hoc, no verdict): for every
E2_dinowm_EO point the paired difference to the E1 reference point (top-30, opt_steps 30) of D_det, D_sh-iter, the
sh-iter gain D_det - D_sh-iter, and det drift / sel_indep / sel_same(elite), same bootstrap.
usage: python e2_posthoc_dinowm.py <results dir> --out JSON [--md MD]
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A
import analyze_e2 as E

QS = ('D', 'oei', 'drift', 'sel_indep', 'sel_same')
EO = [('top6', 'top6 (2%)'), ('top38', 'top38 (12.5%)'), ('top75', 'top75 (25%)'), ('it5', 'opt_steps 5'), ('it10', 'opt_steps 10')]
EOQ = (('det', 'D'), ('shiter', 'D'), ('gain', 'gain'), ('det', 'drift'), ('det', 'sel_indep'), ('det', 'sel_same'))


def series(s, v, q):
    """per-pair values of quantity q for variant v at one point (aligned on s['eps'])."""
    if q == 'D':
        return s['_S']['none'] - s['_S'][v]
    return s['_C'][v][q]


def series2(s, v, q):
    if v == 'gain':                       # D_det - D_sh-iter per pair
        return s['_S']['shiter'] - s['_S']['det']
    return series(s, v, q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('res'); ap.add_argument('--out', required=True); ap.add_argument('--md', default=None)
    a = ap.parse_args()
    pts = {}
    for N, man, tag in E.NPTS:
        st = E.point_stats(E.load_point(a.res, man, tag))
        if st is None or not st['complete']:
            raise SystemExit(f'point N={N} incomplete')
        pts[N] = st
    Ns = sorted(pts)
    eps = pts[Ns[0]]['eps']
    assert all(pts[N]['eps'] == eps for N in Ns)
    out = dict(note='post-hoc description (after the E2 tables were seen); no verdict', N=Ns, rows={})
    for v in ('det', 'shiter'):
        for q in QS:
            m = [float(series(pts[N], v, q).mean()) for N in Ns]
            row = dict(values=m, rho_N=E.spearman(Ns, m))
            for lo in (64, 300):
                d = series(pts[1024], v, q) - series(pts[lo], v, q)
                row[f'diff_1024_minus_{lo}'] = float(d.mean())
                row[f'diff_1024_minus_{lo}_ci'] = A.boot(lambda i: d[i].mean(), len(d))
            out['rows'][f'{v}:{q}'] = row
    ref = pts[300]
    out['eo'] = {}
    for tag, lab in EO:
        st = E.point_stats(E.load_point(a.res, 'E2_dinowm_EO', tag))
        if st is None or not st['complete']:
            continue
        assert st['eps'] == ref['eps']
        row = {}
        for v, q in EOQ:
            d = series2(st, v, q) - series2(ref, v, q)
            row[f'{v}:{q}'] = dict(diff=float(d.mean()), ci=A.boot(lambda i: d[i].mean(), len(d)))
        out['eo'][lab] = row
    json.dump(out, open(a.out, 'w'), indent=1, default=float, ensure_ascii=False)
    L = ['| 변형 | 양 | ρ_s(N, 양) | N=1024 − N=64 [95% CI] | N=1024 − N=300 [95% CI] |', '|---|---|---|---|---|']
    for key, r in out['rows'].items():
        v, q = key.split(':')
        sc = 100 if q == 'D' else 1
        fmt = (lambda x: f'{x:+.0f}') if q == 'D' else (lambda x: f'{x:+.3f}')
        cells = [f"{fmt(sc * r[f'diff_1024_minus_{lo}'])} [{fmt(sc * r[f'diff_1024_minus_{lo}_ci'][0])}, "
                 f"{fmt(sc * r[f'diff_1024_minus_{lo}_ci'][1])}]" for lo in (64, 300)]
        L.append(f"| {'sh-iter' if v == 'shiter' else v} | {q + (' %p' if q == 'D' else '')} | {r['rho_N']:+.2f} | {cells[0]} | {cells[1]} |")
    if out['eo']:
        L += ['', '| EO 점 − E1 기준점(top-30, opt_steps 30) | D_det %p | D_sh-iter %p | 이득 D_det − D_sh-iter %p | drift det | '
              'sel_indep det | sel_same det |', '|---|---|---|---|---|---|---|']
        for lab, row in out['eo'].items():
            c = []
            for v, q in EOQ:
                r = row[f'{v}:{q}']
                c.append(f"{100 * r['diff']:+.0f} [{100 * r['ci'][0]:+.0f}, {100 * r['ci'][1]:+.0f}]" if q in ('D', 'gain')
                         else f"{r['diff']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]")
            L.append(f'| {lab} | ' + ' | '.join(c) + ' |')
    txt = '\n'.join(L) + '\n'
    print(txt)
    if a.md:
        open(a.md, 'w').write(txt)


if __name__ == '__main__':
    main()
