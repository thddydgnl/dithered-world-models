"""[post hoc] analysis of one extra E11 cell (coordinator 2026-11-22T01:36Z, user approved): PushT 3-bit, PT ranges,
eval8, none + 8 variants on pod dw.6. The E11 rules of PLAN 2.8 are applied to this cell alone; the pre-specified
verdicts B2, B3 and Dg1 (DINO-WM unit = PushT 4, PointMaze 3, Wall 3) are NOT changed. Pure python; runs on a pod.

usage: python analyze_posthoc.py <raw dir with <pod>/jobs, <pod>/splits> <main analysis_dw.json> [--pod dw.6] [--k 3]
                                 [--out DIR]
  damage real : one-sided paired bootstrap p(D_det <= 0) < 0.05 for this cell alone (no Holm: a single cell)
  class       : bias (det+BC red >= 0.5) / persistence (not bias, Delta_pers CI lower > 0) / resolution (not bias,
                det+shift-fixed red >= 0.5 and Delta_pers CI holds 0) / mixed, as analyze_tmlr.e11
  also        : D, red, D_v - D_det, Delta_bias, Delta_pers, auxiliary contrasts, drift ratios, planning-time 1/3/5-step
                errors, the D1 overlap-cluster bootstrap (D, Delta_bias, Delta_pers, cluster-based reading), the E1 PushT
                3-bit cell (A/main/results/E1/e1_dinowm_cells.csv) and a [post hoc] descriptive DINO-WM reading that adds
                this cell to the damaged cells of the main analysis (Wall 3-bit)
Statistics as analyze_tmlr (10,000 paired resamples, RandomState(0), 95% percentile CI, tolerance 1e-9)."""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_tmlr as AT
import podplan as PP

# E1 PushT 3-bit (eval6, confirmatory, damaged): A/main/results/E1/e1_dinowm_cells.csv
E1_PUSHT3 = dict(source='A/main/results/E1/e1_dinowm_cells.csv (pusht, confirmatory, k=3)', none=0.83, det=0.47,
                 D_det=0.36, D_det_ci=[0.27, 0.45],
                 red=dict(ind=0.97222, shiter=0.91667, shstep=0.88889, detshift=0.91667),
                 success=dict(ind=0.82, shiter=0.80, shstep=0.79, detshift=0.80))


def classify(V, d_pers):
    fx = lambda v: AT.ge(V.get(v, {}).get('red'), 0.5)
    lo, hi = d_pers['ci'] if d_pers else [np.nan, np.nan]
    if 'detbc' not in V:
        return 'undetermined'
    if fx('detbc'):
        return 'bias'
    if AT.gt(lo, 0.0):
        return 'persistence'
    if fx('detshiftfix') and AT.le(lo, 0.0) and AT.ge(hi, 0.0):
        return 'resolution'
    return 'mixed'


def cell_analysis(R, raw, pod, task, k):
    none = AT.none_of(R, pod, task)
    V = {v: AT.cell(R, pod, task, 'main', v, k) for v in PP.E11_VARS}
    out = dict(task=task, k=k, pod=pod, none=dict(n=len(none), success=float(np.mean([r['success'] for r in none.values()]))),
               variants={})
    for v, c in V.items():
        if not c:
            continue
        s = AT.dstats(none, c)
        if v != 'det':
            s.update(AT.red_stats(none, V['det'], c) or {})
            s['Dv_minus_Ddet'] = AT.contrast(none, c, V['det'])
        for st in ('1', '3', '5'):
            s[f'rel_err{st}'] = AT.mean_rec(c, 'err', (st, 'rel'))
        dr = [r['oei']['last_pop_dJ'] - r['oei']['it0_rand_dJ'] for r in c.values() if 'oei' in r]
        s['drift'] = float(np.mean(dr)) if dr else None
        out['variants'][v] = s
    dd = out['variants']['det'].get('drift')
    for v, s in out['variants'].items():
        if v != 'det' and s.get('drift') is not None and dd is not None and abs(dd) > 1e-12:
            s['drift_ratio_vs_det'] = s['drift'] / dd
    det = out['variants']['det']
    out['d_bias'] = AT.contrast(none, V['det'], V['detbc'])
    out['d_pers'] = AT.contrast(none, V['detshiftfix'], V['detshift'])
    out['aux'] = {name: AT.contrast(none, V[a], V[b]) for name, a, b in (
        ('D(sh-plan)-D(sh-iter)', 'shstep', 'shiter'), ('D(sh-iter+b)-D(sh-iter)', 'shiterb', 'shiter'),
        ('D(det+noise)-D(det)', 'detnoise', 'det'), ('D(det+shift)-D(sh-iter)', 'detshift', 'shiter'))}
    out['damaged'] = bool(det['p_boot_le0'] < AT.ALPHA and det['D'] > 0)
    out['class'] = classify(out['variants'], out['d_pers']) if out['damaged'] else None
    out['fixes'] = {v: AT.ge(out['variants'][v].get('red'), 0.5) for v in PP.E11_VARS if v != 'det'}
    comp = AT.clusters_of(raw)
    if comp is not None:
        cl = dict(D={v: AT.cluster_contrast(none, V[v], None, comp) for v in PP.E11_VARS if V[v]},
                  d_bias=AT.cluster_contrast(none, V['det'], V['detbc'], comp),
                  d_pers=AT.cluster_contrast(none, V['detshiftfix'], V['detshift'], comp))
        cl['damaged'] = bool(cl['D']['det']['p_le0'] < AT.ALPHA and det['D'] > 0)
        cl['class'] = classify(out['variants'], cl['d_pers']) if cl['damaged'] else None
        cl['reading_differs'] = bool(cl['damaged'] != out['damaged'] or cl['class'] != out['class'])
        out['cluster'] = cl
    out['E1'] = E1_PUSHT3
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw'); ap.add_argument('main_analysis'); ap.add_argument('--pod', default='dw.6')
    ap.add_argument('--k', type=int, default=3); ap.add_argument('--task', default='pusht'); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    R, metas = AT.load(a.raw)
    C = cell_analysis(R, a.raw, a.pod, a.task, a.k)
    M = json.load(open(a.main_analysis))
    main_damaged = [c for c in M['E11']['cells'] if c.get('damaged')]
    cells = [dict(task=c['task'], k=c['k'], cls=c.get('class'), detnoise_fixes=c['fixes'].get('detnoise'))
             for c in main_damaged]
    if C['damaged']:
        cells.append(dict(task=C['task'], k=C['k'], cls=C['class'], detnoise_fixes=C['fixes'].get('detnoise')))
    counts = {}
    for c in cells:
        counts[c['cls']] = counts.get(c['cls'], 0) + 1
    maj = [k for k, n in counts.items() if n > len(cells) / 2 + AT.EPS]
    combined = dict(note='[post hoc, descriptive] main-analysis damaged cells plus this cell; B2, B3, Dg1 are not changed',
                    cells=cells, class_counts=counts, reading=(maj[0] if maj else 'mixed') if cells else 'no damaged cell',
                    detnoise_fixes=sum(1 for c in cells if c['detnoise_fixes']), n_damaged=len(cells))
    out = dict(posthoc=True, decided='coordinator 2026-11-22T01:36Z (user approved)', cell=C, combined_dinowm=combined,
               n_jobs=len([m for m in metas if m['tag'].startswith(a.pod)]),
               gpus=sorted({m['meta'].get('gpu') for m in metas if m['tag'].startswith(a.pod)}))
    od = a.out or os.path.join(os.path.dirname(os.path.abspath(a.raw)), 'analysis_posthoc')
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(f'{od}/posthoc_pusht{a.k}.json', 'w'), indent=1, default=float)
    keys = ['variant', 'n', 'success', 'D', 'D_ci_lo', 'D_ci_hi', 'p_boot_le0', 'red', 'red_ci_lo', 'red_ci_hi', 'DvDdet',
            'DvDdet_lo', 'DvDdet_hi', 'rel_err1', 'rel_err3', 'rel_err5', 'drift_ratio_vs_det', 'cluster_D_lo', 'cluster_D_hi']
    with open(f'{od}/posthoc_pusht{a.k}.csv', 'w') as f:
        f.write(','.join(keys) + '\n')
        for v in PP.E11_VARS:
            s = C['variants'].get(v)
            if not s:
                continue
            dv = s.get('Dv_minus_Ddet') or {}
            cl = (C.get('cluster') or {}).get('D', {}).get(v) or {}
            row = [v, s['n'], s['success'], s['D'], s['D_ci'][0], s['D_ci'][1], s['p_boot_le0'], s.get('red'),
                   (s.get('red_ci') or [None, None])[0], (s.get('red_ci') or [None, None])[1], dv.get('diff'),
                   (dv.get('ci') or [None, None])[0], (dv.get('ci') or [None, None])[1], s.get('rel_err1'),
                   s.get('rel_err3'), s.get('rel_err5'), s.get('drift_ratio_vs_det'),
                   (cl.get('ci') or [None, None])[0], (cl.get('ci') or [None, None])[1]]
            f.write(','.join('' if x is None else (f'{x:.6g}' if isinstance(x, float) else str(x)) for x in row) + '\n')
    pp = lambda x, c=None: '—' if x is None else (f'{100 * x:+.0f}' + (f' [{100 * c[0]:.0f}, {100 * c[1]:.0f}]' if c else ''))
    L = [f"## [사후] PushT {a.k}-bit E11 칸 (pod {a.pod})", '',
         f"- none {100 * C['none']['success']:.0f}% (n {C['none']['n']}); det 손상 실재(한쪽 p < 0.05, 이 칸 하나): "
         f"{C['damaged']} (p = {C['variants']['det']['p_boot_le0']:.4g}); 분류: {C['class']}", '',
         '| 변형 | 성공 | D %p [CI] | red [CI] | D_v − D_det %p [CI] | 1-step / 5-step | drift 비 | 덩어리 D CI |',
         '|---|---|---|---|---|---|---|---|']
    for v in PP.E11_VARS:
        s = C['variants'].get(v)
        if not s:
            continue
        red = '—' if s.get('red') is None else f"{s['red']:.2f} [{s['red_ci'][0]:.2f}, {s['red_ci'][1]:.2f}]"
        dv = s.get('Dv_minus_Ddet')
        cl = (C.get('cluster') or {}).get('D', {}).get(v)
        L.append(f"| {AT.VN[v]} | {100 * s['success']:.0f}% | {pp(s['D'], s['D_ci'])} | {red} | "
                 f"{pp(dv['diff'], dv['ci']) if dv else '—'} | {s.get('rel_err1') or float('nan'):.4f} / "
                 f"{s.get('rel_err5') or float('nan'):.4f} | {s.get('drift_ratio_vs_det', float('nan')):.2f} | "
                 f"{'[%.0f, %.0f]' % (100 * cl['ci'][0], 100 * cl['ci'][1]) if cl else '—'} |")
    L += ['', f"- Δ_bias {pp(C['d_bias']['diff'], C['d_bias']['ci'])}, Δ_pers {pp(C['d_pers']['diff'], C['d_pers']['ci'])}; 보조 " +
          '; '.join(f"{k} {pp(v['diff'], v['ci'])}" for k, v in C['aux'].items() if v)]
    if C.get('cluster'):
        cl = C['cluster']
        L.append(f"- 덩어리 bootstrap({cl['D']['det']['n_clusters']}덩어리): det D CI [{100 * cl['D']['det']['ci'][0]:.0f}, "
                 f"{100 * cl['D']['det']['ci'][1]:.0f}], p {cl['D']['det']['p_le0']:.4g}; Δ_bias [{100 * cl['d_bias']['ci'][0]:.0f}, "
                 f"{100 * cl['d_bias']['ci'][1]:.0f}], Δ_pers [{100 * cl['d_pers']['ci'][0]:.0f}, {100 * cl['d_pers']['ci'][1]:.0f}]; "
                 f"덩어리 손상 {cl['damaged']}, 분류 {cl['class']}, 읽기 차이 {cl['reading_differs']}")
    L.append(f"- E1 같은 칸(eval6): none 83%, det 47%, D_det +36 [27, 45], red ind 0.97, sh-iter 0.92, sh-plan 0.89, det+shift 0.92 "
             f"({E1_PUSHT3['source']})")
    L.append(f"- [사후 기술] DINO-WM 손상 칸(본 분석 Wall 3 + 이 칸): {combined['cells']} → 읽기 {combined['reading']}, "
             f"det+noise가 고친 칸 {combined['detnoise_fixes']}/{combined['n_damaged']} (B2·B3·Dg1 판정은 바꾸지 않음)")
    open(f'{od}/tables_posthoc.md', 'w').write('\n'.join(L) + '\n')
    print(json.dumps(dict(damaged=C['damaged'], p=C['variants']['det']['p_boot_le0'], D_det=C['variants']['det']['D'],
                          cls=C['class'], cluster_differs=(C.get('cluster') or {}).get('reading_differs'),
                          combined=combined['reading']), default=float))
    print('wrote', od)


if __name__ == '__main__':
    main()
