"""[post hoc] X17 (D5, DESIGN_tmlr.md 11.1, fixed before the data): PushT 4-bit (PT ranges) on E1's evaluation split
eval6 (pair seed 6, pairs 0-99, planner seeds 4000-4099, salt eval6) on an RTX 4090 (dw.6); none, det, sh-iter x 100.
Descriptive reading written before the data (D5):
  (i)  CI(D_det) lower > 0 and CI(D_det) overlaps E1's [0.14, 0.34]  -> the eval8 non-replication is a split difference
  (ii) CI(D_det) holds 0 and its upper bound < 0.14                  -> GPU / numerics sensitivity of 4-bit PushT det
  (iii) otherwise                                                     -> both explanations stay open
Also: none success and its episode agreement with E1's none (as the D3 Wall rerun), D and red of sh-iter.
Statistics: paired bootstrap 10,000, RandomState(0), 95% percentile CI; tolerance 1e-9 (analyze_tmlr).
usage: python analyze_x17.py <raw dir> <e1 reference json> [--pod dw.6] [--out DIR]"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_tmlr as AT

E1 = dict(source='A/main/results/E1/e1_dinowm_cells.csv (pusht, confirmatory, k=4; eval6, RTX A5000)', none=0.83,
          det=0.59, D_det=0.24, D_det_ci=[0.14, 0.34], shiter=0.83, D_shiter=0.0, D_shiter_ci=[-0.04, 0.04], red_shiter=1.0,
          red_shiter_ci=[0.84615, 1.2222])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw'); ap.add_argument('e1ref'); ap.add_argument('--pod', default='dw.6')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    R, metas = AT.load(a.raw)
    pod, task, split, k = a.pod, 'pusht', 'eval6', 4
    none = AT.none_of(R, pod, task, split=split)
    det = AT.cell(R, pod, task, 'main', 'det', k, split=split)
    sh = AT.cell(R, pod, task, 'main', 'shiter', k, split=split)
    sd, ss = AT.dstats(none, det), AT.dstats(none, sh)
    ss.update(AT.red_stats(none, det, sh) or {})
    ss['Dv_minus_Ddet'] = AT.contrast(none, sh, det)
    lo, hi = sd['D_ci']
    e1lo, e1hi = E1['D_det_ci']
    if AT.gt(lo, 0.0) and hi >= e1lo - AT.EPS and lo <= e1hi + AT.EPS:
        reading, label = '(i)', 'split difference (eval8) explains the non-replication'
    elif lo <= 0.0 + AT.EPS and hi >= 0.0 - AT.EPS and hi < e1lo - AT.EPS:
        reading, label = '(ii)', 'GPU / numerics sensitivity of 4-bit PushT det damage on the same pairs'
    else:
        reading, label = '(iii)', 'both explanations stay open'
    ref = json.load(open(a.e1ref))['success']
    agree = {}
    for v, cellv in (('none', none), ('det', det), ('shiter', sh)):
        m = {int(e): s for e, s in ref[v].items()}
        eps = sorted(set(cellv) & set(m))
        same = [bool(cellv[e]['success']) == m[e] for e in eps]
        both = np.array([bool(cellv[e]['success']) for e in eps]); e1s = np.array([m[e] for e in eps])
        n10, n01, p = AT.mcnemar(both, e1s)
        agree[v] = dict(n=len(eps), agreement=float(np.mean(same)), successes=int(both.sum()), e1_successes=int(e1s.sum()),
                        mcnemar=dict(this_only=n10, e1_only=n01, p=p))
    out = dict(posthoc=True, decision='D5 (DESIGN_tmlr.md 11.1), coordinator 2026-11-22T02:08Z, user approved',
               cell=dict(task=task, k=k, split=split, ranges='main', pod=pod,
                         none=dict(n=len(none), success=float(np.mean([r['success'] for r in none.values()])))),
               det=sd, shiter=ss, reading=reading, reading_label=label,
               rule='(i) CI lower > 0 and overlaps E1 [0.14, 0.34]; (ii) CI holds 0 and upper < 0.14; (iii) otherwise',
               agreement_with_E1=agree, E1=E1, gpus=sorted({m['meta'].get('gpu') for m in metas if m['tag'].startswith(pod)}))
    od = a.out or os.path.join(os.path.dirname(os.path.abspath(a.raw)), 'analysis_posthoc')
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(f'{od}/x17_pusht4_eval6.json', 'w'), indent=1, default=float)
    with open(f'{od}/x17_pusht4_eval6.csv', 'w') as f:
        f.write('variant,n,success,D,D_ci_lo,D_ci_hi,p_boot_le0,red,red_ci_lo,red_ci_hi,agreement_with_E1,E1_success\n')
        f.write(f"none,{len(none)},{out['cell']['none']['success']:.4g},,,,,,,,{agree['none']['agreement']:.4g},{E1['none']}\n")
        for v, s in (('det', sd), ('shiter', ss)):
            rc = s.get('red_ci') or ['', '']
            f.write(f"{v},{s['n']},{s['success']:.4g},{s['D']:.4g},{s['D_ci'][0]:.4g},{s['D_ci'][1]:.4g},{s['p_boot_le0']:.4g},"
                    f"{'' if s.get('red') is None else round(s['red'], 4)},{rc[0]},{rc[1]},{agree[v]['agreement']:.4g},"
                    f"{E1['det'] if v == 'det' else E1['shiter']}\n")
    pp = lambda x, c: f'{100 * x:+.0f} [{100 * c[0]:.0f}, {100 * c[1]:.0f}]'
    L = ['## [사후] X17: PushT 4-bit E1 분할(eval6) 재실행, RTX 4090 (D5)', '',
         '| 변형 | 성공 | D %p [CI] | red [CI] | E1 성공 | E1과 episode 일치 |', '|---|---|---|---|---|---|',
         f"| none | {100 * out['cell']['none']['success']:.0f}% | — | — | {100 * E1['none']:.0f}% | {100 * agree['none']['agreement']:.0f}% |",
         f"| det | {100 * sd['success']:.0f}% | {pp(sd['D'], sd['D_ci'])} | — | {100 * E1['det']:.0f}% | {100 * agree['det']['agreement']:.0f}% |",
         f"| sh-iter | {100 * ss['success']:.0f}% | {pp(ss['D'], ss['D_ci'])} | "
         f"{'—' if ss.get('red') is None else '%.2f [%.2f, %.2f]' % (ss['red'], ss['red_ci'][0], ss['red_ci'][1])} | "
         f"{100 * E1['shiter']:.0f}% | {100 * agree['shiter']['agreement']:.0f}% |", '',
         f"- 판정(D5, 미리 적은 기술): **{reading}** {label}. D_det {pp(sd['D'], sd['D_ci'])} (E1 +24 [14, 34]); 한쪽 p {sd['p_boot_le0']:.4g}",
         f"- E1 none과 비교: 이 pod {agree['none']['successes']}/100, E1 {agree['none']['e1_successes']}/100, episode 일치 "
         f"{100 * agree['none']['agreement']:.0f}%, McNemar p {agree['none']['mcnemar']['p']:.3g}"]
    open(f'{od}/tables_x17.md', 'w').write('\n'.join(L) + '\n')
    print(json.dumps(dict(reading=reading, D_det=sd['D'], D_det_ci=sd['D_ci'], none=out['cell']['none']['success'],
                          agree_none=agree['none']['agreement'], D_shiter=ss['D'], red_shiter=ss.get('red'))))


if __name__ == '__main__':
    main()
