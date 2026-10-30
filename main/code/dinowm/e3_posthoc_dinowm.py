"""POST-HOC description of the DINO-WM E3 table-size sweep (written 2026-10-23 after the analyze_e3.py tables were seen;
not part of any pre-registered rule, no verdict). Paired contrasts between the shared fixed-table levels on the same
pairs (the per-level CIs vs none mostly carry the between-pair spread): D(L1) - D(L0), D(L2) - D(L1), D(L3) - D(L2),
D(L3) - D(L0), and drift(L3) - drift(L0), OEI(L3) - OEI(L0); paired bootstrap (10,000, RandomState(0), percentile 95%).
usage: python e3_posthoc_dinowm.py <results dir> --out JSON [--md MD]
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A
import analyze_e3 as E

PAIRS = [('L1', 'L0'), ('L2', 'L1'), ('L3', 'L2'), ('L3', 'L0')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('res'); ap.add_argument('--out', required=True); ap.add_argument('--md', default=None)
    a = ap.parse_args()
    R = {}
    for sub in ('E1_dinowm_wp', 'E1_dinowm_pusht', 'E3_dinowm'):
        R.update(A.load(f'{a.res}/{sub}')[0])
    var = {lab: v for lab, v, _ in E.LEVELS}
    out = dict(note='post-hoc description (after the E3 tables were seen); no verdict', tasks={})
    drift = lambda r: r['oei']['last_pop_dJ'] - r['oei']['it0_rand_dJ']
    for t, k, role in E.SWEEP:
        x = {}
        for hi, lo in PAIRS:
            x[f'D_{hi}_minus_D_{lo}'] = E.paired_diff(R, t, k, var[hi], var[lo])
        x['drift_L3_minus_drift_L0'] = E.paired_field_diff(R, t, k, var['L3'], var['L0'], drift)
        x['OEI_L3_minus_OEI_L0'] = E.paired_field_diff(R, t, k, var['L3'], var['L0'], lambda r: r['oei']['oei_ref_mf'])
        out['tasks'][f'{t}_k{k}'] = dict(task=t, k=k, role=role, contrasts=x)
    json.dump(out, open(a.out, 'w'), indent=1, default=float, ensure_ascii=False)
    L = ['| 과제 | k | 역할 | D(L1) − D(L0) %p | D(L2) − D(L1) %p | D(L3) − D(L2) %p | D(L3) − D(L0) %p | drift(L3) − drift(L0) | OEI(L3) − OEI(L0) |',
         '|' + '---|' * 9]
    for key, y in out['tasks'].items():
        c = y['contrasts']
        pp = lambda d: f"{100 * d['diff']:+.0f} [{100 * d['ci'][0]:+.0f}, {100 * d['ci'][1]:+.0f}]"
        f3 = lambda d: f"{d['diff']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]"
        L.append(f"| {E.TN[y['task']]} | {y['k']} | {y['role']} | " + ' | '.join(pp(c[f'D_{hi}_minus_D_{lo}']) for hi, lo in PAIRS)
                 + f" | {f3(c['drift_L3_minus_drift_L0'])} | {f3(c['OEI_L3_minus_OEI_L0'])} |")
    txt = '\n'.join(L) + '\n'
    print(txt)
    if a.md:
        open(a.md, 'w').write(txt)


if __name__ == '__main__':
    main()
