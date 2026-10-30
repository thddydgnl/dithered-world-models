"""H4b inputs of the DINO-WM family (DESIGN_main v5 7.2) from E1 data only, with the functions of analyze_e3.py
(near_tie, paired_diff). CPU only. The 15-task correlation (with the 12 TD-MPC2 tasks) is the coordinator's.

usage: python h4b_dinowm.py <dinowm_main dir> --out JSON [--md MD]

Per task at the primary k (PushT 4, Wall 5, PointMaze 3):
  near-tie density = -median over the 100 E1 none episodes of (top1 - top2 gap) / std of the unquantized costs of the
    last CEM iteration's candidate set (all N = 300 rows; analyze_e3.near_tie default); auxiliary: the iteration-0
    candidate set. CI: bootstrap over the episodes of -median (10,000, RandomState(0)).
  CRN gain = D_ind - D_sh-iter (= success(sh-iter) - success(ind) on the same pairs) of the E1 primary-k cell, paired
    bootstrap CI (10,000, RandomState(0)); D_ind and D_sh-iter vs none with their CIs for reference.
Post-hoc sensitivity (labelled, not an input): the density with the std over the sampled rows 1.. only (TD-MPC2
convention: the policy-prior proposals are excluded from its std, here row 0 = the current CEM mean).
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A
import analyze_e3 as E3

SPLIT = dict(pusht='eval6', wall='eval', pointmaze='eval')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root'); ap.add_argument('--out', required=True); ap.add_argument('--md', default=None)
    a = ap.parse_args()
    res, pops = f'{a.root}/results', f'{a.root}/pops'
    R = {}
    for sub in ('E1_dinowm_wp', 'E1_dinowm_pusht'):
        R.update(A.load(f'{res}/{sub}')[0])
    out = dict(rule='DESIGN_main v5 7.2 H4b inputs, DINO-WM: near-tie density = -median(gap12 / candidate std) of the '
                    'E1 none last-iteration population (auxiliary: first iteration); CRN gain = D_ind - D_sh-iter at '
                    'the primary k (E1 cell). Confirmatory H4b uses the 12 TD-MPC2 tasks; the DINO-WM tasks enter the '
                    'auxiliary 15-task Spearman (coordinator).',
               tasks={})
    for t, k in E3.PRIMARY.items():
        g = E3.paired_diff(R, t, k, 'ind', 'shiter')
        ci_ind, ci_sh = A.cell_stats(R, t, 'ind', k), A.cell_stats(R, t, 'shiter', k)
        none = A.cell_stats(R, t, 'none', 0)
        nt = E3.near_tie(pops, t)
        nt1 = E3.near_tie(pops, t, key='it0_J_none')
        sens = E3.near_tie(pops, t, std_rows='rand')
        sens1 = E3.near_tie(pops, t, key='it0_J_none', std_rows='rand')
        out['tasks'][t] = dict(
            k=k, split=SPLIT[t], n_pairs=g['n'],
            near_tie_density_last=nt['density'], near_tie_density_last_ci=nt['density_ci'],
            near_tie_density_first=nt1['density'], near_tie_density_first_ci=nt1['density_ci'],
            crn_gain=g['diff'], crn_gain_ci=g['ci'],
            D_ind=ci_ind['D'], D_ind_ci=ci_ind['D_ci'], D_shiter=ci_sh['D'], D_shiter_ci=ci_sh['D_ci'],
            success=dict(none=none['success'], ind=ci_ind['success'], shiter=ci_sh['success']),
            near_tie_last_detail=nt, near_tie_first_detail=nt1,
            posthoc_sensitivity_std_rows_rand=dict(last=sens, first=sens1))
    json.dump(out, open(a.out, 'w'), indent=1, default=float, ensure_ascii=False)
    print(json.dumps({t: {kk: v for kk, v in x.items() if not kk.endswith('detail') and not kk.startswith('posthoc')}
                      for t, x in out['tasks'].items()}, indent=1, default=float))
    if a.md:
        open(a.md, 'w').write(table(out))


def table(out):
    def f3(x, c):
        return f'{x:+.3f} [{c[0]:+.3f}, {c[1]:+.3f}]'

    def pp(x, c):
        return f'{100 * x:+.0f} [{100 * c[0]:.0f}, {100 * c[1]:.0f}]'
    TN = dict(pusht='PushT', wall='Wall', pointmaze='PointMaze')
    L = ['| 과제 | k | n | near-tie 밀도(마지막 반복) [95% CI] | 보조: 밀도(첫 반복) [95% CI] | gap/std 중앙값 [사분위](마지막) | '
         'CRN 이득 D_ind − D_sh-iter %p [95% CI] | D_ind %p | D_sh-iter %p |', '|' + '---|' * 9]
    for t, x in out['tasks'].items():
        d = x['near_tie_last_detail']
        L.append(f"| {TN[t]} | {x['k']} | {x['n_pairs']} | {f3(x['near_tie_density_last'], x['near_tie_density_last_ci'])} | "
                 f"{f3(x['near_tie_density_first'], x['near_tie_density_first_ci'])} | {d['gap_over_std_median']:.3f} "
                 f"[{d['gap_over_std_q25']:.3f}, {d['gap_over_std_q75']:.3f}] | {pp(x['crn_gain'], x['crn_gain_ci'])} | "
                 f"{pp(x['D_ind'], x['D_ind_ci'])} | {pp(x['D_shiter'], x['D_shiter_ci'])} |")
    L += ['', '사후 민감도(입력 아님): std를 무작위 후보(행 1..)로만 잰 밀도', '', '| 과제 | 마지막 반복 | 첫 반복 |', '|---|---|---|']
    for t, x in out['tasks'].items():
        s = x['posthoc_sensitivity_std_rows_rand']
        L.append(f"| {TN[t]} | {f3(s['last']['density'], s['last']['density_ci'])} | {f3(s['first']['density'], s['first']['density_ci'])} |")
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()
