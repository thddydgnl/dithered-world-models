"""E4: markdown tables for reports/E4_theory.md from the analysis outputs (e4_comparisons.json, cells_*.csv).
usage: python e4_report_tables.py <analysis_dir> > e4_tables.md"""
import csv
import json
import math
import os
import sys


def f(x, nd=3):
    if x is None:
        return '—'
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(x):
        return '—'
    if x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e4):
        return f'{x:.2e}'
    return f'{x:.{nd}g}'


def _cell(c):
    return str(c).replace('|', ' / ')


def table(hdr, rows):
    out = ['| ' + ' | '.join(_cell(h) for h in hdr) + ' |', '|' + '---|' * len(hdr)]
    for r in rows:
        out.append('| ' + ' | '.join(_cell(c) for c in r) + ' |')
    return '\n'.join(out)


def load_cells(d, pl):
    rows = []
    with open(os.path.join(d, f'cells_{pl}.csv')) as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return rows


def num(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return float('nan')


def main():
    d = sys.argv[1]
    C = json.load(open(os.path.join(d, 'e4_comparisons.json')))
    P = []
    # ---- C1
    c1 = C.get('C1') or {}
    if c1:
        P.append('### C1 단발 선택 (블록 A)\n')
        rows = [[op, s.get('n'), f(s.get('frac_within_2se')), f(s.get('median_rel_err')), f(s.get('slope_through_origin'))]
                for op, s in sorted(c1.get('normal_by_op', {}).items())]
        P.append(table(['연산자(정규 J, e)', '칸', '2 SE 안 비율', '상대 오차 중앙값', '기울기'], rows))
        P.append('')
        qt = [r for r in c1.get('quad_table', []) if r['op'] == 'argmin']
        rows = [[r['dist'], f(r['ratio']), f(r['meas_sub'] if r.get('meas_sub') is not None else r['meas']), f(r['pred']),
                 f(r.get('cond_exact')), f(r.get('near_tie'))] for r in qt]
        P.append(table(['2차 첫 모집단', 'σ_e/σ_J', '측정(σ_J 단위)', '2변량', '고정 J 정확식', 'near-tie 수'], rows))
        qa = c1.get('quad_argmin_refined', {})
        P.append('\n- argmin 요약(셀 기준): ' + '; '.join(
            f"{k}: 상대 오차 중앙값 {f(v.get('median_rel_err'))}, 2 SE 안 {f(v.get('frac_within_2se'))}" for k, v in qa.items()))
        qb = c1.get('quad_bivariate', {})
        P.append('- 2변량 공식(2차 첫 모집단): ' + '; '.join(
            f"{k}: 상대 오차 중앙값 {f(v.get('median_rel_err'))}" for k, v in qb.items()))
        P.append('')
    # ---- C2
    c2 = C.get('C2') or {}
    if c2:
        P.append('### C2 평가 안 선택 항(같은 평가값) 대 예측\n')
        rows = []
        for k in sorted(c2):
            s = c2[k]
            parts = k.split('|')
            if (len(parts) < 3 and parts[-1] != 'all') or not s.get('n'):
                continue
            rows.append([k, s.get('n'), s.get('n_degenerate', 0), f(s.get('slope_through_origin')), f(s.get('R2')),
                         f(s.get('median_rel_err_signif')), f(s.get('frac_within_2se')), f(s.get('corr'))])
        P.append(table(['비교|모형군|단계', '칸', '퇴화 제외(사후)', '기울기', 'R²', '상대 오차 중앙값(유의 칸)', '2 SE 안', 'corr'], rows))
        P.append('')
    # ---- C3
    c3 = C.get('C3') or {}
    if c3:
        P.append('### C3 OEI 분해 점검\n')
        rows = [[k, s['cells'], f(s['frac_drift_ci_covers_0']), f(s['frac_drift_ci_below_0']), f(s['frac_redraw_sel_ci_covers_0']),
                 f(s['frac_samedraw_sel_ci_below_0'])] for k, s in sorted(c3.items())]
        P.append(table(['계획기|모형군', '칸', 'drift CI가 0 포함', 'drift CI < 0', '재평가 sel CI가 0 포함', '같은 평가 sel CI < 0'], rows))
        P.append('')
    # ---- C4
    c4 = C.get('C4') or {}
    if c4:
        P.append('### C4 SAA(M1)\n')
        rows = []
        for k, o in sorted(c4.items()):
            dv = o.get('drift_vs_pred', {})
            om = o.get('oei_mf_vs_pred', {})
            rows.append([k, o['n_cells'], f(o['median_shift_slope']), f(o['median_shift_R2']), f(dv.get('slope_through_origin')),
                         f(dv.get('R2')), f(om.get('slope_through_origin')), f(om.get('R2'))])
        P.append(table(['계획기|범위', '칸', '이동 기울기 중앙값', '이동 R² 중앙값', 'drift 기울기', 'drift R²', 'OEI_mf 기울기', 'OEI_mf R²'], rows))
        rows = []
        for k, o in sorted(c4.items()):
            for sc, b in sorted(o.get('by_scale', {}).items(), key=lambda kv: float(kv[0])):
                rows.append([k, sc, b['n'], f(b['shift_slope']), f(b['shift_R2'])])
        P.append('\n' + table(['계획기|범위', 'r', '칸', '이동 기울기', '이동 R²'], rows))
        P.append('')
    # ---- C5
    c5 = C.get('C5') or {}
    if c5.get('rows'):
        P.append('### C5 새 독립 잡음의 정상 손상(MPPI 100회, 블록 V)\n')
        rows = [[r['k'], f(r['scale']), f(r['keff']), f(r['S']), f(r['sigma_eps']), f(r['dmg_mean']) + ' ± ' + f(r['dmg_mean_se']),
                 f(r['pred_regret_mean'])] for r in sorted(c5['rows'], key=lambda r: (r['scale'], r['k']))]
        P.append(table(['k', 'r', 'k_eff', 'S', 'σ_ε', '손상(평균 계획)', '예측 dσ/(4Sk_eff)'], rows))
        P.append('\n- 적합: ' + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in c5.get('fit', {}).items()}, ensure_ascii=False))
        P.append('- log-log 기울기(손상 대 k_eff): ' + ', '.join(f"{k.split('|')[1]}: {f(v)}" for k, v in c5.items() if k.startswith('loglog')))
        P.append('')
    # ---- C6
    c6 = C.get('C6') or {}
    if c6:
        P.append('### C6 반복 수\n')
        rows = [[k, s['groups'], f(s['frac_dmg_decreases_first_to_last']), f(s['mean_ratio_last_over_first'])]
                for k, s in sorted(c6.items()) if k != 'closed_loop']
        P.append(table(['계획기|모형군', 'group', '첫 → 마지막 기록에서 손상 감소 비율', '손상 비(마지막/첫) 평균'], rows))
        W = c6.get('closed_loop', {})
        if W:
            rows = []
            for ws in (1, 0):
                for m in ('M1j', 'Mstep', 'M3', 'M2'):
                    for sc in ('8', '32'):
                        vals = [W.get(f'ws{ws}|n{n}|{m}|{sc}') for n in (1, 3, 6, 12)]
                        rows.append([f'ws{ws}', m, sc] + [f(v['rel_damage']) + ' ± ' + f(v['se'], 2) if v else '—' for v in vals])
                for sc in ('8', '32'):
                    vals = [W.get(f'ws{ws}|n{n}|Mstep_minus_M3|{sc}') for n in (1, 3, 6, 12)]
                    rows.append([f'ws{ws}', 'Mstep − M3', sc] + [f(v['rel_diff']) + ' ± ' + f(v['se'], 2) if v else '—' for v in vals])
            P.append('\n' + table(['warm start', '모형', 'r', 'n = 1', 'n = 3', 'n = 6', 'n = 12'], rows))
        P.append('')
    # ---- C7
    c7 = C.get('C7') or {}
    if c7:
        P.append('### C7 N 용량-반응 (H2c 예측, 블록 M, 마지막 반복)\n')
        rows = [[k, s['groups'], f(s['median_spearman_N_absoei']), f(s['frac_spearman_N_absoei_ge_0p8']),
                 f(s['median_ratio_absoei_1024_64']), f(s['median_spearman_N_dmg']), f(s['frac_spearman_dmg_absoei_ge_0p8'])]
                for k, s in sorted(c7.items())]
        P.append(table(['계획기|elite 규칙|모형군', 'group', 'Spearman(N, |OEI|) 중앙값', '≥ 0.8 비율', '|OEI| 비(1024/64) 중앙값',
                        'Spearman(N, 손상) 중앙값', 'Spearman(손상, |OEI|) ≥ 0.8 비율'], rows))
        cur = C.get('C7_curves', {})
        rows = []
        for key in sorted(cur):
            pl, reg, rule, model, sc = key.split('|')
            if reg != 'sep' or sc not in ('8.0', '32.0') or model not in ('M1', 'R-det', 'R-fixed', 'M2', 'M3', 'R-shiter', 'R-detshift'):
                continue
            c = cur[key]
            rows.append([pl, rule, model, sc] + [f(abs(x), 3) for x in c['abs_oei']] + [f(c['spearman_N_absoei'], 2)])
        P.append('\n- |OEI| 대 N(64, 128, 256, 512, 1024), sep\n')
        P.append(table(['계획기', 'elite', '모형', 'r', 'N64', 'N128', 'N256', 'N512', 'N1024', 'Spearman'], rows))
        P.append('')
    # ---- C8
    c8 = C.get('C8') or {}
    if c8:
        P.append('### C8 CRN(블록 C, 같은 주변 진폭)\n')
        rows = []
        for k, L in sorted(c8.items()):
            for x in sorted(L, key=lambda x: x['rho_c']):
                rows.append([k, x['model'], f(x['rho_c']), f(x['sel']), f(x['dmg']), f(x['oei']), f(x.get('se0'))])
        P.append(table(['계획기|regime|r', '모형', 'ρ_c', 'sel_last', '손상', 'OEI', 'σ_e0(측정)'], rows))
        P.append('')
    # ---- C9
    c9 = C.get('C9') or {}
    if c9:
        P.append('### C9 고정 표 크기 N_eff\n')
        ev = c9.get('exact_vs_measured', {})
        P.append(f"- 정확식(2.6.1) 대 측정(표 32개 평균): 칸 {ev.get('n')}, 측정/정확 비 중앙값 {f(ev.get('median_ratio_meas_over_exact'))}, "
                 f"2 SE 안 {f(ev.get('frac_within_2se'))}, R² {f(ev.get('R2'))}\n")
        rows = []
        for k, o in sorted(c9.get('fits', {}).items(), key=lambda kv: (kv[0].split('|')[0], float(kv[0].split('|')[1]))):
            v = o['var_exact']
            rows.append([k, f(o['sigma_y']), f(o['A']), f(o['B']), f(o['R2']), f(o['loglog_slope']), f(v[0] / v[-1]),
                         f(o['var_det'] / v[0]), f(o['var_shiter'] / v[0]), f(o['var_ind'] / v[0])])
        P.append(table(['toy|퍼짐', 'σ_y', 'A', 'B', 'R²(A/N+B)', 'log-log 기울기', 'Var(N_eff=4096)/Var(1)',
                        'det/fixed4096', 'shiter/fixed4096', 'ind/fixed4096'], rows))
        pv = c9.get('planner_vs_neff', {})
        rows = []
        for k, o in sorted(pv.items()):
            pts = o['points']
            rows.append([k] + [f(x['dmg']) for x in pts] + [f(o['spearman_neff_dmg'], 2), f(o['spearman_neff_absoei'], 2),
                                                             f(o.get('posthoc_loglog_slope_dmg_vs_var0'), 2),
                                                             f(o.get('ref_det', {}).get('dmg')), f(o.get('ref_shiter', {}).get('dmg')),
                                                             f(o.get('ref_detshift', {}).get('dmg')), f(o.get('ref_ind', {}).get('dmg'))])
        if rows:
            ne = [str(x['n_eff']) for x in next(iter(pv.values()))['points']]
            P.append('\n- 계획기 손상(고정 표) 대 N_eff\n')
            P.append(table(['계획기|regime|toy|r'] + [f'N_eff {n}' for n in ne] + ['Spearman(N_eff, 손상)', 'Spearman(N_eff, abs OEI)',
                                                                         '(사후) log 손상 / log 분산 기울기', 'det', 'shiter', 'det+shift', 'ind'], rows))
        P.append('')
    # ---- C10
    c10 = C.get('C10') or {}
    if c10.get('summary'):
        P.append('### C10 det+shift 대 sh-iter\n')
        rows = [[k, s['cells'], f(s['frac_ds_le_sh']), f(s['frac_ds_worse_signif']), f(s['median_rel_dmg'])]
                for k, s in sorted(c10['summary'].items())]
        P.append(table(['블록|계획기', '칸', 'det+shift 손상 ≤ sh-iter 비율', 'det+shift가 유의하게 나쁜 비율(2 SE)', '손상 비 중앙값(ds/sh)'], rows))
        P.append('')
    # ---- C11
    c11 = C.get('C11') or {}
    if c11:
        P.append('### C11 Jensen 편향\n')
        bs = c11.get('bit_slopes', {})
        rows = [[k, f(v['log2_sd_bias_per_bit']), f(v['log2_sd_det_per_bit'])] for k, v in sorted(bs.items())]
        P.append(table(['퍼짐|κ', 'log2 sd(b2) / bit (예측 −2)', 'log2 sd(det) / bit (예측 −1)'], rows))
        dr = c11.get('direct', {})
        rows = [[k, f(v['sd_det']), f(v['sd_bias_SR']), f(v['sd_bias_detshift']), f(v['ratio_bias_det'])]
                for k, v in sorted(dr.items(), key=lambda kv: (float(kv[0].split('|')[0]), float(kv[0].split('|')[1]), int(kv[0].split('|')[2])))]
        P.append('\n' + table(['퍼짐|κ|bits', 'sd(det 오차)', 'sd(b2), SR', 'sd(b2), det+shift', 'sd 비 SR/det'], rows))
        rt = c11.get('ratio', {})
        rows = [[k, f(v['drift_det']), f(v['drift_SR']) + ' ± ' + f(v['drift_SR_se'], 2), f(v['drift_detshift']), f(v['meas_ratio']),
                 f(v['pred_ratio_var'])] for k, v in sorted(rt.items())]
        P.append('\n' + table(['계획기|regime|κ|bits', 'drift det', 'drift SR(ind, shiter 평균)', 'drift det+shift', '측정 비 (SR − ds)/det',
                               '예측 비 (sd 비)²'], rows))
        P.append('')
    # ---- C12
    c12 = C.get('C12') or {}
    if c12:
        P.append('### C12 지속 지수와 계획 편차\n')
        rows = [[k, s['cells'], f(s['median_persistence']), f(s['median_dev_ratio']), f(s['median_spread_ratio'])]
                for k, s in sorted(c12.items())]
        P.append(table(['계획기|모형군', '칸', '지속 지수 중앙값', '첫 행동 편차 / seed 바닥', '재실행 퍼짐 / seed 바닥'], rows))
        P.append('')
    # ---- C13 post-hoc
    c13 = C.get('C13_posthoc') or {}
    if c13:
        P.append('### C13 (사후) 손상의 예측 변수: −OEI, −drift, −sel_last (블록 M, group 안 Spearman)\n')
        rows = []
        for pl, dd in sorted(c13.items()):
            for k, v in sorted(dd.items()):
                rows.append([pl, k.replace('spearman_dmg_neg_', '−'), v['groups'], f(v['median'], 2), f(v['q25'], 2), f(v['q75'], 2)])
        P.append(table(['계획기', '예측 변수', 'group', 'Spearman 중앙값', 'q25', 'q75'], rows))
        P.append('')
    # ---- default decomposition (cells csv)
    for pl, sel_key, rd_key, oei_key, dmg_key, filt in (
            ('mppi', 'sel_last', 'sel_last_rd', 'oei', 'dmg_exec',
             lambda r: r['block'] == 'M' and r['N'] == '512' and r['rule'] == 'q125' and r['n'] in ('6', '12')),
            ('cem', 'sel_last_elite', 'sel_last_mf', 'oei_mf', 'dmg',
             lambda r: r['block'] == 'D' and r['d'] == '6' and r['N'] == '300' and r['n'] == '30')):
        try:
            cells = load_cells(d, pl)
        except FileNotFoundError:
            continue
        rows = []
        for r in cells:
            if not filt(r) or r['model'] not in ('M1', 'R-det', 'R-fixed', 'M2', 'M3', 'R-shiter', 'R-detshift'):
                continue
            if r['scale'] not in ('8', '32', '8.0', '32.0'):
                continue
            sJ0 = num(r, 'sJ0')
            rows.append((r['regime'], int(float(r['scale'])), int(r['n']), r['model'],
                         [num(r, oei_key) / sJ0, num(r, sel_key) / sJ0, num(r, rd_key) / sJ0, num(r, 'drift') / sJ0,
                          num(r, dmg_key) / sJ0, num(r, 'persistence')]))
        rows.sort()
        P.append(f'### 기본 설정 분해 표 ({pl}, σ_J0 단위)\n')
        P.append(table(['regime', 'r', 'n', '모형', 'OEI', 'sel_last 같은 평가', 'sel_last 재평가' if pl == 'mppi' else 'sel_last_mf(재평가)',
                        'drift', '손상', '지속 지수'],
                       [[a, b, c, m] + [f(x) for x in v] for a, b, c, m, v in rows]))
        P.append('')
    meta = C.get('meta', {})
    P.append(f"- 메타: job {meta.get('n_jobs')}, 행 {meta.get('n_rows')}, 계획기 실행 {meta.get('n_planner_runs')}, "
             f"job CPU 합 {f(meta.get('cpu_s'))} s, 벽시계 합 {f(meta.get('wall_s_sum'))} s")
    print('\n'.join(P))


if __name__ == '__main__':
    main()
