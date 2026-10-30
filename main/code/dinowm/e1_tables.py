"""Markdown tables for reports/E1_dinowm.md from analyze_e1.py's output JSON, plus the job metas (GPU per cell).
Pure python + numpy. Prints to stdout.

usage: python e1_tables.py <analyze_e1 out.json> <results dir holding E1_dinowm_wp/ and E1_dinowm_pusht/> [--errdir DIR]
Variant names follow the design: shstep = sh-plan (stands for sh-step and sh-ep, DESIGN_main v2 section 4.2).
red_v is shown only in qualifying cells (DESIGN_main v4 section 6.1). No OEI decomposition (held until v5)."""
import os, sys, json, glob, argparse, collections

CONF = {'pusht': [3, 4, 5], 'wall': [4, 5, 6], 'pointmaze': [2, 3, 4]}
EXPL = {'wall': [3]}
VARS = ['det', 'ind', 'shiter', 'shstep', 'detshift']
VN = dict(det='det', ind='ind', shiter='sh-iter', shstep='sh-plan', detshift='det+shift', none='none', nonealt='nonealt')
TN = dict(pusht='PushT', wall='Wall', pointmaze='PointMaze')
ORDER = ['pusht', 'wall', 'pointmaze']


def pp(x, nd=0):
    return '—' if x is None else f'{100 * x:+.{nd}f}'


def ci(c, scale=100.0, nd=0):
    if c is None or any(v is None for v in c):
        return '—'
    return f'[{scale * c[0]:.{nd}f}, {scale * c[1]:.{nd}f}]'


def f3(x, nd=3):
    return '—' if x is None else f'{x:+.{nd}f}'


def g4(x):
    return '—' if x is None else f'{x:.4f}'


def pv(p):
    return '—' if p is None else (f'{p:.4f}' if p >= 1e-4 else f'{p:.1e}')


def table(head, rows):
    out = ['| ' + ' | '.join(head) + ' |', '|' + '|'.join('---' for _ in head) + '|']
    out += ['| ' + ' | '.join(str(x) for x in r) + ' |' for r in rows]
    return '\n'.join(out)


def gpus(res):
    """(task, k) -> (Counter of CUDA_VISIBLE_DEVICES under PCI_BUS_ID order, GPU seconds, first start, last end)."""
    G = {}
    for sub in ('E1_dinowm_wp', 'E1_dinowm_pusht'):
        for f in glob.glob(f'{res}/{sub}/*/*.json'):
            d = json.load(open(f))
            j, m = d['job'], d['meta']
            key = (j['task'], int(j['k']))
            c, s, t0, t1 = G.get(key, (collections.Counter(), 0.0, None, None))
            c[f"GPU {m.get('cuda_visible')}"] += 1
            s += float(m.get('gpu_s') or 0)
            ts, te = m.get('t_start'), m.get('t_finalized')
            t0 = ts if t0 is None or (ts and ts < t0) else t0
            t1 = te if t1 is None or (te and te > t1) else t1
            G[key] = (c, s, t0, t1)
    return G


SPLIT = dict(pusht=('eval6', 4000, 4500, 1), wall=('eval', 3000, 3500, 2), pointmaze=('eval', 3000, 3500, 1))


def check(res):
    """Data check before any judgment: per (task, var, k) cell, episodes 0-99 present once, the split, planner seed
    (base + ep; nonealt base + 500 + ep), batch size E, and the diagnostics (err for quantized variants, oei for all)."""
    seen = collections.defaultdict(list)
    for sub in ('E1_dinowm_wp', 'E1_dinowm_pusht'):
        for f in glob.glob(f'{res}/{sub}/*/*.json'):
            for r in json.load(open(f))['records']:
                seen[(r['task'], r['var'], int(r['k']))].append(r)
    rows, bad = [], []
    for t in ORDER:
        keys = [key for key in seen if key[0] == t]
        if not keys:
            continue
        split, base, alt, E = SPLIT[t]
        n_full = 0
        for key in keys:
            R = seen[key]
            eps = sorted(int(r['ep']) for r in R)
            ok_eps = eps == list(range(100))
            n_full += ok_eps
            sb = alt if key[1] == 'nonealt' else base
            errs = [f'{key}: ' + m for m, c in (
                ('episodes not 0-99 once', not ok_eps),
                ('split', any(r['split'] != split for r in R)),
                ('planner seed', any(int(r['plan_seed']) != sb + int(r['ep']) for r in R)),
                ('E', any(int(r['E']) != E for r in R)),
                ('err missing', key[1] not in ('none', 'nonealt') and any('err' not in r for r in R)),
                ('oei missing', any('oei' not in r for r in R))) if c]
            bad += errs
        rows.append([TN[t], len(keys), n_full, split, f'{base}+ep / {alt}+ep', E])
    print(table(['과제', '칸(변형 × k) 수', '100 episode 다 있는 칸', '분할', 'planner seed (nonealt)', 'E'], rows))
    print(f"\n불일치: {'없음' if not bad else '; '.join(bad[:20]) + (' ...' if len(bad) > 20 else '')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('json'); ap.add_argument('res'); ap.add_argument('--errdir', default=None)
    a = ap.parse_args()
    out = json.load(open(a.json))
    F = out['families']['dinowm']
    tasks = [t for t in ORDER if t in out['tasks']]
    G = gpus(a.res)
    qual = {(t, int(k)) for t, k in F['qualifying_cells']}

    print('### 자료 점검\n')
    check(a.res)

    print('\n### 표 0. 실행 요약\n')
    rows = []
    for t in tasks:
        T = out['tasks'][t]; nn = T['none']; fl = T['floor']
        for k in [0] + CONF[t] + EXPL.get(t, []):
            if (t, k) not in G:
                continue
            c, s, t0, t1 = G[(t, k)]
            role = 'none·nonealt' if k == 0 else ('탐색' if k in EXPL.get(t, []) else '확증')
            rows.append([TN[t], '—' if k == 0 else k, role, ', '.join(f'{g} ({n} job)' for g, n in sorted(c.items())),
                         f'{s / 3600:.1f}', f'{t0} ~ {t1}'])
    print(table(['과제', 'k', '역할', 'GPU (PCI_BUS_ID 순서)', 'GPU-h', '시작 ~ 끝 (UTC)'], rows))
    print()
    rows = []
    for t in tasks:
        T = out['tasks'][t]; nn = T['none']; fl = T['floor'] or {}
        rows.append([TN[t], f"{nn['s']}/{nn['n']} ({100 * nn['success']:.0f}%)",
                     f"{100 * fl['success_nonealt']:.0f}%" if fl else '—',
                     f"{fl['mcnemar'][2]:.3g}" if fl else '—', f"{fl['floor_first']:.3f}" if fl else '—'])
    print(table(['과제', 'none 성공', 'nonealt 성공', 'none vs nonealt McNemar p', '계획 편차 seed 바닥(첫 행동 RMS)'], rows))

    print('\n### 표 1. det 하락과 자격 칸 (확증 칸)\n')
    rows = []
    for t in tasks:
        T = out['tasks'][t]
        ks = [k for k in CONF[t] if str(k) in T['confirmatory']]
        ps = sorted((T['confirmatory'][str(k)]['det']['p_boot_drop'], k) for k in ks)
        thr = {k: 0.025 / (len(ps) - r) for r, (p, k) in enumerate(ps)}
        for k in ks:
            d = T['confirmatory'][str(k)]['det']
            rows.append([TN[t], k, d['n'], f"{100 * (d['success'] + d['D']):.0f}%", f"{100 * d['success']:.0f}%",
                         f"{pp(d['D'])} {ci(d['D_ci'])}", pv(d['mcnemar_p']), pv(d['p_boot_drop']), f'{thr[k]:.4f}',
                         '예' if d.get('drop_real_holm') else '아니오', '예' if d.get('drop_real_ci95') else '아니오'])
    print(table(['과제', 'k', 'n', 'none', 'det', 'D_det %p [95% CI]', 'McNemar p', 'bootstrap 한쪽 p', 'Holm 기준',
                 '실재(Holm)', '보조: CI 하한 > 0'], rows))
    print(f"\n자격 칸: {', '.join(f'{TN[t]} k={k}' for t, k in sorted(qual)) or '없음'}")

    print('\n### 표 2. H1: 자격 칸의 계획 시점 1-step 오차 (det vs sh-iter)\n')
    rows = []
    ds = {}
    for t in tasks:
        if a.errdir and os.path.exists(f'{a.errdir}/err_{t}_main.json'):
            ds[t] = json.load(open(f'{a.errdir}/err_{t}_main.json'))['ds']
    for t in tasks:
        T = out['tasks'][t]
        for k in CONF[t]:
            C = T['confirmatory'].get(str(k))
            if C is None:
                continue
            d, s = C['det'], C.get('shiter', {})
            dsd, dss = ds.get(t, {}).get(f'det{k}', {}), ds.get(t, {}).get(f'shiter{k}', {})
            rows.append([TN[t], k, '예' if (t, k) in qual else '아니오', g4(d.get('rel_err1')), g4(s.get('rel_err1')),
                         f"{d.get('err1_det_minus_shiter', float('nan')):+.4f} {ci(d.get('err1_det_minus_shiter_ci'), 1, 4)}",
                         '예' if d.get('rel_err1') is not None and s.get('rel_err1') is not None and d['rel_err1'] <= s['rel_err1'] else '아니오',
                         g4(dsd.get('rel_err1')), g4(dss.get('rel_err1'))])
    print(table(['과제', 'k', '자격', 'det 1-step', 'sh-iter 1-step', 'det − sh-iter [95% CI]', 'det ≤ sh-iter',
                 '보조: 데이터셋형 det', '보조: 데이터셋형 sh-iter'], rows))
    print(f"\nH1 성립: {F['H1']['holds']} (만족 칸: {F['H1']['cells_meeting'] or '없음'})")

    print('\n### 표 3. H3a: red_sh-iter (자격 칸)와 해 점검 (모든 확증 칸)\n')
    rows = []
    for t in tasks:
        T = out['tasks'][t]
        for k in CONF[t]:
            C = T['confirmatory'].get(str(k))
            if C is None or 'shiter' not in C:
                continue
            s = C['shiter']
            q = (t, k) in qual
            rows.append([TN[t], k, '예' if q else '아니오', f"{s['red']:.2f} {ci(s['red_ci'], 1, 2)}" if q and s.get('red') is not None else '—',
                         f"{pp(s['Dv_minus_Ddet'])} {ci(s['Dv_minus_Ddet_ci'])}",
                         '예' if s['Dv_minus_Ddet_ci'][0] > 0.05 else '아니오'])
    print(table(['과제', 'k', '자격', 'red_sh-iter [95% CI]', 'D_sh-iter − D_det %p [95% CI]', 'CI 하한 > +5%p'], rows))
    H = F['H3a']
    print(f"\nH3a 성립: {H['holds']} (자격 칸 {H['n_qualifying']}개, red ≥ 0.5 비율 {H['share_red_ge_half']}, "
          f"해 칸 {H['harm_cells'] or '없음'})")

    def cell_rows(role, ks_of):
        rows = []
        for t in tasks:
            T = out['tasks'][t]
            for k in ks_of(t):
                C = T[role].get(str(k))
                if C is None:
                    continue
                for v in VARS:
                    c = C.get(v)
                    if c is None:
                        continue
                    q = (t, k) in qual or role == 'exploratory'
                    red = c.get('red') if (v != 'det' and q and C['det']['D'] > 0) else None
                    rows.append([TN[t], k, VN[v], f"{c['s']}/{c['n']}", f"{pp(c['D'])} {ci(c['D_ci'])}",
                                 '—' if v == 'det' else f"{pp(c['Dv_minus_Ddet'])} {ci(c['Dv_minus_Ddet_ci'])}",
                                 '—' if red is None else f"{red:.2f} {ci(c['red_ci'], 1, 2)}",
                                 g4(c.get('rel_err1')), g4(c.get('rel_err3')), g4(c.get('rel_err5')),
                                 f"{c['plan_dev_first_norm']:.2f}" if 'plan_dev_first_norm' in c else '—'])
        return rows
    head = ['과제', 'k', '변형', '성공', 'D_v %p [95% CI]', 'D_v − D_det %p [95% CI]', 'red_v [95% CI]',
            '1-step', '3-step', '5-step', '계획 편차 / seed 바닥']
    print('\n### 표 4. 확증 칸의 모든 변형\n')
    print(table(head, cell_rows('confirmatory', lambda t: CONF[t])))
    if any(EXPL.get(t) for t in tasks):
        print('\n### 표 5. 탐색 칸 (Wall k=3, H1·H3a에 넣지 않음)\n')
        print(table(head, cell_rows('exploratory', lambda t: EXPL.get(t, []))))
        print('\n(탐색 칸의 red_v는 det 하락이 양수일 때 참고로 적는다.)')

    print('\n### 표 6. OEI (확증 지표 oei_ref_mf, 음수 = 이용)\n')
    rows = []
    for role, ks_of in (('confirmatory', lambda t: CONF[t]), ('exploratory', lambda t: EXPL.get(t, []))):
        for t in tasks:
            T = out['tasks'][t]
            for k in ks_of(t):
                C = T[role].get(str(k))
                if C is None:
                    continue
                for v in VARS:
                    c = C.get(v)
                    if c is None or 'oei_ref_mf' not in c:
                        continue
                    rows.append([TN[t], k, '확증' if role == 'confirmatory' else '탐색', VN[v],
                                 f"{f3(c['oei_ref_mf'])} {ci(c.get('oei_ref_mf_ci'), 1, 3)}",
                                 f"{f3(c.get('oei_ref_argmin'))} {ci(c.get('oei_ref_argmin_ci'), 1, 3)}",
                                 f3(c.get('oei_ref_elite')), f3(c.get('oei_true_mf')), f3(c.get('oei_true_argmin'))])
    print(table(['과제', 'k', '역할', '변형', 'oei_ref_mf [95% CI]', '보조: oei_ref_argmin [95% CI]', '보조: oei_ref_elite',
                 '보조: oei_true_mf', '보조: oei_true_argmin'], rows))

    print('\n### 표 7. H2a (PushT 1차 k = 4)\n')
    h = F.get('H2a')
    if h:
        print(table(['칸', 'det oei_ref_mf [95% CI]', 'det CI 상한 < 0', 'sh-iter oei_ref_mf [95% CI]', 'sh-iter CI가 0 포함', 'PushT 성립'],
                    [[h['cell'], f"{f3(h['det_oei'])} {ci(h['det_oei_ci'], 1, 3)}", h['det_upper_lt0'],
                      f"{f3(h['shiter_oei'])} {ci(h['shiter_oei_ci'], 1, 3)}", h['shiter_ci_contains0'], h['holds']]]))
    else:
        print('PushT k=4 칸이 아직 없다.')

    print('\n### 표 8. H2b (확증 칸, 점 = (과제, k, 변형))\n')
    h = F.get('H2b')
    if h:
        rows = [[TN[t], v['n'], f"{v['rho_negOEI_D']:+.3f}", f"{v['rho_err1_D']:+.3f}", f"{v['rho_err3_D']:+.3f}", f"{v['diff']:+.3f}"]
                for t, v in sorted(h['per_task'].items(), key=lambda kv: ORDER.index(kv[0]))]
        rows.append(['**과제 평균 (주 계산)**', '', '', '', '', f"**{h['primary_task_average']:+.3f}**"])
        aux = [('보조: 과제 섞음, 원값', h.get('aux_pooled_raw')), ('보조: 과제 섞음, OEI / none 첫 반복 비용 std', h.get('aux_pooled_oei_norm')),
               ('보조: 데이터셋형 오차, 과제 섞음', h.get('aux_dataset_errors_pooled'))]
        for name, v in aux:
            if v:
                rows.append([name, v['n'], f"{v['rho_negOEI_D']:+.3f}", f"{v['rho_err1_D']:+.3f}", f"{v['rho_err3_D']:+.3f}", f"{v['diff']:+.3f}"])
        for t, v in sorted((h.get('aux_dataset_errors_per_task') or {}).items(), key=lambda kv: ORDER.index(kv[0])):
            rows.append([f'보조: 데이터셋형 오차, {TN[t]}', v['n'], f"{v['rho_negOEI_D']:+.3f}", f"{v['rho_err1_D']:+.3f}",
                         f"{v['rho_err3_D']:+.3f}", f"{v['diff']:+.3f}"])
        if 'aux_dataset_errors_task_average' in h:
            rows.append(['보조: 데이터셋형 오차, 과제 평균', '', '', '', '', f"{h['aux_dataset_errors_task_average']:+.3f}"])
        print(table(['계산', '점 수', 'ρ(−OEI, D)', 'ρ(1-step, D)', 'ρ(3-step, D)', 'ρ(−OEI) − max(ρ1, ρ3)'], rows))
        print(f"\nH2b 성립(과제 평균 ≥ 0.15): {h['holds']}")


if __name__ == '__main__':
    main()
