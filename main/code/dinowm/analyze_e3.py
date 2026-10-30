"""E3 analysis, DINO-WM part (DESIGN_main v5 section 7.2, v5.1 section 8.3). Written before any E3 data existed.
Pure python + numpy/scipy on the synced job JSONs (and, for the H4b input, the none populations of E1).

usage: python analyze_e3.py <results dir holding E1_dinowm_wp/, E1_dinowm_pusht/, E3_dinowm/> [--e1 E1 summary JSON]
                            [--pops DIR with <man>/<task>/*none*.npz] [--out JSON] [--md MD] [--family JSON]

Two-axis contrast (open loop: plan-fixed = episode-fixed): sh-iter, sh-plan (shstep), ind from E1, indplan from E3, at
PushT k=4, Wall k=5, PointMaze k=3 (primary k) plus PushT k=3 and Wall k=3 (exploratory). Expectation stated before the
run: indplan ~ ind. Reported: D_v, D_indplan - D_ind (paired CI), OEI, drift.
Table-size sweep (H4a): shared fixed tables L0 shstep (E1; per layer and horizon step one u per token x unit),
L1 shunit (one per unit), L2 shgroup (one per 16 units), L3 shlayer (one per layer); N_eff L0 > L1 > L2 > L3.
  task passes if Spearman(N_eff, D_level) <= -0.8 over the 4 levels (one adjacent swap allowed: -0.8 exactly);
  eligible tasks: E1 primary-k cell with a real det drop (v4 6.1, from the E1 summary JSON); DINO-WM passes if 2 of
  the 3 tasks pass (both families must pass, combined by the coordinator). L+ = indplan is reported separately.
  Auxiliary: per-task Spearman median; var_cand per level = mean over episodes of the across-candidate variance of
  e = J_v - J_none on the iteration-0 random candidates (= it0_rand_dJ_sd^2 of the record), with
  Spearman(var_cand, D) per task.
H4b inputs (computed once, after E3 is complete): near-tie density = -median over episodes of gap12 / std of the
  none variant's last-iteration population (unquantized cost); CRN gain = D_ind - D_sh-iter at the primary k (E1).
  Auxiliary (v5 7.2 'first set', added 2026-10-23): the same density on the iteration-0 population (near_tie_first).
  The DINO-WM inputs use E1 data only; h4b_dinowm.py computes them alone with these functions (coordinator request
  2026-10-23; the 15-task correlation stays with the coordinator, after E3).

Changes 2026-10-23T22:30Z (E3 analysis worker, after E3 finished and BEFORE any E3 number was computed; the H4a rule
computation is unchanged):
  - threshold tolerance: scipy returns -0.7999999999999999 for an exact -0.8 (one adjacent swap of 4 points). The
    pre-written test already had `<= -0.8 + 1e-12`; it now uses EPS_RHO = 1e-9 like the TD-MPC2 family
    (v5_common.le). Both give the exact rational decision on all 256 4-level patterns (checked by enumeration).
  - joint bootstrap of Spearman(N_eff, D) per sweep task (coordinator request): the pairs are resampled with the same
    index for the 4 levels and none (10,000, RandomState(0)); 95% interval, median, share <= -0.8 (same tolerance)
  - H4a block: per-task eligibility detail (E1 det drop, CI, Holm), counts, the strict-majority count; auxiliary
    median of the per-task Spearman over the eligible tasks (the TD-MPC2 convention) next to the pre-written median
    over the 3 primary tasks; Spearman(N_eff, var_cand) (descriptive)
  - two-axis descriptive contrasts besides D_indplan - D_ind: paired OEI and drift differences indplan - ind, and the
    2 x 2 contrasts D_shplan - D_shiter, D_ind - D_shiter, D_indplan - D_shplan (paired CIs)
  - completeness flags (100 episodes per level) and --family (the per-family summary for the coordinator)
"""
import os, sys, json, glob, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A

PRIMARY = {'pusht': 4, 'wall': 5, 'pointmaze': 3}
TWO_AXIS = [('pusht', 4, 'primary'), ('wall', 5, 'primary'), ('pointmaze', 3, 'primary'), ('pusht', 3, 'extra'),
            ('wall', 3, 'exploratory')]
SWEEP = [('pusht', 4, 'primary'), ('wall', 5, 'primary'), ('pointmaze', 3, 'primary'), ('wall', 3, 'exploratory')]
LEVELS = [('L0', 'shstep', 4), ('L1', 'shunit', 3), ('L2', 'shgroup', 2), ('L3', 'shlayer', 1)]   # rank of N_eff
TN = dict(pusht='PushT', wall='Wall', pointmaze='PointMaze')
EPS_RHO = 1e-9
NB = 10000
N_EP = 100


def le(x, thr):
    return bool(x is not None and np.isfinite(x) and x <= thr + EPS_RHO)


def spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(x, y).correlation
    return float(r) if np.isfinite(r) else float('nan')


def spearman_rows(X, y):
    """Spearman of every row of X (B, L) with the fixed vector y (L,), average ranks (scipy convention); constant
    row -> NaN."""
    from scipy.stats import rankdata
    rx = rankdata(np.asarray(X, float), axis=1).astype(float)
    ry = rankdata(np.asarray(y, float)).astype(float)
    rx -= rx.mean(1, keepdims=True); ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum())
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(den > 0, (rx * ry).sum(1) / np.where(den > 0, den, 1.0), np.nan)


def stats(R, t, v, k):
    c = A.cell_stats(R, t, v, k)
    if c is None:
        return None
    eps = sorted(e for (tt, vv, kk, e) in R if tt == t and vv == v and kk == k and (t, 'none', 0, e) in R)
    vc = np.array([R[(t, v, k, e)]['oei']['it0_rand_dJ_sd'] ** 2 for e in eps])
    dr = np.array([R[(t, v, k, e)]['oei']['last_pop_dJ'] - R[(t, v, k, e)]['oei']['it0_rand_dJ'] for e in eps])
    c.update(var_cand=float(vc.mean()), var_cand_ci=A.boot(lambda i: vc[i].mean(), len(vc)),
             drift=float(dr.mean()), drift_ci=A.boot(lambda i: dr[i].mean(), len(dr)), complete=bool(len(eps) == N_EP))
    return c


def paired_diff(R, t, k, v1, v2):
    eps = sorted(e for (tt, vv, kk, e) in R if tt == t and vv == v1 and kk == k and (t, v2, k, e) in R)
    d = np.array([float(R[(t, v2, k, e)]['success']) - float(R[(t, v1, k, e)]['success']) for e in eps])   # D_v1 - D_v2
    return dict(n=len(eps), diff=float(d.mean()), ci=A.boot(lambda i: d[i].mean(), len(d)))


def paired_field_diff(R, t, k, v1, v2, fn):
    """mean over the common episodes of fn(record v1) - fn(record v2), paired bootstrap CI (descriptive)."""
    eps = sorted(e for (tt, vv, kk, e) in R if tt == t and vv == v1 and kk == k and (t, v2, k, e) in R)
    d = np.array([fn(R[(t, v1, k, e)]) - fn(R[(t, v2, k, e)]) for e in eps])
    return dict(n=len(eps), diff=float(d.mean()), ci=A.boot(lambda i: d[i].mean(), len(d)))


def joint_boot_rho(R, t, k):
    """Bootstrap distribution of Spearman(N_eff, D) over the 4 shared levels: one pair index per resample for the
    4 levels and none (paired, 10,000, RandomState(0))."""
    vs = [v for _, v, _ in LEVELS]
    eps = sorted(set.intersection(*[{e for (tt, vv, kk, e) in R if tt == t and vv == v and kk == k} for v in vs])
                 & {e for (tt, vv, kk, e) in R if tt == t and vv == 'none'})
    if not eps:
        return None
    Sn = np.array([float(R[(t, 'none', 0, e)]['success']) for e in eps])
    S = np.stack([np.array([float(R[(t, v, k, e)]['success']) for e in eps]) for v in vs], 1)      # (n, 4)
    idx = np.random.RandomState(0).randint(0, len(eps), size=(NB, len(eps)))
    Db = (Sn[idx][:, :, None] - S[idx]).mean(1)                                                       # (NB, 4)
    rb = spearman_rows(Db, [rk for _, _, rk in LEVELS])
    ok = rb[np.isfinite(rb)]
    return dict(n_pairs=len(eps), ci=[float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))],
                median=float(np.median(ok)), n_defined=int(len(ok)), share_le_m0_8=float(np.mean(ok <= -0.8 + EPS_RHO)),
                D_ci_check=[[float(np.percentile(Db[:, j], 2.5)), float(np.percentile(Db[:, j], 97.5))] for j in range(len(vs))])


def near_tie(pops_dir, t, key='last_J_none', std_rows='all'):
    """H4b near-tie input from the E1 none variant of task t (v5 7.2): per episode g = (top1 - top2 gap of the
    unquantized costs) / std of the candidate set's unquantized costs (all N rows, row 0 = the current mean included;
    np.std), density = -median over the episodes.
    key: 'last_J_none' = the last CEM iteration's population (the v5 7.2 definition), 'it0_J_none' = the iteration-0
    population (v5 7.2 auxiliary 'first set'). std_rows 'rand' (post-hoc sensitivity, 2026-10-23): std over rows 1..
    only (the TD-MPC2 convention, std over the sampled candidates); the gap still uses every row.
    CI of the density: bootstrap over the episodes (10,000, RandomState(0)), episodes in index order.
    (2026-10-23: key / std_rows / CI / quartiles added; the default value is the pre-registered computation.)"""
    import re
    man = 'E1_dinowm_pusht' if t == 'pusht' else 'E1_dinowm_wp'
    split = 'eval6' if t == 'pusht' else 'eval'
    fs = [f for f in sorted(glob.glob(f'{pops_dir}/{man}/{t}/{split}_none_e???-???_e*.npz')) if not f.endswith('_truth.npz')]
    ep = [int(re.search(r'_e(\d+)\.npz$', f).group(1)) for f in fs]
    if len(set(ep)) != len(ep):
        raise SystemExit(f'{t}: more than one none population per episode')
    g = []
    for e, f in sorted(zip(ep, fs)):
        J0 = np.load(f)[key].astype(np.float64)
        J = np.sort(J0)
        sd = J0[1:].std() if std_rows == 'rand' else J.std()
        g.append((J[1] - J[0]) / (sd + 1e-12))
    if not g:
        return None
    g = np.array(g)
    return dict(n=len(g), episodes=[min(ep), max(ep)], density=float(-np.median(g)),
                density_ci=A.boot(lambda i: -np.median(g[i]), len(g)), gap_over_std_median=float(np.median(g)),
                gap_over_std_q25=float(np.percentile(g, 25)), gap_over_std_q75=float(np.percentile(g, 75)),
                key=key, std_rows=std_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('res'); ap.add_argument('--e1', default=None); ap.add_argument('--pops', default=None)
    ap.add_argument('--out', default=None); ap.add_argument('--md', default=None); ap.add_argument('--family', default=None)
    a = ap.parse_args()
    R = {}
    for sub in ('E1_dinowm_wp', 'E1_dinowm_pusht', 'E3_dinowm'):
        if os.path.isdir(f'{a.res}/{sub}'):
            R.update(A.load(f'{a.res}/{sub}')[0])
    real, e1det = {}, {}
    if a.e1:
        S = json.load(open(a.e1))
        for t, k in PRIMARY.items():
            c = S['tasks'].get(t, {}).get('confirmatory', {}).get(str(k), {}).get('det', {})
            real[t] = bool(c.get('drop_real_holm'))
            e1det[t] = dict(k=k, drop_real_holm=real[t], D_det=c.get('D'), D_det_ci=c.get('D_ci'),
                            mcnemar_p=c.get('mcnemar_p'))
    out = dict(two_axis={}, sweep={}, H4a={}, H4b_inputs={})
    for t, k, role in TWO_AXIS:
        C = {v: stats(R, t, v, k) for v in ('det', 'ind', 'indplan', 'shiter', 'shstep')}
        C = {v: c for v, c in C.items() if c}
        con = {}
        if 'indplan' in C and 'ind' in C:
            C['indplan_minus_ind'] = paired_diff(R, t, k, 'indplan', 'ind')
            con['D_indplan_minus_D_ind'] = C['indplan_minus_ind']
            con['OEI_indplan_minus_OEI_ind'] = paired_field_diff(R, t, k, 'indplan', 'ind', lambda r: r['oei']['oei_ref_mf'])
            con['drift_indplan_minus_drift_ind'] = paired_field_diff(
                R, t, k, 'indplan', 'ind', lambda r: r['oei']['last_pop_dJ'] - r['oei']['it0_rand_dJ'])
        for name, v1, v2 in (('D_shplan_minus_D_shiter', 'shstep', 'shiter'), ('D_ind_minus_D_shiter', 'ind', 'shiter'),
                             ('D_indplan_minus_D_shplan', 'indplan', 'shstep')):
            if v1 in C and v2 in C:
                con[name] = paired_diff(R, t, k, v1, v2)
        out['two_axis'][f'{t}_k{k}'] = dict(task=t, k=k, role=role, cells=C, contrasts=con)
    for t, k, role in SWEEP:
        L = {}
        for lab, v, rank in LEVELS:
            c = stats(R, t, v, k)
            if c:
                L[lab] = dict(c, rank=rank)
        ip = stats(R, t, 'indplan', k)
        res = dict(task=t, k=k, role=role, levels=L, Lplus_indplan=ip)
        if len(L) == 4:
            ranks = [L[l]['rank'] for l, _, _ in LEVELS]; D = [L[l]['D'] for l, _, _ in LEVELS]
            vc = [L[l]['var_cand'] for l, _, _ in LEVELS]
            res.update(rho_Neff_D=spearman(ranks, D), rho_varcand_D=spearman(vc, D), rho_Neff_varcand=spearman(ranks, vc))
            res['pass'] = le(res['rho_Neff_D'], -0.8)
            res['complete'] = bool(all(L[l]['complete'] for l, _, _ in LEVELS))
            res['rho_boot'] = joint_boot_rho(R, t, k)
        out['sweep'][f'{t}_k{k}'] = res
    elig = [t for t in PRIMARY if real.get(t) and 'pass' in out['sweep'].get(f'{t}_k{PRIMARY[t]}', {})]
    passed = [t for t in elig if out['sweep'][f'{t}_k{PRIMARY[t]}']['pass']]
    rhos = [out['sweep'][f'{t}_k{PRIMARY[t]}']['rho_Neff_D'] for t in PRIMARY if 'rho_Neff_D' in out['sweep'].get(f'{t}_k{PRIMARY[t]}', {})]
    rhos_el = [out['sweep'][f'{t}_k{PRIMARY[t]}']['rho_Neff_D'] for t in elig]
    out['H4a'] = dict(eligible=elig, passed=passed, needed=2, holds_dinowm=bool(len(elig) >= 2 and len(passed) >= 2) if a.e1 else None,
                      note='needs 2 of the 3 tasks; only E1-eligible tasks count (if fewer than 2 are eligible, the coordinator rules)',
                      aux_median_rho=float(np.median(rhos)) if rhos else None,
                      aux_median_rho_eligible=float(np.median(rhos_el)) if rhos_el else None,
                      n_eligible=len(elig), n_passed=len(passed), strict_majority_needed=len(elig) // 2 + 1,
                      complete=bool(all(out['sweep'].get(f'{t}_k{PRIMARY[t]}', {}).get('complete') for t in PRIMARY)),
                      eligibility=e1det, eps_rho=EPS_RHO)
    for t, k in PRIMARY.items():
        g = paired_diff(R, t, k, 'ind', 'shiter') if (t, 'ind', k, 0) in R else None     # D_ind - D_shiter
        nt = near_tie(a.pops, t) if a.pops else None
        nt1 = near_tie(a.pops, t, key='it0_J_none') if a.pops else None
        out['H4b_inputs'][t] = dict(k=k, crn_gain=g, near_tie=nt, near_tie_first=nt1)
    print(json.dumps(dict(H4a={k: v for k, v in out['H4a'].items() if k != 'eligibility'},
                          sweep={key: {kk: x.get(kk) for kk in ('rho_Neff_D', 'pass', 'complete')} for key, x in out['sweep'].items()}),
                     indent=1, default=float, ensure_ascii=False))
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1, default=float, ensure_ascii=False)
    if a.md:
        open(a.md, 'w').write(tables(out))
    if a.family:
        json.dump(family(out), open(a.family, 'w'), indent=1, default=float, ensure_ascii=False)


def family(out):
    """Per-family summary of the DINO-WM E3 share of H4a for the coordinator (plus the two-axis contrasts)."""
    h = out['H4a']
    tasks = {}
    for key, x in out['sweep'].items():
        t, k = x['task'], x['k']
        el = h['eligibility'].get(t, {}) if x['role'] == 'primary' else {}
        lv = []
        for lab, v, rank in LEVELS:
            c = x['levels'].get(lab)
            if c:
                lv.append(dict(level=lab, variant=v, source='E1' if v == 'shstep' else 'E3', N_eff_rank=rank, n=c['n'],
                               D=c['D'], D_ci=c['D_ci'], var_cand=c['var_cand'], var_cand_ci=c['var_cand_ci'],
                               OEI=c.get('oei_ref_mf'), drift=c['drift']))
        ip = x.get('Lplus_indplan')
        tasks[key] = dict(task=t, k=k, role=x['role'], eligible=(t in h['eligible']) if x['role'] == 'primary' else None,
                          E1_det=el or None, levels=lv, rho_Neff_D=x.get('rho_Neff_D'), passes=x.get('pass'),
                          complete=x.get('complete'), rho_boot=x.get('rho_boot'), aux_rho_varcand_D=x.get('rho_varcand_D'),
                          aux_rho_Neff_varcand=x.get('rho_Neff_varcand'),
                          Lplus_indplan=dict(D=ip['D'], D_ci=ip['D_ci'], var_cand=ip['var_cand']) if ip else None)
    two = {key: dict(task=x['task'], k=x['k'], role=x['role'],
                     D={v: dict(D=c['D'], D_ci=c['D_ci'], OEI=c.get('oei_ref_mf'), drift=c['drift'])
                        for v, c in x['cells'].items() if v != 'indplan_minus_ind'},
                     contrasts=x['contrasts']) for key, x in out['two_axis'].items()}
    return dict(
        family='DINO-WM',
        rule='DESIGN_main v5 7.2 H4a: a task passes iff Spearman(N_eff, D_sh-fixed) <= -0.8 over the 4 shared fixed-table '
             'levels L0 shstep (E1) > L1 shunit > L2 shgroup > L3 shlayer (one adjacent swap = exactly -0.8 allowed; '
             f'tolerance {EPS_RHO}); only sweep tasks whose E1 primary-k det drop is real (v4 6.1) count; the family '
             'holds with a majority of them (DINO-WM: 2); H4a needs both families. D = success-rate drop vs the E1 none '
             '(fraction, x100 = %p). Auxiliary: var_cand = mean it0_rand_dJ_sd^2, Spearman(var_cand, D) (v5.1 8.3)',
        H4a_dinowm=dict(holds=h['holds_dinowm'], eligible=h['eligible'], passed=h['passed'], n_eligible=h['n_eligible'],
                        n_passed=h['n_passed'], needed=h['needed'], complete=h['complete'],
                        aux_median_rho_eligible=h['aux_median_rho_eligible'], aux_median_rho_3_primary=h['aux_median_rho']),
        tasks=tasks, two_axis=two)


def tables(out):
    def pp(x, c):
        return f'{100 * x:+.0f} [{100 * c[0]:.0f}, {100 * c[1]:.0f}]'

    def f3(x, c):
        return f'{x:+.3f} [{c[0]:+.3f}, {c[1]:+.3f}]'
    L = ['### 2축 대조', '', '| 과제 | k | 역할 | 변형 | 성공 | D %p [95% CI] | OEI | drift | var_cand |', '|' + '---|' * 9]
    for key, x in out['two_axis'].items():
        for v, c in x['cells'].items():
            if v == 'indplan_minus_ind':
                L.append(f"| {TN[x['task']]} | {x['k']} | {x['role']} | D_indplan − D_ind | | {pp(c['diff'], c['ci'])} | | | |")
                continue
            L.append(f"| {TN[x['task']]} | {x['k']} | {x['role']} | {v} | {c['s']}/{c['n']} | {pp(c['D'], c['D_ci'])} | "
                     f"{c.get('oei_ref_mf', float('nan')):+.3f} | {c['drift']:+.3f} | {c['var_cand']:.2e} |")
    L += ['', '### 2축 짝 대비 (기술)', '', '| 과제 | k | 역할 | D_indplan − D_ind %p | OEI indplan − ind | drift indplan − ind | '
          'D_sh-plan − D_sh-iter %p | D_ind − D_sh-iter %p | D_indplan − D_sh-plan %p |', '|' + '---|' * 9]
    for key, x in out['two_axis'].items():
        c = x['contrasts']
        def g(name, pct=True):
            if name not in c:
                return '—'
            return pp(c[name]['diff'], c[name]['ci']) if pct else f3(c[name]['diff'], c[name]['ci'])
        L.append(f"| {TN[x['task']]} | {x['k']} | {x['role']} | {g('D_indplan_minus_D_ind')} | {g('OEI_indplan_minus_OEI_ind', False)} | "
                 f"{g('drift_indplan_minus_drift_ind', False)} | {g('D_shplan_minus_D_shiter')} | {g('D_ind_minus_D_shiter')} | "
                 f"{g('D_indplan_minus_D_shplan')} |")
    L += ['', '### 표 크기 sweep (H4a)', '', '| 과제 | k | 역할 | L0 D | L1 D | L2 D | L3 D | ρ(N_eff, D) | 성립 | ρ bootstrap 95% | '
          'P(ρ* ≤ −0.8) | ρ(var_cand, D) | L+ indplan D |', '|' + '---|' * 13]
    for key, x in out['sweep'].items():
        Lv = x['levels']
        cells = [pp(Lv[l]['D'], Lv[l]['D_ci']) if l in Lv else '—' for l in ('L0', 'L1', 'L2', 'L3')]
        ip = x.get('Lplus_indplan')
        b = x.get('rho_boot') or {}
        L.append(f"| {TN[x['task']]} | {x['k']} | {x['role']} | " + ' | '.join(cells) +
                 f" | {x.get('rho_Neff_D', float('nan')):+.2f} | {x.get('pass', '—')} | "
                 + (f"[{b['ci'][0]:+.2f}, {b['ci'][1]:+.2f}] | {b['share_le_m0_8']:.2f}" if b else '— | —')
                 + f" | {x.get('rho_varcand_D', float('nan')):+.2f} | " + (pp(ip['D'], ip['D_ci']) if ip else '—') + ' |')
    L += ['', '### var_cand (수준별, 보조)', '', '| 과제 | k | L0 | L1 | L2 | L3 | L+ indplan | ρ(N_eff, var_cand) |', '|' + '---|' * 8]
    for key, x in out['sweep'].items():
        Lv = x['levels']
        cells = [f"{Lv[l]['var_cand']:.2e}" if l in Lv else '—' for l in ('L0', 'L1', 'L2', 'L3')]
        ip = x.get('Lplus_indplan')
        ips = f"{ip['var_cand']:.2e}" if ip else '—'
        L.append(f"| {TN[x['task']]} | {x['k']} | " + ' | '.join(cells) + f" | {ips} | {x.get('rho_Neff_varcand', float('nan')):+.2f} |")
    h = out['H4a']
    L += ['', '### H4a (DINO-WM 몫)', '',
          f"- 자격 과제(E1 1차 k det 하락 실재): {h['eligible']} ({h['n_eligible']}개), 성립: {h['passed']} ({h['n_passed']}개), "
          f"필요 {h['needed']} → DINO-WM {'성립' if h['holds_dinowm'] else '불성립' if h['holds_dinowm'] is not None else '—'}"
          + ('' if h['complete'] else ' (자료 불완전: 잠정)'),
          f"- 보조: 과제별 Spearman 중앙값(자격 과제) {h['aux_median_rho_eligible']}, (1차 k 3과제) {h['aux_median_rho']}"]
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()
