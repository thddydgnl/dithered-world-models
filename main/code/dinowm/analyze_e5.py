"""E5 analysis, DINO-WM part (DESIGN_main v5 section 7.3). Written before any E5 data existed. Pure python + numpy.

usage: python analyze_e5.py <results dir holding E1_dinowm_wp/, E1_dinowm_pusht/, E5_dinowm_held/, E5_dinowm_eval/,
                             E5_dinowm_w/, E5_dinowm_qat/> [--qatlog JSON] [--out JSON] [--md MD] [--family JSON]

H3b (DINO-WM): over the E1-eligible 3-4-bit confirmatory cells (selection.json of e5_select.py), the family passes if
in a majority of the cells the paired 95% CI lower bound of D(B+det) - D(B+sh-iter) = S(B+sh-iter) - S(B+det) is > 0.
B = the held-selected activation-PTQ baseline; B = 'main' (the E1 grid) uses the E1 det and sh-iter cells.
D is against E1 none (same pairs and planner seeds). Paired bootstrap over pairs (10,000, RandomState(0)).
Weights (PushT k=4, activations fp) and QAT (PushT k=4, model qat_k4: none, det, sh-iter): D against E1 none,
descriptive; QAT flag: D(QAT+det) CI upper < 5pp -> dither's value is reported as a training-free PTQ prescription
(design 6.5).

Changes 2026-10-24T12:22Z (E5 analysis worker, after the E5 runs ended and BEFORE any E5 number was computed; the
H3b and QAT rule computations are unchanged):
  - threshold comparisons with the 1e-9 tolerance of the E2 / E3 analyses (coordinator): a value within 1e-9 of the
    threshold counts as equal to it, so it fails a strict inequality (CI lower > 0, CI upper < 0.05); the plain
    comparison is stored next to it (they can differ only by floating-point noise)
  - records are filtered by split (eval6 / eval) and the per-cell completeness (100 pairs) is reported
  - H3b cells: McNemar exact p of B+det vs B+sh-iter (auxiliary, design 6.8), red = (D(B+det) - D(B+sh-iter)) /
    D(B+det) with its bootstrap CI (descriptive), the held selection rows of selection.json and the tie at the maximum
  - QAT: the rule's flag plus descriptive paired contrasts: D(QAT+det) - D(B+det) (v5.3 10.1 pattern check against
    the selected baseline of PushT k=4), D(QAT+det) - D(E1 det) (the E1 grid the QAT was trained on),
    D(QAT+det) - D(QAT+sh-iter), D(QAT+none) against E1 none (did the training itself change the model), the damage
    relative to the QAT model's own none; McNemar for QAT+none vs E1 none; the training-log summary (--qatlog)
  - weights: paired contrasts w4rtn - w4sr1 / w4sr2 / w4sr4 / w4awq, w8rtn - w4rtn (the TD-MPC2 set) and the SR
    steps w4sr1 - w4sr2, w4sr2 - w4sr4 (direction check against TD-MPC2: damage falls as K grows)
  - --family (the per-family summary for the coordinator)
"""
import os, sys, json, glob, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A

SPLIT = dict(pusht='eval6', wall='eval', pointmaze='eval')
TN = dict(pusht='PushT', wall='Wall', pointmaze='PointMaze')
EPS = 1e-9
THR_QAT = 0.05
N_EP = 100
W_ORDER = ('w8rtn', 'w4rtn', 'w4awq', 'w4sr1', 'w4sr2', 'w4sr4')
W_CONTRASTS = (('w4rtn', 'w4sr1'), ('w4rtn', 'w4sr2'), ('w4rtn', 'w4sr4'), ('w4rtn', 'w4awq'), ('w8rtn', 'w4rtn'),
               ('w4sr1', 'w4sr2'), ('w4sr2', 'w4sr4'))


def gt(x, thr):
    return bool(x is not None and np.isfinite(x) and x > thr + EPS)


def lt(x, thr):
    return bool(x is not None and np.isfinite(x) and x < thr - EPS)


def load_jobs(d):
    out = []
    for f in glob.glob(f'{d}/*/*.json'):
        x = json.load(open(f))
        out.append((x['job'], x['records']))
    return out


def by_ep(jobs, pred):
    R = {}
    for j, recs in jobs:
        if pred(j) and j['split'] == SPLIT[j['task']]:
            for r in recs:
                R[int(r['ep'])] = r
    return R


def dstats(none, v):
    eps = sorted(set(none) & set(v))
    Sn = np.array([float(none[e]['success']) for e in eps]); Sv = np.array([float(v[e]['success']) for e in eps])
    d = Sn - Sv
    return dict(n=len(eps), success=float(Sv.mean()), D=float(d.mean()), D_ci=A.boot(lambda i: d[i].mean(), len(eps)),
                complete=bool(len(eps) == N_EP))


def contrast(none, a, b):
    """D_a - D_b = S_b - S_a on the pairs common to none, a and b; paired bootstrap CI and McNemar (a vs b)."""
    eps = sorted(set(none) & set(a) & set(b))
    Sa = np.array([float(a[e]['success']) for e in eps]); Sb = np.array([float(b[e]['success']) for e in eps])
    g = Sb - Sa
    n10, n01, p = A.mcnemar(Sb.astype(bool), Sa.astype(bool))
    return dict(n=len(eps), diff=float(g.mean()), ci=A.boot(lambda i: g[i].mean(), len(eps)),
                mcnemar=dict(b_only=n10, a_only=n01, p=p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('res'); ap.add_argument('--out', default=None); ap.add_argument('--md', default=None)
    ap.add_argument('--qatlog', default=None); ap.add_argument('--family', default=None)
    a = ap.parse_args()
    E1 = load_jobs(f'{a.res}/E1_dinowm_wp') + load_jobs(f'{a.res}/E1_dinowm_pusht')
    none = {t: by_ep(E1, lambda j, t=t: j['task'] == t and j['var'] == 'none') for t in SPLIT}
    out = dict(H3b={}, weights={}, weight_contrasts={}, qat={}, qat_contrasts={}, eps=EPS)
    selp = f'{a.res}/E5_dinowm_held/selection.json'
    sel = json.load(open(selp)) if os.path.exists(selp) else {}
    EV = load_jobs(f'{a.res}/E5_dinowm_eval') if os.path.isdir(f'{a.res}/E5_dinowm_eval') else []
    cells = []
    for key, s in sel.items():
        t, k, B = s['task'], int(s['k']), s['chosen']
        if B == 'main':
            det = by_ep(E1, lambda j: j['task'] == t and j['var'] == 'det' and int(j['k']) == k)
            sh = by_ep(E1, lambda j: j['task'] == t and j['var'] == 'shiter' and int(j['k']) == k)
        else:
            det = by_ep(EV, lambda j: j['task'] == t and j['var'] == 'det' and int(j['k']) == k and j.get('baseline') == B)
            sh = by_ep(EV, lambda j: j['task'] == t and j['var'] == 'shiter' and int(j['k']) == k and j.get('baseline') == B)
        eps = sorted(set(det) & set(sh) & set(none[t]))
        if not eps:
            continue
        g = np.array([float(sh[e]['success']) - float(det[e]['success']) for e in eps])
        c = dict(task=t, k=k, B=B, n=len(eps), Bdet=dstats(none[t], det), Bsh=dstats(none[t], sh),
                 gain=float(g.mean()), gain_ci=A.boot(lambda i: g[i].mean(), len(eps)))
        c['gain_lower_gt0'] = gt(c['gain_ci'][0], 0.0)
        c['gain_lower_gt0_plain'] = bool(c['gain_ci'][0] > 0)
        c['complete'] = bool(len(eps) == N_EP and c['Bdet']['complete'] and c['Bsh']['complete'])
        Sn = np.array([float(none[t][e]['success']) for e in eps])
        Sd = np.array([float(det[e]['success']) for e in eps]); Ss = np.array([float(sh[e]['success']) for e in eps])
        n10, n01, p = A.mcnemar(Ss.astype(bool), Sd.astype(bool))
        c['mcnemar'] = dict(shiter_only=n10, det_only=n01, p=p)

        def red(i):
            dd = (Sn - Sd)[i].mean(); dv = (Sn - Ss)[i].mean()
            return (dd - dv) / dd if dd > 0 else np.nan
        r0 = red(np.arange(len(eps)))
        c['red'] = None if not np.isfinite(r0) else float(r0)
        c['red_ci'] = A.boot(red, len(eps))
        c['held'] = s.get('held'); c['held_tied_at_max'] = [r['tag'] for r in s.get('held', []) if r['s'] == max(x['s'] for x in s['held'])]
        cells.append(c); out['H3b'][key] = c
    if cells:
        npos = sum(c['gain_lower_gt0'] for c in cells)
        out['H3b_verdict'] = dict(n_cells=len(cells), n_lower_gt0=npos, holds=bool(npos > len(cells) / 2),
                                  n_lower_gt0_plain=sum(c['gain_lower_gt0_plain'] for c in cells),
                                  complete=bool(all(c['complete'] for c in cells)),
                                  rule='majority of the E1-eligible 3-4-bit confirmatory cells with D(B+det) - D(B+sh-iter) '
                                       '95% CI lower > 0 (v5 7.3)')
    W = load_jobs(f'{a.res}/E5_dinowm_w') if os.path.isdir(f'{a.res}/E5_dinowm_w') else []
    WR = {v: by_ep(W, lambda j, v=v: j['var'] == v) for v in sorted({j['var'] for j, _ in W})}
    for v in [w for w in W_ORDER if w in WR] + [w for w in WR if w not in W_ORDER]:
        out['weights'][v] = dstats(none['pusht'], WR[v])
    for wa, wb in W_CONTRASTS:
        if wa in WR and wb in WR:
            out['weight_contrasts'][f'{wa}-{wb}'] = contrast(none['pusht'], WR[wa], WR[wb])
    Q = load_jobs(f'{a.res}/E5_dinowm_qat') if os.path.isdir(f'{a.res}/E5_dinowm_qat') else []
    QR = {v: by_ep(Q, lambda j, v=v: j['var'] == v) for v in ('none', 'det', 'shiter')}
    for v in ('none', 'det', 'shiter'):
        if QR[v]:
            out['qat'][v] = dstats(none['pusht'], QR[v])
    if 'det' in out['qat']:
        up = out['qat']['det']['D_ci'][1]
        out['qat_rule'] = dict(flag_training_free_only=lt(up, THR_QAT), flag_plain=bool(up < THR_QAT), D_qatdet_ci_upper=up,
                               rule='design 6.5 / v5 7.3: if D(QAT+det) 95% CI upper < 5 pp, the value of dither is reported '
                                    'as a training-free PTQ prescription')
        cq = out['H3b'].get('pusht_k4')
        if cq:      # v5.3 10.1 pattern check: QAT+det against the selected baseline's B+det on the same pairs
            Bdet = by_ep(EV, lambda j: j['task'] == 'pusht' and j['var'] == 'det' and int(j['k']) == 4 and j.get('baseline') == cq['B']) \
                if cq['B'] != 'main' else by_ep(E1, lambda j: j['task'] == 'pusht' and j['var'] == 'det' and int(j['k']) == 4)
            out['qat_contrasts']['D_qatdet_minus_D_Bdet'] = dict(contrast(none['pusht'], QR['det'], Bdet), B=cq['B'])
        e1det = by_ep(E1, lambda j: j['task'] == 'pusht' and j['var'] == 'det' and int(j['k']) == 4)
        out['qat_contrasts']['D_qatdet_minus_D_E1det'] = contrast(none['pusht'], QR['det'], e1det)
        out['qat_contrasts']['E1det'] = dstats(none['pusht'], e1det)
        if QR['shiter']:
            out['qat_contrasts']['D_qatdet_minus_D_qatshiter'] = contrast(none['pusht'], QR['det'], QR['shiter'])
        if QR['none']:
            out['qat_contrasts']['D_qatnone_vs_E1none_mcnemar'] = contrast(none['pusht'], QR['none'], none['pusht'])
            for v in ('det', 'shiter'):
                if QR[v]:
                    out['qat_contrasts'][f'D_{v}_vs_QATnone'] = dstats(QR['none'], QR[v])
    if a.qatlog and os.path.exists(a.qatlog):
        L = json.load(open(a.qatlog))
        ls = L.get('losses') or []
        out['qat_train'] = dict(final=L.get('final'), step=L.get('step'), minutes=L.get('minutes'), lr=L.get('lr'), bs=L.get('bs'),
                                ranges=L.get('ranges'), k=L.get('k'), first=ls[0] if ls else None, last=ls[-1] if ls else None,
                                min_loss=min((x['loss'] for x in ls), default=None))
    print(json.dumps(dict(H3b_verdict=out.get('H3b_verdict'), qat_rule=out.get('qat_rule')), indent=1, default=float))
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1, default=float, ensure_ascii=False)
    if a.md:
        open(a.md, 'w').write(tables(out))
    if a.family:
        json.dump(family(out), open(a.family, 'w'), indent=1, default=float, ensure_ascii=False)


def family(out):
    v = out.get('H3b_verdict') or {}
    cells = {}
    for key, c in out['H3b'].items():
        cells[key] = dict(task=c['task'], k=c['k'], B=c['B'], held=c['held'], held_tied_at_max=c['held_tied_at_max'], n=c['n'],
                          D_Bdet=c['Bdet']['D'], D_Bdet_ci=c['Bdet']['D_ci'], D_Bshiter=c['Bsh']['D'], D_Bshiter_ci=c['Bsh']['D_ci'],
                          Da_minus_Db=c['gain'], Da_minus_Db_ci=c['gain_ci'], passes=c['gain_lower_gt0'], red=c['red'],
                          red_ci=c['red_ci'], mcnemar=c['mcnemar'])
    return dict(
        source='analyze_e5.py (DINO-WM; eval pairs 0-99 (PushT eval6, PointMaze eval), 100 pairs; B selected on held pairs 0-19)',
        rule_H3b=v.get('rule'), units='D: success-rate drop (fraction) vs the E1 none of the same pairs and planner seeds',
        H3b=dict(cells=v.get('n_cells'), passing=v.get('n_lower_gt0'), holds=v.get('holds'), complete=v.get('complete')),
        H3b_cells=cells,
        qat=dict(rule=(out.get('qat_rule') or {}).get('rule'), flag_training_free_only=(out.get('qat_rule') or {}).get('flag_training_free_only'),
                 conditions={k: dict(D=x['D'], D_ci=x['D_ci'], success=x['success'], n=x['n']) for k, x in out['qat'].items()},
                 contrasts=out['qat_contrasts'], train=out.get('qat_train')),
        weights=dict(conditions={k: dict(D=x['D'], D_ci=x['D_ci'], success=x['success'], n=x['n']) for k, x in out['weights'].items()},
                     contrasts={k: dict(diff=x['diff'], ci=x['ci'], mcnemar_p=x['mcnemar']['p']) for k, x in out['weight_contrasts'].items()}))


def tables(out):
    def pp(x, c):
        return f'{100 * x:+.0f} [{100 * c[0]:.0f}, {100 * c[1]:.0f}]'
    L = ['| 칸 | B | n | D(B+det) %p | D(B+sh-iter) %p | D(B+det) − D(B+sh-iter) %p [95% CI] | CI 하한 > 0 | McNemar p | red [95% CI] |',
         '|' + '---|' * 9]
    for key, c in out['H3b'].items():
        red = '—' if c['red'] is None else f"{c['red']:.2f} [{c['red_ci'][0]:.2f}, {c['red_ci'][1]:.2f}]"
        L.append(f"| {TN[c['task']]} k={c['k']} | {c['B']} | {c['n']} | {pp(c['Bdet']['D'], c['Bdet']['D_ci'])} | "
                 f"{pp(c['Bsh']['D'], c['Bsh']['D_ci'])} | {pp(c['gain'], c['gain_ci'])} | {c['gain_lower_gt0']} | "
                 f"{c['mcnemar']['p']:.3g} | {red} |")
    v = out.get('H3b_verdict')
    if v:
        L += ['', f"- H3b DINO-WM: CI 하한 > 0인 칸 {v['n_lower_gt0']}/{v['n_cells']} → {'성립' if v['holds'] else '불성립'}"]
    L += ['', '| 조건 | n | 성공 | D %p [95% CI] |', '|---|---|---|---|']
    for name, grp in (('가중치', out['weights']), ('QAT', out['qat'])):
        for k, s in grp.items():
            L.append(f"| {name} {k} | {s['n']} | {100 * s['success']:.0f}% | {pp(s['D'], s['D_ci'])} |")
    L += ['', '| 대비 | n | 차 %p [95% CI] | McNemar p |', '|---|---|---|---|']
    for k, c in list(out['weight_contrasts'].items()) + [(k, c) for k, c in out['qat_contrasts'].items() if 'diff' in c]:
        L.append(f"| {k} | {c['n']} | {pp(c['diff'], c['ci'])} | {c['mcnemar']['p']:.3g} |")
    q = out.get('qat_rule')
    if q:
        L += ['', f"- QAT 규칙: D(QAT+det) CI 상한 {100 * q['D_qatdet_ci_upper']:.1f}%p < 5%p → {q['flag_training_free_only']}"]
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()
