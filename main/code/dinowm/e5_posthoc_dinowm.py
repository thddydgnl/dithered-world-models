"""POST-HOC description of the DINO-WM E5 results (written 2026-10-24 after the analyze_e5.py tables were seen; no rule,
no verdict). The E5 evaluation reuses the E1 pairs and planner seeds, so the selected baseline B can be compared with
the E1 grid on the same pairs: per H3b cell D(B+det) - D(E1 det) and D(B+sh-iter) - D(E1 sh-iter) (paired bootstrap,
10,000, RandomState(0); McNemar), and for QAT (PushT k=4) the QAT model's det against its own none (McNemar).
usage: python e5_posthoc_dinowm.py <results dir> --out JSON [--md MD]
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_e5 as E5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('res'); ap.add_argument('--out', required=True); ap.add_argument('--md', default=None)
    a = ap.parse_args()
    E1 = E5.load_jobs(f'{a.res}/E1_dinowm_wp') + E5.load_jobs(f'{a.res}/E1_dinowm_pusht')
    EV = E5.load_jobs(f'{a.res}/E5_dinowm_eval')
    Q = E5.load_jobs(f'{a.res}/E5_dinowm_qat')
    sel = json.load(open(f'{a.res}/E5_dinowm_held/selection.json'))
    none = {t: E5.by_ep(E1, lambda j, t=t: j['task'] == t and j['var'] == 'none') for t in E5.SPLIT}
    out = dict(note='post-hoc description (after the E5 tables were seen); no verdict', cells={}, qat={})
    for key, s in sel.items():
        t, k, B = s['task'], int(s['k']), s['chosen']
        row = dict(B=B)
        for v in ('det', 'shiter'):
            e1 = E5.by_ep(E1, lambda j: j['task'] == t and j['var'] == v and int(j['k']) == k)
            bv = E5.by_ep(EV, lambda j: j['task'] == t and j['var'] == v and int(j['k']) == k and j.get('baseline') == B)
            row[f'E1_{v}'] = E5.dstats(none[t], e1)
            row[f'D_B{v}_minus_D_E1{v}'] = E5.contrast(none[t], bv, e1)
        out['cells'][key] = row
    qn = E5.by_ep(Q, lambda j: j['var'] == 'none'); qd = E5.by_ep(Q, lambda j: j['var'] == 'det')
    qs = E5.by_ep(Q, lambda j: j['var'] == 'shiter')
    out['qat']['D_qatdet_minus_D_qatnone'] = E5.contrast(none['pusht'], qd, qn)
    out['qat']['D_qatshiter_minus_D_qatnone'] = E5.contrast(none['pusht'], qs, qn)
    json.dump(out, open(a.out, 'w'), indent=1, default=float, ensure_ascii=False)
    pp = lambda d: f"{100 * d['diff']:+.0f} [{100 * d['ci'][0]:+.0f}, {100 * d['ci'][1]:+.0f}]"
    L = ['| 칸 | B | E1 det D %p | D(B+det) − D(E1 det) %p [CI] | McNemar p | E1 sh-iter D %p | D(B+sh-iter) − D(E1 sh-iter) %p [CI] | McNemar p |',
         '|' + '---|' * 8]
    for key, r in out['cells'].items():
        cd, cs = r['D_Bdet_minus_D_E1det'], r['D_Bshiter_minus_D_E1shiter']
        L.append(f"| {key} | {r['B']} | {100 * r['E1_det']['D']:+.0f} | {pp(cd)} | {cd['mcnemar']['p']:.3g} | "
                 f"{100 * r['E1_shiter']['D']:+.0f} | {pp(cs)} | {cs['mcnemar']['p']:.3g} |")
    for k, c in out['qat'].items():
        L.append(f"\n- QAT {k}: {pp(c)}, McNemar p {c['mcnemar']['p']:.3g} (b_only {c['mcnemar']['b_only']}, a_only {c['mcnemar']['a_only']})")
    txt = '\n'.join(L) + '\n'
    print(txt)
    if a.md:
        open(a.md, 'w').write(txt)


if __name__ == '__main__':
    main()
