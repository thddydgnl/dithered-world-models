"""[post hoc] cross-check of analyze_posthoc.py from the job JSONs (independent code; WORKER_RULES 7), plus a count of
the auxiliary OEI keys that need the iteration-0 simulator replay (missing where a job was rerun after a driver
restart: the it0 file existed without truth, so the replay was not resubmitted; not used by any verdict).
usage: python recount_posthoc.py <raw dir> <posthoc json> [--pod dw.6] [--k 3] [--out JSON]"""
import os, sys, json, glob, argparse
import numpy as np

VARS = ('det', 'detbc', 'shstep', 'detshiftfix', 'shiterb', 'shiter', 'detshift', 'detnoise')
TRUTH_KEYS = ('oei_true_mf', 'oei_true_argmin', 'rho_it0', 'calib_a')


def ci(d):
    rng = np.random.RandomState(0)
    m = d[rng.randint(0, len(d), size=(10000, len(d)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw'); ap.add_argument('posthoc'); ap.add_argument('--pod', default='dw.6')
    ap.add_argument('--k', type=int, default=3); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    S, gap = {}, {}
    for f in glob.glob(f'{a.raw}/*/jobs/*/*.json'):
        tag = f.split('/')[-4]
        d = json.load(open(f))
        j = d['job']
        for r in d['records']:
            if j['diag'] and 'oei' in r:
                g = gap.setdefault(tag, [0, 0])
                g[1] += 1
                g[0] += int(not all(kk in r['oei'] for kk in TRUTH_KEYS))
        if tag != a.pod or j['split'] != 'eval8' or (j.get('planner') or {}).get('mppi_c') is not None:
            continue
        key = 'none' if j['var'] == 'none' else (j['var'] if (j['ranges'] == 'main' and int(j['k']) == a.k) else None)
        if key is None:
            continue
        for r in d['records']:
            assert int(r['ep']) not in S.get(key, {}), ('duplicate', key, r['ep'])
            S.setdefault(key, {})[int(r['ep'])] = float(bool(r['success']))
    P = json.load(open(a.posthoc))['cell']
    rows, bad = [], 0

    def cmp(name, mine, theirs):
        nonlocal bad
        ok = np.allclose(np.asarray(mine, float), np.asarray(theirs, float), atol=1e-12)
        bad += 0 if ok else 1
        rows.append(dict(item=name, recount=mine, analysis=theirs, match=bool(ok)))
    none = S['none']
    cmp('none success', sum(none.values()) / len(none), P['none']['success'])
    for v in VARS:
        c = S[v]
        eps = sorted(set(c) & set(none))
        d = np.array([none[e] - c[e] for e in eps])
        cmp(f'{v} n', len(eps), P['variants'][v]['n'])
        cmp(f'{v} successes', sum(c[e] for e in eps), round(P['variants'][v]['success'] * len(eps)))
        cmp(f'{v} D', d.mean(), P['variants'][v]['D'])
        cmp(f'{v} D CI', ci(d), P['variants'][v]['D_ci'])
        if v != 'det':
            g = np.array([S['det'][e] - c[e] for e in eps])
            cmp(f'{v} D_v - D_det', g.mean(), P['variants'][v]['Dv_minus_Ddet']['diff'])
    for name, x, y in (('d_bias', 'det', 'detbc'), ('d_pers', 'detshiftfix', 'detshift')):
        eps = sorted(set(S[x]) & set(S[y]) & set(none))
        g = np.array([S[y][e] - S[x][e] for e in eps])
        cmp(f'{name}', g.mean(), P[name]['diff'])
        cmp(f'{name} CI', ci(g), P[name]['ci'])
    out = dict(n_items=len(rows), mismatches=bad, rows=rows,
               oei_truth_gap={t: dict(records_missing=g[0], diag_records=g[1]) for t, g in sorted(gap.items())})
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)
    print(json.dumps(dict(n_items=len(rows), mismatches=bad, oei_truth_gap=out['oei_truth_gap'])))
    for r in rows:
        if not r['match']:
            print('MISMATCH', r)


if __name__ == '__main__':
    main()
