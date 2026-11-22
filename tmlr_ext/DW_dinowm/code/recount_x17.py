"""[post hoc] cross-check of analyze_x17.py from the job JSONs (independent code; WORKER_RULES 7): success counts,
D and CI of det and sh-iter, red(sh-iter), D(sh-iter) - D(det), episode agreement with E1, and the reading rule.
usage: python recount_x17.py <raw dir> <x17 json> <e1 reference json> [--pod dw.6] [--out JSON]"""
import json, glob, argparse
import numpy as np


def ci(d):
    rng = np.random.RandomState(0)
    m = d[rng.randint(0, len(d), size=(10000, len(d)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw'); ap.add_argument('x17'); ap.add_argument('e1ref'); ap.add_argument('--pod', default='dw.6')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    S = {}
    for f in glob.glob(f'{a.raw}/*/jobs/*/*.json'):
        tag = f.split('/')[-4]
        d = json.load(open(f))
        j = d['job']
        if tag != a.pod or j['task'] != 'pusht' or j['split'] != 'eval6' or (j.get('planner') or {}).get('mppi_c') is not None:
            continue
        key = 'none' if j['var'] == 'none' else (j['var'] if (j['ranges'] == 'main' and int(j['k']) == 4) else None)
        if key is None:
            continue
        for r in d['records']:
            assert int(r['ep']) not in S.get(key, {}), ('duplicate', key, r['ep'])
            S.setdefault(key, {})[int(r['ep'])] = float(bool(r['success']))
    X = json.load(open(a.x17))
    ref = json.load(open(a.e1ref))['success']
    rows, bad = [], 0

    def cmp(name, mine, theirs):
        nonlocal bad
        ok = (mine == theirs) if isinstance(mine, str) else np.allclose(np.asarray(mine, float), np.asarray(theirs, float),
                                                                         atol=1e-12)
        bad += 0 if ok else 1
        rows.append(dict(item=name, recount=mine, analysis=theirs, match=bool(ok)))
    none = S['none']
    cmp('none n', len(none), X['cell']['none']['n'])
    cmp('none success', sum(none.values()) / len(none), X['cell']['none']['success'])
    for v in ('det', 'shiter'):
        c = S[v]
        eps = sorted(set(c) & set(none))
        d = np.array([none[e] - c[e] for e in eps])
        cmp(f'{v} n', len(eps), X[v]['n'])
        cmp(f'{v} successes', sum(c[e] for e in eps), round(X[v]['success'] * len(eps)))
        cmp(f'{v} D', d.mean(), X[v]['D'])
        cmp(f'{v} D CI', ci(d), X[v]['D_ci'])
    eps = sorted(set(S['det']) & set(S['shiter']) & set(none))
    dd = np.mean([none[e] - S['det'][e] for e in eps]); dv = np.mean([none[e] - S['shiter'][e] for e in eps])
    if dd > 0:
        cmp('shiter red', (dd - dv) / dd, X['shiter']['red'])
    else:
        rows.append(dict(item='shiter red', recount=None, analysis=X['shiter'].get('red'), match=X['shiter'].get('red') is None))
        bad += 0 if X['shiter'].get('red') is None else 1
    cmp('D_shiter - D_det', np.mean([S['det'][e] - S['shiter'][e] for e in eps]), X['shiter']['Dv_minus_Ddet']['diff'])
    for v in ('none', 'det', 'shiter'):
        m = {int(e): s for e, s in ref[v].items()}
        ee = sorted(set(S[v]) & set(m))
        cmp(f'{v} agreement with E1', np.mean([bool(S[v][e]) == m[e] for e in ee]), X['agreement_with_E1'][v]['agreement'])
        cmp(f'{v} E1 successes', sum(m[e] for e in ee), X['agreement_with_E1'][v]['e1_successes'])
    lo, hi = X['det']['D_ci']
    if lo > 1e-9 and hi >= 0.14 - 1e-9 and lo <= 0.34 + 1e-9:
        r = '(i)'
    elif lo <= 1e-9 and hi >= -1e-9 and hi < 0.14 - 1e-9:
        r = '(ii)'
    else:
        r = '(iii)'
    cmp('reading', r, X['reading'])
    out = dict(n_items=len(rows), mismatches=bad, rows=rows)
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)
    print(json.dumps(dict(n_items=len(rows), mismatches=bad)))
    for r in rows:
        if not r['match']:
            print('MISMATCH', r)


if __name__ == '__main__':
    main()
