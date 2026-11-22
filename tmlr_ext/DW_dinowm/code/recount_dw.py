"""Phase B cross-check (WORKER_RULES 7): recount key numbers straight from the job JSONs, independently of
analyze_tmlr.py, and compare them with its output (analysis_dw.json). Pure python + numpy; runs on the pod.
usage: python recount_dw.py <raw dir> <analysis_dw.json> [--out JSON]
Recounts: E11 none and variant successes, D, D_v - D_det and the paired bootstrap CI of D(det) (own implementation of
the RandomState(0), 10,000-resample percentile rule); E12 cell sizes and D; E16 lambda choice and none MPPI success."""
import os, sys, json, glob, argparse
import numpy as np


def load(raw):
    R = {}
    for f in glob.glob(f'{raw}/*/jobs/*/*.json'):
        pod = f.split('/')[-4].rstrip('bcdefgh')
        d = json.load(open(f))
        j = d['job']
        pl = j.get('planner') or {}
        pt = 'cem' if 'mppi_c' not in pl else f"mppi{pl['mppi_c']:g}"
        key = (pod, j['task'], j['split'], pt, '-' if j['var'] == 'none' else j['ranges'], j['var'],
               0 if j['var'] == 'none' else int(j['k']))
        for r in d['records']:
            assert int(r['ep']) not in R.get(key, {}), ('duplicate episode', key, r['ep'])
            R.setdefault(key, {})[int(r['ep'])] = bool(r['success'])
    return R


def ci(d):
    rng = np.random.RandomState(0)
    n = len(d)
    m = d[rng.randint(0, n, size=(10000, n))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw'); ap.add_argument('analysis'); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    R = load(a.raw)
    A = json.load(open(a.analysis))
    rows, bad = [], 0

    def cmp(name, mine, theirs, tol=1e-12):
        nonlocal bad
        ok = (mine is None and theirs is None) or (mine is not None and theirs is not None and
                                                   np.allclose(np.asarray(mine, float), np.asarray(theirs, float), atol=tol))
        bad += 0 if ok else 1
        rows.append(dict(item=name, recount=mine, analysis=theirs, match=bool(ok)))
    cells = {(c['task'], c['k']): c for c in A['E11']['cells']}
    for task, k, pod in (('pusht', 4, 'dw.1'), ('pointmaze', 3, 'dw.3'), ('wall', 3, 'dw.4')):
        none = R[(pod, task, 'eval8', 'cem', '-', 'none', 0)]
        C = cells[(task, k)]
        cmp(f'E11 {task} none success', sum(none.values()) / len(none), C['none']['success'])
        det = R[(pod, task, 'eval8', 'cem', 'main', 'det', k)]
        for v, s in C['variants'].items():
            c = R[(pod, task, 'eval8', 'cem', 'main', v, k)]
            eps = sorted(set(c) & set(none))
            d = np.array([float(none[e]) - float(c[e]) for e in eps])
            cmp(f'E11 {task} {v} n', len(eps), s['n'])
            cmp(f'E11 {task} {v} successes', sum(c[e] for e in eps), round(s['success'] * s['n']))
            cmp(f'E11 {task} {v} D', float(d.mean()), s['D'])
            if v == 'det':
                cmp(f'E11 {task} det D CI', ci(d), s['D_ci'])
            else:
                e2 = sorted(set(c) & set(det) & set(none))
                g = np.array([float(det[e]) - float(c[e]) for e in e2])
                cmp(f'E11 {task} {v} D_v-D_det', float(g.mean()), s['Dv_minus_Ddet']['diff'])
    for task, T in A['E12'].items():
        for kind, byv in T['cells'].items():
            for var, byk in byv.items():
                for k, s in byk.items():
                    rid = s['ranges']
                    c = R[(s['pod'], task, 'eval8', 'cem', rid, var, int(k))]
                    none = R[(s['pod'], task, 'eval8', 'cem', '-', 'none', 0)]
                    eps = sorted(set(c) & set(none))
                    cmp(f'E12 {task} {rid} {var} k{k} n', len(eps), s['n'])
                    cmp(f'E12 {task} {rid} {var} k{k} D', float(np.mean([float(none[e]) - float(c[e]) for e in eps])), s['D'])
    for task, r in A['E16'].items():
        pod = r['pod']
        rows_l = {}
        for c in (0.25, 0.5, 1.0, 2.0):
            h = R[(pod, task, 'held', f'mppi{c:g}', '-', 'none', 0)]
            rows_l[c] = sum(h.values()) / len(h)
        best = max(rows_l, key=lambda c: (rows_l[c], c))
        cmp(f'E16 {task} chosen c', best, r['lambda_selection']['chosen_c'])
        nm = R[(pod, task, 'eval8', f'mppi{best:g}', '-', 'none', 0)]
        cmp(f'E16 {task} none MPPI success', sum(nm.values()) / len(nm), r['none_mppi'])
    out = dict(n_items=len(rows), mismatches=bad, rows=rows)
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)
    print(json.dumps(dict(n_items=len(rows), mismatches=bad)))
    for r in rows:
        if not r['match']:
            print('MISMATCH', r)


if __name__ == '__main__':
    main()
