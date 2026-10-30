"""S3 rule selection on the held split (and evaluation helper). Thresholds tau_q = per-task q-quantile of R_boot at the
middle opt step of the held baseline runs. ES(q) is evaluated from the recorded per-step outcomes of the baseline run.
usage: python select7.py taus | select | evalES <q>"""
import sys, os, json
import numpy as np
RES = '/workspace/results'
MID = {'pusht': 14, 'wall': 4}
QS = [0.25, 0.5, 0.75]


def load(task, split, sched, tag=''):
    R = {}
    p = f'{RES}/cem7_{task}_{split}_{sched}{tag}.jsonl'
    if os.path.exists(p):
        for ln in open(p):
            r = json.loads(ln); R[r['ep']] = r
    return R


def es_outcome(r, tau):
    for p in r['prox']:
        if p['step'] >= 1 and p['R_boot'] > tau:
            t = p['step']
            return r['traj_success'][t], r['traj_margin'][t], t
    return r['traj_success'][-1], r['traj_margin'][-1], None


def taus():
    out = {}
    for task in MID:
        B = load(task, 'held', 'base')
        v = np.array([r['prox'][MID[task]]['R_boot'] for r in B.values()])
        out[task] = {str(q): float(np.quantile(v, q)) for q in QS}
        out[task]['n'] = len(v)
    json.dump(out, open(f'{RES}/taus7.json', 'w'), indent=1)
    print(json.dumps(out))


def select():
    tau = json.load(open(f'{RES}/taus7.json'))
    rows = []
    cands = [('es', q) for q in QS] + [('fv', 0.5), ('ak', 0.5)]
    for kind, q in cands:
        row = dict(rule=f'{kind}({q})', kind=kind, q=q)
        for task in MID:
            B = load(task, 'held', 'base')
            if kind == 'es':
                S = {e: dict(success=es_outcome(r, tau[task][str(q)])[0], margin=es_outcome(r, tau[task][str(q)])[1])
                     for e, r in B.items()}
            else:
                S = load(task, 'held', kind)
            eps = sorted(set(B) & set(S))
            sb = np.array([B[e]['success'] for e in eps], float); ss = np.array([S[e]['success'] for e in eps], float)
            mb = np.array([B[e]['margin'] for e in eps]); ms = np.array([S[e]['margin'] for e in eps])
            row[f'{task}_n'] = len(eps)
            row[f'{task}_dsucc_pp'] = float(100 * (ss.mean() - sb.mean())) if eps else float('nan')
            row[f'{task}_dmargin'] = float(np.mean(ms - mb)) if eps else float('nan')
            if kind != 'es':
                row[f'{task}_trig_frac'] = float(np.mean([S[e]['trig'] is not None for e in eps])) if eps else float('nan')
            else:
                row[f'{task}_trig_frac'] = float(np.mean([es_outcome(B[e], tau[task][str(q)])[2] is not None for e in eps]))
        row['avg_dsucc_pp'] = float(np.mean([row[f'{t}_dsucc_pp'] for t in MID]))
        row['avg_dmargin'] = float(np.mean([row[f'{t}_dmargin'] for t in MID]))
        row['compute_rank'] = 0 if kind == 'es' else 1
        rows.append(row)
    best = sorted(rows, key=lambda r: (-round(r['avg_dsucc_pp'], 6), round(r['avg_dmargin'], 6), r['compute_rank']))[0]
    json.dump(dict(rows=rows, chosen=best, taus=tau), open(f'{RES}/select7.json', 'w'), indent=1)
    for r in rows:
        print(r)
    print('CHOSEN', best['rule'])
    with open(f'{RES}/chosen7.txt', 'w') as f:
        f.write(f"{best['kind']} {best['q']}\n")


if __name__ == '__main__':
    {'taus': taus, 'select': select}[sys.argv[1]]()
