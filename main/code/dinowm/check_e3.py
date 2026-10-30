"""Data check of the DINO-WM E3 runs (manifest E3_dinowm, DESIGN_main v5 7.2) and of their links to the E1 cells, before
the E3 analysis. CPU only: reads the manifests, the job JSONs and the population arrays, never a GPU.

usage: python check_e3.py <dinowm_main dir> [--out JSON]

Per E3 job: the JSON exists and its job equals the manifest job; meta: default planner (no override), opt_steps of the
task (PushT 30, Wall 10, PointMaze 10), salt / pair seed of the split (eval6: 'eval6' / 6, eval: 'eval5' / 5), ranges
main, no qopts / ckpt, bf16 autocast, TF32, SDPA, CUDA device order PCI_BUS_ID; records: episodes ep0..ep1-1 once,
task / split / var / k of the job, planner seed (4000 eval6, 3000 eval) + ep, E of the job, one best cost per CEM
iteration, finite OEI inputs incl. it0_rand_dJ_sd (var_cand), planning-time errors; hook calls constant per
(cell, variant). Per cell: 100 episodes per variant, every job on one GPU.
E1 links (the cells E3 reuses): none (k = 0), det, ind, sh-iter, sh-plan (shstep) at the same (task, k): 100 episodes
each, same split, the same planner seed per episode, the same E, one GPU per variant; the iteration-0 unquantized costs
of every E3 episode equal those of the E1 none episode (population arrays it0_J_none, max |diff|), i.e. the same
start / goal pair and the same candidate stream.
"""
import os, json, glob, argparse
from collections import defaultdict
import numpy as np

OPT = dict(pusht=30, wall=10, pointmaze=10)
SPLIT = dict(pusht='eval6', wall='eval', pointmaze='eval')
SALT = dict(eval6=('eval6', 6, 4000), eval=('eval5', 5, 3000))
E1MAN = dict(pusht='E1_dinowm_pusht', wall='E1_dinowm_wp', pointmaze='E1_dinowm_wp')
E1VARS = ('none', 'det', 'ind', 'shiter', 'shstep')
OEI_KEYS = ('it0_rand_dJ', 'it0_rand_dJ_sd', 'last_pop_dJ', 'last_elite_dJ', 'mf_dJ', 'oei_ref_mf', 'oei_ref_argmin')


def check_records(jid, job, d, problems):
    m, recs = d['meta'], d['records']
    t, sp = job['task'], job['split']
    salt, pseed, s0 = SALT[sp]
    if m.get('planner'):
        problems.append(f"{jid}: planner override {m.get('planner')}")
    for key, want in (('opt_steps', OPT[t]), ('salt', salt), ('pair_seed', pseed), ('ranges', 'main'), ('qopts', None),
                      ('ckpt', None), ('autocast', 'bf16'), ('tf32', True), ('sdpa', True),
                      ('cuda_device_order', 'PCI_BUS_ID')):
        if key in ('qopts', 'ckpt', 'opt_steps') and key not in m:
            continue                                     # E1 jobs started before these meta keys existed
        if m.get(key) != want:
            problems.append(f'{jid}: meta {key} = {m.get(key)!r} (expected {want!r})')
    eps = [int(r['ep']) for r in recs]
    if eps != list(range(int(job['ep0']), int(job['ep1']))):
        problems.append(f'{jid}: episodes {eps}')
    for r in recs:
        e = int(r['ep'])
        for key in ('task', 'split', 'var'):
            if r[key] != job[key]:
                problems.append(f'{jid} ep {e}: {key} {r[key]} != {job[key]}')
        if int(r['k']) != int(job['k']):
            problems.append(f"{jid} ep {e}: k {r['k']} != {job['k']}")
        if int(r['plan_seed']) != s0 + e:
            problems.append(f"{jid} ep {e}: plan seed {r['plan_seed']}")
        if int(r['E']) != int(job['E']):
            problems.append(f"{jid} ep {e}: E {r['E']} (job {job['E']})")
        if len(r['best']) != OPT[t]:
            problems.append(f"{jid} ep {e}: {len(r['best'])} per-iteration bests")
        o = r.get('oei') or {}
        miss = [k for k in OEI_KEYS if not np.isfinite(o.get(k, np.nan))]
        if miss:
            problems.append(f'{jid} ep {e}: OEI keys missing / not finite {miss}')
        if job['var'] != 'none' and 'err' not in r:
            problems.append(f'{jid} ep {e}: no planning-time errors')
    return {int(r['ep']): r for r in recs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root'); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    R = a.root
    problems, rep = [], dict(cells={}, e1_links={})
    man = json.load(open(f'{R}/code/manifests/E3_dinowm.json'))
    jobs = man['jobs']
    rep['manifest'] = dict(name=man['name'], n_jobs=len(jobs), cells=man['config']['cells'])
    want_jobs = dict(pusht_k4=40, pointmaze_k3=40, pusht_k3=10, wall_k5=20, wall_k3=20)
    n_by_cell = defaultdict(int)
    for j in jobs:
        n_by_cell[j['cell']] += 1
    if dict(n_by_cell) != want_jobs:
        problems.append(f'manifest jobs per cell {dict(n_by_cell)} != {want_jobs}')
    cfg = {(c['task'], c['k']): c for c in man['config']['cells']}
    rec3 = defaultdict(dict)                     # (cell, var) -> {ep: record}
    gpus, hooks, t0, t1, gs, sec = (defaultdict(set), defaultdict(set), defaultdict(list), defaultdict(list),
                                    defaultdict(float), defaultdict(list))
    for j in jobs:
        t, k = j['task'], int(j['k'])
        c = cfg.get((t, k))
        if c is None or j['cell'] != f'{t}_k{k}' or j['var'] not in c['vars'] or j['split'] != SPLIT[t] \
                or int(j['E']) != (2 if t == 'wall' else 1) or j['ranges'] != 'main' or not j['diag'] or j.get('planner'):
            problems.append(f"manifest job {j['id']} does not match the E3 design")
        p = f"{R}/results/{man['name']}/{t}/{j['id']}.json"
        if not os.path.exists(p):
            problems.append(f"{j['id']}: no result JSON"); continue
        d = json.load(open(p))
        if d['job'] != j:
            problems.append(f"{j['id']}: job in JSON differs from the manifest job")
        rr = check_records(j['id'], j, d, problems)
        key = (j['cell'], j['var'])
        for e, r in rr.items():
            if e in rec3[key]:
                problems.append(f"{j['id']}: episode {e} twice")
            rec3[key][e] = r
        gpus[j['cell']].add((d['meta']['gpu'], d['meta']['cuda_visible'], d['meta']['host']))
        hooks[key] |= {int(r['hook_calls']) for r in d['records']}
        t0[j['cell']].append(d['meta']['t_start']); t1[j['cell']].append(d['meta']['t_finalized'])
        gs[j['cell']] += float(d['meta']['gpu_s']); sec[key] += [float(r['time_s']) for r in d['records']]
    extra = sorted(set(os.path.basename(f)[:-5] for f in glob.glob(f"{R}/results/{man['name']}/*/*.json")) - {j['id'] for j in jobs})
    others = sorted(f for f in glob.glob(f"{R}/results/{man['name']}/*/*") if not f.endswith('.json'))
    if extra:
        problems.append(f'result JSONs outside the manifest: {extra}')
    if others:
        problems.append(f'non-JSON files in the result folder: {others}')
    # E1 cells reused by E3 (and their none for D)
    e1 = {}
    for t in ('pusht', 'wall', 'pointmaze'):
        m1 = {jj['id']: jj for jj in json.load(open(f'{R}/code/manifests/{E1MAN[t]}.json'))['jobs']}
        for f in sorted(glob.glob(f'{R}/results/{E1MAN[t]}/{t}/*.json')):
            d = json.load(open(f))
            j = d['job']
            if j['split'] != SPLIT[t] or j['var'] not in E1VARS:
                continue
            if m1.get(j['id']) != j:
                problems.append(f"E1 job {j['id']}: not equal to its manifest job")
            rr = check_records(j['id'], j, d, problems)
            k = 0 if j['var'] == 'none' else int(j['k'])
            for e, r in rr.items():
                e1.setdefault((t, j['var'], k), {})[e] = (r, d['meta']['cuda_visible'], f)
    for c in sorted(cfg.values(), key=lambda c: (c['task'], -c['k'])):
        t, k = c['task'], int(c['k'])
        cell = f'{t}_k{k}'
        row = dict(task=t, k=k, role=c['role'], vars=c['vars'], gpu=sorted(map(list, gpus[cell])), t_start=min(t0[cell]),
                   t_end=max(t1[cell]), gpu_h=gs[cell] / 3600)
        if len(gpus[cell]) != 1:
            problems.append(f'{cell}: jobs on more than one GPU {gpus[cell]}')
        for v in c['vars']:
            e = sorted(rec3[(cell, v)])
            row[f'n_{v}'] = len(e); row[f'hook_calls_{v}'] = sorted(hooks[(cell, v)])
            row[f's_per_ep_{v}'] = float(np.mean(sec[(cell, v)])) if sec[(cell, v)] else None
            if e != list(range(100)):
                problems.append(f'{cell} {v}: episodes cover {len(e)} (expected 0..99 once)')
            if len(hooks[(cell, v)]) != 1:
                problems.append(f'{cell} {v}: hook calls not constant {sorted(hooks[(cell, v)])}')
        # links to E1: the E1 cells of this (task, k) and the none episodes
        link = {}
        for v in E1VARS:
            kk = 0 if v == 'none' else k
            got = e1.get((t, v, kk), {})
            link[f'n_{v}'] = len(got)
            link[f'gpu_{v}'] = sorted({g for _, g, _ in got.values()})
            if sorted(got) != list(range(100)):
                problems.append(f'E1 {t} {v} k={kk}: episodes cover {len(got)}')
            if len(link[f'gpu_{v}']) > 1:
                problems.append(f'E1 {t} {v} k={kk}: more than one GPU {link[f"gpu_{v}"]}')
        none = e1.get((t, 'none', 0), {})
        n_seed, n_fp, n_fpa, dmax, n_arr = 0, 0, 0, 0.0, 0
        for v in c['vars']:
            for e, r in rec3[(cell, v)].items():
                if e not in none:
                    continue
                rn = none[e][0]
                n_seed += int(r['plan_seed']) == int(rn['plan_seed']) and int(r['E']) == int(rn['E'])
                if 'it0_rand_J_none' in r['oei'] and 'it0_rand_J_none' in rn['oei']:
                    n_fpa += 1
                    n_fp += r['oei']['it0_rand_J_none'] == rn['oei']['it0_rand_J_none']
                jid3 = [jj['id'] for jj in jobs if jj['cell'] == cell and jj['var'] == v and jj['ep0'] <= e < jj['ep1']][0]
                f3 = f"{R}/pops/{man['name']}/{t}/{jid3}_e{e}.npz"
                f1 = none[e][2].replace('/results/', '/pops/').replace('.json', f'_e{e}.npz')
                if os.path.exists(f3) and os.path.exists(f1):
                    z3, z1 = np.load(f3), np.load(f1)
                    if z3['it0_J_none'].shape != z1['it0_J_none'].shape:
                        problems.append(f'{cell} {v} ep {e}: it0 population shapes differ')
                        continue
                    dmax = max(dmax, float(np.abs(z3['it0_J_none'].astype(np.float64) - z1['it0_J_none']).max()))
                    n_arr += 1
                else:
                    problems.append(f'{cell} {v} ep {e}: population array missing ({os.path.basename(f3)} / {os.path.basename(f1)})')
        tot = sum(len(rec3[(cell, v)]) for v in c['vars'])
        link.update(n_e3_episodes=tot, same_plan_seed_and_E=n_seed, it0_rand_J_none_compared=n_fpa, same_it0_rand_J_none=n_fp,
                    it0_arrays_compared=n_arr, it0_J_none_max_abs_diff=dmax)
        if n_seed != tot or n_arr != tot:
            problems.append(f'{cell}: planner seed / E or population link mismatch ({n_seed}, {n_arr} of {tot})')
        if n_fp != n_fpa or dmax != 0.0:
            problems.append(f'{cell}: iteration-0 unquantized costs differ from E1 none ({n_fp}/{n_fpa} equal, max |diff| {dmax:.3g})')
        rep['cells'][cell] = row
        rep['e1_links'][cell] = link
    rep['problems'] = problems
    rep['n_problems'] = len(problems)
    print(json.dumps(rep, indent=1, default=str))
    if a.out:
        json.dump(rep, open(a.out, 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
