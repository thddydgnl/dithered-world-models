"""Data check of the DINO-WM E2 candidate-count sweep (manifest E2_dinowm_N) and of its sixth point (the E1 PushT k=4
cell, N = 300, top-30), before the E2 analysis (DESIGN_main v5 7.0 / 7.1, v5.1 8.2). CPU only: reads the manifest,
the job JSONs and the population arrays, never a GPU.

usage: python check_e2n.py <dinowm_main dir> [--man E2_dinowm_N | E2_dinowm_EO] [--out JSON]
(--man E2_dinowm_EO, 2026-10-24: the elite-ratio / opt_steps manifest; per cell N = num_samples (default 300) and
opt_steps (default 30) come from its planner setting, so the opt_steps / per-iteration-best checks use that value.)

Per job: the JSON exists and its job equals the manifest job; meta: planner = the job's planner, opt_steps 30, split
eval6 (pair seed 6, salt eval6), ranges main, no qopts / ckpt, bf16 autocast, TF32, SDPA, CUDA device order PCI_BUS_ID;
records: episodes ep0..ep1-1 once each, task / split / var / k of the job, planner seed 4000 + ep, E = 1, one best cost
per CEM iteration (30), the OEI inputs, planning-time errors for the quantized variants; hook calls constant per
(cell, variant).
Per cell: 100 episodes per variant (none, det, sh-iter), every job on one GPU; population arrays with N rows; the
iteration-0 unquantized costs (it0_J_none) identical for the three variants of a cell and episode (same random stream).
The E1 point: none (k = 0), det k=4, sh-iter k=4 of E1_dinowm_pusht, the default planner (no planner key, 300 / 30).
"""
import os, sys, json, glob, argparse
from collections import defaultdict
import numpy as np

VARS = ('none', 'det', 'shiter')
OEI_KEYS = ('it0_rand_dJ', 'it0_rand_dJ_sd', 'last_pop_dJ', 'last_elite_dJ', 'mf_dJ', 'oei_ref_mf', 'oei_ref_argmin')
OPT_STEPS = 30


def check_job(job, d, problems, planner_expected, lenient=False, opt=OPT_STEPS):
    """lenient (E1 jobs): meta keys added to the runner after E1 started (opt_steps, planner, qopts, ckpt) may be
    absent; the number of CEM iterations is then checked through the 30 per-iteration bests of every record."""
    jid = job['id']
    m, recs = d['meta'], d['records']
    if not lenient and d['job'] != job:
        problems.append(f'{jid}: job in JSON differs from the manifest job')
    if (m.get('planner') or None) != (planner_expected or None):
        problems.append(f"{jid}: meta planner {m.get('planner')} != {planner_expected}")
    for key, want in (('opt_steps', opt), ('salt', 'eval6'), ('pair_seed', 6), ('ranges', 'main'), ('qopts', None),
                      ('ckpt', None), ('autocast', 'bf16'), ('tf32', True), ('sdpa', True),
                      ('cuda_device_order', 'PCI_BUS_ID')):
        if lenient and key in ('opt_steps', 'qopts', 'ckpt') and key not in m:
            continue
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
        if int(r['plan_seed']) != 4000 + e:
            problems.append(f"{jid} ep {e}: plan seed {r['plan_seed']}")
        if int(r['E']) != 1:
            problems.append(f"{jid} ep {e}: E {r['E']}")
        if len(r['best']) != opt:
            problems.append(f"{jid} ep {e}: {len(r['best'])} per-iteration bests")
        o = r.get('oei') or {}
        miss = [k for k in OEI_KEYS if not np.isfinite(o.get(k, np.nan))]
        if miss:
            problems.append(f'{jid} ep {e}: OEI keys missing / not finite {miss}')
        if job['var'] != 'none' and 'err' not in r:
            problems.append(f'{jid} ep {e}: no planning-time errors')
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root'); ap.add_argument('--out', default=None); ap.add_argument('--man', default='E2_dinowm_N')
    a = ap.parse_args()
    R = a.root
    problems, report = [], dict(cells={}, e1_point={})
    man = json.load(open(f'{R}/code/manifests/{a.man}.json'))
    settings = man['config']['settings']
    jobs = man['jobs']
    report['manifest'] = dict(name=man['name'], n_jobs=len(jobs), settings=settings, config_k=man['config']['k'],
                              config_split=man['config']['split'])
    cover = defaultdict(list)          # (cell, var) -> episodes
    gpus, hooks, t0, t1, gpu_s, peak = (defaultdict(set), defaultdict(set), defaultdict(list), defaultdict(list),
                                        defaultdict(float), defaultdict(float))
    sec_ep = defaultdict(list)
    for j in jobs:
        tag = j['id'].rsplit('_', 1)[1]
        if j['cell'] != f'pusht_k4_{tag}' or j['task'] != 'pusht' or j['split'] != 'eval6' or int(j['E']) != 1 \
                or j['ranges'] != 'main' or not j['diag'] or j['var'] not in VARS \
                or int(j['k']) != (0 if j['var'] == 'none' else 4) or j['planner'] != settings[tag]:
            problems.append(f"manifest job {j['id']} does not match the E2 sweep design")
        p = f"{R}/results/{man['name']}/{j['task']}/{j['id']}.json"
        if not os.path.exists(p):
            problems.append(f"{j['id']}: no result JSON"); continue
        d = json.load(open(p))
        eps = check_job(j, d, problems, settings[tag], opt=int(settings[tag].get('opt_steps', OPT_STEPS)))
        key = (j['cell'], j['var'])
        cover[key] += eps
        gpus[j['cell']].add((d['meta']['gpu'], d['meta']['cuda_visible'], d['meta']['host']))
        hooks[key] |= {int(r['hook_calls']) for r in d['records']}
        t0[j['cell']].append(d['meta']['t_start']); t1[j['cell']].append(d['meta']['t_finalized'])
        gpu_s[j['cell']] += float(d['meta']['gpu_s']); peak[j['cell']] = max(peak[j['cell']], float(d['meta']['peak_mem_gb']))
        sec_ep[key] += [float(r['time_s']) for r in d['records']]
    extra = sorted(set(os.path.basename(f)[:-5] for f in glob.glob(f"{R}/results/{man['name']}/*/*.json")) - {j['id'] for j in jobs})
    others = sorted(f for f in glob.glob(f"{R}/results/{man['name']}/*/*") if not f.endswith('.json'))
    if extra:
        problems.append(f'result JSONs outside the manifest: {extra}')
    if others:
        problems.append(f'non-JSON files in the result folder: {others}')
    cells = sorted({j['cell'] for j in jobs}, key=lambda c: (-settings[c.rsplit('_', 1)[1]].get('num_samples', 300), c))
    for c in cells:
        tag = c.rsplit('_', 1)[1]
        N = int(settings[tag].get('num_samples', 300))
        row = dict(N=N, planner=settings[tag], gpu=sorted(map(list, gpus[c])), t_start=min(t0[c]), t_end=max(t1[c]),
                   gpu_h=gpu_s[c] / 3600, peak_mem_gb=peak[c])
        if len(gpus[c]) != 1:
            problems.append(f'{c}: jobs on more than one GPU {gpus[c]}')
        for v in VARS:
            e = sorted(cover[(c, v)])
            row[f'n_{v}'] = len(e)
            row[f'hook_calls_{v}'] = sorted(hooks[(c, v)])
            row[f's_per_ep_{v}'] = float(np.mean(sec_ep[(c, v)])) if sec_ep[(c, v)] else None
            if e != list(range(100)):
                problems.append(f'{c} {v}: episodes cover {len(e)} (expected 0..99 once)')
            if len(hooks[(c, v)]) != 1:
                problems.append(f'{c} {v}: hook calls not constant {sorted(hooks[(c, v)])}')
        # population arrays: N rows, and the same iteration-0 unquantized costs for none / det / sh-iter
        n_pop, n_same, bad = 0, 0, []
        for e in range(100):
            ch = f'e{10 * (e // 10):03d}-{10 * (e // 10) + 10:03d}'
            Z = {}
            for v in VARS:
                vk = 'none' if v == 'none' else f'{v}_k4'
                f = f"{R}/pops/{man['name']}/pusht/eval6_{vk}_{ch}_{tag}_e{e}.npz"
                if not os.path.exists(f):
                    bad.append(f'missing {os.path.basename(f)}'); continue
                z = np.load(f)
                n_pop += 1
                if z['it0_J'].shape != (N,) or z['last_J'].shape != (N,) or z['last_J_none'].shape != (N,) \
                        or z['last_actions'].shape != (N, 5, 10):
                    bad.append(f'{os.path.basename(f)} shapes {z["it0_J"].shape} {z["last_actions"].shape}')
                Z[v] = z['it0_J_none']
            if len(Z) == 3 and np.array_equal(Z['none'], Z['det']) and np.array_equal(Z['none'], Z['shiter']):
                n_same += 1
        row.update(pop_files=n_pop, it0_J_none_identical_episodes=n_same)
        if bad:
            problems.append(f'{c}: population arrays {bad[:5]} ({len(bad)})')
        if n_same != 100:
            problems.append(f'{c}: iteration-0 unquantized costs identical across variants in {n_same}/100 episodes')
        report['cells'][c] = row
    # the sixth point: E1 PushT k=4 (N = 300, top-30)
    cov1, g1, gs1, t01, t11 = defaultdict(list), defaultdict(set), defaultdict(float), defaultdict(list), defaultdict(list)
    man1 = {j['id']: j for j in json.load(open(f'{R}/code/manifests/E1_dinowm_pusht.json'))['jobs']}
    for f in sorted(glob.glob(f'{R}/results/E1_dinowm_pusht/pusht/*.json')):
        d = json.load(open(f))
        j = d['job']
        if man1.get(j['id']) != j:
            problems.append(f"E1 job {j['id']}: not equal to the E1_dinowm_pusht manifest job")
        if j['var'] not in VARS or (j['var'] != 'none' and int(j['k']) != 4):
            continue
        if j.get('planner') or j.get('ckpt') or j.get('qopts'):
            problems.append(f"E1 job {j['id']} has planner / ckpt / qopts"); continue
        if j['split'] != 'eval6' or int(j['E']) != 1 or j['ranges'] != 'main' or not j['diag']:
            problems.append(f"E1 job {j['id']}: split / E / ranges / diag {j['split']} {j['E']} {j['ranges']} {j['diag']}")
        eps = check_job(j, d, problems, None, lenient=True)
        cov1[j['var']] += eps
        g1[j['var']].add((d['meta']['gpu'], d['meta']['cuda_visible'], d['meta']['host']))
        gs1[j['var']] += float(d['meta']['gpu_s']); t01[j['var']].append(d['meta']['t_start']); t11[j['var']].append(d['meta']['t_finalized'])
    for v in VARS:
        e = sorted(cov1[v])
        report['e1_point'][v] = dict(n=len(e), gpu=sorted(map(list, g1[v])), gpu_h=gs1[v] / 3600,
                                     t_start=min(t01[v]) if t01[v] else None, t_end=max(t11[v]) if t11[v] else None)
        if e != list(range(100)):
            problems.append(f'E1 point {v}: episodes cover {len(e)} (expected 0..99 once)')
    if len({x for v in ('det', 'shiter') for x in g1[v]}) != 1:
        problems.append(f'E1 point: det / sh-iter cell on more than one GPU {dict(g1)}')
    report['problems'] = problems
    report['n_problems'] = len(problems)
    print(json.dumps(report, indent=1, default=str))
    if a.out:
        json.dump(report, open(a.out, 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
