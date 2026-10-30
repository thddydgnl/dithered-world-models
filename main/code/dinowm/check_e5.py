"""Data check of the DINO-WM E5 runs (DESIGN_main v5 7.3): manifests E5_dinowm_held, E5_dinowm_eval, E5_dinowm_w,
E5_dinowm_qat against their results, the baseline selection (recomputed from the held results), the QAT training log
and the links to the E1 cells that E5 reuses (E1 none as the D reference; E1 det / sh-iter when B = 'main'). CPU only.

usage: python check_e5.py <dinowm_main dir> [--out JSON]

Per job: the JSON exists and its job equals the manifest job; meta: ranges / qopts / ckpt of the job, default planner,
opt_steps of the task, salt / pair seed of the split, bf16 / TF32 / SDPA, CUDA device order PCI_BUS_ID; records:
episodes ep0..ep1-1 once, task / split / var / k of the job, planner seed of the split + ep (held 1000, eval6 4000,
eval 3000), E of the job, one best cost per CEM iteration, finite OEI inputs; hook calls constant per (cell, variant).
Per cell: complete episode ranges (held 0-19, eval 0-99), every job on one GPU.
Selection: per (task, k) the held det successes of the 7 baselines; B = highest, ties -> earlier in make_manifest.E5B;
compared with results/E5_dinowm_held/selection.json; the eval jobs use B's ranges / qopts.
E1 links: E1 none of the task (100 episodes, same split and planner seeds); per E5 eval / weight / QAT episode the
iteration-0 unquantized costs (population arrays it0_J_none) vs the E1 none episode (max |diff|; the QAT predictor
is a different model, so its J_none is expected to differ).
"""
import os, sys, json, glob, argparse
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_manifest import E5B, E5W

OPT = dict(pusht=30, wall=10, pointmaze=10)
SPLIT_EVAL = dict(pusht='eval6', wall='eval', pointmaze='eval')
SPL = dict(held=('held', 3, 1000), eval6=('eval6', 6, 4000), eval=('eval5', 5, 3000))
E1MAN = dict(pusht='E1_dinowm_pusht', wall='E1_dinowm_wp', pointmaze='E1_dinowm_wp')
OEI_KEYS = ('it0_rand_dJ', 'it0_rand_dJ_sd', 'last_pop_dJ', 'last_elite_dJ', 'mf_dJ', 'oei_ref_mf')
MANS = ('E5_dinowm_held', 'E5_dinowm_eval', 'E5_dinowm_w', 'E5_dinowm_qat')


def check_job(jid, job, d, problems, lenient=False):
    m, recs = d['meta'], d['records']
    t, sp = job['task'], job['split']
    salt, pseed, s0 = SPL[sp]
    if m.get('planner'):
        problems.append(f"{jid}: planner override {m.get('planner')}")
    want = (('opt_steps', OPT[t]), ('salt', salt), ('pair_seed', pseed), ('ranges', job.get('ranges')),
            ('qopts', job.get('qopts')), ('ckpt', job.get('ckpt')), ('autocast', 'bf16'), ('tf32', True), ('sdpa', True),
            ('cuda_device_order', 'PCI_BUS_ID'))
    for key, w in want:
        if lenient and key in ('qopts', 'ckpt', 'opt_steps') and key not in m:
            continue
        if m.get(key) != w:
            problems.append(f'{jid}: meta {key} = {m.get(key)!r} (expected {w!r})')
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
            problems.append(f"{jid} ep {e}: E {r['E']}")
        if len(r['best']) != OPT[t]:
            problems.append(f"{jid} ep {e}: {len(r['best'])} per-iteration bests")
        if job.get('diag'):
            o = r.get('oei') or {}
            miss = [k for k in OEI_KEYS if not np.isfinite(o.get(k, np.nan))]
            if miss:
                problems.append(f'{jid} ep {e}: OEI keys missing / not finite {miss}')
    return {int(r['ep']): r for r in recs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root'); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    R = a.root
    problems, rep = [], dict(manifests={}, cells={}, selection={}, qat_log={}, e1_links={})
    recs = {}                                   # (man, cell, var, baseline) -> {ep: record}
    pops = {}                                   # (man, cell, var, baseline) -> {ep: npz path}
    for name in MANS:
        man = json.load(open(f'{R}/code/manifests/{name}.json'))
        jobs = man['jobs']
        ids = {j['id'] for j in jobs}
        rep['manifests'][name] = dict(n_jobs=len(jobs), cells=sorted({j['cell'] for j in jobs}))
        gpus, hooks, t0, t1, gs = defaultdict(set), defaultdict(set), defaultdict(list), defaultdict(list), defaultdict(float)
        for j in jobs:
            p = f"{R}/results/{name}/{j['task']}/{j['id']}.json"
            if not os.path.exists(p):
                problems.append(f"{name} {j['id']}: no result JSON"); continue
            d = json.load(open(p))
            if d['job'] != j:
                problems.append(f"{name} {j['id']}: job in JSON differs from the manifest job")
            want_split = 'held' if name == 'E5_dinowm_held' else SPLIT_EVAL[j['task']]
            if j['split'] != want_split:
                problems.append(f"{name} {j['id']}: split {j['split']} (expected {want_split})")
            rr = check_job(j['id'], j, d, problems)
            key = (name, j['cell'], j['var'], j.get('baseline'))
            for e, r in rr.items():
                if e in recs.setdefault(key, {}):
                    problems.append(f"{name} {j['id']}: episode {e} twice")
                recs[key][e] = r
                pops.setdefault(key, {})[e] = f"{R}/pops/{name}/{j['task']}/{j['id']}_e{e}.npz"
            gpus[j['cell']].add((d['meta']['gpu'], d['meta']['cuda_visible'], d['meta']['host']))
            hooks[(j['cell'], j['var'], j.get('baseline'))] |= {int(r['hook_calls']) for r in d['records']}
            t0[j['cell']].append(d['meta']['t_start']); t1[j['cell']].append(d['meta']['t_finalized'])
            gs[j['cell']] += float(d['meta']['gpu_s'])
        files = glob.glob(f'{R}/results/{name}/*/*')
        extra = sorted(os.path.basename(f) for f in files if not f.endswith('.json') or os.path.basename(f)[:-5] not in ids)
        if extra:
            problems.append(f'{name}: files outside the manifest {extra[:5]} ({len(extra)})')
        for c in sorted(gpus):
            row = dict(gpu=sorted(map(list, gpus[c])), t_start=min(t0[c]), t_end=max(t1[c]), gpu_h=gs[c] / 3600,
                       hook_calls={f'{v}|{b}': sorted(h) for (cc, v, b), h in hooks.items() if cc == c})
            if len(gpus[c]) != 1:
                problems.append(f'{name} {c}: jobs on more than one GPU {gpus[c]}')
            for (cc, v, b), h in hooks.items():
                if cc == c and len(h) != 1:
                    problems.append(f'{name} {c} {v} {b}: hook calls not constant {sorted(h)}')
            n_ep = {f'{v}|{b}': len(recs[(nm, cc, v, b)]) for (nm, cc, v, b) in recs if nm == name and cc == c}
            row['episodes'] = n_ep
            want = 20 if name == 'E5_dinowm_held' else 100
            for kk, n in n_ep.items():
                if n != want:
                    problems.append(f'{name} {c} {kk}: {n} episodes (expected {want})')
            rep['cells'][f'{name}|{c}'] = row
    # selection recomputed from the held results
    order = [t for t, _, _ in E5B]
    sel = json.load(open(f'{R}/results/E5_dinowm_held/selection.json'))
    held_man = json.load(open(f'{R}/code/manifests/E5_dinowm_held.json'))
    ev_man = json.load(open(f'{R}/code/manifests/E5_dinowm_eval.json'))
    for task, ks in held_man['config']['cells'].items():
        for k in ks:
            cell = f'{task}_k{k}'
            rows = []
            for tag in order:
                rr = recs.get(('E5_dinowm_held', f'{cell}_held', 'det', tag), {})
                rows.append(dict(tag=tag, s=int(sum(bool(r['success']) for r in rr.values())), n=len(rr)))
            best = max(rows, key=lambda r: (r['s'], -order.index(r['tag'])))
            top = max(r['s'] for r in rows)
            s = sel.get(cell)
            ok = bool(s and s['chosen'] == best['tag'] and
                      [(r['tag'], r['s'], r['n']) for r in s['held']] == [(r['tag'], r['s'], r['n']) for r in rows])
            if not ok:
                problems.append(f'selection {cell}: recomputed {best["tag"]} {rows} != selection.json {s}')
            # the eval jobs of the cell use B's ranges and qopts
            if best['tag'] != 'main':
                _, rid, qo = E5B[order.index(best['tag'])]
                ej = [j for j in ev_man['jobs'] if j['task'] == task and int(j['k']) == k]
                bad = [j['id'] for j in ej if j.get('baseline') != best['tag'] or j['ranges'] != rid.format(k=k)
                       or (j.get('qopts') or None) != (dict(qo) if qo else None)]
                if not ej or bad:
                    problems.append(f'eval jobs of {cell}: {len(ej)} jobs, mismatching baseline / ranges / qopts {bad[:3]}')
            rep['selection'][cell] = dict(recomputed=best['tag'], selection_json=s['chosen'] if s else None, held=rows,
                                          tied_at_max=[r['tag'] for r in rows if r['s'] == top], matches=ok)
    # QAT training log
    lp = f'{R}/qat/pusht_qat_k4/log.json'
    if os.path.exists(lp):
        L = json.load(open(lp))
        ls = L.get('losses') or []
        rep['qat_log'] = dict(final=L.get('final'), step=L.get('step'), minutes=L.get('minutes'), max_minutes=L.get('max_minutes'),
                              max_steps=L.get('max_steps'), lr=L.get('lr'), bs=L.get('bs'), ranges=L.get('ranges'), k=L.get('k'),
                              gpu=L.get('gpu'), gpu_name=L.get('gpu_name'), n_loss_entries=len(ls),
                              first=ls[0] if ls else None, last=ls[-1] if ls else None,
                              all_finite=bool(all(np.isfinite(float(x.get('loss', np.nan))) for x in ls)) if ls else None)
        if not L.get('final'):
            problems.append('QAT log not final')
    else:
        problems.append('QAT log missing')
    # E1 links: none of the task (D reference), E1 det / sh-iter (B = main), iteration-0 populations
    e1none = {}
    for t in ('pusht', 'pointmaze'):
        rr = {}
        for f in glob.glob(f'{R}/results/{E1MAN[t]}/{t}/*.json'):
            d = json.load(open(f))
            j = d['job']
            if j['split'] == SPLIT_EVAL[t] and j['var'] == 'none':
                for r in d['records']:
                    rr[int(r['ep'])] = (r, f"{R}/pops/{E1MAN[t]}/{t}/{j['id']}_e{int(r['ep'])}.npz")
        e1none[t] = rr
        if sorted(rr) != list(range(100)):
            problems.append(f'E1 none {t}: {len(rr)} episodes')
    for key, rr in recs.items():
        name, cell, var, b = key
        if name == 'E5_dinowm_held':
            continue
        t = cell.split('_')[0]
        n_seed, dmax, n_arr = 0, 0.0, 0
        for e, r in rr.items():
            if e not in e1none[t]:
                continue
            rn, f1 = e1none[t][e]
            n_seed += int(r['plan_seed']) == int(rn['plan_seed']) and int(r['E']) == int(rn['E'])
            f3 = pops[key][e]
            if os.path.exists(f3) and os.path.exists(f1):
                z3, z1 = np.load(f3), np.load(f1)
                dmax = max(dmax, float(np.abs(z3['it0_J_none'].astype(np.float64) - z1['it0_J_none']).max()))
                n_arr += 1
        rep['e1_links'][f'{name}|{cell}|{var}|{b}'] = dict(n=len(rr), same_plan_seed_and_E=n_seed, it0_arrays=n_arr,
                                                           it0_J_none_max_abs_diff_vs_E1_none=dmax)
        if n_seed != len(rr) or n_arr != len(rr):
            problems.append(f'{name} {cell} {var}: E1 none link {n_seed}/{n_arr} of {len(rr)}')
        if name != 'E5_dinowm_qat' and dmax != 0.0:
            problems.append(f'{name} {cell} {var}: iteration-0 unquantized costs differ from E1 none (max {dmax:.3g})')
    rep['problems'] = problems
    rep['n_problems'] = len(problems)
    print(json.dumps(rep, indent=1, default=str))
    if a.out:
        json.dump(rep, open(a.out, 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
