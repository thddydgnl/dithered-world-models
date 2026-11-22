"""Build job manifests for runner.py (DINO-WM part of paper A).
Jobs are listed task-major, then chunk-major (all variants of an episode chunk before the next chunk), so a stopped
run leaves every variant with the same episodes. One job = (task, split, variant, k, episode chunk).

usage:
  python make_manifest.py repro  [--out F]      R2-13 reproduction: pilot eval split (repro), pilot k and batch E,
                                                pilot ranges, the 5 pilot variants (none det ind shiter shfixed)
  python make_manifest.py ksel   --E pusht=2,wall=4 [--ranges main]
                                                k selection on the held split: none + det at k = 3, 4, 5, 6
  python make_manifest.py e1     --k pusht=3,4,5 --k wall=2,3,4 --E pusht=2,wall=4 [--chunk pusht=10,wall=20]
                                                E1: eval split, none + nonealt + 6 quantized variants x 3 k
  python make_manifest.py smoke  --E pusht=2,wall=4
                                                tiny jobs on the 'smoke' split (never touches eval)
  generic: python make_manifest.py custom --name N --split S --vars v1,v2 --k task=k1,k2 --E task=E --ep0 0 --ep1 20
           [--planner num_samples=64,topk=8,opt_steps=10 --ptag N64]     (planner sweeps, E2)
  python make_manifest.py e2n | e2eo | e3                 DESIGN_main v5 (section 7.1, 7.2) DINO-WM manifests:
        E2_dinowm_N (N sweep), E2_dinowm_EO (elite ratio, opt_steps), E3_dinowm (indplan, table-size sweep).
        Every job carries a 'cell' (one GPU per cell, v5_chain.py); cells are listed longest first.
  python make_manifest.py e5w | e5qat                     v5 section 7.3 E5 parts that do not depend on E1 eligibility:
        E5_dinowm_w (weight quantization, PushT, activations fp), E5_dinowm_qat (QAT model qat_k4: none, det, sh-iter)
  python make_manifest.py e5held --k pusht=3,4 [--k pointmaze=3]   E5 baseline selection on the held split (det,
        20 ep) for the E1-eligible 3-4-bit confirmatory cells; e5_select.py then writes E5_dinowm_eval
  python make_manifest.py v5smoke                          smoke split checks before E3/E5 (Wall E=2, PointMaze)
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

QV6 = ['det', 'ind', 'shiter', 'shstep', 'shep', 'detshift']


def parse_map(items, cast=int, multi=False):
    """['pusht=3,4,5', 'wall=2'] or ['pusht=2,wall=4'] -> dict."""
    out = {}
    for it in items or []:
        if multi:
            t, v = it.split('=')
            out[t] = [cast(x) for x in v.split(',')]
        else:
            for kv in it.split(','):
                t, v = kv.split('=')
                out[t] = cast(v)
    return out


def job_id(split, var, k, e0, e1):
    tag = var if var in ('none', 'nonealt') else f'{var}_k{k}'
    return f'{split}_{tag}_e{e0:03d}-{e1:03d}'


def build(name, split, tasks, ep0, ep1, ranges, diag=True, with_none=True, with_nonealt=False, config=None):
    """tasks: {task: dict(E, ks, vars, chunk)}."""
    jobs = []
    for task, c in tasks.items():
        E, chunk = c['E'], c.get('chunk', c['E'])
        t0, t1 = c.get('ep0', ep0), c.get('ep1', ep1)
        assert chunk % E == 0 or (t1 - t0) <= chunk, (task, chunk, E)
        for e0 in range(t0, t1, chunk):
            e1 = min(e0 + chunk, t1)
            specs = ([('none', 0)] if with_none else []) + ([('nonealt', 0)] if with_nonealt else [])
            specs += [(v, k) for k in c['ks'] for v in c['vars']]
            specs = list(dict.fromkeys((v, 0 if v in ('none', 'nonealt') else k) for v, k in specs))   # none is k-free
            for var, k in specs:
                jobs.append(dict(id=job_id(split, var, k, e0, e1), task=task, split=split, var=var, k=k,
                                 ep0=e0, ep1=e1, E=E, ranges=ranges, diag=diag))
    ids = [j['task'] + '/' + j['id'] for j in jobs]
    assert len(ids) == len(set(ids))
    return dict(name=name, config=dict(split=split, ep0=ep0, ep1=ep1, ranges=ranges, diag=diag, tasks=tasks,
                                       **(config or {})), jobs=jobs)


# DESIGN_main v5 section 7.1 + v5.1 section 8.2: PushT k=4, none/det/sh-iter, E1 eval6 pairs and planner seeds,
# 100 episodes per point. Confirmatory N sweep (H2c) with the elite COUNT fixed at top-30 (v5.1; E1's N=300 is the
# sixth point); the ratio-fixed sweep of v5 is not run. N=1024 in row chunks of 300 (bitwise-identical plans,
# E2E3 smoke). Elite ratio {2, 12.5, 25}% at N=300 and opt_steps {5, 10} (E1: 10%, 30).
E2N = [('N1024t30', dict(num_samples=1024, topk=30, max_rows=300)), ('N512t30', dict(num_samples=512, topk=30)),
       ('N256t30', dict(num_samples=256, topk=30)), ('N128t30', dict(num_samples=128, topk=30)),
       ('N64t30', dict(num_samples=64, topk=30))]
E2EO = [('top6', dict(topk=6)), ('top38', dict(topk=38)), ('top75', dict(topk=75)), ('it10', dict(opt_steps=10)),
        ('it5', dict(opt_steps=5))]
# DESIGN_main v5 section 7.2: indplan at the primary k (PushT 4, Wall 5, PointMaze 3) plus PushT k=3 and Wall k=3;
# table-size sweep L1 shunit, L2 shgroup, L3 shlayer at the primary k plus Wall k=3 (L0 = E1 shstep, L+ = indplan).
# The E1 cells (none, ind, sh-iter, sh-plan) of the same pairs and seeds are the paired references.
E3 = [('pusht', 4, ['indplan', 'shunit', 'shgroup', 'shlayer'], 'primary'),
      ('pointmaze', 3, ['indplan', 'shunit', 'shgroup', 'shlayer'], 'primary'),
      ('pusht', 3, ['indplan'], 'extra'),
      ('wall', 5, ['indplan', 'shunit', 'shgroup', 'shlayer'], 'primary'),
      ('wall', 3, ['indplan', 'shunit', 'shgroup', 'shlayer'], 'exploratory')]
# DESIGN_main v5 section 7.3 (PushT k=4, 100 ep, activations fp): W8 / W4 RTN, AWQ-style W4, W4 SR with K = 1, 2, 4
E5W = [('w4rtn', 4), ('w4awq', 4), ('w4sr1', 4), ('w4sr2', 4), ('w4sr4', 4), ('w8rtn', 8)]
# DESIGN_main v5 section 7.3 activation-PTQ baselines B, in list order (ties in held det success go to the earlier
# one): per-channel (256 windows), percentile 99.9 / 99.99 (= E1 'main' grid, 64 windows) / 99.999, MSE-optimal,
# SmoothQuant alpha 0.5, last block 8-bit (others k-bit on the E1 grid)
E5B = [('pc', 'pc', None), ('p999', 'p999', None), ('main', 'main', None), ('p99999', 'p99999', None),
       ('mse', 'mse_k{k}', None), ('smooth', 'smooth', None), ('last8', 'main', {'last_bits': 8})]
V5E = dict(pusht=1, wall=2, pointmaze=1)                 # batch E as in E1 (DESIGN_main v2 section 4.3)
V5SPLIT = dict(pusht='eval6', wall='eval', pointmaze='eval')
V5CHUNK = dict(pusht=10, wall=20, pointmaze=10)


def v5_manifest(preset, kmap=None):
    jobs = []
    if preset in ('e2n', 'e2eo'):
        name = 'E2_dinowm_N' if preset == 'e2n' else 'E2_dinowm_EO'
        for ptag, pl in (E2N if preset == 'e2n' else E2EO):
            m = build(name, 'eval6', {'pusht': dict(E=1, ks=[4], vars=['none', 'det', 'shiter'], chunk=10)}, 0, 100, 'main',
                      with_none=False)
            for j in m['jobs']:
                j.update(id=f"{j['id']}_{ptag}", planner=dict(pl), cell=f'pusht_k4_{ptag}', role=f'E2 {ptag}')
            jobs += m['jobs']
        cfg = dict(design='DESIGN_main v5 section 7.1, v5.1 section 8.2 (N sweep: top-30 fixed)', task='pusht', k=4,
                   split='eval6', episodes=[0, 100],
                   settings={t: p for t, p in (E2N if preset == 'e2n' else E2EO)},
                   reference='E1_dinowm_pusht (N=300, topk 30, opt_steps 30) is the E1 point of every sweep')
    elif preset == 'e3':
        name = 'E3_dinowm'
        for task, k, vs, role in E3:
            m = build(name, V5SPLIT[task], {task: dict(E=V5E[task], ks=[k], vars=vs, chunk=V5CHUNK[task])}, 0, 100, 'main',
                      with_none=False)
            for j in m['jobs']:
                j.update(cell=f'{task}_k{k}', role=f'E3 {role}')
            jobs += m['jobs']
        cfg = dict(design='DESIGN_main v5 section 7.2', cells=[dict(task=t, k=k, vars=v, role=r) for t, k, v, r in E3],
                   reference='E1 cells of the same task, k, pairs and planner seeds (none, ind, shiter, shstep)')
    elif preset == 'e5w':
        name = 'E5_dinowm_w'
        for var, k in E5W:
            m = build(name, 'eval6', {'pusht': dict(E=1, ks=[k], vars=[var], chunk=10)}, 0, 100, 'main', with_none=False)
            for j in m['jobs']:
                j.update(cell=f'pusht_{var}', role='E5 weights (activations fp)')
            jobs += m['jobs']
        cfg = dict(design='DESIGN_main v5 section 7.3', task='pusht', split='eval6', episodes=[0, 100],
                   conditions=[f'{v} (k={k})' for v, k in E5W],
                   note='weights of every predictor Linear quantized per output channel, activations fp; reference: '
                        'E1_dinowm_pusht none (same pairs and planner seeds); SR copy i mod K per CEM iteration')
    elif preset == 'e5qat':
        name = 'E5_dinowm_qat'
        m = build(name, 'eval6', {'pusht': dict(E=1, ks=[4], vars=['none', 'det', 'shiter'], chunk=10)}, 0, 100, 'main',
                  with_none=False)
        for j in m['jobs']:
            j.update(id=f"{j['id']}_qat_k4", ckpt='qat_k4', cell='pusht_k4_qat', role='E5 QAT')
        jobs += m['jobs']
        cfg = dict(design='DESIGN_main v5 section 7.3', task='pusht', k=4, split='eval6', episodes=[0, 100], ckpt='qat_k4',
                   note='QAT predictor (qat.py pusht --k 4 --name qat_k4, about 2 GPU-h); damage against E1 none')
    elif preset == 'e5held':
        name = 'E5_dinowm_held'
        for task, ks in kmap.items():
            for k in ks:
                for tag, rid, qo in E5B:
                    m = build(name, 'held', {task: dict(E=V5E[task], ks=[k], vars=['det'], chunk=V5CHUNK[task])}, 0, 20,
                              rid.format(k=k), with_none=False)
                    for j in m['jobs']:
                        j.update(id=f"{j['id']}_{tag}", cell=f'{task}_k{k}_held', role=f'E5 held selection {tag}', baseline=tag)
                        if qo:
                            j['qopts'] = dict(qo)
                    jobs += m['jobs']
        cfg = dict(design='DESIGN_main v5 section 7.3', split='held', episodes=[0, 20], cells=kmap,
                   baselines=[dict(tag=t, ranges=r, qopts=q) for t, r, q in E5B],
                   rule='per (task, k): highest held det success; ties -> earlier in the list (e5_select.py)')
    elif preset == 'v5smoke':
        name = 'v5_smoke2'
        for task, k in (('wall', 5), ('pointmaze', 3)):
            m = build(name, 'smoke', {task: dict(E=V5E[task], ks=[k], vars=['shstep', 'indplan', 'shunit', 'shgroup', 'shlayer'],
                                                 chunk=V5E[task])}, 0, V5E[task], 'main', with_none=False)
            for j in m['jobs']:
                j.update(cell=f'{task}_smoke', role='smoke')
            jobs += m['jobs']
        m = build(name, 'smoke', {'wall': dict(E=1, ks=[5], vars=['indplan', 'shunit'], chunk=1)}, 0, 2, 'main', with_none=False)
        for j in m['jobs']:          # E-invariance check of the per-episode tables (compare with the E=2 jobs above)
            j.update(id=f"{j['id']}_E1", cell='wall_smoke', role='smoke')
        jobs += m['jobs']
        m = build(name, 'smoke', {'pusht': dict(E=1, ks=[4], vars=['none', 'det'], chunk=1)}, 0, 1, 'main', with_none=False)
        for j in m['jobs']:          # v5.1 section 8.2: N=64 with top-30 (30 of 64 elites) must run
            j.update(id=f"{j['id']}_N64t30", planner=dict(num_samples=64, topk=30), cell='pusht_smoke', role='smoke')
        jobs += m['jobs']
        m = build(name, 'smoke', {'pusht': dict(E=1, ks=[4], vars=['shiter'], chunk=1)}, 0, 1, 'smooth', with_none=False)
        for j in m['jobs']:
            j.update(id=f"{j['id']}_smooth", cell='pusht_smoke', role='smoke')
        jobs += m['jobs']
        m = build(name, 'smoke', {'pusht': dict(E=1, ks=[4], vars=['shiter'], chunk=1)}, 0, 1, 'main', with_none=False)
        for j in m['jobs']:
            j.update(id=f"{j['id']}_last8", qopts={'last_bits': 8}, cell='pusht_smoke', role='smoke')
        jobs += m['jobs']
        cfg = dict(note='smoke split only: E3 variants with Wall E=2 (and E=1) and PointMaze, PushT N=64 top-30, '
                        'PushT B+sh-iter for smooth and last8')
    ids = [j['task'] + '/' + j['id'] for j in jobs]
    assert len(ids) == len(set(ids))
    return dict(name=name, config=cfg, jobs=jobs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('preset'); ap.add_argument('--out', default=None); ap.add_argument('--name', default=None)
    ap.add_argument('--E', action='append'); ap.add_argument('--k', action='append'); ap.add_argument('--chunk', action='append')
    ap.add_argument('--ranges', default='main'); ap.add_argument('--split', default=None); ap.add_argument('--vars', default=None)
    ap.add_argument('--ep0', type=int, default=0); ap.add_argument('--ep1', type=int, default=None)
    ap.add_argument('--tasks', default='pusht,wall'); ap.add_argument('--nodiag', action='store_true')
    ap.add_argument('--planner', default=None); ap.add_argument('--ptag', default=None)
    a = ap.parse_args()
    Emap = parse_map(a.E); kmap = parse_map(a.k, multi=True); cmap = parse_map(a.chunk)
    tasks = a.tasks.split(',')
    if a.preset == 'repro':
        # R2-13 main run: PushT k=4 E=2, Wall k=3 E=4, eval pairs 0-49 of pair seed 4, planner seed 2000+e,
        # chunk = one pilot block of E episodes (same block composition as the pilot), pilot ranges
        T = {'pusht': dict(E=2, ks=[4], vars=['det', 'ind', 'shiter', 'shfixed'], chunk=2),
             'wall': dict(E=4, ks=[3], vars=['det', 'ind', 'shiter', 'shfixed'], chunk=4)}
        man = build(a.name or 'E0_repro', 'repro', {t: T[t] for t in tasks}, 0, a.ep1 or 50, 'pilot', diag=not a.nodiag)
    elif a.preset == 'ksel':
        T = {t: dict(E=Emap[t], ks=[3, 4, 5, 6], vars=['det'], chunk=cmap.get(t, 10)) for t in tasks}
        man = build(a.name or 'E0_ksel', 'held', T, 0, a.ep1 or 20, a.ranges, diag=not a.nodiag,
                    config=dict(rule='largest k in {3,4,5,6} with det success drop >= 20pp on the 20 held episodes'))
    elif a.preset == 'e1':
        # --split / --vars override the defaults (E1 v3 PushT: --split eval6 --vars det,ind,shiter,shstep,detshift)
        V = a.vars.split(',') if a.vars else QV6
        T = {t: dict(E=Emap[t], ks=kmap[t], vars=V, chunk=cmap.get(t, 10)) for t in tasks}
        man = build(a.name or 'E1_dinowm', a.split or 'eval', T, 0, a.ep1 or 100, a.ranges, diag=not a.nodiag,
                    with_nonealt=True)
    elif a.preset == 'smoke':
        T = {t: dict(E=Emap[t], ks=[kmap.get(t, [4])[0]], vars=QV6, chunk=Emap[t], ep0=0, ep1=Emap[t]) for t in tasks}
        man = build(a.name or 'E0_smoke', 'smoke', T, 0, 4, a.ranges, diag=True, with_nonealt=True)
    elif a.preset == 'custom':
        T = {t: dict(E=Emap[t], ks=kmap.get(t, [0]), vars=a.vars.split(','), chunk=cmap.get(t, Emap[t])) for t in tasks}
        man = build(a.name, a.split, T, a.ep0, a.ep1, a.ranges, diag=not a.nodiag, with_none=False)
        if a.planner:      # planner sweep (E2): e.g. --planner num_samples=64,topk=6,opt_steps=10 --ptag N64
            pl = {kv.split('=')[0]: int(kv.split('=')[1]) for kv in a.planner.split(',')}
            assert a.ptag, '--ptag is required with --planner (keeps job ids unique)'
            for j in man['jobs']:
                j['planner'] = pl
                j['id'] = f"{j['id']}_{a.ptag}"
            man['config']['planner'] = pl
    elif a.preset in ('e2n', 'e2eo', 'e3', 'e5w', 'e5qat', 'e5held', 'v5smoke'):
        man = v5_manifest(a.preset, kmap)
    else:
        raise SystemExit(f'unknown preset {a.preset}')
    out = a.out or f"{os.path.dirname(os.path.abspath(__file__))}/manifests/{man['name']}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(man, open(out, 'w'), indent=1)
    by_task = {}
    for j in man['jobs']:
        by_task[j['task']] = by_task.get(j['task'], 0) + (j['ep1'] - j['ep0'])
    print(f"{out}: {len(man['jobs'])} jobs, episodes per task {by_task}")


if __name__ == '__main__':
    main()
