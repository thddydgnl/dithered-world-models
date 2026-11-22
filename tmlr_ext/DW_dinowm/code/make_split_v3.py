"""PushT eval split of DESIGN_main v3 (split 'eval6'): builds the cached pairs with the constrained sampler
(qlib.SPLITS['eval6'], wmlib.sample_pairs_constrained) and writes the overlap bookkeeping
results/E0_dinowm/splits_pusht_v3.json. CPU only; no planning is run on the new pairs.
Overlap = dataset frames shared by two 26-frame segments of the same validation trajectory.
usage: CUDA_VISIBLE_DEVICES= <venv python> make_split_v3.py"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wmlib as L
import qlib as QL

SEG = 5 * L.GOAL_H + 1


def main():
    T = L.load_task('pusht', device='cpu')
    env = L.make_env('pusht')
    sp = QL.SPLITS['eval6']
    # reference splits (trajectory, offset); check the no-render replica against the caches of the splits run in E0
    refs = {}
    for name, seed, n, role in (('held', 3, 20, 'hard'), ('smoke', 7, 4, 'hard'), ('pilot_eval_R2-7_R2-13', 4, 50, 'soft'),
                                ('R2-1_cem_eval', 2, 50, 'info'), ('R2-1_sel', 1, 100, 'info'),
                                ('E1_eval_v2_unused', 5, 100, 'info')):
        segs = L.pair_segments(T, n, seed)
        refs[name] = dict(pair_seed=seed, n=n, role=role, segs=segs)
    for name, split in (('held', 'held'), ('smoke', 'smoke'), ('pilot_eval_R2-7_R2-13', 'repro'), ('E1_eval_v2_unused', 'eval')):
        fj = f'{L.MAIN}/pairs/pairs_pusht_{split}.json'
        if os.path.exists(fj):
            cached = [tuple(x) for x in json.load(open(fj))['traj']]
            assert cached == [tuple(x) for x in refs[name]['segs']], f'replica differs from the {split} cache'
    pairs = QL.get_pairs(T, env, 'eval6', f'{L.MAIN}/pairs')
    meta = json.load(open(f'{L.MAIN}/pairs/pairs_pusht_eval6.json'))
    new = [tuple(x) for x in pairs['traj']]
    rows = []
    for j, (bi, o) in enumerate(new):
        r = dict(pair=j, traj=bi, offset=o, frames=[o, o + SEG - 1], env_seed=pairs['eval_seed'][j],
                 planner_seed=sp['plan_seed'] + j, planner_seed_nonealt=sp['plan_seed'] + QL.ALT_SEED_OFFSET + j,
                 sampler=meta['sampler_stats'][j])
        for name, ref in refs.items():
            hits = [(i, L.shared_frames(o, ro, SEG)) for i, (rb, ro) in enumerate(ref['segs']) if rb == bi]
            hits = [(i, f) for i, f in hits if f > 0]
            r[f'overlap_{name}'] = dict(pairs=[i for i, _ in hits], shared_frames=[f for _, f in hits],
                                        total_shared=int(sum(f for _, f in hits)))
        within = [(i, L.shared_frames(o, oo, SEG)) for i, (bb, oo) in enumerate(new) if i != j and bb == bi]
        within = [(i, f) for i, f in within if f > 0]
        r['overlap_within_eval6'] = dict(pairs=[i for i, _ in within], total_shared=int(sum(f for _, f in within)))
        rows.append(r)
    totals = {}
    for name, ref in refs.items():
        k = f'overlap_{name}'
        totals[name] = dict(role=ref['role'], pair_seed=ref['pair_seed'], n_ref_pairs=ref['n'],
                            new_pairs_with_overlap=sum(1 for r in rows if r[k]['total_shared'] > 0),
                            total_shared_frames=int(sum(r[k]['total_shared'] for r in rows)),
                            max_shared_frames_one_pair=int(max(r[k]['total_shared'] for r in rows)),
                            identical_segments=sum(1 for r in rows if SEG in r[k]['shared_frames']))
    totals['within_eval6'] = dict(new_pairs_overlapping_another=sum(1 for r in rows if r['overlap_within_eval6']['pairs']),
                                  distinct_trajectories=len({bi for bi, _ in new}),
                                  pairs_per_trajectory={int(b): sum(1 for bi, _ in new if bi == b) for b in sorted({bi for bi, _ in new})})
    out = dict(task='pusht', split='eval6', design='DESIGN_main v3 (2026-10-21T22:54Z)', pair_seed=sp['pair_seed'], n=sp['n'],
               episode_index=[0, sp['n'] - 1], env_seed=[pairs['eval_seed'][0], pairs['eval_seed'][-1]],
               planner_seed=[sp['plan_seed'], sp['plan_seed'] + sp['n'] - 1],
               planner_seed_nonealt=[sp['plan_seed'] + QL.ALT_SEED_OFFSET, sp['plan_seed'] + QL.ALT_SEED_OFFSET + sp['n'] - 1],
               dither_salt=sp['salt'], segment_frames=SEG, n_val_trajectories=len(T['dset']),
               rule=('DINO-WM prepare_targets draws (uniform validation trajectory, uniform offset; rng = random.Random(6)); '
                     'reject any segment sharing a frame with the held pairs (seed 3, 0-19) or the E0 smoke pairs '
                     '(seed 7, 0-3); reject exact duplicates; keep drawing until a segment shares no frame with the pilot '
                     'eval pairs (seed 4, 0-49), else keep the one with the fewest shared frames after 5000 draws'),
               totals=totals, pairs=rows,
               reference_segments={name: dict(pair_seed=ref['pair_seed'], role=ref['role'], segs=ref['segs'])
                                   for name, ref in refs.items()})
    path = f'{L.MAIN}/results/E0_dinowm/splits_pusht_v3.json'
    L.jdump(out, path)
    print(json.dumps(totals, indent=1))
    print('sampler draws: mean %.1f max %d; hard rejections total %d; duplicates %d' % (
        np.mean([r['sampler']['draws'] for r in rows]), max(r['sampler']['draws'] for r in rows),
        sum(r['sampler']['hard_rejections'] for r in rows), sum(r['sampler']['duplicate_rejections'] for r in rows)))
    print('wrote', path)


if __name__ == '__main__':
    main()
