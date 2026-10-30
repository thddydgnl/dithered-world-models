"""Start/goal split bookkeeping (CPU only): builds (or loads) the cached pairs of every split and writes the index
ranges, environment seeds, planner seeds and, for PushT (pairs cut from the 21 validation trajectories), the
trajectory segments and their overlaps across splits. Wall/PointMaze pairs are random states (env seed per pair).
usage: CUDA_VISIBLE_DEVICES= bash run.sh pairs_info.py pusht wall   |   bash run_pm.sh pairs_info.py pointmaze"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wmlib as L
import qlib as QL

SEG = 5 * L.GOAL_H + 1    # PushT pair = 26 consecutive dataset frames (frameskip 5 x goal_H 5)


def main():
    out_dir = f'{L.MAIN}/results/E0_dinowm'
    os.makedirs(out_dir, exist_ok=True)
    for task in sys.argv[1:]:
        T = L.load_task(task, device='cpu')
        env = L.make_env(task)
        info = dict(task=task, splits={})
        segs = {}
        for split in ('held', 'repro', 'eval', 'smoke'):
            sp = QL.SPLITS[split]
            pairs = QL.get_pairs(T, env, split, f'{L.MAIN}/pairs')
            n = sp['n']
            info['splits'][split] = dict(pair_seed=sp['pair_seed'], n=n, episode_index=[0, n - 1],
                                         env_seed=[pairs['eval_seed'][0], pairs['eval_seed'][-1]],
                                         planner_seed=[sp['plan_seed'], sp['plan_seed'] + n - 1],
                                         planner_seed_nonealt=[sp['plan_seed'] + QL.ALT_SEED_OFFSET,
                                                               sp['plan_seed'] + QL.ALT_SEED_OFFSET + n - 1],
                                         dither_salt=sp['salt'], traj=pairs['traj'],
                                         state0_first=np.asarray(pairs['state0'][0]).round(4).tolist())
            segs[split] = [tuple(x) for x in pairs['traj']]
        if task == 'pusht':
            def overlaps(a, b):
                hits = []
                for i, (ta, oa) in enumerate(segs[a]):
                    for j, (tb, ob) in enumerate(segs[b]):
                        if ta == tb and abs(oa - ob) < SEG:
                            hits.append((i, j, int(ta), int(oa), int(ob)))
                return hits
            ov = {}
            for a, b in (('held', 'eval'), ('repro', 'eval'), ('held', 'repro'), ('smoke', 'eval')):
                h = overlaps(a, b)
                ov[f'{a}_vs_{b}'] = dict(n_overlapping_pairs=len(h), n_eval_like_affected=len({x[1] for x in h}),
                                         identical_start=sum(1 for x in h if x[3] == x[4]), pairs=h[:50])
            info['segment_overlap'] = ov
            info['n_val_trajectories'] = len(T['dset'])
            info['note'] = ('PushT pairs are 26-frame segments of the 21 validation trajectories (DINO-WM prepare_targets, '
                            'goal_source dset); two pairs overlap if they share a trajectory and their offsets differ by < 26')
            for k, v in ov.items():
                print(task, k, 'overlapping pairs', v['n_overlapping_pairs'], 'distinct second-split pairs affected',
                      v['n_eval_like_affected'], 'identical start frame', v['identical_start'])
        L.jdump(info, f'{out_dir}/splits_{task}.json')
        print(task, {s: (d['episode_index'], d['env_seed'], d['planner_seed']) for s, d in info['splits'].items()}, flush=True)


if __name__ == '__main__':
    main()
