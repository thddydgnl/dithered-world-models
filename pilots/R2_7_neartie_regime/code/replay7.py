"""Ground-truth costs of every recorded CEM population (simulator replay, 300 candidates per population).
Incremental: skips episodes that already have a truth file. usage: python replay7.py <task> <split> [workers]"""
import sys, os, glob, time
import numpy as np
import torch
from multiprocessing import get_context
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r21lib as L
import r27lib as R7
from cem7 import SPLITS


def main():
    task, split = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    T = L.load_task(task)
    pre = T['pre']
    env = L.make_env(task)
    sp = SPLITS[split]
    pairs = L.sample_pairs(T, env, sp['n'], seed=sp['pair_seed'])
    files = sorted(glob.glob(f'{R7.POPS}/{task}_{split}_e*.npz'))
    files = [f for f in files if not f.endswith('_truth.npz') and not os.path.exists(f.replace('.npz', '_truth.npz'))]
    jobs, keys = [], []
    for fpath in files:
        e = int(fpath.split('_e')[-1].split('.')[0])
        z = np.load(fpath)
        steps = sorted({int(k.split('_')[0][1:]) for k in z.files})
        for s in steps:
            ex = L.to_exec_actions(pre, torch.tensor(z[f's{s}_actions']))
            jobs.append((task, pairs['env_info'][e], pairs['eval_seed'][e], pairs['state0'][e], ex, pairs['stateg'][e]))
            keys.append((fpath, s))
    t0 = time.time()
    with get_context('spawn').Pool(workers) as pool:
        outs = pool.map(L._replay_job, jobs, chunksize=1)
    res = {}
    for (fpath, s), o in zip(keys, outs):
        d = res.setdefault(fpath, {})
        d[f's{s}_margin'] = o[3].astype(np.float32); d[f's{s}_sd'] = o[1].astype(np.float32); d[f's{s}_succ'] = o[2]
    for fpath, d in res.items():
        np.savez_compressed(fpath.replace('.npz', '_truth.npz'), **d)
    print(f'replayed {len(files)} episodes, {len(jobs)} populations in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
