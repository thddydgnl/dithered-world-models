"""Ground-truth costs of recorded CEM populations (simulator replay of all 300 candidates). The last iteration is replayed
for every variant; the first iteration only for 'none' (its population is identical in every variant: same planner
generator, mu = 0, sigma = 1). Incremental: skips files that already have a truth file.
usage: python replay13.py <task> <split+tag, e.g. eval or heldsmk> [workers] [--loop SECONDS]"""
import sys, os, glob, time
import numpy as np
import torch
from multiprocessing import get_context
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r21lib as L
import qlib13 as Q13


def once(task, split_tag, workers, T, pre, pairs):
    files = sorted(glob.glob(f'{Q13.POPS}/{task}_{split_tag}_*_e*.npz'))
    files = [f for f in files if not f.endswith('_truth.npz') and not os.path.exists(f.replace('.npz', '_truth.npz'))]
    if not files:
        return 0
    last = L.TASKS[task]['opt_steps'] - 1
    jobs, keys = [], []
    for fpath in files:
        e = int(fpath.split('_e')[-1].split('.')[0])
        vk = fpath.split('/')[-1].split('_')[2]
        try:
            z = np.load(fpath)
            its = [last] + ([0] if vk.startswith('none') else [])
            for i in its:
                ex = L.to_exec_actions(pre, torch.tensor(z[f'i{i}_actions']))
                jobs.append((task, pairs['env_info'][e], pairs['eval_seed'][e], pairs['state0'][e], ex, pairs['stateg'][e]))
                keys.append((fpath, i))
        except Exception as ex_:   # file still being written
            print('skip', fpath, ex_, flush=True)
    t0 = time.time()
    with get_context('spawn').Pool(workers) as pool:
        outs = pool.map(L._replay_job, jobs, chunksize=1)
    res = {}
    for (fpath, i), o in zip(keys, outs):
        d = res.setdefault(fpath, {})
        d[f'i{i}_margin'] = o[3].astype(np.float32); d[f'i{i}_sd'] = o[1].astype(np.float32); d[f'i{i}_succ'] = o[2]
    for fpath, d in res.items():
        np.savez_compressed(fpath.replace('.npz', '_truth.npz'), **d)
    print(f'{time.strftime("%H:%M:%S")} replayed {len(res)} files, {len(jobs)} populations in {time.time() - t0:.0f}s',
          flush=True)
    return len(res)


def main():
    task, split_tag = sys.argv[1], sys.argv[2]
    split = 'held' if split_tag.startswith('held') else 'eval'
    workers = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else 4
    loop = float(sys.argv[sys.argv.index('--loop') + 1]) if '--loop' in sys.argv else 0
    T = L.load_task(task, device='cpu')
    pre = T['pre']
    env = L.make_env(task)
    sp = Q13.SPLITS[split]
    pairs = L.sample_pairs(T, env, sp['n'], seed=sp['pair_seed'])
    while True:
        once(task, split_tag, workers, T, pre, pairs)
        if not loop:
            break
        if os.path.exists(f'/workspace/logs/STOP_REPLAY_{task}_{split_tag}'):
            once(task, split_tag, workers, T, pre, pairs)
            break
        time.sleep(loop)
    print(f'REPLAY13_DONE {task} {split_tag}', flush=True)


if __name__ == '__main__':
    main()
