"""PointMaze functional check on CPU (no GPU): env creation, OSMesa rendering of start/goal frames, start/goal pair
sampling (L.sample_pairs, random_state), fast replay (no rendering) == env.rollout final states, evaluator,
dataset + preprocessor loading. usage: bash run_pm.sh test_pointmaze.py"""
import os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wmlib as L


def main():
    t0 = time.time()
    env = L.make_env('pointmaze')
    print('env', type(env).__name__, f'{time.time() - t0:.1f}s', flush=True)
    s0, sg = env.sample_random_init_goal_states(7001)
    t1 = time.time()
    o0, st0 = env.prepare(7001, s0)
    print('prepare: visual', o0['visual'].shape, o0['visual'].dtype, 'mean', float(np.mean(o0['visual'])),
          'proprio', o0['proprio'], f'{time.time() - t1:.2f}s', flush=True)
    T = L.load_task('pointmaze', device='cpu')
    pre = T['pre']
    print('dataset: valid', len(T['dset']), 'train', len(T['dset_train']), 'action mean', pre.action_mean, flush=True)
    rng = np.random.RandomState(0)
    acts = rng.randn(4, 25, 2).astype(np.float32) * 0.8
    fast = L.replay_final_states('pointmaze', {}, 7001, s0, acts, fast=True)
    slow = L.replay_final_states('pointmaze', {}, 7001, s0, acts, fast=False)
    print('fast replay == env.rollout:', bool(np.allclose(fast, slow, atol=0, rtol=0)), 'max diff', float(np.abs(fast - slow).max()))
    sd, sc, mg = L.true_costs('pointmaze', sg, fast)
    print('eval: state_dist', sd, 'success', sc, 'margin', mg)
    t2 = time.time()
    pairs = L.sample_pairs(T, env, 3, seed=7)
    print('pairs', {k: (v.shape if hasattr(v, 'shape') else v) for k, v in pairs.items()}, f'{time.time() - t2:.1f}s')
    print('TEST_PM_DONE', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
