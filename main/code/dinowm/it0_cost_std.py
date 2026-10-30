"""Per-task normalizer for the auxiliary H2b (DESIGN_main v4): the none variant's iteration-0 candidate cost std
(std over the 299 random iteration-0 candidates of the unquantized model cost J, averaged over the task's E1
episodes; row 0 = the mu = 0 candidate is left out, as in the OEI reference it0_rand_*), read from the server
population arrays. usage: ~/pmx/envs/a-dinowm/bin/python it0_cost_std.py <out.json>  (server; numpy)"""
import os, sys, glob, json
import numpy as np

MAIN = os.path.expanduser('~/pmx/A/dinowm_main')
SRC = {'pusht': ('E1_dinowm_pusht', 'eval6'), 'wall': ('E1_dinowm_wp', 'eval'), 'pointmaze': ('E1_dinowm_wp', 'eval')}
out = {}
for task, (name, split) in SRC.items():
    files = sorted(glob.glob(f'{MAIN}/pops/{name}/{task}/{split}_none_e*_e*.npz'))
    files = [f for f in files if not f.endswith('_truth.npz')]
    stds = [float(np.load(f)['it0_J'][1:].std()) for f in files]
    if stds:
        out[task] = float(np.mean(stds))
        out[f'{task}_n_episodes'] = len(stds)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
print(out)
