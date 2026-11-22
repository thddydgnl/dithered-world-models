"""Diagnostic for a failed check 0.7-3 (not a design cell): none on an E1 split on this pod, to separate a split
population difference from a pipeline / GPU difference. Writes out/diag/<task>/<split>_none_*.json and a summary with
the episode-level agreement with the E1 records (inputs/e1_<task>_<split>_none.json).
usage: python diag_e1split.py <task> <split> [ep0 ep1]"""
import os, sys, json, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wmlib as L
import qlib as QL
import runner as RN
import podplan as PP

task, split = sys.argv[1], sys.argv[2]
e0, e1 = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 100)
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_num_threads(4)
L.patch_attention(True)
out = f'/workspace/atm/out/diag/{task}'
os.makedirs(out, exist_ok=True)
ctx = RN.TaskCtx(task, True, 'diag')
recs = []
for a, b in PP.chunks(task, e0, e1):
    j = PP.job(task, split, 'none', 0, a, b, diag=False, exp=('diag',))
    p = RN.run_job(ctx, j, 'diag', None, out)
    p.finalize(None)
    recs += p.records
s = [bool(r['success']) for r in recs]
ref = json.load(open(f'{RN.CODE}/inputs/e1_{task}_{split}_none.json'))
refm = {int(k): v for k, v in ref['success'].items()}
same = [s_ == refm[r['ep']] for s_, r in zip(s, recs) if r['ep'] in refm]
summ = dict(task=task, split=split, eps=[e0, e1], gpu=torch.cuda.get_device_name(0), successes=int(sum(s)), n=len(s),
            e1_successes=int(sum(refm[r['ep']] for r in recs if r['ep'] in refm)), episode_agreement=float(np.mean(same)),
            time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
L.jdump(summ, f'{out}/summary_{split}.json')
print('DIAG', json.dumps(summ))
