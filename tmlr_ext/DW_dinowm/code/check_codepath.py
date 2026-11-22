"""DESIGN_tmlr.md 0.7-1: the untouched E1 qlib (code/e1ref/qlib.py, md5 f667047f...) and the extended qlib give
bit-identical plans on the same pod, model and inputs for none, det, sh-iter, sh-plan (shstep) and det+shift (also ind
and shep, the other E1 variants). Both run batched CEM with diagnostics (E1 run_cem arguments, E1 batch size E) on the
smoke split; compared per episode: final plan mean, best cost of every iteration, iteration-0 and last populations
(actions and costs), unquantized costs, planning-time error draws, J_v / J_none of the final plan. The new code runs
first and once more at the end (run-to-run determinism).
usage: python check_codepath.py <task> [--eps 0,1] [--k K] [--out JSON]"""
import os, sys, json, time, argparse, importlib.util
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wmlib as L
import qlib as QL
import runner as RN
import podplan as PP

VARS = ('none', 'det', 'shiter', 'shstep', 'detshift', 'ind', 'shep')


def load_e1ref():
    spec = importlib.util.spec_from_file_location('qlib_e1ref', f'{RN.CODE}/e1ref/qlib.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def flat(outs):
    """Comparable arrays of a list of run_cem episode dicts."""
    res = []
    for o in outs:
        d = dict(mu_final=o['mu_final'], best=np.array(o['best']))
        for i, p in o['pops'].items():
            for kk, v in p.items():
                d[f'pop{i}_{kk}'] = np.asarray(v)
        for kk in ('J_none_it0', 'J_v_mf', 'J_none_mf'):
            if kk in o:
                d[kk] = np.asarray(o[kk])
        if 'err_draws' in o:
            for s, v in o['err_draws'].items():
                d[f'err{s}'] = np.array(v)
            for s, v in o['pred_var'].items():
                d[f'pv{s}'] = np.array(v)
        res.append(d)
    return res


def same(a, b):
    bad = []
    for x, y in zip(a, b):
        if set(x) != set(y):
            bad.append(('keys', sorted(set(x) ^ set(y))))
            continue
        for kk in x:
            if not np.array_equal(x[kk], y[kk]):
                bad.append((kk, float(np.max(np.abs(np.asarray(x[kk], float) - np.asarray(y[kk], float))))))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task'); ap.add_argument('--eps', default='0,1'); ap.add_argument('--k', type=int, default=None)
    ap.add_argument('--split', default='smoke'); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    task, split = a.task, a.split
    k = a.k or PP.K1[task]
    eps = [int(x) for x in a.eps.split(',')]
    E = PP.BATCH_E[task]
    blocks = [eps[i:i + E] for i in range(0, len(eps), E)]
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(4)
    L.patch_attention(True)
    t0 = time.time()
    ctx = RN.TaskCtx(task, True, 'check')
    m, pre = ctx.m, ctx.pre
    pairs = ctx.get_pairs(split)
    rng = RN.load_ranges(task, 'main')
    salt = QL.SPLITS[split]['salt']

    def run(mod, Q, var):
        outs = []
        for ee in blocks:
            inputs = [mod.episode_inputs(m, pre, pairs, e) for e in ee]
            gens = [torch.Generator().manual_seed(mod.plan_seed(split, var, e)) for e in ee]
            outs += mod.run_cem(m, Q, inputs, gens, ctx.alpha, ctx.opt_steps, ee, var, k, salt, diag=True)
        return flat(outs)

    new, old, rep = {}, {}, {}
    Qn = ctx.Q
    Qn.set_ranges(rng)
    for v in VARS:
        new[v] = run(QL, Qn, v)
        print(f'new {v} done ({time.time() - t0:.0f}s)', flush=True)
    Qn.remove()
    E1 = load_e1ref()
    Qo = E1.Quantizer(m, use_compile=True)
    Qo.set_ranges(rng)
    for v in VARS:
        old[v] = run(E1, Qo, v)
        print(f'e1ref {v} done ({time.time() - t0:.0f}s)', flush=True)
    Qo.remove()
    Qn2 = QL.QuantizerT(m, use_compile=True)
    Qn2.set_ranges(rng)
    for v in ('none', 'det', 'shiter'):
        rep[v] = run(QL, Qn2, v)
    Qn2.remove()
    res = dict(task=task, split=split, eps=eps, E=E, k=k, gpu=torch.cuda.get_device_name(0), time=time.strftime(
        '%Y-%m-%dT%H:%M:%SZ', time.gmtime()), e1ref_md5='f667047f23fe0c457fa60c2f124c0ff7', variants={})
    for v in VARS:
        bad = same(new[v], old[v])
        res['variants'][v] = dict(identical=not bad, diffs=bad[:10],
                                  success_inputs_checked=len(new[v]), n_arrays=sum(len(x) for x in new[v]))
    res['rerun'] = {v: dict(identical=not same(new[v], rep[v]), diffs=same(new[v], rep[v])[:5]) for v in rep}
    res['passed'] = all(x['identical'] for x in res['variants'].values()) and all(x['identical'] for x in res['rerun'].values())
    res['secs'] = time.time() - t0
    out = a.out or f'/workspace/atm/out/checks/check1_codepath_{task}.json'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    L.jdump(res, out)
    print('CHECK1', task, 'PASSED' if res['passed'] else 'FAILED',
          {v: x['identical'] for v, x in res['variants'].items()}, 'rerun', {v: x['identical'] for v, x in res['rerun'].items()})


if __name__ == '__main__':
    main()
