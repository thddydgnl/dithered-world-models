"""R2-13 CEM runner. For every block of E episodes, runs every requested variant (same episodes, same planner seeds)
and appends one JSON line per (variant, k, episode). Resumable: (variant, k, ep) already in the output file are skipped.
usage: python cem13.py <task> <split: held|eval> <specs e.g. none:0,det:4,ind:4> <ep0> <ep1> <E> [--tag T] [--norec]
       [--nocompile]
held split = pair seed 3 (20 pairs), eval split = pair seed 4 (50 pairs); planner seed 1000+e (held) / 2000+e (eval),
identical to R2-7."""
import sys, os, time, json, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r21lib as L
import qlib13 as Q13


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task'); ap.add_argument('split'); ap.add_argument('specs')
    ap.add_argument('ep0', type=int); ap.add_argument('ep1', type=int); ap.add_argument('E', type=int)
    ap.add_argument('--tag', default=''); ap.add_argument('--norec', action='store_true')
    ap.add_argument('--nocompile', action='store_true')
    a = ap.parse_args()
    task = a.task
    torch.backends.cuda.matmul.allow_tf32 = True
    L.patch_attention(True)
    T = L.load_task(task)
    m, pre = T['model'], T['pre']
    alpha = L.TASKS[task]['alpha']; opt_steps = L.TASKS[task]['opt_steps']
    env = L.make_env(task)
    sp = Q13.SPLITS[a.split]
    pairs = L.sample_pairs(T, env, sp['n'], seed=sp['pair_seed'])
    Q = Q13.Quantizer(m, use_compile=not a.nocompile)
    rng = json.load(open(f'{Q13.RES}/ranges_{task}.json'))
    Q.set_ranges([(r['lo'], r['hi']) for r in rng['layers']])
    specs = [(s.split(':')[0], int(s.split(':')[1])) for s in a.specs.split(',')]
    out_path = f'{Q13.RES}/cem13_{task}_{a.split}{a.tag}.jsonl'
    done = set()
    if os.path.exists(out_path):
        for ln in open(out_path):
            r = json.loads(ln); done.add((r['var'], r['k'], r['ep']))
    os.makedirs(Q13.POPS, exist_ok=True)
    t_start = time.time()
    eps_all = list(range(a.ep0, a.ep1))
    for g0 in range(0, len(eps_all), a.E):
        eps = eps_all[g0:g0 + a.E]
        inputs = [Q13.episode_inputs(m, pre, pairs, e) for e in eps]
        for var, k in specs:
            todo = [e for e in eps if (var, k, e) not in done]
            if not todo:
                continue
            if len(todo) != len(eps):   # partial block (resume): rerun the block's missing episodes only
                sub = [eps.index(e) for e in todo]
                ins, ee = [inputs[i] for i in sub], todo
            else:
                ins, ee = inputs, eps
            t0 = time.time()
            gens = [torch.Generator().manual_seed(sp['plan_seed'] + e) for e in ee]
            n0 = Q.n_calls
            outs = Q13.run_cem(m, Q, ins, gens, alpha, opt_steps, ee, var, k, a.split, rec=not a.norec)
            t_cem = time.time() - t0
            for kk, e in enumerate(ee):
                o = outs[kk]
                ex = L.to_exec_actions(pre, torch.tensor(o['mu_final'])[None])
                fin = L.replay_final_states(task, pairs['env_info'][e], pairs['eval_seed'][e], pairs['state0'][e], ex)
                sd, sc, mg = L.true_costs(task, pairs['stateg'][e], fin)
                rec = dict(ep=e, task=task, split=a.split, var=var, k=k, success=bool(sc[0]), margin=float(mg[0]),
                           state_dist=float(sd[0]), best=o['best'], mu_final=o['mu_final'].round(5).tolist(),
                           time_s=t_cem / len(ee), E=len(ee), hook_calls=Q.n_calls - n0)
                with open(out_path, 'a') as f:
                    f.write(json.dumps(rec) + '\n')
                done.add((var, k, e))
                if o['pops']:   # write to a temporary name first: the replay process polls this directory
                    tmp = f'{Q13.POPS}/tmp_{os.getpid()}.npz'
                    np.savez_compressed(tmp, **{f'i{i}_{kk2}': v for i, d in o['pops'].items() for kk2, v in d.items()})
                    os.replace(tmp, f'{Q13.POPS}/{task}_{a.split}{a.tag}_{var}{k}_e{e}.npz')
                print(f"{task} {a.split} {var}:{k} ep {e} success {rec['success']} margin {rec['margin']:.3f} "
                      f"({rec['time_s']:.1f}s/ep, total {time.time() - t_start:.0f}s)", flush=True)
    print(f'CEM13_DONE {task} {a.split} {a.specs} {a.ep0}-{a.ep1} in {time.time() - t_start:.0f}s', flush=True)


if __name__ == '__main__':
    main()
