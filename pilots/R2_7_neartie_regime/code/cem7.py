"""R2-7 CEM runner (batched; DINO-WM CEMPlanner semantics, bf16 world model, open-loop = MPC iteration 1).
usage: python cem7.py <task> <split: held|eval> <sched: base|fv|ak> <ep0> <ep1> <E> [--tau X] [--record] [--tag T]
held split = pair seed 3 (20 pairs), eval split = pair seed 4 (50 pairs); planner seed 1000+e (held) / 2000+e (eval)."""
import sys, os, time, json, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r21lib as L
import r27lib as R7

SPLITS = {'held': dict(pair_seed=3, n=20, plan_seed=1000), 'eval': dict(pair_seed=4, n=50, plan_seed=2000)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task'); ap.add_argument('split'); ap.add_argument('sched')
    ap.add_argument('ep0', type=int); ap.add_argument('ep1', type=int); ap.add_argument('E', type=int)
    ap.add_argument('--tau', type=float, default=float('inf')); ap.add_argument('--record', action='store_true')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    task = a.task
    torch.backends.cuda.matmul.allow_tf32 = True
    L.patch_attention(True)
    T = L.load_task(task)
    m, pre = T['model'], T['pre']
    alpha = L.TASKS[task]['alpha']; opt_steps = L.TASKS[task]['opt_steps']
    env = L.make_env(task)
    sp = SPLITS[a.split]
    pairs = L.sample_pairs(T, env, sp['n'], seed=sp['pair_seed'])
    Cb = R7.boot_counts().cuda()
    sched = dict(kind=a.sched, tau=a.tau, topk_after=90)
    name = f"{task}_{a.split}_{a.sched}{a.tag}"
    out_path = f'{R7.RES}/cem7_{name}.jsonl'
    os.makedirs(R7.POPS, exist_ok=True)
    rec_steps = R7.STEPS_REC[task] if a.record else ()
    t_start = time.time()
    eps_all = list(range(a.ep0, a.ep1))
    for g0 in range(0, len(eps_all), a.E):
        eps = eps_all[g0:g0 + a.E]
        t0 = time.time()
        inputs = [R7.episode_inputs(m, pre, pairs, e) for e in eps]
        gens = [torch.Generator().manual_seed(sp['plan_seed'] + e) for e in eps]
        outs = R7.run_cem_batch(m, inputs, gens, task, sched, alpha, opt_steps, rec_steps=rec_steps, Cb=Cb)
        for k, e in enumerate(eps):
            o = outs[k]
            ex = L.to_exec_actions(pre, torch.tensor(o['mu_final'])[None])
            fin = L.replay_final_states(task, pairs['env_info'][e], pairs['eval_seed'][e], pairs['state0'][e], ex)
            sd, sc, mg = L.true_costs(task, pairs['stateg'][e], fin)
            rec = dict(ep=e, task=task, split=a.split, sched=a.sched, tau=a.tau, trig=o['trig'], success=bool(sc[0]),
                       margin=float(mg[0]), state_dist=float(sd[0]), prox=o['prox'], time_s=(time.time() - t0) / len(eps))
            if a.sched == 'base':
                # outcome of stopping after every step (early-stop schedules are evaluated from these, no extra compute)
                exs = L.to_exec_actions(pre, torch.tensor(np.stack(o['mu_traj'])))
                f2 = L.replay_final_states(task, pairs['env_info'][e], pairs['eval_seed'][e], pairs['state0'][e], exs)
                s2, c2, m2 = L.true_costs(task, pairs['stateg'][e], f2)
                rec['traj_margin'] = [float(x) for x in m2]; rec['traj_success'] = [bool(x) for x in c2]
            with open(out_path, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            if o['pops']:
                np.savez_compressed(f'{R7.POPS}/{task}_{a.split}_e{e}.npz',
                                    **{f's{s}_{kk}': v for s, d in o['pops'].items() for kk, v in d.items()})
            print(f"{name} ep {e} success {rec['success']} margin {rec['margin']:.3f} trig {o['trig']} "
                  f"R_boot last {o['prox'][-1]['R_boot']:.3f} ({rec['time_s']:.1f}s/ep, {time.time()-t_start:.0f}s)",
                  flush=True)


if __name__ == '__main__':
    main()
