"""Manifest-driven, resumable CEM runner for the DINO-WM part of paper A (E0 reproduction and k selection, E1 main
benchmark; reused by E2/E3/E5 through new manifests).

usage: python runner.py <manifest.json> [--task T] [--max-jobs N] [--replay-workers W] [--nocompile] [--no-replay]
       (run through run.sh, which pins CUDA_VISIBLE_DEVICES=0 and the caches)

A manifest (make_manifest.py) is {"name", "config", "jobs": [job, ...]}. A job is
    {id, task, split, var, k, ep0, ep1, E, ranges, diag}
and covers the episodes [ep0, ep1) of one (task, split, variant, k), planned in blocks of E episodes (the batch size
E changes bf16 numerics, so it is fixed per task across all variants). The manifest lists jobs chunk-major (all
variants of an episode chunk before the next chunk), so a stopped run leaves every variant with the same episodes.
Every job writes one JSON results/<name>/<task>/<id>.json with one record per episode; an existing JSON is skipped.
Several runners may share a manifest (per-job lock file holding the owner's pid; a lock of a dead pid is taken over).
Per-episode arrays (iteration-0 and last CEM populations, costs) go to pops/<name>/<task>/ (server only); their
simulator replays (true costs of all 300 candidates) run in a CPU process pool while the GPU plans the next job.

Per-episode record: success, state_dist (evaluator distance), margin; final plan mean (plan deviation vs the
unquantized run of the same seed; 'nonealt' = unquantized with another planner seed gives the seed-noise floor);
planning-time 1/3/5-step errors vs the unquantized model on the iteration-0 population (3 dither draws for the
stochastic variants); OEI inputs: model cost J_v and unquantized cost J_none of the iteration-0 population (random
candidates), of the last population (argmin, elite) and of the final plan mean, with the true costs from the replay;
last-population rank correlations (as R2-13).
"""
import os, sys, json, time, glob, socket, argparse
import numpy as np
import torch
from multiprocessing import get_context
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wmlib as L
import qlib as QL

MAIN = L.MAIN
CODE = os.path.dirname(os.path.abspath(__file__))
REPLAY_CHUNK = 50          # candidates per replay task


def ranges_path(task, rid):
    if rid == 'pilot':
        return f'{CODE}/inputs/ranges_pilot_{task}.json'
    return f'{MAIN}/results/ranges/ranges_{rid}_{task}.json'


def load_ranges(task, rid):
    r = json.load(open(ranges_path(task, rid)))
    return [(x['lo'], x['hi']) for x in r['layers']]


def spearman(x, y):
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def try_lock(path):
    me = f'{os.getpid()}@{socket.gethostname()}'
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, me.encode()); os.close(fd)
        return True
    except FileExistsError:
        try:
            owner = open(path).read().strip()
            pid, host = owner.split('@')
            if host == socket.gethostname() and not pid_alive(int(pid)):
                os.remove(path)
                return try_lock(path)
        except Exception:
            pass
        return False


class TaskCtx:
    def __init__(self, task, use_compile, name):
        self.task = task
        self.T = L.load_task(task)
        self.m, self.pre = self.T['model'], self.T['pre']
        self.alpha = L.TASKS[task]['alpha']; self.opt_steps = L.TASKS[task]['opt_steps']
        self.env = L.make_env(task)
        self.Q = QL.Quantizer(self.m, use_compile=use_compile)
        self.pairs = {}
        self.ranges_id = None
        self.pops = f'{MAIN}/pops/{name}/{task}'
        os.makedirs(self.pops, exist_ok=True)

    def get_pairs(self, split):
        if split not in self.pairs:
            self.pairs[split] = QL.get_pairs(self.T, self.env, split, f'{MAIN}/pairs')
        return self.pairs[split]

    def set_ranges(self, rid):
        if rid != self.ranges_id:
            self.Q.set_ranges(load_ranges(self.task, rid))
            self.ranges_id = rid

    def set_ckpt(self, ckpt):
        """E5 QAT: job['ckpt'] = name of a fine-tuned predictor (MAIN/qat/<task>_<name>/predictor.pt); None = original."""
        if ckpt == getattr(self, 'ckpt', None):
            return
        if not hasattr(self, 'pred_orig'):
            self.pred_orig = {k: v.detach().clone() for k, v in self.m.predictor.state_dict().items()}
        sd = self.pred_orig if ckpt is None else torch.load(f'{MAIN}/qat/{self.task}_{ckpt}/predictor.pt', map_location='cuda')
        self.m.predictor.load_state_dict(sd)
        self.m.predictor.eval()
        self.ckpt = ckpt
        if self.Q.wq is not None:          # weight variants must re-read the (new) fp weights
            self.Q.wq = None

    def ensure_wq(self):
        if self.Q.wq is None:
            import ptq
            st = f'{MAIN}/results/ranges/awq_stats_{self.task}.pt'
            self.Q.wq = ptq.WeightQuant(self.m.predictor, torch.load(st) if os.path.exists(st) else None)


def submit_replay(pool, task, pairs, e, actions_norm, pre):
    """Replay all candidates of a population (normalized actions (N,H,A)) in chunks; returns list of AsyncResults."""
    ex = L.to_exec_actions(pre, torch.as_tensor(actions_norm))
    futs = []
    for s in range(0, len(ex), REPLAY_CHUNK):
        futs.append(pool.apply_async(L._replay_job, ((task, pairs['env_info'][e], pairs['eval_seed'][e],
                                                       pairs['state0'][e], ex[s:s + REPLAY_CHUNK], pairs['stateg'][e]),)))
    return futs


def gather(futs):
    outs = [f.get() for f in futs]
    return dict(sd=np.concatenate([o[1] for o in outs]).astype(np.float32),
                succ=np.concatenate([o[2] for o in outs]).astype(bool),
                margin=np.concatenate([o[3] for o in outs]).astype(np.float32))


def summarize(rec, pop, t_last, t_it0, topk=None):
    """OEI inputs and rank statistics of one episode from its population arrays and replayed true costs.
    Model-cost errors e = J_v - J_none (unquantized-model reference, R2-13 definition) and, provisionally,
    e_true = J_v - (a + b * margin) with (a, b) the OLS fit of J_none on the true margin over the iteration-0
    population of the same episode (true-cost reference in model-cost units)."""
    Jv0, Jn0 = pop['it0_J'], pop['it0_J_none']
    JvL, JnL = pop['last_J'], pop['last_J_none']
    am = int(np.argmin(JvL))
    el = np.argsort(JvL, kind='stable')[:int(topk or L.TOPK)]
    rnd = np.arange(1, len(Jv0))                      # iteration-0 random candidates (row 0 = mu = 0)
    d0, dL = Jv0 - Jn0, JvL - JnL
    o = dict(it0_rand_dJ=float(d0[rnd].mean()), it0_rand_dJ_sd=float(d0[rnd].std()),
             last_pop_dJ=float(dL.mean()), last_argmin_dJ=float(dL[am]), last_elite_dJ=float(dL[el].mean()),
             mf_dJ=float(rec['J_v_mf'] - rec['J_none_mf']),
             J_v_mf=float(rec['J_v_mf']), J_none_mf=float(rec['J_none_mf']),
             last_argmin_J=float(JvL[am]), last_argmin_J_none=float(JnL[am]),
             it0_rand_J=float(Jv0[rnd].mean()), it0_rand_J_none=float(Jn0[rnd].mean()),
             spec_it0=float(d0.std() / (Jn0.std() + 1e-12)), spec_last=float(dL.std() / (JnL.std() + 1e-12)),
             rho_J_Jnone_it0=spearman(Jv0, Jn0))
    o['oei_ref_argmin'] = o['last_argmin_dJ'] - o['it0_rand_dJ']
    o['oei_ref_elite'] = o['last_elite_dJ'] - o['it0_rand_dJ']
    o['oei_ref_mf'] = o['mf_dJ'] - o['it0_rand_dJ']
    if t_last is not None:
        y = t_last['margin']
        o.update(rho_last=spearman(JvL, y), rho_last_Jnone=spearman(JnL, y),
                 last_argmin_margin=float(y[am]), last_elite_margin=float(y[el].mean()),
                 last_argmin_sd=float(t_last['sd'][am]), last_argmin_success=bool(t_last['succ'][am]),
                 last_pop_success=float(t_last['succ'].mean()))
    if t_it0 is not None:
        y0 = t_it0['margin']
        o.update(rho_it0=spearman(Jv0, y0), rho_it0_Jnone=spearman(Jn0, y0),
                 it0_rand_margin=float(y0[rnd].mean()), it0_rand_sd=float(t_it0['sd'][rnd].mean()))
        b, a = np.polyfit(y0.astype(np.float64), Jn0.astype(np.float64), 1)
        e0 = Jv0 - (a + b * y0)
        base = float(e0[rnd].mean())
        o.update(calib_a=float(a), calib_b=float(b), it0_rand_etrue=base)
        if t_last is not None:
            eL = JvL - (a + b * t_last['margin'])
            o.update(oei_true_argmin=float(eL[am] - base), oei_true_elite=float(eL[el].mean() - base))
        o['oei_true_mf'] = float(rec['J_v_mf'] - (a + b * rec['margin']) - base)
    return o


class Pending:
    """A job whose GPU part is done and whose replays are in flight."""

    def __init__(self, job, out_path, lock_path, meta, records, pops, futs_last, futs_it0, it0_paths):
        self.job, self.out_path, self.lock_path, self.meta = job, out_path, lock_path, meta
        self.records, self.pops, self.futs_last, self.futs_it0, self.it0_paths = records, pops, futs_last, futs_it0, it0_paths

    def ready(self):
        return all(f.ready() for fs in list(self.futs_last.values()) + list(self.futs_it0.values()) for f in fs)

    def finalize(self, ctx_pops):
        t_rep0 = time.time()
        for e, futs in self.futs_it0.items():                 # iteration-0 truth (shared by all variants)
            tr = gather(futs)
            path = self.it0_paths[e]
            z = dict(np.load(path)) if os.path.exists(path) else {}
            z.update({f'truth_{k}': v for k, v in tr.items()})
            tmp = f'{path}.tmp{os.getpid()}.npz'
            np.savez_compressed(tmp, **z); os.replace(tmp, path)
        for rec in self.records:
            e = rec['ep']
            t_last = None
            if e in self.futs_last:
                t_last = gather(self.futs_last[e])
                np.savez_compressed(self.pops[e].replace('.npz', '_truth.npz'), **t_last)
            t_it0 = None
            p0 = self.it0_paths.get(e)
            if p0 and os.path.exists(p0):
                z0 = np.load(p0)
                if 'truth_margin' in z0.files:
                    t_it0 = dict(margin=z0['truth_margin'], sd=z0['truth_sd'], succ=z0['truth_succ'])
            if self.job['diag']:
                pz = np.load(self.pops[e])
                rec['oei'] = summarize(rec, {k: pz[k] for k in pz.files}, t_last, t_it0,
                                       topk=(self.job.get('planner') or {}).get('topk'))
        self.meta['replay_wait_s'] = time.time() - t_rep0
        self.meta['t_finalized'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        L.jdump(dict(job=self.job, meta=self.meta, records=self.records), self.out_path)
        if os.path.exists(self.lock_path):
            os.remove(self.lock_path)


def run_job(ctx, job, name, pool, res_dir):
    task, split, var, k, E = job['task'], job['split'], job['var'], int(job['k']), int(job['E'])
    sp = QL.SPLITS[split]
    pairs = ctx.get_pairs(split)
    ctx.set_ranges(job['ranges'])
    m, pre, Q = ctx.m, ctx.pre, ctx.Q
    ctx.set_ckpt(job.get('ckpt'))         # E5 QAT predictor (None = original checkpoint)
    Q.last_bits = (job.get('qopts') or {}).get('last_bits')     # E5 mixed precision (None = all k bits)
    if var in QL.WVARS:
        ctx.ensure_wq()
    pl = job.get('planner') or {}          # optional overrides for planner sweeps (E2): num_samples, topk, opt_steps
    opt_steps = int(pl.get('opt_steps', ctx.opt_steps))
    eps = list(range(int(job['ep0']), int(job['ep1'])))
    records, pops, futs_last, futs_it0, it0_paths = [], {}, {}, {}, {}
    t_job = time.time(); gpu_s = 0.0
    torch.cuda.reset_peak_memory_stats()
    for g0 in range(0, len(eps), E):
        ee = eps[g0:g0 + E]
        inputs = [QL.episode_inputs(m, pre, pairs, e) for e in ee]
        gens = [torch.Generator().manual_seed(QL.plan_seed(split, var, e)) for e in ee]
        n0 = Q.n_calls
        torch.cuda.synchronize(); t0 = time.time()
        outs = QL.run_cem(m, Q, inputs, gens, ctx.alpha, opt_steps, ee, var, k, sp['salt'], diag=job['diag'],
                          n_samples=pl.get('num_samples'), topk=pl.get('topk'), max_rows=pl.get('max_rows'))
        torch.cuda.synchronize(); t_cem = time.time() - t0
        gpu_s += t_cem
        for kk, e in enumerate(ee):
            o = outs[kk]
            ex = L.to_exec_actions(pre, torch.tensor(o['mu_final'])[None])
            fin = L.replay_final_states(task, pairs['env_info'][e], pairs['eval_seed'][e], pairs['state0'][e], ex)
            sd, sc, mg = L.true_costs(task, pairs['stateg'][e], fin)
            rec = dict(ep=e, task=task, split=split, var=var, k=k, success=bool(sc[0]), state_dist=float(sd[0]),
                       margin=float(mg[0]), mu_final=o['mu_final'].round(5).tolist(), best=[round(b, 6) for b in o['best']],
                       time_s=t_cem / len(ee), E=len(ee), hook_calls=Q.n_calls - n0,
                       plan_seed=QL.plan_seed(split, var, e),
                       t_block_start=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t0)),
                       t_block_end=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t0 + t_cem)))
            if job['diag']:
                rec['J_v_mf'], rec['J_none_mf'] = o['J_v_mf'], o['J_none_mf']
                if 'err_draws' in o:
                    rec['err'] = {str(s): dict(rel=float(np.mean(o['err_draws'][s]) / o['pred_var'][s]),
                                               abs=float(np.mean(o['err_draws'][s])),
                                               draws=[round(x, 8) for x in o['err_draws'][s]],
                                               pred_var=o['pred_var'][s]) for s in QL.ERR_STEPS}
                last = opt_steps - 1
                pz = dict(it0_J=o['pops'][0]['J'], it0_J_none=o['J_none_it0'], last_actions=o['pops'][last]['actions'],
                          last_J=o['pops'][last]['J'], last_V=o['pops'][last]['V'], last_Q=o['pops'][last]['Q'],
                          last_J_none=o['pops'][last]['J_none'], mu_final=o['mu_final'])
                path = f"{ctx.pops}/{job['id']}_e{e}.npz"
                np.savez_compressed(path, **pz)
                pops[e] = path
                if pool is not None:
                    futs_last[e] = submit_replay(pool, task, pairs, e, o['pops'][last]['actions'], pre)
                if var != 'nonealt':     # the iteration-0 population of the main planner seed (all variants)
                    nkey = f"_N{int(pl['num_samples'])}" if pl.get('num_samples') else ''   # it0 population depends on N only
                    p0 = f'{ctx.pops}/{split}{nkey}_it0_e{e}.npz'
                    it0_paths[e] = p0
                    if not os.path.exists(p0):
                        np.savez_compressed(p0, actions=o['pops'][0]['actions'], J_none=o['J_none_it0'])
                        if pool is not None:
                            futs_it0[e] = submit_replay(pool, task, pairs, e, o['pops'][0]['actions'], pre)
                    else:
                        z0 = np.load(p0)
                        assert np.array_equal(z0['actions'], o['pops'][0]['actions']), f'it0 population differs ep {e}'
            records.append(rec)
            print(f"{time.strftime('%H:%M:%S', time.gmtime())} {task} {split} {var}:{k} ep {e} success {rec['success']} "
                  f"margin {rec['margin']:.3f} ({rec['time_s']:.1f}s/ep)", flush=True)
    props = torch.cuda.get_device_properties(0)
    meta = dict(gpu=torch.cuda.get_device_name(0), host=socket.gethostname(), pid=os.getpid(),
                torch=torch.__version__, cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
                cuda_device_order=os.environ.get('CUDA_DEVICE_ORDER'), gpu_uuid=str(getattr(props, 'uuid', '')),
                t_start=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t_job)), gpu_s=gpu_s,
                wall_gpu_part_s=time.time() - t_job, peak_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                ranges=job['ranges'], salt=sp['salt'], pair_seed=sp['pair_seed'], opt_steps=opt_steps, planner=pl,
                qopts=job.get('qopts'), ckpt=job.get('ckpt'),
                tf32=torch.backends.cuda.matmul.allow_tf32, sdpa=True, autocast='bf16', name=name)
    Q.last_bits = None
    if Q.wq is not None:
        Q.wq.restore()
    out_path = f"{res_dir}/{job['id']}.json"
    return Pending(job, out_path, out_path.replace('.json', '.lock'), meta, records, pops, futs_last, futs_it0, it0_paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('manifest'); ap.add_argument('--task', default=None)
    ap.add_argument('--max-jobs', type=int, default=0); ap.add_argument('--replay-workers', type=int, default=8)
    ap.add_argument('--nocompile', action='store_true'); ap.add_argument('--no-replay', action='store_true')
    ap.add_argument('--max-pending', type=int, default=3)
    ap.add_argument('--ks', default=None, help='only quantized jobs with these k (comma list); none/nonealt jobs always '
                    'included (with the per-job locks two runners on two GPUs share them, first come first served)')
    ap.add_argument('--cells', default=None, help='only jobs whose "cell" field is in this comma list (v5 chain: '
                    'v5_chain.py claims one cell for one GPU and runs it with this filter)')
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    name = man['name']
    torch.backends.cuda.matmul.allow_tf32 = True          # pilot numerics (R2-13): TF32 matmul, SDPA attention, bf16
    torch.set_num_threads(int(os.environ.get('RUNNER_THREADS', '4')))
    L.patch_attention(True)
    jobs = [j for j in man['jobs'] if a.task is None or j['task'] == a.task]
    if a.ks:
        keep = {int(x) for x in a.ks.split(',')}
        jobs = [j for j in jobs if j['var'] in ('none', 'nonealt') or int(j['k']) in keep]
    if a.cells:
        keep_c = set(a.cells.split(','))
        jobs = [j for j in jobs if j.get('cell') in keep_c]
    pool = None if a.no_replay else get_context('spawn').Pool(a.replay_workers)
    ctxs, pending, n_run, t0 = {}, [], 0, time.time()
    print(f"RUNNER {name}: {len(jobs)} jobs (task filter {a.task}), pid {os.getpid()}", flush=True)
    for job in jobs:
        res_dir = f"{MAIN}/results/{name}/{job['task']}"
        os.makedirs(res_dir, exist_ok=True)
        out_path = f"{res_dir}/{job['id']}.json"
        if os.path.exists(out_path):
            continue
        if not try_lock(out_path.replace('.json', '.lock')):
            continue
        if job['task'] not in ctxs:
            ctxs = {job['task']: TaskCtx(job['task'], not a.nocompile, name)}    # one model in memory at a time
        ctx = ctxs[job['task']]
        tj = time.time()
        p = run_job(ctx, job, name, pool, res_dir)
        pending.append(p)
        n_run += 1
        print(f"JOB_GPU_DONE {job['id']} in {time.time() - tj:.0f}s (total {time.time() - t0:.0f}s)", flush=True)
        while pending and (pending[0].ready() or len(pending) > a.max_pending):
            pending.pop(0).finalize(ctx.pops)
        if a.max_jobs and n_run >= a.max_jobs:
            break
    for p in pending:
        p.finalize(None)
    if pool is not None:
        pool.close(); pool.join()
    print(f"RUNNER_DONE {name} jobs run {n_run} in {time.time() - t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
