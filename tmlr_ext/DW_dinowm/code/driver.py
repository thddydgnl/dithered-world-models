"""On-pod queue driver of the DINO-WM group (dw), paper A TMLR extension (DESIGN_tmlr.md v1.0; WORKER_RULES.md 3).

Stateless and resumable: every pass rebuilds the full list of work items of this pod from the files on disk (job
JSONs, calibration files, decisions), in priority order, and runs the first item that is neither done, failed (3
attempts) nor blocked. The adaptive rules are applied here from the results:
  - check 0.7-3: after the eval8 none (CEM) of a task, its success must be within 10 points of E1's none; otherwise
    every further planning job of that task is held (out/checks/check3_<task>.json, status note), never deleted
  - E12 two-stage sampling (PushT): 50 episodes; the other 50 only if the paired 95% bootstrap CI of D holds tau
  - E12 extension rule: from the grid's lowest k down while D <= tau (to k = 1), from its highest k up while D > tau
    (to k = 8); the pod that holds the boundary k runs the extension (podplan.PODS[...]['ext'])
  - E16 lambda: none-MPPI on the 20 held pairs for c in {0.25, 0.5, 1, 2}; highest success, ties to the larger c
Priority: calibration and pairs -> eval8 none -> E11 -> E12 (grid stage 1, stage 2, extensions) -> E16 -> held none
-> E13. One model in memory; replays of all candidates (E11 diagnostics) in a CPU pool while the GPU plans.
Exit codes: 0 finished, 3 a job failed (supervisor restarts), 10 nothing runnable for a long time (supervisor waits).
"""
import os, sys, json, time, glob, socket, traceback
import numpy as np
import torch
from multiprocessing import get_context
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wmlib as L
import qlib as QL
import runner as RN
import podplan as PP

ATM = os.environ.get('ATM', '/workspace/atm')
OUT, BIG, CODE = f'{ATM}/out', f'{ATM}/big', RN.CODE
JOBS, STATE, CHECKS = f'{OUT}/jobs', f'{OUT}/state', f'{OUT}/checks'
SPLITS_DIR, E13_DIR = f'{OUT}/splits', f'{OUT}/e13'
NAME = 'tmlr'                                   # pops/<NAME>/<task>/ under DINOWM_MAIN (= big/)
MAX_ATTEMPTS = 3                                # 1 + 2 retries (WORKER_RULES 3)


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def jload(p, default=None):
    try:
        return json.load(open(p))
    except Exception:
        return default


def boot_ci(d, seed=0):
    rng = np.random.RandomState(seed)
    n = len(d)
    m = d[rng.randint(0, n, size=(PP.NBOOT, n))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


class Item:
    """kind in {calib, wait, job, e13}; key = unique id; needs = files that must exist; run = callable."""

    def __init__(self, kind, key, task, needs=(), job=None, run=None, eps=0, note=''):
        self.kind, self.key, self.task, self.needs, self.job, self.run, self.eps, self.note = \
            kind, key, task, tuple(needs), job, run, eps, note


class Driver:
    def __init__(self, only=None):
        self.only = only                          # 'calib': calibration and pairs items only (phase A), then exit
        self.tag = open(f'{ATM}/TAG').read().strip()
        self.P = PP.plan_for(self.tag)
        for d in (JOBS, STATE, CHECKS, SPLITS_DIR, E13_DIR, RN.RANGES):
            os.makedirs(d, exist_ok=True)
        self.ctx = None
        self.pending = []                         # runner.Pending (GPU done, replays in flight)
        self.pool = None
        self.recs_cache = {}
        self.t_start = time.time()
        self.fails = jload(f'{STATE}/fails.json', {})
        self.current = '-'
        self.note = ''
        self.blocked_since = None
        self.bias_rule = None                     # decided det+BC iteration rule (decisions.json 'bias_rule')
        self.decisions = {}

    # ---------------------------------------------------------------------------------------------- bookkeeping
    def job_path(self, task, jid):
        return f'{JOBS}/{task}/{jid}.json'

    def is_done(self, it):
        if it.kind in ('job',):
            return os.path.exists(self.job_path(it.task, it.job['id'])) or any(
                p.job['id'] == it.job['id'] and p.job['task'] == it.task for p in self.pending)
        if it.kind == 'calib' or it.kind == 'wait':
            if hasattr(it, 'done'):
                return it.done()
            return all(os.path.exists(f) for f in it.produces)
        if it.kind == 'e13':
            return it.done()
        return False

    def records(self, task, jid):
        """Episode records of a job: finalized JSON or a pending job's in-memory records."""
        for p in self.pending:
            if p.job['id'] == jid and p.job['task'] == task:
                return p.records
        path = self.job_path(task, jid)
        if path in self.recs_cache:
            return self.recs_cache[path]
        d = jload(path)
        if d is None:
            return None
        self.recs_cache[path] = d['records']
        return d['records']

    def cell_records(self, jobs):
        out = {}
        for j in jobs:
            r = self.records(j['task'], j['id'])
            if r is None:
                return None
            for x in r:
                out[int(x['ep'])] = x
        return out

    def failed(self, key):
        return self.fails.get(key, 0) >= MAX_ATTEMPTS

    def add_fail(self, key, err):
        self.fails[key] = self.fails.get(key, 0) + 1
        L.jdump(self.fails, f'{STATE}/fails.json')
        with open(f'{STATE}/fail_log.jsonl', 'a') as f:
            f.write(json.dumps(dict(time=now(), key=key, attempt=self.fails[key], err=err[-3000:])) + '\n')

    # ---------------------------------------------------------------------------------------------- context
    def get_ctx(self, task):
        if self.ctx is None or self.ctx.task != task:
            self.finalize_all()
            self.ctx = None
            torch.cuda.empty_cache()
            t0 = time.time()
            self.ctx = RN.TaskCtx(task, True, NAME)
            print(f'{now()} loaded {task} in {time.time() - t0:.0f}s', flush=True)
        return self.ctx

    def get_pool(self):
        if self.pool is None:
            n = int(os.environ.get('DW_REPLAY_WORKERS', max(2, min(8, (os.cpu_count() or 4) - 2))))
            self.pool = get_context('spawn').Pool(n)
            print(f'{now()} replay pool: {n} workers', flush=True)
        return self.pool

    def finalize_ready(self, block=False):
        keep = []
        for p in self.pending:
            if block or p.ready():
                p.finalize(None)
            else:
                keep.append(p)
        self.pending = keep
        while len(self.pending) > 3:
            self.pending.pop(0).finalize(None)

    def finalize_all(self):
        self.finalize_ready(block=True)

    # ---------------------------------------------------------------------------------------------- decisions
    def load_decisions(self):
        d = jload(f'{CODE}/decisions.json', {}) or {}
        self.decisions = d
        self.bias_rule = d.get('bias_rule')       # None: pending (b computed with the design rule, det+BC jobs held)
        ev = d.get('eval8_pusht')
        if ev:
            QL.EVAL8_PUSHT = dict(hard=[tuple(x) for x in ev['hard']], soft=[tuple(x) for x in ev['soft']])
        return d

    def target_rule(self):
        import calib_tmlr as C
        return dict(self.bias_rule) if self.bias_rule else dict(C.DESIGN_RULE)

    def bias_ok(self, path):
        """bias file present and computed with the target rule (the decided one, else the design rule)."""
        import calib_tmlr as C
        d = jload(path)
        return d is not None and d.get('rule_name') == C.rule_name(self.target_rule())

    def pairs_ready(self, task, split):
        return os.path.exists(f'{BIG}/pairs/pairs_{task}_{split}.npz') and os.path.exists(f'{BIG}/pairs/pairs_{task}_{split}.json')

    def eval8_blocked(self, task):
        return task == 'pusht' and QL.EVAL8_PUSHT is None and not self.pairs_ready(task, 'eval8')

    def eval8_why(self, task):
        """Reason the eval8 pairs of a task cannot be built yet (None: go)."""
        if task != 'pusht':
            return None
        if self.eval8_blocked(task):
            return 'eval8 PushT decision pending'
        pre = jload(f'{CHECKS}/eval8_pusht_precheck.json')
        if pre is None:
            return 'eval8 PushT precheck pending'
        if not pre.get('passed'):
            return 'eval8 PushT precheck FAILED (D1 stop condition): stop and report'
        return None

    def split_violation(self, task):
        """PushT eval8 planning is held while the split file is missing or lists a D1 violation."""
        if task != 'pusht':
            return None
        sj = jload(f'{SPLITS_DIR}/eval8_{task}.json')
        if sj is None:
            return 'eval8 PushT split bookkeeping missing'
        if sj.get('violations'):
            return 'eval8 PushT split violation (stop and report): ' + '; '.join(sj['violations'])
        return None

    # ---------------------------------------------------------------------------------------------- item builders
    def none_jobs(self, task, split='eval8', planner=None, diag=True, n=PP.N_EVAL):
        return [PP.job(task, split, 'none', 0, a, b, diag=diag, planner=planner, exp=('none',))
                for a, b in PP.chunks(task, 0, n)]

    def cell_jobs(self, task, kind, var, k, e0, e1, diag=False, planner=None, exp=('E12',)):
        return [PP.job(task, 'eval8', var, k, a, b, ranges=PP.rid_of(kind, k), diag=diag, planner=planner, exp=exp)
                for a, b in PP.chunks(task, e0, e1)]

    def job_items(self, jobs, note=''):
        out = []
        for j in jobs:
            needs = [RN.ranges_path(j['task'], j['ranges'])]
            if j.get('bias'):
                needs.append(RN.bias_path(j['task'], j['bias']))
            out.append(Item('job', j['id'], j['task'], needs=needs, job=j, eps=j['ep1'] - j['ep0'], note=note))
        return out

    def calib_items(self, task):
        """Owner: compute; other pods: wait for the copied file."""
        owner = task in self.P['owner']
        items = []
        R = RN.RANGES
        pc = [f'{R}/ranges_pc_{task}.json', f'{R}/pctgrid_{task}.json', f'{R}/ptcheck_{task}.json']
        it = Item('calib' if owner else 'wait', f'calib:{task}:pcrec', task, run=lambda t=task: self.do_pcrec(t))
        it.produces = pc
        items.append(it)
        for rid, k in PP.owner_bias_needs(task):
            f = RN.bias_path(task, f'{rid}_k{k}')
            it = Item('calib' if owner else 'wait', f'calib:{task}:bias:{rid}_k{k}', task,
                      needs=[RN.ranges_path(task, rid)], run=lambda t=task, r=rid, kk=k: self.do_bias(t, r, kk))
            it.produces = [f]
            it.done = (lambda p=f: self.bias_ok(p))
            items.append(it)
        for k in PP.owner_rc_needs(task):
            it = Item('calib' if owner else 'wait', f'calib:{task}:rc_k{k}', task, needs=[f'{R}/pctgrid_{task}.json'],
                      run=lambda t=task, kk=k: self.do_rc(t, kk))
            it.produces = [RN.ranges_path(task, f'rc_k{k}')]
            items.append(it)
        return items

    def local_calib_for(self, task, kind, var, k):
        """Extension-k artifacts computed on this pod (bias for detbc, RC ranges)."""
        items = []
        rid = PP.rid_of(kind, k)
        if kind == 'rc' and not os.path.exists(RN.ranges_path(task, rid)):
            it = Item('calib', f'calib:{task}:rc_k{k}', task, needs=[f'{RN.RANGES}/pctgrid_{task}.json'],
                      run=lambda t=task, kk=k: self.do_rc(t, kk))
            it.produces = [RN.ranges_path(task, rid)]
            items.append(it)
        if var in QL.BIAS_VARS and not self.bias_ok(RN.bias_path(task, f'{rid}_k{k}')):
            it = Item('calib', f'calib:{task}:bias:{rid}_k{k}', task, needs=[RN.ranges_path(task, rid)],
                      run=lambda t=task, r=rid, kk=k: self.do_bias(t, r, kk))
            it.produces = [RN.bias_path(task, f'{rid}_k{k}')]
            it.done = (lambda p=it.produces[0]: self.bias_ok(p))
            items.append(it)
        return items

    def ex_rule(self, task):
        """check-3 exception of a task: PushT D1 (eval6 0-49 vs 41/50), otherwise the D3 rule (E1 split none on this pod
        within 10 points of the E1 record)."""
        if task == 'pusht':
            ex = (self.decisions.get('eval8_pusht') or {}).get('check3_exception')
            if ex:
                return ex
        return (self.decisions.get('check3_rule_D3') or {}).get(task)

    def exception_jobs(self, task):
        ex = self.ex_rule(task) or {}
        a, b = ex.get('eps', [0, 50])
        return [PP.job(task, ex.get('split', 'eval6'), 'none', 0, x, y, diag=False, exp=('check3_exception',))
                for x, y in PP.chunks(task, a, b)]

    def check3(self, task):
        """None-success gate (DESIGN 0.7-3, D1 exception for PushT). Returns None (eval8 none incomplete), True, False
        or 'exception_pending' (PushT outside the window: the eval6 0-49 none runs decide, D1)."""
        path = f'{CHECKS}/check3_{task}.json'
        recs = self.cell_records(self.none_jobs(task))
        if recs is None or len(recs) < PP.N_EVAL:
            return None
        s = float(np.mean([float(r['success']) for r in recs.values()]))
        ok = abs(s - PP.E1_NONE[task]) <= PP.CHECK3_TOL + PP.EPS
        res = dict(task=task, split='eval8', n=len(recs), none_success=s, e1_none_success=PP.E1_NONE[task],
                   diff=s - PP.E1_NONE[task], tolerance=PP.CHECK3_TOL, within_window=bool(ok), passed=bool(ok),
                   tag=self.tag)
        ov = (self.decisions.get('check3_override') or {}).get(task)
        if not ok and ov:                             # decided exception (D3 for Wall), failure numbers kept
            res.update(exception=dict(ov), passed=True,
                       deviation='eval8 none outside the window; exception decided by the coordinator: proceed')
        elif not ok and self.ex_rule(task):           # D1 (PushT) / D3 rule: E1 split none on this pod
            ex = self.ex_rule(task)
            exr = self.cell_records(self.exception_jobs(task))
            if exr is None or len(exr) < ex['eps'][1] - ex['eps'][0]:
                return 'exception_pending'
            n = int(sum(bool(r['success']) for r in exr.values()))
            lo, hi = ex['accept']
            res.update(exception=dict(rule=('D1' if task == 'pusht' else 'D3') + f": {ex['split']} pairs {ex['eps'][0]}-"
                                           f"{ex['eps'][1] - 1} none vs E1 {ex['e1_success']}, accept {lo}-{hi}",
                                      split=ex['split'], n=len(exr), successes=n, e1_successes=ex['e1_success'],
                                      accepted=bool(lo <= n <= hi)),
                       passed=bool(lo <= n <= hi),
                       deviation=('eval8 none outside the window; the E1 split none reproduces E1 -> split population '
                                  'difference, proceed') if lo <= n <= hi else 'the E1 split none does not reproduce E1: stop')
        old = jload(path)
        if old is None or old.get('passed') != res['passed'] or ('exception' in res) != ('exception' in old):
            if old is not None:                       # keep every earlier record (e.g. the failure before D3)
                res['history'] = old.get('history', []) + [{k: v for k, v in old.items() if k != 'history'}]
            res['time'] = now()
            L.jdump(res, path)
        return bool(res['passed'])

    def D_of(self, task, jobs, planner=None):
        """Paired D (success drop vs this pod's none of the same planner) on the cell's episodes; None if incomplete."""
        recs = self.cell_records(jobs)
        if recs is None:
            return None
        nrecs = self.cell_records(self.none_jobs(task, planner=planner, diag=planner is None))
        if nrecs is None:
            return None
        eps = sorted(set(recs) & set(nrecs))
        if len(eps) < len(recs):
            return None
        d = np.array([float(nrecs[e]['success']) - float(recs[e]['success']) for e in eps])
        return dict(n=len(eps), D=float(d.mean()), ci=boot_ci(d))

    def cell_plan(self, task, kind, var, k, dec, diag=False, exp=('E12',)):
        """Episode jobs of one E12 cell under the two-stage rule; returns (items, complete, D)."""
        if task in PP.TWO_STAGE:
            s1 = self.cell_jobs(task, kind, var, k, 0, PP.STAGE1, diag=diag, exp=exp)
            items = self.job_items(s1)
            r1 = self.D_of(task, s1)
            if r1 is None:
                return items, False, None
            lo, hi = r1['ci']
            need2 = (lo <= PP.TAU + PP.EPS) and (hi >= PP.TAU - PP.EPS)
            dec[PP.cell_id(task, kind, var, k)] = dict(stage1=r1, stage2=bool(need2))
            if not need2:
                return items, True, r1['D']
            s2 = self.cell_jobs(task, kind, var, k, PP.STAGE1, PP.N_EVAL, diag=diag, exp=exp)
            items += self.job_items(s2)
            r = self.D_of(task, s1 + s2)
            if r is None:
                return items, False, None
            dec[PP.cell_id(task, kind, var, k)]['final'] = r
            return items, True, r['D']
        jobs = self.cell_jobs(task, kind, var, k, 0, PP.N_EVAL, diag=diag, exp=exp)
        r = self.D_of(task, jobs)
        if r is not None:
            dec[PP.cell_id(task, kind, var, k)] = dict(final=r)
        return self.job_items(jobs), r is not None, (None if r is None else r['D'])

    def build(self):
        """All work items of this pod in priority order (rebuilt every pass)."""
        P, items, dec = self.P, [], {}
        self.load_decisions()
        gate = {}
        # 1 calibration (owner computes, others wait for the copies) and pairs
        for task in P['tasks']:
            items += self.calib_items(task)
            it = Item('calib', f'pairs:{task}:held', task, run=lambda t=task: self.do_pairs(t, 'held'))
            it.produces = [f'{BIG}/pairs/pairs_{task}_held.npz']
            items.append(it)
            if task == 'pusht':                                        # D1 precheck before sampling
                it = Item('calib', f'check:{task}:eval8pre', task, run=lambda t=task: self.do_precheck(t))
                it.produces = [f'{CHECKS}/eval8_pusht_precheck.json']
                it.blocked_by = (lambda t=task: 'eval8 PushT decision pending' if self.eval8_blocked(t) else None)
                items.append(it)
            it = Item('calib', f'pairs:{task}:eval8', task, run=lambda t=task: self.do_pairs(t, 'eval8'))
            it.produces = [f'{BIG}/pairs/pairs_{task}_eval8.npz', f'{SPLITS_DIR}/eval8_{task}.json']
            it.blocked_by = (lambda t=task: self.eval8_why(t))
            items.append(it)
        # 2 eval8 none (CEM, diagnostics on: E11 and E13 inputs) and the 0.7-3 gate
        for task in P['tasks']:
            items += self.job_items(self.none_jobs(task), note='none')
            g = self.check3(task)
            if g == 'exception_pending':                              # D1 / D3: eval8 none outside the window
                if (self.ex_rule(task) or {}).get('split') == 'eval6':
                    it = Item('calib', f'pairs:{task}:eval6', task, run=lambda t=task: self.do_pairs(t, 'eval6'))
                    it.produces = [f'{CHECKS}/eval6_regen_{task}.json']
                    items.append(it)
                items += self.job_items(self.exception_jobs(task), note='check3 exception')
            gate[task] = g
        ok = lambda t: gate.get(t) is True
        # 3 E11 (chunk-major: every variant of a chunk before the next chunk)
        for task in P['e11']:
            k = PP.K1[task]
            for a, b in PP.chunks(task, 0, PP.N_EVAL):
                jobs = [PP.job(task, 'eval8', v, k, a, b, ranges='main', diag=True, exp=('E11', 'E12') if v in
                               PP.E12_ROUND['main'] else ('E11',)) for v in PP.E11_VARS]
                items += [self._gated(x, ok) for x in self.job_items(jobs, note='E11')]
        # 4 E12 grid (stage 1 chunk-major across cells, then stage 2), then extensions
        for task in P['e12']:
            cells = PP.grid_cells(self.tag, task)
            s1, rest = [], []
            for kind, var, k in cells:
                ci, complete, D = self.cell_plan(task, kind, var, k, dec)
                n1 = len(PP.chunks(task, 0, PP.STAGE1)) if task in PP.TWO_STAGE else len(ci)
                s1.append(ci[:n1]); rest += ci[n1:]
            for c in range(max((len(x) for x in s1), default=0)):          # chunk-major across the grid cells
                items += [self._gated(x[c], ok) for x in s1 if c < len(x)]
            items += [self._gated(x, ok) for x in rest]
        for task, dirs in P['ext'].items():
            for kind, ks in PP.E12_GRID[task].items():
                for var in PP.E12_ROUND[kind]:
                    for direction in dirs:
                        items += [self._gated(x, ok) for x in self.extension(task, kind, var, ks, direction, dec)]
        L.jdump(dict(time=now(), tag=self.tag, cells=dec), f'{STATE}/decisions.json')
        # 5 E16: lambda selection on held pairs, then eval8 cells under the chosen c
        for task in P['e16']:
            lam_jobs = {c: self.none_jobs(task, split='held', planner={'mppi_c': c}, diag=False, n=PP.N_HELD)
                        for c in PP.E16_C}
            for c in PP.E16_C:
                items += [self._gated(x, ok) for x in self.job_items(lam_jobs[c], note='E16 lambda')]
            sel = self.lambda_choice(task, lam_jobs)
            if sel is not None:
                pl = {'mppi_c': sel}
                for a, b in PP.chunks(task, 0, PP.N_EVAL):
                    jobs = [PP.job(task, 'eval8', 'none', 0, a, b, planner=pl, exp=('E16',))]
                    jobs += [PP.job(task, 'eval8', v, PP.E16_K, a, b, ranges='main', planner=pl, exp=('E16',))
                             for v in PP.E16_VARS]
                    items += [self._gated(x, ok) for x in self.job_items(jobs, note='E16')]
        # 6 E13: held none (CEM, diagnostics) and the offline diagnostics, after every episode job of the task
        for task in P['e13']:
            held = self.none_jobs(task, split='held', n=PP.N_HELD)
            items += self.job_items(held, note='held none')
            items += self.e13_items(task, items)
        return items

    def _gated(self, it, ok):
        def why(t=it.task):
            if ok(t):
                return None
            g = self.check3(t)
            return {False: f'check3 FAIL {t}: stop and report', 'exception_pending': f'check3 {t}: D1 eval6 exception running',
                    None: f'waiting eval8 none {t}'}.get(g, f'check3 {t} {g}')
        it.blocked_by = why
        return it

    def boundary_D(self, task, kind, var, k, dec):
        """(complete, D) of the cell at the grid boundary k, if this pod holds it; (False, None) otherwise."""
        if kind == 'main' and k == PP.K1[task] and task in self.P['e11'] and var in PP.E11_VARS:
            jobs = [PP.job(task, 'eval8', var, k, a, b, ranges='main', diag=True) for a, b in PP.chunks(task, 0, PP.N_EVAL)]
            r = self.D_of(task, jobs)
            if r is not None:
                dec[PP.cell_id(task, kind, var, k)] = dict(final=r, source='E11')
            return (r is not None), (None if r is None else r['D'])
        if (kind, var, k) not in PP.grid_cells(self.tag, task):
            return False, None
        _, complete, D = self.cell_plan(task, kind, var, k, dec)
        return complete, D

    def extension(self, task, kind, var, ks, direction, dec):
        """Extension cells of one (task, ranges kind, rounding) in one direction (DESIGN 2.2): from the grid's highest k
        up while D > tau (to k = 8), from its lowest k down while D <= tau (to k = 1)."""
        items = []
        k = max(ks) if direction == 'up' else min(ks)
        complete, D = self.boundary_D(task, kind, var, k, dec)
        while complete and D is not None:
            go = (D > PP.TAU + PP.EPS) if direction == 'up' else (D <= PP.TAU + PP.EPS)
            nk = k + 1 if direction == 'up' else k - 1
            if not go or nk > PP.K_MAX or nk < PP.K_MIN:
                break
            items += self.local_calib_for(task, kind, var, nk)
            ci, complete, D = self.cell_plan(task, kind, var, nk, dec, exp=('E12', 'ext'))
            items += ci
            k = nk
        return items

    def lambda_choice(self, task, lam_jobs):
        path = f'{STATE}/e16_lambda_{task}.json'
        rows = {}
        for c, jobs in lam_jobs.items():
            r = self.cell_records(jobs)
            if r is None or len(r) < PP.N_HELD:
                return None
            rows[c] = dict(success=float(np.mean([float(x['success']) for x in r.values()])), n=len(r),
                           lam_mean=float(np.mean([x.get('mppi_lam', np.nan) for x in r.values()])))
        best = max(PP.E16_C, key=lambda c: (rows[c]['success'], c))       # ties -> larger c
        if jload(path) is None:
            L.jdump(dict(task=task, split='held', rows={str(c): v for c, v in rows.items()}, chosen_c=best,
                         rule='highest none success on the 20 held pairs; ties to the larger c', time=now()), path)
        return best

    def e13_items(self, task, items_so_far):
        """One item per input state, listed only when every episode job of the task is done."""
        if any(it.kind == 'job' and it.task == task and not self.is_done(it) and not self.failed(it.key)
               for it in items_so_far):
            return []
        cfgs = self.e13_configs(task)
        out = []
        split_eps = [('held', e) for e in range(PP.N_HELD)] + [('eval8', e) for e in range(PP.N_EVAL)]
        for split, e in split_eps:
            path = f'{E13_DIR}/{task}/{split}_e{e:03d}.json'
            it = Item('e13', f'e13:{task}:{split}:{e}', task, run=lambda t=task, s=split, ee=e, p=path, c=cfgs:
                      self.do_e13(t, s, ee, p, c))
            it.done = (lambda p=path, c=cfgs: self._e13_done(p, c))
            out.append(it)
        return out

    def e13_configs(self, task):
        """E11 cell x 8 variants, then the E12 cells of the task within one bit of K1 that exist on any pod's plan or
        were run here (extension cells: if their job files exist on this pod)."""
        k1 = PP.K1[task]
        cfgs = [dict(ranges='main', var=v, k=k1) for v in PP.E11_VARS]
        seen = {(c['ranges'], c['var'], c['k']) for c in cfgs}
        for kind, ks in PP.E12_GRID[task].items():
            for k in (k1 - 1, k1, k1 + 1):
                for var in PP.E12_ROUND[kind]:
                    rid = PP.rid_of(kind, k)
                    in_grid = k in ks
                    ran_here = bool(glob.glob(f'{JOBS}/{task}/eval8_{rid}_{var}_k{k}_cem_*.json'))
                    if (in_grid or ran_here) and (rid, var, k) not in seen and os.path.exists(RN.ranges_path(task, rid)) \
                            and (var not in QL.BIAS_VARS or os.path.exists(RN.bias_path(task, f'{rid}_k{k}'))):
                        cfgs.append(dict(ranges=rid, var=var, k=k)); seen.add((rid, var, k))
        part = self.P.get('e13_part')                    # dw.1 / dw.5 split of the PushT E13 configurations
        if part == 'main_ext':
            cfgs = [c for c in cfgs if c['ranges'] == 'main' and c['k'] > k1]
        elif part == 'rest':
            cfgs = [c for c in cfgs if not (c['ranges'] == 'main' and c['k'] > k1)]
        for c in cfgs:
            if c['var'] in QL.BIAS_VARS:
                c['bias'] = f"{c['ranges']}_k{c['k']}"
        return cfgs

    def _e13_done(self, path, cfgs):
        import e13_diag as E13
        d = jload(path)
        if d is None:
            return False
        return all(E13.cfg_key(c) in d['sets'].get(s, {}) for c in cfgs for s in ('it0', 'last'))

    # ---------------------------------------------------------------------------------------------- runners
    def do_pairs(self, task, split):
        ctx = self.get_ctx(task)
        if split == 'eval6':                  # D1 exception: regenerate E1's eval6 pairs; must match the E1 cache json
            e1 = jload(f'{CODE}/inputs/pairs_pusht_eval6.json')
            pairs = ctx.get_pairs('eval6')            # no npz cache yet -> regenerated (json rewritten at the end)
            match = [list(x) for x in pairs['traj']] == [list(x) for x in e1['traj']]
            L.jdump(dict(task=task, match=bool(match), n=len(pairs['traj']), time=now()), f'{CHECKS}/eval6_regen_{task}.json')
            return
        pairs = ctx.get_pairs(split)
        if split == 'eval8':
            import make_eval8
            make_eval8.bookkeeping(ctx, pairs, f'{SPLITS_DIR}/eval8_{task}.json',
                                   max_repro=(self.decisions.get('eval8_pusht') or {}).get('max_repro_shared_per_pair', 20))

    def do_precheck(self, task):
        import make_eval8
        make_eval8.precheck(self.get_ctx(task), f'{CHECKS}/eval8_pusht_precheck.json',
                            self.decisions['eval8_pusht']['expected_free_offsets'])

    def do_pcrec(self, task):
        import calib_tmlr as C
        C.record_calib(self.get_ctx(task))

    def do_bias(self, task, rid, k):
        import calib_tmlr as C
        out = C.compute_bias(self.get_ctx(task), rid, k, rule=self.target_rule())
        with open(f'{CHECKS}/bias_resid.jsonl', 'a') as f:
            f.write(json.dumps(dict(task=task, ranges=rid, k=k, rule=out['rule_name'], converged=out['converged'],
                                    stage_eta=out['stage_eta'], updates=out['updates'],
                                    final=out['final_mean_abs_resid_over_delta'],
                                    stages=[(s['eta'], s['updates'], s['converged'], s['worst_final']) for s in out['stages']],
                                    passed=out['check_resid_lt_1pct'], time=now())) + '\n')
        if out.get('bc_unconverged'):                       # D2: these cells are run anyway and reported
            path = f'{CHECKS}/bc_unconverged.json'
            lst = jload(path, [])
            lst.append(dict(task=task, ranges=rid, k=k, worst_final=max(out['final_mean_abs_resid_over_delta']),
                            stage_eta=out['stage_eta'], time=now()))
            L.jdump(lst, path)

    def do_rc(self, task, k):
        import calib_tmlr as C
        C.select_rc(self.get_ctx(task), k)

    def do_e13(self, task, split, e, path, cfgs):
        import e13_diag as E13
        ctx = self.get_ctx(task)
        self.finalize_all()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if split == 'held':
            nj = [j for j in self.none_jobs(task, split='held', n=PP.N_HELD) if j['ep0'] <= e < j['ep1']][0]
        else:
            nj = [j for j in self.none_jobs(task) if j['ep0'] <= e < j['ep1']][0]
        pops = ctx.pops
        a0 = torch.as_tensor(np.load(f'{pops}/{split}_it0_e{e}.npz')['actions'])
        aL = torch.as_tensor(np.load(f"{pops}/{nj['id']}_e{e}.npz")['last_actions'])
        sets = {'it0': (a0, 0), 'last': (aL, ctx.opt_steps - 1)}

        def set_cfg(c):
            ctx.set_ranges(c['ranges'])
            if c.get('bias'):
                ctx.set_bias(c['bias'], c['ranges'], c['k'])
        E13.run_state(ctx, split, e, sets, cfgs, path, set_cfg)

    def run_job(self, it):
        j = it.job
        ctx = self.get_ctx(j['task'])
        res_dir = f"{JOBS}/{j['task']}"
        os.makedirs(res_dir, exist_ok=True)
        pool = self.get_pool() if j['diag'] else None
        t0 = time.time()
        p = RN.run_job(ctx, j, NAME, pool, res_dir)
        p.meta['tag'] = self.tag
        p.meta['wall_s'] = time.time() - t0
        self.pending.append(p)
        self.finalize_ready()
        with open(f'{STATE}/timing.jsonl', 'a') as f:
            f.write(json.dumps(dict(task=j['task'], id=j['id'], var=j['var'], diag=j['diag'], eps=it.eps,
                                    s=time.time() - t0, time=now())) + '\n')

    # ---------------------------------------------------------------------------------------------- status
    def write_status(self, items, phase):
        done_eps = sum(it.eps for it in items if it.kind == 'job' and self.is_done(it))
        tot_eps = sum(it.eps for it in items if it.kind == 'job')
        nfail = sum(1 for k, v in self.fails.items() if v >= MAX_ATTEMPTS)
        rem = [it for it in items if not self.is_done(it) and not self.failed(it.key)]
        eta = '-'
        T = [json.loads(x) for x in open(f'{STATE}/timing.jsonl')] if os.path.exists(f'{STATE}/timing.jsonl') else []
        if T:
            spe = {}
            for t in T:
                spe.setdefault(t['task'], []).append(t['s'] / max(1, t['eps']))
            sec = sum(it.eps * float(np.mean(spe.get(it.task, [15.0])[-50:])) for it in rem if it.kind == 'job')
            sec += sum(60.0 for it in rem if it.kind == 'e13') + sum(120.0 for it in rem if it.kind == 'calib')
            eta = time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime(time.time() + sec))
        st = dict(tag=self.tag, phase=phase, done=done_eps, total=tot_eps, failed=nfail, eta=eta, updated=now(),
                  note=self.note[:120], current=self.current, pid=os.getpid(), host=socket.gethostname(),
                  remaining_items=len(rem), e13_left=sum(1 for it in rem if it.kind == 'e13'))
        L.jdump(st, f'{STATE}/status.json')

    # ---------------------------------------------------------------------------------------------- main loop
    def loop(self):
        open(f'{ATM}/logs/driver.pid', 'w').write(str(os.getpid()))
        if os.path.exists(f'{STATE}/FINISHED'):
            os.remove(f'{STATE}/FINISHED')
        waited = 0
        if os.path.exists(f'{STATE}/RESTART'):
            os.remove(f'{STATE}/RESTART')
        while True:
            if os.path.exists(f'{STATE}/RESTART'):        # graceful restart (new code): finish in-flight work, exit 4
                self.finalize_all()
                print(f'{now()} DRIVER_RESTART requested', flush=True)
                return 4
            items = self.build()
            if self.only == 'calib':
                items = [it for it in items if it.kind in ('calib', 'wait')]
            pend = [it for it in items if not self.is_done(it) and not self.failed(it.key)]
            producers = {f: c.key for c in items if c.kind == 'calib' and hasattr(c, 'produces') for f in c.produces}
            runnable, blocked, queued = [], [], 0
            for it in pend:
                why = it.blocked_by() if hasattr(it, 'blocked_by') else None
                if why is None and it.kind == 'wait':
                    why = 'waiting copied calibration file ' + os.path.basename(it.produces[0])
                if why is None:
                    miss = [f for f in it.needs if not os.path.exists(f)]
                    if miss:
                        why = 'missing ' + os.path.basename(miss[0])
                        dep = producers.get(miss[0])
                        if dep and self.failed(dep):          # its producer failed 3 times: do not wait for ever
                            self.add_fail(it.key, f'dependency failed: {dep}')
                            self.fails[it.key] = MAX_ATTEMPTS
                            L.jdump(self.fails, f'{STATE}/fails.json')
                            continue
                        if dep:                               # produced by a calibration item later in the queue
                            queued += 1
                            continue
                if why is None and it.kind == 'job' and it.job['split'] == 'eval8' and self.eval8_blocked(it.task):
                    why = 'eval8 PushT decision pending'
                if why is None and it.kind == 'job' and it.job['split'] == 'eval8':
                    why = self.split_violation(it.task)
                if why is None and it.kind == 'job' and it.job['split'] == 'eval6':
                    rg = jload(f'{CHECKS}/eval6_regen_{it.task}.json')
                    why = None if (rg and rg.get('match')) else 'eval6 pairs regeneration not verified'
                if why is None and it.kind == 'job' and it.job['var'] in QL.BIAS_VARS:
                    if self.bias_rule is None:
                        why = 'det+BC/sh-iter+b: b rule pending'
                    elif not self.bias_ok(RN.bias_path(it.task, it.job['bias'])):
                        why = 'bias file of the decided rule missing: ' + it.job['bias']
                (blocked if why else runnable).append((it, why))
            phase = 'smoke' if any(it.kind in ('calib', 'wait') for it, _ in blocked) else 'main'
            if self.only == 'calib' and (not pend or not any(
                    (it.blocked_by() if hasattr(it, 'blocked_by') else None) is None and it.kind == 'calib' and
                    all(os.path.exists(f) for f in it.needs) for it in pend)):
                self.finalize_all()
                print(f'{now()} CALIB_ONLY_DONE pending={[it.key for it in pend]}', flush=True)
                return 0
            if not pend:
                self.finalize_all()
                self.note = 'all items done' + (f'; {sum(1 for v in self.fails.values() if v >= MAX_ATTEMPTS)} failed'
                                                if any(v >= MAX_ATTEMPTS for v in self.fails.values()) else '')
                self.current = '-'
                self.write_status(items, 'done')
                open(f'{STATE}/FINISHED', 'w').write(now())
                print(f'{now()} DRIVER_FINISHED', flush=True)
                return 0
            if not runnable:
                self.finalize_all()
                self.current = '-'
                self.note = 'blocked: ' + '; '.join(sorted({w for _, w in blocked}))[:110]
                self.write_status(items, phase)
                time.sleep(60)
                waited += 60
                if waited >= 6 * 3600:
                    return 10
                continue
            waited = 0
            it = runnable[0][0]
            phase = {'calib': 'smoke', 'wait': 'smoke', 'job': 'main', 'e13': 'analysis'}[it.kind]
            self.current = it.key
            self.note = (it.note or it.kind) + (f'; blocked: {blocked[0][1]}' if blocked else '') + (
                f'; {queued} jobs wait for queued calib' if queued else '')
            self.write_status(items, phase)
            print(f'{now()} RUN {it.kind} {it.key}', flush=True)
            try:
                if it.kind == 'job':
                    self.run_job(it)
                else:
                    it.run()
                    if it.kind == 'calib':
                        self.finalize_ready()
            except Exception:
                err = traceback.format_exc()
                print(err, flush=True)
                self.add_fail(it.key, err)
                try:
                    self.finalize_all()
                except Exception:
                    print(traceback.format_exc(), flush=True)
                self.note = f'FAILED {it.key} (attempt {self.fails[it.key]})'
                self.write_status(items, phase)
                return 3


def main():
    torch.backends.cuda.matmul.allow_tf32 = True          # E1 numerics: TF32 matmul, SDPA attention, bf16 autocast
    torch.set_num_threads(int(os.environ.get('RUNNER_THREADS', '4')))
    L.patch_attention(True)
    d = Driver(only=sys.argv[sys.argv.index('--only') + 1] if '--only' in sys.argv else None)
    rc = d.loop()
    if d.pool is not None:
        d.pool.close(); d.pool.join()
    sys.exit(rc)


if __name__ == '__main__':
    main()
