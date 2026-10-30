"""E4 (paper A): frozen simulation grid, calibration, resumable workers.

usage (server, one thread per process):
  python e4_jobs.py tables  <out.json>                      selection-intensity tables (e4_formulas.build_tables)
  python e4_jobs.py calib   <res_dir>                       error-scale calibration -> <res_dir>/calib.json
  python e4_jobs.py manifest <res_dir> [--smoke]            job list -> <res_dir>/manifest.json
  python e4_jobs.py work <res_dir> <worker> <n_workers>     run this worker's jobs (skips finished ones)
Every job writes <res_dir>/jobs/<job id>.json (atomic rename); an existing file means done.
"""
import json
import math
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e4_formulas as F  # noqa: E402
import e4_sim as S  # noqa: E402

MASTER = 20261022
# ------------------------------------------------------------------------------------------------ frozen settings
MPPI_BASE = dict(npi=24, iters=12, temp=0.5, min_std=0.05, max_std=2.0, pi_std=0.2)
CEM_BASE = dict(iters=30, var_scale=1.0)
REC = {'mppi': (1, 3, 6, 12), 'cem': (5, 10, 30)}
DEF = {'mppi': dict(N=512, k=64), 'cem': dict(N=300, k=30)}
SHAPE = {2: (1, 2), 6: (3, 2), 12: (3, 4)}            # decision dim -> (horizon steps, action dim)
NS = (64, 128, 256, 512, 1024)
ELITE_RULES = ('q02', 'q125', 'q25', 'kfix')
Q = {'q02': 0.02, 'q125': 0.125, 'q25': 0.25}
KFIX = {'mppi': 64, 'cem': 30}
ELL_DEF, RHO_C_DEF = 0.3, 0.5
SCALES = (0.5, 2.0, 8.0, 32.0, 128.0)    # error scale = it0 across-candidate error std / it0 near-tie gap g0
SCALES_MID = (8.0, 32.0)
N_INST, INST_CHUNK = 128, 32
MAIN_MODELS = ('M1', 'M2', 'M3', 'R-det', 'R-shiter', 'R-fixed', 'R-detshift')
TOY_MAIN = dict(n_tok=1, n_unit=512, nu=3.0, gamma=1.0)
TOY_NEFF = {'coh': dict(n_tok=16, n_unit=256, nu=3.0, gamma=1.0, gamma_tok=0.0, kappa_s=1.0, eta_spread=0.0),
            'incoh': dict(n_tok=16, n_unit=256, nu=3.0, gamma=0.0, gamma_tok=1.0, kappa_s=0.0, eta_spread=1.0)}
NEFF_LEVELS = {4096: (1, 1), 1024: (4, 1), 256: (16, 1), 64: (16, 4), 16: (16, 16), 4: (16, 64), 1: (16, 256)}
TOY_RVAR = {'inc': dict(n_tok=1, n_unit=512, nu=3.0, gamma=1.0, phase_conc=1.0, kappa_w=0.0),
            'clus': dict(n_tok=1, n_unit=512, nu=3.0, gamma=1.0, phase_conc=0.5, kappa_w=0.5)}
NL_STRUCT = dict(n_unit=256, m=8, sx=0.3)
NL_BITS = (3, 4, 5, 6)
NL_KAPPA = (1.0, 10.0)
ELLS = (0.03, 0.1, 0.3, 1.0, 3.0)
RHO_CS = (0.0, 0.5, 0.9, 0.99)
CAL_INST = 64


def code(s):
    return zlib.crc32(str(s).encode()) & 0x7fffffff


def rng_of(*parts):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy=MASTER, spawn_key=tuple(code(p) for p in parts))))


def elite_k(planner, N, rule):
    if rule == 'kfix':
        return min(KFIX[planner], N)
    if rule == 'def':
        return DEF[planner]['k']
    return max(2, int(round(Q[rule] * N)))


def planner_cfg(planner, d, N, k, iters=None):
    H, A = SHAPE[d]
    if planner == 'mppi':
        c = dict(MPPI_BASE, N=N, k=k, A=A)
    else:
        c = dict(CEM_BASE, N=N, k=k, A=A)
    if iters is not None:
        c['iters'] = iters
    return c


# ------------------------------------------------------------------------------------------------ instances and models
def make_instance(block, planner, regime, d, inst):
    r = rng_of('inst', block, planner, regime, d, inst)
    prob = S.Quadratic(r, d, regime)
    prior_center = np.clip(prob.a_star + 0.3 * r.standard_normal(d), -1, 1)
    return prob, prior_center


def model_key(m):
    return m['key']


def build_struct(m, block, planner, regime, d, inst, prob):
    r = rng_of('struct', block, planner, regime, d, inst, m.get('struct_tag', m['key']))
    kind = m['kind']
    if kind in ('M1',):
        return S.RFF(r, d, m.get('ell', ELL_DEF))
    if kind == 'R':
        return S.RoundToy.make_struct(r, d, **m['toy'])
    if kind == 'NL':
        return S.NLToy.make_struct(r, d, **m['toy'])
    if kind == 'M1j':
        return r
    return None


def make_err(m, struct, amp, prob):
    kind = m['kind']
    if kind == 'none':
        return S.NoErr()
    if kind == 'M1':
        return S.PersistentField(struct, amp)
    if kind == 'M2':
        return S.WhiteNoise(amp)
    if kind == 'M3':
        return S.FreshField(amp, m.get('ell', ELL_DEF), m.get('rho_c', RHO_C_DEF), prob.d)
    if kind == 'R':
        tb, ub = m.get('blocks', (1, 1))
        return S.RoundToy(struct, m['variant'], amp, tok_block=tb, unit_block=ub)
    if kind == 'NL':
        return S.NLToy(struct, m['variant'], amp, m['kappa'], m['bits'], prob.a_star)
    raise ValueError(kind)


def model_specs(names, **kw):
    out = []
    for n in names:
        if n == 'M1':
            out.append(dict(key='M1', kind='M1', ell=ELL_DEF))
        elif n == 'M2':
            out.append(dict(key='M2', kind='M2'))
        elif n == 'M3':
            out.append(dict(key='M3', kind='M3', ell=ELL_DEF, rho_c=RHO_C_DEF))
        elif n.startswith('R-'):
            v = n[2:]
            out.append(dict(key=n, kind='R', variant=v, toy=TOY_MAIN, struct_tag='toy_main', blocks=(1, 1)))
        else:
            raise ValueError(n)
    return out


# ------------------------------------------------------------------------------------------------ calibration
def it0_population(planner, prob, cfg, rng):
    """Iteration-0 sampled candidates exactly as the planner draws them (t0)."""
    d = prob.d
    if planner == 'mppi':
        return np.clip(cfg['max_std'] * rng.standard_normal((cfg['N'] - cfg['npi'], d)), -1, 1)
    A = rng.standard_normal((cfg['N'], d)) * cfg['var_scale']
    return A[1:]


def calib_gap(planner, regime, d, n=CAL_INST):
    cfg = planner_cfg(planner, d, DEF[planner]['N'], DEF[planner]['k'])
    gaps, sJ = [], []
    for i in range(n):
        prob, _ = make_instance('cal', planner, regime, d, i)
        A = it0_population(planner, prob, cfg, rng_of('calpop', planner, regime, d, i))
        J = prob.J(A)
        gaps.append(S._gap(J))
        sJ.append(J.std())
    return float(np.median(gaps)), float(np.median(sJ))


def calib_unit(planner, regime, d, m, n=CAL_INST, block='cal'):
    """RMS over calibration instances of the it0 across-candidate error std at amp = 1 (one draw per instance)."""
    cfg = planner_cfg(planner, d, DEF[planner]['N'], DEF[planner]['k'])
    vals = []
    for i in range(n):
        prob, _ = make_instance('cal', planner, regime, d, i)
        st = build_struct(m, 'cal', planner, regime, d, i, prob)
        err = make_err(m, st, 1.0, prob)
        A = it0_population(planner, prob, cfg, rng_of('calpop', planner, regime, d, i))
        err.begin_plan(rng_of('calerr', planner, regime, d, i, m['key']))
        err.begin_iter()
        vals.append(err(A).std())
    return float(np.sqrt(np.mean(np.square(vals))))


def calib_all(res):
    """Gap and unit amplitudes for every (planner, regime, d, model) used by the grid."""
    out = dict(gap={}, sJ={}, unit={}, t=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    combos = [(p, r, d) for p in ('mppi', 'cem') for r in ('sep', 'tie') for d in (2, 6, 12)]
    for p, r, d in combos:
        g, s = calib_gap(p, r, d)
        out['gap'][f'{p}|{r}|{d}'] = g
        out['sJ'][f'{p}|{r}|{d}'] = s
    specs = {}
    for m in model_specs(MAIN_MODELS):
        specs[m['key']] = m
    for ell in ELLS:
        specs[f'M1@{ell}'] = dict(key=f'M1@{ell}', kind='M1', ell=ell, struct_tag=f'M1@{ell}')
        specs[f'M3@{ell}'] = dict(key=f'M3@{ell}', kind='M3', ell=ell, rho_c=RHO_C_DEF)
    for coh, toy in TOY_NEFF.items():
        specs[f'E-det@{coh}'] = dict(key=f'E-det@{coh}', kind='R', variant='det', toy=toy, struct_tag=f'toy_neff_{coh}')
    for coh, toy in TOY_RVAR.items():
        specs[f'RV-det@{coh}'] = dict(key=f'RV-det@{coh}', kind='R', variant='det', toy=toy, struct_tag=f'toy_rvar_{coh}')
    specs['NL-det@k4'] = dict(key='NL-det@k4', kind='NL', variant='det', toy=NL_STRUCT, kappa=1.0, bits=4,
                              struct_tag='toy_nl')
    for p, r, d in combos:
        for key, m in specs.items():
            if d != 6 and key not in MAIN_MODELS:
                continue
            out['unit'][f'{p}|{r}|{d}|{key}'] = calib_unit(p, r, d, m)
    json.dump(out, open(os.path.join(res, 'calib.json'), 'w'), indent=1)
    return out


def amp_for(cal, planner, regime, d, key, scale):
    g = cal['gap'][f'{planner}|{regime}|{d}']
    u = cal['unit'][f'{planner}|{regime}|{d}|{key}']
    return scale * g / u


# ------------------------------------------------------------------------------------------------ manifest
def manifest(res, smoke=False):
    jobs = []
    n_inst = 4 if smoke else N_INST
    chunk = 4 if smoke else INST_CHUNK

    def add(block, planner, regime, d, N, rule, models, scales, extra=None, iters=None, rec=None, k=None):
        k = elite_k(planner, N, rule) if k is None else int(k)
        for i0 in range(0, n_inst, chunk):
            j = dict(block=block, planner=planner, regime=regime, d=d, N=N, rule=rule, k=k, models=models,
                     scales=list(scales), i0=i0, i1=min(n_inst, i0 + chunk), iters=iters, rec=rec)
            if extra:
                j.update(extra)
            j['id'] = f"{block}_{planner}_{regime}_d{d}_N{N}_{rule}_{j.get('tag', 'x')}_i{i0}"
            jobs.append(j)

    main_models = model_specs(MAIN_MODELS)
    Ns = (64, 512) if smoke else NS
    rules = ('q125', 'kfix') if smoke else ELITE_RULES
    for planner in ('mppi', 'cem'):
        for regime in ('sep', 'tie'):
            # block M: main factorial (d = 6)
            for N in Ns:
                for rule in rules:
                    add('M', planner, regime, 6, N, rule, main_models, SCALES)
            # block D: dimension at the planner default (CEM default N = 300 also at d = 6)
            for d in ((2, 12) if planner == 'mppi' else (2, 6, 12)):
                add('D', planner, regime, d, DEF[planner]['N'], 'def', main_models, SCALES_MID)
            # block L: field correlation length
            mods = []
            for ell in ELLS:
                if ell == ELL_DEF:
                    continue
                mods.append(dict(key=f'M1@{ell}', kind='M1', ell=ell, struct_tag=f'M1@{ell}'))
                mods.append(dict(key=f'M3@{ell}', kind='M3', ell=ell, rho_c=RHO_C_DEF))
            add('L', planner, regime, 6, DEF[planner]['N'], 'def', mods, SCALES_MID)
            # block C: CRN at fixed marginal amplitude (amp = scale * g0, unit marginal variance)
            mods = [dict(key=f'M3rho@{rc}', kind='M3', ell=ELL_DEF, rho_c=rc, amp_mode='marginal') for rc in RHO_CS]
            mods.append(dict(key='M2marg', kind='M2', amp_mode='marginal'))
            add('C', planner, regime, 6, DEF[planner]['N'], 'def', mods, SCALES_MID)
            # block E: dither-table size (same bits: amplitude of the toy's det at each scale)
            for coh, toy in TOY_NEFF.items():
                mods = [dict(key=f'E-{v}@{coh}', kind='R', variant=v, toy=toy, struct_tag=f'toy_neff_{coh}',
                             blocks=(1, 1), amp_ref=f'E-det@{coh}') for v in ('det', 'ind', 'shiter', 'detshift')]
                for ne, bl in NEFF_LEVELS.items():
                    mods.append(dict(key=f'E-fixed{ne}@{coh}', kind='R', variant='fixed', toy=toy,
                                     struct_tag=f'toy_neff_{coh}', blocks=bl, n_eff=ne, amp_ref=f'E-det@{coh}'))
                add('E', planner, regime, 6, DEF[planner]['N'], 'def', mods, SCALES_MID, extra=dict(tag=coh))
            # block R: rounding variants at the same bits, incoherent vs clustered features
            for coh, toy in TOY_RVAR.items():
                mods = [dict(key=f'RV-{v}@{coh}', kind='R', variant=v, toy=toy, struct_tag=f'toy_rvar_{coh}',
                             blocks=(1, 1), amp_ref=f'RV-det@{coh}') for v in ('det', 'ind', 'shiter', 'fixed', 'detshift')]
                add('R', planner, regime, 6, DEF[planner]['N'], 'def', mods, SCALES_MID, extra=dict(tag=coh))
            # block J: Jensen (fresh dither through a nonlinear head), bits 3-6, same network amplitude
            mods = []
            for b in NL_BITS:
                for kap in NL_KAPPA:
                    for v in ('det', 'ind', 'shiter', 'detshift', 'fixed'):
                        mods.append(dict(key=f'NL-{v}@k{b}@kap{kap:g}', kind='NL', variant=v, toy=NL_STRUCT, kappa=kap,
                                         bits=b, struct_tag='toy_nl', amp_ref='NL-det@k4'))
            add('J', planner, regime, 6, DEF[planner]['N'], 'def', mods, (8.0,))
        # block V: long runs (validation of the SA / SAA limits), sep only
        mods = model_specs(('M1', 'M2'))
        if planner == 'mppi':
            for k in (8, 32, 64, 128):
                add('V', planner, 'sep', 6, 512, f'k{k}', mods, SCALES_MID, extra=dict(tag=f'k{k}'), k=k,
                    iters=100, rec=(1, 3, 6, 12, 25, 50, 100))
        else:
            add('V', planner, 'sep', 6, 300, 'def', mods, SCALES_MID, iters=60, rec=(5, 10, 30, 60))
    # block A: single-shot order statistics; block W: closed loop; block EV / JV: direct field measurements
    jobs.append(dict(block='A', id='A_singleshot', smoke=smoke))
    for ws in (1, 0):
        for n_it in (1, 3, 6, 12):
            jobs.append(dict(block='W', id=f'W_ws{ws}_n{n_it}', ws=ws, n_it=n_it, n_ep=8 if smoke else 64,
                             scales=list(SCALES_MID)))
    for coh in TOY_NEFF:
        jobs.append(dict(block='EV', id=f'EV_{coh}', coh=coh, n_struct=2 if smoke else 16, n_tab=4 if smoke else 32))
    jobs.append(dict(block='JV', id='JV', n_struct=2 if smoke else 16))
    os.makedirs(os.path.join(res, 'jobs'), exist_ok=True)
    json.dump(jobs, open(os.path.join(res, 'manifest.json'), 'w'), indent=0)
    return jobs


# ------------------------------------------------------------------------------------------------ accumulators
class Acc:
    def __init__(self):
        self.d = {}

    def add(self, name, x):
        if x is None:
            return
        x = float(x)
        if not math.isfinite(x):
            return
        s = self.d.setdefault(name, [0.0, 0.0, 0])
        s[0] += x
        s[1] += x * x
        s[2] += 1

    def addv(self, name, v):
        v = np.asarray(v, float)
        s = self.d.setdefault(name, [np.zeros_like(v), np.zeros_like(v), 0])
        s[0] = s[0] + v
        s[1] = s[1] + v * v
        s[2] += 1

    def out(self):
        o = {}
        for k, (a, b, n) in self.d.items():
            if n == 0:
                continue
            m = a / n
            var = np.maximum(b / n - m * m, 0) * n / max(n - 1, 1)
            if isinstance(m, np.ndarray):
                o[k] = dict(mean=m.tolist(), sd=np.sqrt(var).tolist(), n=n)
            else:
                o[k] = dict(mean=float(m), sd=float(math.sqrt(var)), n=n)
        return o


# ------------------------------------------------------------------------------------------------ planner jobs
def _amp(cal, job, m, scale):
    if m.get('amp_mode') == 'marginal':
        return scale * cal['gap'][f"{job['planner']}|{job['regime']}|{job['d']}"]
    key = m.get('amp_ref', m['key'])
    return amp_for(cal, job['planner'], job['regime'], job['d'], key, scale)


def run_planner_job(job, cal, tabs):
    planner, regime, d, N, k = job['planner'], job['regime'], job['d'], job['N'], job['k']
    cfg = planner_cfg(planner, d, N, k, iters=job.get('iters'))
    rec = tuple(job['rec']) if job.get('rec') else REC[planner]
    block = job['block']
    accs = {}
    pers = {}
    t0 = time.time()
    n_runs = 0

    def acc(mk, sc, n):
        return accs.setdefault(f'{mk}|{sc:g}|{n}', Acc())

    def plan(prob, err, prior, slot, tag):
        rp = rng_of('plan', block, planner, regime, d, job['i0'], prob._inst, N, k, slot)
        re = rng_of('err', block, planner, regime, d, prob._inst, N, k, slot, tag)
        ra = rng_of('aux', block, planner, regime, d, prob._inst, N, k, slot, tag)
        if planner == 'mppi':
            rk = rng_of('pick', block, planner, regime, d, prob._inst, N, k, slot)
            return S.mppi_plan(prob, err, cfg, rp, re, rk, ra, prior, rec=rec)
        return S.cem_plan(prob, err, cfg, rp, re, ra, rec=rec)

    for inst in range(job['i0'], job['i1']):
        prob, prior = make_instance(block, planner, regime, d, inst)
        prob._inst = inst
        structs = {}
        base = {}
        for slot in (0, 1):
            base[slot] = plan(prob, S.NoErr(), prior, slot, 'none')
            n_runs += 1
            for n, r in base[slot]['rec'].items():
                a = acc('none', 0, n)
                for kk in ('J_exec', 'J_mean', 'J_mf'):
                    if kk in r:
                        a.add(kk, r[kk])
                a.add('sJ', r['sJ'])
            a0 = acc('none', 0, 0)
            for kk, v in base[slot]['it0'].items():
                a0.add(kk, v)
        # seed floor of the first action and of the mean
        for n in rec:
            f2 = float(np.sum((base[1]['rec'][n]['a1'] - base[0]['rec'][n]['a1']) ** 2))
            m2 = float(np.sum((base[1]['rec'][n]['mean'] - base[0]['rec'][n]['mean']) ** 2))
            acc('none', 0, n).add('floor_a1', f2)
            acc('none', 0, n).add('floor_mean', m2)
            acc('none', 0, n).add('dist_mean_astar', float(np.sum((base[0]['rec'][n]['mean'] - prob.a_star) ** 2)))
        for m in job['models']:
            mk = m['key']
            if mk not in structs:
                structs[mk] = build_struct(m, block, planner, regime, d, inst, prob)
            for sc in job['scales']:
                amp = _amp(cal, job, m, sc)
                runs = {}
                for slot in (0, 1):
                    err = make_err(m, structs[mk], amp, prob)
                    out = plan(prob, err, prior, slot, mk)
                    n_runs += 1
                    runs[slot] = out
                    b = base[slot]
                    a0 = acc(mk, sc, 0)
                    for kk, v in out['it0'].items():
                        a0.add(kk, v)
                    a0.add('amp', amp)
                    a0.addv('traj_sJ', [t['sJ'] for t in out['traj']])
                    a0.addv('traj_se', [t['se'] for t in out['traj']])
                    a0.addv('traj_std', [t['std'] for t in out['traj']])
                    a0.addv('traj_drift', [t['e_rand'] - out['it0']['e0'] for t in out['traj']])
                    a0.addv('traj_rho', [t['rho'] for t in out['traj']])
                    if planner == 'mppi':
                        a0.addv('traj_theta', [t['theta'] for t in out['traj']])
                    # SAA prediction (persistent random field only)
                    saa = None
                    if m['kind'] == 'M1':
                        saa = S.saa_track(prob, structs[mk], amp, out['sig'], box=1.0 if planner == 'mppi' else None)
                    for n, r in out['rec'].items():
                        a = acc(mk, sc, n)
                        rb = b['rec'][n]
                        for kk, v in r.items():
                            if isinstance(v, (float, int)) and kk not in ('N', 'k'):
                                a.add(kk, v)
                        if planner == 'mppi':
                            a.add('dmg_exec', r['J_exec'] - rb['J_exec'])
                            a.add('dmg_mean', r['J_mean'] - rb['J_mean'])
                            Sop = tabs.mppi(N, k, r['theta'])
                            a.add('S_op', Sop)
                            a.add('wc_pred', float(F.winners_curse(r['se'], r['sJ'], r['rho'], Sop)))
                        else:
                            a.add('dmg', r['J_mf'] - rb['J_mf'])
                            Se, Sa = tabs.topk(N, k), tabs.exp_max(N)
                            a.add('wc_pred_elite', float(F.winners_curse(r['se'], r['sJ'], r['rho'], Se)))
                            a.add('wc_pred_argmin', float(F.winners_curse(r['se'], r['sJ'], r['rho'], Sa)))
                        a.add('dist_mean_astar', float(np.sum((r['mean'] - prob.a_star) ** 2)))
                        if saa is not None:
                            mp = saa[n - 1]
                            sig_n = out['sig'][n - 1]
                            dv = r['mean'] - prob.a_star
                            dp = mp - prob.a_star
                            a.add('saa_dot', float(dv @ dp))
                            a.add('saa_pp', float(dp @ dp))
                            a.add('saa_vv', float(dv @ dv))
                            a.add('saa_res', float(np.sum((r['mean'] - mp) ** 2)))
                            a.add('saa_pred_regret', float(prob.J(mp[None])[0]))
                            bval = float(amp * structs[mk](mp[None])[0])
                            ctr = saa[n - 2] if n >= 2 else np.zeros(d)
                            bsm = float(amp * structs[mk].smoothed(ctr, sig_n)[0])
                            e0 = out['it0']['e0']
                            if planner == 'cem':
                                a.add('saa_pred_oei_mf', bval - e0)
                            a.add('saa_pred_drift', bsm - e0)
                # paired across slots: plan deviation, seed floor, rerun spread, persistence
                for n in rec:
                    r0, r1 = runs[0]['rec'][n], runs[1]['rec'][n]
                    b0, b1 = base[0]['rec'][n], base[1]['rec'][n]
                    a = acc(mk, sc, n)
                    a.add('dev_a1', float(np.sum((r0['a1'] - b0['a1']) ** 2)))
                    a.add('dev_a1', float(np.sum((r1['a1'] - b1['a1']) ** 2)))
                    a.add('spread_a1', float(np.sum((r1['a1'] - r0['a1']) ** 2)))
                    s0, s1 = r0['mean'] - b0['mean'], r1['mean'] - b1['mean']
                    a.add('pers_dot', float(s0 @ s1))
                    a.add('pers_n0', float(s0 @ s0))
                    a.add('pers_n1', float(s1 @ s1))
                    a.add('dev_mean', float(s0 @ s0))
                    a.add('dev_mean', float(s1 @ s1))
    res = {k: v.out() for k, v in accs.items()}
    return dict(job=job, cells=res, n_runs=n_runs, cpu_s=time.time() - t0)


# ------------------------------------------------------------------------------------------------ block A
def run_singleshot(job, tabs):
    """Order-statistics check of the winner's-curse formula: jointly normal (J, e) and the real iteration-0 population
    of the quadratic objective (non-normal J) with independent normal e."""
    rng = rng_of('A')
    reps = 2000 if job.get('smoke') else 20000
    rows = []
    ratios = (0.1, 0.3, 1.0, 3.0, 10.0)
    for N in (64, 128, 256, 512, 1024):
        for rho in (0.0, 0.3):
            for ratio in ratios:
                Z = rng.standard_normal((reps, N))
                X = rng.standard_normal((reps, N))
                J = Z
                e = ratio * (rho * Z + math.sqrt(1 - rho ** 2) * X)
                Jh = J + e
                o = np.argsort(Jh, axis=1)
                ops = {'argmin': (1, None)}
                for q in (0.02, 0.125, 0.25):
                    ops[f'top{q:g}'] = (max(2, int(round(q * N))), None)
                for th in (0.1, 1.0, 10.0):
                    ops[f'mppi_q0.125_th{th:g}'] = (max(2, int(round(0.125 * N))), th)
                for name, (kk, th) in ops.items():
                    idx = o[:, :kk]
                    es = np.take_along_axis(e, idx, 1)
                    if th is None:
                        meas = es.mean(1) - e.mean(1)
                        S_ = F.exp_max(N) if kk == 1 else tabs.topk(N, kk)
                    else:
                        js = np.take_along_axis(Jh, idx, 1)
                        sdh = Jh.std(1, keepdims=True)
                        w = np.exp(-th * (js - js[:, :1]) / sdh)
                        w /= w.sum(1, keepdims=True)
                        meas = (w * es).sum(1) - e.mean(1)
                        S_ = tabs.mppi(N, kk, th)
                    sig_e = ratio
                    pred = float(F.winners_curse(sig_e, 1.0, rho, S_))
                    rows.append(dict(dist='normal', N=N, rho=rho, ratio=ratio, op=name, k=kk, theta=th,
                                     meas=float(meas.mean()), meas_se=float(meas.std() / math.sqrt(reps)), pred=pred))
    # quadratic iteration-0 populations (MPPI and CEM), independent normal e
    for planner in ('mppi', 'cem'):
        for regime in ('sep', 'tie'):
            cfg = planner_cfg(planner, 6, DEF[planner]['N'], DEF[planner]['k'])
            nrep = 200 if job.get('smoke') else 2000
            for ratio in ratios:
                acc_m, acc_p = {}, {}
                acc_x, acc_t, acc_mx = [], [], []
                for i in range(nrep):
                    prob, _ = make_instance('A', planner, regime, 6, i % 256)
                    A = it0_population(planner, prob, cfg, rng)
                    J = prob.J(A)
                    sJ = J.std()
                    e = ratio * sJ * rng.standard_normal(len(J))
                    Jh = J + e
                    o = np.argsort(Jh)
                    Nn = len(J)
                    for name, kk in (('argmin', 1), ('top0.125', max(2, int(round(0.125 * Nn))))):
                        meas = e[o[:kk]].mean() - e.mean()
                        S_ = F.exp_max(Nn) if kk == 1 else F.topk_intensity(Nn, kk)
                        pred = float(F.winners_curse(e.std(), sJ, 0.0, S_))
                        acc_m.setdefault(name, []).append(meas / sJ)
                        acc_p.setdefault(name, []).append(pred / sJ)
                    if i < (20 if job.get('smoke') else 150):     # refined fixed-population predictions (argmin)
                        acc_x.append(F.argmin_cond_exact(J, ratio * sJ) / sJ)
                        acc_t.append(F.argmin_near_tie(J, ratio * sJ) / sJ)
                        acc_mx.append(acc_m['argmin'][-1])
                for name in acc_m:
                    mm, pp = np.array(acc_m[name]), np.array(acc_p[name])
                    row = dict(dist=f'quad_{planner}_{regime}', N=Nn, rho=0.0, ratio=ratio, op=name,
                               meas=float(mm.mean()), meas_se=float(mm.std() / math.sqrt(len(mm))),
                               pred=float(pp.mean()))
                    if name == 'argmin':
                        row.update(pred_cond_exact=float(np.mean(acc_x)), pred_near_tie=float(np.mean(acc_t)),
                                   meas_sub=float(np.mean(acc_mx)), meas_sub_se=float(np.std(acc_mx) / math.sqrt(len(acc_mx))),
                                   n_sub=len(acc_x))
                    rows.append(row)
    return dict(job=job, rows=rows)


# ------------------------------------------------------------------------------------------------ block EV (Var_cand vs N_eff)
def exact_varcand_levels(struct, A, levels, amp=1.0, P=48):
    """Exact E_u Var_cand of fixed shared-offset tables (report eq. 2.5.3), for several sharing patterns at once:
       (amp/K)^2 sum_g sum_p Var_a[Z_gp(a) - S_g(a)] / (2 pi^2 p^2),  Z_gp = sum_{m in g} G_m e^{2 pi i p y_m},
       S_g = sum_{m in g} G_m (Var of a complex variable = E|X - EX|^2, over the population; ddof 0).
    levels: {name: (tok_block, unit_block)}. Truncated at P harmonics (tail < 1.5% of the incoherent part)."""
    ref = S.RoundToy(struct, 'fixed', amp)
    y, G = ref._yG(A)
    G = np.broadcast_to(G, y.shape)
    n, T, U = y.shape
    K = T * U
    tot = {nm: 0.0 for nm in levels}
    S0 = {}
    for nm, (tb, ub) in levels.items():
        S0[nm] = G.reshape(n, T // tb, tb, U // ub, ub).sum((2, 4))
    for p in range(1, P + 1):
        E = G * np.exp(2j * np.pi * p * y)
        for nm, (tb, ub) in levels.items():
            D = E.reshape(n, T // tb, tb, U // ub, ub).sum((2, 4)) - S0[nm]
            tot[nm] += float(np.mean(np.abs(D - D.mean(0, keepdims=True)) ** 2, axis=0).sum()) / (2 * np.pi ** 2 * p ** 2)
    return {nm: (amp / K) ** 2 * v for nm, v in tot.items()}


def run_neff_direct(job):
    coh = job['coh']
    toy = TOY_NEFF[coh]
    rows = []
    spreads = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
    for si in range(job['n_struct']):
        rs = rng_of('EVs', coh, si)
        st = S.RoundToy.make_struct(rs, 6, **toy)
        center = rs.uniform(-0.5, 0.5, 6)
        for sp in spreads:
            A = center + sp * rs.standard_normal((256, 6))
            # det (no offsets)
            ed = S.RoundToy(st, 'det', 1.0)
            vdet = float(ed(A).var())
            # ind and shiter: one evaluation's across-candidate variance (fresh draws)
            ev = {}
            for v in ('ind', 'shiter', 'detshift'):
                er = S.RoundToy(st, v, 1.0)
                er.begin_plan(rng_of('EVu', coh, si, sp, v))
                vals = []
                for _ in range(job['n_tab']):
                    er.begin_iter()
                    vals.append(er(A).var())
                ev[v] = float(np.mean(vals))
            exact = exact_varcand_levels(st, A, NEFF_LEVELS)
            for ne, bl in NEFF_LEVELS.items():
                ef = S.RoundToy(st, 'fixed', 1.0, tok_block=bl[0], unit_block=bl[1])
                vals = []
                rr = rng_of('EVt', coh, si, sp, ne)
                for _ in range(job['n_tab']):
                    ef.begin_plan(rr)
                    vals.append(ef(A).var())
                ex = exact[ne]
                rows.append(dict(coh=coh, struct=si, spread=sp, n_eff=ne, var_meas=float(np.mean(vals)),
                                 var_meas_se=float(np.std(vals) / math.sqrt(len(vals))), var_exact=float(ex),
                                 var_det=vdet, var_ind=ev['ind'], var_shiter=ev['shiter'], var_detshift=ev['detshift'],
                                 sigma_y=float(toy['nu'] * sp)))
    return dict(job=job, rows=rows)


# ------------------------------------------------------------------------------------------------ block JV (Jensen field)
def run_jensen_direct(job, cal):
    rows = []
    amp = amp_for(cal, 'mppi', 'sep', 6, 'NL-det@k4', 8.0)
    for si in range(job['n_struct']):
        rs = rng_of('JVs', si)
        prob = S.Quadratic(rs, 6, 'sep')
        st = S.NLToy.make_struct(rs, 6, **NL_STRUCT)
        for sp in (1.0, 0.3, 0.1, 0.03):
            A = prob.a_star + sp * rs.standard_normal((256, 6))
            for b in NL_BITS:
                for kap in NL_KAPPA:
                    det = S.NLToy(st, 'det', amp, kap, b, prob.a_star)
                    edet = det(A)
                    row = dict(struct=si, spread=sp, bits=b, kappa=kap, sd_det=float(edet.std()),
                               mean_det=float(edet.mean()))
                    for v in ('ind', 'shiter', 'detshift'):
                        m = S.NLToy(st, v, amp, kap, b, prob.a_star)
                        bias = m.bias_exact(A)
                        m.begin_plan(rng_of('JVu', si, sp, b, kap, v))
                        m.begin_iter()
                        e1 = m(A)
                        row[f'sd_bias_{v}'] = float(bias.std())
                        row[f'mean_bias_{v}'] = float(bias.mean())
                        row[f'sd_draw_{v}'] = float(e1.std())
                        # correlation of the bias field with J (does the planner gain by following it?)
                        row[f'corr_bias_J_{v}'] = float(np.corrcoef(bias, prob.J(A))[0, 1]) if bias.std() > 0 else 0.0
                    rows.append(row)
    return dict(job=job, rows=rows, amp=amp)


# ------------------------------------------------------------------------------------------------ block W (closed loop)
def run_loop(job, cal):
    ws, n_it = job['ws'], job['n_it']
    sysm = S.LQLoop()
    d = sysm.d
    cfg = dict(MPPI_BASE, N=512, k=64, A=2, iters=n_it)
    # calibration of the loop's error scale: gap of the iteration-0 population at random states
    gaps = []
    rc = rng_of('Wcal')
    for i in range(64):
        x0 = rc.uniform(-2, 2, 2)
        q = sysm.objective(x0)
        A = np.clip(2.0 * rc.standard_normal((488, d)), -1, 1)
        gaps.append(S._gap(q.J(A)))
    g0 = float(np.median(gaps))
    models = ['none', 'M1j', 'Mstep', 'M3', 'M2']
    acc, rets = {}, {}
    t0 = time.time()
    # unit amplitudes (it0 across-candidate std at amp = 1)
    unit = {}
    for mk in models[1:]:
        vals = []
        for i in range(32):
            x0 = rc.uniform(-2, 2, 2)
            A = np.clip(2.0 * rc.standard_normal((488, d)), -1, 1)
            if mk in ('M1j', 'Mstep'):
                e = S.JointField(rng_of('Wcs', i), 2, d, ELL_DEF, 1.0)
                e.set_state(x0)
                vals.append(e(A).std())
            elif mk == 'M3':
                e = S.FreshField(1.0, ELL_DEF, RHO_C_DEF, d)
                e.begin_plan(rng_of('Wce', i))
                e.begin_iter()
                vals.append(e(A).std())
            else:
                vals.append(1.0)
        unit[mk] = float(np.sqrt(np.mean(np.square(vals))))
    for ep in range(job['n_ep']):
        x0 = rng_of('Wx0', ep).uniform(-2, 2, 2)
        for mk in models:
            for sc in (job['scales'] if mk != 'none' else [0.0]):
                amp = sc * g0 / unit.get(mk, 1.0)
                rs = rng_of('Wst', ep, mk)
                if mk == 'none':
                    err = S.NoErr()
                elif mk in ('M1j', 'Mstep'):
                    err = S.JointField(rs, 2, d, ELL_DEF, amp, per_step=(mk == 'Mstep'))
                elif mk == 'M3':
                    err = S.FreshField(amp, ELL_DEF, RHO_C_DEF, d)
                else:
                    err = S.WhiteNoise(amp)
                x = x0.copy()
                prev = np.zeros(d)
                ret = 0.0
                oei = []
                rp = rng_of('Wplan', ep, ws, n_it)
                re = rng_of('Werr', ep, ws, n_it, mk)
                rk = rng_of('Wpick', ep, ws, n_it)
                ra = rng_of('Waux', ep, ws, n_it, mk)
                rstep = rng_of('Wstep', ep, mk)
                for t in range(25):
                    q = sysm.objective(x)
                    if mk in ('M1j', 'Mstep'):
                        err.set_state(x, rstep if mk == 'Mstep' else None)
                    mean0 = None
                    if ws and t > 0:
                        mean0 = np.concatenate([prev[2:], np.zeros(2)])
                    pc = np.clip(q.a_star + 0.3 * rp.standard_normal(d), -1, 1)
                    out = S.mppi_plan(q, err, cfg, rp, re, rk, ra, pc, rec=(n_it,), mean0=mean0, keep_traj=False)
                    r = out['rec'][n_it]
                    oei.append(r['oei'])
                    prev = r['mean']
                    u = r['a_exec'][:2]
                    x, c = sysm.step(x, u)
                    ret += c
                key = f'{mk}|{sc:g}'
                acc.setdefault(key, Acc())
                acc[key].add('ret', ret)
                acc[key].add('oei', float(np.mean(oei)))
                rets.setdefault(key, []).append(ret)
    out = {kk: v.out() for kk, v in acc.items()}
    return dict(job=job, cells=out, rets=rets, g0=g0, unit=unit, cpu_s=time.time() - t0)


# ------------------------------------------------------------------------------------------------ driver
def job_cost(j):
    """Rough relative CPU cost used only to start the heavy jobs first."""
    b = j['block']
    if b in ('A', 'EV', 'JV', 'W'):
        return {'A': 5e9, 'EV': 3e10, 'JV': 2e9, 'W': 5e9}[b] * (2 if b == 'W' and j['n_it'] >= 6 else 1)
    iters = j.get('iters') or (12 if j['planner'] == 'mppi' else 30)
    width = 0
    for m in j['models']:
        if m['kind'] == 'R':
            width += m['toy']['n_tok'] * m['toy']['n_unit']
        elif m['kind'] == 'NL':
            width += m['toy']['n_unit']
        else:
            width += 150
    return j['N'] * iters * (width + 50) * len(j['scales']) * (j['i1'] - j['i0'])


def work(res, w, nw):
    """Pull unfinished jobs in decreasing cost order; a job is claimed by creating <id>.claim (O_EXCL). The runner
    deletes stale claims before (re)starting the workers."""
    jobs = json.load(open(os.path.join(res, 'manifest.json')))
    cal = json.load(open(os.path.join(res, 'calib.json')))
    tabs = F.Tables(json.load(open(os.path.join(res, 's_tables.json'))))
    jd = os.path.join(res, 'jobs')
    os.makedirs(jd, exist_ok=True)
    order = sorted(range(len(jobs)), key=lambda i: -job_cost(jobs[i]))
    for i in order:
        j = jobs[i]
        path = os.path.join(jd, j['id'] + '.json')
        if os.path.exists(path):
            continue
        try:
            fd = os.open(path + '.claim', os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f'{os.getpid()} w{w}'.encode())
            os.close(fd)
        except FileExistsError:
            continue
        t = time.time()
        b = j['block']
        try:
            if b == 'A':
                out = run_singleshot(j, tabs)
            elif b == 'EV':
                out = run_neff_direct(j)
            elif b == 'JV':
                out = run_jensen_direct(j, cal)
            elif b == 'W':
                out = run_loop(j, cal)
            else:
                out = run_planner_job(j, cal, tabs)
        except Exception as ex:  # keep the other jobs going; the error is in the log and the claim is released
            import traceback
            print(f"{time.strftime('%H:%M:%S')} w{w} ERROR {j['id']}: {ex!r}", flush=True)
            traceback.print_exc()
            os.remove(path + '.claim')
            continue
        out['wall_s'] = time.time() - t
        out['host_t'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        tmp = path + f'.tmp{os.getpid()}'
        with open(tmp, 'w') as f:
            json.dump(out, f, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
        os.replace(tmp, path)
        os.remove(path + '.claim')
        print(f"{time.strftime('%H:%M:%S')} w{w} done {j['id']} {out['wall_s']:.1f}s", flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'tables':
        json.dump(F.build_tables(), open(sys.argv[2], 'w'))
    elif cmd == 'calib':
        calib_all(sys.argv[2])
    elif cmd == 'manifest':
        js = manifest(sys.argv[2], smoke='--smoke' in sys.argv)
        print(len(js), 'jobs')
    elif cmd == 'work':
        work(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        raise SystemExit(__doc__)
