"""Analysis of the DINO-WM part of paper A's TMLR extension (DESIGN_tmlr.md v1.0; written before any main-run data).
Pure python + numpy/scipy (+ matplotlib for one PNG); runs on a pod (WORKER_RULES 1).

usage: python analyze_tmlr.py <raw dir: holds <tag>/jobs/<task>/*.json, <tag>/e13/, <tag>/state/, <tag>/checks/>
                              [--out DIR]          (default <raw>/../analysis)
       python analyze_tmlr.py --fake <raw dir>     write a synthetic data set with the real file layout (dry run)

Statistics (DESIGN 0.3): D = success drop vs the none of the same pod (pod = base tag, dw.3b continues dw.3), same
pairs and planner seeds; paired bootstrap over pairs, 10,000 resamples, numpy RandomState(0), 95% percentile CI;
red_v = (D_det - D_v) / D_det (point estimate decides, CI reported); threshold comparisons with tolerance 1e-9.
  E11 / B2 / B3 (DINO-WM unit = PushT k=4, PointMaze k=3, Wall k=3 on eval8):
    damaged cell: one-sided bootstrap p(D_det <= 0) = (#{D* <= 0} + 1) / (B + 1) < 0.05 after Holm over the 3 cells
    per damaged cell: fixes(v) = red_v >= 0.5; d_bias = D(det) - D(det+BC); d_pers = D(det+shift-fixed) - D(det+shift)
      class bias        det+BC fixes
            persistence det+BC does not fix and the d_pers CI lower bound > 0
            resolution  det+BC does not fix, det+shift-fixed fixes and the d_pers CI contains 0
            mixed       otherwise
    unit reading = the class of more than half of the damaged cells, else mixed
    B2 holds iff the unit reading is bias or resolution; B3 (DINO-WM part) holds iff det+noise fixes at most half of the
    damaged cells. Descriptive: D(sh-plan) - D(sh-iter), D(sh-iter+b) - D(sh-iter), D(det+noise) - D(det),
    D(det+shift) - D(sh-iter), drift ratio (variant / det; drift = last_pop_dJ - it0_rand_dJ of the E1 OEI inputs),
    planning-time 1/3/5-step errors (E1 records) and E13 errors, Wall 3-bit vs E1's exploratory cell
  E12 (per task, ranges PT 'main', PC 'pc', RC 'rc_k<k>'; roundings det, det+BC, sh-iter; RC det, sh-iter):
    k*(range, rounding) = smallest measured k with D <= tau = 0.10 such that every larger measured k also has D <= tau
    G(range) = min(k*(det), k*(det+BC)) - k*(sh-iter) (det+shift is not an E12 rounding for DINO-WM); RC: k*(det) -
    k*(sh-iter). Descriptive: best configuration per task, RC vs PT/PC, bit-success curves (PNG). Two-stage cells (PushT,
    50 + 50) report CIs not adjusted for the interim look.
  Dg1 (DINO-WM unit): in each damaged E11 cell, Spearman over the 8 variants of D with
    pr = rho_p x sd_e / sd_J (E13, none planner's last-iteration candidates, mean over all E13 states) and with the latent
    1-step error rel1 (E13, iteration-0 candidates, the E1 definition); holds iff the unit mean of the difference >= 0.3
  E16 (DINO-WM half, descriptive): lambda choice, qualification none_MPPI >= 0.6 none_CEM (same pod), D under MPPI,
    D(sh-plan) - D(sh-iter) under CEM (E11 cell) and MPPI.
"""
import os, sys, json, glob, argparse, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import podplan as PP

NB, EPS, ALPHA = 10000, 1e-9, 0.05
TN = dict(pusht='PushT', pointmaze='PointMaze', wall='Wall')
VN = dict(none='none', det='det', detbc='det+BC', shstep='sh-plan', detshiftfix='det+shift-fixed', shiterb='sh-iter+b',
          shiter='sh-iter', detshift='det+shift', detnoise='det+noise', ind='ind', shep='sh-ep')
E11_CELLS = (('pusht', 4), ('pointmaze', 3), ('wall', 3))
E1_WALL3 = dict(det_D=0.50, red=dict(ind=1.02, shiter=0.92, shstep=1.08, detshift=1.02))   # E1 report table 5


def base_tag(t):
    return re.sub(r'[a-z]+$', '', t)


def gt(x, t):
    return x is not None and np.isfinite(x) and x > t + EPS


def ge(x, t):
    return x is not None and np.isfinite(x) and x >= t - EPS


def le(x, t):
    return x is not None and np.isfinite(x) and x <= t + EPS


def boot(fn, n, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(NB, n))
    v = np.array([fn(i) for i in idx], float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return [float('nan'), float('nan')]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def boot_mean(d, seed=0):
    rng = np.random.RandomState(seed)
    n = len(d)
    m = d[rng.randint(0, n, size=(NB, n))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], m


def mcnemar(a, b):
    from scipy.stats import binomtest
    n10 = int(np.sum(a & ~b)); n01 = int(np.sum(~a & b))
    return n10, n01, (1.0 if n10 + n01 == 0 else float(binomtest(n10, n10 + n01, 0.5).pvalue))


def holm(pvals, alpha):
    order = np.argsort(pvals)
    m, rej = len(pvals), [False] * len(pvals)
    for r, i in enumerate(order):
        if pvals[i] <= alpha / (m - r) + EPS:
            rej[i] = True
        else:
            break
    return rej


# ------------------------------------------------------------------------------------------------ loading
def load(raw):
    """R[(pod, task, split, ptag, ranges, var, k)] = {ep: record}; ranges/k = '-'/0 for none."""
    R, metas = {}, []
    for f in sorted(glob.glob(f'{raw}/*/jobs/*/*.json')):
        tag = f.split('/')[-4]
        d = json.load(open(f))
        j = d['job']
        pt = PP.ptag_of(j.get('planner'))
        key = (base_tag(tag), j['task'], j['split'], pt, '-' if j['var'] == 'none' else j['ranges'], j['var'],
               0 if j['var'] == 'none' else int(j['k']))
        for r in d['records']:
            R.setdefault(key, {})[int(r['ep'])] = r
        metas.append(dict(tag=tag, job=j, meta=d['meta']))
    return R, metas


def none_of(R, pod, task, split='eval8', pt='cem'):
    return R.get((pod, task, split, pt, '-', 'none', 0), {})


def cell(R, pod, task, ranges, var, k, split='eval8', pt='cem'):
    return R.get((pod, task, split, pt, ranges, var, int(k)), {})


def paired(none, *cells):
    eps = sorted(set(none).intersection(*[set(c) for c in cells]))
    S = [np.array([float(c[e]['success']) for e in eps]) for c in (none,) + cells]
    return eps, S


def dstats(none, v):
    eps, (Sn, Sv) = paired(none, v)
    if not eps:
        return None
    d = Sn - Sv
    ci, m = boot_mean(d)
    n10, n01, p = mcnemar(Sn.astype(bool), Sv.astype(bool))
    return dict(n=len(eps), success=float(Sv.mean()), none_success=float(Sn.mean()), D=float(d.mean()), D_ci=ci,
                p_boot_le0=float((np.sum(m <= 0) + 1) / (NB + 1)), mcnemar=dict(none_only=n10, var_only=n01, p=p))


def contrast(none, a, b):
    """D_a - D_b = S_b - S_a on the common pairs."""
    eps, (Sn, Sa, Sb) = paired(none, a, b)
    if not eps:
        return None
    g = Sb - Sa
    ci, _ = boot_mean(g)
    n10, n01, p = mcnemar(Sb.astype(bool), Sa.astype(bool))
    return dict(n=len(eps), diff=float(g.mean()), ci=ci, mcnemar_p=p)


def red_stats(none, det, v):
    eps, (Sn, Sd, Sv) = paired(none, det, v)
    if not eps:
        return None

    def red(i):
        dd = (Sn - Sd)[i].mean(); dv = (Sn - Sv)[i].mean()
        return (dd - dv) / dd if dd > 0 else np.nan
    r0 = red(np.arange(len(eps)))
    return dict(red=None if not np.isfinite(r0) else float(r0), red_ci=boot(red, len(eps)), n=len(eps))


def mean_rec(c, key, sub=None):
    v = []
    for r in c.values():
        x = r.get(key)
        if sub is not None and x is not None:
            for s in sub:
                x = x.get(s) if isinstance(x, dict) else None
        if x is not None and np.isfinite(x):
            v.append(float(x))
    return float(np.mean(v)) if v else None


# ------------------------------------------------------------------------------------------------ cluster bootstrap (D1)
def clusters_of(raw, task='pusht'):
    """component of every eval8 pair (connected components of mutually overlapping PushT segments, make_eval8)."""
    for f in sorted(glob.glob(f'{raw}/*/splits/eval8_{task}.json')):
        d = json.load(open(f))
        if 'component_of_pair' in d:
            return d['component_of_pair']
    return None


def cluster_stats(d_by_ep, comp, seed=0):
    """Cluster bootstrap of a paired per-pair difference (D1 auxiliary CI): resample overlap components with
    replacement (10,000, RandomState(0)); statistic = mean over the pairs of the drawn components."""
    eps = sorted(d_by_ep)
    labs = sorted({comp[e] for e in eps})
    S = np.array([sum(d_by_ep[e] for e in eps if comp[e] == c) for c in labs], float)
    N = np.array([sum(1 for e in eps if comp[e] == c) for c in labs], float)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(labs), size=(NB, len(labs)))
    m = S[idx].sum(1) / N[idx].sum(1)
    return dict(n_clusters=len(labs), ci=[float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
                p_le0=float((np.sum(m <= 0) + 1) / (NB + 1)))


def cluster_contrast(none, a, b, comp):
    """D_a - D_b (b may be None: D_a) per pair, cluster bootstrap."""
    cells = (a,) if b is None else (a, b)
    eps, S = paired(none, *cells)
    if not eps:
        return None
    d = (S[0] - S[1]) if b is None else (S[2] - S[1])
    return cluster_stats({e: float(x) for e, x in zip(eps, d)}, comp)


# ------------------------------------------------------------------------------------------------ E13
def e13_summary(raw, pod, task):
    """Mean over the E13 states of every configuration. The state files of all pods of the task are merged per
    (split, episode) (PushT: dw.1 holds the PT configurations above K1, dw.5 the others; both use their own none states
    on the same pairs and seeds); `pod` is kept for the call signature."""
    merged = {}
    for f in sorted(glob.glob(f'{raw}/*/e13/{task}/*_e*.json')):
        d = json.load(open(f))
        m = merged.setdefault((d['split'], d['ep']), dict(split=d['split'], ep=d['ep'], sets={}))
        for sname, S in d['sets'].items():
            for key, met in S.items():
                m['sets'].setdefault(sname, {}).setdefault(key, met)
    recs = list(merged.values())
    out = {}
    for d in recs:
        for sname, S in d['sets'].items():
            for key, met in S.items():
                for grp in ('all', d['split']):
                    out.setdefault(key, {}).setdefault(f'{sname}_{grp}', []).append(met)
    summ = {}
    for key, groups in out.items():
        summ[key] = {}
        for g, vals in groups.items():
            agg = {}
            for mk in vals[0]:
                v = np.array([x.get(mk, np.nan) for x in vals], float)
                agg[mk] = float(np.nanmean(v)) if np.isfinite(v).any() else None
            agg['n_states'] = len(vals)
            summ[key][g] = agg
    return summ


def ekey(ranges, var, k):
    return f'{ranges}|{var}|k{k}'


# ------------------------------------------------------------------------------------------------ E11, B2, B3
def unconverged(raw):
    """(task, ranges, k) whose D2 bias vector is flagged bc_unconverged (any pod's calib copy)."""
    out = set()
    for f in glob.glob(f'{raw}/*/calib/bias_*.json'):
        if '.' in os.path.basename(f)[:-5]:
            continue
        b = json.load(open(f))
        if b.get('bc_unconverged'):
            out.add((b['task'], b['ranges'], int(b['k'])))
    return out


def e11(R, raw, drop=frozenset()):
    """drop: (task, ranges, k) whose det+BC and sh-iter+b are treated as missing (D2 version (b))."""
    cells, pvals = [], []
    for task, k in E11_CELLS:
        pod = next(t for t, P in PP.PODS.items() if task in P['e11'])
        none = none_of(R, pod, task)
        V = {v: ({} if (v in ('detbc', 'shiterb') and (task, 'main', k) in drop) else cell(R, pod, task, 'main', v, k))
             for v in PP.E11_VARS}
        if not none or not V['det']:
            continue
        C = dict(task=task, k=k, pod=pod, none=dict(n=len(none), success=float(np.mean([r['success'] for r in none.values()]))),
                 variants={})
        for v, c in V.items():
            if not c:
                continue
            s = dstats(none, c)
            if v != 'det':
                s.update(red_stats(none, V['det'], c) or {})
                s['Dv_minus_Ddet'] = contrast(none, c, V['det'])
            for st in ('1', '3', '5'):
                s[f'rel_err{st}'] = mean_rec(c, 'err', (st, 'rel'))
            dr = [r['oei']['last_pop_dJ'] - r['oei']['it0_rand_dJ'] for r in c.values() if 'oei' in r]
            s['drift'] = float(np.mean(dr)) if dr else None
            C['variants'][v] = s
        det = C['variants']['det']
        pvals.append(det['p_boot_le0'])
        dd = [r['oei']['last_pop_dJ'] - r['oei']['it0_rand_dJ'] for r in V['det'].values() if 'oei' in r]
        for v, s in C['variants'].items():
            if v != 'det' and s.get('drift') is not None and dd and abs(np.mean(dd)) > 1e-12:
                s['drift_ratio_vs_det'] = s['drift'] / float(np.mean(dd))
        C['d_bias'] = contrast(none, V['det'], V['detbc']) if V['detbc'] else None
        C['d_pers'] = contrast(none, V['detshiftfix'], V['detshift']) if V['detshiftfix'] and V['detshift'] else None
        C['aux'] = {name: contrast(none, V[a], V[b]) for name, a, b in (
            ('D(sh-plan)-D(sh-iter)', 'shstep', 'shiter'), ('D(sh-iter+b)-D(sh-iter)', 'shiterb', 'shiter'),
            ('D(det+noise)-D(det)', 'detnoise', 'det'), ('D(det+shift)-D(sh-iter)', 'detshift', 'shiter'))
            if V[a] and V[b]}
        cells.append(C)
    rej = holm(pvals, ALPHA) if pvals else []
    damaged = []
    for C, rj in zip(cells, rej):
        det = C['variants']['det']
        C['damaged'] = bool(rj and det['D'] > 0)
        if not C['damaged']:
            continue
        fx = lambda v: ge(C['variants'].get(v, {}).get('red'), 0.5)
        dp = C['d_pers']
        lo, hi = (dp['ci'] if dp else [np.nan, np.nan])
        if 'detbc' not in C['variants']:
            cls = 'undetermined'                 # det+BC missing (D2 version (b))
        elif fx('detbc'):
            cls = 'bias'
        elif gt(lo, 0.0):
            cls = 'persistence'
        elif fx('detshiftfix') and le(lo, 0.0) and ge(hi, 0.0):
            cls = 'resolution'
        else:
            cls = 'mixed'
        C['class'] = cls
        C['fixes'] = {v: fx(v) for v in PP.E11_VARS if v != 'det'}
        damaged.append(C)
    unit = 'no damaged cell'
    if damaged:
        counts = {c: sum(1 for x in damaged if x['class'] == c) for c in ('bias', 'persistence', 'resolution', 'mixed',
                                                                          'undetermined')}
        maj = [c for c, n in counts.items() if n > len(damaged) / 2 + EPS]
        unit = maj[0] if maj else 'mixed'
    comp = clusters_of(raw)
    for C in cells:                          # D1 auxiliary: overlap-cluster bootstrap for the PushT eval8 cell
        if C['task'] != 'pusht' or comp is None:
            continue
        pod, k = C['pod'], C['k']
        none = none_of(R, pod, 'pusht')
        V = {v: cell(R, pod, 'pusht', 'main', v, k) for v in PP.E11_VARS}
        cl = dict(D={v: cluster_contrast(none, V[v], None, comp) for v in PP.E11_VARS if V[v]},
                  d_bias=cluster_contrast(none, V['det'], V['detbc'], comp) if V['detbc'] else None,
                  d_pers=cluster_contrast(none, V['detshiftfix'], V['detshift'], comp) if V['detshiftfix'] and V['detshift'] else None)
        ps = list(pvals); i = [c['task'] for c in cells].index('pusht'); ps[i] = cl['D']['det']['p_le0']
        cl['damaged'] = bool(holm(ps, ALPHA)[i] and C['variants']['det']['D'] > 0)
        cls = None
        if cl['damaged']:
            fx = lambda v: ge(C['variants'].get(v, {}).get('red'), 0.5)
            lo, hi = cl['d_pers']['ci'] if cl['d_pers'] else [np.nan, np.nan]
            cls = ('bias' if fx('detbc') else 'persistence' if gt(lo, 0.0) else
                   'resolution' if fx('detshiftfix') and le(lo, 0.0) and ge(hi, 0.0) else 'mixed')
        cl['class'] = cls
        cl['reading_differs'] = bool(cl['damaged'] != C['damaged'] or (cls != C.get('class')))
        C['cluster'] = cl
    nfix = sum(1 for x in damaged if x['fixes'].get('detnoise'))
    out = dict(cells=cells, holm=dict(alpha=ALPHA, pvals=pvals, rejected=rej), damaged=[(c['task'], c['k']) for c in damaged],
               unit_reading=unit, class_counts={c['task']: c.get('class') for c in damaged},
               B2=dict(holds=(unit in ('bias', 'resolution')) if damaged else None, reading=unit,
                       rule='DINO-WM unit not persistence-dominated: reading = bias or resolution'),
               B3_dinowm=dict(holds=(nfix <= len(damaged) / 2 + EPS) if damaged else None, n_damaged=len(damaged),
                              n_detnoise_fixes=nfix, rule='det+noise fixes at most half of the damaged cells'))
    w = next((c for c in cells if c['task'] == 'wall'), None)
    if w:
        det = w['variants']['det']
        out['wall3_vs_E1_explor'] = dict(E1=E1_WALL3, det_D=det['D'], det_D_ci=det['D_ci'], det_p=det['p_boot_le0'],
                                         red={v: w['variants'].get(v, {}).get('red') for v in ('shiter', 'shstep', 'detshift')},
                                         replicated=bool(det['p_boot_le0'] < ALPHA and ge(w['variants'].get('shiter', {}).get('red'), 0.5)),
                                         rule='det drop with one-sided p < 0.05 and red(sh-iter) >= 0.5 (descriptive)')
    for C in cells:
        S = e13_summary(raw, C['pod'], C['task'])
        C['e13'] = {v: dict(it0=S.get(ekey('main', v, C['k']), {}).get('it0_all'),
                            last=S.get(ekey('main', v, C['k']), {}).get('last_all')) for v in PP.E11_VARS}
    return out


def dg1(E11):
    from scipy.stats import spearmanr
    rows = []
    for C in E11['cells']:
        if not C.get('damaged'):
            continue
        D, pr, l1, l1L = [], [], [], []
        for v in PP.E11_VARS:
            e = C['e13'].get(v) or {}
            if v not in C['variants'] or not e.get('it0') or not e.get('last'):
                continue
            D.append(C['variants'][v]['D']); pr.append(e['last']['pr']); l1.append(e['it0']['rel1']); l1L.append(e['last']['rel1'])
        if len(D) < 3:
            rows.append(dict(task=C['task'], k=C['k'], n=len(D), note='E13 incomplete'))
            continue
        r_pr, r_l1, r_l1L = (float(spearmanr(x, D).correlation) for x in (pr, l1, l1L))
        rows.append(dict(task=C['task'], k=C['k'], n=len(D), rho_pr_D=r_pr, rho_lat1_D=r_l1, diff=r_pr - r_l1,
                         aux_rho_lat1_last_D=r_l1L, aux_diff_last=r_pr - r_l1L))
    ok = [r for r in rows if 'diff' in r and np.isfinite(r['diff'])]
    mean = float(np.mean([r['diff'] for r in ok])) if ok else None
    return dict(cells=rows, unit_mean_diff=mean, holds=(ge(mean, 0.3) if mean is not None else None),
                rule='mean over damaged E11 cells of Spearman(pr, D) - Spearman(rel1 it0, D) >= 0.3 (8 variants per cell)')


# ------------------------------------------------------------------------------------------------ E12
def pods_of(task):
    return [t for t, P in PP.PODS.items() if task in P['tasks']]


def e12(R, raw=None, drop=frozenset()):
    comp = clusters_of(raw) if raw else None
    out = {}
    for task in ('pusht', 'pointmaze', 'wall'):
        T = {}
        for pod in pods_of(task):
            none = none_of(R, pod, task)
            for (p, t, split, pt, ranges, var, k), c in R.items():
                if p != pod or t != task or split != 'eval8' or pt != 'cem' or var == 'none':
                    continue
                kind = 'rc' if ranges.startswith('rc_') else ranges
                if var not in PP.E12_ROUND.get(kind, ()):
                    continue
                if var in ('detbc', 'shiterb') and (task, ranges, int(k)) in drop:
                    continue
                s = dstats(none, c)
                if s:
                    s.update(pod=pod, ranges=ranges, two_stage=task in PP.TWO_STAGE)
                    if task == 'pusht' and comp is not None:
                        s['cluster'] = cluster_contrast(none, c, None, comp)
                    T.setdefault(kind, {}).setdefault(var, {})[int(k)] = s
        kstar = {}
        for kind, byv in T.items():
            for var, byk in byv.items():
                ks = sorted(byk)
                ok = [k for k in ks if all(le(byk[kk]['D'], PP.TAU) for kk in ks if kk >= k)]
                kstar[f'{kind}|{var}'] = dict(kstar=(min(ok) if ok else None), measured=ks,
                                              D={k: byk[k]['D'] for k in ks}, n={k: byk[k]['n'] for k in ks})
        G = {}
        for kind in T:
            ks = lambda v: kstar.get(f'{kind}|{v}', {}).get('kstar')
            dets = [x for x in (ks('det'), ks('detbc')) if x is not None] if kind != 'rc' else [ks('det')]
            sh = ks('shiter')
            if dets and all(x is not None for x in dets) and sh is not None:
                G[kind] = dict(G=min(dets) - sh, det_side=min(dets), shiter=sh)
            else:
                G[kind] = dict(G=None, note='k* undefined for a rounding', det=ks('det'), detbc=ks('detbc'), shiter=sh)
        best = None
        for key, v in kstar.items():
            if v['kstar'] is not None and (best is None or v['kstar'] < best[1]):
                best = (key, v['kstar'])
        out[task] = dict(cells=T, kstar=kstar, G=G, best=best,
                         note='two-stage cells (PushT): CI not adjusted for the interim look' if task in PP.TWO_STAGE else '')
    return out


def plot_e12(E12, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    sty = dict(det=('o', '-'), detbc=('s', '--'), shiter=('^', ':'))
    col = dict(main='C0', pc='C1', rc='C2')
    for ax, task in zip(axes, ('pusht', 'pointmaze', 'wall')):
        T = E12.get(task, {}).get('cells', {})
        for kind, byv in T.items():
            for var, byk in byv.items():
                ks = sorted(byk)
                ax.plot(ks, [100 * (byk[k]['none_success'] - byk[k]['D']) for k in ks], marker=sty[var][0],
                        ls=sty[var][1], color=col[kind], label=f"{'PT' if kind == 'main' else kind.upper()} {VN[var]}")
        ax.set_title(TN[task]); ax.set_xlabel('bits k'); ax.set_ylabel('success %'); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return path


# ------------------------------------------------------------------------------------------------ E16
def e16(R, raw):
    out = {}
    for task in ('pointmaze', 'wall'):
        pod = next((t for t, P in PP.PODS.items() if task in P['e16']), None)
        lam = None
        for f in glob.glob(f'{raw}/{pod}*/state/e16_lambda_{task}.json'):
            lam = json.load(open(f))
        if lam is None:
            continue
        pt = f"mppi{lam['chosen_c']:g}"
        nm, nc = none_of(R, pod, task, pt=pt), none_of(R, pod, task)
        if not nm:
            continue
        sm, sc = float(np.mean([r['success'] for r in nm.values()])), float(np.mean([r['success'] for r in nc.values()]))
        V = {v: cell(R, pod, task, 'main', v, PP.E16_K, pt=pt) for v in PP.E16_VARS}
        res = dict(pod=pod, lambda_selection=lam, none_mppi=sm, none_cem=sc, ratio=sm / sc if sc > 0 else None,
                   qualified=ge(sm / sc if sc > 0 else None, 0.6), variants={v: dstats(nm, c) for v, c in V.items() if c})
        res['D(sh-plan)-D(sh-iter)'] = dict(
            mppi=contrast(nm, V['shstep'], V['shiter']) if V['shstep'] and V['shiter'] else None,
            cem=contrast(nc, cell(R, pod, task, 'main', 'shstep', PP.E16_K), cell(R, pod, task, 'main', 'shiter', PP.E16_K)))
        res['lam_mean'] = float(np.mean([r.get('mppi_lam', np.nan) for r in nm.values()]))
        out[task] = res
    return out


# ------------------------------------------------------------------------------------------------ report tables
def pp_(x, c=None):
    if x is None:
        return '—'
    s = f'{100 * x:+.0f}'
    return s + (f' [{100 * c[0]:.0f}, {100 * c[1]:.0f}]' if c else '')


def tables(A):
    L = ['## E11 (DINO-WM)', '', '| 칸 | 변형 | n | 성공 | D %p [95% CI] | red [95% CI] | D_v − D_det %p [CI] | 1-step | 5-step | drift 비 |',
         '|' + '---|' * 10]
    for C in A['E11']['cells']:
        for v in PP.E11_VARS:
            s = C['variants'].get(v)
            if not s:
                continue
            red = '—' if s.get('red') is None else f"{s['red']:.2f} [{s['red_ci'][0]:.2f}, {s['red_ci'][1]:.2f}]"
            dvd = s.get('Dv_minus_Ddet')
            L.append(f"| {TN[C['task']]} k={C['k']} | {VN[v]} | {s['n']} | {100 * s['success']:.0f}% | {pp_(s['D'], s['D_ci'])} | {red} | "
                     f"{pp_(dvd['diff'], dvd['ci']) if dvd else '—'} | {s.get('rel_err1') or float('nan'):.4f} | "
                     f"{s.get('rel_err5') or float('nan'):.4f} | {s.get('drift_ratio_vs_det', float('nan')):.2f} |")
    E = A['E11']
    L += ['', f"- 손상 칸(Holm, α 0.05): {E['damaged']}; 단위 읽기: **{E['unit_reading']}**; 칸별 {E['class_counts']}",
          f"- B2: {E['B2']['holds']} · B3(DINO-WM): {E['B3_dinowm']['holds']} (det+noise가 고친 칸 {E['B3_dinowm']['n_detnoise_fixes']}/{E['B3_dinowm']['n_damaged']})"]
    for C in E['cells']:
        L.append(f"- {TN[C['task']]}: Δ_bias {pp_(C['d_bias']['diff'], C['d_bias']['ci']) if C['d_bias'] else '—'}, "
                 f"Δ_pers {pp_(C['d_pers']['diff'], C['d_pers']['ci']) if C['d_pers'] else '—'}; 보조 " +
                 '; '.join(f"{k} {pp_(v['diff'], v['ci'])}" for k, v in C['aux'].items() if v))
    D = A['Dg1']
    L += ['', '## Dg1', '', f"- 단위 평균 차 {D['unit_mean_diff']} → {D['holds']}"] + [f"  - {r}" for r in D['cells']]
    L += ['', '## E12', '']
    for task, T in A['E12'].items():
        L.append(f"- {TN[task]}: k* " + ', '.join(f"{k} {v['kstar']}" for k, v in T['kstar'].items()) +
                 f"; G {json.dumps(T['G'])}; 최선 {T['best']} {T['note']}")
    L += ['', '## E16', '']
    for task, r in A['E16'].items():
        L.append(f"- {TN[task]}: c* {r['lambda_selection']['chosen_c']}, none MPPI {100 * r['none_mppi']:.0f}% / CEM {100 * r['none_cem']:.0f}% "
                 f"(자격 {r['qualified']}); D(sh-plan) − D(sh-iter): MPPI "
                 f"{pp_(r['D(sh-plan)-D(sh-iter)']['mppi']['diff'], r['D(sh-plan)-D(sh-iter)']['mppi']['ci']) if r['D(sh-plan)-D(sh-iter)']['mppi'] else '—'}, "
                 f"CEM {pp_(r['D(sh-plan)-D(sh-iter)']['cem']['diff'], r['D(sh-plan)-D(sh-iter)']['cem']['ci']) if r['D(sh-plan)-D(sh-iter)']['cem'] else '—'}")
    return '\n'.join(L) + '\n'


def checks(raw):
    out = {}
    for f in sorted(glob.glob(f'{raw}/*/checks/*.json')):
        out[f.split('/')[-3] + '/' + os.path.basename(f)] = {k: v for k, v in json.load(open(f)).items()
                                                            if k in ('passed', 'none_success', 'e1_none_success', 'diff')}
    return out


# ------------------------------------------------------------------------------------------------ fake data (dry run)
def fake(raw):
    rng = np.random.RandomState(1)
    p_none = dict(pusht=0.8, pointmaze=0.85, wall=0.6)

    def write(tag, task, split, var, k, ranges, eps, ps, planner=None, diag=False):
        j = PP.job(task, split, var, k, eps[0], eps[-1] + 1, ranges=ranges, diag=diag, planner=planner)
        recs = []
        for e in eps:
            r = dict(ep=e, task=task, split=split, var=var, k=j['k'], success=bool(rng.rand() < ps))
            if diag:
                r['err'] = {s: dict(rel=float(rng.rand() * 0.3)) for s in ('1', '3', '5')}
                r['oei'] = dict(last_pop_dJ=float(rng.randn() * 0.1), it0_rand_dJ=float(rng.randn() * 0.1))
            if planner:
                r['mppi_lam'] = float(rng.rand())
            recs.append(r)
        d = f'{raw}/{tag}/jobs/{task}'
        os.makedirs(d, exist_ok=True)
        json.dump(dict(job=j, meta={}, records=recs), open(f"{d}/{j['id']}.json", 'w'))
    for tag, P in PP.PODS.items():
        for task in P['tasks']:
            write(tag, task, 'eval8', 'none', 0, 'main', range(100), p_none[task], diag=True)
        for task in P['e11']:
            for v in PP.E11_VARS:
                drop = dict(det=0.3, detbc=0.25, shstep=0.02, detshiftfix=0.03, shiterb=0.1, shiter=0.0, detshift=0.0, detnoise=0.28)[v]
                write(tag, task, 'eval8', v, PP.K1[task], 'main', range(100), p_none[task] - drop, diag=True)
            S = f'{raw}/{tag}/e13/{task}'
            os.makedirs(S, exist_ok=True)
            for split, n in (('held', 20), ('eval8', 100)):
                for e in range(n):
                    sets = {s: {ekey('main', v, PP.K1[task]): dict(rel1=float(rng.rand()), rel3=0.2, rel5=0.3, ratio=float(rng.rand()),
                                                                     rho_p=1.0 if v in ('det', 'detbc', 'shstep', 'detshiftfix') else 0.1,
                                                                     pr=float(rng.rand()), spearman=0.9, elite_overlap=0.5)
                                for v in PP.E11_VARS} for s in ('it0', 'last')}
                    json.dump(dict(task=task, split=split, ep=e, sets=sets), open(f'{S}/{split}_e{e:03d}.json', 'w'))
        for task, kinds in P['e12'].items():
            for kind, ks in kinds.items():
                for k in ks:
                    for v in PP.E12_ROUND[kind]:
                        if kind == 'main' and k == PP.K1[task] and task in P['e11']:
                            continue
                        n = 50 if task in PP.TWO_STAGE else 100
                        write(tag, task, 'eval8', v, k, PP.rid_of(kind, k), range(n), p_none[task] - 0.1 * (5 - k) * (v != 'shiter'))
        for task in P['e16']:
            os.makedirs(f'{raw}/{tag}/state', exist_ok=True)
            json.dump(dict(task=task, chosen_c=0.5, rows={}), open(f'{raw}/{tag}/state/e16_lambda_{task}.json', 'w'))
            pl = {'mppi_c': 0.5}
            write(tag, task, 'eval8', 'none', 0, 'main', range(100), p_none[task] - 0.05, planner=pl)
            for v in PP.E16_VARS:
                write(tag, task, 'eval8', v, PP.E16_K, 'main', range(100), p_none[task] - 0.1, planner=pl)
    os.makedirs(f'{raw}/dw.1/splits', exist_ok=True)
    json.dump(dict(component_of_pair=[int(x) for x in rng.randint(0, 30, size=100)]), open(f'{raw}/dw.1/splits/eval8_pusht.json', 'w'))
    print('fake data written to', raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw'); ap.add_argument('--out', default=None); ap.add_argument('--fake', action='store_true')
    a = ap.parse_args()
    if a.fake:
        fake(a.raw)
        return
    out_dir = a.out or os.path.join(os.path.dirname(os.path.abspath(a.raw)), 'analysis')
    os.makedirs(out_dir, exist_ok=True)
    R, metas = load(a.raw)
    A = dict(E11=e11(R, a.raw))
    A['Dg1'] = dg1(A['E11'])
    A['E12'] = e12(R, a.raw)
    drop = unconverged(a.raw)                    # D2: version (b) with the bc_unconverged cells' det+BC / sh-iter+b missing
    A['bc_unconverged'] = sorted(list(x) for x in drop)
    if drop:
        Eb, Gb = e11(R, a.raw, drop), e12(R, a.raw, drop)
        A['D2_version_b'] = dict(B2=Eb['B2'], reading=Eb['unit_reading'], B3_dinowm=Eb['B3_dinowm'],
                                 G={t: {k: v['G'] for k, v in T['G'].items()} for t, T in Gb.items()},
                                 kstar={t: {k: v['kstar'] for k, v in T['kstar'].items()} for t, T in Gb.items()})
        Ga = {t: {k: v['G'] for k, v in T['G'].items()} for t, T in A['E12'].items()}
        A['D2_hold'] = dict(B2=bool(Eb['B2']['holds'] != A['E11']['B2']['holds']),
                            G={t: {k: bool(Ga[t].get(k) != A['D2_version_b']['G'][t].get(k)) for k in Ga[t]} for t in Ga},
                            note='True = the verdict depends on unconverged cells: report "보류(미수렴 칸에 의존)"')
    A['E16'] = e16(R, a.raw)
    A['checks'] = checks(a.raw)
    A['n_jobs'] = len(metas)
    A['gpus'] = sorted({(m['tag'], m['meta'].get('gpu')) for m in metas if m['meta'].get('gpu')})
    png = plot_e12(A['E12'], f'{out_dir}/e12_bits_success.png')
    json.dump(A, open(f'{out_dir}/analysis_dw.json', 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    open(f'{out_dir}/tables_dw.md', 'w').write(tables(A))
    print(json.dumps(dict(damaged=A['E11']['damaged'], reading=A['E11']['unit_reading'], B2=A['E11']['B2']['holds'],
                          B3_dinowm=A['E11']['B3_dinowm']['holds'], Dg1=A['Dg1']['holds'],
                          G={t: {k: v['G'] for k, v in T['G'].items()} for t, T in A['E12'].items()},
                          E16={t: r['qualified'] for t, r in A['E16'].items()}, png=png), default=str))
    print('wrote', out_dir)


if __name__ == '__main__':
    main()
