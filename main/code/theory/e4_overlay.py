"""E4 overlay hooks (paper A): put E2/E3 measurements on the same axes as the E4 theory and simulation.

Input: analysis JSONs in the style of the E1 analyses (TD-MPC2 `e1_tdmpc2_summary.json`: {"cells": {"<task>:<k>[:...]":
{"task", "k", ..., "variants": {v: cell}}}}; DINO-WM `analyze_e1.py --out`: {"tasks": {task: {"confirmatory" |
"exploratory" | <sweep>: {k: {v: cell}}}}}). Cells are found by walking the JSON: a dict with an OEI key ('oei' or
'oei_ref_mf') is a cell; task, bit count, variant and sweep parameters are collected from the path (dict keys such as
"mw-push:3:N512:q0.125", "N512", "neff=256", nested {"N": {"512": ...}}) and from fields of the cell or its parents
(N/num_samples, num_elites/topk/elite_ratio, iterations/opt_steps, temperature, n_eff, warm_start).
Minimal fields per cell for each overlay (report section 9):
  dose (H2c)     N, oei (+ oei_ci), D (+ D_ci); optional elites / elite_ratio, temperature
  iterations     iterations, D, oei; optional warm_start
  N_eff (H4a)    n_eff, D, oei; optional var_cand (fixed-table across-candidate error variance), f_e_rand_sd
  decomposition  TD-MPC2: e_chosen, l_e_rand, f_e_rand (sel_last = e_chosen - l_e_rand, drift = l_e_rand - f_e_rand);
                 DINO-WM: mf_dJ, last_pop_dJ, it0_rand_dJ (sel_last_mf = mf_dJ - last_pop_dJ,
                 drift = last_pop_dJ - it0_rand_dJ), last_elite_dJ (same-draw elite selection)
  winner's curse l_e_rand_sd, l_sd_v0 (TD-MPC2) or the last-population std of dJ and J_none (DINO-WM), N, elites,
                 temperature -> predicted sel_last from e4_formulas.winners_curse
usage: python e4_overlay.py <e4_comparisons.json> <s_tables.json> <out_dir> <analysis.json> [more.json ...]
       python e4_overlay.py --selftest <e4_comparisons.json> <s_tables.json> <out_dir>   (synthetic data only)
"""
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e4_formulas as F  # noqa: E402

OEI_KEYS = ('oei', 'oei_ref_mf')
ALIASES = {
    'N': ('N', 'num_samples', 'n_samples', 'samples'),
    'elites': ('elites', 'num_elites', 'topk', 'k_elite', 'n_elites'),
    'elite_ratio': ('elite_ratio', 'q', 'elite_frac'),
    'iterations': ('iterations', 'iters', 'opt_steps', 'n_iter', 'n_it'),
    'temperature': ('temperature', 'temp'),
    'n_eff': ('n_eff', 'neff', 'table_size', 'N_eff'),
    'warm_start': ('warm_start', 'ws'),
}
KEY_PATTERNS = [(re.compile(r'^(?:N|n_samples|num_samples)[=_]?(\d+)$'), 'N', int),
                (re.compile(r'^(?:neff|n_eff|N_eff|table)[=_]?(\d+)$'), 'n_eff', int),
                (re.compile(r'^(?:q|elite_ratio)[=_]?([0-9.]+)$'), 'elite_ratio', float),
                (re.compile(r'^(?:topk|elites|num_elites|kel)[=_]?(\d+)$'), 'elites', int),
                (re.compile(r'^(?:it|iters|iterations|opt_steps|n)[=_]?(\d+)$'), 'iterations', int),
                (re.compile(r'^(?:T|temp|temperature)[=_]?([0-9.]+)$'), 'temperature', float),
                (re.compile(r'^(?:ws|warm_start)[=_]?([01])$'), 'warm_start', int),
                (re.compile(r'^k[=_]?(\d+)$'), 'k', int)]
VARIANT_CANON = {'det': 'det', 'ind': 'ind', 'shiter': 'shiter', 'sh-iter': 'shiter', 'shfwd': 'shiter',
                 'shstep': 'shstep', 'sh-step': 'shstep', 'shep': 'shep', 'sh-ep': 'shep', 'detshift': 'detshift',
                 'det+shift': 'detshift', 'shfixed': 'shstep', 'none': 'none', 'nonealt': 'nonealt'}
CONTAINER_TASK = ('tasks',)
CONTAINER_K = ('confirmatory', 'exploratory', 'by_k')
CONTAINER_VAR = ('variants',)


def _parse_key(key, ctx, parent):
    key = str(key)
    if parent in CONTAINER_TASK:
        ctx['task'] = key
        return
    if parent in CONTAINER_K and re.match(r'^\d+$', key):
        ctx['k'] = int(key)
        return
    if parent in CONTAINER_VAR or key in VARIANT_CANON:
        if key in VARIANT_CANON:
            ctx['variant'] = VARIANT_CANON[key]
            return
    for canon, al in ALIASES.items():          # nested {"N": {"512": ...}}
        if parent in al and re.match(r'^[0-9.]+$', key):
            ctx[canon] = float(key) if '.' in key else int(key)
            return
    toks = key.split(':')
    if len(toks) >= 2 and not re.match(r'^[0-9.]+$', toks[0]):
        ctx['task'] = toks[0]
        if re.match(r'^\d+$', toks[1]):
            ctx['k'] = int(toks[1])
        rest = toks[2:]
    else:
        rest = [key]
    for t in rest:
        for pat, canon, typ in KEY_PATTERNS:
            m = pat.match(t)
            if m:
                ctx[canon] = typ(m.group(1))
                break


def _fields_into(ctx, d):
    for canon, al in ALIASES.items():
        for a in al:
            v = d.get(a)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                ctx[canon] = v
                break
    for a in ('task', 'k', 'family'):
        if isinstance(d.get(a), (str, int)) and not isinstance(d.get(a), bool):
            ctx[a] = d[a]


def extract_cells(obj, ctx=None, parent=None, out=None):
    """Walk an analysis JSON and return flat cell records (context + the cell's scalar / CI fields)."""
    if out is None:
        out = []
    ctx = dict(ctx or {})
    if isinstance(obj, dict):
        _fields_into(ctx, obj)
        if any(k in obj and isinstance(obj[k], (int, float)) for k in OEI_KEYS):
            rec = dict(ctx)
            for k, v in obj.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    rec[k] = float(v)
                elif isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) or x is None for x in v):
                    rec[k] = [np.nan if x is None else float(x) for x in v]
            out.append(rec)
            return out
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                c2 = dict(ctx)
                _parse_key(k, c2, parent)
                extract_cells(v, c2, k, out)
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, (dict, list)):
                extract_cells(v, ctx, parent, out)
    return out


def normalize(rec):
    """Canonical fields: family (tdmpc2 | dinowm), oei, oei_ci, D, D_ci, sel_last, drift (when recorded)."""
    r = dict(rec)
    if 'oei' in rec:
        r['family'] = 'tdmpc2'
        r['oei_val'] = rec['oei']
        r['oei_ci_val'] = rec.get('oei_ci')
        if all(x in rec for x in ('e_chosen', 'l_e_rand')):
            r['sel_last'] = rec['e_chosen'] - rec['l_e_rand']
        if all(x in rec for x in ('l_e_rand', 'f_e_rand')):
            r['drift'] = rec['l_e_rand'] - rec['f_e_rand']
    else:
        r['family'] = 'dinowm'
        r['oei_val'] = rec['oei_ref_mf']
        r['oei_ci_val'] = rec.get('oei_ref_mf_ci')
        if all(x in rec for x in ('mf_dJ', 'last_pop_dJ')):
            r['sel_last'] = rec['mf_dJ'] - rec['last_pop_dJ']
        if all(x in rec for x in ('last_pop_dJ', 'it0_rand_dJ')):
            r['drift'] = rec['last_pop_dJ'] - rec['it0_rand_dJ']
        if all(x in rec for x in ('last_elite_dJ', 'last_pop_dJ')):
            r['sel_elite_same_draw'] = rec['last_elite_dJ'] - rec['last_pop_dJ']
    return r


def load_measurements(paths):
    recs = []
    for p in paths:
        obj = json.load(open(p))
        for r in extract_cells(obj):
            r['_source'] = os.path.basename(p)
            recs.append(normalize(r))
    return recs


# ------------------------------------------------------------------------------------------------ theory per cell
def predict_sel_last(rec, tabs, planner=None):
    """Bivariate winner's-curse prediction of the same-draw selection term from recorded last-iteration constants."""
    fam = rec.get('family')
    planner = planner or ('mppi' if fam == 'tdmpc2' else 'cem')
    se, sJ = rec.get('l_e_rand_sd'), rec.get('l_sd_v0')
    rho = rec.get('l_rho', 0.0)
    N = int(rec.get('N', 512 if planner == 'mppi' else 300))
    k = int(rec.get('elites') or round((rec.get('elite_ratio') or (0.125 if planner == 'mppi' else 0.1)) * N))
    if se is None or sJ is None:
        return np.nan
    if planner == 'mppi':
        temp = rec.get('temperature', 0.5)
        theta = temp * math.sqrt(se ** 2 + sJ ** 2)
        S = tabs.mppi(N, k, theta)
    else:
        S = tabs.topk(N, k)
    return float(F.winners_curse(se, sJ, rho, S))


# ------------------------------------------------------------------------------------------------ overlays
def _mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'figure.dpi': 110})
    return plt


def _ref_norm(xs, ys, x_ref):
    """Value at x_ref by log-log interpolation."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = np.isfinite(ys) & (ys > 0)
    if ok.sum() < 2:
        return np.nan
    return float(np.exp(np.interp(np.log(x_ref), np.log(xs[ok]), np.log(ys[ok]))))


def overlay_dose(meas, e4c, tabs, out_dir, scale='32.0', regime='sep'):
    """H2c: |OEI| and damage vs N, each series normalized at the planner's default N (TD-MPC2 512, DINO-WM 300)."""
    plt = _mpl()
    curves = e4c.get('C7_curves', {})
    fig, ax = plt.subplots(2, 2, figsize=(7.6, 5.4))
    for col_i, (fam, planner, Nref) in enumerate((('tdmpc2', 'mppi', 512), ('dinowm', 'cem', 300))):
        for row_i, (what, key) in enumerate((('|OEI|', 'abs_oei'), ('damage', 'dmg'))):
            a = ax[row_i, col_i]
            for model, colr, lab in (('M1', '#c0392b', 'sim persistent field (det-like)'),
                                     ('R-det', '#e74c3c', 'sim rounding det'),
                                     ('M3', '#2471a3', 'sim fresh shared (sh-iter-like)')):
                if planner == 'cem' and key == 'abs_oei' and model == 'M3':
                    continue        # E[OEI_mf] = 0 for fresh noise: a ratio of noise around 0 is not informative
                for rule, ls in (('q125', '-'), ('kfix', ':')):
                    c = curves.get(f'{planner}|{regime}|{rule}|{model}|{scale}')
                    if not c:
                        continue
                    ys = np.abs(np.asarray(c[key], float))
                    y0 = _ref_norm(c['N'], ys, Nref)
                    a.plot(c['N'], ys / y0, ls, color=colr, lw=1, label=f'{lab}, {rule}' if row_i == 0 else None)
            # operator-only theory curve (fixed elite fraction)
            Ns = np.array([64, 128, 256, 512, 1024])
            if planner == 'mppi':
                S = np.array([tabs.mppi(N, max(2, round(0.125 * N)), 1.0) for N in Ns])
                Sk = np.array([tabs.mppi(N, min(64, N), 1.0) for N in Ns])
            else:
                S = np.array([tabs.topk(N, max(2, round(0.1 * N))) for N in Ns])
                Sk = np.array([tabs.topk(N, min(30, N)) for N in Ns])
            if row_i == 0:
                a.plot(Ns, S / _ref_norm(Ns, S, Nref), 'k-', lw=0.8, alpha=0.6, label='theory S_op, fixed q')
                a.plot(Ns, Sk / _ref_norm(Ns, Sk, Nref), 'k:', lw=0.8, alpha=0.6, label='theory S_op, fixed k')
            # measurements
            M = [m for m in meas if m.get('family') == fam and m.get('N') is not None and m.get('variant') in ('det', 'shiter')]
            groups = {}
            for m in M:
                groups.setdefault((m.get('task'), m.get('variant'), m.get('elite_ratio')), []).append(m)
            for (task, var, q), L in groups.items():
                L = sorted(L, key=lambda m: m['N'])
                xs = np.array([m['N'] for m in L], float)
                if key == 'abs_oei':
                    ys = np.abs([m['oei_val'] for m in L])
                    ci = [m.get('oei_ci_val') for m in L]
                else:
                    ys = np.array([m.get('D', np.nan) for m in L], float)
                    ci = [m.get('D_ci') for m in L]
                y0 = _ref_norm(xs, ys, Nref)
                if not np.isfinite(y0):
                    continue
                mk = 'o' if var == 'det' else 's'
                a.plot(xs, ys / y0, mk, ms=3.5, mfc='none' if var != 'det' else None,
                       label=f'{task} {var}' + (f' q{q:g}' if q else '') if row_i == 0 else None)
                for x, y, c in zip(xs, ys, ci):
                    if c is not None and all(np.isfinite(c)):
                        lo, hi = sorted(np.abs(c)) if key == 'abs_oei' and c[0] * c[1] > 0 else (min(c), max(c))
                        a.plot([x, x], [lo / y0, hi / y0], '-', color='gray', lw=0.6)
            a.set_xscale('log')
            a.set_xlabel('candidates N')
            a.set_ylabel(f'{what} / value at N={Nref}')
            a.set_title(f'{fam} ({planner}): {what} vs N')
            if row_i == 0:
                a.legend(fontsize=5)
    fig.tight_layout()
    p = os.path.join(out_dir, 'overlay_dose_N.png')
    fig.savefig(p)
    plt.close(fig)
    return p


def overlay_iterations(meas, e4c, out_dir, scale='32'):
    plt = _mpl()
    cl = e4c.get('C6', {}).get('closed_loop', {})
    it_curves = e4c.get('C6_curves', {})
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for i, (fam, planner, nref) in enumerate((('tdmpc2', 'mppi', 6), ('dinowm', 'cem', 30))):
        a = ax[i]
        for model, colr in (('M1', '#c0392b'), ('R-fixed', '#e67e22'), ('M3', '#2471a3'), ('M2', '#5dade2')):
            c = it_curves.get(f'{planner}|sep|{model}|{scale}')
            if c:
                ys = np.asarray(c['dmg'], float)
                y0 = _ref_norm(c['n'], ys, nref)
                a.plot(c['n'], ys / y0, '-', color=colr, lw=1, label=f'sim {model}')
        if planner == 'mppi':
            for m, colr in (('M1j', '#922b21'), ('Mstep', '#d35400'), ('M3', '#1f618d')):
                xs, ys = [], []
                for n_it in (1, 3, 6, 12):
                    v = cl.get(f'ws1|n{n_it}|{m}|{scale}')
                    if v:
                        xs.append(n_it)
                        ys.append(v['rel_damage'])
                if len(xs) > 1:
                    y0 = _ref_norm(xs, ys, nref)
                    a.plot(xs, np.array(ys) / y0, '--', color=colr, lw=1, label=f'sim closed loop {m}')
        M = [m for m in meas if m.get('family') == fam and m.get('iterations') is not None and m.get('variant') in ('det', 'shiter', 'shstep')]
        groups = {}
        for m in M:
            groups.setdefault((m.get('task'), m.get('variant'), m.get('warm_start')), []).append(m)
        for (task, var, ws), L in groups.items():
            L = sorted(L, key=lambda m: m['iterations'])
            xs = np.array([m['iterations'] for m in L], float)
            ys = np.array([m.get('D', np.nan) for m in L], float)
            y0 = _ref_norm(xs, ys, nref)
            if np.isfinite(y0):
                a.plot(xs, ys / y0, {'det': 'o', 'shiter': 's', 'shstep': '^'}[var], ms=3.5,
                       label=f'{task} {var}' + (f' ws{ws}' if ws is not None else ''))
        a.set_xscale('log')
        a.set_xlabel('iterations')
        a.set_ylabel(f'damage / value at n={nref}')
        a.set_title(f'{fam}: damage vs iterations')
        a.legend(fontsize=5)
    fig.tight_layout()
    p = os.path.join(out_dir, 'overlay_iterations.png')
    fig.savefig(p)
    plt.close(fig)
    return p


def overlay_neff(meas, e4c, out_dir):
    plt = _mpl()
    C9 = e4c.get('C9', {})
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
    fits = C9.get('fits', {})
    for key, colr in (('coh|0.03', '#1b4f72'), ('coh|0.3', '#5dade2'), ('incoh|0.03', '#a04000'), ('incoh|0.3', '#e59866')):
        f_ = fits.get(key)
        if f_:
            ne = np.array(f_['n_eff'], float)
            v = np.array(f_['var_exact'], float)
            ax[0].plot(ne, v / v[-1], '-', color=colr, lw=1, label=f'theory exact, toy {key}')
    ne = np.logspace(0, np.log10(4096), 30)
    ax[0].plot(ne, ne[-1] / ne, 'k:', lw=0.8, label='1/N_eff')
    pv = C9.get('planner_vs_neff', {})
    for key, colr in (('mppi|sep|coh|32', '#c0392b'), ('cem|sep|coh|32', '#2471a3'), ('mppi|sep|incoh|32', '#e59866'),
                      ('cem|sep|incoh|32', '#85c1e9')):
        if key in pv:
            L = pv[key]['points']
            xs = np.array([x['n_eff'] for x in L], float)
            ys = np.array([x['dmg'] for x in L], float)
            ax[1].plot(xs, ys / ys[-1], '-', color=colr, lw=1, label=f'sim {key}')
    for fam, mk in (('tdmpc2', 'o'), ('dinowm', 's')):
        M = [m for m in meas if m.get('family') == fam and m.get('n_eff') is not None]
        groups = {}
        for m in M:
            groups.setdefault((m.get('task'), m.get('variant')), []).append(m)
        for (task, var), L in groups.items():
            L = sorted(L, key=lambda m: m['n_eff'])
            xs = np.array([m['n_eff'] for m in L], float)
            ys = np.array([m.get('D', np.nan) for m in L], float)
            if np.isfinite(ys[-1]) and ys[-1] != 0:
                ax[1].plot(xs, ys / ys[-1], mk, ms=3.5, label=f'{task} {var}')
            vc = np.array([m.get('var_cand', np.nan) for m in L], float)
            if np.isfinite(vc).sum() > 1:
                ax[0].plot(xs, vc / vc[-1], mk, ms=3.5, label=f'{task} var_cand')
    for a, t in ((ax[0], 'across-candidate variance of a fixed table'), (ax[1], 'fixed-table damage')):
        a.set_xscale('log')
        a.set_yscale('log')
        a.set_xlabel('N_eff')
        a.set_ylabel('relative to the largest N_eff')
        a.set_title(t)
        a.legend(fontsize=5)
    fig.tight_layout()
    p = os.path.join(out_dir, 'overlay_neff.png')
    fig.savefig(p)
    plt.close(fig)
    return p


def overlay_decomposition(meas, e4c, tabs, out_dir):
    plt = _mpl()
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for i, fam in enumerate(('tdmpc2', 'dinowm')):
        M = [m for m in meas if m.get('family') == fam and 'drift' in m and 'sel_last' in m]
        labels, sel, dr, pred = [], [], [], []
        for m in M:
            labels.append(f"{m.get('task', '?')}:{m.get('k', '')} {m.get('variant', '?')}")
            sel.append(m['sel_last'])
            dr.append(m['drift'])
            pred.append(predict_sel_last(m, tabs))
        x = np.arange(len(labels))
        ax[i].bar(x - 0.2, sel, 0.4, color='#e67e22', label='sel_last (measured)')
        ax[i].bar(x + 0.2, dr, 0.4, color='#8e44ad', label='drift (measured)')
        if np.isfinite(pred).any():
            ax[i].plot(x - 0.2, pred, 'k_', ms=10, label='sel_last predicted (winner\'s curse)')
        ax[i].axhline(0, color='k', lw=0.5)
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(labels, rotation=60, fontsize=5)
        ax[i].set_title(f'{fam}: OEI = sel_last + drift')
        ax[i].legend(fontsize=5)
    fig.tight_layout()
    p = os.path.join(out_dir, 'overlay_decomposition.png')
    fig.savefig(p)
    plt.close(fig)
    return p


def run(e4c_path, tabs_path, out_dir, meas_paths):
    os.makedirs(out_dir, exist_ok=True)
    e4c = json.load(open(e4c_path))
    tabs = F.Tables(json.load(open(tabs_path)))
    meas = load_measurements(meas_paths)
    outs = [overlay_dose(meas, e4c, tabs, out_dir), overlay_iterations(meas, e4c, out_dir),
            overlay_neff(meas, e4c, out_dir), overlay_decomposition(meas, e4c, tabs, out_dir)]
    json.dump(dict(n_cells=len(meas), files=outs, cells=meas), open(os.path.join(out_dir, 'overlay_cells.json'), 'w'),
              indent=1, default=float)
    return meas, outs


# ------------------------------------------------------------------------------------------------ self test (synthetic)
def synthetic_inputs(e4c, out_dir):
    """SYNTHETIC E2/E3-style analysis JSONs built from the E4 simulation curves (no experiment data). Two layouts:
    TD-MPC2-like {"cells": {"<task>:<k>:N<N>:q<q>": {"variants": {...}}}} and DINO-WM-like nested
    {"tasks": {task: {"sweep_N": {"N": {"<N>": {"det": {...}}}}}}}, plus an E3-like N_eff sweep."""
    rng = np.random.default_rng(0)
    curves = e4c.get('C7_curves', {})
    td = {'note': 'SYNTHETIC test input for e4_overlay.py (built from E4 simulation curves; not experiment data)',
          'cells': {}}
    for var, model in (('det', 'M1'), ('shiter', 'M3')):
        c = curves.get(f'mppi|sep|q125|{model}|32.0')
        if not c:
            continue
        for N, ao, dm, sel, dr in zip(c['N'], c['abs_oei'], c['dmg'], c['sel'], c['drift']):
            key = f'syn-mw-push:3:N{N}:q0.125'
            cell = td['cells'].setdefault(key, {'task': 'syn-mw-push', 'k': 3, 'N': N, 'elite_ratio': 0.125,
                                                'iterations': 6, 'variants': {}})
            f_e = float(rng.normal(0.5, 0.05))
            cell['variants'][var] = {'D': float(dm), 'D_ci': [float(dm) * 0.8, float(dm) * 1.2],
                                     'oei': float(-ao), 'oei_ci': [float(-ao) * 1.15, float(-ao) * 0.85],
                                     'f_e_rand': f_e, 'l_e_rand': f_e + float(dr), 'e_chosen': f_e + float(dr) + float(sel),
                                     'l_e_rand_sd': 0.3 * float(ao) + 1e-3, 'l_sd_v0': 0.05 * float(ao) + 1e-3}
    dn = {'note': 'SYNTHETIC test input for e4_overlay.py (built from E4 simulation curves; not experiment data)',
          'tasks': {'syn-pusht': {'sweep_N': {'N': {}}}}}
    for var, model in (('det', 'M1'), ('shiter', 'M3')):
        c = curves.get(f'cem|sep|q125|{model}|32.0')
        if not c:
            continue
        for N, ao, dm, dr in zip(c['N'], c['abs_oei'], c['dmg'], c['drift']):
            d = dn['tasks']['syn-pusht']['sweep_N']['N'].setdefault(str(N), {})
            it0 = float(rng.normal(0.3, 0.02))
            d[var] = {'D': float(dm), 'D_ci': [float(dm) * 0.7, float(dm) * 1.3], 'oei_ref_mf': float(-ao),
                      'oei_ref_mf_ci': [float(-ao) * 1.2, float(-ao) * 0.8], 'it0_rand_dJ': it0,
                      'last_pop_dJ': it0 + float(dr), 'mf_dJ': it0 - float(ao), 'last_elite_dJ': it0 + float(dr) - 0.01,
                      'opt_steps': 30, 'topk': max(2, round(0.125 * N))}
    e3 = {'note': 'SYNTHETIC test input for e4_overlay.py (built from E4 simulation curves; not experiment data)',
          'cells': {}}
    pv = e4c.get('C9', {}).get('planner_vs_neff', {}).get('mppi|sep|coh|32', {})
    for x in pv.get('points', []):
        e3['cells'][f"syn-cheetah:5:neff{x['n_eff']}"] = {'variants': {'shstep': {
            'D': float(x['dmg']), 'oei': float(x['oei']), 'n_eff': int(x['n_eff']),
            'var_cand': float(x['se0']) ** 2 if x.get('se0') else None}}}
    paths = []
    for name, obj in (('SYNTHETIC_e2_tdmpc2_like.json', td), ('SYNTHETIC_e2_dinowm_like.json', dn),
                      ('SYNTHETIC_e3_like.json', e3)):
        p = os.path.join(out_dir, name)
        json.dump(obj, open(p, 'w'), indent=1)
        paths.append(p)
    return paths


if __name__ == '__main__':
    if sys.argv[1] == '--selftest':
        e4c_path, tabs_path, out_dir = sys.argv[2], sys.argv[3], sys.argv[4]
        os.makedirs(out_dir, exist_ok=True)
        paths = synthetic_inputs(json.load(open(e4c_path)), out_dir)
        meas, outs = run(e4c_path, tabs_path, out_dir, paths)
        print('selftest: cells', len(meas), 'figures', outs)
        fams = {}
        for m in meas:
            fams[(m['family'], m.get('task'), m.get('variant'))] = fams.get((m['family'], m.get('task'), m.get('variant')), 0) + 1
        print(fams)
    else:
        run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
