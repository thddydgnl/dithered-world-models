"""R2-13 analysis: success by variant, drops with paired bootstrap CIs, McNemar, sh-iter reduction of the det drop,
prediction errors, last-iteration candidate rank correlations from simulator replays, S1-S4 per PLAN.md section 7.1,
two figures. usage: python analyze13.py [RES] [POPS]"""
import os, sys, json, glob
import numpy as np
from scipy.stats import rankdata, binomtest, wilcoxon

RES = sys.argv[1] if len(sys.argv) > 1 else '/workspace/results'
POPS = sys.argv[2] if len(sys.argv) > 2 else '/workspace/pops'
TASKS = ['pusht', 'wall']
VARS = ['none', 'det', 'ind', 'shiter', 'shfixed']
QV = ['det', 'ind', 'shiter', 'shfixed']
LAST = {'pusht': 29, 'wall': 9}
NB = 10000
RNG = np.random.RandomState(0)


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def wilson(s, n, z=1.96):
    if n == 0:
        return [float('nan')] * 2
    p = s / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


def mcnemar(a, b):
    """a, b: paired success arrays. Returns (#a only, #b only, exact two-sided p)."""
    n10 = int(np.sum(a & ~b)); n01 = int(np.sum(~a & b))
    p = 1.0 if n10 + n01 == 0 else float(binomtest(n10, n10 + n01, 0.5).pvalue)
    return n10, n01, p


def boot_ci(stat_fn, n, q=(2.5, 97.5)):
    idx = RNG.randint(0, n, size=(NB, n))
    vals = np.array([stat_fn(i) for i in idx], float)
    vals = vals[np.isfinite(vals)]
    return [float(np.percentile(vals, q[0])), float(np.percentile(vals, q[1]))], int(len(vals))


def load(task):
    p = f'{RES}/cem13_{task}_eval.jsonl'
    R = {}
    if os.path.exists(p):
        for ln in open(p):
            r = json.loads(ln); R[(r['var'], r['ep'])] = r
    ks = sorted({r['k'] for r in R.values() if r['var'] != 'none'})
    eps = sorted(set.intersection(*[{e for (v, e) in R if v == var} for var in VARS])) if R else []
    return R, (ks[0] if ks else None), eps


def rank_stats(task, k, eps):
    """last-iteration Spearman(J_v, margin) and Spearman(J_none, margin) on the same population; iteration-0
    Spearman(J_v, margin) on the shared iteration-0 population (truth from the none replay)."""
    out = {}
    last = LAST[task]
    tn = {}
    for e in eps:
        f = f'{POPS}/{task}_eval_none0_e{e}_truth.npz'
        if os.path.exists(f):
            tn[e] = np.load(f)
    for v in VARS:
        vk = 'none0' if v == 'none' else f'{v}{k}'
        rl, rln, r0, same0, g_last, g_arg, g0, spec0, specl, rJ0 = [], [], [], [], [], [], [], [], [], []
        for e in eps:
            fz, ft = f'{POPS}/{task}_eval_{vk}_e{e}.npz', f'{POPS}/{task}_eval_{vk}_e{e}_truth.npz'
            if os.path.exists(fz) and v != 'none':
                # (post hoc) exploitation of the quantization error: model-cost error J_v - J_none over the last
                # population and at the planner's chosen candidate, vs the same error on the shared iteration-0 population
                z = np.load(fz)
                zn = np.load(f'{POPS}/{task}_eval_none0_e{e}.npz')
                dl = z[f'i{last}_J'] - z[f'i{last}_J_none']
                g_last.append(float(dl.mean())); g_arg.append(float(dl[np.argmin(z[f'i{last}_J'])]))
                d0 = z['i0_J'] - zn['i0_J']
                g0.append(float(d0.mean()))
                # candidate-specific part of the cost error (what can reorder candidates), relative to the cost spread
                spec0.append(float(d0.std() / (zn['i0_J'].std() + 1e-12)))
                specl.append(float(dl.std() / (z[f'i{last}_J_none'].std() + 1e-12)))
                rJ0.append(spearman(z['i0_J'], zn['i0_J']))
            if not (os.path.exists(fz) and os.path.exists(ft)):
                continue
            z, t = np.load(fz), np.load(ft)
            y = t[f'i{last}_margin'].astype(float)
            rl.append(spearman(z[f'i{last}_J'], y))
            rln.append(spearman(z[f'i{last}_J_none'], y) if f'i{last}_J_none' in z.files else rl[-1])
            if e in tn:
                zn = np.load(f'{POPS}/{task}_eval_none0_e{e}.npz')
                same0.append(bool(np.allclose(z['i0_actions'], zn['i0_actions'])))
                r0.append(spearman(z['i0_J'], tn[e]['i0_margin'].astype(float)))
        d = dict(n=len(rl))
        if g_last:
            d.update(err_pop_last=float(np.mean(g_last)), err_argmin_last=float(np.mean(g_arg)),
                     err_pop_it0=float(np.mean(g0)), n_err=len(g_last),
                     spec_err_it0=float(np.mean(spec0)), spec_err_last=float(np.mean(specl)),
                     rho_J_vs_Jnone_it0=float(np.mean(rJ0)))
        if rl:
            rl, rln = np.array(rl), np.array(rln)
            d.update(rho_last=float(rl.mean()), rho_last_ci=boot_ci(lambda i: rl[i].mean(), len(rl))[0],
                     rho_last_Jnone=float(rln.mean()), d_rho_last=float((rl - rln).mean()),
                     d_rho_last_ci=boot_ci(lambda i: (rl - rln)[i].mean(), len(rl))[0],
                     rho_last_eps=[round(float(x), 4) for x in rl])
        if r0:
            d.update(rho_it0=float(np.mean(r0)), it0_pop_identical=f'{sum(same0)}/{len(same0)}')
        out[v] = d
    return out


def main():
    A = dict(tasks={})
    errs = {}
    for task in TASKS:
        R, k, eps = load(task)
        if not eps:
            continue
        n = len(eps)
        S = {v: np.array([R[(v, e)]['success'] for e in eps], bool) for v in VARS}
        M = {v: np.array([R[(v, e)]['margin'] for e in eps], float) for v in VARS}
        T = dict(k=k, n=n, eps=eps, success={}, drop={}, pairs={}, margin={})
        for v in VARS:
            s = int(S[v].sum())
            T['success'][v] = dict(s=s, rate=s / n, wilson=wilson(s, n))
            T['margin'][v] = dict(median=float(np.median(M[v])), mean=float(M[v].mean()))
            if v != 'none':
                dv = S['none'].astype(float) - S[v].astype(float)
                ci, _ = boot_ci(lambda i: dv[i].mean(), n)
                n10, n01, p = mcnemar(S['none'], S[v])
                pw = float(wilcoxon(M[v], M['none']).pvalue) if np.any(M[v] != M['none']) else 1.0
                T['drop'][v] = dict(D=float(dv.mean()), ci95=ci, none_only=n10, var_only=n01, mcnemar_p=p,
                                    margin_wilcoxon_p=pw, median_dmargin=float(np.median(M[v] - M['none'])))
        for a, b in [('shiter', 'det'), ('shfixed', 'det'), ('shiter', 'ind'), ('shiter', 'shfixed'), ('ind', 'det')]:
            n10, n01, p = mcnemar(S[a], S[b])
            dd = S[b].astype(float) - S[a].astype(float)          # = D_a - D_b (positive: a more harmful)
            ci, _ = boot_ci(lambda i: -dd[i].mean(), n)
            T['pairs'][f'{a}_vs_{b}'] = dict(a_only=n10, b_only=n01, mcnemar_p=p, D_a_minus_D_b=float(-dd.mean()),
                                             ci95=ci)
        Dn = S['none'].astype(float)

        def red(v, i=None):
            i = np.arange(n) if i is None else i
            dd = (Dn - S['det'].astype(float))[i].mean()
            dv = (Dn - S[v].astype(float))[i].mean()
            return (dd - dv) / dd if dd > 0 else np.nan
        for v in ['shiter', 'shfixed', 'ind']:
            ci, nb = boot_ci(lambda i: red(v, i), n)
            T[f'red_{v}'] = dict(r=float(red(v)) if np.isfinite(red(v)) else None, ci95=ci, n_boot_valid=nb)
        T['plan_dev'] = {v: float(np.mean([np.linalg.norm(np.array(R[(v, e)]['mu_final']) -
                                                           np.array(R[('none', e)]['mu_final'])) for e in eps]))
                         for v in QV}
        T['time_s'] = {v: float(np.mean([R[(v, e)]['time_s'] for e in eps])) for v in VARS}
        T['rank'] = rank_stats(task, k, eps)
        # prediction errors at the chosen k (main error file; det curve file as fallback for det)
        ef = f'{RES}/err_{task}_main.json'
        E = json.load(open(ef)) if os.path.exists(ef) else {}
        T['err'] = {}
        for src in ('ds', 'plan'):
            for v in QV:
                key = f'{v}{k}'
                if E and key in E.get(src, {}):
                    T['err'].setdefault(src, {})[v] = E[src][key]
        errs[task] = T['err']
        A['tasks'][task] = T
    # ---------------------------------------------------------------- criteria (PLAN 7.1)
    tk = A['tasks']
    D = {t: {v: tk[t]['drop'][v]['D'] for v in QV} for t in tk}
    qual = [t for t in tk if D[t]['det'] >= 0.15 - 1e-9]
    s1_cells = {}
    for t in qual:
        r = tk[t]['red_shiter']['r']
        s1_cells[t] = dict(r=r, ok=bool(r is not None and r >= 0.5))
    side, side_strict = {}, {}
    for t in tk:
        others = [u for u in tk if u != t]
        ok = True; ok_s = True
        for u in others:
            if u in qual:
                ok &= D[u]['shiter'] <= D[u]['det'] + 1e-9
            else:
                ok &= D[u]['shiter'] <= 0.05 + 1e-9
            ok_s &= D[u]['shiter'] <= 0.05 + 1e-9
        side[t], side_strict[t] = bool(ok), bool(ok_s)
    S1 = any(s1_cells[t]['ok'] and side[t] for t in s1_cells)
    S1_strict = any(s1_cells[t]['ok'] and side_strict[t] for t in s1_cells)

    def sig_better(t):
        p = tk[t]['pairs']['shfixed_vs_det']
        return p['mcnemar_p'] < 0.05 and p['a_only'] > p['b_only']
    s2_cells = {t: dict(shfixed_sig_better_than_det=sig_better(t), p=tk[t]['pairs']['shfixed_vs_det']['mcnemar_p'],
                        r_fixed=tk[t]['red_shfixed']['r']) for t in tk}
    S2 = any(not s2_cells[t]['shfixed_sig_better_than_det'] for t in qual)
    S2_literal = any(not s2_cells[t]['shfixed_sig_better_than_det'] for t in tk)
    S3_cells = {t: dict(D_shiter=D[t]['shiter'], D_ind=D[t]['ind'], ok=D[t]['shiter'] < D[t]['ind'] - 1e-9,
                        p=tk[t]['pairs']['shiter_vs_ind']['mcnemar_p']) for t in tk}
    S3 = any(c['ok'] for c in S3_cells.values())
    S4_cases = {'ds': [], 'plan': []}
    for src in ('ds', 'plan'):
        for t in tk:
            e = tk[t]['err'].get(src, {})
            for i, a in enumerate(QV):
                for b in QV[i + 1:]:
                    if a not in e or b not in e:
                        continue
                    ea, eb = e[a]['err1'], e[b]['err1']
                    rel = abs(ea - eb) / max(min(ea, eb), 1e-12)
                    hi, lo = max(D[t][a], D[t][b]), min(D[t][a], D[t][b])
                    if rel <= 0.05 and hi >= 0.10 - 1e-9 and hi >= 2 * max(lo, 0.0) - 1e-9:
                        S4_cases[src].append(dict(task=t, a=a, b=b, err_rel_diff=rel, D_a=D[t][a], D_b=D[t][b]))
    S4 = len(S4_cases['ds']) > 0
    verdict = '있음' if (S1 and S2) else ('조건부 있음' if S1 else '없음')
    A['criteria'] = dict(qualifying_tasks=qual, S1=S1, S1_cells=s1_cells, S1_side_ok=side, S1_strict=S1_strict,
                         S1_side_strict=side_strict, S2=S2, S2_literal_all_tasks=S2_literal, S2_cells=s2_cells,
                         S3=S3, S3_cells=S3_cells, S4=S4, S4_plan_err=len(S4_cases['plan']) > 0, S4_cases=S4_cases,
                         verdict=verdict)
    json.dump(A, open(f'{RES}/analysis13.json', 'w'), indent=1, default=float)
    # ---------------------------------------------------------------- table
    with open(f'{RES}/table_main.csv', 'w') as f:
        f.write('task,k,n,variant,success,wilson_lo,wilson_hi,drop_pp,drop_ci_lo,drop_ci_hi,mcnemar_p_vs_none,'
                'median_margin,ds_rel_err1,plan_rel_err1,rho_last,rho_last_Jnone,rho_it0\n')
        for t, T in tk.items():
            for v in VARS:
                sc = T['success'][v]; dr = T['drop'].get(v, dict(D=0, ci95=[0, 0], mcnemar_p=1))
                ed = T['err'].get('ds', {}).get(v, {}); ep = T['err'].get('plan', {}).get(v, {})
                rk = T['rank'].get(v, {})
                f.write(f"{t},{T['k']},{T['n']},{v},{sc['rate']:.3f},{sc['wilson'][0]:.3f},{sc['wilson'][1]:.3f},"
                        f"{100 * dr['D']:.1f},{100 * dr['ci95'][0]:.1f},{100 * dr['ci95'][1]:.1f},{dr['mcnemar_p']:.4f},"
                        f"{T['margin'][v]['median']:.3f},{ed.get('rel_err1', 0 if v == 'none' else float('nan')):.5f},"
                        f"{ep.get('rel_err1', 0 if v == 'none' else float('nan')):.5f},{rk.get('rho_last', float('nan')):.3f},"
                        f"{rk.get('rho_last_Jnone', float('nan')):.3f},{rk.get('rho_it0', float('nan')):.3f}\n")
    figures(tk)
    # ---------------------------------------------------------------- print
    for t, T in tk.items():
        print(f"== {t} k={T['k']} n={T['n']}")
        for v in VARS:
            sc = T['success'][v]; dr = T['drop'].get(v)
            rk = T['rank'].get(v, {})
            s = f"  {v:8s} {sc['s']:2d}/{T['n']} ({100 * sc['rate']:.0f}%)"
            if dr:
                s += (f" D {100 * dr['D']:+.0f}pp [{100 * dr['ci95'][0]:.0f},{100 * dr['ci95'][1]:.0f}] "
                      f"McN p {dr['mcnemar_p']:.3g} ({dr['none_only']}/{dr['var_only']})")
            s += f" med margin {T['margin'][v]['median']:.3f} rho_last {rk.get('rho_last', float('nan')):.3f}" \
                 f" (J_none {rk.get('rho_last_Jnone', float('nan')):.3f}, n {rk.get('n', 0)}) rho_it0 {rk.get('rho_it0', float('nan')):.3f}"
            if 'err_pop_last' in rk:
                s += (f" | cost err it0 {rk['err_pop_it0']:+.4f} last-pop {rk['err_pop_last']:+.4f}"
                      f" argmin {rk['err_argmin_last']:+.4f} | spec err it0 {rk['spec_err_it0']:.3f}"
                      f" last {rk['spec_err_last']:.3f} rho(J,Jnone) it0 {rk['rho_J_vs_Jnone_it0']:.3f}")
            print(s)
        for key in ('red_shiter', 'red_shfixed', 'red_ind'):
            print(f"  {key}: {T[key]}")
        for key, p in T['pairs'].items():
            print(f"  {key}: {p}")
        print('  err', {src: {v: round(d['rel_err1'], 5) for v, d in e.items()} for src, e in T['err'].items()})
        print('  plan_dev', {v: round(x, 3) for v, x in T['plan_dev'].items()}, 'time', {v: round(x, 1) for v, x in T['time_s'].items()})
    print(json.dumps(A['criteria'], indent=1, default=float))


def figures(tk):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ink, ink2, grid = '#0b0b0b', '#52514e', '#e4e3df'
    blue, gray = '#2a78d6', '#9a9893'
    names = {'none': 'none', 'det': 'det', 'ind': 'ind', 'shiter': 'sh-iter', 'shfixed': 'sh-fixed'}
    # fig 1: success by variant per task
    fig, axs = plt.subplots(1, len(tk), figsize=(4.2 * len(tk), 3.4), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, (t, T) in zip(axs, tk.items()):
        x = np.arange(len(VARS))
        r = np.array([T['success'][v]['rate'] for v in VARS]) * 100
        lo = np.array([T['success'][v]['wilson'][0] for v in VARS]) * 100
        hi = np.array([T['success'][v]['wilson'][1] for v in VARS]) * 100
        cols = [gray if v == 'none' else blue for v in VARS]
        ax.bar(x, r, width=0.62, color=cols, edgecolor='white', linewidth=2, zorder=2)
        ax.errorbar(x, r, yerr=[np.maximum(r - lo, 0), np.maximum(hi - r, 0)], fmt='none', ecolor=ink2, elinewidth=1,
                    capsize=3, zorder=3)
        for xi, ri in zip(x, r):
            ax.text(xi, 2, f'{ri:.0f}', ha='center', va='bottom', color='white', fontsize=9, fontweight='bold', zorder=4)
        ax.set_xticks(x); ax.set_xticklabels([names[v] for v in VARS], color=ink, fontsize=9)
        ax.set_title(f"{'PushT' if t == 'pusht' else 'Wall'} (k={T['k']}, n={T['n']})", color=ink, fontsize=10)
        ax.yaxis.grid(True, color=grid, zorder=0); ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.spines['left'].set_color(grid); ax.spines['bottom'].set_color(grid)
        ax.tick_params(colors=ink2)
        ax.set_ylim(0, 100)
    axs[0].set_ylabel('CEM success rate (%), 95% Wilson CI', color=ink2, fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{RES}/fig1_success_by_variant.png', dpi=150)
    plt.close(fig)
    # fig 2: 1-step error vs success drop (one panel per task; variant = color + marker shape, legend)
    cols = {'det': '#2a78d6', 'ind': '#eb6834', 'shiter': '#1baf7a', 'shfixed': '#eda100'}
    shp = {'det': 's', 'ind': '^', 'shiter': 'o', 'shfixed': 'D'}
    fig, axs = plt.subplots(1, len(tk), figsize=(4.4 * len(tk), 3.5), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, (t, T) in zip(axs, tk.items()):
        e = T['err'].get('ds', {})
        for v in QV:
            if v not in e:
                continue
            x = e[v]['rel_err1']; y = 100 * T['drop'][v]['D']; ci = T['drop'][v]['ci95']
            ax.errorbar([x], [y], yerr=[[max(y - 100 * ci[0], 0)], [max(100 * ci[1] - y, 0)]], fmt=shp[v], ms=8, color=cols[v],
                        mec='white', mew=1.2, ecolor=cols[v], elinewidth=1.2, capsize=3, zorder=3, label=names[v])
        ax.axhline(0, color=gray, lw=1, zorder=1)
        ax.set_title(f"{'PushT' if t == 'pusht' else 'Wall'} (k={T['k']}, n={T['n']})", color=ink, fontsize=10)
        ax.set_xlabel('1-step error vs unquantized (relative)', color=ink2, fontsize=8.5)
        ax.yaxis.grid(True, color=grid, zorder=0); ax.set_axisbelow(True)
        for s_ in ('top', 'right'):
            ax.spines[s_].set_visible(False)
        ax.spines['left'].set_color(grid); ax.spines['bottom'].set_color(grid)
        ax.tick_params(colors=ink2, labelsize=8)
        xs = [e[v]['rel_err1'] for v in QV if v in e]
        if xs:
            ax.set_xlim(0, max(xs) * 1.25)
    axs[0].set_ylabel('success drop vs none (pp), 95% CI', color=ink2, fontsize=8.5)
    h, l = axs[0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc='upper center', ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(f'{RES}/fig2_err1_vs_drop.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
