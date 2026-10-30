"""R2-7 analysis and figures: S1 (range-restriction oracle + resolution ratio), S2 (failure AUROC), S3 (evaluation of the
pre-chosen schedule), S4 summary, verdict. usage: python analyze7.py [RES] [POPS]"""
import os, sys, glob, json
import numpy as np
from scipy.stats import rankdata, binomtest, wilcoxon

RES = sys.argv[1] if len(sys.argv) > 1 else '/workspace/results'
POPS = sys.argv[2] if len(sys.argv) > 2 else '/workspace/pops'
STEPS = {'pusht': [0, 4, 9, 14, 19, 24, 29], 'wall': [0, 2, 4, 6, 9]}
TASKS = ['pusht', 'wall']
M = 64
XI = np.random.RandomState(0).randn(M, 300)
RNG = np.random.RandomState(1)


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def oracle_rho(y, sigma):
    if not np.isfinite(sigma):
        return 0.0
    Yn = y[None] + sigma * XI[:, :len(y)]
    R = np.apply_along_axis(rankdata, 1, Yn); ry = rankdata(y)
    R = R - R.mean(1, keepdims=True); ry = ry - ry.mean()
    return float(np.mean((R * ry).sum(1) / np.sqrt((R ** 2).sum(1) * (ry ** 2).sum())))


def fit_sigma(y0, target):
    if target <= 0:
        return np.inf
    sd = y0.std(); lo, hi = 1e-4 * sd, 1e3 * sd
    for _ in range(30):
        mid = np.sqrt(lo * hi)
        if oracle_rho(y0, mid) > target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def boot_ci(x, n=2000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    bs = [RNG.choice(x, len(x)).mean() for _ in range(n)]
    return [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def auroc(score, fail):
    score = np.asarray(score, float); fail = np.asarray(fail, bool)
    a, b = score[fail], score[~fail]
    if len(a) == 0 or len(b) == 0:
        return float('nan')
    return float(((a[:, None] > b[None]).sum() + 0.5 * (a[:, None] == b[None]).sum()) / (len(a) * len(b)))


def auroc_ci(score, fail, n=2000):
    score = np.asarray(score, float); fail = np.asarray(fail, bool); idx = np.arange(len(score))
    bs = []
    for _ in range(n):
        i = RNG.choice(idx, len(idx))
        v = auroc(score[i], fail[i])
        if np.isfinite(v):
            bs.append(v)
    return [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def load_jsonl(task, split, sched):
    R = {}
    p = f'{RES}/cem7_{task}_{split}_{sched}.jsonl'
    if os.path.exists(p):
        for ln in open(p):
            r = json.loads(ln); R[r['ep']] = r
    return R


def es_outcome(r, tau):
    for p in r['prox']:
        if p['step'] >= 1 and p['R_boot'] > tau:
            return r['traj_success'][p['step']], r['traj_margin'][p['step']], p['step']
    return r['traj_success'][-1], r['traj_margin'][-1], None


def paired(task, A, B):
    eps = sorted(set(A) & set(B))
    sa = np.array([A[e][0] for e in eps], bool); sb = np.array([B[e][0] for e in eps], bool)
    ma = np.array([A[e][1] for e in eps]); mb = np.array([B[e][1] for e in eps])
    n10, n01 = int((sa & ~sb).sum()), int((~sa & sb).sum())
    p = float(binomtest(min(n10, n01), n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
    try:
        pw = float(wilcoxon(mb - ma).pvalue) if np.any(mb != ma) else 1.0
    except ValueError:
        pw = 1.0
    return dict(n=len(eps), succ_base=float(sa.mean()), succ_rule=float(sb.mean()), delta_pp=float(100 * (sb.mean() - sa.mean())),
                base_only=n10, rule_only=n01, mcnemar_p=p, median_dmargin=float(np.median(mb - ma)), wilcoxon_p=pw)


def main():
    out = dict(S1={}, S2={}, S3={}, S4={}, curves={})
    per_ep = {}
    for task in TASKS:
        rows = []
        for split in ['held', 'eval']:
            base = load_jsonl(task, split, 'base')
            for e, r in sorted(base.items()):
                fz, ft = f'{POPS}/{task}_{split}_e{e}.npz', f'{POPS}/{task}_{split}_e{e}_truth.npz'
                if not (os.path.exists(fz) and os.path.exists(ft)):
                    continue
                z, t = np.load(fz), np.load(ft)
                y0, J0 = t['s0_margin'].astype(float), z['s0_J'].astype(float)
                b, a = np.polyfit(J0, y0, 1)
                rho0 = spearman(J0, y0)
                sig = fit_sigma(y0, rho0)
                sig2 = float((y0 - (a + b * J0)).std())
                st = []
                for s in STEPS[task]:
                    y, J = t[f's{s}_margin'].astype(float), z[f's{s}_J'].astype(float)
                    err = a + b * J - y
                    st.append(dict(step=s, rho=spearman(J, y), rho_or=oracle_rho(y, sig), rho_or2=oracle_rho(y, sig2),
                                   R=float(err.std() / max(y.std(), 1e-12)),
                                   R_or=float(sig / max(y.std(), 1e-12)) if np.isfinite(sig) else float('inf'),
                                   sd_y=float(y.std()), rho_sd=spearman(J, t[f's{s}_sd']),
                                   rho_V=spearman(z[f's{s}_V'], y), rho_Q=spearman(z[f's{s}_Q'], y),
                                   J=J, y=y))
                rows.append(dict(split=split, ep=e, success=r['success'], margin=r['margin'], sigma=sig, sigma2=sig2,
                                 slope=float(b), steps=st, prox_last=r['prox'][-1], prox_first=r['prox'][0],
                                 traj_margin=r['traj_margin'], traj_success=r['traj_success']))
        per_ep[task] = rows
        # pooled calibration (secondary R)
        J0all = np.concatenate([rw['steps'][0]['J'] for rw in rows]); y0all = np.concatenate([rw['steps'][0]['y'] for rw in rows])
        bp, ap = np.polyfit(J0all, y0all, 1)
        # ---- S1 summary per step
        summ = []
        for k, s in enumerate(STEPS[task]):
            rho = [rw['steps'][k]['rho'] for rw in rows]; ror = [rw['steps'][k]['rho_or'] for rw in rows]
            ror2 = [rw['steps'][k]['rho_or2'] for rw in rows]
            d = np.array(ror) - np.array(rho); d2 = np.array(ror2) - np.array(rho)
            Rp = [float((ap + bp * rw['steps'][k]['J'] - rw['steps'][k]['y']).std() / max(rw['steps'][k]['sd_y'], 1e-12)) for rw in rows]
            summ.append(dict(step=s, n=len(rows), rho_model=float(np.mean(rho)), rho_oracle=float(np.mean(ror)),
                             rho_oracle2=float(np.mean(ror2)), delta=float(d.mean()), delta_ci=boot_ci(d),
                             delta2=float(d2.mean()), delta2_ci=boot_ci(d2),
                             R_median=float(np.median([rw['steps'][k]['R'] for rw in rows])),
                             R_oracle_median=float(np.median([rw['steps'][k]['R_or'] for rw in rows])),
                             R_pooled_median=float(np.median(Rp)),
                             frac_R_gt1=float(np.mean([rw['steps'][k]['R'] > 1 for rw in rows])),
                             sd_y_mean=float(np.mean([rw['steps'][k]['sd_y'] for rw in rows])),
                             rho_sd=float(np.mean([rw['steps'][k]['rho_sd'] for rw in rows])),
                             rho_V=float(np.mean([rw['steps'][k]['rho_V'] for rw in rows])),
                             rho_Q=float(np.mean([rw['steps'][k]['rho_Q'] for rw in rows]))))
        last = summ[-1]
        out['S1'][task] = dict(per_step=summ, n_episodes=len(rows),
                               pass_R=bool(last['R_median'] > 1), pass_delta=bool(last['delta'] >= 0.1),
                               pass_=bool(last['R_median'] > 1 and last['delta'] >= 0.1))
        # ---- S2 AUROC (failure)
        fail = np.array([not rw['success'] for rw in rows])
        preds = {
            'R_last (GT, primary)': [rw['steps'][-1]['R'] for rw in rows],
            'R_boot_last (GT-free, primary)': [rw['prox_last']['R_boot'] for rw in rows],
            '-boot_rank_last (secondary)': [-rw['prox_last']['boot_rank'] for rw in rows],
            '-rho_VQ_last (secondary)': [-rw['prox_last']['rho_VQ'] for rw in rows],
            'sdJ_last/sdJ_first (secondary)': [rw['prox_last']['sd_J'] / rw['prox_first']['sd_J'] for rw in rows],
            '-rho_last (GT, secondary)': [-rw['steps'][-1]['rho'] for rw in rows],
        }
        s2 = dict(n=len(rows), n_fail=int(fail.sum()))
        for k, v in preds.items():
            s2[k] = dict(auroc=auroc(v, fail), ci=auroc_ci(v, fail))
        s2['pass_'] = bool(max(s2['R_last (GT, primary)']['auroc'], s2['R_boot_last (GT-free, primary)']['auroc']) >= 0.65)
        out['S2'][task] = s2
        # ---- trajectory curves (eval baseline): mean true margin / success of the plan after each opt step
        ev = [rw for rw in rows if rw['split'] == 'eval']
        if ev:
            tm = np.array([rw['traj_margin'] for rw in ev]); ts = np.array([rw['traj_success'] for rw in ev], float)
            out['curves'][task] = dict(mean_margin=tm.mean(0).tolist(), median_margin=np.median(tm, 0).tolist(),
                                       success=ts.mean(0).tolist(), best_step_success=float(ts.max(1).mean()),
                                       final_success=float(ts[:, -1].mean()))
    # ---- S3
    sel = json.load(open(f'{RES}/select7.json')) if os.path.exists(f'{RES}/select7.json') else None
    if sel:
        ch = sel['chosen']; taus = sel['taus']
        out['S3']['chosen'] = ch['rule']; out['S3']['held_rows'] = sel['rows']
        res = {}
        for task in TASKS:
            base = load_jsonl(task, 'eval', 'base')
            A = {e: (r['success'], r['margin']) for e, r in base.items()}
            if ch['kind'] == 'es':
                B = {e: es_outcome(r, taus[task][str(ch['q'])])[:2] for e, r in base.items()}
            else:
                RR = load_jsonl(task, 'eval', ch['kind'])
                B = {e: (r['success'], r['margin']) for e, r in RR.items()}
            res[task] = paired(task, A, B)
            # exploratory: every ES(q) on eval, and oracle (ground-truth) best-step stopping
            for q in ['0.25', '0.5', '0.75']:
                res[task][f'explor_ES({q})'] = paired(task, A, {e: es_outcome(r, taus[task][q])[:2] for e, r in base.items()})
            res[task]['explor_oracle_best_step'] = paired(task, A, {e: (bool(max(r['traj_success'])),
                                                                       float(min(r['traj_margin']))) for e, r in base.items()})
        d = [res[t]['delta_pp'] for t in TASKS]
        out['S3']['eval'] = res
        out['S3']['pass_'] = bool(all(res[t]['n'] == 50 for t in TASKS) and max(d) >= 5 and min(d) >= -3)
    # ---- S4
    for task in TASKS:
        p = f'{RES}/s4_{task}_a0.2.json'
        if os.path.exists(p):
            s4 = json.load(open(p))
            out['S4'][task] = {k: v for k, v in s4.items() if k != 'rows'}
    if out['S4']:
        out['S4']['pass_'] = bool(any(v.get('S4_pass') for k, v in out['S4'].items() if isinstance(v, dict)))
    s1 = any(out['S1'][t]['pass_'] for t in TASKS)
    s2 = any(out['S2'][t]['pass_'] for t in TASKS)
    s3 = bool(out['S3'].get('pass_'))
    out['verdict'] = '있음' if (s1 and (s2 or s3)) else ('조건부' if s1 else '없음')
    out['flags'] = dict(S1=s1, S2=s2, S3=s3, S4=out['S4'].get('pass_'))
    json.dump(out, open(f'{RES}/analysis7.json', 'w'), indent=1, default=float)
    # console
    for task in TASKS:
        print('=====', task, 'episodes', out['S1'][task]['n_episodes'])
        for r in out['S1'][task]['per_step']:
            print('  step %2d  rho %.3f  oracle %.3f (d %+.3f [%.3f,%.3f])  oracle2 %.3f  R_med %.2f  R_or %.2f  R_pool %.2f  sd_y %.3f  rhoV %.2f rhoQ %.2f' % (
                r['step'], r['rho_model'], r['rho_oracle'], r['delta'], *r['delta_ci'], r['rho_oracle2'], r['R_median'],
                r['R_oracle_median'], r['R_pooled_median'], r['sd_y_mean'], r['rho_V'], r['rho_Q']))
        print('  S1 pass', out['S1'][task]['pass_'])
        print('  S2', {k: (round(v['auroc'], 3) if isinstance(v, dict) else v) for k, v in out['S2'][task].items()})
        if task in out['curves']:
            c = out['curves'][task]
            print('  plan margin by step', [round(x, 2) for x in c['mean_margin']], 'success by step', [round(x, 2) for x in c['success']])
    if out['S3']:
        print('S3 chosen', out['S3'].get('chosen'))
        for t in TASKS:
            if t in out['S3'].get('eval', {}):
                print(' ', t, {k: v for k, v in out['S3']['eval'][t].items() if not k.startswith('explor')})
                for k, v in out['S3']['eval'][t].items():
                    if k.startswith('explor'):
                        print('    ', k, 'd %+.1f pp (%d/%d) p=%.3f' % (v['delta_pp'], v['base_only'], v['rule_only'], v['mcnemar_p']))
    print('S4', json.dumps(out['S4'])[:1500])
    print('VERDICT', out['verdict'], out['flags'])
    figures(out)


def figures(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
    INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e6e5e1'
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                         'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                         'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                         'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb'})
    names = {'pusht': 'PushT', 'wall': 'Wall'}
    # fig1: Spearman model vs oracle, and R
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.6))
    for j, t in enumerate(TASKS):
        S = out['S1'][t]['per_step']; x = [r['step'] + 1 for r in S]
        ax = axes[0, j]
        ax.plot(x, [r['rho_model'] for r in S], color=BLUE, lw=1.8, marker='o', ms=4, label='world-model cost')
        ax.plot(x, [r['rho_oracle'] for r in S], color=MUTED, lw=1.4, ls='--', marker='s', ms=4,
                label='constant-noise oracle (Spearman-matched at step 1)')
        ax.plot(x, [r['rho_oracle2'] for r in S], color=MUTED, lw=1.0, ls=':', marker='^', ms=3,
                label='oracle, noise = step-1 calibration residual')
        ax.fill_between(x, [r['rho_oracle'] - r['delta_ci'][1] for r in S], [r['rho_oracle'] - r['delta_ci'][0] for r in S],
                        color=BLUE, alpha=0.12, lw=0)
        ax.set_ylim(0, 1.02); ax.set_xlabel('CEM opt step'); ax.set_ylabel('Spearman(cost, true cost) in population')
        ax.set_title(f'{names[t]}: ranking vs range-restriction oracle', fontsize=9.5, color=INK)
        ax = axes[1, j]
        ax.plot(x, [r['R_median'] for r in S], color=BLUE, lw=1.8, marker='o', ms=4, label='R (model, median)')
        ax.plot(x, [r['R_oracle_median'] for r in S], color=MUTED, lw=1.4, ls='--', marker='s', ms=4, label='R of the oracle noise')
        ax.axhline(1, color=INK2, lw=0.8, ls='-.')
        ax.text(x[0], 1.05, 'R = 1', fontsize=7.5, color=INK2)
        ax.set_yscale('log'); ax.set_xlabel('CEM opt step'); ax.set_ylabel('sd(cost error) / sd(true cost)')
        ax.set_title(f'{names[t]}: resolution ratio R', fontsize=9.5, color=INK)
    axes[0, 0].legend(fontsize=6.8, loc='lower left'); axes[1, 0].legend(fontsize=6.8, loc='upper left')
    fig.tight_layout(); fig.savefig(f'{RES}/fig1_rank_vs_oracle.png', dpi=150)
    # fig2: AUROC
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    keys = list(k for k in out['S2']['pusht'] if isinstance(out['S2']['pusht'][k], dict))
    xx = np.arange(len(keys))
    for i, (t, col) in enumerate(zip(TASKS, [BLUE, ORANGE])):
        v = [out['S2'][t][k]['auroc'] for k in keys]
        lo = [out['S2'][t][k]['auroc'] - out['S2'][t][k]['ci'][0] for k in keys]
        hi = [out['S2'][t][k]['ci'][1] - out['S2'][t][k]['auroc'] for k in keys]
        ax.bar(xx + (i - 0.5) * 0.38, v, 0.36, color=col, label=f"{names[t]} (n={out['S2'][t]['n']}, fail={out['S2'][t]['n_fail']})",
               yerr=[lo, hi], capsize=2, error_kw=dict(ecolor=INK2, lw=0.7))
    ax.axhline(0.65, color=INK2, lw=0.8, ls='-.', label='S2 threshold 0.65')
    ax.axhline(0.5, color=MUTED, lw=0.6)
    ax.set_xticks(xx); ax.set_xticklabels([k.replace(' (', '\n(') for k in keys], fontsize=6.8)
    ax.set_ylim(0, 1.18); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_ylabel('AUROC for episode failure')
    ax.legend(fontsize=7, loc='upper center', ncol=3)
    ax.set_title('Does last-step resolution predict failure? (95% bootstrap CI)', fontsize=9.5, color=INK)
    fig.tight_layout(); fig.savefig(f'{RES}/fig2_failure_auroc.png', dpi=150)
    # fig3: plan quality along CEM and schedule evaluation
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3))
    ax = axes[0]
    for t, col in zip(TASKS, [BLUE, ORANGE]):
        if t in out['curves']:
            c = out['curves'][t]
            ax.plot(np.arange(1, len(c['success']) + 1), [100 * s for s in c['success']], color=col, lw=1.6,
                    label=f"{names[t]} (best-step oracle {100*c['best_step_success']:.0f}%)")
    ax.set_xlabel('CEM opt step at which the plan is executed'); ax.set_ylabel('success of the current plan (%)')
    ax.set_ylim(0, 100); ax.legend(fontsize=7, loc='lower right')
    ax.set_title('Baseline CEM, eval split: plan success by step', fontsize=9.5, color=INK)
    ax = axes[1]
    if out['S3'].get('eval'):
        labels = ['baseline', out['S3']['chosen']]
        for i, (t, col) in enumerate(zip(TASKS, [BLUE, ORANGE])):
            r = out['S3']['eval'][t]
            vals = [100 * r['succ_base'], 100 * r['succ_rule']]
            ax.bar(np.arange(2) + (i - 0.5) * 0.38, vals, 0.36, color=col, label=f"{names[t]} (McNemar p={r['mcnemar_p']:.2f})")
            for k, v in enumerate(vals):
                ax.text(k + (i - 0.5) * 0.38, v + 1.5, f'{v:.0f}%', ha='center', fontsize=7.5, color=INK)
        ax.set_xticks(np.arange(2)); ax.set_xticklabels(labels)
        ax.set_ylim(0, 110); ax.set_ylabel('success, eval split (n=50, same seeds)'); ax.legend(fontsize=7, loc='upper right')
        ax.set_title('Pre-chosen resolution-aware schedule', fontsize=9.5, color=INK)
    fig.tight_layout(); fig.savefig(f'{RES}/fig3_schedule.png', dpi=150)


if __name__ == '__main__':
    main()
