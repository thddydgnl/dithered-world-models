"""E13 (DESIGN_tmlr.md v1.0 section 3), DINO-WM part: offline diagnostics on common planning states.

Input states (all variants use the same ones): the none (CEM) runs of this pod on the held pairs (20) and on eval8 (100).
Per state two candidate sets: the iteration-0 population (300 N(0,1) sequences, row 0 = 0; identical for every variant)
and the none planner's last-iteration population (near-tie region). No environment rollout.

Per (state, set, configuration = (ranges id, variant, k[, bias])), against the unquantized model on the same set:
  rel1 / rel3 / rel5   latent error, E1 definition: mean over candidates of (visual MSE + alpha proprio MSE) between the
                       quantized and the unquantized prediction at horizon step s, divided by the variance of the
                       unquantized predictions across the candidates
  rew_err              "reward error": DINO-WM has no reward; its planning cost J (final-step latent distance to the goal)
                       takes the role of the return, so rew_err = median_a |J_q(a) - J_0(a)| / std_a J_0
  e(a) = J_q - J_0     cost (negative return) error: common part mean_a e, share mean_e^2 / (mean_e^2 + sd_e^2), sd_e,
                       sd_J = std_a J_0, ratio sd_e / sd_J
  rho_p                persistence: Pearson correlation over candidates of e(a) under the dither states of two consecutive
                       planner iterations (it = i0 and i0 + 1, same plan). det, detbc and the plan-fixed tables (shstep,
                       shep, detshiftfix) have the same error field in both -> 1 exactly (not re-evaluated)
  spearman             Spearman(J_q, J_0) over the candidates; elite_overlap = |top30(J_q) & top30(J_0)| / 30
Draws: A (it = i0) and, for fresh variants, B (it = i0 + 1); latent/cost metrics are the mean over the draws. i0 = 0 for
the iteration-0 set and opt_steps - 1 for the last set. Dither salt 'e13|<split>', planner episode id e.
Results: one JSON per state (out/e13/<task>/<split>_e<e>.json) with every configuration; resumable."""
import os, sys, json, time
import numpy as np
import torch
from scipy.stats import spearmanr, rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wmlib as L
import qlib as QL

FRESH = ('ind', 'shiter', 'detshift', 'shiterb', 'detnoise')       # new table/noise at every forward call
FIXED = ('det', 'detbc', 'shstep', 'shep', 'detshiftfix', 'shfixed')  # same error field across iterations of a plan
TOPK = 30


def cfg_key(c):
    return f"{c['ranges']}|{c['var']}|k{c['k']}"


def _metrics(J0, Jq_list, steps_list, pv):
    """J0 (N,) float64; Jq_list: per draw (N,); steps_list: per draw {s: abs err}; pv {s: pred var}."""
    out = {}
    for s in QL.ERR_STEPS:
        out[f'rel{s}'] = float(np.mean([d[s] for d in steps_list]) / pv[s])
    sdJ = float(np.std(J0))
    es = [Jq - J0 for Jq in Jq_list]
    out['rew_err'] = float(np.mean([np.median(np.abs(e)) / sdJ for e in es]))
    out['mean_e'] = float(np.mean([e.mean() for e in es]))
    out['sd_e'] = float(np.mean([e.std() for e in es]))
    out['sd_J'] = sdJ
    out['ratio'] = out['sd_e'] / sdJ
    out['common_share'] = float(np.mean([e.mean() ** 2 / (e.mean() ** 2 + e.var() + 1e-30) for e in es]))
    rs, ov = [], []
    top0 = set(np.argsort(J0, kind='stable')[:TOPK].tolist())
    for Jq in Jq_list:
        r = spearmanr(Jq, J0).correlation
        rs.append(float(r) if np.isfinite(r) else float('nan'))
        ov.append(len(top0 & set(np.argsort(Jq, kind='stable')[:TOPK].tolist())) / TOPK)
    out['spearman'] = float(np.mean(rs)); out['elite_overlap'] = float(np.mean(ov))
    if len(es) == 2:
        a, b = es
        den = a.std() * b.std()
        out['rho_p'] = float(((a - a.mean()) * (b - b.mean())).mean() / den) if den > 0 else float('nan')
    return out


@torch.no_grad()
def _rollout(m, Q, inp, actions):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        zv, zp, va, pa = QL.rollout_q(m, Q, inp[0], inp[1], actions[None], all_frames=True)
    return zv.float(), zp.float(), va, pa


@torch.no_grad()
def run_state(ctx, split, e, sets, configs, out_path, set_cfg):
    """sets: {name: (actions (N,H,A) cpu tensor, i0)}; configs: list of dicts (ranges, var, k, bias)."""
    res = json.load(open(out_path)) if os.path.exists(out_path) else dict(task=ctx.task, split=split, ep=e, sets={})
    todo = [c for c in configs if any(cfg_key(c) not in res['sets'].get(s, {}) for s in sets)]
    if not todo:
        return res, 0
    pairs = ctx.get_pairs(split)
    inp = QL.episode_inputs(ctx.m, ctx.pre, pairs, e)
    m, Q, alpha = ctx.m, ctx.Q, ctx.alpha
    salt = f'e13|{split}'
    n_roll = 0
    for sname, (acts, i0) in sets.items():
        S = res['sets'].setdefault(sname, {})
        need = [c for c in todo if cfg_key(c) not in S]
        if not need:
            continue
        a = acts.cuda()
        Q.configure('none')
        zv0, zp0, rva, rpa = _rollout(m, Q, inp, a); n_roll += 1
        J0 = QL.cost(zv0[0], zp0[0], inp[2], inp[3], alpha)[0].double().cpu().numpy()
        pv = {s: float(v[0]) for s, v in QL._pred_var(rva, rpa, alpha).items()}
        for c in need:
            set_cfg(c)                                     # ranges (+ bias) of the configuration
            Q.configure(c['var'], c['k'])
            Q.begin_plan([e], salt)
            draws = [i0, i0 + 1] if c['var'] in FRESH else [i0]
            Jq_list, steps_list = [], []
            for it in draws:
                Q.it = it
                zvq, zpq, qva, qpa = _rollout(m, Q, inp, a); n_roll += 1
                Jq_list.append(QL.cost(zvq[0], zpq[0], inp[2], inp[3], alpha)[0].double().cpu().numpy())
                st = QL._step_errors(qva, qpa, rva, rpa, alpha)
                steps_list.append({s: float(v[0]) for s, v in st.items()})
                del qva, qpa
            Q.configure('none')
            met = _metrics(J0, Jq_list, steps_list, pv)
            if c['var'] in FIXED:
                met['rho_p'] = 1.0                         # identical error field in consecutive iterations
            met['pr'] = met['rho_p'] * met['ratio']        # Dg1 predictor rho_p x sd_e / sd_J (per state)
            met['n_draws'] = len(draws)
            S[cfg_key(c)] = met
        del rva, rpa
    res['time'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    L.jdump(res, out_path)
    return res, n_roll


def summarize(task, e13_dir, configs, splits=('held', 'eval8')):
    """Mean over states of every metric, per configuration and set; 'all' = held + eval8 states."""
    import glob
    recs = []
    for f in sorted(glob.glob(f'{e13_dir}/*_e*.json')):
        d = json.load(open(f))
        if d['split'] in splits:
            recs.append(d)
    out = {}
    for c in configs:
        key = cfg_key(c)
        row = {}
        for sname in ('it0', 'last'):
            for grp, sp in (('all', splits),) + tuple((s, (s,)) for s in splits):
                vals = [d['sets'][sname][key] for d in recs if d['split'] in sp and key in d['sets'].get(sname, {})]
                if not vals:
                    continue
                agg = {}
                for mk in vals[0]:
                    v = np.array([x.get(mk, np.nan) for x in vals], float)
                    agg[mk] = float(np.nanmean(v)) if np.isfinite(v).any() else float('nan')
                agg['n_states'] = len(vals)
                row[f'{sname}_{grp}'] = agg
        out[key] = dict(cfg=c, **row)
    return out
