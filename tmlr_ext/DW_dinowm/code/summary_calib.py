"""Phase B: summary of the calibration artifacts of every pod (D2 bias vectors, RC ranges, PT re-measure, PC guards),
for REPORT.md (D2 asks for: convergence stage, updates, converged, sample size, per-layer residuals, |b|/Delta, MSE and
propagated 5-step error at b = 0 and at the final b; counts per stage, unconverged cells, max updates, worst residual).
usage: python summary_calib.py <raw dir> <out json>"""
import os, sys, json, glob

raw, out = sys.argv[1], sys.argv[2]
bias, seen = [], set()
for f in sorted(glob.glob(f'{raw}/*/calib/bias_*.json')):
    if '.' in os.path.basename(f)[:-5]:
        continue                                   # archived design-rule vectors (bias_*.design.json)
    b = json.load(open(f))
    key = (b['task'], b['ranges'], int(b['k']))
    pod = f.split('/')[-3]
    owner = key not in seen
    seen.add(key)
    bias.append(dict(pod=pod, file=os.path.basename(f), task=b['task'], ranges=b['ranges'], k=int(b['k']),
                     rule=b.get('rule_name'), converged=b.get('converged'), unconverged_flag=b.get('bc_unconverged'),
                     stage_eta=b.get('stage_eta'), updates=b.get('updates'),
                     worst_final=max(b['final_mean_abs_resid_over_delta']),
                     final_mean_abs_resid_over_delta=b['final_mean_abs_resid_over_delta'],
                     final_max_abs_resid_over_delta=b.get('final_max_abs_resid_over_delta'),
                     b_mean_abs_over_delta=b.get('b_mean_abs_over_delta'), m0_mean_abs_over_delta=b.get('m0_mean_abs_over_delta'),
                     mse_over_delta2_b0=b.get('mse_over_delta2_b0'), mse_over_delta2_final=b.get('mse_over_delta2_final'),
                     latent_rel5_b0=(b.get('latent_rel_b0') or {}).get('5'), latent_rel5_final=(b.get('latent_rel_final') or {}).get('5'),
                     rows=b.get('rows'), elements_per_unit=b.get('elements_per_unit'), gpu=b.get('gpu'), first_copy=owner))
uniq = {}
for r in bias:                                     # one row per (task, ranges, k): the copies are identical files
    uniq.setdefault((r['task'], r['ranges'], r['k']), r)
U = list(uniq.values())
by_stage = {}
for r in U:
    by_stage[str(r['stage_eta'])] = by_stage.get(str(r['stage_eta']), 0) + 1
rc = []
for f in sorted(glob.glob(f'{raw}/*/calib/ranges_rc_k*_*.json')):
    r = json.load(open(f))
    rc.append(dict(pod=f.split('/')[-3], method=r['method'], choice=[l['choice'] for l in r['layers']],
                   objective=r['objective'], objective_init=r['objective_init'], n_evals=r['n_evals']))
pt = {}
for f in sorted(glob.glob(f'{raw}/*/calib/ptcheck_*.json')):
    d = json.load(open(f))
    pt[d['task']] = dict(exact_match_64=d['exact_match_64'], max_abs_d_hi_64=max(abs(c['d_hi_64']) for c in d['layers']),
                         max_abs_d_lo_64=max(abs(c['d_lo_64']) for c in d['layers']), gpu=d['gpu'])
pcg = {}
for f in sorted(glob.glob(f'{raw}/*/calib/ranges_pc_*.json')):
    d = json.load(open(f))
    pcg[d['task']] = d.get('guarded')
res = dict(bias_cells=sorted(U, key=lambda r: (r['task'], r['ranges'], r['k'])), n_bias_cells=len(U),
           converged_by_stage=by_stage, n_unconverged=sum(1 for r in U if not r['converged']),
           max_updates=max(r['updates'] for r in U), worst_final_residual=max(r['worst_final'] for r in U),
           rc=rc, pt_remeasure=pt, pc_guarded=pcg)
json.dump(res, open(out, 'w'), indent=1)
print(json.dumps(dict(n_bias_cells=len(U), converged_by_stage=by_stage, n_unconverged=res['n_unconverged'],
                      max_updates=res['max_updates'], worst_final_residual=round(res['worst_final_residual'], 4))))
