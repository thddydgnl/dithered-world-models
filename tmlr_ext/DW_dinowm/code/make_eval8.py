"""eval8 split (DESIGN_tmlr.md v1.0 section 0.4 and deviation D1 of section 11) bookkeeping and checks.

PushT (D1, coordinator 2026-11-21T18:41:18Z): wmlib.sample_pairs_constrained, pair seed 8, 100 pairs, max_tries 5000,
hard = held (seed 3, 0-19), smoke (seed 7, 0-3), eval6 (its stored 100 segments); soft = repro (seed 4, 0-49);
exact duplicates rejected.
  precheck(ctx, path)     before sampling: real validation lengths; admissible 26-frame starts under held+smoke and
                          held+smoke+repro must be 1,294 and 485 (D1); the held / smoke / repro replicas
                          (wmlib.pair_segments) and the copied eval6 segments must equal splits_pusht_v3.json
  bookkeeping(ctx, pairs) after sampling, before any planning: out/splits/eval8_<task>.json with seeds, salt, sha256;
                          PushT also: frames shared with held / smoke / eval6 (must be 0), repro overlap (pairs with
                          overlap, per-pair shared frames mean / max, nearest repro start distance, identical segments),
                          concentration (trajectories, pairs per trajectory, connected components of mutually
                          overlapping eval8 pairs: count, sizes, component of every pair = the clusters of the auxiliary
                          cluster bootstrap), sampler statistics, and 'violations' (D1 stop conditions: any frame shared
                          with held / smoke / eval6, a segment identical to a repro segment, a pair sharing > 20 repro
                          frames). The driver plans no PushT eval8 episode while 'violations' is non-empty.
usage (pod, CPU): python make_eval8.py count_free [out.json]   (the precheck numbers only)
"""
import os, sys, json, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wmlib as L
import qlib as QL

SEG = 5 * L.GOAL_H + 1
REFS = (('held', 20), ('smoke', 4), ('repro', 50), ('eval6', 100))
CODE = os.path.dirname(os.path.abspath(__file__))
V3_NAMES = dict(held='held', smoke='smoke', repro='pilot_eval_R2-7_R2-13')


def arrays_sha(pairs):
    h = hashlib.sha256()
    for k in ('obs0_visual', 'obs0_proprio', 'obsg_visual', 'obsg_proprio', 'state0', 'stateg'):
        a = np.ascontiguousarray(pairs[k])
        h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def val_lengths(T):
    dset = T['dset']
    out = []
    for i in range(len(dset)):
        base, bi = L._base_traj(dset, i)
        out.append(int(base.get_seq_length(bi)))
    return out


def admissible(lens, segs_lists):
    tot = 0
    per = []
    for b, Lb in enumerate(lens):
        used = set()
        for segs in segs_lists:
            for rb, ro in segs:
                if rb == b:
                    used.update(range(ro, ro + SEG))
        n = sum(1 for o in range(0, Lb - SEG + 1) if not any(f in used for f in range(o, o + SEG)))
        per.append(n); tot += n
    return tot, per


def precheck(ctx, path, expected):
    T = ctx.T
    cache = f'{L.MAIN}/pairs'
    lens = val_lengths(T)
    refs = {name: QL.split_segments(T, name, n, cache) for name, n in REFS}
    v3 = json.load(open(f'{CODE}/inputs/splits_pusht_v3.json'))
    rep = {name: [list(x) for x in refs[name]] == [list(x) for x in v3['reference_segments'][V3_NAMES[name]]['segs']]
           for name in ('held', 'smoke', 'repro')}
    rep['eval6'] = [list(x) for x in refs['eval6']] == [[r['traj'], r['offset']] for r in v3['pairs']]
    a_hs, _ = admissible(lens, [refs['held'], refs['smoke']])
    a_hsr, _ = admissible(lens, [refs['held'], refs['smoke'], refs['repro']])
    a_hse, per_hse = admissible(lens, [refs['held'], refs['smoke'], refs['eval6']])
    a_all, per_all = admissible(lens, [refs[n] for n, _ in REFS])
    ok_counts = (a_hs == expected['held+smoke'] and a_hsr == expected['held+smoke+repro'])
    res = dict(n_val_trajectories=len(lens), lengths=lens, total_frames=int(sum(lens)), segment_frames=SEG,
               admissible={'held+smoke': a_hs, 'held+smoke+repro': a_hsr, 'held+smoke+eval6 (D1 hard set)': a_hse,
                           'held+smoke+repro+eval6 (design literal)': a_all},
               admissible_per_traj={'held+smoke+eval6': per_hse, 'all four': per_all},
               expected=expected, counts_match=bool(ok_counts), replicas_match_v3=rep,
               passed=bool(ok_counts and all(rep.values())))
    L.jdump(res, path)
    print('PRECHECK', json.dumps(dict(admissible=res['admissible'], replicas=rep, passed=res['passed'])), flush=True)
    return res


def components(segs):
    """Connected components of the overlap graph (same trajectory and at least one shared frame)."""
    n = len(segs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if segs[i][0] == segs[j][0] and L.shared_frames(segs[i][1], segs[j][1], SEG) > 0:
                parent[find(i)] = find(j)
    roots = {}
    comp = [roots.setdefault(find(i), len(roots)) for i in range(n)]
    sizes = [comp.count(c) for c in range(len(roots))]
    return comp, sizes


def bookkeeping(ctx, pairs, path, max_repro=20):
    task = ctx.task
    sp = QL.SPLITS['eval8']
    out = dict(task=task, split='eval8', design='DESIGN_tmlr.md v1.0 section 0.4' + (' + D1' if task == 'pusht' else ''),
               pair_seed=sp['pair_seed'], n=sp['n'], episode_index=[0, sp['n'] - 1],
               env_seed=[pairs['eval_seed'][0], pairs['eval_seed'][-1]],
               planner_seed=[sp['plan_seed'], sp['plan_seed'] + sp['n'] - 1], dither_salt=sp['salt'],
               arrays_sha256=arrays_sha(pairs), state0=np.asarray(pairs['state0']).round(6).tolist(),
               stateg=np.asarray(pairs['stateg']).round(6).tolist(), env_info=pairs['env_info'], violations=[])
    if task == 'pusht':
        cache = f'{L.MAIN}/pairs'
        meta = json.load(open(f'{cache}/pairs_pusht_eval8.json'))
        new = [tuple(int(v) for v in x) for x in pairs['traj']]
        refs = {name: QL.split_segments(ctx.T, name, n, cache) for name, n in REFS}
        rows = []
        for j, (bi, o) in enumerate(new):
            r = dict(pair=j, traj=bi, offset=o, sampler=meta.get('sampler_stats', [{}] * len(new))[j])
            for name, segs in refs.items():
                sh = [L.shared_frames(o, ro, SEG) for rb, ro in segs if rb == bi]
                r[f'shared_{name}'] = int(sum(sh))
                r[f'identical_{name}'] = int(sum(1 for x in sh if x == SEG))
            dist = [abs(o - ro) for rb, ro in refs['repro'] if rb == bi]
            r['nearest_repro_start'] = min(dist) if dist else None
            rows.append(r)
        comp, sizes = components(new)
        for r, c in zip(rows, comp):
            r['component'] = c
        rep = [r['shared_repro'] for r in rows]
        per_traj = {}
        for bi, _ in new:
            per_traj[bi] = per_traj.get(bi, 0) + 1
        out.update(constraints={k: [list(x) for x in v] for k, v in QL.EVAL8_PUSHT.items()}, pairs=rows,
                   shared_with={name: dict(pairs_with_overlap=sum(1 for r in rows if r[f'shared_{name}'] > 0),
                                           total_shared_frames=sum(r[f'shared_{name}'] for r in rows),
                                           identical_segments=sum(r[f'identical_{name}'] for r in rows))
                                for name, _ in REFS},
                   repro=dict(pairs_with_overlap=sum(1 for x in rep if x > 0), shared_mean=float(np.mean(rep)),
                              shared_max=int(max(rep)), zero_overlap_pairs=sum(1 for x in rep if x == 0),
                              nearest_start_median=float(np.median([r['nearest_repro_start'] for r in rows
                                                                    if r['nearest_repro_start'] is not None]))),
                   concentration=dict(distinct_trajectories=len(per_traj), pairs_per_trajectory=per_traj,
                                      components=len(sizes), component_sizes=sorted(sizes, reverse=True),
                                      pairs_overlapping_another=sum(1 for c, s in zip(comp, [sizes[c] for c in comp]) if s > 1)),
                   component_of_pair=comp,
                   sampler=dict(draws_mean=float(np.mean([r['sampler'].get('draws', 0) for r in rows])),
                                draws_max=int(max(r['sampler'].get('draws', 0) for r in rows)),
                                hard_rejections=int(sum(r['sampler'].get('hard_rejections', 0) for r in rows)),
                                duplicate_rejections=int(sum(r['sampler'].get('duplicate_rejections', 0) for r in rows))))
        v = []
        for name in ('held', 'smoke', 'eval6'):
            if out['shared_with'][name]['total_shared_frames'] > 0:
                v.append(f'frames shared with {name}')
        if out['shared_with']['repro']['identical_segments'] > 0:
            v.append('segment identical to a repro segment')
        if out['repro']['shared_max'] > max_repro:
            v.append(f"a pair shares {out['repro']['shared_max']} > {max_repro} frames with repro")
        out['violations'] = v
    L.jdump(out, path)
    return out


def count_free(out_path):
    class C:
        pass
    c = C()
    c.T = L.load_task('pusht', device='cpu')
    d = json.load(open(f'{CODE}/decisions.json'))['eval8_pusht']
    precheck(c, out_path, d['expected_free_offsets'])


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'count_free':
        count_free(sys.argv[2] if len(sys.argv) > 2 else '/workspace/atm/out/checks/eval8_pusht_precheck.json')
