"""E5 (DESIGN_main v5 section 7.3, DINO-WM): pick the best activation-PTQ baseline B per E1-eligible (task, k) from the
held det runs (manifests/E5_dinowm_held.json), then write the eval manifest manifests/E5_dinowm_eval.json with
B + det and B + sh-iter on the E1 eval pairs and planner seeds (100 episodes).

Rule: highest held det success (20 episodes); ties -> the earlier baseline in make_manifest.E5B (the TD-MPC2 rule of
v5 section 7.3, "same -> earlier in the list"). If B is the E1 grid ('main'), B + det and B + sh-iter are exactly the
E1 det and sh-iter cells (same code, pairs, seeds, grid), so no new jobs are written for that cell and the E1 cells
are used. Writes results/E5_dinowm_held/selection.json. Pure python (server: python3 e5_select.py)."""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_manifest import build, E5B, V5E, V5SPLIT, V5CHUNK

CODE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.dirname(CODE)


def main():
    held = json.load(open(f'{CODE}/manifests/E5_dinowm_held.json'))
    rec = {}
    for f in glob.glob(f"{MAIN}/results/{held['name']}/*/*.json"):
        d = json.load(open(f))
        tag = d['job']['baseline']
        for r in d['records']:
            rec[(r['task'], int(r['k']), tag, int(r['ep']))] = r
    order = [t for t, _, _ in E5B]
    sel, jobs = {}, []
    for task, ks in held['config']['cells'].items():
        for k in ks:
            rows = []
            for tag in order:
                eps = [rec[(task, k, tag, e)] for e in range(20) if (task, k, tag, e) in rec]
                if len(eps) != 20:
                    raise SystemExit(f'{task} k={k} {tag}: {len(eps)}/20 held episodes, not complete')
                rows.append(dict(tag=tag, s=int(sum(r['success'] for r in eps)), n=20,
                                 err1=sum(r['err']['1']['rel'] for r in eps) / 20 if all('err' in r for r in eps) else None))
            best = max(rows, key=lambda r: (r['s'], -order.index(r['tag'])))
            sel[f'{task}_k{k}'] = dict(task=task, k=k, held=rows, chosen=best['tag'],
                                       eval='E1 cells reused (B = E1 grid)' if best['tag'] == 'main' else 'E5_dinowm_eval')
            if best['tag'] == 'main':
                continue
            _, rid, qo = E5B[order.index(best['tag'])]
            m = build('E5_dinowm_eval', V5SPLIT[task], {task: dict(E=V5E[task], ks=[k], vars=['det', 'shiter'],
                                                                   chunk=V5CHUNK[task])}, 0, 100, rid.format(k=k), with_none=False)
            for j in m['jobs']:
                j.update(id=f"{j['id']}_{best['tag']}", cell=f'{task}_k{k}_B', role=f"E5 eval B={best['tag']}",
                         baseline=best['tag'])
                if qo:
                    j['qopts'] = dict(qo)
            jobs += m['jobs']
    os.makedirs(f"{MAIN}/results/{held['name']}", exist_ok=True)
    json.dump(sel, open(f"{MAIN}/results/{held['name']}/selection.json", 'w'), indent=1)
    man = dict(name='E5_dinowm_eval', config=dict(design='DESIGN_main v5 section 7.3', selection=sel,
                                                  reference='E1 none of the same pairs and seeds'), jobs=jobs)
    tmp = f'{CODE}/manifests/E5_dinowm_eval.json.tmp'
    json.dump(man, open(tmp, 'w'), indent=1)
    os.replace(tmp, f'{CODE}/manifests/E5_dinowm_eval.json')
    for c, v in sel.items():
        print(c, 'chosen', v['chosen'], {r['tag']: f"{r['s']}/{r['n']}" for r in v['held']})
    print(f'E5_SELECT wrote manifests/E5_dinowm_eval.json with {len(jobs)} jobs')


if __name__ == '__main__':
    main()
