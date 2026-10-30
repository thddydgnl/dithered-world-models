"""Progress of runner manifests: done / total jobs and episodes per task; exit code 0 iff every job has its JSON.
usage: python3 status.py manifests/<name>.json [...]   (plain python3, no torch)"""
import os, sys, json
MAIN = os.environ.get('DINOWM_MAIN', os.path.expanduser('~/pmx/A/dinowm_main'))
all_done = True
for mp in sys.argv[1:]:
    man = json.load(open(mp))
    tot, done = {}, {}
    for j in man['jobs']:
        t = j['task']; n = j['ep1'] - j['ep0']
        tot[t] = tot.get(t, 0) + n
        if os.path.exists(f"{MAIN}/results/{man['name']}/{t}/{j['id']}.json"):
            done[t] = done.get(t, 0) + n
    for t in tot:
        print(f"{man['name']} {t}: {done.get(t, 0)}/{tot[t]} episode-runs")
    all_done &= all(done.get(t, 0) == tot[t] for t in tot)
sys.exit(0 if all_done else 1)
