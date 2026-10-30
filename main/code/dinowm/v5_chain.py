"""DINO-WM v5 chain driver (DESIGN_main v5 section 7.5): runs the v5 stages on ONE GPU, in the order of a plan file.

- The plan file is re-read after every cell, so stages can be added or reordered while the chain runs (the E5
  manifests are written after the E1 analysis). Lines (# comments allowed):
      man <manifest name>            run every cell of code/manifests/<name>.json (waits until the file exists)
      cmd <name> [dep manifest ...]  run `bash v5_cmd.sh <name>` once, after the listed manifests are complete
- One cell = one GPU (v5 section 7.0): a cell is claimed atomically with mkdir claims/v5/<manifest>__<cell> (the GPU
  is written inside). A second GPU can run this script later with its own plan; it takes only unclaimed cells.
  A restarted chain resumes its own claimed cells (finished jobs are skipped by the runner). cmd stages are claimed
  the same way (claims/v5/cmd__<name>, with a 'done' file at the end); the other GPU waits for 'done'.
- A cell is run as `run.sh|run_pm.sh runner.py manifests/<m>.json --task <t> --cells <cell>`, retried up to 5 times
  (5 min apart); after that the cell is marked 'failed' in its claim dir and the chain goes on (fix, delete the
  'failed' file, restart).
- --after-e1 <log>: first wait until the given E1 PushT driver log says "PDRV ... DONE" (GPU 1: its PushT k=5 cell).

usage (tmux, GPU explicit):
  tmux new-session -d -s A-v5-dinowm-g1 "python3 ~/pmx/A/dinowm_main/code/v5_chain.py --gpu 1 \
      --plan ~/pmx/A/dinowm_main/code/plans/v5_g1.txt --after-e1 ~/pmx/A/logs/E1_dinowm_pusht_g1.log \
      >> ~/pmx/A/logs/v5_dinowm_g1.log 2>&1"
Log lines: V5C <time> gpu<g> ..."""
import os, sys, json, time, argparse, subprocess

CODE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.dirname(CODE)
CLAIMS = f'{MAIN}/claims/v5'


def log(gpu, msg):
    print(f"V5C {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} gpu{gpu} {msg}", flush=True)


def read_plan(path):
    out = []
    for line in open(path):
        line = line.split('#')[0].split()
        if line:
            out.append(line)
    return out


def load_man(name):
    p = f'{CODE}/manifests/{name}.json'
    return json.load(open(p)) if os.path.exists(p) else None


def done_job(name, j):
    return os.path.exists(f"{MAIN}/results/{name}/{j['task']}/{j['id']}.json")


def cells_of(man):
    cells = {}
    for j in man['jobs']:
        cells.setdefault(j['cell'], []).append(j)
    return cells


def man_complete(name):
    man = load_man(name)
    return man is not None and all(done_job(name, j) for j in man['jobs'])


def claim(key, gpu):
    """-> 'mine' (new or resumed), 'other', or 'failed' (own claim marked failed)."""
    d = f'{CLAIMS}/{key}'
    try:
        os.makedirs(d)
        open(f'{d}/gpu', 'w').write(str(gpu))
        return 'mine'
    except FileExistsError:
        owner = open(f'{d}/gpu').read().strip() if os.path.exists(f'{d}/gpu') else '?'
        if owner != str(gpu):
            return 'other'
        return 'failed' if os.path.exists(f'{d}/failed') else 'mine'


def run_cell(gpu, name, cell, jobs, tries=5):
    task = jobs[0]['task']
    assert all(j['task'] == task for j in jobs), cell
    launcher = 'run_pm.sh' if task == 'pointmaze' else 'run.sh'
    env = dict(os.environ, DINOWM_GPU=str(gpu))
    for t in range(1, tries + 1):
        log(gpu, f'run {name} cell {cell} ({len(jobs)} jobs, {sum(not done_job(name, j) for j in jobs)} left) try {t}')
        subprocess.call(['bash', launcher, 'runner.py', f'manifests/{name}.json', '--task', task, '--cells', cell,
                         '--replay-workers', '8'], cwd=CODE, env=env)
        left = sum(not done_job(name, j) for j in jobs)
        if left == 0:
            log(gpu, f'cell done {name} {cell}')
            return True
        log(gpu, f'cell {name} {cell}: {left} jobs missing after try {t}')
        time.sleep(300)
    open(f'{CLAIMS}/{name}__{cell}/failed', 'w').write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    log(gpu, f'FAIL cell {name} {cell}: marked failed, going on')
    return False


def run_cmd(gpu, name, tries=3):
    env = dict(os.environ, DINOWM_GPU=str(gpu))
    for t in range(1, tries + 1):
        log(gpu, f'cmd {name} try {t}')
        rc = subprocess.call(['bash', 'v5_cmd.sh', name], cwd=CODE, env=env)
        if rc == 0:
            open(f'{CLAIMS}/cmd__{name}/done', 'w').write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            log(gpu, f'cmd done {name}')
            return True
        log(gpu, f'cmd {name} exit {rc}')
        time.sleep(300)
    open(f'{CLAIMS}/cmd__{name}/failed', 'w').write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    log(gpu, f'FAIL cmd {name}: marked failed; the chain stops here (later stages may depend on it)')
    return False


def step(gpu, plan, poll, said):
    """Do one unit of work. -> 'ran' (re-read the plan and call again), 'wait' (sleep, call again), 'end', 'stop'."""
    for st in read_plan(plan):
        kind, name, deps = st[0], st[1], st[2:]
        if kind == 'man':
            man = load_man(name)
            if man is None:
                if said.get(name) != 'missing':
                    log(gpu, f'waiting for manifest {name}'); said[name] = 'missing'
                return 'wait'
            for cell, jobs in cells_of(man).items():
                if all(done_job(name, j) for j in jobs):
                    continue
                c = claim(f'{name}__{cell}', gpu)
                if c != 'mine':
                    continue
                run_cell(gpu, name, cell, jobs)
                return 'ran'
        elif kind == 'cmd':
            d = f'{CLAIMS}/cmd__{name}'
            if os.path.exists(f'{d}/done'):
                continue
            miss = [m for m in deps if not man_complete(m)]
            if miss:
                if said.get(name) != str(miss):
                    log(gpu, f'cmd {name} waits for {miss}'); said[name] = str(miss)
                return 'wait'
            c = claim(f'cmd__{name}', gpu)
            if c == 'other':
                if said.get(name) != 'other':
                    log(gpu, f'cmd {name} runs on the other GPU; waiting for it'); said[name] = 'other'
                return 'wait'
            if c == 'failed':
                return 'stop'
            return 'ran' if run_cmd(gpu, name) else 'stop'
        else:
            log(gpu, f'unknown plan line {st}'); return 'stop'
    return 'end'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, required=True); ap.add_argument('--plan', required=True)
    ap.add_argument('--after-e1', default=None); ap.add_argument('--poll', type=int, default=600)
    a = ap.parse_args()
    os.makedirs(CLAIMS, exist_ok=True)
    log(a.gpu, f'start plan {a.plan} pid {os.getpid()}')
    if a.after_e1:
        log(a.gpu, f'waiting for "PDRV ... DONE" in {a.after_e1}')
        while not (os.path.exists(a.after_e1) and any(l.startswith('PDRV') and l.rstrip().endswith(' DONE')
                                                      for l in open(a.after_e1))):
            time.sleep(a.poll)
        log(a.gpu, 'E1 PushT driver done')
    said = {}
    while True:
        r = step(a.gpu, a.plan, a.poll, said)
        if r == 'wait':
            time.sleep(a.poll)
        elif r == 'end':
            log(a.gpu, 'ALL_DONE (every stage of the plan is complete or claimed by the other GPU)'); return
        elif r == 'stop':
            log(a.gpu, 'STOPPED'); sys.exit(1)


if __name__ == '__main__':
    main()
