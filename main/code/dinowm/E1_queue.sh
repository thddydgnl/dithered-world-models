#!/bin/bash
# E1 (DINO-WM part) launcher. NOT started by the E0 worker: the coordinator starts it after DESIGN_main.md is frozen.
# Usage (on the server; the GPU must be given explicitly, tmux inline commands inherit nothing):
#   tmux new-session -d -s A-E1-dinowm "DINOWM_GPU=<assigned GPU> bash ~/pmx/A/dinowm_main/code/E1_queue.sh >> ~/pmx/A/logs/E1_dinowm.log 2>&1"
# Options (environment): TASKS="wall pointmaze pusht" (default order), KS=3,4 (only quantized cells with these k;
# none/nonealt jobs always included and, through the per-job locks, run once by whichever runner gets them first),
# MAN=<manifest name>. Two GPUs: start one session per GPU with disjoint TASKS (or disjoint KS for one task); each
# (task, k) cell then stays on one GPU (the two A5000s give bitwise-identical plans, E0 report section 2).
# One runner process per task, one after another: every job writes one JSON (results/E1_dinowm/<task>/<job id>.json)
# and is skipped when it exists, so rerunning this script resumes the run. Progress: python3 status.py manifests/E1_dinowm.json
cd "$(dirname "$0")"
export DINOWM_GPU=${DINOWM_GPU:?set DINOWM_GPU explicitly: the GPU assigned by the coordinator (every cell stays on it)}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$DINOWM_GPU
MAN=${MAN:-E1_dinowm}
echo "E1Q $(date -u +%FT%TZ) start $MAN"
for task in ${TASKS:-wall pointmaze pusht}; do
  L=run.sh; [ "$task" = pointmaze ] && L=run_pm.sh
  for try in $(seq 1 20); do
    echo "E1Q $(date -u +%FT%TZ) $task try $try"
    bash $L runner.py manifests/$MAN.json --task $task --replay-workers 8 ${KS:+--ks $KS}
    python3 - "$MAN" "$task" << 'PY' && break
import json, os, sys
man = json.load(open(f'manifests/{sys.argv[1]}.json')); t = sys.argv[2]
ks = os.environ.get('KS'); keep = {int(x) for x in ks.split(',')} if ks else None
sel = [j for j in man['jobs'] if j['task'] == t and (keep is None or j['var'] in ('none', 'nonealt') or int(j['k']) in keep)]
miss = [j for j in sel if not os.path.exists(os.path.expanduser(f"~/pmx/A/dinowm_main/results/{man['name']}/{t}/{j['id']}.json"))]
print('missing', len(miss)); sys.exit(1 if miss else 0)
PY
    sleep 300
  done
done
echo "E1Q $(date -u +%FT%TZ) ALL_DONE"
