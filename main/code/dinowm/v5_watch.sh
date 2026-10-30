#!/bin/bash
# DINO-WM v5 chain watcher for the worker's periodic checks: waits up to $1 seconds (default 570) and returns early on
# a new real Traceback or a new V5C line in the chain logs; then prints one status block (progress per manifest).
L=$HOME/pmx/A/logs; C=$HOME/pmx/A/dinowm_main/code; M=$HOME/pmx/A/dinowm_main
LOGS="$L/v5_dinowm_g0.log $L/v5_dinowm_g1.log"
tb() { echo $(( $(cat $LOGS 2>/dev/null | grep -c Traceback) - $(cat $LOGS 2>/dev/null | grep -c "Exception ignored in") )); }
vc() { cat $LOGS 2>/dev/null | grep -c "^V5C"; }
t0=$(tb); v0=$(vc); end=$(( $(date +%s) + ${1:-570} ))
while [ $(date +%s) -lt $end ]; do
  [ "$(tb)" != "$t0" ] && break; [ "$(vc)" != "$v0" ] && break; sleep 30
done
echo "$(date -u +%FT%TZ) tracebacks=$(tb) chain_alive=$(tmux has-session -t A-v5-dinowm-g1 2>/dev/null && echo yes || echo NO)"
python3 - "$C" "$M" << 'PY'
import json, os, sys
C, M = sys.argv[1], sys.argv[2]
for line in open(f'{C}/plans/v5_g1.txt'):
    w = line.split('#')[0].split()
    if len(w) < 2 or w[0] != 'man':
        continue
    p = f'{C}/manifests/{w[1]}.json'
    if not os.path.exists(p):
        print(f'{w[1]}: manifest not yet written'); continue
    man = json.load(open(p))
    cells = {}
    for j in man['jobs']:
        d = os.path.exists(f"{M}/results/{man['name']}/{j['task']}/{j['id']}.json")
        c = cells.setdefault(j['cell'], [0, 0]); c[0] += d; c[1] += 1
    done = sum(c[0] for c in cells.values()); tot = sum(c[1] for c in cells.values())
    part = ' '.join(f'{k}:{a}/{b}' for k, (a, b) in cells.items() if 0 < a < b)
    print(f"{w[1]}: {done}/{tot} jobs" + (f' (in progress {part})' if part else ''))
PY
cat $LOGS 2>/dev/null | grep "^V5C" | tail -2
grep -E "s/ep\)" $L/v5_dinowm_g1.log 2>/dev/null | tail -1
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';'; df -h ~ | tail -1 | awk '{print "disk free " $4}'
