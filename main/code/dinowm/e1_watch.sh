#!/bin/bash
# Compact E1 + v5 (DINO-WM) watcher for the worker's periodic checks: waits up to $1 seconds (default 570), returns
# early when a new Traceback appears in an E1/v5 DINO-WM log, a PushT driver logs a new line, or the v5 chain logs a
# new V5C line; then prints one status block.
L=$HOME/pmx/A/logs; C=$HOME/pmx/A/dinowm_main/code
LOGS="$L/E1_dinowm.log $L/E1_dinowm_pusht_g0.log $L/E1_dinowm_pusht_g1.log $L/v5_dinowm_g0.log $L/v5_dinowm_g1.log"
# real tracebacks only: multiprocessing's benign semaphore cleanup at pool shutdown prints "Exception ignored in"
tb() { echo $(( $(cat $LOGS 2>/dev/null | grep -c Traceback) - $(cat $LOGS 2>/dev/null | grep -c "Exception ignored in") )); }
pd() { cat $L/E1_dinowm_pusht_g0.log $L/E1_dinowm_pusht_g1.log $L/v5_dinowm_g0.log $L/v5_dinowm_g1.log 2>/dev/null | grep -c -E "^PDRV|^V5C"; }
t0=$(tb); p0=$(pd); end=$(( $(date +%s) + ${1:-570} ))
while [ $(date +%s) -lt $end ]; do
  [ "$(tb)" != "$t0" ] && break; [ "$(pd)" != "$p0" ] && break; sleep 30
done
echo "$(date -u +%FT%TZ) tracebacks=$(tb) B_procs=$(ps -u $USER -o args | grep -c '[b]-dinowm/bin/python')"
python3 $C/status.py $C/manifests/E1_dinowm_wp.json $C/manifests/E1_dinowm_pusht.json | tr '\n' ';'; echo
grep "E1Q" $L/E1_dinowm.log | tail -1
cat $L/E1_dinowm_pusht_g0.log $L/E1_dinowm_pusht_g1.log 2>/dev/null | grep -E "^PDRV|E1Q" | tail -2
cat $L/v5_dinowm_g0.log $L/v5_dinowm_g1.log 2>/dev/null | grep -E "^V5C" | tail -2
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';'; df -h ~ | tail -1 | awk '{print "disk free " $4}'
