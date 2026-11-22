#!/bin/bash
# E17 closed-loop queue driver (WORKER_RULES 3). Usage: driver.sh pre|main
#   pre   bench + calibration episodes + FP32 none (pre.jsonl), then pcstats      (before PLAN.md: checks 3 and 4)
#   main  calibration, pcstats (no-ops when done), then every eval job (main.jsonl; finished jobs are skipped)
# Each manifest runs up to 3 passes: runner.py skips finished jobs and jobs with 3 starts, so a failed job is retried at
# most twice. The driver writes out/driver.pid, out/DRIVER_MODE and out/DRIVER_STATE (running | finished).
source /workspace/atm/code/env.sh
W=/workspace/atm
O=$W/out
C=$W/code
M=$O/manifests
MODE=${1:-main}
echo $$ > $O/driver.pid
echo $MODE > $O/DRIVER_MODE
echo running > $O/DRIVER_STATE
bash $C/mps.sh start
cd $C
$PY $C/make_manifest_e17.py all --outdir $M || { echo "manifest failed"; exit 1; }
run() {
  for pass in 1 2 3; do
    echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) $1 pass $pass workers ${WORKERS:-8}"
    $PY $C/runner.py --manifest $M/$1 --out $O --big $W/big --workers ${WORKERS:-8} --max_attempts 3
    left=$(python3 - "$M/$1" << 'PY'
import json, os, sys
O = "/workspace/atm/out"
n = 0
for line in open(sys.argv[1]):
    if not line.strip():
        continue
    j = json.loads(line)
    if os.path.exists(f"{O}/jobs/{j['type']}/{j['id']}.json.gz"):
        continue
    try:
        att = int(open(f"{O}/attempts/{j['id']}.n").read().strip() or 0)
    except OSError:
        att = 0
    n += att < 3
print(n)
PY
)
    echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) $1 pass $pass done; runnable jobs left: $left"
    [ "$left" = "0" ] && break
  done
}
if [ "$MODE" = pre ]; then
  echo smoke > $O/PHASE
  run pre.jsonl
  run pcstats.jsonl
else
  echo main > $O/PHASE
  run calib.jsonl
  run pcstats.jsonl
  run main.jsonl
fi
echo finished > $O/DRIVER_STATE
rm -f $O/driver.pid
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) DRIVER_FINISHED $MODE"
