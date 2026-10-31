#!/bin/bash
# E10 main run on one pod: (if missing) the mt80 dims JSON via the official multi-task env build, then the driver.
# usage: run_e10.sh <task1,task2,...> <workers>      (log: /workspace/logs/driver.log; restart = same command)
source /workspace/code/env_e10.sh
cd /workspace/code || exit 1
mkdir -p /workspace/e10 /workspace/logs
if [ ! -s "$E10_DIMS" ]; then
  $PY mt_dims.py "$E10_DIMS" /workspace/e10/mt80_ckpt_report.json > /workspace/logs/mt_dims.log 2>&1 || exit 1
fi
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) run_e10.sh tasks=$1 workers=$2" >> /workspace/logs/driver.log
$PY e10_driver.py --out /workspace/e10 --tasks "$1" --workers "$2" >> /workspace/logs/driver.log 2>&1
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) driver exit $?" >> /workspace/logs/driver.log
