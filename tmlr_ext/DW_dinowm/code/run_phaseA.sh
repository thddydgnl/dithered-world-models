# Phase A on an owner pod: calibration items only (driver --only calib), then the 0.7-2 unit checks per task.
source /workspace/atm/code/env.sh
cd /workspace/atm/code
$PY -u driver.py --only calib > /workspace/atm/logs/calib_only.log 2>&1
echo "CALIB_ONLY_EXIT $?" >> /workspace/atm/logs/checks.log
for t in $*; do $PY -u check_variants.py $t > /workspace/atm/logs/check2_$t.log 2>&1; echo "CHECK2_EXIT $t $?" >> /workspace/atm/logs/checks.log; done
