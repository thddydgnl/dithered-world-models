#!/bin/bash
# Supervisor of the dw queue driver (WORKER_RULES 3): restarts driver.py after a failed job (exit 3) or a crash, stops
# when the driver reports FINISHED (exit 0) or has had nothing runnable for 6 h (exit 10). Started by restart.sh.
source /workspace/atm/code/env.sh
LOG=$ATM/logs/driver.log
echo $$ > $ATM/logs/supervisor.pid
n=0
while true; do
  echo "SUPERVISOR $(date -u +%FT%TZ) start driver (restart $n)" >> $LOG
  $PY -u $ATM/code/driver.py >> $LOG 2>&1
  rc=$?
  echo "SUPERVISOR $(date -u +%FT%TZ) driver exit $rc" >> $LOG
  [ $rc -eq 0 ] && break
  [ $rc -eq 10 ] && break
  n=$((n + 1))
  [ $n -ge 60 ] && { echo "SUPERVISOR giving up after $n restarts" >> $LOG; break; }
  sleep 15
done
rm -f $ATM/logs/supervisor.pid
