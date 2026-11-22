#!/bin/bash
# (Re)start the E14 queue driver on this pod (WORKER_RULES 3). Safe to call any number of times: does nothing if the
# driver is alive; finished jobs are skipped and failed jobs are re-queued at most twice by the driver itself.
# usage: restart.sh [--force|--stop]   (--force: stop a live (e.g. hung) driver and its workers first, then start;
#                                      --stop: stop the driver and its workers and do not start a new one)
A=/workspace/atm
source $A/code/env_mt.sh
mkdir -p $A/logs $A/out
exec 9> $A/logs/restart.lock
flock -n 9 || { echo "another restart.sh is running"; exit 0; }
ours() { tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | grep -qE "mt14_driver|multiprocessing"; }
alive() { [ -n "$1" ] && kill -0 "$1" 2>/dev/null && tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | grep -q mt14_driver; }
kill_group() {   # members of the old driver's process group (setsid: pgid = driver pid) that are our python processes
  local g=$1 p n=0
  for p in $(pgrep -g "$g" 2>/dev/null); do ours "$p" && { kill -TERM "$p" 2>/dev/null; n=$((n + 1)); }; done
  [ $n -gt 0 ] || return 0
  sleep 4
  for p in $(pgrep -g "$g" 2>/dev/null); do ours "$p" && kill -KILL "$p" 2>/dev/null; done
  echo "stopped $n process(es) of old driver group $g"
}
OLD=$(cat $A/logs/driver.pid 2>/dev/null)
if alive "$OLD"; then
  if [ "${1:-}" != "--force" ] && [ "${1:-}" != "--stop" ]; then echo "driver alive (pid $OLD)"; exit 0; fi
  kill_group "$OLD"
elif [ -n "$OLD" ]; then
  kill_group "$OLD"   # the driver is gone; its spawned workers may still hold the GPU (E10 lesson)
fi
if [ "${1:-}" = "--stop" ]; then
  alive "$OLD" && { echo "driver $OLD still alive"; exit 1; }
  echo "driver stopped"; exit 0
fi
[ -f $A/SETUP_DONE ] || { echo "setup not done"; exit 1; }
[ -s $A/TASKS ] || { echo "no $A/TASKS"; exit 1; }
WORKERS=$(cat $A/WORKERS 2>/dev/null || echo 1)
if [ ! -s "$E10_DIMS" ]; then
  $PY $A/code/mt_dims.py "$E10_DIMS" $A/out/env/mt80_ckpt_report.json > $A/logs/mt_dims.log 2>&1 || { echo "mt_dims failed"; exit 1; }
fi
rm -f $A/PHASE
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) restart.sh tag=$ATM_TAG tasks=$(cat $A/TASKS) workers=$WORKERS" >> $A/logs/driver.log
cd $A/code
setsid nohup $PY $A/code/mt14_driver.py --out $A/out --tasks "$(cat $A/TASKS)" --workers "$WORKERS" --tag "$ATM_TAG" \
  >> $A/logs/driver.log 2>&1 < /dev/null 9>&- &
for i in 1 2 3 4 5 6 7 8 9 10; do
  sleep 1
  NEW=$(cat $A/logs/driver.pid 2>/dev/null)
  if [ "$NEW" != "$OLD" ] && alive "$NEW"; then echo "driver started (pid $NEW)"; exit 0; fi
done
echo "driver did not report a pid; see $A/logs/driver.log"; tail -n 5 $A/logs/driver.log; exit 1
