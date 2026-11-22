#!/bin/bash
# (Re)start the queue driver of this pod (WORKER_RULES 3). Safe to call any number of times: if the driver is alive it
# does nothing; otherwise it starts MPS (idempotent) and a new driver session, which skips finished jobs (result files)
# and re-queues failed jobs (2 retries per session). The driver itself holds an exclusive lock (one instance only).
A=/workspace/atm
source $A/code/env_td.sh
PID=$(cat $A/logs/driver.pid 2>/dev/null)
if [ -n "$PID" ] && [ -d /proc/$PID ] && grep -q driver.py /proc/$PID/cmdline 2>/dev/null; then
  echo "driver alive (pid $PID); nothing to do"; exit 0
fi
[ -f $A/SETUP_DONE ] || { echo "setup not done; not starting"; exit 1; }
[ -f $A/driver_conf.json ] || { echo "no $A/driver_conf.json; not starting"; exit 1; }
bash $A/code/mps_td.sh start > /dev/null
cd $A/code
setsid nohup $PY $A/code/driver.py --conf $A/driver_conf.json >> $A/logs/driver_stdout.log 2>&1 < /dev/null &
sleep 3
PID=$(cat $A/logs/driver.pid 2>/dev/null)
if [ -n "$PID" ] && [ -d /proc/$PID ]; then echo "driver started (pid $PID)"; else echo "driver did not start; see $A/logs/driver_stdout.log"; tail -5 $A/logs/driver_stdout.log; exit 1; fi
