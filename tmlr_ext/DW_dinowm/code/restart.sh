#!/bin/bash
# (Re)start the dw queue supervisor. Safe to call any number of times: if it is alive nothing happens; finished jobs
# are skipped by the driver; failed jobs are retried by the driver up to 2 times (3 attempts in all).
ATM=/workspace/atm
P=$ATM/logs/supervisor.pid
if [ -f $P ] && kill -0 "$(cat $P)" 2> /dev/null; then
  echo "supervisor alive (pid $(cat $P)); nothing to do"
  exit 0
fi
mkdir -p $ATM/logs
setsid nohup bash $ATM/code/driver_loop.sh > /dev/null 2>&1 < /dev/null &
sleep 1
echo "supervisor started (pid $(cat $P 2>/dev/null || echo '?'))"
