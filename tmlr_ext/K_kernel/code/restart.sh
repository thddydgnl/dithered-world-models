#!/bin/bash
# (Re)start the E17 queue driver unless it is alive (safe to call any number of times: finished jobs are skipped by
# runner.py, failed jobs are retried at most twice, a live driver is left alone).
# Usage: restart.sh [pre|main]   (default: the last mode in out/DRIVER_MODE, else main)
O=/workspace/atm/out
C=/workspace/atm/code
mkdir -p /workspace/atm/logs
MODE=${1:-$(cat $O/DRIVER_MODE 2>/dev/null || echo main)}
PID=$(cat $O/driver.pid 2>/dev/null)
if [ -n "$PID" ] && [ -e /proc/$PID ]; then echo "driver alive (pid $PID, mode $(cat $O/DRIVER_MODE)); nothing to do"; exit 0; fi
if [ -f $O/latency/progress.json ] && grep -q '"stage": "main"' $O/latency/progress.json && pgrep -f "e17_lat.py main" > /dev/null; then
  echo "latency measurement running (exclusive GPU); not starting the driver"; exit 0
fi
if [ "$(cat $O/DRIVER_STATE 2>/dev/null)" = finished ] && [ "$(cat $O/DRIVER_MODE 2>/dev/null)" = "$MODE" ] && \
   [ "$(python3 $C/qstat.py left $MODE)" = "0" ]; then echo "driver finished ($MODE), no runnable jobs left"; exit 0; fi
LOG=/workspace/atm/logs/driver_${MODE}_$(date -u +%Y%m%dT%H%M%S).log
setsid nohup bash $C/driver.sh $MODE > $LOG 2>&1 < /dev/null &
sleep 1
echo "driver started (mode $MODE, pid $(cat $O/driver.pid 2>/dev/null), log $LOG)"
