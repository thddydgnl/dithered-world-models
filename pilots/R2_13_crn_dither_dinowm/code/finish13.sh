#!/bin/bash
# Final stage on the pod: wait for the PushT main run, flush the replay loops, run the analysis.
cd /workspace/code
export PYTHONWARNINGS=ignore
while pgrep -f "[c]em13.py pusht eval" > /dev/null; do sleep 10; done
echo "$(date -u +%H:%M:%S) pusht main done" >> /workspace/logs/finish.log
touch /workspace/logs/STOP_REPLAY_pusht_eval /workspace/logs/STOP_REPLAY_wall_eval
while pgrep -f "[r]eplay13.py" > /dev/null; do sleep 5; done
echo "$(date -u +%H:%M:%S) replays done" >> /workspace/logs/finish.log
OMP_NUM_THREADS=4 python analyze13.py > /workspace/results/analysis13_stdout.txt 2>&1
echo "$(date -u +%H:%M:%S) FINISH13_DONE" >> /workspace/logs/finish.log
