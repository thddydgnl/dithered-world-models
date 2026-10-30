#!/bin/bash
# E1 (TD-MPC2 part) launcher. NOT started by the E0 worker: the coordinator starts it after DESIGN_main.md is frozen:
#   ssh kiis-mvf-gpu 'tmux new-session -d -s A-E1-tdmpc2 "WORKERS=16 bash ~/pmx/A/tdmpc2_main/code/run_e1.sh"'
# Builds the manifest from <out>/ktable.json (E0 selection) once, then runs it (resumable: finished jobs are skipped;
# a second pass retries jobs that failed). Progress: python status.py --manifest <out>/manifests/e1.jsonl --out <out>
source ~/pmx/A/tdmpc2_main/code/env.sh
cd ~/pmx/A/tdmpc2_main/code
bash mps.sh start
O=~/pmx/A/tdmpc2_main/out
LOG=~/pmx/A/logs/E1_tdmpc2.log
[ -f $O/manifests/e1.jsonl ] || $PY make_manifest.py e1 --ktable $O/ktable.json --out $O/manifests/e1.jsonl >> $LOG 2>&1
for PASS in 1 2; do
  echo "=== E1 pass $PASS $(date -u +%FT%TZ)" >> $LOG
  $PY runner.py --manifest $O/manifests/e1.jsonl --out $O --workers ${WORKERS:-16} >> $LOG 2>&1
done
$PY status.py --manifest $O/manifests/e1.jsonl --out $O >> $LOG 2>&1
echo "E1_DONE $(date -u +%FT%TZ)" >> $LOG
