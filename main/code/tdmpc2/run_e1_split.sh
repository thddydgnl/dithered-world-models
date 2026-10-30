#!/bin/bash
# E1 (TD-MPC2 part) launcher, split by job type (replaces run_e1.sh after the 2026-10-21 19:10Z OOM check).
# Episode jobs are light (~0.4-0.9 GB each) and run 16-wide under MPS; prederr/decision jobs hold the held-out
# candidate sets on the GPU (~2-3.3 GB each) and ran out of memory when 16 of them shared GPU 0, so they run
# narrow afterwards. Same manifest as run_e1.sh; finished jobs are skipped, stale locks are taken over.
#   ssh kiis-mvf-gpu 'tmux new-session -d -s A-E1-tdmpc2 "bash ~/pmx/A/tdmpc2_main/code/run_e1_split.sh"'
source ~/pmx/A/tdmpc2_main/code/env.sh
cd ~/pmx/A/tdmpc2_main/code
bash mps.sh start
O=~/pmx/A/tdmpc2_main/out
LOG=~/pmx/A/logs/E1_tdmpc2.log
[ -f $O/manifests/e1.jsonl ] || $PY make_manifest.py e1 --ktable $O/ktable.json --out $O/manifests/e1.jsonl >> $LOG 2>&1
if [ -d $O/errors ] && [ "$(ls -A $O/errors)" ]; then mkdir -p $O/errors_archive && mv $O/errors/* $O/errors_archive/; fi
echo "=== E1 episodes $(date -u +%FT%TZ)" >> $LOG
$PY runner.py --manifest $O/manifests/e1.jsonl --out $O --types episodes --workers ${WORKERS:-16} >> $LOG 2>&1
for PASS in 1 2; do
  echo "=== E1 prederr+decision pass $PASS $(date -u +%FT%TZ)" >> $LOG
  $PY runner.py --manifest $O/manifests/e1.jsonl --out $O --types prederr,decision --workers ${CELL_WORKERS:-3} >> $LOG 2>&1
done
echo "=== E1 final retry, all types $(date -u +%FT%TZ)" >> $LOG
$PY runner.py --manifest $O/manifests/e1.jsonl --out $O --workers 2 >> $LOG 2>&1
$PY status.py --manifest $O/manifests/e1.jsonl --out $O >> $LOG 2>&1
echo "E1_DONE $(date -u +%FT%TZ)" >> $LOG
