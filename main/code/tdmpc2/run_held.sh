#!/bin/bash
# E0 task / bit-width selection on the held split (seeds 12000-12019; MW 20, DMC 10 episodes per checkpoint seed): none and det at
# k = 3..6 for the 12 candidate tasks x checkpoint seeds 1-3 (episode-major chunks of 5: the first 5 episodes of every
# cell finish before the next 5 start; DMC: 10 episodes per checkpoint seed). Then analyze_e0.py held -> <out>/ktable.json.
#   tmux new-session -d -s A-E0-held "WORKERS=16 bash ~/pmx/A/tdmpc2_main/code/run_held.sh"
source ~/pmx/A/tdmpc2_main/code/env.sh
cd ~/pmx/A/tdmpc2_main/code
bash mps.sh start
O=~/pmx/A/tdmpc2_main/out
LOG=~/pmx/A/logs/E0_tdmpc2_held.log
[ -f $O/manifests/held.jsonl ] || $PY make_manifest.py held --out $O/manifests/held.jsonl >> $LOG 2>&1
for PASS in 1 2; do
  echo "=== held pass $PASS $(date -u +%FT%TZ)" >> $LOG
  $PY runner.py --manifest $O/manifests/held.jsonl --out $O --workers ${WORKERS:-16} >> $LOG 2>&1
done
$PY analyze_e0.py held --res $O --out $O >> $LOG 2>&1
echo "HELD_DONE $(date -u +%FT%TZ)" >> $LOG
