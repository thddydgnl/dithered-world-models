#!/bin/bash
# After the held selection (run_held.sh): (1) supplementary held cells at k = 2 (det and sh-iter) for the tasks whose
# primary k is 3, as information for the k-1 = 2 decision (not part of the selection rule); (2) analysis again;
# (3) the throughput benchmark (run_bench.sh). Log: ~/pmx/A/logs/E0_tdmpc2_post.log
#   tmux new-session -d -s A-E0-post "bash ~/pmx/A/tdmpc2_main/code/run_post_held.sh"
source ~/pmx/A/tdmpc2_main/code/env.sh
cd ~/pmx/A/tdmpc2_main/code
O=~/pmx/A/tdmpc2_main/out
LOG=~/pmx/A/logs/E0_tdmpc2_post.log
until grep -q HELD_DONE ~/pmx/A/logs/E0_tdmpc2_held.log; do sleep 30; done
echo "=== post start $(date -u +%FT%TZ)" >> $LOG
TASKS=$($PY -c "import json; kt = json.load(open('$O/ktable.json')); print(','.join(t for t, v in kt.items() if v['primary'] == 3))")
echo "tasks with primary k = 3: $TASKS" >> $LOG
if [ -n "$TASKS" ]; then
  $PY make_manifest.py heldx --tasks $TASKS --ks 2 --variants det,shiter --out $O/manifests/heldx_k2.jsonl >> $LOG 2>&1
  $PY runner.py --manifest $O/manifests/heldx_k2.jsonl --out $O --workers ${WORKERS:-16} >> $LOG 2>&1
  $PY analyze_e0.py held --res $O --out $O >> $LOG 2>&1
fi
echo "=== bench start $(date -u +%FT%TZ)" >> $LOG
DMC_S=${DMC_S:-420} MW_S=${MW_S:-300} bash run_bench.sh >> $LOG 2>&1
$PY analyze_e0.py bench --res ~/pmx/A/logs --out $O >> $LOG 2>&1
echo "POST_DONE $(date -u +%FT%TZ)" >> $LOG
