#!/bin/bash
# E1 (TD-MPC2) cell jobs (prederr, decision) without CUDA MPS. Written after the 2026-10-22 06:08Z failure: once the
# episode phase ended, the PushT E1 runner started on GPU 0 as a non-MPS process, and every new MPS client then failed
# at CUDA init ("Error 205: mapping of buffer object failed"). The MPS daemon is stopped before this script runs.
#   ssh kiis-mvf-gpu 'tmux new-session -d -s A-E1-tdmpc2-cells "bash ~/pmx/A/tdmpc2_main/code/run_e1_cells.sh"'
source ~/pmx/A/tdmpc2_main/code/env.sh
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
cd ~/pmx/A/tdmpc2_main/code
O=~/pmx/A/tdmpc2_main/out
LOG=~/pmx/A/logs/E1_tdmpc2.log
if [ -d $O/errors ] && [ "$(ls -A $O/errors)" ]; then mkdir -p $O/errors_archive2 && mv $O/errors/* $O/errors_archive2/; fi
echo "=== E1 cells without MPS $(date -u +%FT%TZ)" >> $LOG
$PY runner.py --manifest $O/manifests/e1.jsonl --out $O --types prederr,decision --workers ${CELL_WORKERS:-2} >> $LOG 2>&1
$PY status.py --manifest $O/manifests/e1.jsonl --out $O >> $LOG 2>&1
echo "E1_CELLS_DONE $(date -u +%FT%TZ)" >> $LOG
