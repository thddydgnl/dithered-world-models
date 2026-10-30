#!/bin/bash
# v5 runs of the TD-MPC2 part (PREPARED; start only after the coordinator's go):
#   bash run_v5.sh <tag> <manifest> <out dir> <gpu 0|1> <mps|nomps> <workers>
#   e.g. post-hoc H2a rerun beside PushT on GPU 0:  run_v5.sh h2a_indep ../v5/manifests/h2a_indep_min.jsonl \
#            ~/pmx/A/tdmpc2_main/out_h2a_indep 0 nomps 2
#        E2 on a whole free GPU with MPS:          run_v5.sh e2 ../v5/manifests/e2.jsonl ~/pmx/A/tdmpc2_main/out_e2 1 mps 16
#   in tmux: tmux new-session -d -s A-v5-<tag> "bash ~/pmx/A/tdmpc2_main/code/run_v5.sh ..."
# mps refuses to start when a non-MPS process is on that GPU (MPS clients then fail with CUDA error 205).
# Same-GPU rule (design 6.0): every job of one manifest runs on the GPU given here (recorded per job in env.cvd).
# relative manifest / out paths are taken relative to ~/pmx/A/tdmpc2_main/code
set -u
cd ~/pmx/A/tdmpc2_main/code
TAG=$1; MAN=$(readlink -f "$2"); OUT=$(readlink -m "$3"); GPU=${4:-0}; MODE=${5:-nomps}; W=${6:-2}
[ -f "$MAN" ] || { echo "no manifest $MAN"; exit 1; }
export A_GPU=$GPU
source ~/pmx/A/tdmpc2_main/code/env.sh
LOG=~/pmx/A/logs/v5_${TAG}.log
mkdir -p "$OUT"
[ -e "$OUT/scales" ] || ln -s ~/pmx/A/tdmpc2_main/out/scales "$OUT/scales"
[ -e "$OUT/calib" ] || ln -s ~/pmx/A/tdmpc2_main/out/calib "$OUT/calib"
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$GPU" '$1 == g {print $2}')
echo "=== v5 $TAG gpu $GPU ($UUID) $MODE workers $W $(date -u +%FT%TZ)" >> "$LOG"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader >> "$LOG"
if [ "$MODE" = mps ]; then
  # own daemon already up (e.g. a restart): reuse it. Otherwise the GPU must be free: process names are not visible in
  # this container, so any compute process on that GPU blocks the start
  if [ -S "$CUDA_MPS_PIPE_DIRECTORY/control" ] && echo get_server_list | nvidia-cuda-mps-control > /dev/null 2>&1; then
    echo "own MPS daemon already running ($CUDA_MPS_PIPE_DIRECTORY)" >> "$LOG"
  elif nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$UUID"; then
    echo "REFUSED: processes on GPU $GPU; MPS needs the whole GPU" >> "$LOG"; exit 1
  else
    bash mps.sh start "$GPU" >> "$LOG" 2>&1
  fi
else
  unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
fi
nice -n 5 $PY runner.py --manifest "$MAN" --out "$OUT" --workers "$W" >> "$LOG" 2>&1
$PY status.py --manifest "$MAN" --out "$OUT" --workers "$W" >> "$LOG" 2>&1
echo "V5_${TAG}_DONE $(date -u +%FT%TZ)" >> "$LOG"
