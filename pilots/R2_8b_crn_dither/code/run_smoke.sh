#!/bin/bash
# R2-8b smoke (one cell: mw-push k=3): baseline + 5 variants x 2 episodes on smoke seeds 9000-9001 (disjoint from the
# main seeds 2000-2019) with forward-call logging; then decision on 10 states (5 plan reruns).
source /root/venv_mw/bin/activate
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH}
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
pgrep -f nvidia-cuda-mps-control > /dev/null || nvidia-cuda-mps-control -d
cd /workspace/code
S=/workspace/out_smoke
mkdir -p $S/results && ln -sfn /workspace/out_main/data $S/data
rm -f /workspace/SMOKE_DONE /workspace/SMOKE_FAILED
NT_LOG_CALLS=1 python grid.py --cells ${SMOKE_CELL:-mw-push:3} --episodes 2 --seed0 9000 --workers 12 --out $S && \
python decision.py --tasks ${SMOKE_TASK:-mw-push} --condlist Qdet3:3,Qind3:3,Qshfwd3:3,Qshstep3:3,Qshep3:3 \
  --plan_conds Qdet3:3,Qind3:3,Qshfwd3:3,Qshstep3:3,Qshep3:3 --max_states 10 --plan_states 10 --out $S && \
touch /workspace/SMOKE_DONE && exit 0
touch /workspace/SMOKE_FAILED
