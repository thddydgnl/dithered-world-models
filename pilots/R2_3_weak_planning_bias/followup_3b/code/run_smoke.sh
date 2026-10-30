#!/bin/bash
# Smoke (R2-3b): 1 task, 2 calib episodes, alpha 0.2, 1 episode per condition, decision on 10 states.
source /root/venv_mw/bin/activate
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH}
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
pgrep -f nvidia-cuda-mps-control > /dev/null || nvidia-cuda-mps-control -d
cd /workspace/code
O=/workspace/out_smoke
T=${SMOKE_TASK:-mw-push}
rm -f /workspace/SMOKE_DONE /workspace/SMOKE_FAILED
mkdir -p $O/data $O/results
python calib.py --tasks $T --episodes 2 --workers 2 --out $O && \
python prep.py --tasks $T --alphas 0.2 --out $O && \
python grid.py --tasks $T --episodes 1 --alphas 0.2 --workers 7 --out $O && \
python decision.py --tasks $T --alphas 0.2 --conds A,Dfix1_s0,Dfix3_s0,Ddir1_s0 --max_states 10 --plan_states 5 --out $O && \
touch /workspace/SMOKE_DONE && exit 0
touch /workspace/SMOKE_FAILED
