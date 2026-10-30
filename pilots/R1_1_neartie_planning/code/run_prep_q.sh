#!/bin/bash
source /root/venv_mw/bin/activate
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/nvidia MUJOCO_GL=egl OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
cd /workspace/code
python prep_q.py --tasks walker-walk,cheetah-run,mw-push,mw-pick-place --alphas 0.05,0.1,0.2,0.3,0.5 --out /workspace/out_main && touch /workspace/PREPQ_DONE || touch /workspace/PREPQ_FAILED
