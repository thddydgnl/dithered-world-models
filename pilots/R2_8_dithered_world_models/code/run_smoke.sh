#!/bin/bash
# R2-8 smoke / bit-width choice: mw-push, det and ind at k = 2..6, 3 episodes, with the main scales; decision on 10 states.
source /root/venv_mw/bin/activate
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH}
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
cd /workspace/code
S=/workspace/out_smoke
mkdir -p $S/results && ln -sfn /workspace/out_main/data $S/data
rm -f /workspace/SMOKE_DONE /workspace/SMOKE_FAILED
python grid.py --tasks ${SMOKE_TASK:-mw-push} --episodes 3 --kbits 2,3,4,5,6 --variants det,ind --workers 7 --out $S && \
python decision.py --tasks ${SMOKE_TASK:-mw-push} --condlist Qdet4:4,Qind4:4,Qsh4:4,A:0.2,Dfix1_s0:0.2 --plan_conds Qdet4:4,Qind4:4 --max_states 10 --plan_states 5 --out $S && \
touch /workspace/SMOKE_DONE && exit 0
touch /workspace/SMOKE_FAILED
