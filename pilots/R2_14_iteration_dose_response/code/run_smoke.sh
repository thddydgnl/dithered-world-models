#!/bin/bash
# R2-14 smoke (mw-push k=3): iterations 1 and 12, none/det/shstep/shfwd x 1 episode on smoke seed 9100 (disjoint from
# the main seeds 3000-3014) with forward-call logging; plandev on 5 states; timing with 5 timed steps.
source /root/venv_mw/bin/activate
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH}
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
pgrep -f nvidia-cuda-mps-control > /dev/null || nvidia-cuda-mps-control -d
cd /workspace/code
S=/workspace/out_smoke
mkdir -p $S/results && ln -sfn /workspace/out_main/data $S/data
rm -f /workspace/SMOKE_DONE /workspace/SMOKE_FAILED
NT_LOG_CALLS=1 python grid.py --cells mw-push:3 --iters 1,12 --episodes 1 --seed0 9100 --workers 8 --out $S && \
python plandev_r214.py --task mw-push --k 3 --iters 1,12 --max_states 5 --out $S && \
python timing_r214.py --cells mw-push:3 --iters 1,12 --warm 2 --steps 5 --out $S && \
python analyze_r214.py --cells mw-push:3 --iters 1,12 --out $S && \
touch /workspace/SMOKE_DONE && exit 0
touch /workspace/SMOKE_FAILED
