#!/bin/bash
# R2-8b main: grid (4 cells x (baseline + 5 variants) x 20 episodes, seeds 2000-2019) -> decision (150 states, value
# metrics + plan reruns for the 5 variants per cell) -> analyze.
source /root/venv_mw/bin/activate
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH}
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
pgrep -f nvidia-cuda-mps-control > /dev/null || nvidia-cuda-mps-control -d
cd /workspace/code
O=/workspace/out_main
CELLS=mw-push:3,mw-pick-place:3,cheetah-run:4,walker-walk:4
rm -f /workspace/MAIN_FAILED
step() { echo "=== $1 $(date -u +%H:%M:%S)"; }
fail() { echo "MAIN FAILED at $1"; touch /workspace/MAIN_FAILED; exit 1; }
step grid
python grid.py --cells $CELLS --episodes 20 --seed0 2000 --workers ${WORKERS:-16} --out $O || fail grid
touch /workspace/MAIN_GRID_DONE
step decision
for T in mw-push:3 mw-pick-place:3 cheetah-run:4 walker-walk:4; do
  K=${T#*:}; C=Qdet$K:$K,Qind$K:$K,Qshfwd$K:$K,Qshstep$K:$K,Qshep$K:$K
  python decision.py --tasks ${T%%:*} --condlist $C --plan_conds $C --max_states 150 --plan_states 150 --out $O \
    > /workspace/logs/decision_${T%%:*}.log 2>&1 &
done
wait
grep -h "DECISION_DONE" /workspace/logs/decision_*.log | wc -l | grep -q 4 || fail decision
touch /workspace/MAIN_DECISION_DONE
step analyze
python analyze_r28b.py --cells $CELLS --out $O || fail analyze
touch /workspace/MAIN_DONE
step done
