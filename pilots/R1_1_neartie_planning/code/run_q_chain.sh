#!/bin/bash
# Amendment 1 (PLAN.md section 7): Q-only and WM+Q ("all") ablation episodes, started as soon as the WM grid is done
# (runs alongside decision.py), then the full analysis. Writes /workspace/Q_DONE or /workspace/Q_FAILED.
source /root/venv_mw/bin/activate
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH}
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
cd /workspace/code
O=/workspace/out_main
TASKS=walker-walk,cheetah-run,mw-push,mw-pick-place
until [ -f /workspace/MAIN_GRID_DONE ] && [ -f /workspace/PREPQ_DONE ]; do sleep 10; done
echo "=== q grid $(date -u +%H:%M:%S)"
python grid.py --tasks $TASKS --scope q,all --conds weak,rand_pred --alphas 0.1,0.2,0.3 --episodes 10 \
  --workers ${WORKERS:-7} --out $O || { touch /workspace/Q_FAILED; exit 1; }
until [ -f /workspace/MAIN_DONE ] || [ -f /workspace/MAIN_FAILED ]; do sleep 10; done
echo "=== analyze $(date -u +%H:%M:%S)"
python analyze.py --tasks $TASKS --alphas 0.05,0.1,0.2,0.3,0.5 --out $O && touch /workspace/Q_DONE || touch /workspace/Q_FAILED
echo "=== done $(date -u +%H:%M:%S)"
