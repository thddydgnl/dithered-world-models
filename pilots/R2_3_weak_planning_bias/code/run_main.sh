#!/bin/bash
# Main run (R2-3). Phase markers /workspace/MAIN_<PHASE>_DONE, failure /workspace/MAIN_FAILED.
source /root/venv_mw/bin/activate
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH}
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
pgrep -f nvidia-cuda-mps-control > /dev/null || nvidia-cuda-mps-control -d
cd /workspace/code
O=/workspace/out_main
A=0.1,0.2,0.3
TASKS=${TASKS:-mw-push,mw-pick-place,cheetah-run,walker-walk}
W=${WORKERS:-7}
mkdir -p $O/data $O/results
rm -f /workspace/MAIN_FAILED
step() { echo "=== $1 $(date -u +%H:%M:%S)"; }
fail() { echo "MAIN FAILED at $1"; touch /workspace/MAIN_FAILED; exit 1; }
step calib;    python calib.py --tasks $TASKS --episodes 10 --workers $W --out $O || fail calib
touch /workspace/MAIN_CALIB_DONE
step prep;     python prep.py --tasks $TASKS --alphas $A --out $O || fail prep
touch /workspace/MAIN_PREP_DONE
step grid;     python grid.py --tasks $TASKS --episodes 10 --alphas $A --workers $W --out $O || fail grid
touch /workspace/MAIN_GRID_DONE
step decision; python decision.py --tasks $TASKS --alphas $A --max_states 300 --plan_states 150 --out $O || fail decision
touch /workspace/MAIN_DECISION_DONE
step analyze;  python analyze.py --tasks $TASKS --alphas $A --out $O || fail analyze
touch /workspace/MAIN_DONE
step done
