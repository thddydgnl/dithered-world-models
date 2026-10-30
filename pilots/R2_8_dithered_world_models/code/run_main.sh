#!/bin/bash
# R2-8 main: grid (520 episodes) -> decision (S1 metric for 51 conditions; plan reruns for the 12 R2-8 conditions) -> analyze.
source /root/venv_mw/bin/activate
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH}
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
pgrep -f nvidia-cuda-mps-control > /dev/null || nvidia-cuda-mps-control -d
cd /workspace/code
O=/workspace/out_main
TASKS=mw-push,mw-pick-place,cheetah-run,walker-walk
rm -f /workspace/MAIN_FAILED
step() { echo "=== $1 $(date -u +%H:%M:%S)"; }
fail() { echo "MAIN FAILED at $1"; touch /workspace/MAIN_FAILED; exit 1; }
step grid;     python grid.py --tasks $TASKS --episodes 10 --kbits 3,4,5 --workers 7 --out $O || fail grid
touch /workspace/MAIN_GRID_DONE
step decision; python decision.py --tasks $TASKS --condlist A:0.1,B:0.1,C:0.1,D1z:0.1,D3z:0.1,D1m:0.1,D3m:0.1,A:0.2,B:0.2,C:0.2,D1z:0.2,D3z:0.2,D1m:0.2,D3m:0.2,A:0.3,B:0.3,C:0.3,D1z:0.3,D3z:0.3,D1m:0.3,D3m:0.3,Dfix1_s0:0.1,Dfix1_s1:0.1,Dfix1_s2:0.1,Dfix3_s0:0.1,Dfix3_s1:0.1,Dfix3_s2:0.1,Ddir1_s0:0.1,Ddir1_s1:0.1,Ddir1_s2:0.1,Dfix1_s0:0.2,Dfix1_s1:0.2,Dfix1_s2:0.2,Dfix3_s0:0.2,Dfix3_s1:0.2,Dfix3_s2:0.2,Ddir1_s0:0.2,Ddir1_s1:0.2,Ddir1_s2:0.2,Qdet3:3,Qind3:3,Qsh3:3,Qsub3:3,Qdet4:4,Qind4:4,Qsh4:4,Qsub4:4,Qdet5:5,Qind5:5,Qsh5:5,Qsub5:5 --plan_conds Qdet3:3,Qind3:3,Qsh3:3,Qsub3:3,Qdet4:4,Qind4:4,Qsh4:4,Qsub4:4,Qdet5:5,Qind5:5,Qsh5:5,Qsub5:5 --max_states 150 --plan_states 150 --out $O || fail decision
touch /workspace/MAIN_DECISION_DONE
step analyze;  python analyze_r28.py --tasks $TASKS --kbits 3,4,5 --out $O --inputs /workspace/inputs || fail analyze
touch /workspace/MAIN_DONE
step done
