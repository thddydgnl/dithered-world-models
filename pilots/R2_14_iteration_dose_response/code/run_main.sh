#!/bin/bash
# R2-14 main: grid (2 cells x 4 iteration counts x (none + 3 variants) x 15 episodes, seeds 3000-3014) -> plandev
# (150 states per task, both tasks in parallel) -> timing (single process, GPU otherwise idle) -> analyze.
source /root/venv_mw/bin/activate
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH}
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
pgrep -f nvidia-cuda-mps-control > /dev/null || nvidia-cuda-mps-control -d
cd /workspace/code
O=/workspace/out_main
CELLS=mw-push:3,cheetah-run:4
rm -f /workspace/MAIN_FAILED
step() { echo "=== $1 $(date -u +%H:%M:%S)"; }
fail() { echo "MAIN FAILED at $1"; touch /workspace/MAIN_FAILED; exit 1; }
step grid
python grid.py --cells $CELLS --iters 1,3,6,12 --episodes 15 --seed0 3000 --workers ${WORKERS:-10} --out $O || fail grid
touch /workspace/MAIN_GRID_DONE
step plandev
python plandev_r214.py --task mw-push --k 3 --iters 1,3,6,12 --max_states 150 --out $O > /workspace/logs/plandev_mw-push.log 2>&1 &
python plandev_r214.py --task cheetah-run --k 4 --iters 1,3,6,12 --max_states 150 --out $O > /workspace/logs/plandev_cheetah-run.log 2>&1 &
wait
grep -h "PLANDEV .* done" /workspace/logs/plandev_*.log | wc -l | grep -q 2 || fail plandev
step timing
python timing_r214.py --cells $CELLS --iters 1,3,6,12 --out $O > /workspace/logs/timing.log 2>&1 || fail timing
step analyze
python analyze_r214.py --cells $CELLS --iters 1,3,6,12 --out $O || fail analyze
touch /workspace/MAIN_DONE
step done
