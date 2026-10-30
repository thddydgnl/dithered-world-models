#!/bin/bash
# R2-8b phase A+B: calibration rollouts (unquantized, seeds 1000-1009 as in R2-8; 10 episodes x 4 tasks), then
# prep_r28b for the 4 tasks in parallel (R2-8 quantizer ranges + held-out 1-/3-step errors of the 5 variants).
source /root/venv_mw/bin/activate
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH}
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
pgrep -f nvidia-cuda-mps-control > /dev/null || nvidia-cuda-mps-control -d
cd /workspace/code
O=/workspace/out_main
TASKS="mw-push mw-pick-place cheetah-run walker-walk"
mkdir -p $O/data $O/results /workspace/logs
rm -f /workspace/PREP_DONE /workspace/PREP_FAILED
echo "=== calib $(date -u +%H:%M:%S)"
python calib.py --tasks mw-push,mw-pick-place,cheetah-run,walker-walk --episodes 10 --workers 16 --out $O || { touch /workspace/PREP_FAILED; exit 1; }
echo "=== prep $(date -u +%H:%M:%S)"
for T in $TASKS; do python prep_r28b.py --task $T --kbits 3,4 --nseeds 3 --out $O --inputs /workspace/inputs > /workspace/logs/prep_$T.log 2>&1 & done
wait
grep -h "PREP .* done" /workspace/logs/prep_*.log | wc -l | grep -q 4 && touch /workspace/PREP_DONE || touch /workspace/PREP_FAILED
echo "=== prep end $(date -u +%H:%M:%S)"
