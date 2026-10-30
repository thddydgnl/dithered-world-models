#!/bin/bash
# R2-14 phase A+B: unquantized calibration rollouts (default 6 iterations, seeds 1000-1009 as in R2-8/R2-8b; used only
# for the decision states of plandev and the held-out set of prep), then prep_r28b (writes the scales file with R2-8's
# quantizer ranges; prediction errors are a by-product, nseeds 3).
source /root/venv_mw/bin/activate
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH}
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
pgrep -f nvidia-cuda-mps-control > /dev/null || nvidia-cuda-mps-control -d
cd /workspace/code
O=/workspace/out_main
mkdir -p $O/data $O/results /workspace/logs
rm -f /workspace/PREP_DONE /workspace/PREP_FAILED
echo "=== calib $(date -u +%H:%M:%S)"
python calib.py --tasks mw-push,cheetah-run --episodes 10 --workers 10 --out $O || { touch /workspace/PREP_FAILED; exit 1; }
echo "=== prep $(date -u +%H:%M:%S)"
python prep_r28b.py --task mw-push --kbits 3 --nseeds 3 --out $O --inputs /workspace/inputs > /workspace/logs/prep_mw-push.log 2>&1 &
python prep_r28b.py --task cheetah-run --kbits 4 --nseeds 3 --out $O --inputs /workspace/inputs > /workspace/logs/prep_cheetah-run.log 2>&1 &
wait
grep -h "PREP .* done" /workspace/logs/prep_*.log | wc -l | grep -q 2 && touch /workspace/PREP_DONE || touch /workspace/PREP_FAILED
echo "=== prep end $(date -u +%H:%M:%S)"
