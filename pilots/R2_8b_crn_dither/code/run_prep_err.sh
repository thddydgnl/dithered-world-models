#!/bin/bash
# R2-8b: re-estimate the held-out prediction errors with 20 dither seeds (reporting only; runs alongside the main grid,
# never rewrites the scales file). Overwrites <out>/results/prep_<task>.json (3-seed version -> 20-seed version).
source /root/venv_mw/bin/activate
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH}
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
cd /workspace/code
O=/workspace/out_main
for T in mw-push mw-pick-place cheetah-run walker-walk; do
  cp $O/results/prep_$T.json $O/results/prep3_$T.json
  python prep_r28b.py --task $T --kbits 3,4 --nseeds 20 --out $O --inputs /workspace/inputs > /workspace/logs/prep20_$T.log 2>&1 &
done
wait
grep -h "PREP .* done" /workspace/logs/prep20_*.log | wc -l | grep -q 4 && touch /workspace/PREP20_DONE || touch /workspace/PREP20_FAILED
