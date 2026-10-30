#!/bin/bash
# HISTORICAL: ran 14:07-16:28Z on GPU 0 (k selection of PushT and Wall; its later steps exited unrun after the
# 15:53Z GPU reassignment, see E0_queue_gpu1.sh). Since then run.sh/run_queue.sh require DINOWM_GPU; to rerun this
# script set DINOWM_GPU explicitly.
# E0 DINO-WM GPU queue on GPU 0 (tmux A-E0-dinowm, log ~/pmx/A/logs/E0_dinowm_queue.log): one GPU process at a time
# (GPU 0 is shared with the TD-MPC2 worker). Every step is resumable; rerunning this script skips finished work.
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0      # GPU 0 only (tmux inline commands do not inherit the login env); run.sh pins it too
echo "E0Q $(date -u +%FT%TZ) start"
bash run_queue.sh E0_ksel                                                     # 1. k selection PushT, Wall (held)
[ -f ~/pmx/A/dinowm_main/results/ranges/ranges_main_pointmaze.json ] || bash run_pm.sh calib.py pointmaze --ranges_only
LAUNCH=run_pm.sh bash run_queue.sh E0_smoke_pm E0_ksel_pm                     # 2. PointMaze smoke + k selection
bash run_queue.sh E0_repro                                                    # 3. R2-13 reproduction
for t in pusht wall; do                                                       # 4. prediction errors per k, variant
  [ -f ~/pmx/A/dinowm_main/results/E0_dinowm/err_${t}_main.json ] || bash run.sh calib.py $t --kbits 2,3,4,5,6,7
done
[ -f ~/pmx/A/dinowm_main/results/E0_dinowm/err_pointmaze_main.json ] || bash run_pm.sh calib.py pointmaze --kbits 2,3,4,5,6,7
echo "E0Q $(date -u +%FT%TZ) ALL_DONE"
