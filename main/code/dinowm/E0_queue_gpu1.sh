#!/bin/bash
# E0 DINO-WM queue on GPU 1 (coordinator reassignment 2026-10-21 ~15:55Z: B's server E0 finished, GPU 1 idle).
# tmux A-E0-dinowm-g1, log ~/pmx/A/logs/E0_dinowm_queue_gpu1.log. The held k selection of PushT and Wall stays on
# GPU 0 (all its cells were run there); everything below runs on GPU 1 only. Resumable: rerunning skips finished work.
export DINOWM_GPU=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
cd "$(dirname "$0")"
echo "G1Q $(date -u +%FT%TZ) start (GPU $DINOWM_GPU)"
[ -f ~/pmx/A/dinowm_main/results/ranges/ranges_main_pointmaze.json ] || bash run_pm.sh calib.py pointmaze --ranges_only
bash run_queue.sh E0_bench_g1 && echo "G1Q $(date -u +%FT%TZ) bench PushT/Wall done"        # 1. solo throughput
LAUNCH=run_pm.sh bash run_queue.sh E0_bench_g1_pm
LAUNCH=run_pm.sh bash run_queue.sh E0_smoke_pm E0_ksel_pm                                    # 2. PointMaze
bash run_queue.sh E0_repro                                                                  # 3. R2-13 reproduction
for t in pusht wall; do                                                                     # 4. prediction errors
  [ -f ~/pmx/A/dinowm_main/results/E0_dinowm/err_${t}_main.json ] || bash run.sh calib.py $t --kbits 2,3,4,5,6,7
done
[ -f ~/pmx/A/dinowm_main/results/E0_dinowm/err_pointmaze_main.json ] || bash run_pm.sh calib.py pointmaze --kbits 2,3,4,5,6,7
echo "G1Q $(date -u +%FT%TZ) ALL_DONE"
