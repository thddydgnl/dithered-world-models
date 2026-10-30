#!/bin/bash
# E0 throughput on the server A5000 (GPU 0; the DINO-WM worker may share it - its load is sampled every 5 s).
# Fixed-duration runs of E1-like job sets (det / sh-iter at the pilot k, R2-8 ranges, checkpoint 1):
#   dmc: cheetah-run, 1 episode per job, run for DMC_S seconds;  mw: mw-push + mw-pick-place, 2 episodes per job, MW_S s
# Configurations: mpsW = W workers under the own MPS daemon with the E1 instrumentation; mpsWn = same without
# instrumentation (overhead); nompsW = W workers without MPS. Each run starts in a fresh directory (every job really
# runs) and is stopped by runner.py --max_seconds; analyze_e0.py bench uses the completions after the warm-up.
# Logs: ~/pmx/A/logs/E0_bench_<cfg>_<part>.log, GPU samples: ~/pmx/A/logs/E0_bench_gpu.csv
source ~/pmx/A/tdmpc2_main/code/env.sh
cd ~/pmx/A/tdmpc2_main/code
bash mps.sh start
B=~/pmx/A/tdmpc2_main/out_bench
DMC_S=${DMC_S:-480}
MW_S=${MW_S:-360}
mkdir -p $B
$PY make_manifest.py bench --part mw --mw_eps 400 --out $B/bench_mw.jsonl
$PY make_manifest.py bench --part dmc --dmc_eps 120 --out $B/bench_dmc.jsonl
$PY make_manifest.py bench --part mw --mw_eps 400 --noinstr --out $B/benchn_mw.jsonl
$PY make_manifest.py bench --part dmc --dmc_eps 120 --noinstr --out $B/benchn_dmc.jsonl
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 5 >> ~/pmx/A/logs/E0_bench_gpu.csv &
SMI=$!
for CFG in ${BENCH_CFGS:-mps16 mps8 mps24 mps16n nomps16}; do
  W=$(echo $CFG | tr -cd '0-9')
  M=bench; [[ $CFG == *n ]] && M=benchn
  for PART in dmc mw; do
    S=$DMC_S; [ $PART = mw ] && S=$MW_S
    rm -rf $B/$CFG/$PART
    echo "=== $CFG $PART W=$W start $(date -u +%H:%M:%S)"
    if [[ $CFG == nomps* ]]; then
      (unset CUDA_MPS_PIPE_DIRECTORY; $PY runner.py --manifest $B/${M}_$PART.jsonl --out $B/$CFG/$PART --workers $W \
        --max_seconds $S > ~/pmx/A/logs/E0_bench_${CFG}_$PART.log 2>&1)
    else
      $PY runner.py --manifest $B/${M}_$PART.jsonl --out $B/$CFG/$PART --workers $W --max_seconds $S \
        > ~/pmx/A/logs/E0_bench_${CFG}_$PART.log 2>&1
    fi
    echo "=== $CFG $PART end $(date -u +%H:%M:%S) $(tail -1 ~/pmx/A/logs/E0_bench_${CFG}_$PART.log)"
    sleep 10
  done
done
kill $SMI
echo BENCH_DONE
