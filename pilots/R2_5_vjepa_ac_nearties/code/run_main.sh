#!/bin/bash
# Main run: prep (160 eval + 20 calibration windows) -> stats -> match -> window-major eval (fp16) -> analyze.
W=/workspace/r25
export TORCH_HOME=/workspace/cache/torch OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd $W
mkdir -p main logs
[ -f main/prep/prep.pt ] || python3 code/prep.py $W/main/prep 160 20 square > logs/main_prep.log 2>&1 || { echo FAIL prep > main/FAIL; exit 1; }
python3 code/run.py $W/main/prep $W/main/run --prec fp16 --nmatch 6 --save_every 5 --num_floor bf16:1000000,tf32:30,fp32:30 \
  > logs/main_run.log 2>&1 || { echo FAIL run > main/FAIL; exit 1; }
python3 code/analyze.py $W/main/prep $W/main/run > logs/main_analyze.log 2>&1 || echo FAIL analyze > main/FAIL_analyze
echo DONE > main/DONE
