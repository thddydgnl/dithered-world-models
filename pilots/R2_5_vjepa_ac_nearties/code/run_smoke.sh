#!/bin/bash
W=/workspace/r25
export TORCH_HOME=/workspace/cache/torch OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd $W
mkdir -p smoke logs
python3 code/prep.py $W/smoke/prep 5 4 > logs/smoke_prep.log 2>&1 || { echo FAIL prep > $W/smoke/FAIL; exit 1; }
python3 code/checks.py $W/smoke/prep $W/smoke/checks.json > logs/smoke_checks.log 2>&1 || echo "FAIL checks" > $W/smoke/FAIL_checks
python3 code/run.py $W/smoke/prep $W/smoke/run --prec tf32 --nmatch 4 --save_every 1 > logs/smoke_run.log 2>&1 || { echo FAIL run > $W/smoke/FAIL; exit 1; }
echo SMOKE1_DONE > $W/smoke/SMOKE1_DONE
