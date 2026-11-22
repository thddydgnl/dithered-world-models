#!/bin/bash
# td throughput bench (bench seeds 108000+, never analysed): fixed-time runs of one-episode jobs of the main-run mix.
# Needs the smoke prep (bias vectors, PC ranges) of the bench tasks. Usage: run_bench.sh <mw|dmc> <workers> <seconds>
source /workspace/atm/code/env_td.sh
A=/workspace/atm
C=$A/code
bash $C/mps_td.sh start
O=$A/bench/$1_w$2
mkdir -p $O
ln -sfn $A/smoke_out/prep $O/prep
$PY $C/plan_td.py bench --part $1 --n 80 --out $A/bench/bench_$1.jsonl
echo "=== $(date -u +%H:%M:%S) bench $1 workers $2 for $3 s"
(cd $C && ATM_BIG=$A/smoke_big $PY runner.py --manifest $A/bench/bench_$1.jsonl --out $O --workers $2 --max_seconds $3) | tail -3
$PY $C/bench_rate.py --res $O --prefix bench_ --workers $2 | tee $O/rate.json
echo "=== $(date -u +%H:%M:%S) BENCH_DONE $1 $2"
