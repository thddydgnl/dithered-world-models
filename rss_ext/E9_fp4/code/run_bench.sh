#!/bin/bash
# E9 throughput bench (seeds 88000+, never analysed): fixed-time runs of one-episode jobs of all 5 variants.
# Usage: run_bench.sh <part mw|dmc> <workers> <seconds>
source /root/e9/code/env_e9.sh
C=/root/e9/code
bash $C/mps_e9.sh start
O=/workspace/bench_$1_w$2
mkdir -p /workspace/manifests $O
$PY $C/make_manifest_e9.py bench --part $1 --eps 60 --out /workspace/manifests/bench_$1.jsonl
echo "=== $(date -u +%H:%M:%S) bench $1 workers $2 for $3 s"
(cd $C && $PY runner.py --manifest /workspace/manifests/bench_$1.jsonl --out $O --workers $2 --max_seconds $3) | tail -3
$PY $C/bench_rate.py --res $O --prefix bench_ --workers $2 | tee $O/rate.json
echo "=== $(date -u +%H:%M:%S) BENCH_DONE $1 $2"
