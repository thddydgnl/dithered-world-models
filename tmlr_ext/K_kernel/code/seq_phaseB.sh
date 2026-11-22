#!/bin/bash
# Phase-B analysis on the pod (WORKER_RULES 6): the pre-defined computations of PLAN.md 1 / 7.
# Usage: seq_phaseB.sh [<fq_csv on the pod: the td worker's td_episodes.csv>]
#   out/analysis/      analyze_e17.py (K2 integer side or the K2 comparison with --fq_csv, K3, part-4 descriptive,
#                      check 3, integrity)
#   out/analysis_lat/  e17_lat_analyze.py (K1; E8 cells as the descriptive cross-pod reference)
source /workspace/atm/code/env.sh
O=/workspace/atm/out
C=/workspace/atm/code
echo analysis > $O/PHASE
FQ=""
[ -n "$1" ] && FQ="--fq_csv $1"
$PY $C/e17_lat_analyze.py --main $O/latency/main --outdir $O/analysis_lat --e8 $C/inputs/e8_cells.csv | tail -2
$PY $C/analyze_e17.py --res $O --e1 $C/inputs/e1_tdmpc2_baseline.json --outdir $O/analysis $FQ | tail -2
$PY $C/bench_e17.py --res $O --workers 8 --out $O/analysis/throughput_all.json > /dev/null
md5sum $C/analyze_e17.py $C/e17_lat_analyze.py > $O/analysis/analysis_code_md5.txt
echo done > $O/PHASE
echo "=== $(date -u +%H:%M:%SZ) SEQ_PHASEB_DONE"
