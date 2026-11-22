#!/bin/bash
# td phase A smoke on the first pod (smoke folders only; never analysed): DESIGN_tmlr 0.7 check 1 (untouched E1 qlib vs
# this qlib, same episodes), the smoke manifest (calib -> prep -> episodes / e13 / timing) with 8 workers, check 2
# (test_variants.py) and the smoke report (check_smoke.py). Everything under MPS.
source /workspace/atm/code/env_td.sh
A=/workspace/atm
C=$A/code
SO=$A/smoke_out
SB=$A/smoke_big
echo smoke > $A/logs/phase
bash $C/mps_td.sh start
mkdir -p $SO $SB
echo "=== $(date -u +%H:%M:%S) check 1: E1 code paths (untouched E1 qlib vs td qlib, same pod, same episodes)"
$PY $C/check_e1_paths.py --qlib_dir $C/inputs/e1_reference > $SO/e1paths_e1.txt 2> $A/logs/e1paths_e1.err &
P1=$!
$PY $C/check_e1_paths.py --qlib_dir $C > $SO/e1paths_td.txt 2> $A/logs/e1paths_td.err &
P2=$!
echo "=== $(date -u +%H:%M:%S) smoke manifest"
$PY $C/plan_td.py smoke --out $SO/smoke.jsonl
cd $C
for T in calib prep episodes,e13,timing; do
  echo "=== $(date -u +%H:%M:%S) smoke jobs: $T"
  ATM_BIG=$SB $PY runner.py --manifest $SO/smoke.jsonl --out $SO --workers ${WORKERS:-8} --types $T 2>&1 | tail -n 60
done
wait $P1 $P2
grep -h E1PATHS $SO/e1paths_e1.txt $SO/e1paths_td.txt
echo "=== $(date -u +%H:%M:%S) check 2: test_variants"
ATM_BIG=$SB $PY $C/test_variants.py --out $SO/test_variants.json --prep_out $SO --big $SB 2>&1 | tail -n 80
echo "=== $(date -u +%H:%M:%S) smoke report"
$PY $C/check_smoke.py --res $SO --e1paths $SO
echo "=== $(date -u +%H:%M:%S) SMOKE_ALL_DONE"
