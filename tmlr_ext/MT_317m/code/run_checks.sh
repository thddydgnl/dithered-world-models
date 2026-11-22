#!/bin/bash
# E14 phase-A checks on the first pod (DESIGN_tmlr 0.7; WORKER_RULES 6): dims, check 1 (E10 code vs this code, same pod),
# the held + bias jobs of task 1 (main-run jobs), check 2 (new variant), analysis dry run on fake data.
# usage: run_checks.sh <task1> <task2>          (log: /workspace/atm/logs/checks.log; results in /workspace/atm/out/smoke)
source /workspace/atm/code/env_mt.sh
A=/workspace/atm; C=$A/code; S=$A/out/smoke
mkdir -p $S $A/logs
echo smoke > $A/PHASE
T1=$1; T2=$2
cd $C
echo "=== $(date -u +%H:%M:%S) dims"
[ -s "$E10_DIMS" ] || $PY $C/mt_dims.py "$E10_DIMS" $A/out/env/mt80_ckpt_report.json > $A/logs/mt_dims.log 2>&1 || echo "MT_DIMS_FAILED"
$PY -c "import json; a=json.load(open('$E10_DIMS')); b=json.load(open('$C/inputs/mt80_dims_e10.json')); print('DIMS_SAME_AS_E10', a == b)"
echo "=== $(date -u +%H:%M:%S) check 1 (E10 qlib/mtlib vs E14 qlib/mtlib, two processes, same GPU)"
$PY $C/check_paths.py --impl $C/e10_ref --tasks $T1,$T2 --seed 108000 --out $S/paths_e10 > $A/logs/paths_e10.log 2>&1 &
P1=$!
$PY $C/check_paths.py --impl $C --tasks $T1,$T2 --seed 108000 --out $S/paths_new > $A/logs/paths_new.log 2>&1 &
P2=$!
wait $P1; wait $P2
$PY $C/check_paths.py --compare $S/paths_e10.npz $S/paths_new.npz | tee $S/check1.txt
echo "=== $(date -u +%H:%M:%S) held + bias jobs of $T1 (main-run jobs)"
$PY $C/run_jobs.py --out $A/out --task $T1 --stages held,bias > $A/logs/run_jobs_$T1.log 2>&1 || echo "RUN_JOBS_FAILED"
grep -h "JOB " $A/logs/run_jobs_$T1.log | cut -c1-400
echo "=== $(date -u +%H:%M:%S) check 2"
$PY $C/check_new.py --task $T1 --seed 108001 --out $S/check2.json > $A/logs/check2.log 2>&1 || echo "CHECK2_FAILED"
grep -h CHECK2 $A/logs/check2.log
echo "=== $(date -u +%H:%M:%S) analysis dry run (fake data)"
rm -rf $A/big/fake
$PY $C/make_fake_e14.py --out $A/big/fake > $A/logs/fake.log 2>&1 && \
  $PY $C/analyze_e14.py --res $A/big/fake/mt.1,$A/big/fake/mt.2 --outdir $S/fake_analysis > $S/fake_analysis.txt 2>&1 && \
  echo "FAKE_ANALYSIS_OK" || echo "FAKE_ANALYSIS_FAILED"
rm -rf $A/big/fake_incomplete
$PY $C/make_fake_e14.py --out $A/big/fake_incomplete --drop_task mw-push > $A/logs/fake2.log 2>&1 && \
  $PY $C/analyze_e14.py --res $A/big/fake_incomplete/mt.1,$A/big/fake_incomplete/mt.2 --outdir $S/fake_analysis_incomplete > $S/fake_analysis_incomplete.txt 2>&1 && \
  echo "FAKE_ANALYSIS_INCOMPLETE_OK" || echo "FAKE_ANALYSIS_INCOMPLETE_FAILED"
rm -f $A/PHASE
echo "=== $(date -u +%H:%M:%S) CHECKS_DONE"
