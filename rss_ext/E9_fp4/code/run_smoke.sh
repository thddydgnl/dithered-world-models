#!/bin/bash
# E9 smoke on the pod: unit checks (test_fp4.py), then the smoke manifest (E0 reproduction, FP4 variants with both
# implementations, heldperr) with 8 workers, then check_smoke.py. Everything under MPS.
source /root/e9/code/env_e9.sh
C=/root/e9/code
bash $C/mps_e9.sh start
mkdir -p /workspace/manifests /workspace/results/smoke
echo "=== $(date -u +%H:%M:%S) test_fp4"
$PY $C/test_fp4.py --out /workspace/results/smoke/test_fp4.json || echo "TEST_FP4_ERROR"
echo "=== $(date -u +%H:%M:%S) E1 code paths (untouched E1 qlib vs E9 qlib, same pod, same episodes)"
$PY $C/check_e1_paths.py --qlib_dir $C/inputs/e1_reference > /workspace/results/smoke/e1paths_e1.txt 2> /workspace/logs/e1paths_e1.err &
$PY $C/check_e1_paths.py --qlib_dir $C > /workspace/results/smoke/e1paths_e9.txt 2> /workspace/logs/e1paths_e9.err &
wait
grep -h E1PATHS /workspace/results/smoke/e1paths_e1.txt /workspace/results/smoke/e1paths_e9.txt
echo "=== $(date -u +%H:%M:%S) smoke manifest"
$PY $C/make_manifest_e9.py smoke --out /workspace/manifests/smoke.jsonl
(cd $C && $PY runner.py --manifest /workspace/manifests/smoke.jsonl --out /workspace/results --workers ${WORKERS:-8})
echo "=== $(date -u +%H:%M:%S) check"
$PY $C/check_smoke.py --res /workspace/results
echo "=== $(date -u +%H:%M:%S) SMOKE_ALL_DONE"
