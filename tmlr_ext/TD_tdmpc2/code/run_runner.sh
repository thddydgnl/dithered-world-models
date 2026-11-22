#!/bin/bash
# Run one manifest with runner.py under MPS. Usage: run_runner.sh <manifest> <workers> [extra runner args]
# Results: /workspace/results/jobs/<type>/<id>.json.gz (existing results are skipped, so a rerun resumes).
source /root/e9/code/env_e9.sh
bash /root/e9/code/mps_e9.sh start
cd /root/e9/code
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) start $1 workers $2 gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) nproc $(nproc)"
$PY runner.py --manifest "$1" --out /workspace/results --workers "$2" "${@:3}"
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) end $1"
