#!/bin/bash
# E17 latency stage on the exclusive GPU. Usage: run_lat.sh smoke|quick|main
# Refuses while the queue driver / runner is alive; stops the MPS daemon; runs as a non-MPS single process.
source /workspace/atm/code/env.sh
O=/workspace/atm/out
if pgrep -f "runner.py --manifest" > /dev/null; then echo "REFUSED: runner.py alive"; exit 2; fi
bash /workspace/atm/code/mps.sh stop
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
[ "$1" = main ] && echo main > $O/PHASE
cd /workspace/atm/code
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) latency $1 start; mps: $(pgrep -af nvidia-cuda-mps | head -2)"
$PY /workspace/atm/code/e17_lat.py $1 --out $O/latency
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) latency $1 end rc $?"
