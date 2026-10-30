#!/bin/bash
# E4 simulation runner (paper A). Server only, CPU only: <= 8 worker processes, one thread each, nice 10, no GPU.
# usage: bash e4_run.sh <res_dir> [n_workers]      (run inside tmux A-E4-sim; resumable: finished jobs are skipped)
set -u
RES=${1:-$HOME/pmx/A/theory/results/E4}
NW=${2:-8}
CODE=$HOME/pmx/A/theory/code
PY=$HOME/pmx/envs/a-theory/bin/python
LOG=$HOME/pmx/A/logs/E4_sim.log
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
mkdir -p "$RES/jobs"
rm -f "$RES"/jobs/*.claim "$RES"/jobs/*.tmp*
echo "$(date -u +%FT%TZ) E4 start res=$RES workers=$NW" >> "$LOG"
for w in $(seq 0 $((NW-1))); do
  nice -n 10 "$PY" "$CODE/e4_jobs.py" work "$RES" "$w" "$NW" >> "$LOG" 2>&1 &
done
wait
N_DONE=$(ls "$RES"/jobs/*.json 2>/dev/null | wc -l)
echo "$(date -u +%FT%TZ) E4 workers exited, $N_DONE job files" >> "$LOG"
