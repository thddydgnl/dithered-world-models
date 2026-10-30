#!/bin/bash
# Post-hoc GPU-0 queue of the TD-MPC2 part (DESIGN_main v5.3 section 10, user-approved 2026-10-24; descriptive checks).
# Armed in tmux A-posthoc-g0:
#   tmux new-session -d -s A-posthoc-g0 "bash ~/pmx/A/tdmpc2_main/code/run_posthoc_g0.sh"
# (a) start trigger, at two consecutive polls every POSTHOC_POLL s (default 300): the GPU-0 DINO-WM chain has ended
#     (tmux session A-v5-dinowm-g0 absent, or the last V5C line of ~/pmx/A/logs/v5_dinowm_g0.log is
#     "gpu0 ALL_DONE" / "gpu0 STOPPED") and GPU 0 has no compute process
# (b) starts the own MPS daemon on GPU 0 and runs QAT+none (v5.3 10.2: posthoc/manifests/qatnone.jsonl ->
#     out_posthoc_qatnone, POSTHOC_WORKERS MPS workers, default 16); before every worker launch: no foreign non-MPS
#     compute process on GPU 0 and a test MPS client starts (the run_queue_g0.sh rule); one retry; then the CPU analysis
# (c) stops the MPS daemon
# (d) waits until both DINO-WM chains (A-v5-dinowm-g0, A-v5-dinowm-g1; same test) have ended and GPU 0 has no compute
#     process at two consecutive polls, then runs the paired latency re-measurement (v5.3 10.3: e7_latency_paired.py,
#     single non-MPS process, exclusive-GPU check, 10 rounds -> v5/e7_latency_posthoc.json; not niced, as the first
#     measurement); one retry; then the CPU analysis
# (e) logs POSTHOC_DONE
# Resumable: posthoc/queue/<stage>.done marks finished stages and the runner skips finished jobs; a stage still
# incomplete after its retry is marked <stage>.failed and the queue moves on (fix, delete .failed, re-arm).
# POSTHOC_DRYRUN=1: evaluate and log the trigger tests once, change nothing, exit.
set -u
CODE=~/pmx/A/tdmpc2_main/code
PH=~/pmx/A/tdmpc2_main/posthoc
M=$PH/manifests
Q=$PH/queue
LOG=~/pmx/A/logs/posthoc_g0.log
W=${POSTHOC_WORKERS:-16}
POLL=${POSTHOC_POLL:-300}
OUTQ=~/pmx/A/tdmpc2_main/out_posthoc_qatnone
LATOUT=~/pmx/A/tdmpc2_main/v5/e7_latency_posthoc.json
mkdir -p "$Q"
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
unset A_GPU
source "$CODE/env.sh"          # CUDA_VISIBLE_DEVICES=0, MPS pipe ~/pmx/A/mps/pipe
cd "$CODE"
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1 == 0 {print $2}')
gpu0_procs() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$UUID"; }
chain_done() {  # chain_done <gpu>: the DINO-WM v5 chain of that GPU has ended
  local g=$1 lg=~/pmx/A/logs/v5_dinowm_g$1.log
  tmux has-session -t "=A-v5-dinowm-g$g" 2>/dev/null || return 0
  [ -f "$lg" ] && grep '^V5C' "$lg" | tail -1 | grep -qE " gpu$g (ALL_DONE|STOPPED)"
}
mps_up() { [ -S "$CUDA_MPS_PIPE_DIRECTORY/control" ] && echo get_server_list | nvidia-cuda-mps-control > /dev/null 2>&1; }
gpu0_types() {  # "<n type C> <n type M+C>" compute processes on GPU 0 (nvidia-smi XML; names are not visible here)
  nvidia-smi -q -x | python3 -c "
import sys, xml.etree.ElementTree as ET
r = ET.fromstring(sys.stdin.read())
c = m = 0
for g in r.findall('gpu'):
    if g.findtext('uuid') != sys.argv[1]:
        continue
    for p in g.findall('processes/process_info'):
        t = (p.findtext('type') or '').strip()
        c += t == 'C'
        m += t == 'M+C'
print(c, m)" "$UUID"
}
mps_client_ok() { timeout 120 $PY -c "import torch; torch.ones(1, device='cuda'); torch.cuda.synchronize()" > /dev/null 2>&1; }
wait_mps_clear() {  # before launching MPS workers: at most one type-C process (our MPS server) and a test client works
  local stage=$1 n=0 tc nc nm
  while true; do
    tc=$(gpu0_types); nc=${tc%% *}; nm=${tc##* }
    if [ "${nc:-9}" -le 1 ] && mps_client_ok; then
      [ $n -gt 0 ] && log "stage $stage: GPU 0 clear again ($nc type C, $nm MPS clients)"
      return 0
    fi
    [ $((n % 12)) = 0 ] && log "stage $stage waits: GPU 0 has $nc type-C (non-MPS; our MPS server counts as one) and $nm MPS-client processes, or a test MPS client failed"
    n=$((n + 1))
    sleep 300
  done
}
not_done() {   # number of jobs of manifest $1 without a result file in $2
  $PY -c "
import json, os, sys
n = 0
for l in open(sys.argv[1]):
    if l.strip():
        j = json.loads(l)
        n += not os.path.exists(f\"{sys.argv[2]}/jobs/{j['type']}/{j['id']}.json.gz\")
print(n)" "$1" "$2"
}
prep_out() {
  mkdir -p "$1"
  [ -e "$1/scales" ] || ln -s ~/pmx/A/tdmpc2_main/out/scales "$1/scales"
  [ -e "$1/calib" ] || ln -s ~/pmx/A/tdmpc2_main/out/calib "$1/calib"
  [ -e "$1/qat" ] || ln -s ~/pmx/A/tdmpc2_main/out_e5/qat "$1/qat"     # E5 QAT weights, read only
}
wait_free() {  # wait_free <label> <gpu of a chain> ...: those chains ended and 0 compute processes on GPU 0, 2 polls
  local label=$1; shift
  local free=0 polls=0 ok n c
  log "$label waits for: DINO-WM chain(s) of GPU $* ended and 0 compute processes on GPU 0, at two consecutive ${POLL}s polls"
  while [ $free -lt 2 ]; do
    ok=1
    for c in "$@"; do chain_done "$c" || ok=0; done
    n=$(gpu0_procs)
    if [ $ok = 1 ] && [ "$n" = 0 ]; then free=$((free + 1)); else free=0; fi
    polls=$((polls + 1))
    [ $((polls % (3600 / POLL))) = 0 ] && log "$label still waiting (chains ended: $ok, compute processes on GPU 0: $n)"
    [ $free -lt 2 ] && sleep "$POLL"
  done
  log "$label: trigger met"
}
cpu_py() {     # CPU-only analysis: no GPU, no MPS client, thread caps, nice 10
  env -u CUDA_MPS_PIPE_DIRECTORY -u CUDA_MPS_LOG_DIRECTORY CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 nice -n 10 $PY "$@"
}

if [ "${POSTHOC_DRYRUN:-0}" = 1 ]; then
  for g in 0 1; do if chain_done $g; then s=ended; else s=running; fi; echo "chain g$g: $s"; done
  echo "GPU 0 compute processes: $(gpu0_procs); types (C, M+C): $(gpu0_types)"
  if mps_up; then echo "own MPS daemon: up"; else echo "own MPS daemon: down"; fi
  echo "done markers: $(ls "$Q" 2>/dev/null | tr '\n' ' ')"
  echo "manifest jobs: $(grep -c . "$M/qatnone.jsonl")"
  exit 0
fi

log "=== post-hoc queue armed (pid $$, workers $W, poll ${POLL}s, GPU 0 $UUID)"
# ---------------------------------------------------------------- (a) + (b) QAT+none (v5.3 10.2)
if [ ! -e "$Q/qatnone.done" ] && [ ! -e "$Q/qatnone.failed" ]; then
  if mps_up; then
    log "own MPS daemon already running: resuming QAT+none"
  else
    wait_free "(a) start" 0
    bash "$CODE/mps.sh" start 0 >> "$LOG" 2>&1
    if ! mps_up; then log "MPS daemon did not start; queue stops"; exit 1; fi
    log "(b) MPS daemon started"
  fi
  prep_out "$OUTQ"
  left=x
  for attempt in 1 2; do
    if [ -d "$OUTQ/errors" ] && [ "$(ls -A "$OUTQ/errors")" ]; then
      mkdir -p "$OUTQ/errors_prev"; mv "$OUTQ"/errors/* "$OUTQ/errors_prev/"
    fi
    wait_mps_clear qatnone
    log "stage qatnone attempt $attempt start ($(grep -c . "$M/qatnone.jsonl") jobs in manifest)"
    nice -n 5 $PY runner.py --manifest "$M/qatnone.jsonl" --out "$OUTQ" --workers "$W" >> ~/pmx/A/logs/posthoc_qatnone.log 2>&1
    left=$(not_done "$M/qatnone.jsonl" "$OUTQ")
    log "stage qatnone attempt $attempt end: $left jobs not done"
    if [ "$left" = 0 ]; then touch "$Q/qatnone.done"; break; fi
  done
  [ -e "$Q/qatnone.done" ] || { touch "$Q/qatnone.failed"; log "stage qatnone FAILED: $left jobs not done; moving on"; }
fi
if [ -e "$Q/qatnone.done" ] && [ ! -e "$Q/qatnone_analysis.done" ]; then
  if cpu_py analyze_posthoc_qatnone.py --e5 ../out_e5 --res "$OUTQ" --manifest "$M/qatnone.jsonl" \
       --outdir ../analysis_posthoc_qatnone >> "$LOG" 2>&1; then
    touch "$Q/qatnone_analysis.done"; log "QAT+none analysis done (analysis_posthoc_qatnone)"
  else
    log "QAT+none analysis FAILED (see above)"
  fi
fi
# ---------------------------------------------------------------- (c) stop MPS
if mps_up; then
  bash "$CODE/mps.sh" stop 0 >> "$LOG" 2>&1
  sleep 10
fi
if mps_up; then log "MPS daemon still up after stop; queue stops"; exit 1; fi
touch "$Q/mps_stopped.done"
log "(c) MPS daemon stopped"
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
# ---------------------------------------------------------------- (d) paired latency re-measurement (v5.3 10.3)
if [ ! -e "$Q/e7lat.done" ] && [ ! -e "$Q/e7lat.failed" ]; then
  rc=x
  for attempt in 1 2; do
    wait_free "(d) latency attempt $attempt" 0 1
    log "stage e7lat attempt $attempt start (e7_latency_paired.py, 10 rounds, waits up to 2 h for an exclusive GPU 0)"
    $PY e7_latency_paired.py --gpu_check --wait_exclusive 7200 --out "$LATOUT" >> ~/pmx/A/logs/posthoc_e7lat.log 2>&1
    rc=$?
    log "stage e7lat attempt $attempt end: rc $rc"
    if [ "$rc" = 0 ]; then touch "$Q/e7lat.done"; break; fi
  done
  [ -e "$Q/e7lat.done" ] || { touch "$Q/e7lat.failed"; log "stage e7lat FAILED (rc $rc, see posthoc_e7lat.log)"; }
fi
if [ -e "$Q/e7lat.done" ] && [ ! -e "$Q/e7lat_analysis.done" ]; then
  if cpu_py analyze_e7_posthoc_latency.py --posthoc "$LATOUT" --first ../v5/e7_latency.json \
       --outdir ../analysis_posthoc_e7lat >> "$LOG" 2>&1; then
    touch "$Q/e7lat_analysis.done"; log "latency analysis done (analysis_posthoc_e7lat)"
  else
    log "latency analysis FAILED (see above)"
  fi
fi
# ---------------------------------------------------------------- (e)
log "stages: $(cd "$Q" && ls | tr '\n' ' ')"
touch "$Q/POSTHOC_DONE"
log "POSTHOC_DONE"
