#!/bin/bash
# GPU-0 queue of the TD-MPC2 part after E1 (DESIGN_main v5 7.5). Armed in tmux A-v5-queue-g0:
#   tmux new-session -d -s A-v5-queue-g0 "bash ~/pmx/A/tdmpc2_main/code/run_queue_g0.sh"
# 1. Start trigger (both, at two consecutive polls every QUEUE_POLL s, default 300): the last "PDRV" line of the DINO-WM
#    PushT g0 driver log ~/pmx/A/logs/E1_dinowm_pusht_g0.log is "PDRV ... gpu0 DONE" (its exit signal, coordinator
#    2026-10-22), and GPU 0 has no compute process. Then starts the own CUDA MPS daemon on GPU 0 (pipe as in E1).
# 2. Runs with QUEUE_WORKERS (default 16) MPS workers, in this order:
#      [H2a re-measure, fresh 3 variants, only if v5/FLAG_h2a_indep_fresh3 exists; checked again at every stage
#       boundary and while waiting] -> E2 -> E3 ->
#      E5 (needs v5/E5_READY): statistics -> held selection episodes + QAT data -> QAT training -> baseline
#      selection (e5_select.py, CPU) -> evaluation + weight-only + QAT evaluation ->
#      E7 success rates (needs v5/E7_READY; the queue waits for it).
# 3. Stops the MPS daemon, then runs the E7 latency measurement as the last step (e7_latency.py, single non-MPS
#    process; waits up to 2 h for an exclusive GPU 0, else gives up and logs). GPU 0 then goes back to DINO-WM.
# Resumable: the runner skips finished jobs; v5/queue/<stage>.done marks finished stages. A stage with jobs still
# missing after one retry is marked <stage>.failed and the queue moves on (fix, delete .failed, re-arm the queue).
# GPU-0 MPS rule: from the MPS start until QUEUE_DONE no non-MPS process may use GPU 0 (E7 tests run as MPS clients or
# wait). Before every worker launch the queue checks for foreign non-MPS processes and waits (and logs) while any is
# there, instead of launching workers that would die with CUDA error 205.
set -u
CODE=~/pmx/A/tdmpc2_main/code
V=~/pmx/A/tdmpc2_main/v5
M=$V/manifests
Q=$V/queue
LOG=~/pmx/A/logs/v5_queue_g0.log
W=${QUEUE_WORKERS:-16}
POLL=${QUEUE_POLL:-300}
mkdir -p "$Q"
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
unset A_GPU
source "$CODE/env.sh"          # CUDA_VISIBLE_DEVICES=0, MPS pipe ~/pmx/A/mps/pipe
cd "$CODE"
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1 == 0 {print $2}')
gpu0_procs() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$UUID"; }
PUSHT_LOG=~/pmx/A/logs/E1_dinowm_pusht_g0.log
pusht_done() { [ -f "$PUSHT_LOG" ] && grep '^PDRV' "$PUSHT_LOG" | tail -1 | grep -q 'gpu0 DONE'; }
mps_up() { [ -S "$CUDA_MPS_PIPE_DIRECTORY/control" ] && echo get_server_list | nvidia-cuda-mps-control > /dev/null 2>&1; }
prep_out() {
  mkdir -p "$1"
  [ -e "$1/scales" ] || ln -s ~/pmx/A/tdmpc2_main/out/scales "$1/scales"
  [ -e "$1/calib" ] || ln -s ~/pmx/A/tdmpc2_main/out/calib "$1/calib"
}
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
wait_mps_clear() {  # wait_mps_clear <stage>: before launching workers (coordinator rule, 2026-10-22): no foreign
  # non-MPS compute process on GPU 0 (at most one type-C process = our MPS server) and a test MPS client can start
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
run_stage() {  # run_stage <name> <manifest> <out dir>
  local name=$1 man=$2 out=$3 left=x
  [ -e "$Q/$name.done" ] && return 0
  prep_out "$out"
  for attempt in 1 2; do
    if [ -d "$out/errors" ] && [ "$(ls -A "$out/errors")" ]; then
      mkdir -p "$out/errors_prev"; mv "$out"/errors/* "$out/errors_prev/"
    fi
    wait_mps_clear "$name"
    log "stage $name attempt $attempt start ($(grep -c . "$man") jobs in manifest)"
    nice -n 5 $PY runner.py --manifest "$man" --out "$out" --workers "$W" >> ~/pmx/A/logs/v5_$name.log 2>&1
    left=$(not_done "$man" "$out")
    log "stage $name attempt $attempt end: $left jobs not done"
    if [ "$left" = 0 ]; then touch "$Q/$name.done"; rm -f "$Q/$name.failed"; return 0; fi
  done
  touch "$Q/$name.failed"
  log "stage $name FAILED: $left jobs not done; moving on"
  return 1
}
h2a_if_flag() {
  if [ -e "$V/FLAG_h2a_indep_fresh3" ] && [ ! -e "$Q/h2a.done" ] && [ ! -e "$Q/h2a.failed" ]; then
    run_stage h2a "$M/h2a_indep_min.jsonl" ~/pmx/A/tdmpc2_main/out_h2a_indep
  fi
}
wait_flag() {  # wait_flag <file> <stage>: poll every 10 min (H2a may run meanwhile)
  local first=1
  while [ ! -e "$1" ]; do
    [ $first = 1 ] && log "stage $2 waits for $(basename "$1")"
    first=0
    sleep 600
    h2a_if_flag
  done
}

log "=== queue armed (pid $$, workers $W, poll ${POLL}s, GPU 0 $UUID)"
if mps_up; then
  log "own MPS daemon already running: resuming"
else
  free=0; polls=0
  cond="last PDRV line of $(basename "$PUSHT_LOG") = 'PDRV ... gpu0 DONE' and 0 compute processes on GPU 0, at two consecutive ${POLL}s polls"
  log "waiting for: $cond"
  while [ $free -lt 2 ]; do
    n=$(gpu0_procs)
    if pusht_done; then pd=present; else pd=absent; fi
    if [ "$pd" = present ] && [ "$n" = 0 ]; then free=$((free + 1)); else free=0; fi
    polls=$((polls + 1))
    [ $((polls % (3600 / POLL))) = 0 ] && log "waiting for: $cond (now: DONE line $pd, $n compute processes on GPU 0)"
    [ $free -lt 2 ] && sleep "$POLL"
  done
  log "start trigger met: PushT g0 DONE line present and GPU 0 free at two consecutive polls"
  bash "$CODE/mps.sh" start 0 >> "$LOG" 2>&1
  if ! mps_up; then log "MPS daemon did not start; queue stops"; exit 1; fi
fi

h2a_if_flag
run_stage e2 "$M/e2.jsonl" ~/pmx/A/tdmpc2_main/out_e2
h2a_if_flag
run_stage e3 "$M/e3.jsonl" ~/pmx/A/tdmpc2_main/out_e3
h2a_if_flag
wait_flag "$V/E5_READY" e5
O5=~/pmx/A/tdmpc2_main/out_e5
run_stage e5stats "$M/e5_stats.jsonl" "$O5"
cat "$M/e5_held.jsonl" "$M/e5_qatdata.jsonl" > "$M/e5_held_qatdata.jsonl"
run_stage e5held "$M/e5_held_qatdata.jsonl" "$O5"
run_stage e5qattrain "$M/e5_qattrain.jsonl" "$O5"
if [ ! -e "$O5/e5/selection.json" ]; then
  $PY e5_select.py --res "$O5" --manifest "$M/e5_held.jsonl" >> "$LOG" 2>&1 || log "E5 selection refused (held jobs missing)"
fi
cat "$M/e5_eval.jsonl" "$M/e5_w.jsonl" "$M/e5_qateval.jsonl" > "$M/e5_eval_all.jsonl"
if [ -e "$O5/e5/selection.json" ]; then run_stage e5eval "$M/e5_eval_all.jsonl" "$O5"; else touch "$Q/e5eval.failed"; fi
h2a_if_flag
wait_flag "$V/E7_READY" e7
run_stage e7 "$M/e7.jsonl" ~/pmx/A/tdmpc2_main/out_e7
h2a_if_flag
log "stages: $(cd "$Q" && ls | tr '\n' ' ')"
bash "$CODE/mps.sh" stop 0 >> "$LOG" 2>&1
log "MPS daemon stopped"
# E7 latency (coordinator 2026-10-22): last step, after MPS is stopped; single non-MPS process; e7_latency.py checks
# exclusivity itself and waits up to 2 h (5-minute polls) for an exclusive GPU 0, then gives up (exit 2)
if [ ! -e "$Q/e7lat.done" ] && [ ! -e "$Q/e7lat.gaveup" ]; then
  unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
  log "E7 latency start (waits up to 2 h for an exclusive GPU 0)"
  $PY e7_latency.py --gpu_check --wait_exclusive 7200 --out "$V/e7_latency.json" >> ~/pmx/A/logs/v5_e7lat.log 2>&1
  rc=$?
  if [ $rc = 0 ]; then touch "$Q/e7lat.done"; log "E7 latency done"
  elif [ $rc = 2 ]; then touch "$Q/e7lat.gaveup"; log "E7 latency GAVE UP: GPU 0 not exclusive within 2 h (schedule by hand)"
  else touch "$Q/e7lat.failed"; log "E7 latency FAILED (rc $rc, see v5_e7lat.log)"; fi
fi
log "QUEUE_DONE"

