#!/bin/bash
# PushT E1 driver (DESIGN_main v3): waits for its start condition, then runs whole PushT cells (task, k) one after
# another on its GPU. Each k cell is claimed atomically (mkdir ~/pmx/A/dinowm_main/claims/pusht_k<k>), so a cell never
# runs on two GPUs; the none/nonealt jobs ride along with every cell run and run once (per-job locks), GPU recorded
# in every job JSON. Restarting a driver resumes its own claimed cells (finished jobs are skipped).
# usage (tmux, GPU explicit):
#   DINOWM_GPU=0 WAIT=tdmpc2 ORDER="4 3 5" bash E1_pusht_driver.sh   # GPU 0 after the TD-MPC2 E1 episode phase
#   DINOWM_GPU=1 WAIT=wp     ORDER="5 3 4" bash E1_pusht_driver.sh   # GPU 1 after the Wall/PointMaze E1 queue
#   WAIT=none starts at once.  Log lines: PDRV ...
set -u
export DINOWM_GPU=${DINOWM_GPU:?set DINOWM_GPU explicitly}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$DINOWM_GPU
cd "$(dirname "$0")"
CL=$HOME/pmx/A/dinowm_main/claims
mkdir -p "$CL"
log() { echo "PDRV $(date -u +%FT%TZ) gpu$DINOWM_GPU $*"; }
log "waiting for start condition WAIT=${WAIT:-none} (checked every 10 min)"
case "${WAIT:-none}" in
  tdmpc2) until grep -q "=== E1 prederr+decision pass 1" "$HOME/pmx/A/logs/E1_tdmpc2.log" 2>/dev/null; do sleep 600; done ;;
  wp)     until grep -q "E1Q .* ALL_DONE" "$HOME/pmx/A/logs/E1_dinowm.log" 2>/dev/null; do sleep 600; done ;;
  none)   ;;
  *) log "unknown WAIT=$WAIT"; exit 1 ;;
esac
log "start condition met"
for k in ${ORDER:?set ORDER, e.g. \"4 3 5\"}; do
  if mkdir "$CL/pusht_k$k" 2>/dev/null; then
    echo "$DINOWM_GPU" > "$CL/pusht_k$k/gpu"; log "claimed pusht k=$k"
  elif [ "$(cat "$CL/pusht_k$k/gpu" 2>/dev/null)" = "$DINOWM_GPU" ]; then
    log "resuming own pusht k=$k"
  else
    log "pusht k=$k is claimed by gpu $(cat "$CL/pusht_k$k/gpu" 2>/dev/null); skip"; continue
  fi
  MAN=E1_dinowm_pusht TASKS=pusht KS=$k bash E1_queue.sh
  log "pusht k=$k queue returned"
done
log "DONE"
