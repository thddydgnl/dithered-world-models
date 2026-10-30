#!/bin/bash
# Sequential GPU-0 queue of runner manifests for the DINO-WM part (one runner process at a time). A manifest is rerun
# after a crash (e.g. CUDA OOM while the GPU is shared): the runner skips finished jobs and takes over stale locks.
# usage: DINOWM_GPU=<0|1> [LAUNCH=run_pm.sh] bash run_queue.sh <manifest name> [...]   (log lines: QUEUE ...)
cd "$(dirname "$0")"
: "${DINOWM_GPU:?set DINOWM_GPU (0 or 1) explicitly}"
export DINOWM_GPU CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$DINOWM_GPU
for man in "$@"; do
  for try in $(seq 1 20); do
    echo "QUEUE $(date -u +%FT%TZ) start $man try $try (GPU $DINOWM_GPU)"
    bash ${LAUNCH:-run.sh} runner.py manifests/$man.json --replay-workers 8
    echo "QUEUE $(date -u +%FT%TZ) runner exit $? ($man)"
    python3 status.py manifests/$man.json && break
    sleep 300
  done
done
echo "QUEUE $(date -u +%FT%TZ) ALL_DONE $*"
