#!/bin/bash
# One-off stages of the DINO-WM v5 chain (v5_chain.py runs `bash v5_cmd.sh <name>` once, after its dependencies).
# DINOWM_GPU is set by the chain. Exit 0 = done.
set -u
: "${DINOWM_GPU:?set DINOWM_GPU}"
cd "$(dirname "$0")"
case "$1" in
  qat_pusht_k4)
    # DESIGN_main v5 section 7.3: QAT on PushT k=4, about 2 GPU-hours (cumulative over resumed sessions), det
    # fake-quant + STE at the E1 site and ranges, DINO-WM latent loss, AdamW lr 5e-5, batch 32
    bash run.sh qat.py pusht --k 4 --name qat_k4 --steps 1000000 --bs 32 --workers 12 --max_minutes 120 &&
      python3 -c "import json,sys; d=json.load(open('$HOME/pmx/A/dinowm_main/qat/pusht_qat_k4/log.json')); sys.exit(0 if d.get('final') else 1)"
    ;;
  e5_select)
    # best activation-PTQ baseline per eligible (task, k) from the held det runs -> manifests/E5_dinowm_eval.json
    [ -f e5_select.py ] || { echo "e5_select.py not written yet"; exit 2; }
    python3 e5_select.py
    ;;
  *) echo "unknown v5 command $1"; exit 2 ;;
esac
