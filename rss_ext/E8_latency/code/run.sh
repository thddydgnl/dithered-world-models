#!/bin/bash
# Launch one E8 stage on the pod (detached by the caller with setsid nohup). Usage: run.sh smoke|main|quick|analyze [args]
export TDMPC2_DIR=/root/tdmpc2/tdmpc2 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0
PY=/root/venv/bin/python
C=/workspace/e8/code
cd $C
case "$1" in
  smoke) shift; exec $PY $C/e8_smoke.py --out /workspace/results/smoke "$@" ;;
  quick) shift; exec $PY $C/e8_main.py --smoke /workspace/results/smoke --out /workspace/results/quick --quick "$@" ;;
  main) shift; exec $PY $C/e8_main.py --smoke /workspace/results/smoke --out /workspace/results/main "$@" ;;
  analyze) shift; exec $PY $C/e8_analyze.py --main /workspace/results/main --smoke /workspace/results/smoke \
             --outdir /workspace/results/analysis "$@" ;;
  *) echo "unknown stage $1"; exit 2 ;;
esac
