#!/bin/bash
# Launch one E8c stage on the pod (detached by the caller with setsid nohup). Usage: run.sh smoke|quick|main|analyze [args]
export TDMPC2_DIR=/root/tdmpc2/tdmpc2 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0
PY=/root/venv/bin/python
C=/workspace/e8c/code
IN=/workspace/e8c/e8_inputs
cd $C
case "$1" in
  smoke) shift; exec $PY $C/e8c_smoke.py --e8in $IN --out /workspace/results/smoke "$@" ;;
  quick) shift; exec $PY $C/e8c_main.py --e8in $IN --smoke /workspace/results/smoke --out /workspace/results/quick \
           --quick "$@" ;;
  main) shift; exec $PY $C/e8c_main.py --e8in $IN --smoke /workspace/results/smoke --out /workspace/results/main "$@" ;;
  analyze) shift; exec $PY $C/e8c_analyze.py --main /workspace/results/main --outdir /workspace/results/analysis \
             --e8 $IN/e8_summary.json "$@" ;;
  analyze-quick) shift; exec $PY $C/e8c_analyze.py --main /workspace/results/quick \
                   --outdir /workspace/results/quick_analysis --e8 $IN/e8_summary.json "$@" ;;
  *) echo "unknown stage $1"; exit 2 ;;
esac
