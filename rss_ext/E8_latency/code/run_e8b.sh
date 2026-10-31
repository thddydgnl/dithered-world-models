#!/bin/bash
# E8b on the Blackwell pod (image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404): the image's python and PyTorch only,
# no installs besides rsync. Usage (detached): run_e8b.sh
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0
mkdir -p /workspace/results/e8b /workspace/logs
PY=$(command -v python3)
$PY -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))" \
  > /workspace/results/e8b/env_python.txt 2>&1
nvidia-smi > /workspace/results/e8b/nvidia_smi.txt 2>&1
$PY -m pip freeze 2>/dev/null | grep -iE "^(torch|numpy|triton|nvidia-cublas)" > /workspace/results/e8b/freeze.txt
exec $PY /workspace/e8/code/e8b_bench.py --out /workspace/results/e8b/e8b.json
