#!/bin/bash
# start|stop|status of the pod's CUDA MPS daemon (default pipe directory /tmp/nvidia-mps)
source /root/e9/code/env_e9.sh
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
case "$1" in
  start) if pgrep -f "nvidia-cuda-mps-control -d" > /dev/null; then echo "MPS already running"; else nvidia-cuda-mps-control -d && echo "MPS started"; fi ;;
  stop) echo quit | nvidia-cuda-mps-control && echo "MPS stopped" ;;
  status) pgrep -af nvidia-cuda-mps; echo get_server_list | nvidia-cuda-mps-control 2>&1 | head -3 ;;
  *) echo "usage: $0 start|stop|status"; exit 1 ;;
esac
