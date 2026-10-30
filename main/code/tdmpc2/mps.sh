#!/bin/bash
# Own CUDA MPS control daemon for GPU 0 only (pipe/log dirs under ~/pmx/A/mps, so only processes that source env.sh use
# it). Usage: bash mps.sh start|stop|status [gpu]   (gpu 0 = default, the E0/E1 pipe; gpu 1 = pipe_g1, v5)
G=${2:-0}
export CUDA_VISIBLE_DEVICES=$G
if [ "$G" = 0 ]; then
  export CUDA_MPS_PIPE_DIRECTORY=$HOME/pmx/A/mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/pmx/A/mps/log
else
  export CUDA_MPS_PIPE_DIRECTORY=$HOME/pmx/A/mps/pipe_g$G CUDA_MPS_LOG_DIRECTORY=$HOME/pmx/A/mps/log_g$G
fi
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
case "$1" in
  start)
    if [ -S $CUDA_MPS_PIPE_DIRECTORY/control ] && echo get_server_list | nvidia-cuda-mps-control > /dev/null 2>&1; then
      echo "MPS already running"
    else
      nvidia-cuda-mps-control -d && echo "MPS started"
    fi ;;
  stop)
    echo quit | nvidia-cuda-mps-control && echo "MPS stopped" ;;
  status)
    ps -u $USER -o pid,etimes,args | grep "[n]vidia-cuda-mps" ;
    echo get_server_list | nvidia-cuda-mps-control 2>&1 | head -5 ;;
  *) echo "usage: $0 start|stop|status [gpu]"; exit 1 ;;
esac
