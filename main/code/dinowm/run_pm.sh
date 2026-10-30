#!/bin/bash
# Launcher for PointMaze DINO-WM jobs (venv a-dinowm-pm with mujoco-py/OSMesa); otherwise identical to run.sh.
: "${DINOWM_GPU:?set DINOWM_GPU (0 or 1) explicitly in the calling script}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$DINOWM_GPU     # nvidia-smi index = CUDA index
export TORCH_HOME=$HOME/pmx/torch HF_HOME=$HOME/pmx/hf
export SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=$HOME/pmx/A/dinowm_main/inductor_cache_pm TRITON_CACHE_DIR=$HOME/pmx/A/dinowm_main/triton_cache_pm
export DINOWM_POINTMAZE=1
export PATH=$HOME/pmx/envs/a-dinowm-pm/bin:$PATH
export MUJOCO_PY_MUJOCO_PATH=$HOME/pmx/envs/mujoco210
export LD_LIBRARY_PATH=$HOME/pmx/envs/mujoco210/bin:$HOME/pmx/envs/micromamba/envs/gl24/lib
cd "$(dirname "$0")"
exec $HOME/pmx/envs/a-dinowm-pm/bin/python "$@"
