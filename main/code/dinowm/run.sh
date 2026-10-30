#!/bin/bash
# Launcher for every DINO-WM job of paper A on the university server: user-space caches, headless pygame, and the GPU
# given explicitly by the caller (DINOWM_GPU=0|1; the coordinator assigns GPUs, tmux inline commands inherit nothing).
# usage: DINOWM_GPU=1 bash run.sh <python script> [args...]   e.g. DINOWM_GPU=1 bash run.sh runner.py manifests/E0_repro.json
: "${DINOWM_GPU:?set DINOWM_GPU (0 or 1) explicitly in the calling script}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$DINOWM_GPU     # nvidia-smi index = CUDA index
export TORCH_HOME=$HOME/pmx/torch HF_HOME=$HOME/pmx/hf
export SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=$HOME/pmx/A/dinowm_main/inductor_cache TRITON_CACHE_DIR=$HOME/pmx/A/dinowm_main/triton_cache
cd "$(dirname "$0")"
exec $HOME/pmx/envs/a-dinowm/bin/python "$@"
