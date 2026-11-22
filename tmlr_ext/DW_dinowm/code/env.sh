# Environment of every dw job on a RunPod pod (source this). Paper A TMLR extension, DINO-WM group.
export ATM=/workspace/atm
export DINOWM_MAIN=$ATM/big DINOWM_REPO=$ATM/big/dino_wm DINOWM_CKPT=$ATM/big/data/dinowm DINOWM_DS=$ATM/big/ds
export DW_RANGES=$ATM/out/calib
export DINOWM_POINTMAZE=$(cat $ATM/WITH_PM 2>/dev/null || echo 0)
export TORCH_HOME=$ATM/big/torch HF_HOME=$ATM/big/hf
export SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=$ATM/big/inductor_cache TRITON_CACHE_DIR=$ATM/big/triton_cache
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export MUJOCO_PY_MUJOCO_PATH=$ATM/big/envs/mujoco210 MUJOCO_PY_FORCE_CPU=1
export LD_LIBRARY_PATH=$MUJOCO_PY_MUJOCO_PATH/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PATH=$ATM/big/envs/dw/bin:$PATH
export PY=$ATM/big/envs/dw/bin/python
