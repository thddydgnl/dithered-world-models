# source me: runtime environment of every E9 process on the pod (all CUDA processes are clients of one MPS daemon
# with the default pipe directory, so no process runs outside MPS).
export PATH=/root/venv/bin:$HOME/.local/bin:$PATH
export MUJOCO_GL=disable MUJOCO_PY_FORCE_CPU=1 MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TDMPC2_DIR=/root/tdmpc2/tdmpc2 NT_CKPT=/root/ckpt E9_RANGES=/root/e9/code/inputs/ranges
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
export PY=/root/venv/bin/python
