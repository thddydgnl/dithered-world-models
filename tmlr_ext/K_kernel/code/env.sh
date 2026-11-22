# source me: runtime environment of the E17 closed-loop processes on the pod (E9 env_e9.sh with the atm layout). Every
# CUDA process of the closed-loop runs is a client of one MPS daemon with the default pipe directory. The latency
# measurement (run_lat.sh) stops the daemon and unsets the pipe variables (exclusive, non-MPS GPU).
export PATH=/root/venv/bin:$HOME/.local/bin:$PATH
export MUJOCO_GL=disable MUJOCO_PY_FORCE_CPU=1 MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TDMPC2_DIR=/root/tdmpc2/tdmpc2 NT_CKPT=/root/ckpt E9_RANGES=/workspace/atm/code/inputs/ranges
export E17_BIG=/workspace/atm/big CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
export PY=/root/venv/bin/python
