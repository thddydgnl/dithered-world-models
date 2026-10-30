# source me: runtime environment of every TD-MPC2 job of paper A on kiis-mvf-gpu (GPU 0 only).
# One venv (a-mw) for Meta-World and DMC so that every cell runs with the same binaries; DMC needs no renderer
# (MUJOCO_GL=disable); mujoco-py (Meta-World) needs MuJoCo 2.1 and the user-space OSMesa from micromamba env gl24.
# A_GPU (optional, v5): run on GPU 1 with its own MPS pipe; unset = GPU 0 and the E0/E1 pipe (unchanged default)
export CUDA_VISIBLE_DEVICES=${A_GPU:-0}
export MUJOCO_GL=disable
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TDW=$HOME/pmx/A/tdmpc2_main
export TDMPC2_DIR=$TDW/tdmpc2/tdmpc2 NT_CKPT=$TDW/ckpt R28_INPUTS=$TDW/inputs/r28
export MUJOCO_PY_MUJOCO_PATH=$HOME/pmx/envs/mujoco210
export LD_LIBRARY_PATH=$HOME/pmx/envs/mujoco210/bin:$HOME/pmx/envs/micromamba/envs/gl24/lib
if [ "${A_GPU:-0}" = 0 ]; then
  export CUDA_MPS_PIPE_DIRECTORY=$HOME/pmx/A/mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/pmx/A/mps/log
else
  export CUDA_MPS_PIPE_DIRECTORY=$HOME/pmx/A/mps/pipe_g$A_GPU CUDA_MPS_LOG_DIRECTORY=$HOME/pmx/A/mps/log_g$A_GPU
fi
export PY=$HOME/pmx/envs/a-mw/bin/python
