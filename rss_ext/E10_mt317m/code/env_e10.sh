# source me: runtime environment of every E10 job on a RunPod pod (setup_pod_e10.sh).
export PATH=/root/venv/bin:$PATH
export MUJOCO_GL=disable MUJOCO_PY_FORCE_CPU=1
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
export TDMPC2_DIR=/root/tdmpc2/tdmpc2
export E10_CKPT=/workspace/ckpt/mt80-317M.pt E10_OUT=/workspace/e10 E10_DIMS=/workspace/e10/mt80_dims.json
export PY=/root/venv/bin/python
