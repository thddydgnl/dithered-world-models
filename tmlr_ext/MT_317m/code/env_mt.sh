# source me: runtime environment of every E14 job on a RunPod pod (setup_pod_mt.sh; standard layout of WORKER_RULES 3)
export PATH=/root/venv/bin:$PATH
export MUJOCO_GL=disable MUJOCO_PY_FORCE_CPU=1
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
export TDMPC2_DIR=/root/tdmpc2/tdmpc2
export ATM=/workspace/atm ATM_OUT=/workspace/atm/out ATM_BIG=/workspace/atm/big
export ATM_TAG=$(cat /workspace/atm/TAG 2>/dev/null || echo "?")
# names read by E10's mtlib.py (unchanged): checkpoint and the mt80 dims JSON built on this pod by mt_dims.py
export E10_CKPT=/workspace/atm/big/ckpt/mt80-317M.pt E10_DIMS=/workspace/atm/out/env/mt80_dims.json
export PY=/root/venv/bin/python
