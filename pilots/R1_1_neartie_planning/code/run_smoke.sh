#!/bin/bash
# Smoke test: 1 task, 2 episodes, 2 alphas, whole pipeline. Writes /workspace/SMOKE_DONE or /workspace/SMOKE_FAILED.
source ${VENV:-/root/venv_mw}/bin/activate
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH}
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /workspace/code
O=${SMOKE_OUT:-/workspace/out_smoke}
T=${SMOKE_TASK:-walker-walk}
A=0.1,0.3
M=${SMOKE_MARK:-SMOKE}; rm -f /workspace/${M}_DONE /workspace/${M}_FAILED
mkdir -p $O/data $O/results
python calib.py --tasks $T --episodes 2 --workers 2 --out $O && \
python prep.py --tasks $T --alphas $A --out $O && \
python grid.py --tasks $T --episodes 2 --alphas $A --workers ${WORKERS:-6} --out $O && \
python decision.py --tasks $T --alphas $A --max_states 40 --out $O && \
python analyze.py --tasks $T --alphas $A --out $O && touch /workspace/${M}_DONE && exit 0
touch /workspace/${M}_FAILED
