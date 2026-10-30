#!/bin/bash
# PointMaze (DINO-WM) environment in user space, separate venv so the PushT/Wall env is never disturbed:
#   ~/pmx/envs/a-dinowm-pm = the a-dinowm packages + mujoco-py 2.1.2.14 (CPU/OSMesa builder) + d4rl (no deps).
# Reuses read-only the MuJoCo 2.1.0 binaries (~/pmx/envs/mujoco210) and the micromamba GL env with OSMesa
# (~/pmx/envs/micromamba/envs/gl24: conda-forge mesalib 24.3.4 + glew + libgl-devel) built by the TD-MPC2 worker
# (code/tdmpc2/setup_mw_server.sh). PointMaze dataset: OSF vr5gy (DINO-WM data), only the .pth metadata is unpacked
# (frames are rendered by the env, as for Wall). Idempotent; writes PM_ENV_DONE / PM_ENV_FAILED in the log.
set -x
UV=$HOME/.local/bin/uv
export UV_PYTHON_INSTALL_DIR=$HOME/pmx/envs/uv-python
V=$HOME/pmx/envs/a-dinowm-pm
MJ=$HOME/pmx/envs/mujoco210
GL=$HOME/pmx/envs/micromamba/envs/gl24
fail() { echo "PM_ENV_FAILED: $1"; exit 1; }
[ -f $MJ/bin/libmujoco210.so ] || fail mujoco210_missing
[ -f $GL/lib/libOSMesa.so.8 ] || [ -f $GL/lib/libOSMesa.so ] || fail osmesa_missing
# dataset (metadata only)
DL=$HOME/pmx/data/dinowm/pm_dl
if [ ! -d $HOME/pmx/data/dinowm/point_maze ]; then
  [ -f $DL/point_maze.zip ] || fail zip_missing
  (cd $HOME/pmx/data/dinowm && unzip -q $DL/point_maze.zip 'point_maze/*.pth' -x 'point_maze/obses/*' && mkdir -p point_maze/obses) || fail unzip
fi
ls -la $HOME/pmx/data/dinowm/point_maze
ln -sfn $HOME/pmx/data/dinowm/point_maze $HOME/pmx/A/dinowm_main/ds/point_maze
# venv
[ -x $V/bin/python ] || $UV venv -q --python 3.11 $V || fail venv
$UV pip install -q --python $V/bin/python torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124 || fail torch
$UV pip install -q --python $V/bin/python "setuptools==65.5.0" "wheel==0.38.0" "cython<3" patchelf "numpy<2" \
  "hydra-core==1.3.2" "gym==0.23.1" einops "pymunk==6.2.1" shapely opencv-python-headless pygame scikit-image \
  matplotlib scipy pandas tqdm accelerate decord h5py termcolor click || fail base
$UV pip install -q --python $V/bin/python --no-build-isolation "mujoco-py==2.1.2.14" || fail mujoco_py_pip
$UV pip install -q --python $V/bin/python --no-deps "d4rl @ git+https://github.com/Farama-Foundation/D4RL.git@89141a689b0353b0dac3da5cba60da4b1b16254d" || fail d4rl
# build + import (runtime env = run_pm.sh)
export PATH=$V/bin:$PATH MUJOCO_PY_MUJOCO_PATH=$MJ CPATH=$GL/include LIBRARY_PATH=$GL/lib   # patchelf from the venv
export LD_LIBRARY_PATH=$MJ/bin:$GL/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
$V/bin/python -c "import mujoco_py; print('mujoco_py ok', mujoco_py.__file__)" || fail mujoco_py_import
$V/bin/python -c "import d4rl.offline_env; print('d4rl ok')" || fail d4rl_import
$UV pip freeze --python $V/bin/python > $HOME/pmx/A/dinowm_main/pip_freeze_pm.txt
echo PM_ENV_DONE
