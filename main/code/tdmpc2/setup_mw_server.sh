#!/bin/bash
# E0 (paper A, TD-MPC2 part): Meta-World env in user space (approach 1 = the pre-planned route of design 6.0).
# Separate venv ~/pmx/envs/a-mw so the DMC venv is never disturbed. Follows the pilot setup_mw.sh
# (cython<3, wheel 0.38.0, setuptools 65.5.0, pip 23.3.1 for gym 0.21.0, Metaworld@04be337, mujoco-py 2.1.2.14) but
# replaces apt with:
#   - MuJoCo 2.1.0 binaries under ~/pmx/envs/mujoco210 (MUJOCO_PY_MUJOCO_PATH)
#   - micromamba (static binary) with root prefix ~/pmx/envs/micromamba and an env 'gl' holding conda-forge Mesa
#     (OSMesa + GL headers/libs) and GLEW; pip patchelf in the venv
# mujoco-py compiles its Cython extension at first import (CPU/OSMesa builder: no /usr/lib/nvidia on this host).
# Writes ~/pmx/A/tdmpc2_main/MW_DONE or MW_FAILED.
set -x
UV=$HOME/.local/bin/uv
export UV_PYTHON_INSTALL_DIR=$HOME/pmx/envs/python UV_PYTHON_PREFERENCE=only-managed
W=$HOME/pmx/A/tdmpc2_main
E=$HOME/pmx/envs
V=$E/a-mw
MJ=$E/mujoco210
MR=$E/micromamba
GL=$MR/envs/gl24   # mesalib 24.3.4: Mesa >= 25.1 (current conda-forge 26.2.5) no longer ships OSMesa
rm -f $W/MW_DONE $W/MW_FAILED
fail() { echo "MW FAILED: $1"; touch $W/MW_FAILED; exit 1; }

# 1. MuJoCo 2.1.0 binaries
if [ ! -f $MJ/bin/libmujoco210.so ]; then
  mkdir -p $E/dl_mujoco210 && rm -f $E/dl_mujoco210/*
  curl -fsSL -o $E/dl_mujoco210/mujoco210.tar.gz \
    https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz || fail mujoco210_download
  tar --no-same-owner -xzf $E/dl_mujoco210/mujoco210.tar.gz -C $E || fail mujoco210_extract
  rm -rf $E/dl_mujoco210
fi
ls $MJ/bin
ldd $MJ/bin/libmujoco210.so $MJ/bin/libglewosmesa.so | grep -E "not found|=>" | sort -u

# 2. micromamba + conda-forge Mesa (with OSMesa) / GLEW / libGL headers
mkdir -p $MR/bin
[ -x $MR/bin/micromamba ] || curl -fsSL -o $MR/bin/micromamba \
  https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64 || fail micromamba_download
chmod +x $MR/bin/micromamba
export MAMBA_ROOT_PREFIX=$MR
$MR/bin/micromamba --version || fail micromamba_run
# first attempt (13:28Z) used the latest mesalib (26.2.5): no libOSMesa / GL/osmesa.h -> osmesashim.c failed to compile
[ -d $GL/conda-meta ] || $MR/bin/micromamba create -y -p $GL -c conda-forge --override-channels "mesalib=24.3.4" glew \
  libgl-devel || fail mamba_gl
$MR/bin/micromamba list -p $GL | grep -E -i "mesa|glew|libgl|llvm|osmesa"
ls $GL/include/GL $GL/lib | grep -E -i "osmesa|libGL|glapi|glew|gl.h|^GL" | head -40

# 3. venv a-mw (pilot pins) + gym 0.21 + Metaworld + patchelf + mujoco-py
[ -x $V/bin/python ] || $UV venv --python 3.11 $V || fail venv
$UV pip install --python $V/bin/python torch==2.7.1 tensordict==0.8.3 dm-control==1.0.16 mujoco==3.1.2 numpy==1.24.4 \
  gymnasium==0.29.1 hydra-core==1.3.2 omegaconf==2.3.0 termcolor tqdm pandas==2.3.3 scipy==1.15.3 matplotlib==3.10.9 \
  "cython<3" setuptools==65.5.0 wheel==0.38.0 patchelf || fail base
# uv rejects gym 0.21.0's legacy metadata ("opencv-python>=3."); pip<24.1 (vendored packaging 21) still accepts it
$UV pip install --python $V/bin/python pip==23.3.1 || fail pip23
$V/bin/python -m pip install --no-build-isolation gym==0.21.0 || fail gym
$UV pip install --python $V/bin/python --no-build-isolation --no-deps \
  "metaworld @ git+https://github.com/Farama-Foundation/Metaworld.git@04be337a12305e393c0caf0cbf5ec7755c7c8feb" || fail metaworld
$UV pip install --python $V/bin/python --no-build-isolation mujoco-py==2.1.2.14 || fail mujoco-py
ls -la $V/bin/patchelf || fail patchelf

# 4. build + import (env of env_mw.sh)
export PATH=$V/bin:$PATH MUJOCO_PY_MUJOCO_PATH=$MJ MUJOCO_GL=disable
export CPATH=$GL/include LIBRARY_PATH=$GL/lib
export LD_LIBRARY_PATH=$MJ/bin:$GL/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
$V/bin/python -c "import mujoco_py; print('mujoco_py ok', mujoco_py.__file__)" || fail mujoco_py_import
SO=$(ls $V/lib/python3.11/site-packages/mujoco_py/generated/*.so | head -1)
ldd $SO | grep -E "not found|mujoco|OSMesa|GL|glew"
$UV pip freeze --python $V/bin/python > $W/freeze_a-mw.txt
touch $W/MW_DONE
echo MW_ALL_DONE
