#!/bin/bash
# E10 (paper A, RSS extension): one-shot RunPod setup for the mt80-317M experiment.
# Follows the pilot setup_pod.sh / setup_mw.sh (apt available on RunPod) with the E1 package pins
# (main/code/tdmpc2/setup_server.sh, setup_mw_server.sh):
#   - one uv venv /root/venv: torch 2.7.1 (cu126), tensordict 0.8.3, dm-control 1.0.16, mujoco 3.1.2, numpy 1.24.4,
#     gymnasium 0.29.1, hydra-core 1.3.2, omegaconf 2.3.0, scipy 1.15.3, pandas 2.3.3, matplotlib 3.10.9
#     + Meta-World: cython<3, setuptools 65.5.0, wheel 0.38.0, pip 23.3.1, gym 0.21.0, Metaworld@04be337,
#       mujoco-py 2.1.2.14 on MuJoCo 2.1.0 (CPU/OSMesa builder, as on the E1 server; physics does not depend on it)
#   - tdmpc2 at commit 332f19e
#   - the official multi-task checkpoint mt80-317M (HF nicklashansen/tdmpc2, multitask/mt80-317M.pt), sha256 checked
#     against the HF LFS oid
# Writes /workspace/SETUP_DONE or /workspace/SETUP_FAILED.
set -x
export DEBIAN_FRONTEND=noninteractive
W=/workspace
V=/root/venv
TDMPC2_COMMIT=332f19e4aba4bffe2dc736f45ea7f7bdc585903a
CKPT_URL=https://huggingface.co/nicklashansen/tdmpc2/resolve/main/multitask/mt80-317M.pt
CKPT_SHA=38efd60ba341bff89282898201483e38c81030a6b716f152de7f82f5567b99cc
mkdir -p $W/logs $W/ckpt $W/e10
rm -f $W/SETUP_DONE $W/SETUP_FAILED
fail() { echo "SETUP FAILED: $1"; touch $W/SETUP_FAILED; exit 1; }

# checkpoint (2.0 GB) in the background while the environment installs
(
  F=$W/ckpt/mt80-317M.pt
  for i in 1 2 3 4 5 6; do
    [ -s "$F" ] && break
    curl -fsSL --retry 3 -o "$F.part" "$CKPT_URL" && mv "$F.part" "$F" && break
    sleep $((10 * i))
  done
  ls -la $W/ckpt
  (cd $W/ckpt && sha256sum mt80-317M.pt > SHA256SUMS)
) > $W/logs/ckpt_download.log 2>&1 &
DL=$!

apt-get update -qq && apt-get install -y -qq rsync libegl1 libgl1 libglew-dev libosmesa6 libosmesa6-dev \
  libgl1-mesa-dev libglfw3 patchelf gcc > /dev/null || fail apt
curl -LsSf https://astral.sh/uv/install.sh | sh || fail uv
export PATH=$HOME/.local/bin:$PATH

cd /root
[ -d tdmpc2 ] || git clone -q https://github.com/nicklashansen/tdmpc2.git || fail clone
git -C tdmpc2 checkout -q $TDMPC2_COMMIT || fail checkout
git -C tdmpc2 log -1 --format='tdmpc2 %H %cd'

[ -x $V/bin/python ] || uv venv --python 3.11 $V || fail venv
source $V/bin/activate
uv pip install torch==2.7.1 tensordict==0.8.3 dm-control==1.0.16 mujoco==3.1.2 numpy==1.24.4 \
  gymnasium==0.29.1 hydra-core==1.3.2 omegaconf==2.3.0 termcolor tqdm pandas==2.3.3 scipy==1.15.3 matplotlib==3.10.9 \
  "cython<3" setuptools==65.5.0 wheel==0.38.0 || fail pip
# uv rejects gym 0.21.0's legacy metadata ("opencv-python>=3."); pip<24.1 (vendored packaging 21) still accepts it
uv pip install pip==23.3.1 || fail pip23
python -m pip install -q --no-build-isolation gym==0.21.0 || fail gym
uv pip install --no-build-isolation --no-deps \
  "metaworld @ git+https://github.com/Farama-Foundation/Metaworld.git@04be337a12305e393c0caf0cbf5ec7755c7c8feb" || fail metaworld
mkdir -p /root/.mujoco
[ -d /root/.mujoco/mujoco210 ] || { curl -fsSL -o /tmp/mujoco210.tar.gz \
  https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz && \
  tar --no-same-owner -xzf /tmp/mujoco210.tar.gz -C /root/.mujoco; } || fail mujoco210
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH}
export MUJOCO_PY_FORCE_CPU=1 MUJOCO_GL=disable
uv pip install --no-build-isolation mujoco-py==2.1.2.14 || fail mujoco-py
python -c "import mujoco_py; print('mujoco_py ok', mujoco_py.__file__)" || fail mujoco_py_import
python -c "import torch, tensordict, dm_control, mujoco, numpy, gym, metaworld; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), 'td', tensordict.__version__, 'mujoco', mujoco.__version__, 'numpy', numpy.__version__, 'gym', gym.__version__)" || fail import
uv pip freeze > $W/e10/freeze_venv.txt
wait $DL
cat $W/logs/ckpt_download.log
cat $W/ckpt/SHA256SUMS
grep -q $CKPT_SHA $W/ckpt/SHA256SUMS || fail ckpt_sha256
touch $W/SETUP_DONE
echo SETUP_ALL_DONE
