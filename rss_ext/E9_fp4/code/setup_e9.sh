#!/bin/bash
# E9 pod setup (one venv for Meta-World and DMC, as E1's a-mw venv): apt libraries, uv venv with the E1 pins
# (setup_server.sh / setup_mw_server.sh: torch 2.7.1 cu126, tensordict 0.8.3, dm-control 1.0.16, mujoco 3.1.2,
# numpy 1.24.4, gymnasium 0.29.1, hydra-core 1.3.2, omegaconf 2.3.0, pandas 2.3.3, scipy 1.15.3, matplotlib 3.10.9,
# cython<3, setuptools 65.5.0, wheel 0.38.0, pip 23.3.1, gym 0.21.0, Metaworld@04be337, mujoco-py 2.1.2.14 on
# MuJoCo 2.1.0 with the CPU/OSMesa builder like the server), tdmpc2 at 332f19e, the official 5M checkpoints of the
# 12 E1 tasks x seeds 1-3 (HF nicklashansen/tdmpc2) checked against E0's SHA256SUMS.
# Writes /workspace/SETUP_DONE or /workspace/SETUP_FAILED.
set -x
export DEBIAN_FRONTEND=noninteractive PATH=$HOME/.local/bin:$PATH
W=/workspace
C=/root/e9/code
mkdir -p $W/logs $W/results /root/ckpt
rm -f $W/SETUP_DONE $W/SETUP_FAILED
fail() { echo "SETUP FAILED: $1"; touch $W/SETUP_FAILED; exit 1; }
T0=$(date +%s)

apt-get update -qq && apt-get install -y -qq rsync libegl1 libgl1 libglew-dev libosmesa6 libosmesa6-dev \
  libgl1-mesa-dev libglfw3 patchelf gcc > /dev/null || fail apt
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh || fail uv
uv venv --python 3.11 /root/venv || fail venv
source /root/venv/bin/activate
uv pip install torch==2.7.1 tensordict==0.8.3 dm-control==1.0.16 mujoco==3.1.2 numpy==1.24.4 gymnasium==0.29.1 \
  hydra-core==1.3.2 omegaconf==2.3.0 termcolor tqdm pandas==2.3.3 scipy==1.15.3 matplotlib==3.10.9 \
  "cython<3" setuptools==65.5.0 wheel==0.38.0 patchelf || fail base
# uv rejects gym 0.21.0's legacy metadata; pip<24.1 accepts it (E1 setup_mw_server.sh)
uv pip install pip==23.3.1 || fail pip23
python -m pip install -q --no-build-isolation gym==0.21.0 || fail gym
uv pip install --no-build-isolation --no-deps \
  "metaworld @ git+https://github.com/Farama-Foundation/Metaworld.git@04be337a12305e393c0caf0cbf5ec7755c7c8feb" || fail metaworld
mkdir -p /root/.mujoco
[ -d /root/.mujoco/mujoco210 ] || { curl -fsSL -o /tmp/mujoco210.tar.gz \
  https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz && \
  tar --no-same-owner -xzf /tmp/mujoco210.tar.gz -C /root/.mujoco; } || fail mujoco210
source $C/env_e9.sh
uv pip install --no-build-isolation mujoco-py==2.1.2.14 || fail mujoco-py
python -c "import mujoco_py; print('mujoco_py ok', mujoco_py.__file__)" || fail mujoco_py_import
python -c "import torch, tensordict, dm_control, mujoco, numpy, triton; print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), 'td', tensordict.__version__, 'mujoco', mujoco.__version__, 'numpy', numpy.__version__, 'triton', triton.__version__)" || fail import
uv pip freeze > $W/results/freeze_e9_$(hostname).txt

cd /root
[ -d tdmpc2 ] || git clone -q https://github.com/nicklashansen/tdmpc2.git || fail clone
git -C tdmpc2 checkout -q 332f19e4aba4bffe2dc736f45ea7f7bdc585903a || fail checkout
git -C tdmpc2 log -1 --format='tdmpc2 %H %cd'

for T in metaworld/mw-push metaworld/mw-pick-place metaworld/mw-door-open metaworld/mw-drawer-open \
         metaworld/mw-button-press-topdown metaworld/mw-peg-insert-side metaworld/mw-window-open metaworld/mw-sweep-into \
         dmcontrol/cheetah-run dmcontrol/walker-walk dmcontrol/quadruped-walk dmcontrol/finger-turn-hard; do
  for S in 1 2 3; do
    F=/root/ckpt/$(basename $T)-$S.pt
    for i in 1 2 3 4 5 6; do
      [ -s "$F" ] && break
      curl -fsSL --retry 3 -o "$F.part" "https://huggingface.co/nicklashansen/tdmpc2/resolve/main/$T-$S.pt" && mv "$F.part" "$F" && break
      sleep $((10 * i))
    done
    [ -s "$F" ] || fail "checkpoint $T-$S"
  done
done
(cd /root/ckpt && sha256sum -c $C/inputs/ckpt_SHA256SUMS) || fail sha256
echo "setup secs $(( $(date +%s) - T0 ))"
touch $W/SETUP_DONE
echo SETUP_ALL_DONE
