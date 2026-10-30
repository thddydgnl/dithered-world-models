#!/bin/bash
# Meta-World attempt (time-boxed): separate venv /root/venv_mw so the DMControl venv is never disturbed.
# Follows the commented "Gym" + "Meta-World" blocks of tdmpc2's docker/environment.yaml
# (cython<3, wheel 0.38.0, setuptools 65.5.0, gym 0.21.0, Metaworld@04be337, mujoco-py 2.1.2.14 if required).
# Writes /workspace/MW_DONE or /workspace/MW_FAILED.
set -x
export PATH=$HOME/.local/bin:$PATH DEBIAN_FRONTEND=noninteractive
W=/workspace
rm -f $W/MW_DONE $W/MW_FAILED
fail() { echo "MW FAILED: $1"; touch $W/MW_FAILED; exit 1; }
[ -x /root/venv_mw/bin/python ] || uv venv --python 3.11 /root/venv_mw || fail venv
source /root/venv_mw/bin/activate
uv pip install torch==2.7.1 tensordict==0.8.3 dm-control==1.0.16 mujoco==3.1.2 numpy==1.24.4 \
  gymnasium==0.29.1 hydra-core==1.3.2 omegaconf==2.3.0 termcolor tqdm pandas scipy matplotlib \
  "cython<3" setuptools==65.5.0 wheel==0.38.0 || fail base
# uv rejects gym 0.21.0's legacy metadata ("opencv-python>=3."); pip<24.1 (vendored packaging 21) still accepts it
uv pip install pip==23.3.1 || fail pip23
python -m pip install --no-build-isolation gym==0.21.0 || fail gym
uv pip install --no-build-isolation --no-deps \
  "metaworld @ git+https://github.com/Farama-Foundation/Metaworld.git@04be337a12305e393c0caf0cbf5ec7755c7c8feb" || fail metaworld
SP=$(python -c "import site; print(site.getsitepackages()[0])")
grep -rl "mujoco_py" $SP/metaworld | head -3
if grep -rqs "import mujoco_py" $SP/metaworld; then
  apt-get install -y -qq libosmesa6-dev libgl1-mesa-dev libglfw3 patchelf gcc > /dev/null || fail apt
  mkdir -p /root/.mujoco
  [ -d /root/.mujoco/mujoco210 ] || { curl -fsSL -o /tmp/mujoco210.tar.gz https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz && tar --no-same-owner -xzf /tmp/mujoco210.tar.gz -C /root/.mujoco; } || fail mujoco210
  export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH}
  uv pip install --no-build-isolation mujoco-py==2.1.2.14 || fail mujoco-py
  MUJOCO_GL=egl python -c "import mujoco_py; print('mujoco_py ok')" || fail mujoco_py_import
fi
cd /root/tdmpc2/tdmpc2
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH}
MUJOCO_GL=egl python - <<'PY' || fail envtest
import sys, numpy as np
sys.path.insert(0, '.')
from omegaconf import OmegaConf
from envs.metaworld import make_env
cfg = OmegaConf.create({'task': 'mw-push', 'obs': 'state', 'seed': 1})
env = make_env(cfg)
obs = env.reset()
tot = 0
for t in range(5):
    obs, r, done, info = env.step(np.zeros(env.action_space.shape, np.float32))
    tot += r
print('mw ok', obs.shape, tot, info.get('success'), env.max_episode_steps)
PY
touch $W/MW_DONE
echo MW_ALL_DONE
