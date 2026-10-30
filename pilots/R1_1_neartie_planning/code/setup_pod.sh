#!/bin/bash
# One-shot pod setup for R1-1 (r1n1): system libs, uv venv matching tdmpc2's docker/environment.yaml
# (torch 2.7.1, tensordict 0.8.3, dm-control 1.0.16, mujoco 3.1.2), the official tdmpc2 repo and the single-task
# 5M checkpoints (DMControl + Meta-World). Writes /workspace/SETUP_DONE or /workspace/SETUP_FAILED.
set -x
export DEBIAN_FRONTEND=noninteractive
W=/workspace
mkdir -p $W/logs $W/ckpt
rm -f $W/SETUP_DONE $W/SETUP_FAILED
fail() { echo "SETUP FAILED: $1"; touch $W/SETUP_FAILED; exit 1; }

apt-get update -qq && apt-get install -y -qq rsync libegl1 libgl1 libglew-dev libosmesa6 > /dev/null || fail apt
curl -LsSf https://astral.sh/uv/install.sh | sh || fail uv
export PATH=$HOME/.local/bin:$PATH

cd /root
[ -d tdmpc2 ] || git clone https://github.com/nicklashansen/tdmpc2.git || fail clone
git -C tdmpc2 log -1 --format='tdmpc2 %H %cd'

uv venv --python 3.11 /root/venv || fail venv
source /root/venv/bin/activate
uv pip install torch==2.7.1 tensordict==0.8.3 dm-control==1.0.16 mujoco==3.1.2 numpy==1.24.4 \
  gymnasium==0.29.1 hydra-core==1.3.2 omegaconf==2.3.0 termcolor tqdm pandas scipy matplotlib || fail pip
python -c "import torch, tensordict, dm_control, mujoco; print('torch', torch.__version__, torch.cuda.is_available(), 'td', tensordict.__version__, 'mujoco', mujoco.__version__)" || fail import

# single-task checkpoints from the official HF repo (retry on 429)
for T in dmcontrol/walker-walk dmcontrol/cheetah-run dmcontrol/finger-turn-hard dmcontrol/quadruped-walk \
         metaworld/mw-push metaworld/mw-pick-place metaworld/mw-door-open; do
  F=$W/ckpt/$(basename $T)-1.pt
  for i in 1 2 3 4 5 6; do
    [ -s "$F" ] && break
    curl -fsSL --retry 3 -o "$F.part" "https://huggingface.co/nicklashansen/tdmpc2/resolve/main/$T-1.pt" && mv "$F.part" "$F" && break
    sleep $((10 * i))
  done
  [ -s "$F" ] || echo "WARN: missing checkpoint $T"
done
ls -la $W/ckpt
touch $W/SETUP_DONE
echo SETUP_ALL_DONE
