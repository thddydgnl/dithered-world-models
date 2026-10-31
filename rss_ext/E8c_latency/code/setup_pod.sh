#!/bin/bash
# E8 pod setup (runpod-torch-v240 template): uv venv /root/venv with torch 2.7.1 (PyPI cu126 wheel, bundled Triton) and
# the package pins of main/code/tdmpc2/setup_server.sh that E8 needs (tensordict 0.8.3, numpy 1.24.4, hydra-core 1.3.2,
# omegaconf 2.3.0, matplotlib), plus nvidia-ml-py for per-block GPU state; tdmpc2 at commit 332f19e.
# No simulation environments (E8 builds randomly initialized models without them).
# Writes /workspace/SETUP_DONE or /workspace/SETUP_FAILED.
set -x
export DEBIAN_FRONTEND=noninteractive
W=/workspace
mkdir -p $W/logs $W/results
rm -f $W/SETUP_DONE $W/SETUP_FAILED
fail() { echo "SETUP FAILED: $1"; touch $W/SETUP_FAILED; exit 1; }

apt-get update -qq && apt-get install -y -qq rsync > /dev/null || fail apt
curl -LsSf https://astral.sh/uv/install.sh | sh || fail uv
export PATH=$HOME/.local/bin:$PATH

cd /root
[ -d tdmpc2 ] || git clone -q https://github.com/nicklashansen/tdmpc2.git || fail clone
git -C tdmpc2 checkout -q 332f19e4aba4bffe2dc736f45ea7f7bdc585903a || fail checkout
git -C tdmpc2 log -1 --format='tdmpc2 %H %cd'

[ -x /root/venv/bin/python ] || uv venv --python 3.11 /root/venv || fail venv
uv pip install --python /root/venv/bin/python torch==2.7.1 tensordict==0.8.3 numpy==1.24.4 hydra-core==1.3.2 \
  omegaconf==2.3.0 termcolor matplotlib==3.10.9 nvidia-ml-py || \
  uv pip install --python /root/venv/bin/python torch==2.7.1 tensordict==0.8.3 numpy==1.24.4 hydra-core==1.3.2 \
  omegaconf==2.3.0 termcolor "matplotlib<3.11" nvidia-ml-py || fail pip
/root/venv/bin/python -c "import torch, triton, tensordict, numpy, pynvml; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), 'triton', triton.__version__, 'td', tensordict.__version__, 'numpy', numpy.__version__, torch.cuda.get_device_name(0))" || fail import
uv pip freeze --python /root/venv/bin/python > $W/results/freeze.txt
git -C /root/tdmpc2 rev-parse HEAD > $W/results/tdmpc2_commit.txt
nvidia-smi > $W/results/nvidia_smi_setup.txt
nproc > $W/results/nproc.txt
touch $W/SETUP_DONE
echo SETUP_ALL_DONE
