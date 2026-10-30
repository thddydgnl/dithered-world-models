#!/bin/bash
# Pod setup for R2-13 (copied from R2-7 setup7.sh). Only the needed files are extracted: checkpoints pusht and
# wall_single, PushT without obs videos, Wall meta .pth only (unpacked wall_single is 59 GB). All frames used here are
# rendered by the simulator from dataset states. Downloads land in separate empty dirs.
set -u
mkdir -p /workspace/logs /workspace/results /workspace/pops /root/data/ckpt /root/data/pusht /root/data/wall /root/data/ds
export DEBIAN_FRONTEND=noninteractive
(apt-get update -qq && apt-get install -y -qq rsync unzip libgl1 libglib2.0-0 > /workspace/logs/apt.log 2>&1; echo APT_DONE >> /workspace/logs/apt.log) &
VO="view_only=a56a296ce3b24cceaf408383a175ce28"
waitunzip() { while [ ! -x /usr/bin/unzip ]; do sleep 2; done; }
(cd /root/data/ckpt && curl -sSL --retry 3 -o outputs.zip "https://osf.io/download/xvzs4/?$VO" && waitunzip && \
   unzip -q outputs.zip 'outputs/pusht/*' 'outputs/wall_single/*' && rm -f outputs.zip && echo CKPT_DONE) > /workspace/logs/dl_ckpt.log 2>&1 &
(cd /root/data/pusht && curl -sSL --retry 3 -o pusht_noise.zip "https://osf.io/download/k2d8w/?$VO" && waitunzip && \
   unzip -q pusht_noise.zip -x '*/obses/*' && rm -f pusht_noise.zip && echo PUSHT_DONE) > /workspace/logs/dl_pusht.log 2>&1 &
(cd /root/data/wall && curl -sSL --retry 3 -o wall_single.zip "https://osf.io/download/49rnx/?$VO" && waitunzip && \
   unzip -q wall_single.zip 'wall_single/*.pth' -x 'wall_single/obses/*' && mkdir -p wall_single/obses && rm -f wall_single.zip && \
   echo WALL_DONE) > /workspace/logs/dl_wall.log 2>&1 &
cd /root && [ -d dino_wm ] || git clone -q https://github.com/gaoyuezhou/dino_wm.git
cd /root/dino_wm && git log -1 --format='%H %cd' > /workspace/logs/dino_wm_commit.txt
pip install -q "hydra-core==1.3.2" "gym==0.23.1" einops "pymunk==6.2.1" shapely opencv-python-headless pygame \
  scikit-image matplotlib scipy pandas tqdm accelerate > /workspace/logs/pip.log 2>&1
echo "PIP_EXIT $?" >> /workspace/logs/pip.log
pip install -q decord >> /workspace/logs/pip.log 2>&1; echo "DECORD_EXIT $?" >> /workspace/logs/pip.log
wait
ln -sfn /root/data/pusht/pusht_noise /root/data/ds/pusht_noise
ln -sfn /root/data/wall/wall_single /root/data/ds/wall_single
python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')" > /workspace/logs/hub.log 2>&1
df -h / > /workspace/logs/df.txt
nproc > /workspace/logs/nproc.txt; cat /sys/fs/cgroup/cpu.max >> /workspace/logs/nproc.txt 2>/dev/null
echo SETUP_DONE
