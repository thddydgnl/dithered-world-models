#!/bin/bash
# R2-5 pod setup (runpod-torch-v240: torch 2.4, CUDA 12.4, Python 3.11).
# Adapted from the earlier phaseB setup (05_contact_gates_physical_ai/feasibility/phaseB/code/setup_pod.sh):
# TF/TFDS live in a separate venv, the vjepa2 hub URL is patched, cv2 is installed up front (phaseB pitfall),
# and the 11.8 GB AC checkpoint is fetched with aria2c (16 connections) straight into the torch hub cache.
set -x
export DEBIAN_FRONTEND=noninteractive
W=/workspace
mkdir -p $W/r25/code $W/r25/data $W/r25/results $W/r25/logs $W/cache/torch/hub/checkpoints $W/droid_100/1.0.0
apt-get update -qq && apt-get install -y -qq rsync git aria2 jq > /dev/null

# AC checkpoint in the background (the hub loader finds it in the cache and skips its own download)
( cd $W/cache/torch/hub/checkpoints && \
  aria2c -q -x 16 -s 16 -k 64M --file-allocation=none -o vjepa2-ac-vitg.pt https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt \
  && echo CKPT_OK > $W/r25/CKPT_OK ) > $W/r25/logs/ckpt.log 2>&1 &

# TF venv for RLDS conversion, also in the background
( python3 -m venv $W/tfenv && $W/tfenv/bin/pip install -q tensorflow-cpu tensorflow-datasets && echo TF_OK > $W/r25/TF_OK ) \
  > $W/r25/logs/tf.log 2>&1 &

pip install -q timm einops scipy matplotlib pandas "opencv-python-headless<4.11" 2>&1 | tail -2

# official V-JEPA 2 repo; its hub code may point the weights at localhost:8300, use the official host instead
cd $W && [ -d vjepa2 ] || git clone --depth 1 https://github.com/facebookresearch/vjepa2.git
sed -i 's|^VJEPA_BASE_URL = "http://localhost:8300"|VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"|' $W/vjepa2/src/hub/backbones.py
grep -n "^VJEPA_BASE_URL" $W/vjepa2/src/hub/backbones.py
(cd $W/vjepa2 && git log -1 --format='%H %cd' > $W/r25/results/vjepa2_commit.txt)

# DROID-100 (public Google Research bucket, RLDS, ~2.1 GB)
cd $W/droid_100/1.0.0
for f in dataset_info.json features.json; do curl -sf -O https://storage.googleapis.com/gresearch/robotics/droid_100/1.0.0/$f; done
for i in $(seq -f "%05g" 0 30); do echo "r2d2_faceblur-train.tfrecord-$i-of-00031"; done \
  | xargs -P 8 -I{} curl -sf --retry 5 -O https://storage.googleapis.com/gresearch/robotics/droid_100/1.0.0/{}
ls | wc -l
du -sh .

wait
ls -la $W/cache/torch/hub/checkpoints/
cat $W/r25/logs/ckpt.log | tail -5
[ -f $W/r25/CKPT_OK ] || { echo "aria2c failed; torch.hub will download single-stream"; rm -f $W/cache/torch/hub/checkpoints/vjepa2-ac-vitg.pt*; }
[ -f $W/r25/TF_OK ] || echo "TF venv FAILED"
echo SETUP_DONE > $W/r25/SETUP_DONE
