#!/bin/bash
# User-space setup of the DINO-WM environment on the university server (paper A, E0/E1, GPU 0).
# Adapted from the pilot setup13.sh (RunPod, root + apt + system pip) to uv: no sudo, apt, conda or Docker.
# Idempotent: re-running skips finished steps. Data are NOT downloaded here (shared, read-only):
#   checkpoints  ~/pmx/data/dinowm/outputs/{pusht,wall_single,point_maze}
#   Wall meta    ~/pmx/data/dinowm/wall/wall_single/*.pth (obses/ empty: Wall frames are rendered by the env)
#   PushT        ~/pmx/data/pusht_noise/{train,val} (symlink into another project's data, read-only)
set -u
ROOT=$HOME/pmx/A/dinowm_main
ENV=$HOME/pmx/envs/a-dinowm
UV=$HOME/.local/bin/uv
export UV_PYTHON_INSTALL_DIR=$HOME/pmx/envs/uv-python      # keep the managed interpreter inside ~/pmx
export TORCH_HOME=$HOME/pmx/torch HF_HOME=$HOME/pmx/hf
PILOT_COMMIT=0a9492fa12044b852ae9e001cc74604b79c8bb0c        # dino_wm commit used by R2-1/R2-7/R2-13
mkdir -p "$ROOT"/{logs,results,pops,ds} "$TORCH_HOME" "$HF_HOME"
cd "$ROOT"
if [ ! -d dino_wm ]; then
  git clone -q https://github.com/gaoyuezhou/dino_wm.git
  git -C dino_wm log -1 --format='%H %cd' > dino_wm_head_at_clone.txt
fi
git -C dino_wm checkout -q "$PILOT_COMMIT"
git -C dino_wm log -1 --format='%H %cd' > dino_wm_commit.txt
# dataset dir layout expected by the dino_wm hydra configs: $DATASET_DIR/pusht_noise, $DATASET_DIR/wall_single
ln -sfn "$HOME/pmx/data/pusht_noise" ds/pusht_noise
ln -sfn "$HOME/pmx/data/dinowm/wall/wall_single" ds/wall_single
"$UV" python install 3.11
[ -x "$ENV/bin/python" ] || "$UV" venv -q --python 3.11 "$ENV"
PY="$ENV/bin/python"
"$UV" pip install -q --python "$PY" torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
echo "TORCH_EXIT $?"
# setuptools: triton 3.0 (torch.compile of the quantizer kernels) imports it; uv venvs do not ship it
"$UV" pip install -q --python "$PY" "setuptools<71" "numpy<2" "hydra-core==1.3.2" "gym==0.23.1" einops "pymunk==6.2.1" shapely \
  opencv-python-headless pygame scikit-image matplotlib scipy pandas tqdm accelerate decord
echo "PIP_EXIT $?"
# DINOv2 ViT-S/14 via torch hub (cached under $TORCH_HOME/hub)
"$PY" -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14'); print('HUB_OK', torch.hub.get_dir())" 2>&1 | grep -v Warn
"$UV" pip freeze --python "$PY" > pip_freeze.txt
"$PY" - << 'PYEOF' > versions.txt 2>&1
import sys, torch, numpy, gym, pymunk, hydra, cv2, pygame, shapely, scipy
print('python', sys.version.split()[0]); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version())
print('numpy', numpy.__version__, 'gym', gym.__version__, 'pymunk', pymunk.version, 'hydra', hydra.__version__)
print('opencv', cv2.__version__, 'pygame', pygame.version.ver, 'shapely', shapely.__version__, 'scipy', scipy.__version__)
PYEOF
cat versions.txt
df -h ~ | tail -1
echo SETUP_DONE
