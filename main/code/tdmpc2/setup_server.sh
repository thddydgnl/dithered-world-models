#!/bin/bash
# E0 (paper A, TD-MPC2 part): user-space environment on kiis-mvf-gpu (no root, no conda, no GL libs).
# Recreates the pilot setup_pod.sh (R2-8b) without apt:
#   - uv-managed CPython 3.11 under ~/pmx/envs/python (never the interpreters linked from other home folders)
#   - venv ~/pmx/envs/a-tdmpc2 with the pilot pins (torch 2.7.1 cu126 wheel, tensordict 0.8.3, dm-control 1.0.16,
#     mujoco 3.1.2, numpy 1.24.4, gymnasium 0.29.1, hydra-core 1.3.2, omegaconf 2.3.0, ...)
#   - tdmpc2 repo at the pilot commit 332f19e (R2-8b setup.log)
#   - official single-task 5M checkpoints, seeds 1-3, for the 12 candidate tasks (HF nicklashansen/tdmpc2)
# Writes ~/pmx/A/tdmpc2_main/SETUP_DONE or SETUP_FAILED. DMC runs with MUJOCO_GL=disable (no renderer).
set -x
UV=$HOME/.local/bin/uv
export UV_PYTHON_INSTALL_DIR=$HOME/pmx/envs/python UV_PYTHON_PREFERENCE=only-managed
W=$HOME/pmx/A/tdmpc2_main
V=$HOME/pmx/envs/a-tdmpc2
TDMPC2_COMMIT=332f19e4aba4bffe2dc736f45ea7f7bdc585903a
mkdir -p $W/ckpt $W/out $HOME/pmx/A/logs
rm -f $W/SETUP_DONE $W/SETUP_FAILED
fail() { echo "SETUP FAILED: $1"; touch $W/SETUP_FAILED; exit 1; }

$UV python install 3.11 || fail python
[ -x $V/bin/python ] || $UV venv --python 3.11 $V || fail venv
$UV pip install --python $V/bin/python torch==2.7.1 tensordict==0.8.3 dm-control==1.0.16 mujoco==3.1.2 numpy==1.24.4 \
  gymnasium==0.29.1 hydra-core==1.3.2 omegaconf==2.3.0 termcolor tqdm pandas==2.3.3 scipy==1.15.3 matplotlib==3.10.9 \
  || fail pip
MUJOCO_GL=disable $V/bin/python -c "import torch, tensordict, dm_control, mujoco, numpy; print('torch', torch.__version__, torch.cuda.is_available(), 'td', tensordict.__version__, 'mujoco', mujoco.__version__, 'numpy', numpy.__version__)" || fail import
$UV pip freeze --python $V/bin/python > $W/freeze_a-tdmpc2.txt

cd $W
[ -d tdmpc2 ] || git clone https://github.com/nicklashansen/tdmpc2.git || fail clone
git -C tdmpc2 checkout -q $TDMPC2_COMMIT || fail checkout
git -C tdmpc2 log -1 --format='tdmpc2 %H %cd'

# single-task checkpoints (retry on 429); file name <task>-<seed>.pt
for T in metaworld/mw-push metaworld/mw-pick-place metaworld/mw-door-open metaworld/mw-drawer-open \
         metaworld/mw-button-press-topdown metaworld/mw-peg-insert-side metaworld/mw-window-open metaworld/mw-sweep-into \
         dmcontrol/cheetah-run dmcontrol/walker-walk dmcontrol/quadruped-walk dmcontrol/finger-turn-hard; do
  for S in 1 2 3; do
    F=$W/ckpt/$(basename $T)-$S.pt
    for i in 1 2 3 4 5 6; do
      [ -s "$F" ] && break
      curl -fsSL --retry 3 -o "$F.part" "https://huggingface.co/nicklashansen/tdmpc2/resolve/main/$T-$S.pt" && mv "$F.part" "$F" && break
      sleep $((10 * i))
    done
    [ -s "$F" ] || echo "WARN: missing checkpoint $T-$S"
  done
done
ls -la $W/ckpt
(cd $W/ckpt && sha256sum *.pt > SHA256SUMS)
touch $W/SETUP_DONE
echo SETUP_ALL_DONE
