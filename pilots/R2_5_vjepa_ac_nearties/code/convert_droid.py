"""Convert DROID-100 (RLDS) into per-episode npz files. Runs in the TF venv.

Copied from phaseB/code/convert_droid.py and trimmed to what R2-5 needs: frames of exterior_image_1_left at stride 4
of the 15 Hz stream (about 3.75 fps, the official V-JEPA 2-AC DROID loader uses fstp = ceil(15 / 4) = 4) and the
proprio state (cartesian_position 6 + gripper_position 1) at those frames.
Usage: python convert_droid.py <rlds_dir> <out_dir>
"""
import json
import sys

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

tf.config.set_visible_devices([], "GPU")
SRC, OUT = sys.argv[1], sys.argv[2]
STRIDE = 4

ds = tfds.builder_from_directory(SRC).as_dataset(split="train")
meta = []
for ei, ep in enumerate(ds):
    steps = list(ep["steps"].as_numpy_iterator())
    n = len(steps)
    obs = [s["observation"] for s in steps]
    idx = np.arange(0, n, STRIDE)
    cart = np.stack([obs[i]["cartesian_position"] for i in idx]).astype(np.float64)        # [n_sub, 6] xyz + euler xyz
    grip = np.stack([obs[i]["gripper_position"] for i in idx]).astype(np.float64)[:, 0]    # [n_sub]
    img1 = np.stack([obs[i]["exterior_image_1_left"] for i in idx])                         # [n_sub, 180, 320, 3] uint8
    np.savez(f"{OUT}/ep{ei:03d}.npz", cart=cart, grip=grip, idx=idx, img1=img1)
    meta.append(dict(ep=ei, n=n, n_sub=int(len(idx)), instr=steps[0]["language_instruction"].decode(),
                     path=ep["episode_metadata"]["file_path"].numpy().decode()))
    print(ei, n, len(idx), meta[-1]["instr"][:60], flush=True)
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
print("CONVERT_DONE", len(meta))
