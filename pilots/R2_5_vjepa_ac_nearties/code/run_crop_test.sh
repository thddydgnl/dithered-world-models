#!/bin/bash
# Model-competence check before the main run (created on the pod at 05:44Z; copied here): same 32 windows, two input crops.
export TORCH_HOME=/workspace/cache/torch OMP_NUM_THREADS=8
cd /workspace/r25
python3 code/prep.py /workspace/r25/croptest/square 30 2 square > logs/crop_prep_square.log 2>&1
python3 code/prep.py /workspace/r25/croptest/train 30 2 train > logs/crop_prep_train.log 2>&1
python3 code/crop_test.py /workspace/r25/croptest/square /workspace/r25/croptest/train /workspace/r25/croptest/crop_test.json > logs/crop_test.log 2>&1
echo DONE > /workspace/r25/croptest/DONE
