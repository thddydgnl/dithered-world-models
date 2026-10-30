#!/bin/bash
cd /workspace/code
export OMP_NUM_THREADS=2
F='warn\|gym\|migration\|torch.load\|pkg_resources\|next_location\|^\[\['
for T in wall pusht; do
  python cem7.py $T held base 0 2 2 --record --tag _smoke 2>&1 | grep -v -i "$F" | tail -3
  python cem7.py $T held fv 0 2 2 --tau 0.3 --tag _smoke 2>&1 | grep -v -i "$F" | tail -2
  python cem7.py $T held ak 0 2 2 --tau 0.3 --tag _smoke 2>&1 | grep -v -i "$F" | tail -2
  OMP_NUM_THREADS=1 python replay7.py $T held 6 2>&1 | grep -v -i "$F" | tail -2
done
ls -la /workspace/pops /workspace/results
echo SMOKE7_DONE
