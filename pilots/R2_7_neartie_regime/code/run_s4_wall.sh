#!/bin/bash
# Side job: Wall eval replay + S4 for Wall as soon as the Wall eval baselines exist.
cd /workspace/code
export OMP_NUM_THREADS=1
F='warn\|gym\|migration\|torch.load\|pkg_resources\|next_location\|^\[\['
while pgrep -f "[c]em7.py wall eval base" > /dev/null || [ $(grep -c "^wall_eval_base" /workspace/logs/w_eval_base.log) -lt 50 ]; do sleep 10; done
true
python s4_7.py wall 0.2 2>&1 | grep -v -i "$F" >> /workspace/logs/s4_wall.log
echo S4WALL_DONE >> /workspace/logs/s4_wall.log
