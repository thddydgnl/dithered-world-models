#!/bin/bash
# R2-7 main orchestration. Stage A: baseline CEM with population recording on held and eval splits.
# Stage B: taus from held baselines -> FV/AK on held -> selection. Stage C: eval of the chosen rule (if not ES).
# Replays of recorded populations run on CPU as populations become available.
cd /workspace/code
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=/workspace/logs/main7.log
F='warn\|gym\|migration\|torch.load\|pkg_resources\|next_location\|^\[\['
echo "$(date -u +%H:%M:%S) stage A start" >> $LOG
( python cem7.py wall held base 0 20 5 --record 2>&1 | grep --line-buffered -v -i "$F" > /workspace/logs/w_held_base.log
  python cem7.py wall eval base 0 50 5 --record 2>&1 | grep --line-buffered -v -i "$F" > /workspace/logs/w_eval_base.log
  echo "$(date -u +%H:%M:%S) wall base held+eval done" >> $LOG ) &
python cem7.py pusht held base 0 20 3 --record 2>&1 | grep --line-buffered -v -i "$F" > /workspace/logs/p_held_base.log &
PH=$!
python cem7.py pusht eval base 0 50 3 --record 2>&1 | grep --line-buffered -v -i "$F" > /workspace/logs/p_eval_base.log &
PE=$!
wait $PH
while pgrep -f "[c]em7.py wall held base" > /dev/null; do sleep 5; done
echo "$(date -u +%H:%M:%S) held baselines done" >> $LOG
python select7.py taus >> $LOG 2>&1
TW=$(python -c "import json;print(json.load(open('/workspace/results/taus7.json'))['wall']['0.5'])")
TP=$(python -c "import json;print(json.load(open('/workspace/results/taus7.json'))['pusht']['0.5'])")
echo "$(date -u +%H:%M:%S) stage B: tau_0.5 wall $TW pusht $TP" >> $LOG
( while pgrep -f "[c]em7.py wall eval base" > /dev/null; do sleep 5; done
  python cem7.py wall held fv 0 20 5 --tau $TW 2>&1 | grep --line-buffered -v -i "$F" > /workspace/logs/w_held_fv.log
  python cem7.py wall held ak 0 20 5 --tau $TW 2>&1 | grep --line-buffered -v -i "$F" > /workspace/logs/w_held_ak.log ) &
python cem7.py pusht held fv 0 20 3 --tau $TP 2>&1 | grep --line-buffered -v -i "$F" > /workspace/logs/p_held_fv.log &
python cem7.py pusht held ak 0 20 3 --tau $TP 2>&1 | grep --line-buffered -v -i "$F" > /workspace/logs/p_held_ak.log &
OMP_NUM_THREADS=1 python replay7.py wall held 3 >> $LOG 2>&1
OMP_NUM_THREADS=1 python replay7.py pusht held 3 >> $LOG 2>&1
while pgrep -f "[c]em7.py .* held (fv|ak)" > /dev/null; do sleep 10; done
echo "$(date -u +%H:%M:%S) held FV/AK done" >> $LOG
python select7.py select >> $LOG 2>&1
read KIND Q < /workspace/results/chosen7.txt
echo "$(date -u +%H:%M:%S) chosen $KIND $Q" >> $LOG
wait $PE
while pgrep -f "[c]em7.py wall eval base" > /dev/null; do sleep 5; done
echo "$(date -u +%H:%M:%S) eval baselines done" >> $LOG
if [ "$KIND" != "es" ]; then
  TW=$(python -c "import json;print(json.load(open('/workspace/results/taus7.json'))['wall']['$Q'])")
  TP=$(python -c "import json;print(json.load(open('/workspace/results/taus7.json'))['pusht']['$Q'])")
  python cem7.py wall eval $KIND 0 50 5 --tau $TW 2>&1 | grep --line-buffered -v -i "$F" > /workspace/logs/w_eval_chosen.log &
  python cem7.py pusht eval $KIND 0 25 3 --tau $TP 2>&1 | grep --line-buffered -v -i "$F" > /workspace/logs/p_eval_chosen_a.log &
  python cem7.py pusht eval $KIND 25 50 3 --tau $TP 2>&1 | grep --line-buffered -v -i "$F" > /workspace/logs/p_eval_chosen_b.log &
fi
OMP_NUM_THREADS=1 python replay7.py wall eval 3 >> $LOG 2>&1
OMP_NUM_THREADS=1 python replay7.py pusht eval 3 >> $LOG 2>&1
wait
echo "$(date -u +%H:%M:%S) MAIN7_DONE" >> $LOG
