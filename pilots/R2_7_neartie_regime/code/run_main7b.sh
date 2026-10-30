#!/bin/bash
# Replaces the tail of run_main7.sh after the PushT held FV/AK runs died of GPU OOM (06:55Z).
# Same jobs and parameters; launched sequentially to fit in 44 GB.
cd /workspace/code
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=/workspace/logs/main7.log
F='warn\|gym\|migration\|torch.load\|pkg_resources\|next_location\|^\[\['
TW=$(python -c "import json;print(json.load(open('/workspace/results/taus7.json'))['wall']['0.5'])")
TP=$(python -c "import json;print(json.load(open('/workspace/results/taus7.json'))['pusht']['0.5'])")
echo "$(date -u +%H:%M:%S) run_main7b: restart held FV/AK (pusht OOM)" >> $LOG
while pgrep -f "[c]em7.py wall held fv" > /dev/null; do sleep 5; done
python cem7.py wall held ak 0 20 5 --tau $TW 2>&1 | grep --line-buffered -v -i "$F" > /workspace/logs/w_held_ak.log &
while pgrep -f "[s]4_7.py wall" > /dev/null; do sleep 5; done
rm -f /workspace/results/cem7_pusht_held_fv.jsonl /workspace/results/cem7_pusht_held_ak.jsonl
python cem7.py pusht held fv 0 20 3 --tau $TP 2>&1 | grep --line-buffered -v -i "$F" > /workspace/logs/p_held_fv.log &
while pgrep -f "[c]em7.py wall held ak" > /dev/null; do sleep 5; done
python cem7.py pusht held ak 0 20 3 --tau $TP 2>&1 | grep --line-buffered -v -i "$F" > /workspace/logs/p_held_ak.log &
sleep 30
while pgrep -f "[c]em7.py pusht held" > /dev/null; do sleep 10; done
echo "$(date -u +%H:%M:%S) held FV/AK done" >> $LOG
python select7.py select >> $LOG 2>&1
read KIND Q < /workspace/results/chosen7.txt
echo "$(date -u +%H:%M:%S) chosen $KIND $Q" >> $LOG
if [ "$KIND" != "es" ]; then
  TW=$(python -c "import json;print(json.load(open('/workspace/results/taus7.json'))['wall']['$Q'])")
  TP=$(python -c "import json;print(json.load(open('/workspace/results/taus7.json'))['pusht']['$Q'])")
  python cem7.py wall eval $KIND 0 50 5 --tau $TW 2>&1 | grep --line-buffered -v -i "$F" > /workspace/logs/w_eval_chosen.log &
  python cem7.py pusht eval $KIND 0 50 4 --tau $TP 2>&1 | grep --line-buffered -v -i "$F" > /workspace/logs/p_eval_chosen.log &
fi
while pgrep -f "[c]em7.py pusht eval base" > /dev/null; do sleep 10; done
echo "$(date -u +%H:%M:%S) eval baselines done" >> $LOG
OMP_NUM_THREADS=1 python replay7.py pusht eval 3 >> $LOG 2>&1
OMP_NUM_THREADS=1 python replay7.py wall eval 3 >> $LOG 2>&1
python s4_7.py pusht 0.2 2>&1 | grep -v -i "$F" > /workspace/logs/s4_pusht.log
wait
echo "$(date -u +%H:%M:%S) all CEM done" >> $LOG
python analyze7.py > /workspace/logs/analyze7.log 2>&1
echo "$(date -u +%H:%M:%S) MAIN7B_DONE" >> $LOG
