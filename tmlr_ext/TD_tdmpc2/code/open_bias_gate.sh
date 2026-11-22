#!/bin/bash
# Mac-side helper (td worker): write the coordinator's bias-vector rule to every live td pod and open the bias gate.
# The driver then runs the bias jobs (rule read at run time and stored in each result) and the det+BC / sh-iter+b /
# E13 jobs waiting behind them. Usage: open_bias_gate.sh '<rule json>' [tag ...]
#   e.g. open_bias_gate.sh '{"eta": 0.5, "until_below": 0.01, "max_sweeps": 40, "decided": "<who, when>"}'
N="/Users/song-yonghwi/Documents/physical ai - Neural Networks/runpod_tmlr"
RULE=$1
shift
python3 -c "import json,sys; r=json.loads(sys.argv[1]); assert 'eta' in r and ('sweeps' in r or 'until_below' in r)" "$RULE" || { echo "bad rule"; exit 1; }
TAGS=("$@")
[ ${#TAGS[@]} -eq 0 ] && TAGS=(td.1 td.2 td.3 td.4 td.5 td.6)
for T in "${TAGS[@]}"; do
  [ -f "$N/pods/$T.env" ] || { echo "skip $T (no pod)"; continue; }
  echo "$RULE" | perl -e 'alarm 60; exec @ARGV' bash "$N/rsh.sh" $T 'cat > /workspace/atm/bias_rule.json; python3 -c "import json; json.load(open(\"/workspace/atm/bias_rule.json\"))" && date -u +%Y-%m-%dT%H:%M:%SZ > /workspace/atm/GATE_bias && echo "$(cat /workspace/atm/TAG) gate open: $(cat /workspace/atm/bias_rule.json)"'
done
