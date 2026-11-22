source /workspace/atm/code/env.sh
cd /workspace/atm/code
for t in $*; do $PY -u check_codepath.py $t > /workspace/atm/logs/check1_$t.log 2>&1; echo "CHECK1_EXIT $t $?" >> /workspace/atm/logs/checks.log; done
