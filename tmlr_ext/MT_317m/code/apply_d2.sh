#!/bin/bash
# Apply deviation D2 (det+BC bias rule, DESIGN_tmlr 11) on this pod: stop the driver, move the bias results of the
# frozen 3-sweep rule out of out/jobs and out/bias (kept as a record in out/bias_frozen_rule/, same file names),
# then start the driver; it recomputes every bias job with code/inputs/bias_rule.json (= runpod_tmlr/bias_rule_D2.json).
A=/workspace/atm
grep -q '"stages"' $A/code/inputs/bias_rule.json || { echo "no D2 rule file"; exit 1; }
bash $A/code/restart.sh --stop || exit 1
mkdir -p $A/out/bias_frozen_rule/jobs $A/out/bias_frozen_rule/npz
for f in $A/out/jobs/bias/*.json.gz; do [ -f "$f" ] && mv "$f" $A/out/bias_frozen_rule/jobs/; done
for f in $A/out/bias/*.npz; do [ -f "$f" ] && mv "$f" $A/out/bias_frozen_rule/npz/; done
echo "moved to out/bias_frozen_rule: $(ls $A/out/bias_frozen_rule/jobs | wc -l) job files, $(ls $A/out/bias_frozen_rule/npz | wc -l) npz"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) D2 applied" >> $A/logs/driver.log
bash $A/code/restart.sh
