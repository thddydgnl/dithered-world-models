#!/bin/bash
# After the main run: post-hoc analysis, compressed raw energies, and a results bundle to pull.
W=/workspace/r25
cd $W
python3 code/posthoc.py main/prep main/run > logs/main_posthoc.log 2>&1
python3 - <<'PY'
import numpy as np
R = np.load("/workspace/r25/main/run/eval_raw.npz", allow_pickle=True)
np.savez_compressed("/workspace/r25/main/run/eval_raw_f32.npz", **{k: (R[k].astype(np.float32) if R[k].dtype == np.float64 else R[k]) for k in R.files})
PY
B=$W/bundle
rm -rf $B && mkdir -p $B/smoke $B/pod_logs
cp main/run/{summary.json,table_conditions.csv,table_flip_quartiles.csv,table_criteria.csv,posthoc.json,table_posthoc.csv,match.json,stats.json,eval_progress.json,eval_raw_f32.npz} $B/
cp main/run/fig_*.png $B/
cp main/prep/prep.json $B/prep_windows.json
cp main/massive.json $B/massive_coords.json
cp croptest/crop_test.json $B/crop_test.json
cp smoke/checks.json $B/smoke/checks.json
cp smoke/run/{summary.json,match.json,stats.json} $B/smoke/
cp results/vjepa2_commit.txt $B/
cp logs/*.log $B/pod_logs/
du -sh $B; ls $B
