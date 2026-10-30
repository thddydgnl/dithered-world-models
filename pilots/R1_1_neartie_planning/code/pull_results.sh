#!/bin/bash
# Pull the small result artifacts (JSON/CSV/PNG + logs) from the r1n1 pod. Calibration npz files stay on the pod.
ROOT="/Users/song-yonghwi/Documents/physical ai - Neural Networks"
D="$ROOT/A_dithered_world_models/pilots/R1_1_neartie_planning/results"
mkdir -p "$D/pod_logs" "$D/smoke"
bash "$ROOT/runpod_infra/rsh.sh" r1n1 'cd /workspace/out_main/results && tar -cf - *.json *.jsonl *.csv *.png 2>/dev/null' | tar -C "$D" -xf -
bash "$ROOT/runpod_infra/rsh.sh" r1n1 'cd /workspace/logs && tar -cf - *.log' | tar -C "$D/pod_logs" -xf -
bash "$ROOT/runpod_infra/rsh.sh" r1n1 'cd /workspace/out_smoke/results && tar -cf - *.json *.jsonl *.csv 2>/dev/null' | tar -C "$D/smoke" -xf -
ls -la "$D"
