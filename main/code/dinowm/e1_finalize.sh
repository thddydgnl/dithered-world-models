#!/bin/bash
# E1 (DINO-WM) final analysis, run on the Mac after every E1 cell is done:
#   1. server: H2b auxiliary normalizer (none iteration-0 candidate cost std per task) -> results/E1/it0_cost_std.json
#   2. sync the E1 job JSONs (results/E1/E1_dinowm_wp, results/E1/E1_dinowm_pusht)
#   3. analyze_e1.py -> results/E1/e1_dinowm_summary.json, e1_dinowm_cells.csv; e1_tables.py -> e1_dinowm_tables.md
#   4. v5.2 (b1)/(b2)/(b3): sync results/drift_ref and results/mf_same (drift_ref.py, run on the server GPU), then
#      analyze_e1_v52.py -> results/E1/e1_dinowm_v52.json, e1_dinowm_v52_tables.md
# usage: bash e1_finalize.sh
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
MAIN="$(dirname "$(dirname "$HERE")")"          # .../A_dithered_world_models/main
R="$MAIN/results/E1"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
mkdir -p "$R"
$SSH kiis-mvf-gpu '~/pmx/envs/a-dinowm/bin/python ~/pmx/A/dinowm_main/code/it0_cost_std.py ~/pmx/A/dinowm_main/results/it0_cost_std.json'
rsync -a -e "$SSH" kiis-mvf-gpu:pmx/A/dinowm_main/results/it0_cost_std.json "$R/"
rsync -a -e "$SSH" kiis-mvf-gpu:pmx/A/dinowm_main/results/E1_dinowm_wp kiis-mvf-gpu:pmx/A/dinowm_main/results/E1_dinowm_pusht "$R/"
cd "$HERE"
python3 analyze_e1.py "$R" --errdir "$MAIN/results/E0_dinowm" --it0std "$R/it0_cost_std.json" \
    --out "$R/e1_dinowm_summary.json" --csv "$R/e1_dinowm_cells.csv" | tee "$R/e1_dinowm_summary.txt"
python3 e1_tables.py "$R/e1_dinowm_summary.json" "$R" --errdir "$MAIN/results/E0_dinowm" > "$R/e1_dinowm_tables.md"
rsync -a -e "$SSH" kiis-mvf-gpu:pmx/A/dinowm_main/results/drift_ref kiis-mvf-gpu:pmx/A/dinowm_main/results/mf_same "$R/" 2>/dev/null || true
python3 analyze_e1_v52.py "$R" --drift-ref "$R/drift_ref" --mf-same "$R/mf_same" --out "$R/e1_dinowm_v52.json" \
    --md "$R/e1_dinowm_v52_tables.md" > "$R/e1_dinowm_v52.txt"
echo "E1_FINALIZE done: $R"
