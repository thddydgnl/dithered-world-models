#!/bin/bash
# Phase-A sequence on the pod (before PLAN.md). Usage: seq_phaseA.sh checks_lat | pc_check | dryrun | after_pre
#   checks_lat  e17_checks paths, shift, bf16p (single process, no MPS daemon), then the latency smoke and the
#               latency quick run (run_lat.sh: refuses while a runner is alive, stops MPS; exclusive GPU)
#   pc_check    e17_checks pc (needs the pcstats results of the pre driver)
#   dryrun      analysis code paths on void data: analyze_e17.py on synthetic results (fake_e17.py) and
#               e17_lat_analyze.py on the latency quick run
#   after_pre   waits for the pre driver, then pc_check, check 3 (analyze_e17.py on the FP32 none -> out/analysis_pre)
#               and the throughput summary (bench_e17.py -> out/analysis_pre/throughput.json)
source /workspace/atm/code/env.sh
W=/workspace/atm
O=$W/out
C=$W/code
L=$W/logs
cd $C
case "$1" in
  checks_lat)
    echo smoke > $O/PHASE
    echo "=== $(date -u +%H:%M:%SZ) e17_checks paths,shift,bf16p"
    $PY $C/e17_checks.py --sections paths,shift,bf16p --out $O/checks > $L/checks.log 2>&1
    grep E17_CHECKS $L/checks.log
    echo "=== $(date -u +%H:%M:%SZ) latency smoke"
    bash $C/run_lat.sh smoke > $L/lat_smoke.log 2>&1
    grep -E "ALL_PASS|calib:|E17_LAT_SMOKE_DONE|Error|error" $L/lat_smoke.log | tail -5
    echo "=== $(date -u +%H:%M:%SZ) latency quick"
    bash $C/run_lat.sh quick > $L/lat_quick.log 2>&1
    tail -2 $L/lat_quick.log
    echo "=== $(date -u +%H:%M:%SZ) SEQ_CHECKS_LAT_DONE" ;;
  pc_check)
    $PY $C/e17_checks.py --sections pc --out $O/checks --res $O > $L/checks_pc.log 2>&1
    grep E17_CHECKS $L/checks_pc.log
    echo "=== $(date -u +%H:%M:%SZ) SEQ_PC_CHECK_DONE" ;;
  dryrun)
    D=$W/big/dryrun
    rm -rf $D && mkdir -p $D
    $PY $C/make_manifest_e17.py all --outdir $D/manifests > /dev/null
    $PY $C/fake_e17.py --manifest $D/manifests/main.jsonl --dst $D/fake
    $PY $C/analyze_e17.py --res $D/fake --e1 $C/inputs/e1_tdmpc2_baseline.json --outdir $D/fake_analysis \
      --fq_csv $D/fake/fq_fake.csv | tail -2
    $PY $C/analyze_e17.py --res $D/fake --e1 $C/inputs/e1_tdmpc2_baseline.json --outdir $D/fake_analysis_td \
      --fq_csv $D/fake/fq_fake_td.csv | tail -2
    cmp <(python3 -c "import json;d=json.load(open('$D/fake_analysis/e17_summary.json'));print(json.dumps([d['K2'],d['pc_descriptive']],sort_keys=True))") \
        <(python3 -c "import json;d=json.load(open('$D/fake_analysis_td/e17_summary.json'));print(json.dumps([d['K2'],d['pc_descriptive']],sort_keys=True))") \
      && echo "generic and td csv formats give the same K2 / pc comparison"
    $PY $C/analyze_e17.py --res $D/fake --e1 $C/inputs/e1_tdmpc2_baseline.json --outdir $D/fake_analysis_nofq | tail -2
    $PY $C/e17_lat_analyze.py --main $O/latency/quick --outdir $D/lat_quick_analysis --e8 $C/inputs/e8_cells.csv | tail -2
    ls -la $D/fake_analysis $D/lat_quick_analysis
    echo "=== $(date -u +%H:%M:%SZ) SEQ_DRYRUN_DONE" ;;
  after_pre)
    # wait for the pre driver to finish, then the per-channel check, check 3 on the FP32 none and the throughput
    while [ "$(cat $O/DRIVER_STATE 2>/dev/null)" != finished ] || [ "$(cat $O/DRIVER_MODE 2>/dev/null)" != pre ]; do
      sleep 30
    done
    echo "=== $(date -u +%H:%M:%SZ) pre driver finished; pc check"
    $PY $C/e17_checks.py --sections pc --out $O/checks --res $O > $L/checks_pc.log 2>&1
    grep E17_CHECKS $L/checks_pc.log
    echo "=== $(date -u +%H:%M:%SZ) check 3 (FP32 none vs E1) and throughput"
    $PY $C/analyze_e17.py --res $O --e1 $C/inputs/e1_tdmpc2_baseline.json --outdir $O/analysis_pre | tail -2
    $PY $C/bench_e17.py --res $O --workers 8 --out $O/analysis_pre/throughput.json > /dev/null
    echo "=== $(date -u +%H:%M:%SZ) SEQ_AFTER_PRE_DONE" ;;
  *) echo "usage: $0 checks_lat|pc_check|dryrun|after_pre"; exit 2 ;;
esac
