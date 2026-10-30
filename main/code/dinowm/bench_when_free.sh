#!/bin/bash
# Solo GPU-0 throughput benchmark (coordinator request): wait until GPU 0 carries no other worker's compute process
# (no MPS client of 100 MiB..1.5 GiB and at most one large process = my own queue runner) for 3 checks in a row,
# pause my own E0 queue (SIGSTOP; memory stays allocated, no GPU work), run manifests/E0_bench.json alone, resume
# (SIGCONT). Bench rows record their block times; ~/pmx/A/logs/E0_gpu0_occupancy.log documents the window.
# usage: nohup bash bench_when_free.sh >> ~/pmx/A/logs/E0_bench.log 2>&1 &
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0
U0=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==0{print $2}')
free_checks=0
while true; do
  mems=$(nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v u="$U0" '$1==u{print $2}')
  clients=$(echo "$mems" | awk '$1 != "" && $1 >= 100 && $1 < 1536' | wc -l)
  large=$(echo "$mems" | awk '$1 != "" && $1 >= 1536' | wc -l)
  if [ "$clients" -eq 0 ] && [ "$large" -le 1 ]; then free_checks=$((free_checks + 1)); else free_checks=0; fi
  if [ "$free_checks" -ge 3 ]; then break; fi
  sleep 20
done
mine=$(pgrep -u "$USER" -f "envs/a-dinowm(-pm)?/bin/python (runner|calib).py" | while read p; do
         grep -q "E0_bench" /proc/$p/cmdline 2>/dev/null || echo $p; done)
echo "BENCH $(date -u +%FT%TZ) GPU 0 free of other workers; pausing my queue pids: $mine"
[ -n "$mine" ] && kill -STOP $mine
bash run.sh runner.py manifests/E0_bench.json --task pusht --replay-workers 8
bash run.sh runner.py manifests/E0_bench.json --task wall --replay-workers 8
bash run_pm.sh runner.py manifests/E0_bench.json --task pointmaze --replay-workers 8
[ -n "$mine" ] && kill -CONT $mine
echo "BENCH $(date -u +%FT%TZ) done; resumed $mine"
