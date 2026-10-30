#!/bin/bash
# Passive occupancy log of both GPUs for the throughput split (solo vs shared): every 20 s one line per GPU
#   <UTC time> gpu=<i> util=<%> mem=<MiB> n_apps=<compute apps> n_small=<apps < 1.5 GiB (e.g. MPS clients)> n_large=<apps >= 1.5 GiB>
# Host PIDs are not visible inside this container, so processes are classified by memory only.
# usage: nohup bash gpu_occupancy.sh >> ~/pmx/A/logs/E0_gpu_occupancy.log 2>&1 &
while true; do
  for i in 0 1; do
    u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i=$i '$1==i{print $2}')
    apps=$(nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v u="$u" '$1==u{print $2}')
    n=$(echo "$apps" | grep -c '[0-9]'); ns=$(echo "$apps" | awk '$1 != "" && $1 < 1536' | wc -l); nl=$(echo "$apps" | awk '$1 != "" && $1 >= 1536' | wc -l)
    g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i $i)
    echo "$(date -u +%FT%TZ) gpu=$i util=${g%%,*} mem=${g##*, } n_apps=$n n_small=$ns n_large=$nl"
  done
  sleep 20
done
