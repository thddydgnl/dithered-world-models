#!/bin/bash
# Passive GPU-0 occupancy log for the throughput split (solo vs shared): every 20 s one line
#   <UTC time> util=<GPU0 util %> mem=<GPU0 used MiB> n_apps=<compute apps on GPU 0> n_small=<apps < 1.5 GiB (MPS
#   server/clients of the TD-MPC2 worker)> n_large=<apps >= 1.5 GiB (DINO-WM runner or other)>
# Host PIDs are not visible inside this container, so processes are classified by memory only.
# usage: nohup bash gpu0_occupancy.sh >> ~/pmx/A/logs/E0_gpu0_occupancy.log 2>&1 &
U0=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==0{print $2}')
while true; do
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v u="$U0" '$1==u{print $2}')
  n=$(echo "$apps" | grep -c '[0-9]'); ns=$(echo "$apps" | awk '$1 != "" && $1 < 1536' | wc -l); nl=$(echo "$apps" | awk '$1 != "" && $1 >= 1536' | wc -l)
  g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i 0)
  echo "$(date -u +%FT%TZ) util=${g%%,*} mem=${g##*, } n_apps=$n n_small=$ns n_large=$nl"
  sleep 20
done
