#!/usr/bin/env bash
set -euo pipefail

cd /people/cs/o/oxx220000/safefix

mkdir -p logs
ts=$(date +%Y%m%d_%H%M%S)
log="logs/assign_attr_kitti_val_test_qwen3_gpu3_${ts}.log"
pidfile="logs/assign_attr_kitti_val_test_qwen3_gpu3_${ts}.pid"

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate qwen

setsid nohup bash scripts/run_kitti_attr_qwen3_gpu3_val_test_worker.sh > "${log}" 2>&1 < /dev/null &

pid=$!
echo "${pid}" > "${pidfile}"
echo "LOG=${log}"
echo "PIDFILE=${pidfile}"
echo "PID=${pid}"
