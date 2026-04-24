#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_dir="log_pose/base_reference_old015544_top14"
mkdir -p "$out_dir"

ts="$(date +%Y%m%d_%H%M%S)"
logfile="${out_dir}/posthoc_pose_old015544_top14_compare_${ts}.log"
pidfile="${out_dir}/posthoc_pose_old015544_top14_compare_${ts}.pid"

setsid nohup /people/cs/o/oxx220000/.conda/envs/ad/bin/python \
  scripts/posthoc_pose_base_compare.py \
  --base-run-dir log_pose/pose_keypointrcnn_train20k_internvl3_bs4_e2_rare0p97_gpuX_20260412_015544 \
  --base-log-file log_pose/pose_train20k_internvl3_bs4_e2_rare0p97_gpuX_20260412_015544.log \
  --copy-base-log-dir "$out_dir" \
  --analysis-tag manual \
  --top-k 14 \
  > "$logfile" 2>&1 < /dev/null &

echo "$!" > "$pidfile"
echo "LOG=$logfile"
echo "PIDFILE=$pidfile"
cat "$pidfile"
