#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mkdir -p log_pose/base_reference

ts="$(date +%Y%m%d_%H%M%S)"
logfile="log_pose/base_reference/posthoc_pose_base_compare_${ts}.log"
pidfile="log_pose/base_reference/posthoc_pose_base_compare_${ts}.pid"

setsid nohup /people/cs/o/oxx220000/.conda/envs/ad/bin/python \
  scripts/posthoc_pose_base_compare.py \
  > "$logfile" 2>&1 < /dev/null &

echo "$!" > "$pidfile"
echo "LOG=$logfile"
echo "PIDFILE=$pidfile"
cat "$pidfile"
