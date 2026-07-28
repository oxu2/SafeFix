#!/usr/bin/env bash
#SBATCH --job-name=img-attrseq
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch scripts/juno_imagenet_attribute_aware_sequential.sbatch \
    --start-task 3 --end-task 59

Runs a contiguous range of the 60-task ImageNet10 remaining-seed mapping
inside one H200 allocation. Each child task uses the `remaining` mode in
scripts/juno_imagenet_attribute_aware_baselines.sbatch, so the allocation is
retained between ratios, seeds, and methods.

Options:
  --start-task N    First remaining-mode task ID, inclusive. Default: 0
  --end-task N      Last remaining-mode task ID, inclusive. Default: 59
  --repo-dir PATH   Default: /work/$USER/safefix_rebuttal
EOF
}

START_TASK=0
END_TASK=59
REPO_DIR="/work/${USER}/safefix_rebuttal"

require_value() {
  [[ $# -ge 2 && -n "$2" ]] || { echo "Missing value for $1" >&2; exit 2; }
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start-task) require_value "$@"; START_TASK="$2"; shift 2 ;;
    --end-task) require_value "$@"; END_TASK="$2"; shift 2 ;;
    --repo-dir) require_value "$@"; REPO_DIR="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$START_TASK" =~ ^[0-9]+$ && "$END_TASK" =~ ^[0-9]+$ ]] || {
  echo "Task IDs must be non-negative integers." >&2
  exit 2
}
(( START_TASK >= 0 && END_TASK < 60 && START_TASK <= END_TASK )) || {
  echo "Expected 0 <= start-task <= end-task < 60; got $START_TASK..$END_TASK" >&2
  exit 2
}
[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo "Submit this driver with sbatch." >&2
  exit 2
}

LAUNCHER="$REPO_DIR/scripts/juno_imagenet_attribute_aware_baselines.sbatch"
NOTIFICATION_ENV_FILE="$REPO_DIR/.vscode/.env"
OUTPUT_ROOT="$REPO_DIR/logs/juno/imagenet_attribute_aware"
DRIVER_DIR="$OUTPUT_ROOT/sequential/$SLURM_JOB_ID"
DRIVER_LOG="$DRIVER_DIR/driver.log"

[[ -s "$LAUNCHER" ]] || { echo "Missing launcher: $LAUNCHER" >&2; exit 1; }
mkdir -p "$DRIVER_DIR"
exec > >(tee -a "$DRIVER_LOG") 2>&1

if [[ -z "${NOTIFICATION_WEBHOOK:-}" && -f "$NOTIFICATION_ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$NOTIFICATION_ENV_FILE"
  set +a
fi

cleanup_and_notify() {
  local exit_code="$1"
  [[ -n "${NOTIFICATION_WEBHOOK:-}" ]] || return 0
  [[ -f "$REPO_DIR/scripts/test_notification.py" ]] || return 0
  local state="failed"
  local emoji="❌"
  if [[ "$exit_code" -eq 0 ]]; then
    state="completed"
    emoji="✅"
  fi
  python3 "$REPO_DIR/scripts/test_notification.py" \
    "$emoji ImageNet sequential remaining-seed job $SLURM_JOB_ID $state: tasks=$START_TASK-$END_TASK artifacts=$DRIVER_DIR" \
    || true
}
trap 'cleanup_and_notify $?' EXIT

echo "driver_job_id=$SLURM_JOB_ID"
echo "host=$(hostname)"
echo "task_range=$START_TASK-$END_TASK"
echo "launcher=$LAUNCHER"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for task_id in $(seq "$START_TASK" "$END_TASK"); do
  echo "starting remaining task $task_id at $(date --iso-8601=seconds)"
  SLURM_ARRAY_JOB_ID="$SLURM_JOB_ID" \
  SLURM_ARRAY_TASK_ID="$task_id" \
    bash "$LAUNCHER" --mode remaining --repo-dir "$REPO_DIR"
  echo "finished remaining task $task_id at $(date --iso-8601=seconds)"
done

echo "completed sequential task range $START_TASK-$END_TASK"
