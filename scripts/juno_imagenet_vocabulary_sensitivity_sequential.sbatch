#!/usr/bin/env bash
#SBATCH --job-name=img-vocabseq
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

START_TASK=1
END_TASK=4
REPO_DIR="/work/${USER}/safefix_rebuttal"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start-task) START_TASK="$2"; shift 2 ;;
    --end-task) END_TASK="$2"; shift 2 ;;
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ "$START_TASK" =~ ^[0-9]+$ && "$END_TASK" =~ ^[0-9]+$ ]] || {
  echo "Task IDs must be non-negative integers." >&2
  exit 2
}
(( START_TASK >= 0 && END_TASK < 5 && START_TASK <= END_TASK )) || {
  echo "Expected 0 <= start-task <= end-task < 5; got $START_TASK..$END_TASK" >&2
  exit 2
}
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Submit with sbatch." >&2; exit 2; }

LAUNCHER="$REPO_DIR/scripts/juno_imagenet_vocabulary_sensitivity.sbatch"
NOTIFICATION_ENV_FILE="$REPO_DIR/.vscode/.env"
DRIVER_DIR="$REPO_DIR/logs/juno/imagenet_vocabulary_sensitivity/sequential/$SLURM_JOB_ID"
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
    "$emoji ImageNet vocabulary sequential job $SLURM_JOB_ID $state: tasks=$START_TASK-$END_TASK artifacts=$DRIVER_DIR" \
    || true
}
trap 'cleanup_and_notify $?' EXIT

echo "driver_job_id=$SLURM_JOB_ID"
echo "host=$(hostname)"
echo "task_range=$START_TASK-$END_TASK"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for task_id in $(seq "$START_TASK" "$END_TASK"); do
  echo "starting vocabulary task $task_id at $(date --iso-8601=seconds)"
  SLURM_ARRAY_JOB_ID="$SLURM_JOB_ID" \
  SLURM_ARRAY_TASK_ID="$task_id" \
    bash "$LAUNCHER"
  echo "finished vocabulary task $task_id at $(date --iso-8601=seconds)"
done

echo "completed sequential task range $START_TASK-$END_TASK"
