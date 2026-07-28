#!/usr/bin/env bash
#SBATCH --job-name=img-vocabsweep
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

REPO_DIR="/work/${USER}/safefix_rebuttal"
LAUNCHER="$REPO_DIR/scripts/juno_imagenet_vocabulary_sensitivity.sbatch"
NOTIFICATION_ENV_FILE="$REPO_DIR/.vscode/.env"
DRIVER_DIR="$REPO_DIR/logs/juno/imagenet_vocabulary_sensitivity/sweep/$SLURM_JOB_ID"
DRIVER_LOG="$DRIVER_DIR/driver.log"
RATIOS=(0.1 0.3 0.7 0.9)
ALL_VARIANTS=(0 1 2 3 4)
RATIO_ONE_VARIANTS=(0 3 4)

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Submit with sbatch." >&2; exit 2; }
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
    "$emoji ImageNet vocabulary ratio sweep $SLURM_JOB_ID $state: artifacts=$DRIVER_DIR" \
    || true
}
trap 'cleanup_and_notify $?' EXIT

run_variant() {
  local ratio="$1"
  local task_id="$2"
  local ratio_tag="${ratio//./p}"
  echo "starting ratio=$ratio task=$task_id at $(date --iso-8601=seconds)"
  VOCAB_RATIO="$ratio" \
  VOCAB_OUTPUT_SUFFIX="ratio${ratio_tag}" \
  SLURM_ARRAY_JOB_ID="$SLURM_JOB_ID" \
  SLURM_ARRAY_TASK_ID="$task_id" \
    bash "$LAUNCHER"
  echo "finished ratio=$ratio task=$task_id at $(date --iso-8601=seconds)"
}

echo "driver_job_id=$SLURM_JOB_ID"
echo "host=$(hostname)"
echo "ratios=0.1,0.3,0.7,0.9 with variants 0-4; ratio 1.0 with variants 0,3,4"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for ratio in "${RATIOS[@]}"; do
  for task_id in "${ALL_VARIANTS[@]}"; do
    run_variant "$ratio" "$task_id"
  done
done

for task_id in "${RATIO_ONE_VARIANTS[@]}"; do
  run_variant 1.0 "$task_id"
done

echo "completed 23-run vocabulary and slice ratio sweep"
