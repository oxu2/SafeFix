#!/usr/bin/env bash
#SBATCH --job-name=img-general-attrsrc
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch --qos=juno-pri --array=0-11%8 scripts/juno_imagenet_general_attribute_sources.sbatch
  bash scripts/juno_imagenet_general_attribute_sources.sbatch --dry-run

Runs the general ImageNet10 AttriSelect protocol with two alternative
attribute sources. For each source, tasks 0-4 run the five one-shot ratios and
task 5 runs the complete six-round active-learning experiment.

Options:
  --repo-dir PATH    Default: /work/$USER/safefix_rebuttal
  --python PATH      Default: /work/$USER/venvs/kitti/bin/python
  --data-dir PATH    Default: scratch ImageNet10 ImageFolder
  --output-root PATH Default: durable Juno log root
  --dry-run          Print the resolved task without running it
EOF
}

REPO_DIR="/work/${USER}/safefix_rebuttal"
PYTHON_EXE="/work/${USER}/venvs/kitti/bin/python"
DATA_DIR="/home/${USER}/scratch/safefix_cache/data/imagenet_subset"
OUTPUT_ROOT=""
DRY_RUN=0

require_value() {
  [[ $# -ge 2 && -n "$2" ]] || { echo "Missing value for $1" >&2; exit 2; }
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir) require_value "$@"; REPO_DIR="$2"; shift 2 ;;
    --python) require_value "$@"; PYTHON_EXE="$2"; shift 2 ;;
    --data-dir) require_value "$@"; DATA_DIR="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$OUTPUT_ROOT" ]] || OUTPUT_ROOT="$REPO_DIR/logs/juno/imagenet_general_attribute_sources"

SOURCE_NAMES=(blip internvl3)
SOURCE_FILES=(
  "$REPO_DIR/save/attribute10.json"
  "$REPO_DIR/save/attribute10_internvl3_8B.json"
)
RATIOS=(0.1 0.3 0.5 0.7 0.9)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
(( TASK_ID >= 0 && TASK_ID < 12 )) || {
  echo "Expected SLURM_ARRAY_TASK_ID in 0-11, got $TASK_ID" >&2
  exit 2
}

TASKS_PER_SOURCE=6
SOURCE_INDEX=$((TASK_ID / TASKS_PER_SOURCE))
WITHIN_SOURCE=$((TASK_ID % TASKS_PER_SOURCE))
SOURCE_NAME="${SOURCE_NAMES[$SOURCE_INDEX]}"
ATTRIBUTE_FILE="${SOURCE_FILES[$SOURCE_INDEX]}"
SEED=42
EPOCHS=30

if (( WITHIN_SOURCE < 5 )); then
  EXPERIMENT="data_selection"
  METHOD="attriselect"
  RATIO="${RATIOS[$WITHIN_SOURCE]}"
  ROUNDS=1
else
  EXPERIMENT="active_learning"
  METHOD="attriselect_al"
  RATIO=0.1
  ROUNDS=6
fi

RATIO_TAG="${RATIO//./p}"
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  JOB_KEY="${SLURM_ARRAY_JOB_ID}_${TASK_ID}"
else
  JOB_KEY="dryrun_${TASK_ID}"
fi
TASK_TAG="${SOURCE_NAME}_${EXPERIMENT}_${METHOD}_ratio${RATIO_TAG}"
TASK_DIR="$OUTPUT_ROOT/$JOB_KEY"
LOG_FILE="$TASK_DIR/${TASK_TAG}.log"
MANIFEST_FILE="$TASK_DIR/${TASK_TAG}_selection.json"
RESULT_FILE="$TASK_DIR/${TASK_TAG}_result.json"
NOTIFICATION_ENV_FILE="$REPO_DIR/.vscode/.env"

CMD=(
  env
  CUDA_VISIBLE_DEVICES=0
  "$PYTHON_EXE" -u "$REPO_DIR/scripts/run_imagenet_general_selection.py"
  --experiment "$EXPERIMENT"
  --method "$METHOD"
  --data-dir "$DATA_DIR"
  --attribute-file "$ATTRIBUTE_FILE"
  --attribute-fields color,background,texture
  --ratio "$RATIO"
  --rounds "$ROUNDS"
  --initial-ratio 0.1
  --query-ratio 0.1
  --epochs "$EPOCHS"
  --batch-size 64
  --workers 4
  --learning-rate 0.001
  --weight-decay 0.001
  --lr-step-size 3
  --lr-gamma 0.5
  --seed "$SEED"
  --gpu 0
  --result-json "$RESULT_FILE"
  --selection-manifest "$MANIFEST_FILE"
)

print_resolved_task() {
  echo "task_id=$TASK_ID"
  echo "attribute_source=$SOURCE_NAME"
  echo "attribute_file=$ATTRIBUTE_FILE"
  echo "experiment=$EXPERIMENT"
  echo "method=$METHOD"
  echo "ratio=$RATIO"
  echo "rounds=$ROUNDS"
  echo "epochs=$EPOCHS"
  echo "task_dir=$TASK_DIR"
  printf 'command='
  printf '%q ' "${CMD[@]}"
  printf '\n'
}

if [[ "$DRY_RUN" -eq 1 ]]; then
  print_resolved_task
  exit 0
fi

[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo "Submit with sbatch, or pass --dry-run." >&2
  exit 2
}

for required_file in \
  "$PYTHON_EXE" \
  "$REPO_DIR/scripts/run_imagenet_general_selection.py" \
  "$REPO_DIR/utils/imagenet_general_selection.py" \
  "$ATTRIBUTE_FILE"; do
  [[ -s "$required_file" ]] || { echo "Missing or empty file: $required_file" >&2; exit 1; }
done
[[ -d "$DATA_DIR" ]] || { echo "Missing data directory: $DATA_DIR" >&2; exit 1; }

mkdir -p "$TASK_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
print_resolved_task

if [[ -z "${NOTIFICATION_WEBHOOK:-}" && -f "$NOTIFICATION_ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$NOTIFICATION_ENV_FILE"
  set +a
fi

cleanup_and_notify() {
  local exit_code="$1"
  [[ "$exit_code" -ne 0 ]] || return 0
  [[ -n "${NOTIFICATION_WEBHOOK:-}" ]] || return 0
  python3 "$REPO_DIR/scripts/test_notification.py" \
    "❌ ImageNet alternative-attribute task $JOB_KEY failed with exit $exit_code: source=$SOURCE_NAME experiment=$EXPERIMENT method=$METHOD ratio=$RATIO artifacts=$TASK_DIR" \
    || true
}
trap 'cleanup_and_notify $?' EXIT

echo "host=$(hostname)"
echo "slurm_job_id=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"${CMD[@]}"

[[ -s "$MANIFEST_FILE" ]] || { echo "Missing manifest: $MANIFEST_FILE" >&2; exit 1; }
[[ -s "$RESULT_FILE" ]] || { echo "Missing result: $RESULT_FILE" >&2; exit 1; }
echo "completed task=$JOB_KEY source=$SOURCE_NAME experiment=$EXPERIMENT method=$METHOD ratio=$RATIO"
