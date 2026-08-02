#!/usr/bin/env bash
#SBATCH --job-name=img-general-select
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
  sbatch --qos=juno-pri --array=0-14%8 scripts/juno_imagenet_general_selection.sbatch --mode smoke
  sbatch --qos=juno-pri --array=0-46%8 scripts/juno_imagenet_general_selection.sbatch --mode pilot
  sbatch --qos=juno-pri --array=0-4%4 scripts/juno_imagenet_general_selection.sbatch --mode semantic_correction
  bash scripts/juno_imagenet_general_selection.sbatch --mode smoke --dry-run

Runs source-only ImageNet10 experiments outside model repair:
  smoke  8 one-shot methods at ratio 0.1 and 7 active-learning methods for
         two rounds, seed 42, one epoch per trained model. Expected array 0-14.
  pilot  8 one-shot methods x 5 ratios and 7 six-round active-learning methods,
         seed 42, 30 epochs per trained model. Expected array 0-46.
  semantic_correction
         Corrected class-balanced Semantic CoreSet at five ratios, seed 42,
         30 epochs. Expected array 0-4.

Options:
  --mode smoke|pilot|semantic_correction
  --repo-dir PATH       Default: /work/$USER/safefix_rebuttal
  --python PATH         Default: /work/$USER/venvs/kitti/bin/python
  --data-dir PATH       Default: scratch ImageNet10 ImageFolder
  --proxy-model PATH    Full-data ImageNet10 checkpoint for one-shot model-based selectors
  --attribute-file PATH Qwen3-VL attribute metadata
  --output-root PATH    Durable result root under the Juno worktree
  --dry-run             Print the resolved task without running it
EOF
}

MODE="smoke"
REPO_DIR="/work/${USER}/safefix_rebuttal"
PYTHON_EXE="/work/${USER}/venvs/kitti/bin/python"
DATA_DIR="/home/${USER}/scratch/safefix_cache/data/imagenet_subset"
PROXY_MODEL=""
ATTRIBUTE_FILE=""
OUTPUT_ROOT=""
DRY_RUN=0

require_value() {
  [[ $# -ge 2 && -n "$2" ]] || { echo "Missing value for $1" >&2; exit 2; }
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) require_value "$@"; MODE="$2"; shift 2 ;;
    --repo-dir) require_value "$@"; REPO_DIR="$2"; shift 2 ;;
    --python) require_value "$@"; PYTHON_EXE="$2"; shift 2 ;;
    --data-dir) require_value "$@"; DATA_DIR="$2"; shift 2 ;;
    --proxy-model) require_value "$@"; PROXY_MODEL="$2"; shift 2 ;;
    --attribute-file) require_value "$@"; ATTRIBUTE_FILE="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$PROXY_MODEL" ]] || PROXY_MODEL="$REPO_DIR/save/models/model_imagenet_resnet_base.pth"
[[ -n "$ATTRIBUTE_FILE" ]] || ATTRIBUTE_FILE="$REPO_DIR/save/attribute10_qwen3_8B.json"
[[ -n "$OUTPUT_ROOT" ]] || OUTPUT_ROOT="$REPO_DIR/logs/juno/imagenet_general_selection"

DATA_METHODS=(
  random
  stratified
  entropy
  coreset
  badge
  attribute_rarity
  semantic_coreset
  attriselect
)
AL_METHODS=(
  random
  entropy
  coreset
  badge
  attribute_rarity
  object_attribute_entropy_reduction
  attriselect_al
)
RATIOS=(0.1 0.3 0.5 0.7 0.9)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED=42
case "$MODE" in
  smoke)
    (( TASK_ID >= 0 && TASK_ID < 15 )) || {
      echo "Smoke mode expects SLURM_ARRAY_TASK_ID in 0-14, got $TASK_ID" >&2
      exit 2
    }
    EPOCHS=1
    if (( TASK_ID < 8 )); then
      EXPERIMENT="data_selection"
      METHOD="${DATA_METHODS[$TASK_ID]}"
      RATIO=0.1
      ROUNDS=1
    else
      EXPERIMENT="active_learning"
      METHOD="${AL_METHODS[$((TASK_ID - 8))]}"
      RATIO=0.1
      ROUNDS=2
    fi
    ;;
  pilot)
    (( TASK_ID >= 0 && TASK_ID < 47 )) || {
      echo "Pilot mode expects SLURM_ARRAY_TASK_ID in 0-46, got $TASK_ID" >&2
      exit 2
    }
    EPOCHS=30
    if (( TASK_ID < 40 )); then
      EXPERIMENT="data_selection"
      METHOD_INDEX=$((TASK_ID / ${#RATIOS[@]}))
      RATIO_INDEX=$((TASK_ID % ${#RATIOS[@]}))
      METHOD="${DATA_METHODS[$METHOD_INDEX]}"
      RATIO="${RATIOS[$RATIO_INDEX]}"
      ROUNDS=1
    else
      EXPERIMENT="active_learning"
      METHOD="${AL_METHODS[$((TASK_ID - 40))]}"
      RATIO=0.1
      ROUNDS=6
    fi
    ;;
  semantic_correction)
    (( TASK_ID >= 0 && TASK_ID < 5 )) || {
      echo "Semantic-correction mode expects SLURM_ARRAY_TASK_ID in 0-4, got $TASK_ID" >&2
      exit 2
    }
    EXPERIMENT="data_selection"
    METHOD="semantic_coreset"
    RATIO="${RATIOS[$TASK_ID]}"
    ROUNDS=1
    EPOCHS=30
    ;;
  *)
    echo "--mode must be smoke, pilot, or semantic_correction, got: $MODE" >&2
    exit 2
    ;;
esac

RATIO_TAG="${RATIO//./p}"
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  JOB_KEY="${SLURM_ARRAY_JOB_ID}_${TASK_ID}"
else
  JOB_KEY="dryrun_${MODE}_${TASK_ID}"
fi
TASK_TAG="${EXPERIMENT}_${METHOD}_ratio${RATIO_TAG}_seed${SEED}"
TASK_DIR="$OUTPUT_ROOT/$MODE/$JOB_KEY"
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

if [[ "$EXPERIMENT" == "data_selection" && "$METHOD" =~ ^(entropy|coreset|badge)$ ]]; then
  CMD+=(--proxy-checkpoint "$PROXY_MODEL")
fi

print_resolved_task() {
  echo "mode=$MODE"
  echo "task_id=$TASK_ID"
  echo "experiment=$EXPERIMENT"
  echo "method=$METHOD"
  echo "ratio=$RATIO"
  echo "rounds=$ROUNDS"
  echo "epochs=$EPOCHS"
  echo "seed=$SEED"
  echo "repo_dir=$REPO_DIR"
  echo "python=$PYTHON_EXE"
  echo "data_dir=$DATA_DIR"
  echo "proxy_model=$PROXY_MODEL"
  echo "attribute_file=$ATTRIBUTE_FILE"
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
if [[ "$EXPERIMENT" == "data_selection" && "$METHOD" =~ ^(entropy|coreset|badge)$ ]]; then
  [[ -s "$PROXY_MODEL" ]] || { echo "Missing proxy model: $PROXY_MODEL" >&2; exit 1; }
fi
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
  [[ -f "$REPO_DIR/scripts/test_notification.py" ]] || return 0
  python3 "$REPO_DIR/scripts/test_notification.py" \
    "❌ ImageNet general-selection task $JOB_KEY failed with exit $exit_code: experiment=$EXPERIMENT method=$METHOD ratio=$RATIO seed=$SEED artifacts=$TASK_DIR" \
    || true
}
trap 'cleanup_and_notify $?' EXIT

echo "host=$(hostname)"
echo "slurm_job_id=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"${CMD[@]}"

[[ -s "$MANIFEST_FILE" ]] || { echo "Missing selection manifest: $MANIFEST_FILE" >&2; exit 1; }
[[ -s "$RESULT_FILE" ]] || { echo "Missing result JSON: $RESULT_FILE" >&2; exit 1; }

"$PYTHON_EXE" - "$RESULT_FILE" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    result = json.load(handle)
if result["experiment"] == "data_selection":
    summary = {
        "experiment": result["experiment"],
        "method": result["method"],
        "ratio": result["ratio"],
        "test_accuracy": result["metrics"]["test_accuracy"],
        "macro_attribute_accuracy": result["attribute_test_metrics"][
            "macro_attribute_accuracy_min_support"
        ],
    }
else:
    summary = {
        "experiment": result["experiment"],
        "method": result["method"],
        "rounds": result["rounds"],
        "test_accuracies": [row["metrics"]["test_accuracy"] for row in result["round_results"]],
        "labeled_ratios": [row["labeled_ratio"] for row in result["round_results"]],
    }
print("result_summary", json.dumps(summary, sort_keys=True))
PY

echo "completed task=$JOB_KEY experiment=$EXPERIMENT method=$METHOD ratio=$RATIO seed=$SEED"
