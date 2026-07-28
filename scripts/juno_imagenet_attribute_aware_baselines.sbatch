#!/usr/bin/env bash
#SBATCH --job-name=img-attrbase
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch --array=0-3%2 scripts/juno_imagenet_attribute_aware_baselines.sbatch --mode smoke
  sbatch --array=0-19%2 scripts/juno_imagenet_attribute_aware_baselines.sbatch --mode full
  sbatch --array=0-59%2 scripts/juno_imagenet_attribute_aware_baselines.sbatch --mode remaining
  bash scripts/juno_imagenet_attribute_aware_baselines.sbatch --mode smoke --dry-run

Runs the ImageNet10 same-information comparison requested in the rebuttal:
  Target-Slice + Stratified, Attribute-Filtered Random,
  Attribute-Filtered Entropy, and AttriSelect.

Modes:
  smoke  Four methods at ratio 0.1, seed 42, one repair epoch.
         Expected array: 0-3.
  full   Four methods x five ratios at seed 42, 15 repair epochs.
         Expected array: 0-19.
  remaining
         Three attribute-aware baselines x five ratios x the four remaining
         seeds (0, 1234, 3407, 2024), 15 repair epochs.
         Expected array: 0-59.

Options:
  --mode smoke|full|remaining
                           Default: smoke
  --repo-dir PATH         Default: /work/$USER/safefix_rebuttal
  --python PATH           Default: /work/$USER/venvs/kitti/bin/python
  --data-dir PATH         ImageNet10 source ImageFolder root in scratch
  --generated-dir PATH    Fixed pink-color/fabric-texture generated pool
  --base-model PATH       Diagnosis-stage ResNet-18 checkpoint
  --attribute-file PATH   Qwen3-VL source attribute metadata
  --output-root PATH      Durable log/result root under the repo
  --dry-run               Print the resolved task and command without training
EOF
}

MODE="smoke"
REPO_DIR="/work/${USER}/safefix_rebuttal"
PYTHON_EXE="/work/${USER}/venvs/kitti/bin/python"
DATA_DIR="/home/${USER}/scratch/safefix_cache/data/imagenet_subset"
GENERATED_DIR="/home/${USER}/scratch/safefix_cache/data/imagenet_pinkcolor_fabrictexture"
BASE_MODEL=""
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
    --generated-dir) require_value "$@"; GENERATED_DIR="$2"; shift 2 ;;
    --base-model) require_value "$@"; BASE_MODEL="$2"; shift 2 ;;
    --attribute-file) require_value "$@"; ATTRIBUTE_FILE="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$BASE_MODEL" ]] || BASE_MODEL="$REPO_DIR/save/models/model_imagenet_resnet_base.pth"
[[ -n "$ATTRIBUTE_FILE" ]] || ATTRIBUTE_FILE="$REPO_DIR/save/attribute10_qwen3_8B.json"
[[ -n "$OUTPUT_ROOT" ]] || OUTPUT_ROOT="$REPO_DIR/logs/juno/imagenet_attribute_aware"

METHODS=(
  target_slice_stratified
  attribute_filtered_random
  attribute_filtered_entropy
  attribute
)
BASELINE_METHODS=(
  target_slice_stratified
  attribute_filtered_random
  attribute_filtered_entropy
)
RATIOS=(0.1 0.3 0.5 0.7 0.9)
REMAINING_SEEDS=(0 1234 3407 2024)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
case "$MODE" in
  smoke)
    (( TASK_ID >= 0 && TASK_ID < 4 )) || {
      echo "Smoke mode expects SLURM_ARRAY_TASK_ID in 0-3, got $TASK_ID" >&2
      exit 2
    }
    METHOD="${METHODS[$TASK_ID]}"
    RATIO="0.1"
    SEED="42"
    EPOCHS=1
    ;;
  full)
    (( TASK_ID >= 0 && TASK_ID < 20 )) || {
      echo "Full mode expects SLURM_ARRAY_TASK_ID in 0-19, got $TASK_ID" >&2
      exit 2
    }
    METHOD_INDEX=$((TASK_ID / 5))
    RATIO_INDEX=$((TASK_ID % 5))
    METHOD="${METHODS[$METHOD_INDEX]}"
    RATIO="${RATIOS[$RATIO_INDEX]}"
    SEED="42"
    EPOCHS=15
    ;;
  remaining)
    (( TASK_ID >= 0 && TASK_ID < 60 )) || {
      echo "Remaining mode expects SLURM_ARRAY_TASK_ID in 0-59, got $TASK_ID" >&2
      exit 2
    }
    TASKS_PER_METHOD=$(( ${#REMAINING_SEEDS[@]} * ${#RATIOS[@]} ))
    METHOD_INDEX=$((TASK_ID / TASKS_PER_METHOD))
    WITHIN_METHOD=$((TASK_ID % TASKS_PER_METHOD))
    SEED_INDEX=$((WITHIN_METHOD / ${#RATIOS[@]}))
    RATIO_INDEX=$((WITHIN_METHOD % ${#RATIOS[@]}))
    METHOD="${BASELINE_METHODS[$METHOD_INDEX]}"
    RATIO="${RATIOS[$RATIO_INDEX]}"
    SEED="${REMAINING_SEEDS[$SEED_INDEX]}"
    EPOCHS=15
    ;;
  *)
    echo "--mode must be smoke, full, or remaining, got: $MODE" >&2
    exit 2
    ;;
esac

RATIO_TAG="${RATIO//./p}"
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  JOB_KEY="${SLURM_ARRAY_JOB_ID}_${TASK_ID}"
else
  JOB_KEY="dryrun_${MODE}_${TASK_ID}"
fi
TASK_TAG="${METHOD}_ratio${RATIO_TAG}_seed${SEED}_genseed42"
TASK_DIR="$OUTPUT_ROOT/$MODE/$JOB_KEY"
LOG_FILE="$TASK_DIR/${TASK_TAG}.log"
MANIFEST_FILE="$TASK_DIR/${TASK_TAG}_selection.json"
RESULT_FILE="$TASK_DIR/${TASK_TAG}_result.json"
NOTIFICATION_ENV_FILE="$REPO_DIR/.vscode/.env"

STRATEGY_ARGS=(--al_strategy "$METHOD")
if [[ "$METHOD" == "attribute" ]]; then
  STRATEGY_ARGS+=(
    --al_attribute_rarity_power 0.03
    --al_attribute_frequency_penalty 0.5
    --al_attribute_class_balance 15
    --al_conditional_rarity True
    --al_attribute_use_entropy False
  )
fi

CMD=(
  env
  TRANSFORMERS_OFFLINE=1
  HF_HUB_OFFLINE=1
  MPLCONFIGDIR="$TASK_DIR/matplotlib"
  CUDA_VISIBLE_DEVICES=0
  "$PYTHON_EXE" -u "$REPO_DIR/alpred.py"
  --dataset imagenet
  --model resnet
  --data_dir "$DATA_DIR"
  --gen_image_dir "$GENERATED_DIR"
  --num_added_img 1000
  --generated_seed 42
  --attributes pinkcolor_fabrictexture
  --resume_model "$BASE_MODEL"
  --attribute_file "$ATTRIBUTE_FILE"
  --al_attribute_focus color,background,texture
  --num_epochs "$EPOCHS"
  --lr 1e-5
  --batch_size 64
  --workers 4
  --sub_ratio "$RATIO"
  --seed "$SEED"
  --gpu 0
  --use_active_learning True
  --skip_feasible True
  --freeze_backbone False
  --save_pred_model best
  --save_preds False
  --save_model False
  --save_report False
  --verbose True
  --selection_manifest "$MANIFEST_FILE"
  --result_json "$RESULT_FILE"
  "${STRATEGY_ARGS[@]}"
)

print_resolved_task() {
  echo "mode=$MODE"
  echo "task_id=$TASK_ID"
  echo "method=$METHOD"
  echo "ratio=$RATIO"
  echo "seed=$SEED"
  echo "generated_seed=42"
  echo "epochs=$EPOCHS"
  echo "repo_dir=$REPO_DIR"
  echo "python=$PYTHON_EXE"
  echo "data_dir=$DATA_DIR"
  echo "generated_dir=$GENERATED_DIR"
  echo "base_model=$BASE_MODEL"
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

for required_file in "$PYTHON_EXE" "$REPO_DIR/alpred.py" "$BASE_MODEL" "$ATTRIBUTE_FILE"; do
  [[ -s "$required_file" ]] || { echo "Missing or empty file: $required_file" >&2; exit 1; }
done
for required_dir in "$DATA_DIR" "$GENERATED_DIR"; do
  [[ -d "$required_dir" ]] || { echo "Missing directory: $required_dir" >&2; exit 1; }
done

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
  [[ -n "${NOTIFICATION_WEBHOOK:-}" ]] || return 0
  [[ -f "$REPO_DIR/scripts/test_notification.py" ]] || return 0
  local state="failed"
  local emoji="❌"
  if [[ "$exit_code" -eq 0 && -s "$RESULT_FILE" ]]; then
    state="completed"
    emoji="✅"
  fi
  python3 "$REPO_DIR/scripts/test_notification.py" \
    "$emoji ImageNet attribute-aware task $JOB_KEY $state: method=$METHOD ratio=$RATIO seed=$SEED artifacts=$TASK_DIR" \
    || true
}
trap 'cleanup_and_notify $?' EXIT

echo "host=$(hostname)"
echo "slurm_job_id=$SLURM_JOB_ID"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-set by command}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"${CMD[@]}"

[[ -s "$MANIFEST_FILE" ]] || { echo "Missing selection manifest: $MANIFEST_FILE" >&2; exit 1; }
[[ -s "$RESULT_FILE" ]] || { echo "Missing result JSON: $RESULT_FILE" >&2; exit 1; }

"$PYTHON_EXE" - "$RESULT_FILE" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    result = json.load(handle)
metrics = result.get("test_slice_metrics") or {}
print("result_summary", json.dumps({
    "strategy": result.get("strategy"),
    "ratio": result.get("source_ratio"),
    "seed": result.get("source_seed"),
    "overall": (metrics.get("overall") or {}).get("accuracy"),
    "target": (metrics.get("target") or {}).get("accuracy"),
    "target_count": (metrics.get("target") or {}).get("count"),
    "non_target": (metrics.get("non_target") or {}).get("accuracy"),
    "selection_seconds": result.get("selection_time_seconds"),
    "training_seconds": result.get("training_time_seconds"),
}, sort_keys=True))
PY

echo "completed task=$JOB_KEY method=$METHOD ratio=$RATIO seed=$SEED"
