#!/usr/bin/env bash
#SBATCH --job-name=img-fr-epochs
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

usage() {
  cat <<'EOF'
Usage:
  sbatch scripts/juno_imagenet_fr_epoch_multiplier.sbatch --mode smoke
  sbatch --array=0-4%2 scripts/juno_imagenet_fr_epoch_multiplier.sbatch --mode pilot
  bash scripts/juno_imagenet_fr_epoch_multiplier.sbatch --mode pilot --task-id 0 --dry-run

Runs the ImageNet10 single-seed repair-epoch study at source ratio 0.9.

Modes:
  smoke  AttriSelect, seed 42, two epochs, milestones 1 and 2.
  pilot  Random, Entropy, CoreSet, BADGE, and AttriSelect, seed 42,
         60 epochs, milestones 15, 30, 45, and 60.
EOF
}

MODE="smoke"
TASK_ID_OVERRIDE=""
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
    --task-id) require_value "$@"; TASK_ID_OVERRIDE="$2"; shift 2 ;;
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
[[ -n "$OUTPUT_ROOT" ]] || OUTPUT_ROOT="$REPO_DIR/logs/juno/imagenet_fr_epoch_multiplier"

METHODS=(random entropy coreset badge attribute)
TASK_ID="${TASK_ID_OVERRIDE:-${SLURM_ARRAY_TASK_ID:-0}}"

case "$MODE" in
  smoke)
    [[ "$TASK_ID" -eq 0 ]] || {
      echo "Smoke mode expects task ID 0, got $TASK_ID" >&2
      exit 2
    }
    METHOD="attribute"
    NUM_EPOCHS=2
    MILESTONES="1,2"
    ;;
  pilot)
    (( TASK_ID >= 0 && TASK_ID < ${#METHODS[@]} )) || {
      echo "Pilot mode expects task ID in 0-4, got $TASK_ID" >&2
      exit 2
    }
    METHOD="${METHODS[$TASK_ID]}"
    NUM_EPOCHS=60
    MILESTONES="15,30,45,60"
    ;;
  *)
    echo "--mode must be smoke or pilot, got: $MODE" >&2
    exit 2
    ;;
esac

SOURCE_RATIO="0.9"
SOURCE_SEED=42
GENERATED_SEED=42
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  JOB_KEY="${SLURM_ARRAY_JOB_ID}_${TASK_ID}"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  JOB_KEY="${SLURM_JOB_ID}"
else
  JOB_KEY="dryrun_${MODE}_${TASK_ID}"
fi

TASK_TAG="${METHOD}_ratio0p9_seed42_e${NUM_EPOCHS}"
TASK_DIR="$OUTPUT_ROOT/$MODE/$JOB_KEY"
LOG_FILE="$TASK_DIR/${TASK_TAG}.log"
MANIFEST_FILE="$TASK_DIR/${TASK_TAG}_selection.json"
RESULT_FILE="$TASK_DIR/${TASK_TAG}_result.json"
NOTIFICATION_ENV_FILE="$REPO_DIR/.vscode/.env"

STRATEGY_ARGS=(--al_strategy "$METHOD")
case "$METHOD" in
  random)
    ;;
  entropy)
    ;;
  coreset|badge)
    STRATEGY_ARGS+=(--al_initial_labeled 400 --al_embedding_dim 256)
    ;;
  attribute)
    STRATEGY_ARGS+=(
      --al_attribute_rarity_power 0.03
      --al_attribute_frequency_penalty 0.5
      --al_attribute_class_balance 15
      --al_conditional_rarity True
      --al_attribute_use_entropy False
    )
    ;;
esac

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
  --generated_seed "$GENERATED_SEED"
  --attributes pinkcolor_fabrictexture
  --resume_model "$BASE_MODEL"
  --attribute_file "$ATTRIBUTE_FILE"
  --al_attribute_focus color,background,texture
  --num_epochs "$NUM_EPOCHS"
  --evaluation_milestones "$MILESTONES"
  --lr 1e-5
  --batch_size 64
  --workers 4
  --sub_ratio "$SOURCE_RATIO"
  --seed "$SOURCE_SEED"
  --gpu 0
  --use_active_learning True
  --skip_feasible True
  --freeze_backbone False
  --save_pred_model best
  --save_preds False
  --save_model False
  --save_report False
  --verbose False
  --selection_manifest "$MANIFEST_FILE"
  --result_json "$RESULT_FILE"
  "${STRATEGY_ARGS[@]}"
)

print_resolved_task() {
  echo "mode=$MODE"
  echo "task_id=$TASK_ID"
  echo "method=$METHOD"
  echo "source_ratio=$SOURCE_RATIO"
  echo "source_seed=$SOURCE_SEED"
  echo "generated_seed=$GENERATED_SEED"
  echo "num_epochs=$NUM_EPOCHS"
  echo "evaluation_milestones=$MILESTONES"
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
    "$emoji ImageNet FR epoch task $JOB_KEY $state: mode=$MODE method=$METHOD artifacts=$TASK_DIR" \
    || true
}
trap 'cleanup_and_notify $?' EXIT

echo "host=$(hostname)"
echo "slurm_job_id=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"${CMD[@]}"

[[ -s "$MANIFEST_FILE" ]] || { echo "Missing selection manifest: $MANIFEST_FILE" >&2; exit 1; }
[[ -s "$RESULT_FILE" ]] || { echo "Missing result JSON: $RESULT_FILE" >&2; exit 1; }

"$PYTHON_EXE" - "$RESULT_FILE" "$MILESTONES" <<'PY'
import json
import sys

result_path, expected_text = sys.argv[1:]
expected = [int(value) for value in expected_text.split(",")]
with open(result_path, "r", encoding="utf-8") as handle:
    result = json.load(handle)
records = result.get("milestone_results") or []
actual = [int(record["epoch"]) for record in records]
if actual != expected:
    raise SystemExit(f"Milestone mismatch: expected={expected}, actual={actual}")
for record in records:
    metrics = record.get("test_slice_metrics") or {}
    if int((metrics.get("overall") or {}).get("count", -1)) != 1300:
        raise SystemExit(f"Unexpected overall test count at epoch {record['epoch']}")
    if int((metrics.get("target") or {}).get("count", -1)) != 16:
        raise SystemExit(f"Unexpected target test count at epoch {record['epoch']}")
    if int((metrics.get("non_target") or {}).get("count", -1)) != 1284:
        raise SystemExit(f"Unexpected non-target test count at epoch {record['epoch']}")
print(json.dumps({
    "strategy": result.get("strategy"),
    "milestones": [
        {
            "epoch": record["epoch"],
            "best_epoch": record["best_epoch"],
            "test_accuracy": record["test_accuracy"],
        }
        for record in records
    ],
}, sort_keys=True))
PY

echo "completed task=$JOB_KEY mode=$MODE method=$METHOD"
