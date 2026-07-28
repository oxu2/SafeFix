#!/usr/bin/env bash
#SBATCH --job-name=img-vocab
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

REPO_DIR="/work/${USER}/safefix_rebuttal"
PYTHON_EXE="/work/${USER}/venvs/kitti/bin/python"
DATA_ROOT="/home/${USER}/scratch/safefix_cache/data"
SOURCE_DIR="$DATA_ROOT/imagenet_subset"
COMBINED_DIR="$DATA_ROOT/imagenet_pinkcolor_fabrictexture"
PINK_DIR="$DATA_ROOT/imagenet_pinkcolor"
FABRIC_DIR="$DATA_ROOT/imagenet_fabrictexture"
BASE_MODEL="$REPO_DIR/save/models/model_imagenet_resnet_base.pth"
ATTRIBUTE_FILE="$REPO_DIR/save/attribute10_qwen3_8B.json"
OUTPUT_ROOT="$REPO_DIR/logs/juno/imagenet_vocabulary_sensitivity"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RATIO="${VOCAB_RATIO:-0.5}"
OUTPUT_SUFFIX="${VOCAB_OUTPUT_SUFFIX:-}"

case "$RATIO" in
  0.1|0.3|0.5|0.7|0.9|1.0) ;;
  *) echo "Unsupported VOCAB_RATIO: $RATIO" >&2; exit 2 ;;
esac

case "$TASK_ID" in
  0)
    VARIANT="full_conjunction"
    GENERATED_DIR="$COMBINED_DIR"
    SELECTION_ATTRIBUTES="pinkcolor_fabrictexture"
    ATTRIBUTE_FOCUS="color,background,texture"
    ;;
  1)
    VARIANT="missing_texture_fixed_generated"
    GENERATED_DIR="$COMBINED_DIR"
    SELECTION_ATTRIBUTES="pinkcolor"
    ATTRIBUTE_FOCUS="color,background"
    ;;
  2)
    VARIANT="missing_color_fixed_generated"
    GENERATED_DIR="$COMBINED_DIR"
    SELECTION_ATTRIBUTES="fabrictexture"
    ATTRIBUTE_FOCUS="background,texture"
    ;;
  3)
    VARIANT="pink_only_slice_and_generated"
    GENERATED_DIR="$PINK_DIR"
    SELECTION_ATTRIBUTES="pinkcolor"
    ATTRIBUTE_FOCUS="color,background,texture"
    ;;
  4)
    VARIANT="fabric_only_slice_and_generated"
    GENERATED_DIR="$FABRIC_DIR"
    SELECTION_ATTRIBUTES="fabrictexture"
    ATTRIBUTE_FOCUS="color,background,texture"
    ;;
  *)
    echo "Expected SLURM_ARRAY_TASK_ID in 0-4, got $TASK_ID" >&2
    exit 2
    ;;
esac

if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  JOB_KEY="${SLURM_ARRAY_JOB_ID}_${TASK_ID}"
else
  JOB_KEY="dryrun_${TASK_ID}"
fi
TASK_DIR="$OUTPUT_ROOT/$JOB_KEY"
if [[ -n "$OUTPUT_SUFFIX" ]]; then
  TASK_DIR="$TASK_DIR/$OUTPUT_SUFFIX"
fi
LOG_FILE="$TASK_DIR/${VARIANT}.log"
MANIFEST_FILE="$TASK_DIR/${VARIANT}_selection.json"
RESULT_FILE="$TASK_DIR/${VARIANT}_result.json"
NOTIFICATION_ENV_FILE="$REPO_DIR/.vscode/.env"

CMD=(
  env TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
  MPLCONFIGDIR="$TASK_DIR/matplotlib" CUDA_VISIBLE_DEVICES=0
  "$PYTHON_EXE" -u "$REPO_DIR/alpred.py"
  --dataset imagenet
  --model resnet
  --data_dir "$SOURCE_DIR"
  --gen_image_dir "$GENERATED_DIR"
  --num_added_img 1000
  --generated_seed 42
  --attributes "$SELECTION_ATTRIBUTES"
  --evaluation_attributes pinkcolor_fabrictexture
  --resume_model "$BASE_MODEL"
  --attribute_file "$ATTRIBUTE_FILE"
  --al_strategy attribute
  --al_attribute_focus "$ATTRIBUTE_FOCUS"
  --al_attribute_rarity_power 0.03
  --al_attribute_frequency_penalty 0.5
  --al_attribute_class_balance 15
  --al_conditional_rarity True
  --al_attribute_use_entropy False
  --num_epochs 15
  --lr 1e-5
  --batch_size 64
  --workers 4
  --sub_ratio "$RATIO"
  --seed 42
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
)

print_task() {
  echo "variant=$VARIANT"
  echo "generated_dir=$GENERATED_DIR"
  echo "selection_attributes=$SELECTION_ATTRIBUTES"
  echo "evaluation_attributes=pinkcolor_fabrictexture"
  echo "attribute_focus=$ATTRIBUTE_FOCUS"
  echo "source_ratio=$RATIO"
  echo "source_seed=42"
  echo "generated_seed=42"
  echo "task_dir=$TASK_DIR"
  printf 'command='
  printf '%q ' "${CMD[@]}"
  printf '\n'
}

if [[ "${1:-}" == "--dry-run" ]]; then
  mkdir -p "$TASK_DIR"
  print_task
  exit 0
fi

[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo "Submit with sbatch --array=0-4%2, or run with --dry-run." >&2
  exit 2
}

for required_file in "$PYTHON_EXE" "$REPO_DIR/alpred.py" "$BASE_MODEL" "$ATTRIBUTE_FILE"; do
  [[ -s "$required_file" ]] || { echo "Missing or empty file: $required_file" >&2; exit 1; }
done
for required_dir in "$SOURCE_DIR" "$GENERATED_DIR"; do
  [[ -d "$required_dir" ]] || { echo "Missing directory: $required_dir" >&2; exit 1; }
done

mkdir -p "$TASK_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
print_task

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
    "$emoji ImageNet vocabulary task $JOB_KEY $state: variant=$VARIANT artifacts=$TASK_DIR" \
    || true
}
trap 'cleanup_and_notify $?' EXIT

echo "host=$(hostname)"
echo "slurm_job_id=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"${CMD[@]}"

[[ -s "$MANIFEST_FILE" ]] || { echo "Missing manifest: $MANIFEST_FILE" >&2; exit 1; }
[[ -s "$RESULT_FILE" ]] || { echo "Missing result: $RESULT_FILE" >&2; exit 1; }

"$PYTHON_EXE" - "$RESULT_FILE" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    result = json.load(handle)
metrics = result["test_slice_metrics"]
assert metrics["overall"]["count"] == 1300
assert metrics["target"]["count"] == 16
assert metrics["non_target"]["count"] == 1284
print(json.dumps(metrics, indent=2, sort_keys=True))
PY
