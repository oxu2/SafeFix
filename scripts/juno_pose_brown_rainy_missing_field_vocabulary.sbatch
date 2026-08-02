#!/usr/bin/env bash
#SBATCH --job-name=coco-miss-field
#SBATCH --partition=h200
#SBATCH --qos=juno-pri
#SBATCH --array=0-14%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

REPO_DIR="/work/${USER}/safefix_rebuttal"
CONDA_ENV="/groups/yguo/${USER}/conda/envs/ad"
DATA_ROOT="/work/${USER}/data/coco"
GENERATED_DIR="/home/${USER}/scratch/safefix_cache/data/coco-pose-weather-edits/generated_coco_qwen3_1500_brown_rain_icedit"
BASE_MODEL="$REPO_DIR/logs/pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530/best.pth"
ATTRIBUTE_FILE="$REPO_DIR/save/attribute_coco_pose_gemini_train20k_qwen3_8B.json"
OUTPUT_ROOT="$REPO_DIR/logs/juno/coco_missing_field_vocabulary"
RATIOS=(0.1 0.3 0.5 0.7 0.9)
VARIANTS=(full_conjunction missing_weather missing_color)
NUM_ADDED_IMG=1000
MAX_TRAIN_SAMPLES=20000
SEED=42
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
[[ "$TASK_ID" =~ ^[0-9]+$ ]] && (( TASK_ID >= 0 && TASK_ID < 15 )) || {
  echo "Expected task id in 0-14; got $TASK_ID" >&2
  exit 2
}

RATIO_INDEX=$((TASK_ID / 3))
VARIANT_INDEX=$((TASK_ID % 3))
RATIO="${RATIOS[$RATIO_INDEX]}"
VARIANT="${VARIANTS[$VARIANT_INDEX]}"

case "$VARIANT" in
  full_conjunction)
    ATTRIBUTE_FOCUS="clothes_color,weather"
    EXTRA_COUNTS="clothes_color:brown=${NUM_ADDED_IMG},weather:rainy=${NUM_ADDED_IMG}"
    ;;
  missing_weather)
    ATTRIBUTE_FOCUS="clothes_color"
    EXTRA_COUNTS="clothes_color:brown=${NUM_ADDED_IMG}"
    ;;
  missing_color)
    ATTRIBUTE_FOCUS="weather"
    EXTRA_COUNTS="weather:rainy=${NUM_ADDED_IMG}"
    ;;
  *)
    echo "Unsupported variant: $VARIANT" >&2
    exit 2
    ;;
esac

PYTHON_EXE="$CONDA_ENV/bin/python"
JOB_KEY="${SLURM_ARRAY_JOB_ID:-dry_run}_${TASK_ID}"
RUN_DIR="$OUTPUT_ROOT/$JOB_KEY"
MODEL_DIR="$RUN_DIR/model"
TRAIN_LOG="$RUN_DIR/train.log"
SELECTION_MANIFEST="$RUN_DIR/selection_manifest.json"
NOTIFICATION_ENV_FILE="$REPO_DIR/.vscode/.env"
EXTRA_COUNT_SCALE="$(awk -v value="$RATIO" 'BEGIN { printf "%.6g", 1.0 / value }')"

cmd=(
  env
  MPLCONFIGDIR=/tmp/matplotlib-safefix
  PYTHONNOUSERSITE=1
  CUDA_VISIBLE_DEVICES=0
  "$PYTHON_EXE" -u "$REPO_DIR/pose_estimation/train_pose_estimation.py" train
  --data-root "$DATA_ROOT"
  --pretrained true
  --device cuda
  --batch-size 8
  --num-workers 4
  --epochs 2
  --eval-every 1
  --lr 0.005
  --seed "$SEED"
  --max-train-samples "$MAX_TRAIN_SAMPLES"
  --sub-ratio "$RATIO"
  --resume-model "$BASE_MODEL"
  --attribute brown_rainy
  --generated-image-dir "$GENERATED_DIR"
  --num-added-img "$NUM_ADDED_IMG"
  --selection-manifest "$SELECTION_MANIFEST"
  --analyze-after-train true
  --analysis-source qwen3
  --analysis-attributes clothes_color,weather
  --assigner qwen3
  --output-dir "$MODEL_DIR"
  --use-active-learning true
  --al-strategy attribute
  --al-attribute-file "$ATTRIBUTE_FILE"
  --al-attribute-focus "$ATTRIBUTE_FOCUS"
  --al-attribute-rarity-power 0.03
  --al-attribute-uncertainty-boost 1.0
  --al-attribute-log-acc-scale 0.0
  --al-attribute-extra-counts "$EXTRA_COUNTS"
  --al-attribute-extra-count-scale "$EXTRA_COUNT_SCALE"
  --al-attribute-frequency-penalty 0.5
  --al-attribute-class-balance 15
  --al-attribute-use-entropy false
)

print_task() {
  echo "task_id=$TASK_ID"
  echo "ratio=$RATIO"
  echo "variant=$VARIANT"
  echo "attribute_focus=$ATTRIBUTE_FOCUS"
  echo "extra_counts=$EXTRA_COUNTS"
  echo "extra_count_scale=$EXTRA_COUNT_SCALE"
  echo "generated_dir=$GENERATED_DIR"
  echo "evaluation_attributes=clothes_color,weather"
  echo "run_dir=$RUN_DIR"
  printf 'command='
  printf '%q ' "${cmd[@]}"
  printf '\n'
}

if [[ "$DRY_RUN" -eq 1 ]]; then
  print_task
  exit 0
fi

[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo "Submit with sbatch, or pass --dry-run." >&2
  exit 2
}

for required in "$PYTHON_EXE" "$DATA_ROOT" "$GENERATED_DIR" "$BASE_MODEL" "$ATTRIBUTE_FILE"; do
  [[ -e "$required" ]] || { echo "Required path not found: $required" >&2; exit 1; }
done

mkdir -p "$RUN_DIR"
exec > >(tee -a "$TRAIN_LOG") 2>&1
print_task
echo "host=$(hostname)"
echo "slurm_job_id=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

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
  if [[ "$exit_code" -eq 0 && -s "$MODEL_DIR/test_metrics.json" && -s "$SELECTION_MANIFEST" ]]; then
    state="completed"
    emoji="✅"
  fi
  python3 "$REPO_DIR/scripts/test_notification.py" \
    "$emoji COCO missing-field task $JOB_KEY $state: ratio=$RATIO variant=$VARIANT artifacts=$RUN_DIR" \
    || true
}
trap 'cleanup_and_notify $?' EXIT

"${cmd[@]}"

for required_output in \
  "$SELECTION_MANIFEST" \
  "$MODEL_DIR/test_metrics.json" \
  "$MODEL_DIR/slice_analysis/qwen3/test_slice_stats.json"; do
  [[ -s "$required_output" ]] || { echo "Missing output: $required_output" >&2; exit 1; }
done

echo "completed task=$TASK_ID ratio=$RATIO variant=$VARIANT"
