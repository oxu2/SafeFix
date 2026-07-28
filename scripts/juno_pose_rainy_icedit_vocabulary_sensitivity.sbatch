#!/usr/bin/env bash
#SBATCH --job-name=coco-rain-vocab
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

REPO_DIR="/work/${USER}/safefix_rebuttal"
CONDA_ENV="/groups/yguo/${USER}/conda/envs/ad"
DATA_ROOT="/work/${USER}/data/coco"
GENERATED_DIR="/home/${USER}/scratch/safefix_cache/data/coco-pose-weather-edits/generated_coco_qwen3_1500_rain_icedit"
BASE_MODEL=""
ATTRIBUTE_FILE=""
RATIOS="0.1,0.3,0.5,0.7,0.9,1.0"
NUM_ADDED_IMG=1000
MAX_TRAIN_SAMPLES=20000
BATCH_SIZE=8
NUM_WORKERS=4
EPOCHS_TOTAL=2
LR="0.005"
SEED=42
RARITY_POWER="0.03"
FREQUENCY_PENALTY="0.5"
CLASS_BALANCE="15"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  sbatch scripts/juno_pose_rainy_icedit_vocabulary_sensitivity.sbatch [options]

Runs the COCO rain-only ICEdit slice-granularity diagnostic sequentially on
one GPU. It keeps the brown-rainy repair protocol fixed except that source
selection, generated extra counts, and post-train slice analysis use only
weather=rainy.

Options:
  --repo-dir PATH
  --conda-env PATH
  --data-root PATH
  --generated-dir PATH
  --base-model PATH
  --attribute-file PATH
  --ratios CSV             Default: 0.1,0.3,0.5,0.7,0.9,1.0
  --num-added-img N        Default: 1000
  --max-train-samples N    Default: 20000
  --batch-size N           Default: 8
  --epochs-total N         Default: 2
  --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --generated-dir) GENERATED_DIR="$2"; shift 2 ;;
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    --attribute-file) ATTRIBUTE_FILE="$2"; shift 2 ;;
    --ratios) RATIOS="$2"; shift 2 ;;
    --num-added-img) NUM_ADDED_IMG="$2"; shift 2 ;;
    --max-train-samples) MAX_TRAIN_SAMPLES="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --epochs-total) EPOCHS_TOTAL="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "${SLURM_JOB_ID:-}" || "$DRY_RUN" -eq 1 ]] || {
  echo "Submit with sbatch, or pass --dry-run." >&2
  exit 2
}
for value_name in NUM_ADDED_IMG MAX_TRAIN_SAMPLES BATCH_SIZE EPOCHS_TOTAL; do
  value="${!value_name}"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "$value_name must be a positive integer; got $value" >&2
    exit 2
  }
done

if [[ -z "$BASE_MODEL" ]]; then
  BASE_MODEL="$REPO_DIR/logs/pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530/best.pth"
fi
if [[ -z "$ATTRIBUTE_FILE" ]]; then
  ATTRIBUTE_FILE="$REPO_DIR/save/attribute_coco_pose_gemini_train20k_qwen3_8B.json"
fi

PYTHON_EXE="$CONDA_ENV/bin/python"
NOTIFICATION_ENV_FILE="$REPO_DIR/.vscode/.env"
DRIVER_DIR="$REPO_DIR/logs/juno/coco_rainy_icedit_vocabulary_sensitivity/${SLURM_JOB_ID:-dry_run}"
DRIVER_LOG="$DRIVER_DIR/driver.log"

for required in "$PYTHON_EXE" "$DATA_ROOT" "$GENERATED_DIR" "$BASE_MODEL" "$ATTRIBUTE_FILE"; do
  if [[ ! -e "$required" ]]; then
    echo "Required path not found: $required" >&2
    exit 1
  fi
done

IFS=',' read -r -a RATIO_VALUES <<< "$RATIOS"
for ratio in "${RATIO_VALUES[@]}"; do
  awk -v value="$ratio" 'BEGIN { exit !(value > 0 && value <= 1) }' || {
    echo "Ratio must be in (0, 1]: $ratio" >&2
    exit 2
  }
done

mkdir -p "$DRIVER_DIR"
if [[ "$DRY_RUN" -eq 0 ]]; then
  exec > >(tee -a "$DRIVER_LOG") 2>&1
fi

if [[ -z "${NOTIFICATION_WEBHOOK:-}" && -f "$NOTIFICATION_ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$NOTIFICATION_ENV_FILE"
  set +a
fi

cleanup_and_notify() {
  local exit_code="$1"
  [[ "$DRY_RUN" -eq 0 ]] || return 0
  [[ -n "${NOTIFICATION_WEBHOOK:-}" ]] || return 0
  [[ -f "$REPO_DIR/scripts/test_notification.py" ]] || return 0
  local state="failed"
  local emoji="❌"
  if [[ "$exit_code" -eq 0 ]]; then
    state="completed"
    emoji="✅"
  fi
  python3 "$REPO_DIR/scripts/test_notification.py" \
    "$emoji COCO rain-only ICEdit job ${SLURM_JOB_ID:-dry_run} $state: ratios=$RATIOS artifacts=$DRIVER_DIR" \
    || true
}
trap 'cleanup_and_notify $?' EXIT

cd "$REPO_DIR"
echo "job_id=${SLURM_JOB_ID:-dry_run}"
echo "host=$(hostname)"
echo "ratios=$RATIOS"
echo "data_root=$DATA_ROOT"
echo "generated_dir=$GENERATED_DIR"
echo "base_model=$BASE_MODEL"
echo "attribute_file=$ATTRIBUTE_FILE"
if [[ "$DRY_RUN" -eq 0 ]]; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

for ratio in "${RATIO_VALUES[@]}"; do
  ratio_tag="${ratio//./p}"
  run_dir="$DRIVER_DIR/ratio_${ratio_tag}"
  output_dir="$run_dir/model"
  run_log="$run_dir/train.log"
  extra_count_scale="$(awk -v value="$ratio" 'BEGIN { printf "%.6g", 1.0 / value }')"
  mkdir -p "$run_dir"

  cmd=(
    env
    MPLCONFIGDIR=/tmp/matplotlib-safefix
    PYTHONNOUSERSITE=1
    CUDA_VISIBLE_DEVICES=0
    "$PYTHON_EXE" -u pose_estimation/train_pose_estimation.py train
    --data-root "$DATA_ROOT"
    --pretrained true
    --device cuda
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --epochs "$EPOCHS_TOTAL"
    --eval-every 1
    --lr "$LR"
    --seed "$SEED"
    --max-train-samples "$MAX_TRAIN_SAMPLES"
    --sub-ratio "$ratio"
    --resume-model "$BASE_MODEL"
    --attribute rainy
    --generated-image-dir "$GENERATED_DIR"
    --num-added-img "$NUM_ADDED_IMG"
    --analyze-after-train true
    --analysis-source qwen3
    --analysis-attributes weather
    --assigner qwen3
    --output-dir "$output_dir"
    --use-active-learning true
    --al-strategy attribute
    --al-attribute-file "$ATTRIBUTE_FILE"
    --al-attribute-focus weather
    --al-attribute-rarity-power "$RARITY_POWER"
    --al-attribute-uncertainty-boost 1.0
    --al-attribute-log-acc-scale 0.0
    --al-attribute-extra-counts "weather:rainy=${NUM_ADDED_IMG}"
    --al-attribute-extra-count-scale "$extra_count_scale"
    --al-attribute-frequency-penalty "$FREQUENCY_PENALTY"
    --al-attribute-class-balance "$CLASS_BALANCE"
    --al-attribute-use-entropy false
  )

  echo "starting ratio=$ratio extra_count_scale=$extra_count_scale at $(date --iso-8601=seconds)"
  printf '  %q' "${cmd[@]}"
  printf '\n'
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "${cmd[@]}" > "$run_log" 2>&1
  fi
  echo "finished ratio=$ratio at $(date --iso-8601=seconds)"
done

echo "completed ratios=$RATIOS"
