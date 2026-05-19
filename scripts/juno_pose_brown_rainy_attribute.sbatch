#!/usr/bin/env bash
#SBATCH --job-name=safefix-br-rain
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch scripts/juno_pose_brown_rainy_attribute.sbatch [options]

Runs one COCO pose resume job on Juno using attribute selection focused on:
  - clothes_color=brown
  - weather=rainy

Default Juno paths:
  repo:          ~/work/safefix
  COCO:          ~/work/data/coco
  generated dir:~/work/data/coco-pose-weather-edits/generated_coco_qwen3_1500_brown_rain_icedit
  attributes:    ~/work/safefix/save/attribute_coco_pose_gemini_train20k_qwen3_8B.json

Options:
  --repo-dir PATH          Repo path on Juno. Default: $PWD
  --conda-env PATH         Conda environment prefix. Default: /groups/yguo/oxx220000/conda/envs/ad
  --data-root PATH         COCO root on persistent storage. Default: ~/work/data/coco
  --generated-dir PATH     Brown-rainy generated image dir on persistent storage.
  --base-model PATH        Resume checkpoint path.
  --attribute-file PATH    Qwen3 train20k attribute metadata JSON.
  --ratio FLOAT            Original-data sub-ratio. Default: 0.9
  --num-added-img N        Number of generated images to append. Default: 1000
  --max-train-samples N    Original train cap before sub-ratio. Default: 20000
  --batch-size N           Training batch size. Default: 8
  --num-workers N          DataLoader workers. Default: 4
  --epochs-total N         Total epochs passed to pose_estimation/train_pose_estimation.py. Default: 2
  --rarity-power FLOAT     Attribute rarity exponent. Default: 0.0
  --frequency-penalty F    Attribute frequency penalty. Default: 0.0
  --class-balance VALUE    Pose class-balance weight. Default: 15.0
  --no-scratch-copy        Use persistent paths directly instead of scratch copies.
  --dry-run                Print the resolved command and exit.
EOF
}

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
CONDA_ENV="/groups/yguo/oxx220000/conda/envs/ad"
DATA_ROOT="$HOME/work/data/coco"
GENERATED_DIR="$HOME/work/data/coco-pose-weather-edits/generated_coco_qwen3_1500_brown_rain_icedit"
BASE_MODEL=""
ATTRIBUTE_FILE=""
RATIO="0.9"
NUM_ADDED_IMG=1000
MAX_TRAIN_SAMPLES=20000
BATCH_SIZE=8
NUM_WORKERS=4
EPOCHS_TOTAL=2
LR="0.005"
SEED=42
RARITY_POWER="0.0"
FREQUENCY_PENALTY="0.0"
CLASS_BALANCE="15.0"
USE_SCRATCH=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --generated-dir)
      GENERATED_DIR="$2"
      shift 2
      ;;
    --base-model)
      BASE_MODEL="$2"
      shift 2
      ;;
    --attribute-file)
      ATTRIBUTE_FILE="$2"
      shift 2
      ;;
    --ratio)
      RATIO="$2"
      shift 2
      ;;
    --num-added-img)
      NUM_ADDED_IMG="$2"
      shift 2
      ;;
    --max-train-samples)
      MAX_TRAIN_SAMPLES="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --epochs-total)
      EPOCHS_TOTAL="$2"
      shift 2
      ;;
    --rarity-power)
      RARITY_POWER="$2"
      shift 2
      ;;
    --frequency-penalty)
      FREQUENCY_PENALTY="$2"
      shift 2
      ;;
    --class-balance)
      CLASS_BALANCE="$2"
      shift 2
      ;;
    --no-scratch-copy)
      USE_SCRATCH=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

cd "$REPO_DIR"

if [[ -z "$BASE_MODEL" ]]; then
  BASE_MODEL="$REPO_DIR/logs/pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530/best.pth"
fi

if [[ -z "$ATTRIBUTE_FILE" ]]; then
  ATTRIBUTE_FILE="$REPO_DIR/save/attribute_coco_pose_gemini_train20k_qwen3_8B.json"
fi

PYTHON_EXE="$CONDA_ENV/bin/python"
for required in "$PYTHON_EXE" "$DATA_ROOT" "$GENERATED_DIR" "$BASE_MODEL" "$ATTRIBUTE_FILE"; do
  if [[ ! -e "$required" ]]; then
    echo "Required path not found: $required" >&2
    exit 1
  fi
done

RUN_DATA_ROOT="$DATA_ROOT"
RUN_GENERATED_DIR="$GENERATED_DIR"
SCRATCH_RUN_DIR=""

if [[ "$USE_SCRATCH" -eq 1 && "$DRY_RUN" -eq 0 ]]; then
  SCRATCH_BASE="${SCRATCH:-$HOME/scratch}"
  SCRATCH_RUN_DIR="$SCRATCH_BASE/safefix_${SLURM_JOB_ID:-manual}_brown_rainy_ratio${RATIO}"
  mkdir -p "$SCRATCH_RUN_DIR/coco-pose-weather-edits"

  echo "Copying COCO and brown-rainy generated images to scratch: $SCRATCH_RUN_DIR"
  rsync -a "$DATA_ROOT/" "$SCRATCH_RUN_DIR/coco/"
  rsync -a "$GENERATED_DIR/" "$SCRATCH_RUN_DIR/coco-pose-weather-edits/generated_coco_qwen3_1500_brown_rain_icedit/"

  RUN_DATA_ROOT="$SCRATCH_RUN_DIR/coco"
  RUN_GENERATED_DIR="$SCRATCH_RUN_DIR/coco-pose-weather-edits/generated_coco_qwen3_1500_brown_rain_icedit"
elif [[ "$USE_SCRATCH" -eq 1 ]]; then
  echo "Dry-run mode: skipping scratch copy and printing persistent input paths."
fi

mkdir -p "$REPO_DIR/logs/pose/pids"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXTRA_COUNT_SCALE="$(awk -v ratio="$RATIO" 'BEGIN { printf "%.6g", 1.0 / ratio }')"
RUN_NAME="qwen3_brown_rainy_r${RATIO}_r${RARITY_POWER}_p${FREQUENCY_PENALTY}_c${CLASS_BALANCE}"
RUN_NAME="${RUN_NAME//./p}"
OUT_DIR="$REPO_DIR/logs/pose/pose_keypointrcnn_resume_${RUN_NAME}_gpu0_${TIMESTAMP}"
LOG_FILE="$REPO_DIR/logs/pose/pose_resume_${RUN_NAME}_gpu0_${TIMESTAMP}.log"

cmd=(
  env
  MPLCONFIGDIR=/tmp/matplotlib-safefix
  CUDA_VISIBLE_DEVICES=0
  "$PYTHON_EXE" -u pose_estimation/train_pose_estimation.py train
  --data-root "$RUN_DATA_ROOT"
  --pretrained true
  --device cuda
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --epochs "$EPOCHS_TOTAL"
  --eval-every 1
  --lr "$LR"
  --seed "$SEED"
  --max-train-samples "$MAX_TRAIN_SAMPLES"
  --sub-ratio "$RATIO"
  --resume-model "$BASE_MODEL"
  --attribute brown_rainy
  --generated-image-dir "$RUN_GENERATED_DIR"
  --num-added-img "$NUM_ADDED_IMG"
  --analyze-after-train false
  --assigner qwen3
  --output-dir "$OUT_DIR"
  --use-active-learning true
  --al-strategy attribute
  --al-attribute-file "$ATTRIBUTE_FILE"
  --al-attribute-focus clothes_color,weather
  --al-attribute-rarity-power "$RARITY_POWER"
  --al-attribute-uncertainty-boost 1.0
  --al-attribute-log-acc-scale 0.0
  --al-attribute-extra-counts "clothes_color:brown=${NUM_ADDED_IMG},weather:rainy=${NUM_ADDED_IMG}"
  --al-attribute-extra-count-scale "$EXTRA_COUNT_SCALE"
  --al-attribute-frequency-penalty "$FREQUENCY_PENALTY"
  --al-attribute-class-balance "$CLASS_BALANCE"
  --al-attribute-use-entropy false
)

echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual}"
echo "data_root=$RUN_DATA_ROOT"
echo "generated_dir=$RUN_GENERATED_DIR"
echo "attribute_file=$ATTRIBUTE_FILE"
echo "extra_counts=clothes_color:brown=${NUM_ADDED_IMG},weather:rainy=${NUM_ADDED_IMG}"
echo "extra_scale=$EXTRA_COUNT_SCALE"
echo "log=$LOG_FILE"
echo "out=$OUT_DIR"
echo "Command:"
printf '  %q' "${cmd[@]}"
printf '\n'

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

"${cmd[@]}" > "$LOG_FILE" 2>&1

if [[ "$USE_SCRATCH" -eq 1 ]]; then
  echo "Scratch run directory retained for inspection: $SCRATCH_RUN_DIR"
fi
