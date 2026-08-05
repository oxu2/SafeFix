#!/usr/bin/env bash
#SBATCH --job-name=ck-sameinfo
#SBATCH --partition=h200
#SBATCH --qos=juno-pri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

REPO_DIR="/work/${USER}/safefix_rebuttal"
DATA_ROOT="/work/${USER}/data"
POSE_PYTHON="/groups/yguo/oxx220000/conda/envs/ad/bin/python"
KITTI_PYTHON="/work/${USER}/venvs/kitti/bin/python"
OUTPUT_ROOT=""
DRY_RUN=0
COCO_BASELINES_ONLY=0
ENTROPY_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --pose-python) POSE_PYTHON="$2"; shift 2 ;;
    --kitti-python) KITTI_PYTHON="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --coco-baselines-only) COCO_BASELINES_ONLY=1; shift ;;
    --entropy-only) ENTROPY_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$OUTPUT_ROOT" ]] || OUTPUT_ROOT="$REPO_DIR/logs/juno/coco_kitti_same_information"

METHODS=(target_slice_stratified attribute_filtered_random attribute)
RATIOS=(0.1 0.3 0.5 0.7 0.9)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "$ENTROPY_ONLY" -eq 1 ]]; then
  (( TASK_ID >= 0 && TASK_ID < 10 )) || {
    echo "Entropy-only mode expects task ID 0-9, got $TASK_ID" >&2
    exit 2
  }
  METHOD="attribute_filtered_entropy"
  if (( TASK_ID < 5 )); then
    TASK="coco"
    RATIO_INDEX="$TASK_ID"
  else
    TASK="kitti"
    RATIO_INDEX=$((TASK_ID - 5))
  fi
  RATIO="${RATIOS[$RATIO_INDEX]}"
elif [[ "$COCO_BASELINES_ONLY" -eq 1 ]]; then
  (( TASK_ID >= 0 && TASK_ID < 10 )) || {
    echo "COCO-baselines mode expects task ID 0-9, got $TASK_ID" >&2
    exit 2
  }
  TASK="coco"
  LOCAL_ID="$TASK_ID"
else
  (( TASK_ID >= 0 && TASK_ID < 30 )) || {
    echo "Full mode expects task ID 0-29, got $TASK_ID" >&2
    exit 2
  }
  if (( TASK_ID < 15 )); then
    TASK="coco"
    LOCAL_ID="$TASK_ID"
  else
    TASK="kitti"
    LOCAL_ID=$((TASK_ID - 15))
  fi
fi
if [[ "$ENTROPY_ONLY" -eq 0 ]]; then
  METHOD_INDEX=$((LOCAL_ID / 5))
  RATIO_INDEX=$((LOCAL_ID % 5))
  METHOD="${METHODS[$METHOD_INDEX]}"
  RATIO="${RATIOS[$RATIO_INDEX]}"
fi
SEED=42
RATIO_TAG="${RATIO//./p}"
JOB_KEY="${SLURM_ARRAY_JOB_ID:-dryrun}_${TASK_ID}"
TASK_DIR="$OUTPUT_ROOT/$JOB_KEY"
LOG_FILE="$TASK_DIR/${TASK}_${METHOD}_ratio${RATIO_TAG}_seed${SEED}.log"

if [[ "$TASK" == "coco" ]]; then
  POSE_DATA="$DATA_ROOT/coco"
  POSE_GENERATED="$DATA_ROOT/coco-pose-weather-edits/generated_coco_qwen3_1500_brown_rain_icedit"
  POSE_MODEL="$REPO_DIR/logs/pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530/best.pth"
  POSE_ATTRIBUTES="$REPO_DIR/save/attribute_coco_pose_gemini_train20k_qwen3_8B.json"
  CMD=(
    env PYTHONNOUSERSITE=1
    "$POSE_PYTHON" -u "$REPO_DIR/pose_estimation/train_pose_estimation.py" train
    --data-root "$POSE_DATA" --pretrained true --device cuda
    --batch-size 8 --num-workers 4 --epochs 2 --eval-every 1 --lr 0.005
    --seed "$SEED" --max-train-samples 20000 --sub-ratio "$RATIO"
    --resume-model "$POSE_MODEL" --attribute brown_rainy
    --generated-image-dir "$POSE_GENERATED" --num-added-img 1000
    --analyze-after-train false --assigner qwen3 --output-dir "$TASK_DIR/output"
    --use-active-learning true --al-strategy "$METHOD"
    --al-attribute-file "$POSE_ATTRIBUTES"
    --al-attribute-focus clothes_color,weather
    --al-attribute-rarity-power 0.03 --al-attribute-use-rarity true
    --al-attribute-extra-counts "clothes_color:brown=1000,weather:rainy=1000"
    --al-attribute-frequency-penalty 0.5 --al-attribute-class-balance 15
    --al-conditional-rarity true --al-attribute-use-entropy false
  )
else
  KITTI_DATA="$DATA_ROOT/kitti_yolov8_first5000_1240val_1241test"
  [[ -d "$KITTI_DATA" ]] || KITTI_DATA="$DATA_ROOT/kitti_yolov8_first5000"
  KITTI_GENERATED="$DATA_ROOT/kitti-weather-edits/generated_kitti1352_red_cloudy_icedit_upscaled"
  KITTI_MODEL="$REPO_DIR/save/runs/kitti_yolov8/first5000_1240val_1241test_gpu0_bs16_e30_20260429_012232/weights/best.pt"
  [[ -s "$KITTI_MODEL" ]] || KITTI_MODEL="$DATA_ROOT/checkpoints/kitti_yolov8_first5000_best.pt"
  KITTI_ATTRIBUTES="$REPO_DIR/save/attribute_kitti_qwen3_8B.json"
  SCRATCH_ROOT="/home/${USER}/scratch/safefix_rebuttal/coco_kitti_same_information/$JOB_KEY"
  CMD=(
    "$KITTI_PYTHON" -u "$REPO_DIR/object_detection/train_yolov8_kitti_generated.py"
    --base-data-root "$KITTI_DATA" --generated-image-dir "$KITTI_GENERATED"
    --prepared-root "$SCRATCH_ROOT/prepared" --resume-model "$KITTI_MODEL"
    --sub-ratio "$RATIO" --selection-strategy "$METHOD"
    --attribute-file "$KITTI_ATTRIBUTES" --al-attribute-focus color,weather
    --al-attribute-rarity-power 0.03 --al-attribute-use-rarity true
    --al-attribute-extra-counts "color:red=1000;weather:cloudy=1000"
    --al-attribute-frequency-penalty 5 --al-attribute-class-balance 15
    --al-conditional-rarity false --num-added-img 1000
    --require-exact-generated-count true --epochs 10 --batch 16 --imgsz 640
    --optimizer AdamW --lr0 2e-4 --lrf 0.1 --warmup-epochs 1
    --seed "$SEED" --device 0 --workers 4 --project "$TASK_DIR/runs"
    --name "${TASK}_${METHOD}_ratio${RATIO_TAG}_seed${SEED}"
    --plots false --eval-test true --copy-images false
    --overwrite-prepared true --write-log false
    --timing-json "$TASK_DIR/timing.json"
  )
fi

print_task() {
  echo "task_id=$TASK_ID task=$TASK method=$METHOD ratio=$RATIO seed=$SEED"
  echo "task_dir=$TASK_DIR"
  printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'
}

if [[ "$DRY_RUN" -eq 1 ]]; then
  print_task
  exit 0
fi

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Submit with sbatch or use --dry-run." >&2; exit 2; }
mkdir -p "$TASK_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
print_task
echo "host=$(hostname) slurm_job_id=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"${CMD[@]}"
echo "completed task=$TASK method=$METHOD ratio=$RATIO seed=$SEED"
