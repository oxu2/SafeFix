#!/usr/bin/env bash
#SBATCH --job-name=constrained-budget
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

usage() {
  cat <<'EOF'
Usage:
  sbatch --array=0-29%8 scripts/juno_constrained_budget_seed42.sbatch
  SLURM_ARRAY_TASK_ID=0 bash scripts/juno_constrained_budget_seed42.sbatch --dry-run

Runs the seed-42 constrained-source-budget comparison over ImageNet10, COCO,
and KITTI. Each dataset uses source ratios 0.02 and 0.05 with these methods:
  Generated-Complement, Target-Slice + Stratified,
  Attribute-Filtered Random, Attribute-Filtered Entropy, and AttriSelect.

Task layout: 10 tasks per dataset, in ImageNet10/COCO/KITTI order. Within a
dataset, each method occupies two consecutive tasks (ratio 0.02 then 0.05).

Options:
  --repo-dir PATH         Default: /work/$USER/safefix_rebuttal
  --data-root PATH        Default: /work/$USER/data
  --imagenet-python PATH  Default: /work/$USER/venvs/kitti/bin/python
  --pose-python PATH      Default: /groups/yguo/$USER/conda/envs/ad/bin/python
  --kitti-python PATH     Default: /work/$USER/venvs/kitti/bin/python
  --output-root PATH      Durable output root under the remote repository
  --dry-run               Print the resolved task and command without training
EOF
}

REPO_DIR="/work/${USER}/safefix_rebuttal"
DATA_ROOT="/work/${USER}/data"
IMAGENET_PYTHON="/work/${USER}/venvs/kitti/bin/python"
POSE_PYTHON="/groups/yguo/${USER}/conda/envs/ad/bin/python"
KITTI_PYTHON="/work/${USER}/venvs/kitti/bin/python"
OUTPUT_ROOT=""
DRY_RUN=0

require_value() {
  [[ $# -ge 2 && -n "$2" ]] || { echo "Missing value for $1" >&2; exit 2; }
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir) require_value "$@"; REPO_DIR="$2"; shift 2 ;;
    --data-root) require_value "$@"; DATA_ROOT="$2"; shift 2 ;;
    --imagenet-python) require_value "$@"; IMAGENET_PYTHON="$2"; shift 2 ;;
    --pose-python) require_value "$@"; POSE_PYTHON="$2"; shift 2 ;;
    --kitti-python) require_value "$@"; KITTI_PYTHON="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$OUTPUT_ROOT" ]] || OUTPUT_ROOT="$REPO_DIR/logs/juno/constrained_budget_seed42"

DATASETS=(imagenet10 coco kitti)
METHODS=(
  generated_complement
  target_slice_stratified
  attribute_filtered_random
  attribute_filtered_entropy
  attribute
)
RATIOS=(0.02 0.05)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
(( TASK_ID >= 0 && TASK_ID < 30 )) || {
  echo "Expected SLURM_ARRAY_TASK_ID in 0-29, got $TASK_ID" >&2
  exit 2
}

TASKS_PER_DATASET=$(( ${#METHODS[@]} * ${#RATIOS[@]} ))
DATASET_INDEX=$(( TASK_ID / TASKS_PER_DATASET ))
WITHIN_DATASET=$(( TASK_ID % TASKS_PER_DATASET ))
METHOD_INDEX=$(( WITHIN_DATASET / ${#RATIOS[@]} ))
RATIO_INDEX=$(( WITHIN_DATASET % ${#RATIOS[@]} ))

TASK="${DATASETS[$DATASET_INDEX]}"
METHOD="${METHODS[$METHOD_INDEX]}"
RATIO="${RATIOS[$RATIO_INDEX]}"
RATIO_TAG="${RATIO//./p}"
SEED=42
JOB_KEY="${SLURM_ARRAY_JOB_ID:-dryrun}_${TASK_ID}"
TASK_DIR="$OUTPUT_ROOT/$JOB_KEY"
LOG_FILE="$TASK_DIR/${TASK}_${METHOD}_ratio${RATIO_TAG}_seed${SEED}.log"

if [[ "$TASK" == "imagenet10" ]]; then
  IMAGENET_DATA="/home/${USER}/scratch/safefix_cache/data/imagenet_subset"
  IMAGENET_GENERATED="/home/${USER}/scratch/safefix_cache/data/imagenet_pinkcolor_fabrictexture"
  IMAGENET_MODEL="$REPO_DIR/save/models/model_imagenet_resnet_base.pth"
  IMAGENET_ATTRIBUTES="$REPO_DIR/save/attribute10_qwen3_8B.json"
  MANIFEST_FILE="$TASK_DIR/selection.json"
  RESULT_FILE="$TASK_DIR/result.json"
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
    env TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
    MPLCONFIGDIR="$TASK_DIR/matplotlib" CUDA_VISIBLE_DEVICES=0
    "$IMAGENET_PYTHON" -u "$REPO_DIR/alpred.py"
    --dataset imagenet --model resnet --data_dir "$IMAGENET_DATA"
    --gen_image_dir "$IMAGENET_GENERATED" --num_added_img 1000
    --generated_seed 42 --attributes pinkcolor_fabrictexture
    --resume_model "$IMAGENET_MODEL" --attribute_file "$IMAGENET_ATTRIBUTES"
    --al_attribute_focus color,background,texture --num_epochs 15 --lr 1e-5
    --batch_size 64 --workers 4 --sub_ratio "$RATIO" --seed "$SEED" --gpu 0
    --use_active_learning True --skip_feasible True --freeze_backbone False
    --save_pred_model best --save_preds False --save_model False
    --save_report False --verbose True --selection_manifest "$MANIFEST_FILE"
    --result_json "$RESULT_FILE" "${STRATEGY_ARGS[@]}"
  )
elif [[ "$TASK" == "coco" ]]; then
  POSE_DATA="$DATA_ROOT/coco"
  POSE_GENERATED="$DATA_ROOT/coco-pose-weather-edits/generated_coco_qwen3_1500_brown_rain_icedit"
  POSE_MODEL="$REPO_DIR/logs/pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530/best.pth"
  POSE_ATTRIBUTES="$REPO_DIR/save/attribute_coco_pose_gemini_train20k_qwen3_8B.json"
  CMD=(
    env PYTHONNOUSERSITE=1
    "$POSE_PYTHON" -u "$REPO_DIR/pose_estimation/train_pose_estimation.py" train
    --data-root "$POSE_DATA" --pretrained true --device cuda --batch-size 8
    --num-workers 4 --epochs 2 --eval-every 1 --lr 0.005 --seed "$SEED"
    --max-train-samples 20000 --sub-ratio "$RATIO" --resume-model "$POSE_MODEL"
    --attribute brown_rainy --generated-image-dir "$POSE_GENERATED"
    --num-added-img 1000 --analyze-after-train false --assigner qwen3
    --output-dir "$TASK_DIR/output" --use-active-learning true
    --al-strategy "$METHOD" --al-attribute-file "$POSE_ATTRIBUTES"
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
  SCRATCH_ROOT="/home/${USER}/scratch/safefix_rebuttal/constrained_budget_seed42/$JOB_KEY"
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
    --plots false --eval-test true --copy-images false --overwrite-prepared true
    --write-log false --timing-json "$TASK_DIR/timing.json"
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

[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo "Submit with sbatch or pass --dry-run." >&2
  exit 2
}

mkdir -p "$TASK_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
print_task
echo "host=$(hostname) slurm_job_id=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"${CMD[@]}"

if [[ "$TASK" == "imagenet10" ]]; then
  [[ -s "$MANIFEST_FILE" ]] || { echo "Missing selection manifest: $MANIFEST_FILE" >&2; exit 1; }
  [[ -s "$RESULT_FILE" ]] || { echo "Missing result JSON: $RESULT_FILE" >&2; exit 1; }
fi

echo "completed task=$TASK method=$METHOD ratio=$RATIO seed=$SEED"
