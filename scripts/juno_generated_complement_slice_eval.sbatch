#!/usr/bin/env bash
#SBATCH --job-name=gencomp-slice
#SBATCH --partition=h200
#SBATCH --qos=juno-pri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

REPO_DIR="/work/${USER}/safefix_rebuttal"
DATA_ROOT="/work/${USER}/data"
POSE_PYTHON="/groups/yguo/oxx220000/conda/envs/ad/bin/python"
KITTI_PYTHON="/work/${USER}/venvs/kitti/bin/python"
SOURCE_JOB_ID="313018"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
(( TASK_ID >= 0 && TASK_ID < 10 )) || {
  echo "Expected task ID 0-9, got $TASK_ID" >&2
  exit 2
}

RATIOS=(0.1 0.3 0.5 0.7 0.9)
if (( TASK_ID < 5 )); then
  TASK="coco"
  RATIO_INDEX="$TASK_ID"
  SOURCE_TASK_ID=$((TASK_ID + 5))
else
  TASK="kitti"
  RATIO_INDEX=$((TASK_ID - 5))
  SOURCE_TASK_ID=$((TASK_ID + 5))
fi

RATIO="${RATIOS[$RATIO_INDEX]}"
RATIO_TAG="${RATIO//./p}"
SOURCE_DIR="$REPO_DIR/logs/juno/generated_complement_seed42/${SOURCE_JOB_ID}_${SOURCE_TASK_ID}"
OUTPUT_DIR="$SOURCE_DIR/slice_evaluation"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/${TASK}_ratio${RATIO_TAG}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "task=$TASK ratio=$RATIO source_dir=$SOURCE_DIR"
if [[ "$TASK" == "coco" ]]; then
  env PYTHONNOUSERSITE=1 "$POSE_PYTHON" -u "$REPO_DIR/utils/stat_pose_estimation.py" \
    --predictions-json "$SOURCE_DIR/output/test_predictions.json" \
    --attribute-file "$REPO_DIR/save/attribute_coco_pose_gemini_qwen3_8B_val.json" \
    --attribute-split val_full --eval-split test \
    --data-root "$DATA_ROOT/coco" --attributes clothes_color,weather \
    --exact-slice "clothes_color=brown,weather=rainy" --min-images 1 \
    --output-json "$OUTPUT_DIR/exact_slice.json" \
    --output-csv "$OUTPUT_DIR/single_attribute_slices.csv"
else
  KITTI_ROOT="$DATA_ROOT/kitti_yolov8_first5000_1240val_1241test"
  [[ -d "$KITTI_ROOT" ]] || KITTI_ROOT="$DATA_ROOT/kitti_yolov8_first5000"
  WEIGHTS="$SOURCE_DIR/runs/kitti_generated_complement_ratio${RATIO_TAG}_seed42/weights/best.pt"
  "$KITTI_PYTHON" -u "$REPO_DIR/scripts/evaluate_kitti_exact_slice.py" \
    --model-weights "$WEIGHTS" \
    --attribute-file "$REPO_DIR/save/attribute_kitti_qwen3_8B_test.json" \
    --kitti-root "$KITTI_ROOT" --exact-slice "color=red,weather=cloudy" \
    --output-dir "$OUTPUT_DIR/official"
fi

echo "completed task=$TASK ratio=$RATIO"
