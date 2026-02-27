#!/bin/bash

# =============================================================================
# Fine-tune from a base model (trained on original data only) using different
# Active Learning strategies to select generated images.
#
# Usage:
#   ./test_al_resume.sh --sub_ratio 0.3
#   ./test_al_resume.sh --sub_ratio 0.3 --model_path save/models/my_base.pth
# =============================================================================

# Function to check if GPU is free with timestamp
wait_for_gpu_free() {
    local gpu_id=$1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking GPU $gpu_id for availability..."

    while true; do
        gpu_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=$gpu_id 2>/dev/null | tr -d ' \r\n')

        if [ $? -eq 0 ] && [ -n "$gpu_memory" ]; then
            if [ "$gpu_memory" -lt 1000 ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu_id is free (memory usage: ${gpu_memory}MB)."
                break
            else
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu_id is busy (memory usage: ${gpu_memory}MB). Waiting 30s..."
            fi
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] nvidia-smi check failed for GPU $gpu_id. Falling back to process check."
            if ! pgrep -f "python" > /dev/null; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu_id is likely free (no Python processes running)."
                break
            else
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu_id might be busy (Python processes are running). Waiting 30s..."
            fi
        fi

        sleep 30
    done
}

# Ensure logs directory exists
mkdir -p logs_imagenet

# Specify Python executable
PYTHON_EXE="/people/cs/o/oxx220000/.conda/envs/ad/bin/python"

if [ ! -f "$PYTHON_EXE" ]; then
    echo "Error: Python executable $PYTHON_EXE not found. Falling back to 'python3'."
    PYTHON_EXE="python3"
fi

# Default values
SUB_RATIO=0.3
BASE_MODEL="save/models/model_imagenet_resnet_base.pth"
NUM_EPOCHS=15

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sub_ratio)
            SUB_RATIO="$2"
            shift 2
            ;;
        --model_path)
            BASE_MODEL="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Verify base model exists
if [ ! -f "$BASE_MODEL" ]; then
    echo "Error: Base model not found at $BASE_MODEL"
    echo "Please train a base model first with:"
    echo "  python -u alpred.py --dataset imagenet --model resnet --num_added_img 0 \\"
    echo "    --attributes pinkcolor_fabrictexture --num_epochs 30 --save_model True \\"
    echo "    --use_active_learning False --sub_ratio 1"
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting AL resume-from-base comparison sweep..."
echo "Base model: $BASE_MODEL"
echo "Sub ratio:  $SUB_RATIO"
echo "Epochs:     $NUM_EPOCHS"

# Common parameters (fine-tuning: fewer epochs, resume from base)
COMMON_ARGS="--dataset imagenet --model resnet --num_added_img 1000 --attributes pinkcolor_fabrictexture --num_epochs $NUM_EPOCHS --verbose True --resume_model $BASE_MODEL"

# Generate ratio string for filenames
RATIO_STR="ratio${SUB_RATIO}"
LOG_PREFIX="logs_imagenet/resume_lr5_epoch15"

# --- GPU 0: Random Baseline & Stratified ---
(
  echo "=== GPU 0: Random & Stratified (resume) ==="

  wait_for_gpu_free 0
  echo "--- [GPU 0] Starting Random Baseline (resume) ---"
  CUDA_VISIBLE_DEVICES=0 $PYTHON_EXE -u alpred.py \
    $COMMON_ARGS \
    --sub_ratio $SUB_RATIO \
    --use_active_learning False \
    > ${LOG_PREFIX}_random_resnet_imagenet_1000_${RATIO_STR}.log 2>&1

  echo "--- [GPU 0] Starting Stratified Strategy (resume) ---"
  CUDA_VISIBLE_DEVICES=0 $PYTHON_EXE -u alpred.py \
    $COMMON_ARGS \
    --sub_ratio $SUB_RATIO \
    --al_strategy stratified \
    > ${LOG_PREFIX}_stratified_resnet_imagenet_1000_${RATIO_STR}.log 2>&1

  echo "=== [GPU 0] Jobs finished ==="
) &

# --- GPU 1: Core-set & Uncertainty-Stratified ---
(
  echo "=== GPU 1: Core-set & Uncertainty-Stratified (resume) ==="

  wait_for_gpu_free 1
  echo "--- [GPU 1] Starting Core-set Strategy (resume) ---"
  CUDA_VISIBLE_DEVICES=1 $PYTHON_EXE -u alpred.py \
    $COMMON_ARGS \
    --sub_ratio $SUB_RATIO \
    --al_strategy coreset \
    --al_initial_labeled 400 \
    --al_embedding_dim 256 \
    > ${LOG_PREFIX}_coreset_resnet_imagenet_1000_${RATIO_STR}.log 2>&1

  echo "--- [GPU 1] Starting Uncertainty-Stratified Strategy (resume) ---"
  CUDA_VISIBLE_DEVICES=1 $PYTHON_EXE -u alpred.py \
    $COMMON_ARGS \
    --sub_ratio $SUB_RATIO \
    --al_strategy uncertainty_stratified \
    > ${LOG_PREFIX}_uncertainty_stratified_resnet_imagenet_1000_${RATIO_STR}.log 2>&1

  echo "=== [GPU 1] Jobs finished ==="
) &

# --- GPU 2: Diversity & Hierarchical ---
(
  echo "=== GPU 2: Diversity & Hierarchical (resume) ==="

  wait_for_gpu_free 2
  echo "--- [GPU 2] Starting Diversity Strategy (resume) ---"
  CUDA_VISIBLE_DEVICES=2 $PYTHON_EXE -u alpred.py \
    $COMMON_ARGS \
    --sub_ratio $SUB_RATIO \
    --al_strategy diversity \
    --al_clusters 100 \
    --al_embedding_dim 256 \
    > ${LOG_PREFIX}_diversity_resnet_imagenet_1000_${RATIO_STR}.log 2>&1

  echo "--- [GPU 2] Starting Hierarchical Strategy (resume) ---"
  CUDA_VISIBLE_DEVICES=2 $PYTHON_EXE -u alpred.py \
    $COMMON_ARGS \
    --sub_ratio $SUB_RATIO \
    --al_strategy hierarchical \
    --al_alpha 0.4 --al_beta 0.4 --al_gamma 0.2 \
    > ${LOG_PREFIX}_hierarchical_resnet_imagenet_1000_${RATIO_STR}.log 2>&1

  echo "=== [GPU 2] Jobs finished ==="
) &

# --- GPU 3: Hybrid & Attribute-Focused ---
(
  echo "=== GPU 3: Hybrid & Attribute-Focused (resume) ==="

  wait_for_gpu_free 3
  echo "--- [GPU 3] Starting Hybrid Strategy (resume) ---"
  CUDA_VISIBLE_DEVICES=3 $PYTHON_EXE -u alpred.py \
    $COMMON_ARGS \
    --sub_ratio $SUB_RATIO \
    --al_strategy hybrid \
    --al_coreset_ratio 0.3 \
    --al_stratified_ratio 0.5 \
    --al_initial_labeled 200 \
    --al_embedding_dim 256 \
    > ${LOG_PREFIX}_hybrid_resnet_imagenet_1000_${RATIO_STR}.log 2>&1

  echo "--- [GPU 3] Starting Attribute Strategy (resume) ---"
  CUDA_VISIBLE_DEVICES=3 $PYTHON_EXE -u alpred.py \
    $COMMON_ARGS \
    --sub_ratio $SUB_RATIO \
    --al_strategy attribute \
    --al_attribute_focus color,background,texture \
    --al_attribute_balance_strength 5 \
    --al_attribute_class_balance inf \
    > ${LOG_PREFIX}_attribute_resnet_imagenet_1000_${RATIO_STR}.log 2>&1

  echo "=== [GPU 3] Jobs finished ==="
) &

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All resume fine-tuning pipelines launched with sub_ratio=${SUB_RATIO}."
echo "Log files: ${LOG_PREFIX}_*_${RATIO_STR}.log"
