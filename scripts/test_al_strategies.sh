#!/bin/bash

# Function to check if GPU is free with timestamp
wait_for_gpu_free() {
    local gpu_id=$1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking GPU $gpu_id for availability..."
    
    while true; do
        # Check GPU memory usage with nvidia-smi
        # Using 1000MB as a threshold for being "free"
        gpu_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=$gpu_id 2>/dev/null | tr -d ' \r\n')
        
        # Check if the nvidia-smi command was successful and returned a value
        if [ $? -eq 0 ] && [ -n "$gpu_memory" ]; then
            if [ "$gpu_memory" -lt 1000 ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu_id is free (memory usage: ${gpu_memory}MB)."
                break
            else
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu_id is busy (memory usage: ${gpu_memory}MB). Waiting 30s..."
            fi
        else
            # Fallback in case nvidia-smi fails: Check for any running python processes as a general indicator
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

# Verify Python executable exists
if [ ! -f "$PYTHON_EXE" ]; then
    echo "Error: Python executable $PYTHON_EXE not found. Falling back to 'python3'."
    PYTHON_EXE="python3"
fi

# Testing different Active Learning strategies for comparison
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Active Learning strategies comparison sweep..."

# Parse command line arguments for sub_ratio
SUB_RATIO=0.5  # Default value
while [[ $# -gt 0 ]]; do
    case $1 in
        --sub_ratio)
            SUB_RATIO="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "Using sub_ratio: $SUB_RATIO"

# Common parameters
# Note: --num_added_img 1000 is the budget for selected images
COMMON_ARGS="--dataset imagenet --model resnet --num_added_img 1000 --attributes pinkcolor_fabrictexture --num_epochs 30 --verbose True"

# Generate ratio string for filenames
RATIO_STR="ratio${SUB_RATIO}"

# --- GPU 0: Random Baseline & Stratified ---
(
  echo "=== GPU 0: Random & Stratified ==="
  
  wait_for_gpu_free 0
  echo "--- [GPU 0] Starting Random Baseline ---"
  CUDA_VISIBLE_DEVICES=0 $PYTHON_EXE -u alpred.py \
    $COMMON_ARGS \
    --sub_ratio $SUB_RATIO \
    --use_active_learning False \
    > logs_imagenet/AL_random_resnet_imagenet_1000_${RATIO_STR}.log 2>&1
  
  echo "--- [GPU 0] Starting Stratified Strategy ---"
  CUDA_VISIBLE_DEVICES=0 $PYTHON_EXE -u alpred.py \
    $COMMON_ARGS \
    --sub_ratio $SUB_RATIO \
    --al_strategy stratified \
    > logs_imagenet/AL_stratified_resnet_imagenet_1000_${RATIO_STR}.log 2>&1
  
  echo "=== [GPU 0] Jobs finished ==="
) &

# --- GPU 1: Core-set & Uncertainty-Stratified ---
(
  echo "=== GPU 1: Core-set & Uncertainty-Stratified ==="
  
  wait_for_gpu_free 1
  echo "--- [GPU 1] Starting Core-set Strategy ---"
  CUDA_VISIBLE_DEVICES=1 $PYTHON_EXE -u alpred.py \
    $COMMON_ARGS \
    --sub_ratio $SUB_RATIO \
    --al_strategy coreset \
    --al_initial_labeled 400 \
    --al_embedding_dim 256 \
    > logs_imagenet/AL_coreset_resnet_imagenet_1000_${RATIO_STR}.log 2>&1
    
  echo "--- [GPU 1] Starting Uncertainty-Stratified Strategy ---"
  CUDA_VISIBLE_DEVICES=1 $PYTHON_EXE -u alpred.py \
    $COMMON_ARGS \
    --sub_ratio $SUB_RATIO \
    --al_strategy uncertainty_stratified \
    > logs_imagenet/AL_uncertainty_stratified_resnet_imagenet_1000_${RATIO_STR}.log 2>&1

  echo "=== [GPU 1] Jobs finished ==="
) &

# --- GPU 2: Diversity & Hierarchical ---
(
  echo "=== GPU 2: Diversity & Hierarchical ==="
  
  wait_for_gpu_free 2
  echo "--- [GPU 2] Starting Diversity Strategy ---"
  CUDA_VISIBLE_DEVICES=2 $PYTHON_EXE -u alpred.py \
    $COMMON_ARGS \
    --sub_ratio $SUB_RATIO \
    --al_strategy diversity \
    --al_clusters 100 \
    --al_embedding_dim 256 \
    > logs_imagenet/AL_diversity_resnet_imagenet_1000_${RATIO_STR}.log 2>&1
    
  echo "--- [GPU 2] Starting Hierarchical Strategy ---"
  CUDA_VISIBLE_DEVICES=2 $PYTHON_EXE -u alpred.py \
    $COMMON_ARGS \
    --sub_ratio $SUB_RATIO \
    --al_strategy hierarchical \
    --al_alpha 0.4 --al_beta 0.4 --al_gamma 0.2 \
    > logs_imagenet/AL_hierarchical_resnet_imagenet_1000_${RATIO_STR}.log 2>&1

  echo "=== [GPU 2] Jobs finished ==="
) &

# --- GPU 3: Hybrid & Attribute-Focused ---
(
  echo "=== GPU 3: Hybrid & Attribute-Focused ==="
  
  wait_for_gpu_free 3
  echo "--- [GPU 3] Starting Hybrid Strategy ---"
  CUDA_VISIBLE_DEVICES=3 $PYTHON_EXE -u alpred.py \
    $COMMON_ARGS \
    --sub_ratio $SUB_RATIO \
    --al_strategy hybrid \
    --al_coreset_ratio 0.3 \
    --al_stratified_ratio 0.5 \
    --al_initial_labeled 200 \
    --al_embedding_dim 256 \
    > logs_imagenet/AL_hybrid_resnet_imagenet_1000_${RATIO_STR}.log 2>&1
    
  echo "--- [GPU 3] Starting Attribute Strategy ---"
  # Attempt to use attribute strategy if metadata is available
  CUDA_VISIBLE_DEVICES=3 $PYTHON_EXE -u alpred.py \
    $COMMON_ARGS \
    --sub_ratio $SUB_RATIO \
    --al_strategy attribute \
    --al_attribute_focus pinkcolor,fabrictexture \
    > logs_imagenet/AL_attribute_resnet_imagenet_1000_${RATIO_STR}.log 2>&1

  echo "=== [GPU 3] Jobs finished ==="
) &

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All experiment pipelines launched with sub_ratio=${SUB_RATIO}."
echo "Check main log for progress or individual logs in logs_imagenet/ directory."