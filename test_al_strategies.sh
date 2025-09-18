#!/bin/bash

# Function to check if GPU is free
wait_for_gpu_free() {
    local gpu_id=$1
    echo "Checking GPU $gpu_id for availability..."
    
    while true; do
        # Check GPU memory usage with nvidia-smi
        # Using 1000MB as a threshold for being "free"
        gpu_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=$gpu_id 2>/dev/null | tr -d ' ')
        
        # Check if the nvidia-smi command was successful and returned a value
        if [ $? -eq 0 ] && [ -n "$gpu_memory" ]; then
            if [ "$gpu_memory" -lt 1000 ]; then
                echo "GPU $gpu_id is free (memory usage: ${gpu_memory}MB)."
                break
            else
                echo "GPU $gpu_id is busy (memory usage: ${gpu_memory}MB). Waiting..."
            fi
        else
            # Fallback in case nvidia-smi fails: Check for any running python processes as a general indicator
            echo "nvidia-smi check failed for GPU $gpu_id. Falling back to process check."
            if ! pgrep -f "python" > /dev/null; then
                echo "GPU $gpu_id is likely free (no Python processes running)."
                break
            else
                echo "GPU $gpu_id might be busy (Python processes are running). Waiting..."
            fi
        fi
        
        sleep 30  # Wait for 30 seconds before checking again
    done
}

# Testing different Active Learning strategies for comparison
echo "Testing different Active Learning strategies..."

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
COMMON_ARGS="--dataset imagenet --model resnet --num_added_img 1000 --attributes pinkcolor_fabrictexture --num_epochs 30 --verbose True"

# Generate ratio string for filenames
RATIO_STR="ratio${SUB_RATIO}"

# ---
# Grouped sequential jobs for GPU 0
# The commands inside the parentheses will run one after the other.
# The '&' at the end runs the entire group in the background.
# ---
(
  echo "=== Queuing sequential jobs for GPU 0 ==="
  
  # Test 0: Random baseline (for comparison)
  echo "--- Waiting for GPU 0 to start Random Baseline ---"
  wait_for_gpu_free 0
  echo "--- Starting Random Baseline on GPU 0 ---"
  CUDA_VISIBLE_DEVICES=0 python -u alpred.py \
    $COMMON_ARGS \
    --sub_ratio $SUB_RATIO \
    --use_active_learning False \
    > logs_imagenet/AL_random_resnet_imagenet_1000_${RATIO_STR}.log 2>&1
  
  # Test 2: Stratified sampling strategy
  echo "--- Random Baseline finished. Starting Stratified Strategy on GPU 0 ---"
  # No need for another wait_for_gpu_free, as the previous job on this GPU just finished.
  CUDA_VISIBLE_DEVICES=0 python -u alpred.py \
    $COMMON_ARGS \
    --sub_ratio $SUB_RATIO \
    --al_strategy stratified \
    > logs_imagenet/AL_stratified_resnet_imagenet_1000_${RATIO_STR}.log 2>&1
  
  echo "=== All jobs for GPU 0 finished. ==="
) &


# Test 1: Core-set strategy (runs in parallel on GPU 1)
echo "=== Testing Core-set Strategy on GPU 1 ==="
wait_for_gpu_free 1
CUDA_VISIBLE_DEVICES=1 python -u alpred.py \
  $COMMON_ARGS \
  --sub_ratio $SUB_RATIO \
  --al_strategy coreset \
  --al_initial_labeled 400 \
  --al_embedding_dim 256 \
  > logs_imagenet/AL_coreset_resnet_imagenet_1000_${RATIO_STR}.log 2>&1 &

# Test 3: Diversity strategy (runs in parallel on GPU 2)
echo "=== Testing Diversity Strategy on GPU 2 ==="
wait_for_gpu_free 2
CUDA_VISIBLE_DEVICES=2 python -u alpred.py \
  $COMMON_ARGS \
  --sub_ratio $SUB_RATIO \
  --al_strategy diversity \
  --al_clusters 100 \
  --al_embedding_dim 256 \
  > logs_imagenet/AL_diversity_resnet_imagenet_1000_${RATIO_STR}.log 2>&1 &

# Test 4: Hybrid strategy (runs in parallel on GPU 3)
echo "=== Testing Hybrid Strategy on GPU 3 ==="
wait_for_gpu_free 3
CUDA_VISIBLE_DEVICES=3 python -u alpred.py \
  $COMMON_ARGS \
  --sub_ratio $SUB_RATIO \
  --al_strategy hybrid \
  --al_coreset_ratio 0.3 \
  --al_stratified_ratio 0.5 \
  --al_initial_labeled 200 \
  --al_embedding_dim 256 \
  > logs_imagenet/AL_hybrid_resnet_imagenet_1000_${RATIO_STR}.log 2>&1 &

echo "All tests started with sub_ratio=${SUB_RATIO}. Check logs in logs_imagenet/ directory."
echo "Log files pattern: AL_*_resnet_imagenet_1000_${RATIO_STR}.log"
echo "Monitor with: tail -f logs_imagenet/AL_*_${RATIO_STR}.log"