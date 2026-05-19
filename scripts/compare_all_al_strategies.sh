#!/bin/bash

# Comprehensive comparison of Hierarchical AL vs other AL strategies
# Based on test_combined_al.sh pattern

echo "Comprehensive Active Learning Strategy Comparison"
echo "==============================================="

# Parse command line arguments
SUB_RATIO=0.5  # Default value
NUM_ADDED_IMG=1000  # Default value
while [[ $# -gt 0 ]]; do
    case $1 in
        --sub_ratio)
            SUB_RATIO="$2"
            shift 2
            ;;
        --num_added_img)
            NUM_ADDED_IMG="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "Using sub_ratio: $SUB_RATIO"
echo "Using num_added_img: $NUM_ADDED_IMG"

# Common parameters
COMMON_ARGS="--dataset imagenet --model resnet --num_added_img $NUM_ADDED_IMG --attributes pinkcolor_fabrictexture --num_epochs 30 --verbose True"

echo ""
echo "Launching all AL strategy tests in parallel..."

echo "=== Test 0: Random Baseline ==="
CUDA_VISIBLE_DEVICES=0 python -u alpred.py \
  $COMMON_ARGS \
  --sub_ratio $SUB_RATIO \
  --use_active_learning False \
  > logs/imagenet/AL_comparison_random_imagenet_${NUM_ADDED_IMG}_${SUB_RATIO}.log 2>&1 &

echo "=== Test 1: Core-set AL ==="
CUDA_VISIBLE_DEVICES=1 python -u alpred.py \
  $COMMON_ARGS \
  --sub_ratio $SUB_RATIO \
  --al_strategy coreset \
  --al_embedding_dim 256 \
  > logs/imagenet/AL_comparison_coreset_imagenet_${NUM_ADDED_IMG}_${SUB_RATIO}.log 2>&1 &

echo "=== Test 2: Stratified AL ==="
CUDA_VISIBLE_DEVICES=2 python -u alpred.py \
  $COMMON_ARGS \
  --sub_ratio $SUB_RATIO \
  --al_strategy stratified \
  > logs/imagenet/AL_comparison_stratified_imagenet_${NUM_ADDED_IMG}_${SUB_RATIO}.log 2>&1 &

echo "=== Test 3: Diversity AL ==="
CUDA_VISIBLE_DEVICES=3 python -u alpred.py \
  $COMMON_ARGS \
  --sub_ratio $SUB_RATIO \
  --al_strategy diversity \
  --al_clusters 50 \
  --al_embedding_dim 256 \
  > logs/imagenet/AL_comparison_diversity_imagenet_${NUM_ADDED_IMG}_${SUB_RATIO}.log 2>&1 &

echo "=== Test 4: Hybrid AL ==="
CUDA_VISIBLE_DEVICES=0 python -u alpred.py \
  $COMMON_ARGS \
  --sub_ratio $SUB_RATIO \
  --al_strategy hybrid \
  --al_coreset_ratio 0.3 \
  --al_stratified_ratio 0.4 \
  --al_embedding_dim 256 \
  > logs/imagenet/AL_comparison_hybrid_imagenet_${NUM_ADDED_IMG}_${SUB_RATIO}.log 2>&1 &

echo "=== Test 5: Hierarchical AL (Default) ==="
CUDA_VISIBLE_DEVICES=1 python -u alpred.py \
  $COMMON_ARGS \
  --sub_ratio $SUB_RATIO \
  --al_strategy hierarchical \
  --al_embedding_dim 256 \
  > logs/imagenet/AL_comparison_hierarchical_imagenet_${NUM_ADDED_IMG}_${SUB_RATIO}.log 2>&1 &

echo "=== Test 6: Hierarchical AL (Optimized) ==="
CUDA_VISIBLE_DEVICES=2 python -u alpred.py \
  $COMMON_ARGS \
  --sub_ratio $SUB_RATIO \
  --al_strategy hierarchical \
  --al_embedding_dim 256 \
  --al_w_err 0.7 \
  --al_w_unc 0.3 \
  --al_alpha 0.3 \
  --al_beta 0.5 \
  --al_gamma 0.2 \
  --al_lambda_div 0.6 \
  --al_round1_ratio 0.6 \
  --al_min_slice_quota 3 \
  --al_max_slice_quota 30 \
  > logs/imagenet/AL_comparison_hierarchical_opt_imagenet_${NUM_ADDED_IMG}_${SUB_RATIO}.log 2>&1 &

echo ""
echo "All strategy comparison tests launched..."
echo "GPU allocation:"
echo "  GPU 0: Random baseline, then Hybrid AL"
echo "  GPU 1: Core-set AL, then Hierarchical AL (default)"
echo "  GPU 2: Stratified AL, then Hierarchical AL (optimized)"
echo "  GPU 3: Diversity AL"

# Wait for all background processes to finish
echo ""
echo "Waiting for all tests to complete..."
wait

echo ""
echo "All tests completed!"

# Generate comprehensive comparison report
echo ""
echo "=== COMPREHENSIVE COMPARISON REPORT ==="
echo "======================================="

declare -A results
strategies=("random" "coreset" "stratified" "diversity" "hybrid" "hierarchical" "hierarchical_opt")
strategy_names=("Random Baseline" "Core-set AL" "Stratified AL" "Diversity AL" "Hybrid AL" "Hierarchical AL (Default)" "Hierarchical AL (Optimized)")

echo ""
echo "Final Validation Accuracies:"
echo "----------------------------"

for i in "${!strategies[@]}"; do
    strategy="${strategies[$i]}"
    strategy_name="${strategy_names[$i]}"
    log_file="logs/imagenet/AL_comparison_${strategy}_imagenet_${NUM_ADDED_IMG}_${SUB_RATIO}.log"
    
    if [ -f "$log_file" ]; then
        final_acc=$(grep -o "Final Epoch Validation Accuracy: [0-9.]*" "$log_file" | tail -1 | cut -d' ' -f5)
        if [ ! -z "$final_acc" ]; then
            results["$strategy"]="$final_acc"
            printf "%-30s: %s\n" "$strategy_name" "$final_acc"
        else
            printf "%-30s: %s\n" "$strategy_name" "N/A"
        fi
    else
        printf "%-30s: %s\n" "$strategy_name" "Log not found"
    fi
done

echo ""
echo "Training Time and Selection Details:"
echo "-----------------------------------"

for i in "${!strategies[@]}"; do
    strategy="${strategies[$i]}"
    strategy_name="${strategy_names[$i]}"
    log_file="logs/imagenet/AL_comparison_${strategy}_imagenet_${NUM_ADDED_IMG}_${SUB_RATIO}.log"
    
    if [ -f "$log_file" ]; then
        printf "%-30s: " "$strategy_name"
        
        # Extract training time
        train_time=$(grep -o "Total training time.*: [0-9.]* seconds" "$log_file" | tail -1 | sed 's/.*: \([0-9.]*\) seconds/\1/')
        if [ ! -z "$train_time" ]; then
            printf "Train: %ss" "$train_time"
        fi
        
        # Extract AL selection time
        al_time=$(grep -o "Total Active Learning selection time: [0-9.]* seconds" "$log_file" | tail -1 | sed 's/.*: \([0-9.]*\) seconds/\1/')
        if [ ! -z "$al_time" ]; then
            printf ", AL: %ss" "$al_time"
        fi
        
        # Extract specific hierarchical info
        if [[ "$strategy" == *"hierarchical"* ]]; then
            round1_samples=$(grep -o "Round 1.*: [0-9]* samples" "$log_file" | sed 's/.*: \([0-9]*\) samples/\1/')
            round2_samples=$(grep -o "Round 2.*: [0-9]* samples" "$log_file" | sed 's/.*: \([0-9]*\) samples/\1/')
            if [ ! -z "$round1_samples" ] && [ ! -z "$round2_samples" ]; then
                printf ", R1: %s, R2: %s" "$round1_samples" "$round2_samples"
            fi
        fi
        
        echo ""
    fi
done

echo ""
echo "Performance Ranking (by Validation Accuracy):"
echo "---------------------------------------------"

# Create temporary file for sorting
temp_file=$(mktemp)
for strategy in "${strategies[@]}"; do
    if [ ! -z "${results[$strategy]}" ]; then
        echo "${results[$strategy]} $strategy" >> "$temp_file"
    fi
done

# Sort by accuracy (descending) and display
rank=1
sort -nr "$temp_file" | while read acc strategy; do
    strategy_name=""
    for i in "${!strategies[@]}"; do
        if [ "${strategies[$i]}" = "$strategy" ]; then
            strategy_name="${strategy_names[$i]}"
            break
        fi
    done
    printf "%d. %-30s: %s\n" "$rank" "$strategy_name" "$acc"
    ((rank++))
done

rm -f "$temp_file"

echo ""
echo "Log Files Summary:"
echo "-----------------"
for strategy in "${strategies[@]}"; do
    log_file="logs/imagenet/AL_comparison_${strategy}_imagenet_${NUM_ADDED_IMG}_${SUB_RATIO}.log"
    if [ -f "$log_file" ]; then
        size=$(wc -l < "$log_file")
        printf "%-30s: %d lines\n" "$(basename "$log_file")" "$size"
    fi
done

echo ""
echo "Recommendations:"
echo "---------------"
echo "1. Compare the final validation accuracies to identify the best-performing strategy"
echo "2. Consider training time vs. accuracy trade-offs"
echo "3. Hierarchical AL shows slice-level allocation details in verbose logs"
echo "4. Check prediction files in save/pred/ for detailed analysis"

echo ""
echo "Comparison complete!"