#!/bin/bash

# Test script for Hierarchical Active Learning
# Tests the new hierarchical two-stage AL method with parallel GPU execution

echo "Testing Hierarchical Active Learning Implementation"
echo "================================================="

# Parse command line arguments for hierarchical parameters
SUB_RATIO=0.9  # Default value
EMBEDDING_DIM=256  # Default value
while [[ $# -gt 0 ]]; do
    case $1 in
        --sub_ratio)
            SUB_RATIO="$2"
            shift 2
            ;;
        --embedding_dim)
            EMBEDDING_DIM="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "Using sub_ratio: $SUB_RATIO"
echo "Using embedding_dim: $EMBEDDING_DIM"

# Common parameters following test_combined_al.sh pattern
COMMON_ARGS="--dataset imagenet --model resnet --num_added_img 1000 --attributes pinkcolor_fabrictexture --num_epochs 30 --verbose True"

echo ""
echo "=== Test 0: Hierarchical AL with Default Parameters ==="
CUDA_VISIBLE_DEVICES=0 python -u alpred.py \
  $COMMON_ARGS \
  --sub_ratio $SUB_RATIO \
  --al_strategy hierarchical \
  --al_embedding_dim $EMBEDDING_DIM \
  > logs_imagenet/AL_hierarchical_default_imagenet_1000_${SUB_RATIO}.log 2>&1 &

echo "=== Test 1: Hierarchical AL with High Error Weight ==="
CUDA_VISIBLE_DEVICES=1 python -u alpred.py \
  $COMMON_ARGS \
  --sub_ratio $SUB_RATIO \
  --al_strategy hierarchical \
  --al_embedding_dim $EMBEDDING_DIM \
  --al_w_err 0.8 \
  --al_w_unc 0.2 \
  --al_alpha 0.5 \
  --al_beta 0.3 \
  --al_gamma 0.2 \
  > logs_imagenet/AL_hierarchical_higherr_imagenet_1000_${SUB_RATIO}.log 2>&1 &

echo "=== Test 2: Hierarchical AL with High Coverage Weight ==="
CUDA_VISIBLE_DEVICES=2 python -u alpred.py \
  $COMMON_ARGS \
  --sub_ratio $SUB_RATIO \
  --al_strategy hierarchical \
  --al_embedding_dim $EMBEDDING_DIM \
  --al_w_err 0.4 \
  --al_w_unc 0.6 \
  --al_alpha 0.2 \
  --al_beta 0.6 \
  --al_gamma 0.2 \
  --al_lambda_div 0.6 \
  > logs_imagenet/AL_hierarchical_highcov_imagenet_1000_${SUB_RATIO}.log 2>&1 &

echo "=== Test 3: Hierarchical AL with Balanced Round Allocation ==="
CUDA_VISIBLE_DEVICES=3 python -u alpred.py \
  $COMMON_ARGS \
  --sub_ratio $SUB_RATIO \
  --al_strategy hierarchical \
  --al_embedding_dim $EMBEDDING_DIM \
  --al_round1_ratio 0.5 \
  --al_min_slice_quota 3 \
  --al_max_slice_quota 30 \
  --al_lambda_div 0.8 \
  > logs_imagenet/AL_hierarchical_balanced_imagenet_1000_${SUB_RATIO}.log 2>&1 &

echo ""
echo "All hierarchical AL tests launched in parallel..."
echo "GPU 0: Default parameters"
echo "GPU 1: High error weight (w_err=0.8)"
echo "GPU 2: High coverage weight (beta=0.6)"
echo "GPU 3: Balanced rounds (round1_ratio=0.5)"

# Wait for all background processes to finish
echo ""
echo "Waiting for all tests to complete..."
wait

echo ""
echo "Tests completed. Check logs in logs_imagenet/AL_hierarchical_*.log"
echo "Look for lines like 'Hierarchical AL selected X samples total'"

# Check if all log files were created
echo ""
echo "=== Log Files Summary ==="
for log_file in logs_imagenet/AL_hierarchical_*_imagenet_1000_${SUB_RATIO}.log; do
    if [ -f "$log_file" ]; then
        size=$(wc -l < "$log_file")
        echo "✓ $(basename "$log_file"): $size lines"
        
        # Extract final validation accuracy if available
        final_acc=$(grep -o "Final Epoch Validation Accuracy: [0-9.]*" "$log_file" | tail -1 | cut -d' ' -f5)
        if [ ! -z "$final_acc" ]; then
            echo "  Final Val Accuracy: $final_acc"
        fi
        
        # Check for hierarchical-specific output
        slice_info=$(grep -c "selected.*samples" "$log_file")
        if [ $slice_info -gt 0 ]; then
            echo "  Slice selection lines: $slice_info"
        fi
    else
        echo "✗ $(basename "$log_file"): Not found"
    fi
done

# Optional: Check if prediction files were generated
echo ""
echo "=== Prediction Files Summary ==="
PRED_FILES=$(find save/pred/ -name "*hierarchical*" -type f -newer /tmp 2>/dev/null | wc -l)
if [ $PRED_FILES -gt 0 ]; then
    echo "✓ Found $PRED_FILES prediction files from hierarchical AL"
    echo "Latest prediction files:"
    find save/pred/ -name "*hierarchical*" -type f -exec ls -la {} \; | tail -3
else
    echo "⚠ No new prediction files found with 'hierarchical' in name"
fi

echo ""
echo "=== Performance Comparison ==="
echo "Compare final validation accuracies across different hierarchical AL configurations:"
echo ""
for log_file in logs_imagenet/AL_hierarchical_*_imagenet_1000_${SUB_RATIO}.log; do
    if [ -f "$log_file" ]; then
        config=$(basename "$log_file" | cut -d'_' -f3)
        final_acc=$(grep -o "Final Epoch Validation Accuracy: [0-9.]*" "$log_file" | tail -1 | cut -d' ' -f5)
        if [ ! -z "$final_acc" ]; then
            printf "%-15s: %s\n" "$config" "$final_acc"
        else
            printf "%-15s: %s\n" "$config" "N/A"
        fi
    fi
done

echo ""
echo "Testing complete!"
echo "All hierarchical AL configurations have been tested in parallel."