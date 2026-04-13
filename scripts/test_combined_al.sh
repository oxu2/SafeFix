#!/bin/bash

echo "Testing Combined Active Learning approach..."

# Parse command line arguments for combined_ratio
COMBINED_RATIO=0.7  # Default value
while [[ $# -gt 0 ]]; do
    case $1 in
        --combined_ratio)
            COMBINED_RATIO="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "Using combined_ratio: $COMBINED_RATIO"

# Common parameters
COMMON_ARGS="--dataset imagenet --model resnet --num_added_img 1000 --attributes pinkcolor_fabrictexture --num_epochs 30 --verbose True"

echo "=== Test 0: Combined Core-set AL ==="
CUDA_VISIBLE_DEVICES=0 python -u alpred.py \
  $COMMON_ARGS \
  --al_combined True \
  --combined_ratio $COMBINED_RATIO \
  --al_strategy coreset \
  --al_embedding_dim 256 \
  > logs_imagenet/AL_combined_coreset_imagenet_1000_${COMBINED_RATIO}.log 2>&1 &

echo "=== Test 1: Combined Stratified AL ==="
CUDA_VISIBLE_DEVICES=1 python -u alpred.py \
  $COMMON_ARGS \
  --al_combined True \
  --combined_ratio $COMBINED_RATIO \
  --al_strategy stratified \
  > logs_imagenet/AL_combined_stratified_imagenet_1000_${COMBINED_RATIO}.log 2>&1 &

echo "=== Test 2: Combined Diversity AL ==="
CUDA_VISIBLE_DEVICES=2 python -u alpred.py \
  $COMMON_ARGS \
  --al_combined True \
  --combined_ratio $COMBINED_RATIO \
  --al_strategy diversity \
  --al_clusters 50 \
  --al_embedding_dim 256 \
  > logs_imagenet/AL_combined_diversity_imagenet_1000_${COMBINED_RATIO}.log 2>&1 &

echo "=== Test 3: Combined Hybrid AL ==="
CUDA_VISIBLE_DEVICES=3 python -u alpred.py \
  $COMMON_ARGS \
  --al_combined True \
  --combined_ratio $COMBINED_RATIO \
  --al_strategy hybrid \
  --al_coreset_ratio 0.3 \
  --al_stratified_ratio 0.4 \
  --al_clusters 50 \
  --al_embedding_dim 256 \
  > logs_imagenet/AL_combined_hybrid_imagenet_1000_${COMBINED_RATIO}.log 2>&1 &

# Wait for all background processes to finish
wait

echo "Tests completed. Check logs in logs_imagenet/AL_combined_*.log"
echo "Look for lines like 'AL selected from combined dataset: X original + Y generated = Z total'"