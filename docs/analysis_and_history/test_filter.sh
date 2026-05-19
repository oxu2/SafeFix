#!/bin/bash

# Test script for filter.py with very small batch to check for errors
# Usage: ./test_filter.sh

set -e

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "🧪 Testing filter.py with small batch (10 images)..."

# Test with very small parameters
python filter.py \
    --attributes "redhair_brownskin_sademotion" \
    --max_memory 18 \
    --stop_at 10 \
    --verbose true \
    --dataset celeba \
    --file_type "jpg,jpeg"

echo "✅ Test completed successfully!"
