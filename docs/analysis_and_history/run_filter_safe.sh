#!/bin/bash

# Script to run filter.py with proper memory management
# Usage: ./run_filter_safe.sh [attributes] [max_memory_gb] [stop_at]

set -e  # Exit on any error

# Default values
ATTRIBUTES=${1:-"redhair_brownskin_sademotion"}
MAX_MEMORY=${2:-18}  # Reduced from 23 to 18GB for safety
STOP_AT=${3:-5000}   # Reduced from 15000 to 5000 for initial run

# Set environment variables to prevent issues
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1

echo "🚀 Starting filter.py with improved memory management"
echo "📊 Configuration:"
echo "   • Attributes: $ATTRIBUTES"
echo "   • Max Memory per GPU: ${MAX_MEMORY}GB"
echo "   • Stop after: $STOP_AT images"
echo "   • Memory allocation strategy: Conservative"
echo ""

# Check available GPU memory first
if command -v nvidia-smi &> /dev/null; then
    echo "🔍 GPU Memory Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
    echo ""
fi

# Run the filter script with conservative settings
python filter.py \
    --attributes "$ATTRIBUTES" \
    --max_memory "$MAX_MEMORY" \
    --stop_at "$STOP_AT" \
    --verbose true \
    --dataset celeba \
    --file_type "jpg,jpeg" 2>&1 | tee "logs/filter_${ATTRIBUTES}_$(date +%m%d_%H%M).log"

echo ""
echo "✅ Filter.py execution completed successfully!"
echo "📁 Check the logs/ directory for detailed output logs"
echo "📁 Check the save/json/ directory for results and statistics"
