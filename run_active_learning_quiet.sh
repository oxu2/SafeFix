#!/bin/bash

# Quiet Active Learning execution script
# Usage: ./run_active_learning_quiet.sh [strategy] [model] [rounds]

# Set default parameters
STRATEGY=${1:-coreset}
MODEL=${2:-resnet}
ROUNDS=${3:-5}

echo "Starting Active Learning training..."
echo "Strategy: $STRATEGY | Model: $MODEL | Rounds: $ROUNDS"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    DEVICE="cuda"
else
    DEVICE="cpu"
fi

# Create output directories
mkdir -p logs/active_learning

# Run Active Learning with minimal output
python active_learning_coreset.py \
    --model $MODEL \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --num-epochs 10 \
    --initial-labeled 1000 \
    --al-rounds $ROUNDS \
    --samples-per-round 500 \
    --device $DEVICE \
    --quiet \
    > logs/active_learning/al_${STRATEGY}_${MODEL}_$(date +%Y%m%d_%H%M%S).log 2>&1

echo "Training completed! Check logs/active_learning/ for results."
