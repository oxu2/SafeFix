#!/bin/bash

# Quick Active Learning Test Script
# This script runs a fast test of Active Learning with minimal parameters

echo "================================="
echo "Quick Active Learning Test"
echo "================================="

# Create results directory
mkdir -p results/quick_test
mkdir -p logs/quick_test

# Quick test parameters
DATASET="imagenet"
EPOCHS=3
BATCH_SIZE=16
AL_ROUNDS=2
INITIAL_LABELED=100
SAMPLES_PER_ROUND=50

echo "Quick test parameters:"
echo "Dataset: $DATASET"
echo "Initial labeled samples: $INITIAL_LABELED"
echo "AL rounds: $AL_ROUNDS"
echo "Samples per round: $SAMPLES_PER_ROUND"
echo "Epochs per round: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo ""

# Function to handle file versioning - rename existing files with _old1, _old2, etc.
handle_file_versioning() {
    local target_file=$1
    
    if [ ! -f "$target_file" ]; then
        return 0  # File doesn't exist, no need to version
    fi
    
    # Extract directory, basename and extension
    local dir=$(dirname "$target_file")
    local filename=$(basename "$target_file")
    local name_without_ext="${filename%.*}"
    local ext="${filename##*.}"
    
    # Find the next available old version number
    local version=1
    while [ -f "${dir}/${name_without_ext}_old${version}.${ext}" ]; do
        version=$((version + 1))
    done
    
    # Rename existing file to old version
    local old_filename="${dir}/${name_without_ext}_old${version}.${ext}"
    mv "$target_file" "$old_filename"
    echo "  Renamed existing $target_file to $old_filename"
}

# Function to run quick experiment
run_quick_test() {
    local model=$1
    local run_name=$2
    
    echo "----------------------------------------"
    echo "Quick test: $run_name"
    echo "Model: $model"
    echo "----------------------------------------"
    
    log_file="logs/quick_test/${run_name}_$(date +%Y%m%d_%H%M%S).log"
    
    python active_learning_coreset.py \
        --dataset $DATASET \
        --model $model \
        --verbose \
        --al-rounds $AL_ROUNDS \
        --initial-labeled $INITIAL_LABELED \
        --samples-per-round $SAMPLES_PER_ROUND \
        --num-epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        2>&1 | tee "$log_file"
    
    if [ $? -eq 0 ]; then
        echo "✅ $run_name completed successfully"
        # Move results to organized folders with new naming convention
        # New format: al_results_{model}_{dataset}_nopretrained.png
        results_file="al_results_${model}_${DATASET}_nopretrained.png"
        history_file="al_history_${model}_${DATASET}_nopretrained.json"
        
        # Define target paths
        target_results="results/quick_test/${run_name}_results.png"
        target_history="results/quick_test/${run_name}_history.json"
        
        if [ -f "$results_file" ]; then
            # Handle versioning for target file
            handle_file_versioning "$target_results"
            mv "$results_file" "$target_results"
            echo "  Moved $results_file to $target_results"
        else
            echo "  Warning: $results_file not found"
        fi
        
        if [ -f "$history_file" ]; then
            # Handle versioning for target file
            handle_file_versioning "$target_history"
            mv "$history_file" "$target_history"
            echo "  Moved $history_file to $target_history"
        else
            echo "  Warning: $history_file not found"
        fi
    else
        echo "❌ $run_name failed"
        echo "Check log: $log_file"
    fi
    echo ""
}

# Quick tests
echo "Running quick tests..."
run_quick_test "resnet" "resnet_quick"
run_quick_test "clip" "clip_quick"

echo ""
echo "================================="
echo "Quick tests completed!"
echo "================================="
echo ""
echo "Results saved in: results/quick_test/"
echo "Logs saved in: logs/quick_test/"
echo ""

# Generate summary
echo "Quick test summary:"
python -c "
import os
import json
import glob

history_files = glob.glob('results/quick_test/*_history.json')
for file in sorted(history_files):
    try:
        with open(file, 'r') as f:
            data = json.load(f)
        name = os.path.basename(file).replace('_history.json', '')
        final_train_acc = data['train_acc'][-1] if data['train_acc'] else 'N/A'
        final_val_acc = data['val_acc'][-1] if data['val_acc'] else 'N/A'
        rounds = len(data['train_acc'])
        print(f'{name:15} | Rounds: {rounds} | Train: {final_train_acc:6.2f}% | Val: {final_val_acc:6.2f}%')
    except Exception as e:
        print(f'{name:15} | ERROR: {str(e)}')
"
