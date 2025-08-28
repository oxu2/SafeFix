#!/bin/bash

# Active Learning Experiments Runner
# This script runs comprehensive Active Learning experiments with different configurations

echo "================================="
echo "Active Learning Experiments"
echo "================================="

# Create results directory
mkdir -p results/active_learning
mkdir -p logs/active_learning

# Set common parameters
DATASET="imagenet"
EPOCHS=10
BATCH_SIZE=32
AL_ROUNDS=5
INITIAL_LABELED=1000
SAMPLES_PER_ROUND=500

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

echo ""
echo "Starting Active Learning experiments..."
echo "Dataset: $DATASET"
echo "Initial labeled samples: $INITIAL_LABELED"
echo "AL rounds: $AL_ROUNDS"
echo "Samples per round: $SAMPLES_PER_ROUND"
echo "Epochs per round: $EPOCHS"
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

# Function to run experiment with error handling
run_experiment() {
    local model=$1
    local run_name=$2
    local extra_args=$3
    
    echo "----------------------------------------"
    echo "Running: $run_name"
    echo "Model: $model"
    echo "Extra args: $extra_args"
    echo "----------------------------------------"
    
    log_file="logs/active_learning/${run_name}_$(date +%Y%m%d_%H%M%S).log"
    
    python active_learning_coreset.py \
        --dataset $DATASET \
        --model $model \
        --verbose \
        --al-rounds $AL_ROUNDS \
        --initial-labeled $INITIAL_LABELED \
        --samples-per-round $SAMPLES_PER_ROUND \
        --num-epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        $extra_args \
        2>&1 | tee "$log_file"
    
    if [ $? -eq 0 ]; then
        echo "✅ $run_name completed successfully"
        # Move results to organized folders with new naming convention
        # New format: al_results_{model}_{dataset}_nopretrained.png
        results_file="al_results_${model}_${DATASET}_nopretrained.png"
        history_file="al_history_${model}_${DATASET}_nopretrained.json"
        
        # Define target paths
        target_results="results/active_learning/${run_name}_results.png"
        target_history="results/active_learning/${run_name}_history.json"
        
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

# Experiment 1: ResNet baseline
run_experiment "resnet" "resnet_baseline" ""

# Experiment 2: CLIP baseline  
run_experiment "clip" "clip_baseline" ""

# # Experiment 3: ResNet with different AL parameters
# echo "Running ResNet with different parameters..."
# run_experiment "resnet" "resnet_small_init" "--initial-labeled 500"
# run_experiment "resnet" "resnet_large_rounds" "--al-rounds 8 --samples-per-round 300"

# # Experiment 4: CLIP with different AL parameters
# echo "Running CLIP with different parameters..."
# run_experiment "clip" "clip_small_init" "--initial-labeled 500"
# run_experiment "clip" "clip_large_rounds" "--al-rounds 8 --samples-per-round 300"

# # Experiment 5: Quick test runs with fewer epochs for comparison
# echo "Running quick comparison experiments..."
# run_experiment "resnet" "resnet_quick" "--num-epochs 5 --al-rounds 3"
# run_experiment "clip" "clip_quick" "--num-epochs 5 --al-rounds 3"

# # Experiment 6: CPU fallback test (in case GPU fails)
# echo "Running CPU fallback test..."
# run_experiment "resnet" "resnet_cpu" "--device cpu --batch-size 16 --num-epochs 3 --al-rounds 2"

echo ""
echo "================================="
echo "All experiments completed!"
echo "================================="
echo ""
echo "Results saved in: results/active_learning/"
echo "Logs saved in: logs/active_learning/"
echo ""

# Generate summary
echo "Generating experiment summary..."
python -c "
import os
import json
import glob

print('\\n=== EXPERIMENT SUMMARY ===')
history_files = glob.glob('results/active_learning/*_history.json')
for file in sorted(history_files):
    try:
        with open(file, 'r') as f:
            data = json.load(f)
        name = os.path.basename(file).replace('_history.json', '')
        final_train_acc = data['train_acc'][-1] if data['train_acc'] else 'N/A'
        final_val_acc = data['val_acc'][-1] if data['val_acc'] else 'N/A'
        rounds = len(data['train_acc'])
        print(f'{name:20} | Rounds: {rounds:2} | Train: {final_train_acc:6.2f}% | Val: {final_val_acc:6.2f}%')
    except Exception as e:
        print(f'{name:20} | ERROR: {str(e)}')
print('==========================\\n')
"

echo "To view detailed results:"
# echo "  - Images: ls results/active_learning/*.png"
# echo "  - History: ls results/active_learning/*.json" 
# echo "  - Logs: ls logs/active_learning/*.log"
# echo ""
echo "To compare results:"
echo "  python -c \"import json; [print(f'{f}: {json.load(open(f))[\"val_acc\"][-1]:.2f}%') for f in sorted(__import__('glob').glob('results/active_learning/*_history.json'))]\""
    # echo "GPU Information:"
    # nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    # DEVICE="cuda"
else
    echo "No GPU detected, using CPU"
    DEVICE="cpu"
fi

echo "======================================"

# Create output directories
mkdir -p logs/active_learning
mkdir -p results/active_learning

# Run Active Learning
python active_learning_coreset.py 
    --model $MODEL 
    --batch-size 32 
    --learning-rate 1e-4 
    --num-epochs 10 
    --initial-labeled 1000 
    --al-rounds $ROUNDS 
    --samples-per-round 500 
    --device $DEVICE 
    --quiet 
    2>&1 | tee logs/active_learning/al_${STRATEGY}_${MODEL}_$(date +%Y%m%d_%H%M%S).log

echo "======================================"
echo "Training completed!"
echo "Check the following files:"
echo "- Logs: logs/active_learning/"
echo "- Results plot: al_results_${MODEL}.png"
echo "- History: al_history_${MODEL}.json"
echo "- Model: best_model.pth"
echo "======================================"
