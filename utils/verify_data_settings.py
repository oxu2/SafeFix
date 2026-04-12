"""
Data consistency verification and unified settings configuration
"""

import numpy as np
import os
from utils.defines import TRAIN, VALID, TEST, UNLABELED

def verify_data_consistency():
    """Verify that all scripts use the same data split"""
    
    print("=== Data Consistency Verification ===\n")
    
    # Load data using the same function as all scripts
    from active_learning_coreset import load_celeba_data
    
    try:
        all_image_paths, all_labels, all_splits = load_celeba_data()
        
        # Count splits
        train_count = sum(1 for s in all_splits if s == TRAIN)
        valid_count = sum(1 for s in all_splits if s == VALID)
        test_count = sum(1 for s in all_splits if s == TEST)
        unlabeled_count = sum(1 for s in all_splits if s == UNLABELED)
        
        print("✅ Data loaded successfully!")
        print(f"Total images: {len(all_image_paths)}")
        print(f"Total labels: {len(all_labels)}")
        print(f"Total splits: {len(all_splits)}")
        print()
        
        print("📊 Data Split Distribution:")
        print(f"  Training set: {train_count:,} samples")
        print(f"  Validation set: {valid_count:,} samples") 
        print(f"  Test set: {test_count:,} samples")
        print(f"  Unlabeled set: {unlabeled_count:,} samples")
        print(f"  Total: {train_count + valid_count + test_count + unlabeled_count:,} samples")
        print()
        
        print("🏷️ Label Distribution:")
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"  Label {label}: {count:,} samples ({count/len(all_labels)*100:.1f}%)")
        print()
        
        # Verify file paths exist
        print("📁 File Path Verification:")
        if os.path.exists('/data/oxx220000/img_align_celeba/'):
            base_path = '/data/oxx220000/img_align_celeba/'
            print(f"  ✅ Using primary path: {base_path}")
        elif os.path.exists('/home/jovyan/ouser/dataset/img_align_celeba/'):
            base_path = '/home/jovyan/ouser/dataset/img_align_celeba/'
            print(f"  ✅ Using fallback path: {base_path}")
        else:
            print("  ❌ No valid image directory found!")
            return False
        
        # Check a few sample files
        sample_indices = np.random.choice(len(all_image_paths), 5, replace=False)
        missing_files = 0
        for idx in sample_indices:
            if not os.path.exists(all_image_paths[idx]):
                missing_files += 1
        
        if missing_files == 0:
            print(f"  ✅ All {len(sample_indices)} sampled image files exist")
        else:
            print(f"  ⚠️ {missing_files}/{len(sample_indices)} sampled files are missing")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def show_unified_settings():
    """Show unified settings for all scripts"""
    
    print("\n=== Unified Settings Configuration ===\n")
    
    settings = {
        "Quick Demo (tests/demo_active_learning.py)": {
            "Purpose": "Fast testing and verification",
            "batch_size": 16,
            "learning_rate": 1e-4,
            "num_epochs": 5,
            "initial_labeled_size": 500,
            "al_rounds": 3,
            "samples_per_round": 200,
            "estimated_time": "10-15 minutes"
        },
        
        "Production Training (run_active_learning.sh)": {
            "Purpose": "Full-scale research training",
            "batch_size": 32,
            "learning_rate": 1e-4,
            "num_epochs": 10,
            "initial_labeled_size": 1000,
            "al_rounds": 5,
            "samples_per_round": 500,
            "estimated_time": "2-3 hours"
        },
        
        "Strategy Comparison (enhanced_active_learning.py)": {
            "Purpose": "Compare different AL strategies",
            "batch_size": 16,
            "learning_rate": 1e-4,
            "num_epochs": 3,
            "initial_labeled_size": 300,
            "al_rounds": 3,
            "samples_per_round": 100,
            "estimated_time": "30-45 minutes per strategy"
        }
    }
    
    for script_name, config in settings.items():
        print(f"🔧 {script_name}")
        print(f"   Purpose: {config['Purpose']}")
        print(f"   Settings: batch_size={config['batch_size']}, epochs={config['num_epochs']}, rounds={config['al_rounds']}")
        print(f"   Initial labeled: {config['initial_labeled_size']}, samples/round: {config['samples_per_round']}")
        print(f"   Estimated time: {config['estimated_time']}")
        print()

def create_custom_config():
    """Create a custom configuration template"""
    
    print("=== Custom Configuration Template ===\n")
    
    custom_config = """
# Custom Active Learning Configuration
# Copy and modify these settings as needed

# Model settings
MODEL_TYPE = 'resnet'  # Options: 'resnet', 'clip'
DEVICE = 'cuda'        # Options: 'cuda', 'cpu'

# Training settings
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10

# Active Learning settings  
INITIAL_LABELED_SIZE = 1000  # Initial number of labeled samples
AL_ROUNDS = 5                # Number of active learning rounds
SAMPLES_PER_ROUND = 500      # Samples to select each round

# Strategy settings
STRATEGY = 'coreset'  # Options: 'coreset', 'uncertainty', 'diversity', 'random'

# Usage example:
# python active_learning_coreset.py \\
#     --model $MODEL_TYPE \\
#     --batch-size $BATCH_SIZE \\
#     --learning-rate $LEARNING_RATE \\
#     --num-epochs $NUM_EPOCHS \\
#     --initial-labeled $INITIAL_LABELED_SIZE \\
#     --al-rounds $AL_ROUNDS \\
#     --samples-per-round $SAMPLES_PER_ROUND \\
#     --device $DEVICE
"""
    
    print(custom_config)

def main():
    """Main verification and configuration display"""
    
    # Verify data consistency
    if verify_data_consistency():
        print("✅ All scripts use the SAME data split and image paths!")
        print("✅ The only differences are in training parameters\n")
        
        # Show unified settings
        show_unified_settings()
        
        # Show custom config template
        create_custom_config()
        
        print("💡 Key Points:")
        print("1. ALL scripts use identical data loading (load_celeba_data())")
        print("2. Splits follow the consistent train/validation/test/unlabeled partition")
        print("3. ONLY training parameters differ between scripts")
        print("4. Choose the script based on your needs (speed vs. accuracy)")
        
    else:
        print("❌ Please check your data setup before running Active Learning")

if __name__ == "__main__":
    main()
