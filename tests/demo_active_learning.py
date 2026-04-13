import numpy as np
import torch
import os

# Disable progress bars by default
os.environ['TQDM_DISABLE'] = '1'

# Try to disable tqdm if it's imported
try:
    from tqdm import tqdm
    tqdm.disable = True
except ImportError:
    pass

# Try to disable transformers progress bars
try:
    import transformers
    transformers.logging.set_verbosity_error()
except ImportError:
    pass

from utils.active_learning_coreset import ActiveLearningTrainer, load_celeba_data
import matplotlib.pyplot as plt

def quick_demo():
    """Quick demonstration of Active Learning"""
    
    print("=== Active Learning Core-set Demo ===")
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Load data
        print("Loading CelebA data...")
        all_image_paths, all_labels, all_splits = load_celeba_data()
        print(f"Successfully loaded {len(all_image_paths)} images")
        print(f"Label distribution: {np.bincount(all_labels)}")
        
        # Data statistics
        from utils.defines import TRAIN, VALID, UNLABELED
        train_count = sum(1 for s in all_splits if s == TRAIN)
        valid_count = sum(1 for s in all_splits if s == VALID)  
        unlabeled_count = sum(1 for s in all_splits if s == UNLABELED)
        
        print(f"Data distribution:")
        print(f"  Training set: {train_count}")
        print(f"  Validation set: {valid_count}")
        print(f"  Unlabeled set: {unlabeled_count}")
        
        # Initialize Active Learning trainer
        print("\nInitializing Active Learning trainer...")
        trainer = ActiveLearningTrainer(
            model_type='resnet',  # Options: 'resnet', 'clip'
            num_classes=2,
            embedding_dim=512,
            device=device,
            batch_size=16,  # Smaller batch size for demo
            learning_rate=1e-4,
            num_epochs=5,  # Fewer epochs for quick demo
            verbose=True,
            disable_progress_bar=True  # Disable progress bars by default
        )
        
        # Run Active Learning (small-scale demo)
        print("\nStarting Active Learning training...")
        history = trainer.active_learning_loop(
            all_image_paths=all_image_paths,
            all_labels=all_labels,
            all_splits=all_splits,
            initial_labeled_size=500,  # Initial labeled samples
            al_rounds=3,  # Active Learning rounds
            samples_per_round=200  # Samples selected per round
        )
        
        # Display results
        print("\n=== Active Learning Results ===")
        for i, (train_acc, val_acc) in enumerate(zip(history['train_acc'], history['val_acc'])):
            print(f"Round {i+1}: Train Acc {train_acc:.2f}%, Val Acc {val_acc:.2f}%")
        
        # Plot results
        trainer.plot_results('demo_results.png')
        
        return history
        
    except Exception as e:
        print(f"Error during demo: {e}")
        print("Please ensure:")
        print("1. CelebA data is properly downloaded and configured")
        print("2. data/celebA/ directory contains required .npy and .txt files")
        print("3. Sufficient GPU memory (if using GPU)")
        return None

def compare_strategies_demo():
    """Demo for comparing different strategies"""
    
    print("\n=== Strategy Comparison Demo ===")
    
    strategies = ['random', 'uncertainty', 'coreset']
    results = {}
    
    # This can be extended to implement different strategies
    print("Note: Complete strategy comparison requires more implementation")
    print("Currently implemented Core-set strategy")
    print("Can be extended to add:")
    print("- Random sampling")
    print("- Uncertainty sampling")
    print("- Diversity sampling, etc.")

def analyze_selected_samples(history):
    """Analyze selected samples"""
    
    if not history or 'selected_samples' not in history:
        print("No sample selection history to analyze")
        return
    
    print("\n=== Sample Selection Analysis ===")
    
    for round_num, selected in enumerate(history['selected_samples']):
        print(f"Round {round_num + 2}: Selected {len(selected)} samples")
        print(f"  Sample indices example: {selected[:10]}...")
    
    # Can add more analysis
    # - Class distribution of selected samples
    # - Feature distribution of selected samples
    # - Comparison with random selection, etc.

if __name__ == "__main__":
    # Run quick demo
    history = quick_demo()
    
    if history:
        # Analyze results
        analyze_selected_samples(history)
        
        # Strategy comparison demo
        compare_strategies_demo()
        
        print("\nDemo completed!")
        print("Check generated files:")
        print("- demo_results.png (results plot)")
        print("- best_model.pth (best model)")
    else:
        print("Demo failed, please check data and environment configuration")
