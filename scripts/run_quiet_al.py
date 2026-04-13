#!/usr/bin/env python3
"""
Quiet Active Learning Runner - No progress bars, minimal output
"""

import os
import sys

# Set environment variables to disable progress bars BEFORE importing anything else
os.environ['TQDM_DISABLE'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Disable warnings
import warnings
warnings.filterwarnings('ignore')

# Now import the main modules
from utils.active_learning_coreset import ActiveLearningTrainer, load_celeba_data
import torch

def run_quiet_active_learning():
    """Run Active Learning with minimal output"""
    
    # Configuration
    config = {
        'model_type': 'resnet',  # or 'clip'
        'num_classes': 2,
        'embedding_dim': 512,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 10,
        'initial_labeled': 1000,
        'al_rounds': 5,
        'samples_per_round': 500,
        'verbose': False,  # Minimal output
        'disable_progress_bar': True
    }
    
    print("Starting Quiet Active Learning...")
    print(f"Device: {config['device']}")
    print(f"Model: {config['model_type']}")
    print(f"AL Rounds: {config['al_rounds']}")
    print("-" * 50)
    
    # Load data
    all_image_paths, all_labels, all_splits = load_celeba_data()
    print(f"Loaded {len(all_image_paths)} images")
    
    # Initialize trainer
    trainer = ActiveLearningTrainer(
        model_type=config['model_type'],
        num_classes=config['num_classes'],
        embedding_dim=config['embedding_dim'],
        device=config['device'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        num_epochs=config['num_epochs'],
        verbose=config['verbose'],
        disable_progress_bar=config['disable_progress_bar']
    )
    
    # Run Active Learning
    history = trainer.active_learning_loop(
        all_image_paths=all_image_paths,
        all_labels=all_labels,
        all_splits=all_splits,
        initial_labeled_size=config['initial_labeled'],
        al_rounds=config['al_rounds'],
        samples_per_round=config['samples_per_round']
    )
    
    # Summary
    print("\n" + "=" * 50)
    print("ACTIVE LEARNING COMPLETED")
    print("=" * 50)
    
    if history['val_acc']:
        print(f"Final Validation Accuracy: {history['val_acc'][-1]:.2f}%")
        print(f"Best Validation Accuracy: {max(history['val_acc']):.2f}%")
    
    print(f"Total AL Rounds: {len(history['al_rounds'])}")
    print(f"Results saved to: al_results_{config['model_type']}.png")
    print(f"History saved to: al_history_{config['model_type']}.json")
    
    return history

if __name__ == "__main__":
    try:
        history = run_quiet_active_learning()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during training: {e}")
        sys.exit(1)
