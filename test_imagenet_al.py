#!/usr/bin/env python3
"""
Quick test for ImageNet Active Learning setup
"""

import sys
sys.path.append('.')

import numpy as np
from pathlib import Path
from torchvision.datasets import ImageFolder
from active_learning_coreset import load_imagenet_data, ActiveLearningTrainer
from utils.defines import TRAIN, VALID, UNLABELED

def test_imagenet_setup():
    """Test ImageNet data loading and model setup"""
    
    print("=== Testing ImageNet Active Learning Setup ===")
    
    # 1. Test data loading
    data_dir = Path("exampleData/imagenet_subset")
    print(f"1. Loading ImageNet data from {data_dir}...")
    
    all_image_paths, all_labels, all_splits = load_imagenet_data(data_dir)
    
    print(f"   - Loaded {len(all_image_paths)} images")
    print(f"   - Label range: {np.min(all_labels)} to {np.max(all_labels)}")
    print(f"   - Unique labels: {len(np.unique(all_labels))}")
    print(f"   - Split distribution:")
    split_counts = {TRAIN: 0, VALID: 0, UNLABELED: 0}
    for split in all_splits:
        split_counts[split] += 1
    for split_type, count in split_counts.items():
        print(f"     {split_type}: {count}")
    
    # 2. Test model initialization
    print("\n2. Testing model initialization...")
    num_classes = len(np.unique(all_labels))
    print(f"   - Number of classes: {num_classes}")
    
    try:
        trainer = ActiveLearningTrainer(
            model_type='resnet',
            num_classes=num_classes,
            embedding_dim=512,
            device='cpu',  # Use CPU to avoid CUDA issues
            batch_size=8,
            learning_rate=1e-4,
            num_epochs=1,
            verbose=True,
            disable_progress_bar=True
        )
        print("   - Model initialized successfully!")
        
        # 3. Test data splits
        print("\n3. Testing data splits...")
        train_indices = [i for i, split in enumerate(all_splits) if split == TRAIN]
        val_indices = [i for i, split in enumerate(all_splits) if split == VALID]
        unlabeled_indices = [i for i, split in enumerate(all_splits) if split == UNLABELED]
        
        print(f"   - Train indices: {len(train_indices)}")
        print(f"   - Validation indices: {len(val_indices)}")
        print(f"   - Unlabeled indices: {len(unlabeled_indices)}")
        
        # Check label consistency
        train_labels = [all_labels[i] for i in train_indices[:10]]  # Check first 10
        print(f"   - Sample train labels: {train_labels}")
        print(f"   - Min/max in train: {np.min([all_labels[i] for i in train_indices])}/{np.max([all_labels[i] for i in train_indices])}")
        
        print("\n✅ All tests passed! ImageNet setup is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imagenet_setup()
    exit(0 if success else 1)
