#!/usr/bin/env python3
"""
Split ImageNet validation set into new validation and test sets.
Randomly splits the original val_idx_imagenet.npy into two halves.
"""

import numpy as np
import os
from pathlib import Path

def split_imagenet_validation(seed=42):
    """Split ImageNet validation set into new validation and test sets."""

    save_dir = Path('save')
    if not save_dir.exists():
        print(f"Creating save directory: {save_dir}")
        save_dir.mkdir(parents=True, exist_ok=True)

    # Load original indices
    train_idx_path = save_dir / 'train_idx_imagenet.npy'
    val_idx_path = save_dir / 'val_idx_imagenet.npy'

    if not train_idx_path.exists():
        raise FileNotFoundError(f"Train indices file not found: {train_idx_path}")
    if not val_idx_path.exists():
        raise FileNotFoundError(f"Validation indices file not found: {val_idx_path}")

    print("Loading original ImageNet indices...")
    train_idx = np.load(train_idx_path)
    val_idx = np.load(val_idx_path)

    print(f"Original train set size: {len(train_idx)}")
    print(f"Original validation set size: {len(val_idx)}")

    # Split validation set into new validation and test sets
    np.random.seed(seed)  # Use the same seed for reproducibility
    val_indices = np.array(val_idx)
    np.random.shuffle(val_indices)

    split_point = len(val_indices) // 2
    new_val_idx = val_indices[:split_point]
    test_idx = val_indices[split_point:]

    print(f"New validation set size: {len(new_val_idx)}")
    print(f"New test set size: {len(test_idx)}")

    # Save new indices
    new_val_path = save_dir / 'val_idx_imagenet.npy'  # Overwrite original
    test_idx_path = save_dir / 'test_idx_imagenet.npy'  # New file

    np.save(new_val_path, new_val_idx)
    np.save(test_idx_path, test_idx)

    print(f"Saved new validation indices to: {new_val_path}")
    print(f"Saved test indices to: {test_idx_path}")

    # Verification
    loaded_new_val = np.load(new_val_path)
    loaded_test = np.load(test_idx_path)

    assert len(loaded_new_val) == len(new_val_idx), "Validation indices save/load mismatch"
    assert len(loaded_test) == len(test_idx), "Test indices save/load mismatch"
    assert np.array_equal(loaded_new_val, new_val_idx), "Validation indices content mismatch"
    assert np.array_equal(loaded_test, test_idx), "Test indices content mismatch"

    print("Verification successful: all indices saved and loaded correctly")

    return new_val_idx, test_idx

if __name__ == "__main__":
    split_imagenet_validation()