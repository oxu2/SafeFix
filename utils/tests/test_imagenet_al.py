#!/usr/bin/env python3
"""
Quick test for ImageNet Active Learning setup
"""

import sys
sys.path.append('.')

import numpy as np
import os
import tempfile
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from utils.active_learning_coreset import load_imagenet_data, ActiveLearningTrainer
from utils.active_learning import AttributeFocusedStrategy, parse_attribute_priority_log
from utils.defines import TRAIN, VALID, UNLABELED

def test_imagenet_setup():
    """Test ImageNet data loading and model setup"""
    
    print("=== Testing ImageNet Active Learning Setup ===")
    
    # 1. Test data loading
    data_dir = Path("data/imagenet_subset")
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


class RandomTensorDataset(Dataset):
    """Dataset that returns random tensors with stable seeding."""

    def __init__(self, num_samples, seed=0):
        self.num_samples = num_samples
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.randn(num_samples, 3, 64, 64, generator=generator)
        self.labels = torch.randint(low=0, high=2, size=(num_samples,), generator=generator)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], int(self.labels[idx])


def test_attribute_strategy_mock():
    """Ensure attribute-focused strategy prioritizes rare/uncertain slices."""

    print("\n=== Testing Attribute-Focused Strategy (Mock Data) ===")

    # Explicit annotations for thirty-five samples
    sample_annotations = [
        # Blue-ground cases (slightly fewer)
        ("blue", "ground"),
        ("blue", "ground"),
        ("blue", "ground"),
        ("blue", "ground"),
        # Blue-sky cases
        ("blue", "sky"),
        ("blue", "sky"),
        ("blue", "sky"),
        # Green-ground cases
        ("green", "ground"),
        ("green", "ground"),
        ("green", "ground"),
        # Green-sky cases (rarest)
        ("green", "sky"),
        ("green", "sky"),
        # Red-ground cases (most common)
        ("red", "ground"),
        ("red", "ground"),
        ("red", "ground"),
        ("red", "ground"),
        ("red", "ground"),
        ("red", "ground"),
        ("red", "ground"),
        ("red", "ground"),
        ("red", "ground"),
        ("red", "ground"),
        ("red", "ground"),
        ("red", "ground"),
        ("red", "ground"),
        ("red", "ground"),
        ("red", "ground"),
        ("red", "ground"),
        ("red", "ground"),
        ("red", "ground"),
        ("red", "ground"),
        # Red-sky cases (few examples)
        ("red", "sky"),
        ("red", "sky"),
        ("red", "sky"),
        ("red", "sky"),
        ("red", "sky"),
        ("red", "sky"),
    ]

    for idx, (color, background) in enumerate(sample_annotations):
        print(f"Sample {idx:02d}: color={color}, background={background}")

    color_data = [c for c, _ in sample_annotations]
    background_data = [b for _, b in sample_annotations]

    metadata = {
        "color": {
            "data": color_data,
            "word": ["red", "blue", "green"],
            "uncertain_idx": [1],
        },
        "background": {
            "data": background_data,
            "word": ["ground", "sky"],
            "uncertain_idx": [],
        }
    }

    strategy = AttributeFocusedStrategy(
        attribute_metadata=metadata,
        rarity_power=1.0,
        uncertain_boost=1.0,
    )

    # Dataset providing random tensors for the thirty-five entries
    dataset = RandomTensorDataset(num_samples=len(sample_annotations), seed=0)

    raw_selected = strategy.select(dataset, n_select=20, random_seed=0)
    selected = sorted(raw_selected)

    print(f"Selected indices (sorted): {selected}")

    # Expect uncertain sample (idx 1) and rare attribute sample (idx 10 or 11)
    assert 1 in selected, "Uncertain attribute sample should be prioritized"
    assert any(idx in selected for idx in (10, 11)), "Rarest attribute sample should be prioritized"

    blue_sky_indices = [idx for idx, (color, background) in enumerate(sample_annotations) if color == "blue" and background == "sky"]
    assert sum(1 for idx in raw_selected if idx in blue_sky_indices) >= 2, (
        "Low-accuracy blue sky combinations should appear multiple times in the selection"
    )

    # Validate log parsing and value weighting
    log_content = (
        'ACC: 0.60\tRare Case: attribute "color", description "blue"\n'
        'ACC: 0.55\tRare Case: attribute "color", description "green"\n'
        'ACC: 0.40\tRare Case: attribute "background", description "sky"\n'
    )

    tmp_log = tempfile.NamedTemporaryFile("w", delete=False)
    try:
        tmp_log.write(log_content)
        tmp_log.flush()
        tmp_log_name = tmp_log.name
    finally:
        tmp_log.close()

    try:
        priorities = parse_attribute_priority_log(tmp_log_name)
        assert "color" in priorities and "background" in priorities, "Log parser should capture attributes"
        assert abs(priorities["color"]["blue"] - 0.40) < 1e-6, "Blue failure weight should be preserved"
        assert abs(priorities["color"]["green"] - 0.45) < 1e-6, "Green failure weight should be preserved"
        assert abs(priorities["background"]["sky"] - 0.60) < 1e-6, "Sky failure weight should be preserved"

        boosted_strategy = AttributeFocusedStrategy(
            attribute_metadata=metadata,
            rarity_power=0.0,
            uncertain_boost=0.0,
            value_weights=priorities,
            log_acc_scale=1.0,
        )

        boosted_selected = boosted_strategy.select(dataset, n_select=1, random_seed=0)
        green_sky_indices = [idx for idx, (color, background) in enumerate(sample_annotations) if color == "green" and background == "sky"]
        assert boosted_selected[0] in green_sky_indices, (
            "Combined weights should prioritize samples covering the lowest-accuracy attributes"
        )
    finally:
        os.unlink(tmp_log_name)

    print("✅ Attribute-focused strategy mock test passed!")
    return True

if __name__ == "__main__":
    setup_success = test_imagenet_setup()
    attribute_success = test_attribute_strategy_mock()
    exit(0 if (setup_success and attribute_success) else 1)
