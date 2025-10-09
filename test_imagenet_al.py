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
from torchvision.datasets import ImageFolder, FakeData
from active_learning_coreset import load_imagenet_data, ActiveLearningTrainer
from utils.active_learning import AttributeFocusedStrategy, parse_attribute_priority_log
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


def test_attribute_strategy_mock():
    """Ensure attribute-focused strategy prioritizes rare/uncertain slices."""

    print("\n=== Testing Attribute-Focused Strategy (Mock Data) ===")

    # Mock metadata for five samples with color attribute
    metadata = {
        "color": {
            "data": ["red", "blue", "blue", "green", "red"],
            "word": ["red", "blue", "green"],
            "uncertain_idx": [1],
        }
    }

    strategy = AttributeFocusedStrategy(
        attribute_metadata=metadata,
        rarity_power=1.0,
        uncertain_boost=1.0,
    )

    # Simple dataset with five entries
    dataset = FakeData(size=5, image_size=(3, 64, 64), num_classes=2)

    selected = strategy.select(dataset, n_select=2, random_seed=0)

    print(f"Selected indices: {selected}")

    assert len(selected) == 2, "Strategy must return requested number of samples"
    # Expect uncertain sample (idx 1) and rare attribute sample (idx 3)
    assert 1 in selected, "Uncertain attribute sample should be prioritized"
    assert 3 in selected, "Rarest attribute sample should be prioritized"

    # Validate log parsing and value boosts
    log_content = (
        'ACC: 0.60\tRare Case: attribute "color", description "blue"\n'
        'ACC: 0.55\tHard case: attribute "color", description "green"\n'
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
        assert "color" in priorities, "Log parser should capture color attribute"
        assert "blue" in priorities["color"]["rare"], "Blue should be marked as rare"
        assert "green" in priorities["color"]["hard"], "Green should be marked as hard"

        value_boosts = {}
        for attr_name, buckets in priorities.items():
            boost_map = {}
            for value in buckets.get("rare", set()):
                boost_map[value] = boost_map.get(value, 0.0) + 2.0
            for value in buckets.get("hard", set()):
                boost_map[value] = boost_map.get(value, 0.0) + 1.0
            if boost_map:
                value_boosts[attr_name] = boost_map

        boosted_strategy = AttributeFocusedStrategy(
            attribute_metadata=metadata,
            rarity_power=0.0,
            uncertain_boost=0.0,
            value_boosts=value_boosts,
        )

        boosted_selected = boosted_strategy.select(dataset, n_select=1, random_seed=0)
        assert boosted_selected[0] in {1, 2}, "Log boosts should prioritize blue samples"
    finally:
        os.unlink(tmp_log_name)

    print("✅ Attribute-focused strategy mock test passed!")
    return True

if __name__ == "__main__":
    setup_success = test_imagenet_setup()
    attribute_success = test_attribute_strategy_mock()
    exit(0 if (setup_success and attribute_success) else 1)
