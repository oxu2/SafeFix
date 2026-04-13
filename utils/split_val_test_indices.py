#!/usr/bin/env python3
"""Split the ImageNet validation indices into validation and test subsets."""
import argparse
from pathlib import Path
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Split ImageNet validation indices into validation and test sets.")
    parser.add_argument(
        "--input",
        default="save/val_idx_imagenet.npy",
        help="Path to the original validation indices (.npy).",
    )
    parser.add_argument(
        "--val-output",
        default="save/val_idx_imagenet.npy",
        help="Output path for the new validation indices (.npy).",
    )
    parser.add_argument(
        "--test-output",
        default="save/test_idx_imagenet.npy",
        help="Output path for the new test indices (.npy).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when shuffling indices before the split.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input indices file not found: {input_path}")

    indices = np.load(input_path)
    if indices.size == 0:
        raise ValueError("Input indices array is empty; cannot create validation/test split.")

    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(indices)
    split_point = shuffled.size // 2
    new_val = np.sort(shuffled[:split_point])
    new_test = np.sort(shuffled[split_point:])

    val_output = Path(args.val_output)
    test_output = Path(args.test_output)
    val_output.parent.mkdir(parents=True, exist_ok=True)
    test_output.parent.mkdir(parents=True, exist_ok=True)

    np.save(val_output, new_val)
    np.save(test_output, new_test)

    print(f"Saved {new_val.size} validation indices to {val_output}")
    print(f"Saved {new_test.size} test indices to {test_output}")


if __name__ == "__main__":
    main()
