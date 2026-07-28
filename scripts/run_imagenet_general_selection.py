#!/usr/bin/env python3
"""Thin CLI for ImageNet10 general selection experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.imagenet_general_selection import (
    ACTIVE_LEARNING_METHODS,
    DATA_SELECTION_METHODS,
    AttributeConfig,
    TrainingConfig,
    load_dataset_bundle,
    run_active_learning_experiment,
    run_data_selection_experiment,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run source-only ImageNet10 data selection or iterative active learning."
    )
    parser.add_argument("--experiment", choices=("data_selection", "active_learning"), required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--train-index", type=Path, default=Path("save/train_idx_imagenet.npy"))
    parser.add_argument("--val-index", type=Path, default=Path("save/val_idx_imagenet.npy"))
    parser.add_argument("--test-index", type=Path, default=Path("save/test_idx_imagenet.npy"))
    parser.add_argument("--attribute-file", type=Path, default=Path("save/attribute10_qwen3_8B.json"))
    parser.add_argument("--attribute-fields", default="color,background,texture")
    parser.add_argument("--proxy-checkpoint", type=Path, default=None)
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--initial-ratio", type=float, default=0.1)
    parser.add_argument("--query-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--lr-step-size", type=int, default=3)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--rarity-power", type=float, default=0.03)
    parser.add_argument("--frequency-penalty", type=float, default=0.5)
    parser.add_argument("--class-balance", type=float, default=15.0)
    parser.add_argument("--entropy-boost", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fields = tuple(value.strip() for value in args.attribute_fields.split(",") if value.strip())
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
    )
    attribute_config = AttributeConfig(
        fields=fields,
        rarity_power=args.rarity_power,
        frequency_penalty=args.frequency_penalty,
        class_balance=args.class_balance,
        entropy_boost=args.entropy_boost,
    )
    bundle = load_dataset_bundle(
        data_dir=args.data_dir,
        train_index_path=args.train_index,
        val_index_path=args.val_index,
        test_index_path=args.test_index,
        attribute_path=args.attribute_file,
        attribute_fields=fields,
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"device={device}")
    print(f"experiment={args.experiment} method={args.method} seed={args.seed}")
    print(
        f"split_sizes=train:{len(bundle.train_indices)} val:{len(bundle.val_indices)} "
        f"test:{len(bundle.test_indices)}"
    )

    if args.experiment == "data_selection":
        if args.method not in DATA_SELECTION_METHODS:
            raise ValueError(f"Data-selection method must be one of {DATA_SELECTION_METHODS}")
        result, manifest = run_data_selection_experiment(
            bundle=bundle,
            method=args.method,
            ratio=args.ratio,
            seed=args.seed,
            proxy_checkpoint=args.proxy_checkpoint,
            training_config=training_config,
            attribute_config=attribute_config,
            device=device,
            selection_only=args.selection_only,
        )
    else:
        if args.method not in ACTIVE_LEARNING_METHODS:
            raise ValueError(f"Active-learning method must be one of {ACTIVE_LEARNING_METHODS}")
        result, manifest = run_active_learning_experiment(
            bundle=bundle,
            method=args.method,
            seed=args.seed,
            rounds=args.rounds,
            initial_ratio=args.initial_ratio,
            query_ratio=args.query_ratio,
            training_config=training_config,
            attribute_config=attribute_config,
            device=device,
            selection_only=args.selection_only,
        )

    result["data_dir"] = str(args.data_dir)
    result["attribute_file"] = str(args.attribute_file)
    result["proxy_checkpoint"] = str(args.proxy_checkpoint) if args.proxy_checkpoint else None
    write_json_atomic(args.selection_manifest, manifest)
    write_json_atomic(args.result_json, result)
    print(f"selection_manifest={args.selection_manifest}")
    print(f"result_json={args.result_json}")


if __name__ == "__main__":
    main()
