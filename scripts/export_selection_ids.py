#!/usr/bin/env python3
"""Export selection-only ids for ImageNet, CelebA, and COCO.

This script rebuilds the original training candidate pool and runs one active
learning strategy without starting any model training. It writes dataset-local
selected indices plus original image ids for downstream plotting.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "pose_estimation") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "pose_estimation"))

from utils.active_learning import active_learning_selection
from utils.data_utils import CelebADataset, default_img_transform


METHODS = ["random", "stratified", "entropy", "coreset", "badge", "ppal", "uherding", "ours"]


def str_ratio_tag(ratio: float) -> str:
    return f"ratio{ratio:g}".replace(".", "p")


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_selection(out_dir: Path, selected_indices: list[int], selected_original_ids: list[str], metadata: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selected_indices.txt").write_text(
        "\n".join(str(int(idx)) for idx in selected_indices) + "\n",
        encoding="utf-8",
    )
    (out_dir / "selected_original_ids.txt").write_text(
        "\n".join(str(item) for item in selected_original_ids) + "\n",
        encoding="utf-8",
    )
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def common_al_args(args: argparse.Namespace, method: str) -> SimpleNamespace:
    strategy = "attribute" if method == "ours" else method
    return SimpleNamespace(
        dataset=args.dataset,
        model="resnet",
        use_active_learning=True,
        al_strategy=strategy,
        sub_ratio=args.ratio,
        seed=args.seed,
        gpu=str(args.gpu),
        batch_size=args.batch_size,
        workers=args.num_workers,
        al_embedding_dim=512,
        al_initial_labeled=400,
        al_clusters=None,
        pretrained=True,
        resume_model=None,
        attributes="",
        num_added_img=1000,
        attribute_file=None,
        al_attribute_focus=None,
        al_attribute_rarity_power=1.0,
        al_attribute_uncertainty_boost=1.0,
        al_attribute_log_path=None,
        al_attribute_log_acc_scale=0.0,
        al_attribute_extra_counts=None,
        al_attribute_extra_count_scale=1.0,
        al_attribute_frequency_penalty=0.0,
        al_attribute_class_balance=0.0,
        al_attribute_use_entropy=False,
        al_attribute_use_rarity=True,
        al_conditional_rarity=False,
        al_ppal_temperature=1.0,
        al_ppal_uncertainty="binary_entropy",
        al_ppal_pool_expand_ratio=4.0,
        al_ppal_class_weight_ub=0.2,
        al_ppal_class_weight_alpha=0.3,
        al_ppal_diversity_power=1.0,
    )


def imagenet_args(args: argparse.Namespace, method: str) -> SimpleNamespace:
    ns = common_al_args(args, method)
    ns.resume_model = "save/weights/model_imagenet_resnet_base.pth" if method == "entropy" else None
    if method == "ours":
        ns.attributes = "pinkcolor_fabrictexture"
        ns.attribute_file = "save/attribute10_internvl3_8B.json"
        ns.al_attribute_focus = "color,background,texture"
        ns.al_attribute_rarity_power = 0.01
        ns.al_attribute_frequency_penalty = 0.0
        ns.al_attribute_class_balance = "inf"
        ns.al_attribute_extra_counts = "color:pink=1000,texture:fabric=1000"
        ns.al_attribute_extra_count_scale = 1.0 / args.ratio
    return ns


def celeba_args(args: argparse.Namespace, method: str) -> SimpleNamespace:
    ns = common_al_args(args, method)
    ns.resume_model = "save/weights/model_celeba_resnet_base.pth" if method == "entropy" else None
    if method == "ours":
        ns.attributes = "redhair_brownskin_sademotion"
        ns.attribute_file = "save/attribute_celeba_trainval_qwen3_8B.json"
        ns.al_attribute_focus = "hair color,skin,emotion"
        ns.al_attribute_rarity_power = 0.03
        ns.al_attribute_frequency_penalty = 0.5
        ns.al_attribute_class_balance = "inf"
        ns.al_attribute_extra_counts = "hair color:red=1000,skin:brown=1000,emotion:sad=1000"
        ns.al_attribute_extra_count_scale = 1.0 / args.ratio
    return ns


def load_imagenet_candidates(args: argparse.Namespace):
    transform = default_img_transform(train=True)
    val_transform = default_img_transform(train=False)
    root = Path(args.imagenet_root)
    train_idx = np.load("save/train_idx_imagenet.npy")
    val_idx = np.load("save/val_idx_imagenet.npy")
    train_ds = Subset(ImageFolder(root, transform=transform), train_idx)
    val_ds = Subset(ImageFolder(root, transform=val_transform), val_idx)
    original_ids = [str(int(idx)) for idx in train_idx]
    return train_ds, val_ds, original_ids


def load_celeba_candidates(args: argparse.Namespace):
    with Path("data/celebA/list_attr_celeba.txt").open(encoding="utf-8") as handle:
        names = [line.strip().split(" ")[0] for line in handle.readlines()][1:]
    image_root = Path(args.celeba_root)
    all_paths = [str(image_root / name) for name in names]
    train_idx = np.load("data/celebA/train_idx.npy")[:80000]
    valid_idx = np.load("save/val_idx_celeba.npy")
    labels_all = np.load("data/celebA/labels.npy")
    train_imgs = [all_paths[int(idx)] for idx in train_idx]
    valid_imgs = [all_paths[int(idx)] for idx in valid_idx]
    train_labels = labels_all[train_idx]
    valid_labels = labels_all[valid_idx]
    train_ds = CelebADataset(train_imgs, train_labels, default_img_transform(train=True))
    val_ds = CelebADataset(valid_imgs, valid_labels, default_img_transform(train=False))
    original_ids = [str(int(idx)) for idx in train_idx]
    return train_ds, val_ds, original_ids


def export_image_dataset(args: argparse.Namespace) -> None:
    start_time = time.time()
    if args.dataset == "imagenet":
        train_ds, val_ds, original_ids = load_imagenet_candidates(args)
        al_args = imagenet_args(args, args.method)
    elif args.dataset == "celeba":
        train_ds, val_ds, original_ids = load_celeba_candidates(args)
        al_args = celeba_args(args, args.method)
    else:
        raise ValueError(args.dataset)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    selected = active_learning_selection(train_ds, val_ds, al_args, device, verbose=True)
    selected = [int(idx) for idx in selected]
    selected_original_ids = [original_ids[idx] for idx in selected]
    out_dir = args.out_root / str_ratio_tag(args.ratio) / args.dataset / args.method
    write_selection(
        out_dir,
        selected,
        selected_original_ids,
        {
            "dataset": args.dataset,
            "ratio": args.ratio,
            "method": args.method,
            "selected_count": len(selected),
            "candidate_count": len(original_ids),
            "gpu": args.gpu,
            "selection_only": True,
            "selection_time_sec": time.time() - start_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    print(f"Wrote {out_dir}")


def coco_args(args: argparse.Namespace) -> SimpleNamespace:
    ns = SimpleNamespace(
        al_strategy="attribute" if args.method == "ours" else args.method,
        use_active_learning=True,
        seed=args.seed,
        max_train_samples=20000,
        sub_ratio=args.ratio,
        al_proxy_bins=4,
        al_initial_labeled=400,
        al_feature_batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=f"cuda:{args.gpu}",
        al_entropy_batch_size=2,
        al_ppal_temperature=1.0,
        al_ppal_uncertainty="binary_entropy",
        al_ppal_pool_expand_ratio=4.0,
        al_ppal_class_weight_ub=0.2,
        al_ppal_class_weight_alpha=0.3,
        al_ppal_diversity_power=1.0,
        assigner="qwen3",
        generated_attribute="brown_rainy",
        num_added_img=1000,
        al_attribute_file=Path("save/attribute_coco_pose_gemini_train20k_qwen3_8B.json"),
        al_attribute_focus="clothes_color,weather",
        al_attribute_rarity_power=0.03,
        al_attribute_use_rarity=True,
        al_attribute_uncertainty_boost=1.0,
        al_attribute_use_entropy=False,
        al_attribute_log_path=None,
        al_attribute_log_acc_scale=0.0,
        al_attribute_extra_counts="clothes_color:brown=1000,weather:rainy=1000",
        al_attribute_extra_count_scale=1.0 / args.ratio,
        al_attribute_frequency_penalty=0.5,
        al_attribute_class_balance="0",
        al_conditional_rarity=False,
    )
    return ns


def export_coco(args: argparse.Namespace) -> None:
    start_time = time.time()
    from pose_estimation.dataset import build_train_val_test_datasets
    from train_pose_estimation import select_original_train_image_ids

    full_train, val_ds, _ = build_train_val_test_datasets(
        data_root=args.coco_root,
        hflip_prob=0.0,
        seed=args.seed,
        max_train_samples=0,
    )
    al_args = coco_args(args)
    selected_ids, info = select_original_train_image_ids(
        full_train,
        val_ds,
        max_train_samples=20000,
        sub_ratio=args.ratio,
        seed=args.seed,
        args=al_args,
    )
    candidate_ids = list(full_train.image_ids)[:20000]
    id_to_local = {int(image_id): idx for idx, image_id in enumerate(candidate_ids)}
    selected_indices = [id_to_local[int(image_id)] for image_id in selected_ids]
    out_dir = args.out_root / str_ratio_tag(args.ratio) / args.dataset / args.method
    write_selection(
        out_dir,
        selected_indices,
        [str(int(image_id)) for image_id in selected_ids],
        {
            "dataset": args.dataset,
            "ratio": args.ratio,
            "method": args.method,
            "selected_count": len(selected_ids),
            "candidate_count": len(candidate_ids),
            "gpu": args.gpu,
            "selection_only": True,
            "selection_info": info,
            "selection_time_sec": time.time() - start_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    print(f"Wrote {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["imagenet", "celeba", "coco"], required=True)
    parser.add_argument("--ratio", type=float, choices=[0.05, 0.1, 0.3, 0.5], required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out-root", type=Path, default=Path("save/selection_exports"))
    parser.add_argument("--imagenet-root", default="data/imagenet_subset")
    parser.add_argument("--celeba-root", default="/data/oxx220000/img_align_celeba")
    parser.add_argument("--coco-root", default="/data/oxx220000/coco")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_reproducible_seed(args.seed)
    if args.dataset in {"imagenet", "celeba"}:
        export_image_dataset(args)
    else:
        export_coco(args)


if __name__ == "__main__":
    main()
