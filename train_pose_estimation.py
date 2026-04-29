#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import models as tv_models
from torchvision.models import ResNet50_Weights

from pose_estimation.dataset import (
    build_datasets,
    build_test_dataset,
    build_train_val_test_datasets,
    clone_dataset_with_image_ids,
    collate_fn,
    default_coco_paths,
)
from pose_estimation.engine import (
    evaluate_coco_keypoints,
    load_checkpoint,
    load_model_checkpoint,
    read_checkpoint_metadata,
    save_model_checkpoint,
    save_checkpoint,
    train_one_epoch,
)
from pose_estimation.modeling import create_keypointrcnn_model
from pose_estimation.visualize import draw_pose_predictions, image_to_tensor
from utils.active_learning import (
    AttributeFocusedStrategy,
    BadgeStrategy,
    CoreSetStrategy,
    StratifiedStrategy,
    parse_attribute_priority_log,
    parse_attribute_value_counts,
    parse_float_with_infinity,
)
from utils.attribute_aliases import (
    canonicalize_attribute_counts,
    canonicalize_coco_pose_generated_attribute,
    coco_pose_generated_attribute_choices,
)
from utils.stat_pose_estimation import run_analysis as run_pose_slice_analysis


REPO_ROOT = Path(__file__).resolve().parent
POSE_LOG_ROOT = Path("log_pose")
ASSIGNER_MODEL_MAP = {
    "qwen3": "Qwen/Qwen3-VL-8B-Instruct",
    "internvl3": "OpenGVLab/InternVL3-8B",
}
ANALYSIS_SOURCE_DEFAULTS = {
    "internvl3": {
        "attribute_file": Path("save/attribute_coco_pose_gemini_internvl3_8B.json"),
        "attribute_split": "val_full",
        "train_attribute_file": Path("save/attribute_coco_pose_gemini_train20k_internvl3_8B.json"),
        "train_attribute_split": "train",
    },
    "qwen3": {
        "attribute_file": Path("save/attribute_coco_pose_gemini_qwen3_8B.json"),
        "attribute_split": "val_full",
        "train_attribute_file": Path("save/attribute_coco_pose_gemini_train20k_qwen3_8B.json"),
        "train_attribute_split": "train",
    },
    "blip": {
        "attribute_file": Path("save/attribute_coco_pose_gemini_blip_val.json"),
        "attribute_split": "val_full",
        "train_attribute_file": Path("save/attribute_coco_pose_gemini_train20k_blip.json"),
        "train_attribute_split": "train",
    },
}
POSE_GENERATED_ATTRIBUTE_ROOTS = {
    "brown_rainy": Path("/data/oxx220000/coco-pose-weather-edits/generated_coco_qwen3_1500_brown_rain_icedit"),
    "rainy": Path("/data/oxx220000/coco-pose-weather-edits/generated_coco_train1500_rain"),
}
POSE_GENERATED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
POSE_AL_RANDOM_ALIASES = {"none", "0", "false", "no", "random", ""}
POSE_AL_STRATEGIES = ["random", "stratified", "coreset", "badge", "attribute"]
POSE_GENERATED_ATTRIBUTE_EXTRA_COUNTS = {
    "brown_rainy": {
        "clothes_color": {"brown": 1.0},
        "weather": {"rainy": 1.0},
    },
    "rainy": {"weather": {"rainy": 1.0}},
}
POSE_AL_IMAGE_SIZE = 224
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def _format_duration(seconds: float) -> str:
    total = int(max(0, round(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _timestamp_str(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def parse_args():
    parser = argparse.ArgumentParser(description="COCO pose estimation with Keypoint R-CNN.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train model on COCO keypoints.")
    _add_common_data_args(train_parser)
    train_parser.add_argument("--epochs", type=int, default=2)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--num-workers", type=int, default=4)
    train_parser.add_argument("--lr", type=float, default=0.005)
    train_parser.add_argument("--momentum", type=float, default=0.9)
    train_parser.add_argument("--weight-decay", type=float, default=5e-4)
    train_parser.add_argument("--lr-step-size", type=int, default=8)
    train_parser.add_argument("--lr-gamma", type=float, default=0.1)
    train_parser.add_argument("--hflip-prob", type=float, default=0.5)
    train_parser.add_argument("--eval-every", type=int, default=1)
    train_parser.add_argument("--output-dir", type=Path, default=POSE_LOG_ROOT / "pose_keypointrcnn")
    train_parser.add_argument("--resume", type=Path, default=None)
    train_parser.add_argument(
        "--resume-model",
        type=Path,
        default=None,
        help=(
            "Load only model weights from a checkpoint and continue training with a fresh "
            "optimizer / scheduler. If the checkpoint stores an epoch, training resumes "
            "from the next epoch index."
        ),
    )
    train_parser.add_argument("--score-threshold", type=float, default=0.05)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--max-train-samples", type=int, default=20000)
    train_parser.add_argument(
        "--sub-ratio",
        type=float,
        default=1.0,
        help="Fraction of original COCO train images to keep before appending generated images.",
    )
    train_parser.add_argument(
        "--use-active-learning",
        type=str2bool,
        default=True,
        help=(
            "Enable Active Learning based subset selection when --sub-ratio < 1. "
            "If False, falls back to plain random subsampling."
        ),
    )
    train_parser.add_argument(
        "--al-strategy",
        type=str,
        default="random",
        choices=[*POSE_AL_STRATEGIES, "none", "0", "false", "no"],
        help=(
            "Subset selection strategy for the original train split when --sub-ratio < 1. "
            "Supported: random, stratified, coreset, badge, attribute."
        ),
    )
    train_parser.add_argument(
        "--al-initial-labeled",
        type=int,
        default=400,
        help="Proxy labeled set size drawn from the validation split for coreset/badge.",
    )
    train_parser.add_argument(
        "--al-feature-batch-size",
        type=int,
        default=16,
        help="Batch size used for pose Active Learning feature extraction.",
    )
    train_parser.add_argument(
        "--al-proxy-bins",
        type=int,
        default=4,
        help="Number of pose-complexity bins used as pseudo-labels for stratified/badge selection.",
    )
    train_parser.add_argument(
        "--al-attribute-file",
        type=Path,
        default=None,
        help=(
            "Optional attribute JSON for pose attribute-based selection. "
            "Defaults to the train attribute file associated with --assigner."
        ),
    )
    train_parser.add_argument(
        "--al-attribute-focus",
        type=str,
        default=None,
        help="Comma-separated pose attribute keys to prioritize for attribute AL.",
    )
    train_parser.add_argument(
        "--al-attribute-rarity-power",
        type=float,
        default=0.03,
        help="Rarity exponent for pose attribute selection.",
    )
    train_parser.add_argument(
        "--al-attribute-uncertainty-boost",
        type=float,
        default=1.0,
        help="Bonus for uncertain indices in pose attribute metadata.",
    )
    train_parser.add_argument(
        "--al-attribute-use-entropy",
        type=str2bool,
        default=False,
        help=(
            "Reserved for parity with alpred.py. Pose attribute AL currently uses static metadata "
            "scoring and ignores dynamic entropy."
        ),
    )
    train_parser.add_argument(
        "--al-attribute-log-path",
        type=Path,
        default=None,
        help="Optional diagnostic log path for Rare Case weighting in pose attribute AL.",
    )
    train_parser.add_argument(
        "--al-attribute-log-acc-scale",
        type=float,
        default=0.0,
        help="Scale factor for Rare Case log-derived weights in pose attribute AL.",
    )
    train_parser.add_argument(
        "--al-attribute-extra-counts",
        type=str,
        default=None,
        help="Inline attr:value=count map or JSON path for generated pose attribute counts.",
    )
    train_parser.add_argument(
        "--al-attribute-extra-count-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to generated pose attribute counts.",
    )
    train_parser.add_argument(
        "--al-attribute-frequency-penalty",
        type=float,
        default=0.5,
        help="Penalty strength for over-represented pose attribute values.",
    )
    train_parser.add_argument(
        "--al-attribute-class-balance",
        type=str,
        default="0",
        help=(
            "Pose class-balance weight for attribute AL. Uses pose-complexity pseudo-labels; "
            'pass "inf" for hard round-robin balancing.'
        ),
    )
    train_parser.add_argument(
        "--al-conditional-rarity",
        type=str2bool,
        default=False,
        help="Enable class-conditional pose attribute rarity scoring.",
    )
    train_parser.add_argument(
        "--attribute",
        "--generated-attribute",
        dest="generated_attribute",
        choices=["none", *coco_pose_generated_attribute_choices()],
        default="none",
        help=(
            "Generated augmentation attribute to append to the train split. "
            "For COCO weather edits, the canonical value is 'rainy'; legacy 'rain' is accepted as an alias."
        ),
    )
    train_parser.add_argument(
        "--generated-image-dir",
        type=Path,
        default=None,
        help=(
            "Optional generated-image directory override. Pass either the folder that contains "
            "the generated files directly or its parent directory if it has an outputs/ subfolder."
        ),
    )
    train_parser.add_argument(
        "--num-added-img",
        "--generated-max-samples",
        dest="num_added_img",
        type=int,
        default=1000,
        help="Maximum number of generated images to append to the training split.",
    )
    train_parser.add_argument("--pretrained", type=str2bool, default=True)
    train_parser.add_argument("--device", type=str, default="cuda")
    train_parser.add_argument("--analyze-after-train", type=str2bool, default=True)
    train_parser.add_argument(
        "--assigner",
        "--analysis-source",
        dest="assigner",
        choices=["internvl3", "qwen3", "blip"],
        default="internvl3",
        help="Select the precomputed attribute source for post-train analysis.",
    )
    train_parser.add_argument("--analysis-attribute-file", type=Path, default=None)
    train_parser.add_argument(
        "--analysis-attribute-split",
        choices=["train", "val_full", "val", "test", "all"],
        default="val_full",
    )
    train_parser.add_argument("--analysis-train-attribute-file", type=Path, default=None)
    train_parser.add_argument(
        "--analysis-train-attribute-split",
        choices=["train", "val_full", "val", "test", "all"],
        default="train",
    )
    train_parser.add_argument(
        "--analysis-corpus",
        type=Path,
        default=Path("save/corpus_coco_pose_gemini.json"),
    )
    train_parser.add_argument("--analysis-attributes", type=str, default=None)
    train_parser.add_argument("--analysis-min-images", type=int, default=25)
    train_parser.add_argument("--analysis-top-k", type=int, default=20)
    train_parser.add_argument(
        "--analysis-sort-by",
        choices=["ap", "ap_delta", "images"],
        default="ap",
    )
    train_parser.add_argument("--analysis-acc-diff-threshold", type=float, default=0.005)
    train_parser.add_argument("--analysis-distribution-diff-threshold", type=float, default=0.15)
    train_parser.add_argument("--analysis-rare-label-threshold", type=float, default=1.1)
    train_parser.add_argument("--assign-after-train", type=str2bool, default=False)
    train_parser.add_argument(
        "--assigners",
        type=str,
        default="blip",
        help="Comma-separated assigners to run after training. Supported: blip,qwen3,internvl3",
    )
    train_parser.add_argument(
        "--assigner-splits",
        type=str,
        default="train",
        help="Comma-separated COCO splits for assigners. Supported: val,train,all",
    )
    train_parser.add_argument("--assigner-corpus", type=Path, default=None)
    train_parser.add_argument("--assigner-output-root", type=Path, default=None)
    train_parser.add_argument("--assigner-save-every", type=int, default=500)
    train_parser.add_argument("--assigner-max-memory-per-gpu", type=float, default=23.0)
    train_parser.add_argument("--assigner-verbose", type=str2bool, default=False)
    train_parser.add_argument(
        "--save-resume-checkpoint",
        type=str2bool,
        default=False,
        help=(
            "Also save a full optimizer/scheduler resume checkpoint as last_resume.pth. "
            "Default is False to reduce disk usage."
        ),
    )
    train_parser.add_argument(
        "--save-last-checkpoint",
        type=str2bool,
        default=False,
        help=(
            "Save a compact last.pth model checkpoint after each epoch. "
            "Default is False because best.pth is usually sufficient for this pose workflow."
        ),
    )

    eval_parser = subparsers.add_parser("eval", help="Evaluate model on COCO val keypoints.")
    _add_common_data_args(eval_parser)
    eval_parser.add_argument("--batch-size", type=int, default=2)
    eval_parser.add_argument("--num-workers", type=int, default=4)
    eval_parser.add_argument("--checkpoint", type=Path, required=False)
    eval_parser.add_argument("--score-threshold", type=float, default=0.05)
    eval_parser.add_argument("--output-json", type=Path, default=POSE_LOG_ROOT / "pose_eval_results.json")
    eval_parser.add_argument("--pretrained", type=str2bool, default=True)
    eval_parser.add_argument("--device", type=str, default="cuda")
    eval_parser.add_argument("--seed", type=int, default=42)

    test_parser = subparsers.add_parser(
        "test",
        help="Final test evaluation on the held-out half of valid val2017 images.",
    )
    _add_common_data_args(test_parser)
    test_parser.add_argument("--batch-size", type=int, default=2)
    test_parser.add_argument("--num-workers", type=int, default=4)
    test_parser.add_argument("--checkpoint", type=Path, required=True,
                             help="Path to the model checkpoint (e.g. best.pth).")
    test_parser.add_argument("--score-threshold", type=float, default=0.05)
    test_parser.add_argument("--output-json", type=Path, default=POSE_LOG_ROOT / "pose_test_results.json")
    test_parser.add_argument("--pretrained", type=str2bool, default=True)
    test_parser.add_argument("--device", type=str, default="cuda")
    test_parser.add_argument("--seed", type=int, default=42)

    infer_parser = subparsers.add_parser("infer", help="Run inference on one image and draw pose.")
    infer_parser.add_argument("--image", type=Path, required=True)
    infer_parser.add_argument("--checkpoint", type=Path, default=None)
    infer_parser.add_argument("--output", type=Path, default=POSE_LOG_ROOT / "pose_infer.jpg")
    infer_parser.add_argument("--score-threshold", type=float, default=0.5)
    infer_parser.add_argument("--keypoint-threshold", type=float, default=0.1)
    infer_parser.add_argument("--pretrained", type=str2bool, default=True)
    infer_parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def _add_common_data_args(parser):
    parser.add_argument("--data-root", type=str, default="/data/oxx220000/coco")
    parser.add_argument("--train-images", type=str, default=None)
    parser.add_argument("--val-images", type=str, default=None)
    parser.add_argument("--test-images", type=str, default=None)
    parser.add_argument("--train-annotations", type=str, default=None)
    parser.add_argument("--val-annotations", type=str, default=None)
    parser.add_argument("--test-annotations", type=str, default=None)


def str2bool(value: str) -> bool:
    value = value.lower().strip()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_loader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def normalize_pose_al_strategy(raw_strategy: Optional[str]) -> str:
    strategy = str(raw_strategy or "random").strip().lower()
    if strategy in POSE_AL_RANDOM_ALIASES:
        return "random"
    if strategy not in POSE_AL_STRATEGIES:
        raise ValueError(f"Unsupported pose AL strategy '{raw_strategy}'. Expected one of {POSE_AL_STRATEGIES}.")
    return strategy


def normalize_pose_generated_attribute(raw_attribute: Optional[str]) -> str:
    normalized = canonicalize_coco_pose_generated_attribute(raw_attribute)
    return normalized or "none"


def resolve_pose_al_attribute_file(args) -> Optional[Path]:
    if args.al_attribute_file is not None:
        return args.al_attribute_file
    return ANALYSIS_SOURCE_DEFAULTS[args.assigner]["train_attribute_file"]


def resolve_pose_al_focus_attributes(args) -> Optional[List[str]]:
    raw = getattr(args, "al_attribute_focus", None)
    if raw is None:
        return None
    values = [value.strip() for value in str(raw).split(",") if value.strip()]
    return values or None


def resolve_pose_al_extra_counts(args) -> Optional[Dict[str, Dict[str, float]]]:
    raw = getattr(args, "al_attribute_extra_counts", None)
    if isinstance(raw, str) and raw.strip():
        parsed = parse_attribute_value_counts(raw.strip(), default_count=args.num_added_img)
    else:
        generated_attribute = normalize_pose_generated_attribute(
            getattr(args, "generated_attribute", "none")
        )
        template = POSE_GENERATED_ATTRIBUTE_EXTRA_COUNTS.get(generated_attribute)
        if template is None or args.num_added_img <= 0:
            return None
        parsed = {
            attr_name: {
                value: float(multiplier) * float(args.num_added_img)
                for value, multiplier in value_map.items()
            }
            for attr_name, value_map in template.items()
        }

    scale = max(float(getattr(args, "al_attribute_extra_count_scale", 1.0)), 0.0)
    if scale != 1.0:
        parsed = {
            attr_name: {
                value: float(count) * scale
                for value, count in value_map.items()
            }
            for attr_name, value_map in parsed.items()
        }
    return canonicalize_attribute_counts(parsed, dataset_name="coco_pose")


def _count_valid_person_annotations(dataset, image_id: int) -> int:
    ann_ids = dataset.coco.getAnnIds(imgIds=image_id, catIds=[1], iscrowd=None)
    anns = dataset.coco.loadAnns(ann_ids)
    count = 0
    for ann in anns:
        if ann.get("category_id") != 1:
            continue
        if ann.get("iscrowd", 0) == 1:
            continue

        x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
        if w <= 1 or h <= 1:
            continue

        kp = ann.get("keypoints", [])
        if len(kp) != 51:
            continue
        if sum(1 for i in range(2, len(kp), 3) if kp[i] > 0) <= 0:
            continue

        count += 1
    return count


def compute_pose_proxy_targets(dataset, *, n_bins: int = 4) -> Optional[np.ndarray]:
    counts = np.array(
        [_count_valid_person_annotations(dataset, image_id) for image_id in dataset.image_ids],
        dtype=np.int64,
    )
    unique_counts = np.unique(counts)
    if unique_counts.size < 2:
        return None

    n_bins = max(2, min(int(n_bins), unique_counts.size))
    if unique_counts.size <= n_bins:
        mapping = {int(value): idx for idx, value in enumerate(sorted(unique_counts.tolist()))}
        return np.array([mapping[int(value)] for value in counts], dtype=np.int64)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(counts, quantiles))
    if edges.size == 0:
        return None

    targets = np.digitize(counts, edges, right=True).astype(np.int64)
    if np.unique(targets).size < 2:
        return None
    return targets


def build_pose_selection_loader(dataset, *, batch_size: int, num_workers: int, seed: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=build_loader_generator(seed),
    )


def extract_pose_al_embeddings(
    dataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
    feature_model=None,
) -> torch.Tensor:
    loader = build_pose_selection_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    model = feature_model
    if model is None:
        model = tv_models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()
        model = model.to(device)
        model.eval()

    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    embeddings: List[torch.Tensor] = []

    with torch.no_grad():
        for images, _ in loader:
            resized_batch = [
                torch.nn.functional.interpolate(
                    image.unsqueeze(0),
                    size=(POSE_AL_IMAGE_SIZE, POSE_AL_IMAGE_SIZE),
                    mode="bilinear",
                    align_corners=False,
                )
                for image in images
            ]
            batch = torch.cat(resized_batch, dim=0).to(device, non_blocking=True)
            batch = (batch - mean) / std
            features = model(batch)
            embeddings.append(features.detach().cpu())

    if not embeddings:
        return torch.empty((0, 2048), dtype=torch.float32)
    return torch.cat(embeddings, dim=0)


def select_original_train_image_ids(
    full_train_dataset,
    val_dataset,
    *,
    max_train_samples: int,
    sub_ratio: float,
    seed: int,
    args,
) -> Tuple[List[int], Dict[str, object]]:
    if not 0.0 <= sub_ratio <= 1.0:
        raise ValueError(f"--sub-ratio must be within [0, 1], got {sub_ratio}")

    candidate_ids = list(full_train_dataset.image_ids)
    if max_train_samples > 0:
        candidate_ids = candidate_ids[:max_train_samples]

    if sub_ratio >= 1.0 or not candidate_ids:
        return candidate_ids, {"strategy": "full", "selection_time_sec": 0.0}

    selected_count = int(len(candidate_ids) * sub_ratio)
    if 0.0 < sub_ratio < 1.0 and selected_count == 0:
        selected_count = 1
    if selected_count <= 0:
        return [], {"strategy": "empty", "selection_time_sec": 0.0}

    strategy_name = normalize_pose_al_strategy(getattr(args, "al_strategy", "random"))
    effective_strategy = strategy_name
    if not getattr(args, "use_active_learning", True):
        strategy_name = "random"
        effective_strategy = "random"

    selection_start = time.time()

    if strategy_name == "random":
        rng = random.Random(seed)
        selected_image_ids = [candidate_ids[idx] for idx in sorted(rng.sample(range(len(candidate_ids)), selected_count))]
        return selected_image_ids, {
            "strategy": "random",
            "selection_time_sec": time.time() - selection_start,
        }

    selection_dataset = clone_dataset_with_image_ids(
        full_train_dataset,
        candidate_ids,
        train=False,
        hflip_prob=0.0,
    )
    selection_dataset.attribute_metadata_indices = list(range(len(candidate_ids)))
    proxy_targets = compute_pose_proxy_targets(
        selection_dataset,
        n_bins=getattr(args, "al_proxy_bins", 4),
    )
    if proxy_targets is not None:
        selection_dataset.al_targets = proxy_targets.tolist()

    if strategy_name == "stratified":
        if proxy_targets is None:
            print("Pose stratified selection could not build proxy targets; falling back to random.")
            rng = random.Random(seed)
            selected_local_indices = sorted(rng.sample(range(len(candidate_ids)), selected_count))
            effective_strategy = "random"
        else:
            selected_local_indices = StratifiedStrategy().stratified_sample(
                proxy_targets,
                selected_count,
                random_seed=seed,
            )
    elif strategy_name == "attribute":
        attribute_path = resolve_pose_al_attribute_file(args)
        if attribute_path is None or not attribute_path.exists():
            print(f"Pose attribute selection could not find attribute file: {attribute_path}. Falling back to random.")
            rng = random.Random(seed)
            selected_local_indices = sorted(rng.sample(range(len(candidate_ids)), selected_count))
            effective_strategy = "random"
        else:
            attribute_metadata = json.loads(attribute_path.read_text())
            metadata_len = 0
            if attribute_metadata:
                first_attr = next(iter(attribute_metadata.values()))
                if isinstance(first_attr, dict):
                    metadata_len = len(first_attr.get("data", []))
            if metadata_len < len(selection_dataset):
                print(
                    f"Pose attribute selection metadata in {attribute_path} only covers {metadata_len} "
                    f"samples but {len(selection_dataset)} are required; falling back to random."
                )
                rng = random.Random(seed)
                selected_local_indices = sorted(rng.sample(range(len(candidate_ids)), selected_count))
                effective_strategy = "random"
            else:
                if getattr(args, "al_attribute_use_entropy", False):
                    print(
                        "Pose attribute selection currently ignores --al-attribute-use-entropy and "
                        "uses metadata-only scoring."
                    )
                log_path = getattr(args, "al_attribute_log_path", None)
                value_weights = None
                if log_path is not None and Path(log_path).exists():
                    value_weights = parse_attribute_priority_log(str(log_path))
                focus_attributes = resolve_pose_al_focus_attributes(args)
                strategy = AttributeFocusedStrategy(
                    attribute_metadata=attribute_metadata,
                    focus_attributes=focus_attributes,
                    rarity_power=getattr(args, "al_attribute_rarity_power", 0.03),
                    uncertain_boost=getattr(args, "al_attribute_uncertainty_boost", 1.0),
                    value_weights=value_weights,
                    log_acc_scale=getattr(args, "al_attribute_log_acc_scale", 0.0),
                    extra_counts=resolve_pose_al_extra_counts(args),
                    frequency_penalty=getattr(args, "al_attribute_frequency_penalty", 0.5),
                    class_balance_weight=parse_float_with_infinity(
                        getattr(args, "al_attribute_class_balance", "0"),
                        default=0.0,
                    ),
                    conditional_rarity=getattr(args, "al_conditional_rarity", False),
                    use_entropy=False,
                )
                selected_local_indices = strategy.select(
                    selection_dataset,
                    selected_count,
                    random_seed=seed,
                    debug_attribute=(focus_attributes[0] if focus_attributes else None),
                )
    else:
        feature_batch_size = max(1, min(getattr(args, "al_feature_batch_size", 16), 32))
        feature_num_workers = max(0, min(getattr(args, "num_workers", 4), 4))
        selection_device = resolve_device(getattr(args, "device", "cuda"))
        feature_model = tv_models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        feature_model.fc = torch.nn.Identity()
        feature_model = feature_model.to(selection_device)
        feature_model.eval()

        train_embeddings = extract_pose_al_embeddings(
            selection_dataset,
            device=selection_device,
            batch_size=feature_batch_size,
            num_workers=feature_num_workers,
            seed=seed,
            feature_model=feature_model,
        )

        proxy_count = min(len(val_dataset), max(2, int(getattr(args, "al_initial_labeled", 400))))
        proxy_val_dataset = clone_dataset_with_image_ids(
            val_dataset,
            list(val_dataset.image_ids[:proxy_count]),
            train=False,
            hflip_prob=0.0,
        )
        val_embeddings = extract_pose_al_embeddings(
            proxy_val_dataset,
            device=selection_device,
            batch_size=feature_batch_size,
            num_workers=feature_num_workers,
            seed=seed + 1,
            feature_model=feature_model,
        )

        if strategy_name == "coreset":
            selected_local_indices = CoreSetStrategy(
                embedding_dim=int(train_embeddings.shape[1]) if train_embeddings.numel() > 0 else 2048
            ).greedy_k_center(
                labeled_embeddings=val_embeddings,
                unlabeled_embeddings=train_embeddings,
                n_select=selected_count,
                random_seed=seed,
            )
        elif strategy_name == "badge":
            val_proxy_targets = compute_pose_proxy_targets(
                proxy_val_dataset,
                n_bins=getattr(args, "al_proxy_bins", 4),
            )
            if val_proxy_targets is None or np.unique(val_proxy_targets).size < 2:
                print("Pose BADGE could not build proxy targets from validation images; falling back to random.")
                rng = random.Random(seed)
                selected_local_indices = sorted(rng.sample(range(len(candidate_ids)), selected_count))
                effective_strategy = "random"
            else:
                selected_local_indices = BadgeStrategy().select(
                    labeled_embeddings=val_embeddings,
                    unlabeled_embeddings=train_embeddings,
                    labeled_targets=val_proxy_targets,
                    n_select=selected_count,
                    random_seed=seed,
                )
        else:
            raise ValueError(f"Unsupported pose AL strategy '{strategy_name}'")

    selected_local_indices = [int(idx) for idx in selected_local_indices]
    deduped_indices: List[int] = []
    seen = set()
    for idx in selected_local_indices:
        if idx < 0 or idx >= len(candidate_ids) or idx in seen:
            continue
        deduped_indices.append(idx)
        seen.add(idx)
    if len(deduped_indices) < selected_count:
        remaining = [idx for idx in range(len(candidate_ids)) if idx not in seen]
        rng = random.Random(seed)
        fill_count = min(selected_count - len(deduped_indices), len(remaining))
        deduped_indices.extend(sorted(rng.sample(remaining, fill_count)))

    selected_image_ids = [candidate_ids[idx] for idx in sorted(deduped_indices[:selected_count])]
    return selected_image_ids, {
        "strategy": effective_strategy,
        "selection_time_sec": time.time() - selection_start,
    }


def normalize_generated_image_dir(path: Path) -> Path:
    resolved = path.expanduser()
    outputs_dir = resolved / "outputs"
    if outputs_dir.is_dir():
        return outputs_dir
    return resolved


def resolve_generated_image_dir(args) -> Optional[Path]:
    if getattr(args, "generated_image_dir", None) is not None:
        return normalize_generated_image_dir(args.generated_image_dir)

    attribute = normalize_pose_generated_attribute(getattr(args, "generated_attribute", "none"))
    if attribute == "none":
        return None
    return normalize_generated_image_dir(POSE_GENERATED_ATTRIBUTE_ROOTS[attribute])


def discover_generated_pose_images(image_dir: Path) -> Dict[int, str]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Generated image directory not found: {image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Generated image path is not a directory: {image_dir}")

    image_paths: Dict[int, str] = {}
    duplicate_ids = 0
    ignored_files = 0
    for path in sorted(image_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in POSE_GENERATED_IMAGE_SUFFIXES:
            continue
        if not path.stem.isdigit():
            ignored_files += 1
            continue

        image_id = int(path.stem)
        if image_id in image_paths:
            duplicate_ids += 1
            continue
        image_paths[image_id] = str(path)

    if not image_paths:
        raise ValueError(
            f"No generated pose images with numeric stems were found in {image_dir}"
        )
    if duplicate_ids > 0:
        print(
            f"Warning: ignored {duplicate_ids} duplicate generated files in {image_dir} "
            "after the first image_id match."
        )
    if ignored_files > 0:
        print(
            f"Warning: ignored {ignored_files} generated files in {image_dir} because their "
            "filenames do not match a numeric COCO image id."
        )
    return image_paths


def filter_readable_generated_pose_images(image_paths: Dict[int, str]) -> Dict[int, str]:
    readable_paths: Dict[int, str] = {}
    unreadable = []
    for image_id, image_path in sorted(image_paths.items()):
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image.load()
        except Exception as exc:
            unreadable.append((image_id, image_path, exc))
            continue
        readable_paths[image_id] = image_path

    if unreadable:
        preview = ", ".join(
            f"{image_id} ({Path(image_path).name}: {type(exc).__name__})"
            for image_id, image_path, exc in unreadable[:5]
        )
        print(
            f"Warning: filtered out {len(unreadable)} unreadable generated images before training. "
            f"Examples: {preview}"
        )
    return readable_paths


def filter_box_compatible_generated_pose_images(
    image_paths: Dict[int, str],
    coco,
) -> Dict[int, str]:
    compatible_paths: Dict[int, str] = {}
    incompatible = []

    for image_id, image_path in sorted(image_paths.items()):
        with Image.open(image_path) as image:
            width, height = image.size

        ann_ids = coco.getAnnIds(imgIds=image_id, catIds=[1], iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        has_invalid_box = False
        for ann in anns:
            if ann.get("category_id") != 1:
                continue
            if ann.get("iscrowd", 0) == 1:
                continue

            x, y, w, h = ann["bbox"]
            if w <= 1 or h <= 1:
                continue

            kp = ann.get("keypoints", [])
            if len(kp) != 51:
                continue
            if sum(1 for i in range(2, len(kp), 3) if kp[i] > 0) <= 0:
                continue

            x1 = max(0.0, float(x))
            y1 = max(0.0, float(y))
            x2 = max(x1 + 1.0, float(x + w))
            y2 = max(y1 + 1.0, float(y + h))

            x1 = min(x1, float(width))
            x2 = min(x2, float(width))
            y1 = min(y1, float(height))
            y2 = min(y2, float(height))
            if x2 <= x1 or y2 <= y1:
                has_invalid_box = True
                break

        if has_invalid_box:
            incompatible.append((image_id, width, height))
            continue
        compatible_paths[image_id] = image_path

    if incompatible:
        preview = ", ".join(
            f"{image_id} ({width}x{height})"
            for image_id, width, height in incompatible[:5]
        )
        print(
            f"Warning: filtered out {len(incompatible)} generated images whose reused COCO boxes "
            f"collapse under the generated image size. Examples: {preview}"
        )
    return compatible_paths


def select_generated_image_ids(
    image_ids: Sequence[int],
    *,
    max_generated_samples: int,
    seed: int,
) -> List[int]:
    if max_generated_samples < 0:
        raise ValueError(f"--num-added-img must be >= 0, got {max_generated_samples}")

    available_ids = list(image_ids)
    if max_generated_samples == 0 or not available_ids:
        return []
    if len(available_ids) <= max_generated_samples:
        return available_ids

    rng = random.Random(seed)
    return sorted(rng.sample(available_ids, max_generated_samples))


def build_loaders(args) -> Dict[str, DataLoader]:
    full_train_dataset, val_dataset, test_dataset = build_train_val_test_datasets(
        data_root=args.data_root,
        train_images=args.train_images,
        val_images=args.val_images,
        train_annotations=args.train_annotations,
        val_annotations=args.val_annotations,
        hflip_prob=getattr(args, "hflip_prob", 0.5),
        seed=args.seed,
        max_train_samples=0,
    )

    full_train_image_ids = list(full_train_dataset.image_ids)
    original_candidate_ids = list(full_train_image_ids)
    if args.max_train_samples > 0:
        original_candidate_ids = original_candidate_ids[: args.max_train_samples]
    capped_original_count = len(original_candidate_ids)
    selected_original_ids, selection_info = select_original_train_image_ids(
        full_train_dataset,
        val_dataset,
        max_train_samples=args.max_train_samples,
        sub_ratio=args.sub_ratio,
        seed=args.seed,
        args=args,
    )
    original_train_dataset = clone_dataset_with_image_ids(
        full_train_dataset,
        selected_original_ids,
        train=True,
        hflip_prob=getattr(args, "hflip_prob", 0.5),
    )

    generated_available_count = 0
    generated_selected_count = 0
    generated_image_dir = resolve_generated_image_dir(args)
    train_dataset = original_train_dataset

    if generated_image_dir is not None:
        generated_image_paths = discover_generated_pose_images(generated_image_dir)
        generated_image_paths = filter_readable_generated_pose_images(generated_image_paths)
        generated_image_paths = filter_box_compatible_generated_pose_images(
            generated_image_paths,
            full_train_dataset.coco,
        )
        valid_generated_ids = sorted(
            set(generated_image_paths).intersection(full_train_image_ids)
        )
        generated_available_count = len(valid_generated_ids)
        if generated_available_count == 0:
            raise ValueError(
                f"No readable generated image ids from {generated_image_dir} overlap the valid COCO pose train split."
            )
        if generated_available_count < len(generated_image_paths):
            filtered_count = len(generated_image_paths) - generated_available_count
            print(
                f"Filtered out {filtered_count} generated images from {generated_image_dir} "
                "because they are not valid COCO pose-train image ids."
            )

        selected_generated_ids = select_generated_image_ids(
            valid_generated_ids,
            max_generated_samples=args.num_added_img,
            seed=args.seed,
        )
        generated_selected_count = len(selected_generated_ids)
        if generated_selected_count < args.num_added_img:
            print(
                f"Warning: requested {args.num_added_img} generated images, but only "
                f"{generated_selected_count} readable generated images are available in {generated_image_dir}."
            )

        if generated_selected_count > 0:
            generated_train_dataset = clone_dataset_with_image_ids(
                full_train_dataset,
                selected_generated_ids,
                train=True,
                hflip_prob=getattr(args, "hflip_prob", 0.5),
                image_path_overrides={
                    image_id: generated_image_paths[image_id]
                    for image_id in selected_generated_ids
                },
            )
            datasets = []
            if len(original_train_dataset) > 0:
                datasets.append(original_train_dataset)
            datasets.append(generated_train_dataset)
            train_dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    if len(train_dataset) == 0:
        raise ValueError(
            "The training dataset is empty after applying --max-train-samples, --sub-ratio, "
            "and generated-image selection."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=build_loader_generator(args.seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=build_loader_generator(args.seed + 1),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=build_loader_generator(args.seed + 2),
    )

    if generated_image_dir is not None:
        normalized_generated_attribute = normalize_pose_generated_attribute(
            getattr(args, "generated_attribute", "none")
        )
        print(
            f"Generated augmentation source: attribute={normalized_generated_attribute}, "
            f"dir={generated_image_dir}"
        )
    print(
        f"Train sampling summary: original_candidates={capped_original_count}, "
        f"selected_original={len(original_train_dataset)} (sub_ratio={args.sub_ratio}), "
        f"selection_strategy={selection_info['strategy']}, "
        f"selection_time={selection_info['selection_time_sec']:.2f}s, "
        f"generated_available={generated_available_count}, "
        f"generated_selected={generated_selected_count}, final_train={len(train_dataset)}"
    )
    print(
        f"Dataset sizes after split: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
    )
    return {"train": train_loader, "val": val_loader, "test": test_loader}


def parse_csv_choices(raw: str, allowed: Sequence[str], flag_name: str) -> List[str]:
    values = [value.strip().lower() for value in raw.split(",") if value.strip()]
    invalid = [value for value in values if value not in allowed]
    if invalid:
        raise ValueError(f"Invalid values for {flag_name}: {invalid}. Allowed: {sorted(allowed)}")
    deduped: List[str] = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def resolve_assigner_corpus(args) -> Path:
    if args.assigner_corpus is not None:
        return args.assigner_corpus
    if args.analysis_corpus is not None:
        return args.analysis_corpus
    return Path("save/corpus_coco_pose_gemini.json")


def resolve_assigner_output_root(args) -> Path:
    return args.assigner_output_root or (args.output_dir / "assigners")


def resolve_analysis_source_config(
    args,
    assigner_artifacts: Optional[Dict[str, Dict[str, Dict[str, Path]]]] = None,
) -> Dict[str, object]:
    if args.analysis_attribute_file is not None:
        return {
            "tag": "manual",
            "attribute_file": args.analysis_attribute_file,
            "attribute_split": args.analysis_attribute_split,
            "train_attribute_file": args.analysis_train_attribute_file,
            "train_attribute_split": args.analysis_train_attribute_split,
        }

    if assigner_artifacts and args.assigner in assigner_artifacts:
        outputs = assigner_artifacts[args.assigner]
        eval_source = outputs.get("val") or outputs.get("all")
        train_source = outputs.get("train") or outputs.get("all")
        if eval_source is not None:
            return {
                "tag": args.assigner,
                "attribute_file": eval_source["json"],
                "attribute_split": "val_full" if "val" in outputs else "all",
                "train_attribute_file": train_source["json"] if train_source is not None else None,
                "train_attribute_split": "train" if "train" in outputs else ("all" if "all" in outputs else "train"),
            }

    config = ANALYSIS_SOURCE_DEFAULTS[args.assigner]
    return {
        "tag": args.assigner,
        "attribute_file": config["attribute_file"],
        "attribute_split": config["attribute_split"],
        "train_attribute_file": config["train_attribute_file"],
        "train_attribute_split": config["train_attribute_split"],
    }


def run_command(command: Sequence[str]) -> None:
    print("Running command:")
    print(" ".join(str(part) for part in command))
    subprocess.run(
        [str(part) for part in command],
        check=True,
        cwd=str(REPO_ROOT),
    )


def run_single_assigner(args, *, assigner: str, split: str, corpus_path: Path) -> Dict[str, Path]:
    output_root = resolve_assigner_output_root(args)
    artifact_dir = output_root / assigner / split
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if assigner == "blip":
        command: List[object] = [
            sys.executable,
            "-u",
            "utils/blip.py",
            "--dataset",
            "coco_pose",
            "--coco_root",
            args.data_root,
            "--split",
            split,
            "--corpus",
            corpus_path,
            "--with_attribute_selection",
            "--output_labels",
            artifact_dir / "labels.npy",
            "--output_attributes",
            artifact_dir / "attributes.json",
            "--csv_output",
            artifact_dir / "attributes.csv",
            "--text_labels_output",
            artifact_dir / "text_labels.npy",
            "--train_idx_output",
            artifact_dir / "train_idx.npy",
            "--val_idx_output",
            artifact_dir / "val_idx.npy",
        ]
        if split == "train" and args.max_train_samples > 0:
            command.extend(["--stop_at", str(args.max_train_samples)])
    else:
        model_name = ASSIGNER_MODEL_MAP[assigner]
        command = [
            sys.executable,
            "-u",
            "utils/assign_attribute.py",
            "--dataset",
            "coco_pose",
            "--coco_root",
            args.data_root,
            "--split",
            split,
            "--corpus",
            corpus_path,
            "--model",
            model_name,
            "--output",
            artifact_dir / "attributes.json",
            "--csv-output",
            artifact_dir / "attributes.csv",
            "--partial",
            artifact_dir / "attributes_partial.json",
            "--save_every",
            str(args.assigner_save_every),
            "--max_memory_per_gpu",
            str(args.assigner_max_memory_per_gpu),
        ]
        if split == "train" and args.max_train_samples > 0:
            command.extend(["--stop_at", str(args.max_train_samples)])
        if args.assigner_verbose:
            command.append("--verbose")

    run_command(command)
    artifacts = {
        "json": artifact_dir / "attributes.json",
        "csv": artifact_dir / "attributes.csv",
        "dir": artifact_dir,
    }
    if assigner != "blip":
        artifacts["partial"] = artifact_dir / "attributes_partial.json"
    return artifacts


def maybe_run_assigners_after_train(args) -> Dict[str, Dict[str, Dict[str, Path]]]:
    if not args.assign_after_train:
        return {}

    assigners = parse_csv_choices(
        args.assigners,
        allowed={"blip", "qwen3", "internvl3"},
        flag_name="--assigners",
    )
    splits = parse_csv_choices(
        args.assigner_splits,
        allowed={"train", "val", "all"},
        flag_name="--assigner-splits",
    )
    corpus_path = resolve_assigner_corpus(args)

    artifacts: Dict[str, Dict[str, Dict[str, Path]]] = {}
    for assigner in assigners:
        artifacts[assigner] = {}
        for split in splits:
            print(f"Running assigner '{assigner}' on split '{split}'...")
            artifacts[assigner][split] = run_single_assigner(
                args,
                assigner=assigner,
                split=split,
                corpus_path=corpus_path,
            )
    return artifacts


def build_analysis_args(
    *,
    args,
    predictions_json: Path,
    eval_split: str,
    output_json: Path,
    output_csv: Path,
    attribute_file: Path,
    attribute_split: str,
    train_attribute_file: Optional[Path],
    train_attribute_split: str,
) -> argparse.Namespace:
    paths = default_coco_paths(args.data_root)
    if eval_split == "val":
        eval_annotation_file = args.val_annotations or paths["val_annotations"]
        eval_image_dir = args.val_images or paths["val_images"]
    elif eval_split == "test":
        eval_annotation_file = args.test_annotations or paths["test_annotations"]
        eval_image_dir = args.test_images or paths["test_images"]
    else:
        raise ValueError(f"Unsupported analysis split: {eval_split}")

    return argparse.Namespace(
        predictions_json=predictions_json,
        attribute_file=attribute_file,
        attribute_split=attribute_split,
        train_attribute_file=train_attribute_file,
        train_attribute_split=train_attribute_split,
        corpus=args.analysis_corpus or resolve_assigner_corpus(args),
        attributes=args.analysis_attributes,
        data_root=args.data_root,
        eval_annotation_file=eval_annotation_file,
        eval_image_dir=eval_image_dir,
        eval_split=eval_split,
        train_annotation_file=args.train_annotations,
        train_image_dir=args.train_images,
        min_images=args.analysis_min_images,
        top_k=args.analysis_top_k,
        sort_by=args.analysis_sort_by,
        output_json=output_json,
        output_csv=output_csv,
        seed=args.seed,
        acc_diff_threshold=args.analysis_acc_diff_threshold,
        distribution_diff_threshold=args.analysis_distribution_diff_threshold,
        rare_label_threshold=args.analysis_rare_label_threshold,
        train_max_samples=args.max_train_samples,
    )


def maybe_run_post_train_analysis(
    args,
    *,
    val_predictions_path: Optional[Path],
    test_predictions_path: Optional[Path],
    assigner_artifacts: Optional[Dict[str, Dict[str, Dict[str, Path]]]] = None,
) -> None:
    if not args.analyze_after_train:
        return

    analysis_dir = args.output_dir / "slice_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    source = resolve_analysis_source_config(args, assigner_artifacts=assigner_artifacts)
    source_dir = analysis_dir / source["tag"]
    source_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    if val_predictions_path is not None and val_predictions_path.exists():
        tasks.append(
            (
                "val",
                build_analysis_args(
                    args=args,
                    predictions_json=val_predictions_path,
                    eval_split="val",
                    output_json=source_dir / "val_slice_stats.json",
                    output_csv=source_dir / "val_slice_stats.csv",
                    attribute_file=source["attribute_file"],
                    attribute_split=source["attribute_split"],
                    train_attribute_file=source["train_attribute_file"],
                    train_attribute_split=source["train_attribute_split"],
                ),
            )
        )
    if test_predictions_path is not None and test_predictions_path.exists():
        tasks.append(
            (
                "test",
                build_analysis_args(
                    args=args,
                    predictions_json=test_predictions_path,
                    eval_split="test",
                    output_json=source_dir / "test_slice_stats.json",
                    output_csv=source_dir / "test_slice_stats.csv",
                    attribute_file=source["attribute_file"],
                    attribute_split=source["attribute_split"],
                    train_attribute_file=source["train_attribute_file"],
                    train_attribute_split=source["train_attribute_split"],
                ),
            )
        )

    for split_name, analysis_args in tasks:
        print(f"Running post-train slice analysis on {split_name} split using '{source['tag']}' attributes...")
        try:
            run_pose_slice_analysis(analysis_args)
        except Exception as exc:
            print(f"Post-train slice analysis failed on {split_name} using '{source['tag']}': {exc}")


def run_train(args):
    overall_start_time = time.time()
    print("Current time:", datetime.datetime.now().strftime("%m%d%H%M%S"))
    print(f"Training start timestamp: {_timestamp_str(overall_start_time)}")

    if args.resume is not None and args.resume_model is not None:
        raise ValueError("Use only one of --resume or --resume-model.")

    set_seed(args.seed)
    print(
        f"Deterministic settings: seed={args.seed}, cudnn.deterministic={cudnn.deterministic}, "
        f"cudnn.benchmark={cudnn.benchmark}, CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}"
    )
    device = resolve_device(args.device)
    loaders = build_loaders(args)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    model = create_keypointrcnn_model(num_keypoints=17, pretrained=args.pretrained).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma,
    )

    start_epoch = 0
    best_ap = -1.0
    if args.resume is not None:
        resume_metadata = read_checkpoint_metadata(
            checkpoint_path=args.resume,
            map_location="cpu",
        )
        if not resume_metadata.get("has_optimizer", False):
            raise ValueError(
                f"{args.resume} does not contain optimizer state. "
                "Use --resume-model for model-only continuation, or rerun training with "
                "--save-resume-checkpoint true and resume from last_resume.pth."
            )
        start_epoch = load_checkpoint(
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            map_location="cpu",
        )
        if resume_metadata.get("best_ap") is not None:
            best_ap = float(resume_metadata["best_ap"])
    elif args.resume_model is not None:
        resume_metadata = load_model_checkpoint(
            checkpoint_path=args.resume_model,
            model=model,
            map_location="cpu",
        )
        if resume_metadata.get("epoch") is not None:
            start_epoch = int(resume_metadata["epoch"]) + 1
        if resume_metadata.get("best_ap") is not None:
            best_ap = float(resume_metadata["best_ap"])

        print(
            "Using fresh optimizer / scheduler state after --resume-model. "
            f"Training will start from epoch index {start_epoch} with the current "
            f"optimizer hyperparameters (lr={args.lr}, momentum={args.momentum}, "
            f"weight_decay={args.weight_decay}, step_size={args.lr_step_size}, gamma={args.lr_gamma})."
        )

    if start_epoch >= args.epochs:
        print(
            f"Warning: start_epoch={start_epoch} is already >= args.epochs={args.epochs}. "
            "No additional training epochs will run unless you increase --epochs."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    total_train_time = 0.0
    total_eval_time = 0.0
    train_loop_start_time = time.time()
    best_checkpoint_path = args.output_dir / "best.pth"
    last_checkpoint_path = args.output_dir / "last.pth"
    last_resume_checkpoint_path = args.output_dir / "last_resume.pth"
    assigner_artifacts: Dict[str, Dict[str, Dict[str, Path]]] = {}

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        metrics = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            print_freq=50,
        )
        lr_scheduler.step()
        epoch_train_time = float(metrics.get("epoch_time_sec", time.time() - epoch_start_time))
        total_train_time += epoch_train_time

        eval_metrics = {}
        if (epoch + 1) % args.eval_every == 0:
            eval_start_time = time.time()
            eval_metrics = evaluate_coco_keypoints(
                model=model,
                data_loader=val_loader,
                coco_gt=val_loader.dataset.coco,
                device=device,
                score_threshold=args.score_threshold,
                result_json_path=args.output_dir / f"val_predictions_epoch_{epoch:03d}.json",
            )
            eval_duration = time.time() - eval_start_time
            total_eval_time += eval_duration
            eval_metrics["eval_time_sec"] = eval_duration
            if eval_metrics["AP"] > best_ap:
                best_ap = eval_metrics["AP"]
                save_model_checkpoint(
                    checkpoint_path=best_checkpoint_path,
                    model=model,
                    epoch=epoch,
                    extra={"best_ap": best_ap},
                )

        if args.save_last_checkpoint:
            save_model_checkpoint(
                checkpoint_path=last_checkpoint_path,
                model=model,
                epoch=epoch,
                extra={"best_ap": best_ap},
            )
        if args.save_resume_checkpoint:
            save_checkpoint(
                checkpoint_path=last_resume_checkpoint_path,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                epoch=epoch,
                extra={"best_ap": best_ap},
            )

        row = {"epoch": epoch, **metrics, **eval_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

    with (args.output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    final_eval_checkpoint: Optional[Path] = None
    if best_checkpoint_path.exists():
        final_eval_checkpoint = best_checkpoint_path
    elif last_checkpoint_path.exists():
        final_eval_checkpoint = last_checkpoint_path

    best_val_predictions_path: Optional[Path] = None
    test_predictions_path: Optional[Path] = None
    if final_eval_checkpoint is not None:
        load_model_checkpoint(
            checkpoint_path=final_eval_checkpoint,
            model=model,
            map_location="cpu",
        )
        best_val_predictions_path = args.output_dir / "val_predictions_best.json"
        best_val_metrics = evaluate_coco_keypoints(
            model=model,
            data_loader=val_loader,
            coco_gt=val_loader.dataset.coco,
            device=device,
            score_threshold=args.score_threshold,
            result_json_path=best_val_predictions_path,
        )
        with (args.output_dir / "val_metrics_best.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "checkpoint": str(final_eval_checkpoint),
                    "metrics": best_val_metrics,
                },
                f,
                indent=2,
            )
        print("Held-out val metrics (best checkpoint):")
        print(json.dumps(best_val_metrics, ensure_ascii=False, indent=2))

        test_predictions_path = args.output_dir / "test_predictions.json"
        test_metrics = evaluate_coco_keypoints(
            model=model,
            data_loader=test_loader,
            coco_gt=test_loader.dataset.coco,
            device=device,
            score_threshold=args.score_threshold,
            result_json_path=test_predictions_path,
        )
        with (args.output_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "checkpoint": str(final_eval_checkpoint),
                    "metrics": test_metrics,
                },
                f,
                indent=2,
            )
        print("Held-out test metrics:")
        print(json.dumps(test_metrics, ensure_ascii=False, indent=2))
        assigner_artifacts = maybe_run_assigners_after_train(args)
        maybe_run_post_train_analysis(
            args,
            val_predictions_path=best_val_predictions_path,
            test_predictions_path=test_predictions_path,
            assigner_artifacts=assigner_artifacts,
        )
    else:
        print(
            "No saved best/last model checkpoint was found after training. "
            "Skipping the post-train reload step and using the in-memory final model for evaluation."
        )
        best_val_predictions_path = args.output_dir / "val_predictions_best.json"
        best_val_metrics = evaluate_coco_keypoints(
            model=model,
            data_loader=val_loader,
            coco_gt=val_loader.dataset.coco,
            device=device,
            score_threshold=args.score_threshold,
            result_json_path=best_val_predictions_path,
        )
        with (args.output_dir / "val_metrics_best.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "checkpoint": None,
                    "metrics": best_val_metrics,
                },
                f,
                indent=2,
            )
        print("Held-out val metrics (final in-memory model):")
        print(json.dumps(best_val_metrics, ensure_ascii=False, indent=2))

        test_predictions_path = args.output_dir / "test_predictions.json"
        test_metrics = evaluate_coco_keypoints(
            model=model,
            data_loader=test_loader,
            coco_gt=test_loader.dataset.coco,
            device=device,
            score_threshold=args.score_threshold,
            result_json_path=test_predictions_path,
        )
        with (args.output_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "checkpoint": None,
                    "metrics": test_metrics,
                },
                f,
                indent=2,
            )
        print("Held-out test metrics:")
        print(json.dumps(test_metrics, ensure_ascii=False, indent=2))
        assigner_artifacts = maybe_run_assigners_after_train(args)
        maybe_run_post_train_analysis(
            args,
            val_predictions_path=best_val_predictions_path,
            test_predictions_path=test_predictions_path,
            assigner_artifacts=assigner_artifacts,
        )

    end_timestamp = time.time()
    train_loop_elapsed = end_timestamp - train_loop_start_time
    overall_elapsed_time = end_timestamp - overall_start_time
    print(f"Total training time for {args.epochs} epochs: {train_loop_elapsed:.2f} seconds")
    print(f"Total training-only epoch time: {total_train_time:.2f} seconds")
    print(f"Total evaluation time: {total_eval_time:.2f} seconds")
    print(
        "End-to-end wall time (data prep + training + evaluation + setup): "
        f"{overall_elapsed_time:.2f} seconds"
    )
    print(f"Training end timestamp: {_timestamp_str(end_timestamp)}")
    print(f"Training elapsed (HH:MM:SS): {_format_duration(train_loop_elapsed)}")
    print(f"End-to-end elapsed (HH:MM:SS): {_format_duration(overall_elapsed_time)}")
    print(f"Training complete. Artifacts: {args.output_dir}")


def run_eval(args):
    device = resolve_device(args.device)
    paths = default_coco_paths(args.data_root)
    _, val_dataset = build_datasets(
        data_root=args.data_root,
        train_images=args.train_images,
        val_images=args.val_images,
        train_annotations=args.train_annotations,
        val_annotations=args.val_annotations,
        hflip_prob=0.0,
        seed=args.seed,
        max_train_samples=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = create_keypointrcnn_model(num_keypoints=17, pretrained=args.pretrained).to(device)
    if args.checkpoint is not None:
        load_checkpoint(
            checkpoint_path=args.checkpoint,
            model=model,
            optimizer=None,
            lr_scheduler=None,
            map_location="cpu",
        )
    else:
        print("No checkpoint specified; evaluating default pretrained model.")

    print(
        f"Evaluating on held-out val split: images={args.val_images or paths['val_images']}, "
        f"annotations={args.val_annotations or paths['val_annotations']}, seed={args.seed}, "
        f"num_images={len(val_dataset)}"
    )
    metrics = evaluate_coco_keypoints(
        model=model,
        data_loader=val_loader,
        coco_gt=val_loader.dataset.coco,
        device=device,
        score_threshold=args.score_threshold,
        result_json_path=args.output_json,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def run_test(args):
    """Final test evaluation on the held-out half of valid val2017 images."""
    device = resolve_device(args.device)
    paths = default_coco_paths(args.data_root)

    test_dataset = build_test_dataset(
        data_root=args.data_root,
        test_images=args.test_images,
        test_annotations=args.test_annotations,
        seed=args.seed,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = create_keypointrcnn_model(num_keypoints=17, pretrained=args.pretrained).to(device)
    load_checkpoint(
        checkpoint_path=args.checkpoint,
        model=model,
        optimizer=None,
        lr_scheduler=None,
        map_location="cpu",
    )

    test_img_dir = args.test_images or paths["test_images"]
    test_ann_file = args.test_annotations or paths["test_annotations"]
    print(f"Testing on held-out test split: images={test_img_dir}, annotations={test_ann_file}, seed={args.seed}")
    print(f"Test set size: {len(test_dataset)} images")

    metrics = evaluate_coco_keypoints(
        model=model,
        data_loader=test_loader,
        coco_gt=test_loader.dataset.coco,
        device=device,
        score_threshold=args.score_threshold,
        result_json_path=args.output_json,
    )

    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    print(f"  AP:   {metrics['AP']:.4f}")
    print(f"  AP50: {metrics['AP50']:.4f}")
    print(f"  AP75: {metrics['AP75']:.4f}")
    print(f"  AR:   {metrics['AR']:.4f}")
    print("=" * 50)

    # Save test results to JSON
    result_path = args.output_json
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "test_images": test_img_dir,
                "test_annotations": test_ann_file,
                "num_test_images": len(test_dataset),
                "metrics": metrics,
            },
            f,
            indent=2,
        )
    print(f"Test results saved to {result_path}")


@torch.no_grad()
def run_infer(args):
    device = resolve_device(args.device)
    model = create_keypointrcnn_model(num_keypoints=17, pretrained=args.pretrained).to(device)
    if args.checkpoint is not None:
        load_checkpoint(
            checkpoint_path=args.checkpoint,
            model=model,
            optimizer=None,
            lr_scheduler=None,
            map_location="cpu",
        )
    model.eval()

    image = Image.open(args.image).convert("RGB")
    image_tensor = image_to_tensor(image).to(device)

    predictions = model([image_tensor])[0]
    canvas = draw_pose_predictions(
        image=image,
        prediction=predictions,
        score_threshold=args.score_threshold,
        keypoint_threshold=args.keypoint_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"Saved inference visualization to {args.output}")


def main():
    args = parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "eval":
        run_eval(args)
    elif args.command == "test":
        run_test(args)
    elif args.command == "infer":
        run_infer(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
