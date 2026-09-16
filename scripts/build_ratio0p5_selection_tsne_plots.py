#!/usr/bin/env python3
"""Build ratio-specific selection t-SNE snapshots for the four main datasets.

KITTI uses exact persisted selected ids. For datasets whose main runs did not
persist selected ids, this script reconstructs deterministic random/attribute
selections from the same metadata and seed, and marks that source in the CSV.
The embedding is a lightweight image-thumbnail proxy intended for selection
coverage visualization, not a semantic model feature.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.active_learning import AttributeFocusedStrategy


OUT_DIR = Path("figures")
SEED = 42

PALETTE = {
    "Random": "#4C72B0",
    "Stratified": "#55A868",
    "Entropy": "#8172B3",
    "Coreset": "#64B5CD",
    "BADGE": "#CCB974",
    "UHerding": "#8C8C8C",
    "PPAL": "#937860",
    "Ours": "#C44E52",
}


@dataclass(frozen=True)
class Selection:
    name: str
    indices: set[int] | None
    metric: float
    source: str


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    slug: str
    target: str
    metric_name: str
    ids: list[str]
    paths: list[Path]
    selections: list[Selection]
    max_plot_points: int


class MinimalDataset:
    def __init__(self, n: int, labels: list[int] | None = None) -> None:
        self.n = n
        self.labels = labels
        self.targets = labels
        self.attribute_metadata_indices = list(range(n))

    def __len__(self) -> int:
        return self.n


def image_embedding(path: Path, side: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((side, side), Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
    flat = arr.reshape(-1)
    stats = [arr.mean(axis=(0, 1)), arr.std(axis=(0, 1)), arr.min(axis=(0, 1)), arr.max(axis=(0, 1))]
    return np.concatenate([flat, *stats], axis=0).astype(np.float32)


def stratified_indices(labels: list[int], n_select: int, seed: int) -> set[int]:
    rng = random.Random(seed)
    label_to_indices: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(int(label), []).append(idx)
    selected: list[int] = []
    for label, candidates in sorted(label_to_indices.items()):
        quota = int(round(len(candidates) * n_select / len(labels)))
        quota = min(quota, len(candidates))
        selected.extend(rng.sample(candidates, quota))
    if len(selected) < n_select:
        seen = set(selected)
        remaining = [idx for idx in range(len(labels)) if idx not in seen]
        selected.extend(rng.sample(remaining, n_select - len(selected)))
    elif len(selected) > n_select:
        selected = rng.sample(selected, n_select)
    return set(selected)


def random_indices(n_total: int, n_select: int, seed: int) -> set[int]:
    return set(random.Random(seed).sample(range(n_total), n_select))


def ratio_tag(ratio: float) -> str:
    return f"ratio{ratio:g}".replace(".", "p")


def read_exported_selection(dataset: str, ratio: float, method: str, id_to_idx: dict[str, int]) -> set[int] | None:
    method_key = "ours" if method == "Ours" else method.lower()
    path = Path("save/selection_exports") / ratio_tag(ratio) / dataset / method_key / "selected_original_ids.txt"
    if not path.exists():
        return None
    selected = {
        id_to_idx[line.strip()]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() in id_to_idx
    }
    return selected


def selection_from_export_or_fallback(
    *,
    dataset: str,
    ratio: float,
    method: str,
    ids: list[str],
    fallback: set[int] | None,
    metric: float,
    fallback_source: str,
) -> Selection:
    exported = read_exported_selection(dataset, ratio, method, {image_id: idx for idx, image_id in enumerate(ids)})
    if exported is not None:
        return Selection(method, exported, metric, "save/selection_exports")
    return Selection(method, fallback, metric, fallback_source)


def attribute_indices(
    *,
    n_total: int,
    n_select: int,
    attribute_file: Path,
    focus: list[str],
    seed: int,
    labels: list[int] | None = None,
    rarity_power: float,
    frequency_penalty: float,
    class_balance: float,
    use_rarity: bool = True,
    extra_counts: dict[str, dict[str, float]] | None = None,
) -> set[int]:
    metadata = json.loads(attribute_file.read_text(encoding="utf-8"))
    dataset = MinimalDataset(n_total, labels)
    strategy = AttributeFocusedStrategy(
        attribute_metadata=metadata,
        focus_attributes=focus,
        rarity_power=rarity_power,
        frequency_penalty=frequency_penalty,
        class_balance_weight=class_balance,
        log_acc_scale=0.0,
        extra_counts=extra_counts,
        use_entropy=False,
        use_rarity=use_rarity,
    )
    return set(strategy.select(dataset, n_select, random_seed=seed))


def load_imagenet(ratio: float) -> DatasetSpec:
    metrics = {
        0.05: {
            "Random": np.nan,
            "Stratified": np.nan,
            "Entropy": np.nan,
            "Coreset": np.nan,
            "BADGE": np.nan,
            "PPAL": np.nan,
            "UHerding": np.nan,
            "Ours": np.nan,
        },
        0.1: {
            "Random": 0.7269,
            "Stratified": 0.7271,
            "Entropy": 0.7254,
            "Coreset": 0.7269,
            "BADGE": 0.7323,
            "PPAL": 0.7285,
            "UHerding": 0.7331,
            "Ours": 0.7369,
        },
        0.3: {
            "Random": np.nan,
            "Stratified": np.nan,
            "Entropy": np.nan,
            "Coreset": np.nan,
            "BADGE": np.nan,
            "PPAL": np.nan,
            "UHerding": np.nan,
            "Ours": np.nan,
        },
        0.5: {
            "Random": 0.7400,
            "Stratified": 0.7446,
            "Entropy": 0.7331,
            "Coreset": 0.7400,
            "BADGE": 0.7469,
            "PPAL": 0.7400,
            "UHerding": 0.7423,
            "Ours": 0.7508,
        },
    }
    if ratio not in metrics:
        raise ValueError(f"ImageNet metrics are not configured for ratio {ratio}")
    root = Path("data/imagenet_subset")
    classes = sorted(path.name for path in root.iterdir() if path.is_dir())
    samples: list[tuple[Path, int]] = []
    for label, cls in enumerate(classes):
        for suffix in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            samples.extend((path, label) for path in sorted((root / cls).glob(suffix)))
    train_idx = np.load("save/train_idx_imagenet.npy")
    ids = [str(int(idx)) for idx in train_idx]
    paths = [samples[int(idx)][0] for idx in train_idx]
    labels = [samples[int(idx)][1] for idx in train_idx]
    n_select = int(len(ids) * ratio)
    ours = attribute_indices(
        n_total=len(ids),
        n_select=n_select,
        attribute_file=Path("save/attribute10_internvl3_8B.json"),
        focus=["color", "background", "texture"],
        seed=SEED,
        labels=labels,
        rarity_power=0.01,
        frequency_penalty=0.0,
        class_balance=float("inf"),
    )
    return DatasetSpec(
        name="ImageNet",
        slug=f"imagenet_controlnet_{ratio_tag(ratio)}",
        target="pinkcolor_fabrictexture",
        metric_name="Test ACC",
        ids=ids,
        paths=paths,
        selections=[
            selection_from_export_or_fallback(dataset="imagenet", ratio=ratio, method="Random", ids=ids, fallback=random_indices(len(ids), n_select, SEED), metric=metrics[ratio]["Random"], fallback_source="reconstructed"),
            selection_from_export_or_fallback(dataset="imagenet", ratio=ratio, method="Stratified", ids=ids, fallback=stratified_indices(labels, n_select, SEED), metric=metrics[ratio]["Stratified"], fallback_source="reconstructed"),
            selection_from_export_or_fallback(dataset="imagenet", ratio=ratio, method="Entropy", ids=ids, fallback=None, metric=metrics[ratio]["Entropy"], fallback_source="selection ids unavailable"),
            selection_from_export_or_fallback(dataset="imagenet", ratio=ratio, method="Coreset", ids=ids, fallback=None, metric=metrics[ratio]["Coreset"], fallback_source="selection ids unavailable"),
            selection_from_export_or_fallback(dataset="imagenet", ratio=ratio, method="BADGE", ids=ids, fallback=None, metric=metrics[ratio]["BADGE"], fallback_source="selection ids unavailable"),
            selection_from_export_or_fallback(dataset="imagenet", ratio=ratio, method="PPAL", ids=ids, fallback=None, metric=metrics[ratio]["PPAL"], fallback_source="selection ids unavailable"),
            selection_from_export_or_fallback(dataset="imagenet", ratio=ratio, method="UHerding", ids=ids, fallback=None, metric=metrics[ratio]["UHerding"], fallback_source="selection ids unavailable"),
            selection_from_export_or_fallback(dataset="imagenet", ratio=ratio, method="Ours", ids=ids, fallback=ours, metric=metrics[ratio]["Ours"], fallback_source="reconstructed attribute no_p"),
        ],
        max_plot_points=10400,
    )


def load_celeba(ratio: float) -> DatasetSpec:
    metrics = {
        0.1: {
            "Random": 0.9103,
            "Stratified": 0.9080,
            "Entropy": 0.9124,
            "Coreset": 0.9000,
            "BADGE": 0.9087,
            "PPAL": 0.9113,
            "UHerding": 0.9047,
            "Ours": 0.9137,
        },
        0.3: {
            "Random": np.nan,
            "Stratified": np.nan,
            "Entropy": np.nan,
            "Coreset": np.nan,
            "BADGE": np.nan,
            "PPAL": np.nan,
            "UHerding": np.nan,
            "Ours": np.nan,
        },
        0.5: {
            "Random": 0.9114,
            "Stratified": 0.9132,
            "Entropy": 0.9129,
            "Coreset": 0.9134,
            "BADGE": 0.9084,
            "PPAL": 0.9151,
            "UHerding": 0.9078,
            "Ours": 0.9187,
        },
    }
    if ratio not in metrics:
        raise ValueError(f"CelebA metrics are not configured for ratio {ratio}")
    with Path("data/celebA/list_attr_celeba.txt").open(encoding="utf-8") as handle:
        names = [line.strip().split(" ")[0] for line in handle.readlines()][1:]
    image_root = Path("/data/oxx220000/img_align_celeba")
    train_idx = np.load("data/celebA/train_idx.npy")[:80000]
    labels_all = np.load("data/celebA/labels.npy")
    ids = [str(int(idx)) for idx in train_idx]
    paths = [image_root / names[int(idx)] for idx in train_idx]
    labels = [int(labels_all[int(idx)]) for idx in train_idx]
    n_select = int(len(ids) * ratio)
    extra_scale = 1.0 / ratio
    ours = attribute_indices(
        n_total=len(ids),
        n_select=n_select,
        attribute_file=Path("save/attribute_celeba_trainval_qwen3_8B.json"),
        focus=["hair color", "skin", "emotion"],
        seed=SEED,
        labels=labels,
        rarity_power=0.03,
        frequency_penalty=0.5,
        class_balance=float("inf"),
        extra_counts={
            "hair color": {"red": 1000.0 * extra_scale},
            "skin": {"brown": 1000.0 * extra_scale},
            "emotion": {"sad": 1000.0 * extra_scale},
        },
    )
    return DatasetSpec(
        name="CelebA",
        slug=f"celeba_redhair_brownskin_sademotion_{ratio_tag(ratio)}",
        target="redhair_brownskin_sademotion",
        metric_name="Test ACC",
        ids=ids,
        paths=paths,
        selections=[
            selection_from_export_or_fallback(dataset="celeba", ratio=ratio, method="Random", ids=ids, fallback=random_indices(len(ids), n_select, SEED), metric=metrics[ratio]["Random"], fallback_source="reconstructed"),
            selection_from_export_or_fallback(dataset="celeba", ratio=ratio, method="Stratified", ids=ids, fallback=None, metric=metrics[ratio]["Stratified"], fallback_source="selection ids unavailable"),
            selection_from_export_or_fallback(dataset="celeba", ratio=ratio, method="Entropy", ids=ids, fallback=None, metric=metrics[ratio]["Entropy"], fallback_source="selection ids unavailable"),
            selection_from_export_or_fallback(dataset="celeba", ratio=ratio, method="Coreset", ids=ids, fallback=None, metric=metrics[ratio]["Coreset"], fallback_source="selection ids unavailable"),
            selection_from_export_or_fallback(dataset="celeba", ratio=ratio, method="BADGE", ids=ids, fallback=None, metric=metrics[ratio]["BADGE"], fallback_source="selection ids unavailable"),
            selection_from_export_or_fallback(dataset="celeba", ratio=ratio, method="PPAL", ids=ids, fallback=None, metric=metrics[ratio]["PPAL"], fallback_source="selection ids unavailable"),
            selection_from_export_or_fallback(dataset="celeba", ratio=ratio, method="UHerding", ids=ids, fallback=None, metric=metrics[ratio]["UHerding"], fallback_source="selection ids unavailable"),
            selection_from_export_or_fallback(dataset="celeba", ratio=ratio, method="Ours", ids=ids, fallback=ours, metric=metrics[ratio]["Ours"], fallback_source="reconstructed attribute r003_p05_cinf"),
        ],
        max_plot_points=12000,
    )


def _has_valid_person_annotation(anns: list[dict]) -> bool:
    for ann in anns:
        if ann.get("category_id") != 1 or ann.get("iscrowd", 0):
            continue
        if ann.get("num_keypoints", 0) > 0 and ann.get("area", 0) > 0:
            return True
    return False


def load_coco(ratio: float) -> DatasetSpec:
    metrics = {
        0.05: {
            "Random": 0.5730,
            "Stratified": 0.5781,
            "Entropy": 0.5827,
            "Coreset": 0.5693,
            "BADGE": 0.5797,
            "PPAL": 0.5771,
            "UHerding": 0.5738,
            "Ours": 0.5969,
        },
        0.1: {
            "Random": 0.6048,
            "Stratified": 0.6110,
            "Entropy": 0.6132,
            "Coreset": 0.5877,
            "BADGE": 0.5985,
            "PPAL": 0.5950,
            "UHerding": 0.6013,
            "Ours": 0.6183,
        },
        0.3: {
            "Random": np.nan,
            "Stratified": np.nan,
            "Entropy": np.nan,
            "Coreset": np.nan,
            "BADGE": np.nan,
            "PPAL": np.nan,
            "UHerding": np.nan,
            "Ours": np.nan,
        },
        0.5: {
            "Random": 0.6219,
            "Stratified": 0.6230,
            "Entropy": 0.6282,
            "Coreset": 0.6167,
            "BADGE": 0.6206,
            "PPAL": 0.6182,
            "UHerding": 0.6153,
            "Ours": 0.6317,
        },
    }
    if ratio not in metrics:
        raise ValueError(f"COCO metrics are not configured for ratio {ratio}")
    root = Path("/data/oxx220000/coco")
    ann = json.loads((root / "annotations/person_keypoints_train2017.json").read_text(encoding="utf-8"))
    id_to_file = {int(img["id"]): img["file_name"] for img in ann["images"]}
    ann_by_image: dict[int, list[dict]] = {}
    for item in ann["annotations"]:
        ann_by_image.setdefault(int(item["image_id"]), []).append(item)
    candidate_ids = sorted(image_id for image_id in id_to_file if _has_valid_person_annotation(ann_by_image.get(image_id, [])))[:20000]
    ids = [str(image_id) for image_id in candidate_ids]
    paths = [root / "train2017" / id_to_file[image_id] for image_id in candidate_ids]
    n_select = int(len(ids) * ratio)
    extra_scale = 1.0 / ratio
    ours = attribute_indices(
        n_total=len(ids),
        n_select=n_select,
        attribute_file=Path("save/attribute_coco_pose_gemini_train20k_qwen3_8B.json"),
        focus=["clothes_color", "weather"],
        seed=SEED,
        labels=None,
        rarity_power=0.03,
        frequency_penalty=0.5,
        class_balance=0.0,
        extra_counts={"clothes_color": {"brown": 1000.0 * extra_scale}, "weather": {"rainy": 1000.0 * extra_scale}},
    )
    return DatasetSpec(
        name="COCO",
        slug=f"coco_brown_rainy_{ratio_tag(ratio)}",
        target="brown_rainy_pose",
        metric_name="Test AP",
        ids=ids,
        paths=paths,
        selections=[
            selection_from_export_or_fallback(dataset="coco", ratio=ratio, method="Random", ids=ids, fallback=random_indices(len(ids), n_select, SEED), metric=metrics[ratio]["Random"], fallback_source="reconstructed"),
            selection_from_export_or_fallback(dataset="coco", ratio=ratio, method="Stratified", ids=ids, fallback=None, metric=metrics[ratio]["Stratified"], fallback_source="selection ids unavailable"),
            selection_from_export_or_fallback(dataset="coco", ratio=ratio, method="Entropy", ids=ids, fallback=None, metric=metrics[ratio]["Entropy"], fallback_source="selection ids unavailable"),
            selection_from_export_or_fallback(dataset="coco", ratio=ratio, method="Coreset", ids=ids, fallback=None, metric=metrics[ratio]["Coreset"], fallback_source="selection ids unavailable"),
            selection_from_export_or_fallback(dataset="coco", ratio=ratio, method="BADGE", ids=ids, fallback=None, metric=metrics[ratio]["BADGE"], fallback_source="selection ids unavailable"),
            selection_from_export_or_fallback(dataset="coco", ratio=ratio, method="PPAL", ids=ids, fallback=None, metric=metrics[ratio]["PPAL"], fallback_source="selection ids unavailable"),
            selection_from_export_or_fallback(dataset="coco", ratio=ratio, method="UHerding", ids=ids, fallback=None, metric=metrics[ratio]["UHerding"], fallback_source="selection ids unavailable"),
            selection_from_export_or_fallback(dataset="coco", ratio=ratio, method="Ours", ids=ids, fallback=ours, metric=metrics[ratio]["Ours"], fallback_source="reconstructed attribute anchor"),
        ],
        max_plot_points=12000,
    )


def read_kitti_selected(root: Path, id_to_idx: dict[str, int]) -> set[int]:
    path = root / "selected_original_ids.txt"
    return {id_to_idx[line.strip()] for line in path.read_text(encoding="utf-8").splitlines() if line.strip() in id_to_idx}


def load_kitti(ratio: float) -> DatasetSpec:
    image_dir = Path("/data/oxx220000/kitti_yolov8_first5000/images/train")
    image_paths = sorted(image_dir.glob("*.png"))
    ids = [path.stem for path in image_paths]
    id_to_idx = {image_id: idx for idx, image_id in enumerate(ids)}
    roots_by_ratio = {
        0.05: {
            "Random": (Path("/data/oxx220000/kitti_yolov8_ratio0p05_random_red_cloudy_gen1000_seed42"), np.nan),
            "Stratified": (Path("/data/oxx220000/kitti_yolov8_ratio0p05_stratified_red_cloudy_gen1000_seed42"), np.nan),
            "Entropy": (Path("/data/oxx220000/kitti_yolov8_ratio0p05_entropy_red_cloudy_gen1000_seed42"), np.nan),
            "Coreset": (Path("/data/oxx220000/kitti_yolov8_ratio0p05_coreset_fast_red_cloudy_gen1000_seed42"), np.nan),
            "BADGE": (Path("/data/oxx220000/kitti_yolov8_ratio0p05_badge_fast_red_cloudy_gen1000_seed42"), np.nan),
            "PPAL": (Path("/data/oxx220000/kitti_yolov8_ratio0p05_ppal_fast_red_cloudy_gen1000_seed42_20260505_000927"), np.nan),
            "UHerding": (Path("/data/oxx220000/kitti_yolov8_ratio0p05_uherding_fast_red_cloudy_gen1000_seed42_20260505_000927"), np.nan),
            "Ours": (Path("/data/oxx220000/kitti_yolov8_ratio0p05_attribute_r003_p5_c15_red_cloudy_gen1000_seed42"), np.nan),
        },
        0.1: {
            "Random": (Path("/data/oxx220000/kitti_yolov8_ratio0p1_random_red_cloudy_gen1000_seed42"), 0.5204),
            "Stratified": (Path("/data/oxx220000/kitti_yolov8_ratio0p1_stratified_red_cloudy_gen1000_seed42"), 0.5308),
            "Entropy": (Path("/data/oxx220000/kitti_yolov8_ratio0p1_entropy_red_cloudy_gen1000_seed42"), 0.5297),
            "Coreset": (Path("/data/oxx220000/kitti_yolov8_ratio0p1_coreset_fast_red_cloudy_gen1000_seed42"), 0.5237),
            "BADGE": (Path("/data/oxx220000/kitti_yolov8_ratio0p1_badge_fast_red_cloudy_gen1000_seed42"), 0.5322),
            "PPAL": (Path("/data/oxx220000/kitti_yolov8_ratio0p1_ppal_fast_red_cloudy_gen1000_seed42_20260502_185507"), 0.5262),
            "UHerding": (Path("/data/oxx220000/kitti_yolov8_ratio0p1_uherding_fast_red_cloudy_gen1000_seed42_20260502_140255"), 0.5222),
            "Ours": (Path("/data/oxx220000/kitti_yolov8_ratio0p1_attribute_r000_p0_cinf_seed42_red_cloudy_gen1000"), 0.5372),
        },
        0.3: {
            "Random": (Path("/data/oxx220000/kitti_yolov8_ratio0p3_random_red_cloudy_gen1000_seed42"), np.nan),
            "Stratified": (Path("/data/oxx220000/kitti_yolov8_ratio0p3_stratified_red_cloudy_gen1000_seed42"), np.nan),
            "Entropy": (Path("/data/oxx220000/kitti_yolov8_ratio0p3_entropy_red_cloudy_gen1000_seed42"), np.nan),
            "Coreset": (Path("/data/oxx220000/kitti_yolov8_ratio0p3_coreset_fast_red_cloudy_gen1000_seed42"), np.nan),
            "BADGE": (Path("/data/oxx220000/kitti_yolov8_ratio0p3_badge_fast_red_cloudy_gen1000_seed42"), np.nan),
            "PPAL": (Path("/data/oxx220000/kitti_yolov8_ratio0p3_ppal_fast_red_cloudy_gen1000_seed42_20260502_185507"), np.nan),
            "UHerding": (Path("/data/oxx220000/kitti_yolov8_ratio0p3_uherding_fast_red_cloudy_gen1000_seed42_20260502_140255"), np.nan),
            "Ours": (Path("/data/oxx220000/kitti_yolov8_ratio0p3_attribute_r000_p0_c15_red_cloudy_gen1000_seed42"), np.nan),
        },
        0.5: {
            "Random": (Path("/data/oxx220000/kitti_yolov8_ratio0p5_random_red_cloudy_gen1000_seed42"), 0.5719),
            "Stratified": (Path("/data/oxx220000/kitti_yolov8_ratio0p5_stratified_red_cloudy_gen1000_seed42"), 0.5680),
            "Entropy": (Path("/data/oxx220000/kitti_yolov8_ratio0p5_entropy_red_cloudy_gen1000_seed42"), 0.5766),
            "Coreset": (Path("/data/oxx220000/kitti_yolov8_ratio0p5_coreset_fast_red_cloudy_gen1000_seed42"), 0.5759),
            "BADGE": (Path("/data/oxx220000/kitti_yolov8_ratio0p5_badge_fast_red_cloudy_gen1000_seed42"), 0.5726),
            "PPAL": (Path("/data/oxx220000/kitti_yolov8_ratio0p5_ppal_fast_red_cloudy_gen1000_seed42_20260502_185507"), 0.5721),
            "UHerding": (Path("/data/oxx220000/kitti_yolov8_ratio0p5_uherding_fast_red_cloudy_gen1000_seed42_20260502_140255"), 0.5675),
            "Ours": (Path("/data/oxx220000/kitti_yolov8_ratio0p5_attribute_r000_p0_c15_red_cloudy_gen1000_seed42"), 0.5773),
        },
    }
    if ratio not in roots_by_ratio:
        raise ValueError(f"KITTI selected-id roots are not configured for ratio {ratio}")
    roots = roots_by_ratio[ratio]
    selections = [
        Selection(name, read_kitti_selected(root, id_to_idx), metric, "exact selected_original_ids")
        for name, (root, metric) in roots.items()
    ]
    return DatasetSpec(
        name="KITTI",
        slug=f"kitti_red_cloudy_{ratio_tag(ratio)}",
        target="red_cloudy_yolov8",
        metric_name="mAP50-95",
        ids=ids,
        paths=image_paths,
        selections=selections,
        max_plot_points=5000,
    )


def choose_plot_indices(spec: DatasetSpec, seed: int) -> list[int]:
    n_total = len(spec.ids)
    if n_total <= spec.max_plot_points:
        return list(range(n_total))
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_total), spec.max_plot_points))


def compute_or_load_coords(spec: DatasetSpec, plot_indices: list[int], args: argparse.Namespace) -> np.ndarray:
    cache_path = args.out_dir / f"{spec.slug}_proxy_tsne_cache.npz"
    cache_ids = np.asarray([spec.ids[idx] for idx in plot_indices])
    if cache_path.exists() and not args.recompute:
        data = np.load(cache_path, allow_pickle=False)
        if [str(item) for item in data["ids"]] == [str(item) for item in cache_ids]:
            return data["coords"]

    embeddings = []
    for count, idx in enumerate(plot_indices, start=1):
        if count % 1000 == 0:
            print(f"{spec.name}: embedded {count}/{len(plot_indices)} plotted images")
        embeddings.append(image_embedding(spec.paths[idx], args.thumbnail_size))
    x = np.stack(embeddings, axis=0)
    x = StandardScaler().fit_transform(x)
    pca_dim = min(args.pca_dim, x.shape[0] - 1, x.shape[1])
    x = PCA(n_components=pca_dim, random_state=args.seed).fit_transform(x)
    tsne_kwargs = {
        "n_components": 2,
        "perplexity": min(args.perplexity, max(5.0, (len(plot_indices) - 1) / 3)),
        "init": "pca",
        "learning_rate": "auto",
        "random_state": args.seed,
    }
    iter_arg = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    tsne_kwargs[iter_arg] = args.n_iter
    coords = TSNE(**tsne_kwargs).fit_transform(x)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, ids=cache_ids, coords=coords)
    return coords


def write_csv(spec: DatasetSpec, plot_indices: list[int], coords: np.ndarray, out_path: Path) -> None:
    fields = ["dataset", "image_id", "tsne_x", "tsne_y"]
    fields.extend(f"selected_{sel.name.lower()}" for sel in spec.selections)
    fields.extend(f"source_{sel.name.lower()}" for sel in spec.selections)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for local_pos, (idx, coord) in enumerate(zip(plot_indices, coords)):
            row = {
                "dataset": spec.name,
                "image_id": spec.ids[idx],
                "tsne_x": f"{coord[0]:.6f}",
                "tsne_y": f"{coord[1]:.6f}",
            }
            for sel in spec.selections:
                row[f"selected_{sel.name.lower()}"] = "" if sel.indices is None else int(idx in sel.indices)
                row[f"source_{sel.name.lower()}"] = sel.source
            writer.writerow(row)


def plot_spec(spec: DatasetSpec, plot_indices: list[int], coords: np.ndarray, out_path: Path) -> None:
    index_to_plot_pos = {idx: pos for pos, idx in enumerate(plot_indices)}
    n_cols = 4
    n_rows = int(np.ceil(len(spec.selections) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14.5, max(3.7, 3.4 * n_rows)), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, sel in zip(axes, spec.selections):
        selected_positions = [] if sel.indices is None else [index_to_plot_pos[idx] for idx in sel.indices if idx in index_to_plot_pos]
        ax.scatter(coords[:, 0], coords[:, 1], s=5, c="#D0D0D0", alpha=0.18, linewidths=0)
        if sel.indices is None:
            ax.text(0.5, 0.5, "selection ids\nunavailable", transform=ax.transAxes, ha="center", va="center", fontsize=10, color="#555555")
            title_n = "n/a"
        else:
            ax.scatter(
                coords[selected_positions, 0],
                coords[selected_positions, 1],
                s=9 if sel.name != "Ours" else 14,
                c=PALETTE.get(sel.name, "#333333"),
                alpha=0.86,
                linewidths=0,
            )
            title_n = str(len(sel.indices))
        if np.isnan(sel.metric):
            ax.set_title(f"{sel.name}  n={title_n}", fontsize=10)
        else:
            ax.set_title(f"{sel.name}  n={title_n}  {spec.metric_name}={sel.metric:.4f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes[len(spec.selections):]:
        ax.axis("off")
    ratio_label = spec.slug.split("_ratio")[-1].replace("p", ".")
    dataset_display = {
        "COCO": "COCO Pose Estimation",
        "KITTI": "KITTI Object Detection",
        "ImageNet": "ImageNet10 Classification",
        "CelebA": "CelebA Classification",
    }.get(spec.name, spec.name)
    fig.suptitle(f"{dataset_display} (Ratio {ratio_label}) Selection on Shared Proxy t-SNE", fontsize=14, y=0.98)
    fig.text(0.5, 0.025, "Gray: candidate pool. Colored: selected subset.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0.01, 0.06, 0.99, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), format="pdf")
    fig.savefig(out_path.with_suffix(".png"), dpi=220)
    plt.close(fig)


def build_one(spec: DatasetSpec, args: argparse.Namespace) -> None:
    plot_indices = choose_plot_indices(spec, args.seed)
    coords = compute_or_load_coords(spec, plot_indices, args)
    prefix = args.out_dir / f"{spec.slug}_selection_tsne"
    write_csv(spec, plot_indices, coords, prefix.with_suffix(".csv"))
    plot_spec(spec, plot_indices, coords, prefix.with_suffix(".png"))
    print(f"Wrote {prefix.with_suffix('.pdf')}")
    print(f"Wrote {prefix.with_suffix('.png')}")
    print(f"Wrote {prefix.with_suffix('.csv')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["imagenet", "celeba", "coco", "kitti"])
    parser.add_argument("--ratio", type=float, default=0.5, choices=[0.05, 0.1, 0.3, 0.5])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--thumbnail-size", type=int, default=8)
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--n-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaders = {
        "imagenet": load_imagenet,
        "celeba": load_celeba,
        "coco": load_coco,
        "kitti": load_kitti,
    }
    for dataset_name in args.datasets:
        if dataset_name not in loaders:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        spec = loaders[dataset_name](args.ratio)
        build_one(spec, args)


if __name__ == "__main__":
    main()
