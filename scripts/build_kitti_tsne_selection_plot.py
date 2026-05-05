#!/usr/bin/env python3
"""Build KITTI t-SNE selection comparison plots.

The plot uses the same lightweight thumbnail embedding family as the KITTI
active-learning baselines, then overlays selected original images for each
ratio-0.1 method.
"""

from __future__ import annotations

import argparse
import csv
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


DEFAULT_BASE_ROOT = Path("/data/oxx220000/kitti_yolov8_first5000")
DEFAULT_OUT_DIR = Path("figures")

METHODS = [
    (
        "Random",
        Path("/data/oxx220000/kitti_yolov8_ratio0p1_random_red_cloudy_gen1000_seed42"),
        0.5204,
    ),
    (
        "Stratified",
        Path("/data/oxx220000/kitti_yolov8_ratio0p1_stratified_red_cloudy_gen1000_seed42"),
        0.5308,
    ),
    (
        "Entropy",
        Path("/data/oxx220000/kitti_yolov8_ratio0p1_entropy_red_cloudy_gen1000_seed42"),
        0.5297,
    ),
    (
        "Coreset",
        Path("/data/oxx220000/kitti_yolov8_ratio0p1_coreset_fast_red_cloudy_gen1000_seed42"),
        0.5237,
    ),
    (
        "BADGE",
        Path("/data/oxx220000/kitti_yolov8_ratio0p1_badge_fast_red_cloudy_gen1000_seed42"),
        0.5322,
    ),
    (
        "UHerding",
        Path("/data/oxx220000/kitti_yolov8_ratio0p1_uherding_fast_red_cloudy_gen1000_seed42_20260502_140255"),
        0.5222,
    ),
    (
        "PPAL",
        Path("/data/oxx220000/kitti_yolov8_ratio0p1_ppal_fast_red_cloudy_gen1000_seed42_20260502_185507"),
        0.5262,
    ),
    (
        "Ours",
        Path("/data/oxx220000/kitti_yolov8_ratio0p1_attribute_r000_p0_cinf_seed42_red_cloudy_gen1000"),
        0.5372,
    ),
]

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


def list_image_ids(image_dir: Path) -> list[str]:
    paths = []
    for suffix in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        paths.extend(image_dir.glob(suffix))
    return sorted(path.stem for path in paths)


def image_embedding(path: Path, side: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((side, side), Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
    flat = arr.reshape(-1)
    stats = [arr.mean(axis=(0, 1)), arr.std(axis=(0, 1)), arr.min(axis=(0, 1)), arr.max(axis=(0, 1))]
    return np.concatenate([flat, *stats], axis=0).astype(np.float32)


def read_selected_ids(root: Path) -> set[str]:
    path = root / "selected_original_ids.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def compute_or_load_tsne(args: argparse.Namespace, image_ids: list[str], image_dir: Path) -> np.ndarray:
    if args.cache_path.exists() and not args.recompute:
        data = np.load(args.cache_path, allow_pickle=False)
        cached_ids = [str(item) for item in data["image_ids"]]
        if cached_ids == image_ids:
            return data["coords"]

    embeddings = []
    for idx, image_id in enumerate(image_ids, start=1):
        if idx % 1000 == 0:
            print(f"Embedded {idx}/{len(image_ids)} KITTI train images")
        embeddings.append(image_embedding(image_dir / f"{image_id}.png", args.thumbnail_size))
    embeddings_array = np.stack(embeddings, axis=0)

    scaled = StandardScaler().fit_transform(embeddings_array)
    pca_dim = min(args.pca_dim, scaled.shape[0] - 1, scaled.shape[1])
    reduced = PCA(n_components=pca_dim, random_state=args.seed).fit_transform(scaled)
    tsne_kwargs = {
        "n_components": 2,
        "perplexity": args.perplexity,
        "init": "pca",
        "learning_rate": "auto",
        "random_state": args.seed,
    }
    iter_arg = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    tsne_kwargs[iter_arg] = args.n_iter
    coords = TSNE(**tsne_kwargs).fit_transform(reduced)

    args.cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.cache_path, image_ids=np.asarray(image_ids), coords=coords)
    return coords


def write_membership_csv(path: Path, image_ids: list[str], coords: np.ndarray, selections: dict[str, set[str]]) -> None:
    fieldnames = ["image_id", "tsne_x", "tsne_y"] + [f"selected_{name.lower()}" for name in selections]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for image_id, (x_val, y_val) in zip(image_ids, coords):
            row = {"image_id": image_id, "tsne_x": f"{x_val:.6f}", "tsne_y": f"{y_val:.6f}"}
            for name, selected in selections.items():
                row[f"selected_{name.lower()}"] = int(image_id in selected)
            writer.writerow(row)


def plot_facets(path: Path, image_ids: list[str], coords: np.ndarray, selections: dict[str, set[str]]) -> None:
    id_to_idx = {image_id: idx for idx, image_id in enumerate(image_ids)}
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.2), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, (name, _, metric) in zip(axes, METHODS):
        selected = selections[name]
        selected_idx = [id_to_idx[image_id] for image_id in selected if image_id in id_to_idx]
        ax.scatter(coords[:, 0], coords[:, 1], s=5, c="#D0D0D0", alpha=0.18, linewidths=0)
        ax.scatter(
            coords[selected_idx, 0],
            coords[selected_idx, 1],
            s=9 if name != "Ours" else 13,
            c=PALETTE[name],
            alpha=0.86,
            linewidths=0,
        )
        ax.set_title(f"{name}  n={len(selected_idx)}  mAP50-95={metric:.4f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("KITTI Red-Cloudy Ratio 0.1 Selection on Shared t-SNE Space", fontsize=14, y=0.98)
    fig.text(0.5, 0.02, "Gray: all 5000 original train candidates. Colored: selected original subset.", ha="center", fontsize=10)
    fig.tight_layout(rect=[0.01, 0.05, 0.99, 0.95])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_overlay(path: Path, image_ids: list[str], coords: np.ndarray, selections: dict[str, set[str]]) -> None:
    id_to_idx = {image_id: idx for idx, image_id in enumerate(image_ids)}
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    ax.scatter(coords[:, 0], coords[:, 1], s=5, c="#D6D6D6", alpha=0.16, linewidths=0)
    for name in ["Random", "Entropy", "BADGE", "PPAL", "Ours"]:
        selected_idx = [id_to_idx[image_id] for image_id in selections[name] if image_id in id_to_idx]
        ax.scatter(
            coords[selected_idx, 0],
            coords[selected_idx, 1],
            s=8 if name != "Ours" else 16,
            c=PALETTE[name],
            alpha=0.62 if name != "Ours" else 0.88,
            linewidths=0,
            label=name,
        )
    ax.set_title("KITTI Ratio 0.1 Selected Subsets on Shared t-SNE")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(frameon=False, markerscale=2.0, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--thumbnail-size", type=int, default=8)
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--n-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_OUT_DIR / "kitti_red_cloudy_ratio0p1_proxy_tsne_cache.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = args.base_root / "images" / "train"
    image_ids = list_image_ids(image_dir)
    if not image_ids:
        raise RuntimeError(f"No images found under {image_dir}")

    selections = {name: read_selected_ids(root) for name, root, _ in METHODS}
    coords = compute_or_load_tsne(args, image_ids, image_dir)

    out_prefix = args.out_dir / "kitti_red_cloudy_ratio0p1_selection_tsne"
    write_membership_csv(out_prefix.with_suffix(".csv"), image_ids, coords, selections)
    plot_facets(out_prefix.with_suffix(".png"), image_ids, coords, selections)
    plot_overlay(args.out_dir / "kitti_red_cloudy_ratio0p1_selection_tsne_overlay.png", image_ids, coords, selections)

    print(f"Wrote {out_prefix.with_suffix('.png')}")
    print(f"Wrote {args.out_dir / 'kitti_red_cloudy_ratio0p1_selection_tsne_overlay.png'}")
    print(f"Wrote {out_prefix.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
