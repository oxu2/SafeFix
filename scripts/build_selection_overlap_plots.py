#!/usr/bin/env python3
"""Build method-overlap heatmaps from selection t-SNE CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = ["random", "stratified", "entropy", "coreset", "badge", "ppal", "uherding", "ours"]
METHOD_LABELS = {
    "random": "Random",
    "stratified": "Stratified",
    "entropy": "Entropy",
    "coreset": "Coreset",
    "badge": "BADGE",
    "ppal": "PPAL",
    "uherding": "UHerding",
    "ours": "Ours",
}


def read_selection_columns(path: Path) -> dict[str, set[str]]:
    selections: dict[str, set[str]] | None = None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found in {path}")
        available = [method for method in METHODS if f"selected_{method}" in reader.fieldnames]
        selections = {method: set() for method in available}
        for row in reader:
            image_id = row["image_id"]
            for method in available:
                key = f"selected_{method}"
                if key in row and row[key] == "1":
                    selections[method].add(image_id)
    if selections is None:
        raise ValueError(f"Could not read {path}")
    return selections


def compute_overlap(selections: dict[str, set[str]], mode: str) -> np.ndarray:
    methods = list(selections)
    matrix = np.zeros((len(methods), len(methods)), dtype=float)
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            intersection = len(selections[a] & selections[b])
            if mode == "count":
                value = intersection
            elif mode == "jaccard":
                union = len(selections[a] | selections[b])
                value = intersection / union if union else 0.0
            elif mode == "overlap_coef":
                denom = min(len(selections[a]), len(selections[b]))
                value = intersection / denom if denom else 0.0
            else:
                raise ValueError(mode)
            matrix[i, j] = value
    return matrix


def write_matrix_csv(path: Path, methods: list[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        labels = [METHOD_LABELS[method] for method in methods]
        writer.writerow(["method", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[f"{value:.6f}" for value in row]])


def plot_heatmap(path: Path, methods: list[str], matrix: np.ndarray, title: str, mode: str) -> None:
    labels = [METHOD_LABELS[method] for method in methods]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    vmax = float(np.nanmax(matrix)) if matrix.size else 1.0
    if mode != "count":
        vmax = 1.0
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=vmax)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title(title, fontsize=13)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text = f"{matrix[i, j]:.0f}" if mode == "count" else f"{matrix[i, j]:.2f}"
            color = "white" if matrix[i, j] > vmax * 0.55 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label({"count": "Intersection Count", "jaccard": "Jaccard", "overlap_coef": "Overlap Coefficient"}[mode])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def build_one(csv_path: Path, out_dir: Path, mode: str) -> None:
    selections = read_selection_columns(csv_path)
    methods = list(selections)
    matrix = compute_overlap(selections, mode)
    stem = csv_path.stem.replace("_selection_tsne", f"_selection_overlap_{mode}")
    out_csv = out_dir / f"{stem}.csv"
    out_png = out_dir / f"{stem}.png"
    write_matrix_csv(out_csv, methods, matrix)
    title = csv_path.stem.replace("_selection_tsne", "").replace("_", " ")
    plot_heatmap(out_png, methods, matrix, f"{title} Selection Overlap", mode)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csvs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["count", "jaccard", "overlap_coef"], default="jaccard")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for csv_path in args.csvs:
        build_one(csv_path, args.out_dir, args.mode)


if __name__ == "__main__":
    main()
