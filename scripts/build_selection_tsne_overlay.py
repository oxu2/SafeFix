#!/usr/bin/env python3
"""Build a selected-subset overlay plot from a selection t-SNE CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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

METHOD_KEYS = {
    "Random": "random",
    "Stratified": "stratified",
    "Entropy": "entropy",
    "Coreset": "coreset",
    "BADGE": "badge",
    "UHerding": "uherding",
    "PPAL": "ppal",
    "Ours": "ours",
}


def read_csv(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    coords: list[tuple[float, float]] = []
    selected: dict[str, list[bool]] = {name: [] for name in METHOD_KEYS}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        available = {
            name: f"selected_{key}"
            for name, key in METHOD_KEYS.items()
            if f"selected_{key}" in reader.fieldnames
        }
        selected = {name: [] for name in available}
        for row in reader:
            coords.append((float(row["tsne_x"]), float(row["tsne_y"])))
            for name, column in available.items():
                selected[name].append(row[column] == "1")
    return np.asarray(coords, dtype=float), {name: np.asarray(values, dtype=bool) for name, values in selected.items()}


def plot_overlay(
    out_path: Path,
    coords: np.ndarray,
    selected: dict[str, np.ndarray],
    methods: list[str],
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    ax.scatter(coords[:, 0], coords[:, 1], s=5, c="#D6D6D6", alpha=0.16, linewidths=0)
    for name in methods:
        if name not in selected:
            continue
        mask = selected[name]
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=8 if name != "Ours" else 16,
            c=PALETTE[name],
            alpha=0.62 if name != "Ours" else 0.88,
            linewidths=0,
            label=name,
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(frameon=False, markerscale=2.0, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["Random", "Entropy", "BADGE", "PPAL", "Ours"],
        choices=list(METHOD_KEYS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coords, selected = read_csv(args.csv)
    plot_overlay(args.out, coords, selected, args.methods, args.title)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
