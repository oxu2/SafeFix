#!/usr/bin/env python3
"""Build COCO and KITTI per-method ratio plots for report.md."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

PLOTS = {
    "coco_brown_rainy_methods_test_ap": {
        "title": "COCO Brown-Rainy Pose, Test AP",
        "ylabel": "Test AP",
        "safefix": 0.6292,
        "ylim": (0.585, 0.638),
        "methods": {
            "Random": [0.6048, 0.6040, 0.6209, 0.6151, 0.6091, 0.6259],
            "Stratified": [0.6110, 0.6243, 0.6230, 0.6299, 0.6291, 0.6259],
            "Entropy": [0.6132, 0.6215, 0.6282, 0.6293, 0.6170, 0.6259],
            "Coreset": [0.5877, 0.6256, 0.6167, 0.6275, 0.6202, 0.6259],
            "BADGE": [0.5985, 0.6153, 0.6206, 0.6256, 0.6213, 0.6259],
            "UHerding": [0.6013, 0.6079, 0.6153, 0.6155, 0.6191, 0.6259],
            "PPAL": [0.5950, 0.6165, 0.6182, 0.6223, 0.6301, 0.6259],
            "Ours": [0.6183, 0.6306, 0.6317, 0.6319, 0.6355, 0.6259],
        },
    },
    "kitti_red_cloudy_methods_test_map5095": {
        "title": "KITTI Red-Cloudy YOLOv8, Test mAP50-95",
        "ylabel": "Test mAP50-95",
        "safefix": 0.5737,
        "ylim": (0.518, 0.589),
        "methods": {
            "Random": [0.5204, 0.5605, 0.5719, 0.5766, 0.5817, 0.5858],
            "Stratified": [0.5308, 0.5572, 0.5680, 0.5760, 0.5830, 0.5858],
            "Entropy": [0.5297, 0.5636, 0.5766, 0.5772, 0.5830, 0.5858],
            "Coreset": [0.5237, 0.5612, 0.5759, 0.5772, 0.5781, 0.5858],
            "BADGE": [0.5322, 0.5624, 0.5726, 0.5717, 0.5810, 0.5858],
            "UHerding": [0.5222, 0.5573, 0.5675, 0.5768, 0.5838, 0.5858],
            "PPAL": [0.5262, 0.5568, 0.5721, 0.5782, 0.5796, 0.5858],
            "Ours": [0.5372, 0.5646, 0.5773, 0.5780, 0.5850, 0.5858],
        },
    },
    "celeba_redhair_brownskin_sademotion_methods_test_acc": {
        "title": "CelebA Red-Hair Brown-Skin Sad-Emotion, Test ACC",
        "ylabel": "Test ACC",
        "safefix": 0.9155,
        "ylim": (0.899, 0.921),
        "methods": {
            "Random": [0.9103, 0.9114, 0.9114, 0.9084, 0.9087, 0.9114],
            "Stratified": [0.9080, 0.9107, 0.9132, 0.9135, 0.9100, 0.9114],
            "Entropy": [0.9124, 0.9139, 0.9129, 0.9153, 0.9135, 0.9114],
            "Coreset": [0.9000, 0.9132, 0.9134, 0.9119, 0.9114, 0.9114],
            "BADGE": [0.9087, 0.9075, 0.9084, 0.9133, 0.9114, 0.9114],
            "UHerding": [0.9047, 0.9090, 0.9078, 0.9105, 0.9100, 0.9114],
            "PPAL": [0.9113, 0.9095, 0.9151, 0.9114, 0.9119, 0.9114],
            "Ours": [0.9137, 0.9141, 0.9187, 0.9171, 0.9196, 0.9114],
        },
    },
}

COLORS = {
    "Random": "#4C72B0",
    "Stratified": "#55A868",
    "Entropy": "#8172B3",
    "Coreset": "#64B5CD",
    "BADGE": "#DD8452",
    "UHerding": "#8C8C8C",
    "PPAL": "#937860",
    "Ours": "#C44E52",
}

MARKERS = {
    "Random": "o",
    "Stratified": "s",
    "Entropy": "^",
    "Coreset": "D",
    "BADGE": "P",
    "UHerding": "X",
    "PPAL": "v",
    "Ours": "*",
}


def display_label(method: str) -> str:
    return r"$\bf{Ours}$" if method == "Ours" else method


def write_csv(out_dir: Path, stem: str, spec: dict) -> None:
    csv_path = out_dir / f"{stem}.csv"
    methods = spec["methods"]
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ratio", *methods.keys(), "SafeFix"])
        for i, ratio in enumerate(RATIOS):
            writer.writerow(
                [
                    ratio,
                    *[values[i] for values in methods.values()],
                    spec["safefix"] if spec["safefix"] is not None and ratio == 1.0 else "",
                ]
            )


def write_plot(out_dir: Path, stem: str, spec: dict) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    for method, values in spec["methods"].items():
        is_ours = method == "Ours"
        ax.plot(
            RATIOS,
            values,
            marker=MARKERS[method],
            linewidth=2.8 if is_ours else 1.75,
            markersize=8.5 if is_ours else 5.2,
            label=display_label(method),
            color=COLORS[method],
            zorder=5 if is_ours else 3,
        )

    if spec["safefix"] is not None:
        ax.axhline(
            spec["safefix"],
            color="0.25",
            linestyle="--",
            linewidth=1.8,
            label="SafeFix",
        )
        ax.plot(
            [1.0],
            [spec["safefix"]],
            marker="*",
            markersize=11,
            color="0.15",
            linestyle="None",
        )

    ax.set_title(spec["title"])
    ax.set_xlabel("Ratio")
    ax.set_ylabel(spec["ylabel"])
    ax.set_xticks(RATIOS)
    ax.set_ylim(*spec["ylim"])
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, frameon=False, fontsize=8.5)
    fig.savefig(out_dir / f"{stem}.png", dpi=180)


def main() -> None:
    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)
    for stem, spec in PLOTS.items():
        write_csv(out_dir, stem, spec)
        write_plot(out_dir, stem, spec)


if __name__ == "__main__":
    main()
