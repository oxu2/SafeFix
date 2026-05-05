#!/usr/bin/env python3
"""Build the compact main-result CSV and line plot used by report.md."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]

DATA = {
    "ImageNet": {
        "metric": "Test ACC",
        "random": [0.7269, 0.7308, 0.7400, 0.7423, 0.7462],
        "best_attribute": [0.7369, 0.7492, 0.7508, 0.7515, 0.7562],
    },
    "CelebA": {
        "metric": "Test ACC",
        "random": [0.9103, 0.9114, 0.9114, 0.9084, 0.9087],
        "best_attribute": [0.9137, 0.9141, 0.9187, 0.9171, 0.9196],
    },
    "COCO": {
        "metric": "Test AP",
        "random": [0.6048, 0.6040, 0.6209, 0.6151, 0.6091],
        "best_attribute": [0.6183, 0.6306, 0.6317, 0.6319, 0.6355],
    },
    "KITTI": {
        "metric": "Test mAP50-95",
        "random": [0.5204, 0.5605, 0.5719, 0.5766, 0.5817],
        "best_attribute": [0.5372, 0.5646, 0.5773, 0.5780, 0.5850],
    },
}


def main() -> None:
    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)

    csv_path = out_dir / "main_results_random_vs_best_attribute.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "metric", "ratio", "random", "best_attribute"])
        for dataset, spec in DATA.items():
            for ratio, random_value, attr_value in zip(
                RATIOS, spec["random"], spec["best_attribute"]
            ):
                writer.writerow(
                    [
                        dataset,
                        spec["metric"],
                        ratio,
                        "" if random_value is None else random_value,
                        "" if attr_value is None else attr_value,
                    ]
                )

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for ax, (dataset, spec) in zip(axes.ravel(), DATA.items()):
        for key, label, marker in [
            ("random", "Random", "o"),
            ("best_attribute", "Best Attribute", "s"),
        ]:
            xs = []
            ys = []
            for ratio, value in zip(RATIOS, spec[key]):
                if value is not None:
                    xs.append(ratio)
                    ys.append(value)
            if xs:
                ax.plot(xs, ys, marker=marker, linewidth=2, label=label)

        ax.set_title(f"{dataset} ({spec['metric']})")
        ax.set_xlabel("Ratio")
        ax.set_ylabel(spec["metric"])
        ax.set_xticks(RATIOS)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Main Results: Random vs Best Attribute")
    fig.savefig(out_dir / "main_results_random_vs_best_attribute.png", dpi=180)


if __name__ == "__main__":
    main()
