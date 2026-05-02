#!/usr/bin/env python3
"""Plot recent ratio sweeps: random vs best attribute result per ratio."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = Path("figures")
CSV_PATH = OUT_DIR / "recent_ratio_random_vs_best_attribute.csv"
PNG_PATH = OUT_DIR / "recent_ratio_random_vs_best_attribute.png"


EXPERIMENTS = [
    {
        "name": "ImageNet base AL",
        "metric": "Test ACC",
        "rows": [
            (0.10, 0.7269, 0.7346, "attribute"),
            (0.30, 0.7308, 0.7385, "attribute"),
            (0.50, 0.7400, 0.7495, "attribute"),
            (0.70, 0.7423, 0.7446, "attribute"),
            (0.90, 0.7482, 0.7531, "attribute"),
        ],
    },
    {
        "name": "Brown-Rainy COCO pose",
        "metric": "Test AP",
        "rows": [
            (0.05, 0.5797, 0.5871, "attribute"),
            (0.10, 0.6048, 0.6183, "attribute"),
            (0.30, 0.6127, 0.6306, "attribute"),
            (0.50, 0.6219, 0.6287, "attribute"),
            (0.70, 0.6232, 0.6319, "attribute"),
            (0.90, 0.6180, 0.6355, "attribute"),
        ],
    },
    {
        "name": "ImageNet instruct-pix2pix",
        "metric": "Test ACC",
        "rows": [
            (0.10, 0.7062, 0.7146, "attribute"),
            (0.30, 0.7285, 0.7369, "attribute"),
            (0.50, 0.7438, 0.7462, "attribute"),
            (0.70, 0.7477, 0.7501, "attribute"),
            (0.90, 0.7423, 0.7523, "attribute"),
        ],
    },
    {
        "name": "CelebA instruct-pix2pix",
        "metric": "Test ACC",
        "rows": [
            (0.10, 0.9108, 0.9120, "attribute"),
            (0.30, 0.9091, 0.9101, "attribute"),
            (0.50, 0.9098, 0.9110, "attribute"),
            (0.70, 0.9116, 0.9116, "attribute"),
            (0.90, 0.9104, 0.9111, "attribute"),
        ],
    },
    {
        "name": "CelebA HiBUG",
        "metric": "Test ACC",
        "rows": [
            (0.10, 0.9052, 0.9008, "attribute"),
            (0.30, 0.9076, 0.9095, "attribute"),
            (0.50, 0.9096, 0.9131, "attribute"),
            (0.70, 0.9057, 0.9116, "attribute"),
        ],
    },
]


def write_csv() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "experiment",
                "metric",
                "ratio",
                "random",
                "best_attribute",
                "delta_attribute_minus_random",
                "best_attribute_method",
            ]
        )
        for experiment in EXPERIMENTS:
            for ratio, random_value, attr_value, attr_method in experiment["rows"]:
                writer.writerow(
                    [
                        experiment["name"],
                        experiment["metric"],
                        f"{ratio:.2f}",
                        f"{random_value:.4f}",
                        f"{attr_value:.4f}",
                        f"{attr_value - random_value:.4f}",
                        attr_method,
                    ]
                )


def plot() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=False)
    flat_axes = list(axes.flat)

    for ax, experiment in zip(flat_axes, EXPERIMENTS):
        rows = experiment["rows"]
        ratios = [row[0] for row in rows]
        random_values = [row[1] for row in rows]
        attr_values = [row[2] for row in rows]

        ax.plot(ratios, random_values, marker="o", linewidth=2, label="random")
        ax.plot(
            ratios,
            attr_values,
            marker="s",
            linewidth=2,
            label="best attribute",
        )
        ax.set_title(experiment["name"], fontsize=11)
        ax.set_xlabel("ratio")
        ax.set_ylabel(experiment["metric"])
        ax.set_xticks(ratios)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    flat_axes[-1].axis("off")
    fig.suptitle("Recent Ratio Sweeps: Random vs Best Attribute", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(PNG_PATH, dpi=200)


def main() -> None:
    write_csv()
    plot()
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
