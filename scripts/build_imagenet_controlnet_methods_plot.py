#!/usr/bin/env python3
"""Build the ImageNet ControlNet per-method ratio plot for report.md."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
FULL_DATA_ACC = 0.7400
SAFEFIX_RATIO1_NO_RESUME_ACC = 0.7392

METHODS = {
    "Random": [0.7269, 0.7308, 0.7400, 0.7423, 0.7482, FULL_DATA_ACC],
    "Stratified": [0.7271, 0.7369, 0.7446, 0.7346, 0.7469, FULL_DATA_ACC],
    "Entropy": [0.7254, 0.7292, 0.7331, 0.7354, 0.7462, FULL_DATA_ACC],
    "Coreset": [0.7269, 0.7369, 0.7400, 0.7392, 0.7346, FULL_DATA_ACC],
    "BADGE": [0.7323, 0.7423, 0.7469, 0.7477, 0.7438, FULL_DATA_ACC],
    "UHerding": [0.7331, 0.7469, 0.7423, 0.7469, 0.7500, FULL_DATA_ACC],
    "PPAL": [0.7285, 0.7392, 0.7400, 0.7423, 0.7508, FULL_DATA_ACC],
    "Ours": [0.7369, 0.7492, 0.7508, 0.7515, 0.7562, FULL_DATA_ACC],
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


def main() -> None:
    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)

    csv_path = out_dir / "imagenet_controlnet_methods_test_acc.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ratio", *METHODS.keys(), "SafeFix"])
        for i, ratio in enumerate(RATIOS):
            writer.writerow(
                [
                    ratio,
                    *[values[i] for values in METHODS.values()],
                    SAFEFIX_RATIO1_NO_RESUME_ACC if ratio == 1.0 else "",
                ]
            )

    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    for method, values in METHODS.items():
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

    ax.axhline(
        SAFEFIX_RATIO1_NO_RESUME_ACC,
        color="0.25",
        linestyle="--",
        linewidth=1.8,
        label="SafeFix",
    )
    ax.plot(
        [1.0],
        [SAFEFIX_RATIO1_NO_RESUME_ACC],
        marker="*",
        markersize=11,
        color="0.15",
        linestyle="None",
    )

    ax.set_title("ImageNet ControlNet, Test ACC")
    ax.set_xlabel("Ratio")
    ax.set_ylabel("Test ACC")
    ax.set_xticks(RATIOS)
    ax.set_ylim(0.722, 0.759)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, frameon=False, fontsize=8.5)

    fig.savefig(out_dir / "imagenet_controlnet_methods_test_acc.png", dpi=180)


if __name__ == "__main__":
    main()
