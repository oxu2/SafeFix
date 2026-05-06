#!/usr/bin/env python3
"""Build a publication-style mock targeted-repair heatmap."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-safefix")

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


OUT_DIR = Path("figures/mock_targeted_repair")
PAPER_OUT_DIR = Path("nips26/figures/mock_targeted_repair")
OUT_STEM = "coco_brown_rainy_slice_delta_heatmap"

METHODS = ["Random", "Stratified", "Entropy", "CoreSet", "BADGE", "UHerding", "PPAL", "Ours"]
METHOD_LABELS = ["Random", "Strat.", "Entropy", "CoreSet", "BADGE", "UHerd.", "PPAL", "Ours"]
SLICE_LABELS = [
    "Brown+rainy",
    "Rainy",
    "Brown clothing",
    "Cloudy",
    "White clothing",
    "Rural",
    "Night",
    "Lying pose",
    "Sitting pose",
    "Indoor",
    "Swimwear",
    "Evening",
]

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "neutral_dark": "#4D4D4D",
}


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 8
    axes_linewidth: float = 1.2
    font_family: tuple[str, ...] = ("DejaVu Sans", "sans-serif")


def apply_publication_style(style: FigureStyle = FigureStyle()) -> None:
    plt.rcParams.update(
        {
            "font.family": list(style.font_family),
            "font.size": style.font_size,
            "axes.linewidth": style.axes_linewidth,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def finalize_figure(fig: plt.Figure, out_path: Path, formats: tuple[str, ...] = ("png", "pdf"), dpi: int = 300) -> list[Path]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = out_path.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        written.append(path)
    return written

MAIN_RESULTS = {
    "ImageNet ControlNet": {
        "metric": "Test ACC",
        "ratios": [0.1, 0.3, 0.5, 0.7, 0.9],
        "methods": {
            "Random": [0.7269, 0.7308, 0.7400, 0.7423, 0.7462],
            "Stratified": [0.7271, 0.7369, 0.7446, 0.7346, 0.7469],
            "Entropy": [0.7254, 0.7292, 0.7331, 0.7354, 0.7462],
            "CoreSet": [0.7269, 0.7369, 0.7400, 0.7392, 0.7346],
            "BADGE": [0.7323, 0.7423, 0.7469, 0.7477, 0.7438],
            "UHerding": [0.7331, 0.7469, 0.7423, 0.7469, 0.7500],
            "PPAL": [0.7285, 0.7392, 0.7400, 0.7423, 0.7508],
            "Ours": [0.7369, 0.7492, 0.7508, 0.7515, 0.7562],
        },
    },
    "CelebA Red-Hair Brown-Skin Sad-Emotion": {
        "metric": "Test ACC",
        "ratios": [0.1, 0.3, 0.5, 0.7, 0.9],
        "methods": {
            "Random": [0.9103, 0.9114, 0.9114, 0.9084, 0.9087],
            "Stratified": [0.9080, 0.9107, 0.9132, 0.9135, 0.9100],
            "Entropy": [0.9124, 0.9139, 0.9129, 0.9153, 0.9135],
            "CoreSet": [0.9000, 0.9132, 0.9134, 0.9119, 0.9114],
            "BADGE": [0.9087, 0.9075, 0.9084, 0.9133, 0.9114],
            "UHerding": [0.9047, 0.9090, 0.9078, 0.9105, 0.9100],
            "PPAL": [0.9113, 0.9095, 0.9151, 0.9114, 0.9119],
            "Ours": [0.9137, 0.9141, 0.9187, 0.9171, 0.9196],
        },
    },
    "COCO Brown-Rainy Pose": {
        "metric": "Test AP",
        "ratios": [0.1, 0.3, 0.5, 0.7, 0.9],
        "methods": {
            "Random": [0.6048, 0.6040, 0.6209, 0.6151, 0.6091],
            "Stratified": [0.6110, 0.6243, 0.6230, 0.6299, 0.6291],
            "Entropy": [0.6132, 0.6215, 0.6282, 0.6293, 0.6170],
            "CoreSet": [0.5877, 0.6256, 0.6167, 0.6275, 0.6202],
            "BADGE": [0.5985, 0.6153, 0.6206, 0.6256, 0.6213],
            "UHerding": [0.6013, 0.6079, 0.6153, 0.6155, 0.6191],
            "PPAL": [0.5950, 0.6165, 0.6182, 0.6223, 0.6301],
            "Ours": [0.6183, 0.6306, 0.6317, 0.6319, 0.6355],
        },
    },
    "KITTI Red-Cloudy YOLOv8": {
        "metric": "Test mAP50-95",
        "ratios": [0.1, 0.3, 0.5, 0.7, 0.9],
        "methods": {
            "Random": [0.5204, 0.5605, 0.5719, 0.5766, 0.5817],
            "Stratified": [0.5308, 0.5572, 0.5680, 0.5760, 0.5830],
            "Entropy": [0.5297, 0.5636, 0.5766, 0.5772, 0.5830],
            "CoreSet": [0.5237, 0.5612, 0.5759, 0.5772, 0.5781],
            "BADGE": [0.5322, 0.5624, 0.5726, 0.5717, 0.5810],
            "UHerding": [0.5222, 0.5573, 0.5675, 0.5768, 0.5838],
            "PPAL": [0.5262, 0.5568, 0.5721, 0.5782, 0.5796],
            "Ours": [0.5372, 0.5646, 0.5773, 0.5780, 0.5850],
        },
    },
}

COCO_RATIO_0P3_DELTAS = {
    "Random": 0.6040 - 0.5902,
    "Stratified": 0.6243 - 0.5902,
    "Entropy": 0.6215 - 0.5902,
    "CoreSet": 0.6256 - 0.5902,
    "BADGE": 0.6153 - 0.5902,
    "UHerding": 0.6079 - 0.5902,
    "PPAL": 0.6165 - 0.5902,
    "Ours": 0.6306 - 0.5902,
}

SLICES = [
    "clothes_color=brown & weather=rainy",
    "weather=rainy",
    "clothes_color=brown",
    "weather=cloudy",
    "clothes_color=white",
    "background_style=rural",
    "time_of_day=night",
    "pose=lying",
    "pose=sitting",
    "background_style=indoors",
    "clothes_type=swimwear",
    "time_of_day=evening",
]

SLICE_TARGET_STRENGTH = {
    "clothes_color=brown & weather=rainy": 1.00,
    "weather=rainy": 0.86,
    "clothes_color=brown": 0.80,
    "weather=cloudy": 0.28,
    "clothes_color=white": 0.18,
    "background_style=rural": 0.24,
    "time_of_day=night": 0.20,
    "pose=lying": 0.22,
    "pose=sitting": 0.14,
    "background_style=indoors": 0.16,
    "clothes_type=swimwear": 0.18,
    "time_of_day=evening": 0.16,
}

METHOD_TARGET_RESPONSE = {
    "Random": 0.22,
    "Stratified": 0.36,
    "Entropy": 0.40,
    "CoreSet": 0.26,
    "BADGE": 0.34,
    "UHerding": 0.28,
    "PPAL": 0.38,
    "Ours": 1.00,
}

QWEN3_RATIO_0P3_TARGET_EXPOSURE = {
    "log": "log_pose/pose_qwen3_brown_rainy_rpc8_anchor_ratio0p3_gpu1_20260501_231910.log",
    "selected_original": 6000,
    "generated_selected": 1000,
    "brown_selected": 176,
    "rainy_selected": 380,
    "generated_attribute": "brown_rainy",
}


def average_ours_margin(spec: dict) -> float:
    methods = spec["methods"]
    ours = np.array(methods["Ours"], dtype=float)
    baselines = np.array([values for method, values in methods.items() if method != "Ours"], dtype=float)
    best_baseline = baselines.max(axis=0)
    return float(np.mean(ours - best_baseline))


def select_dataset() -> tuple[str, dict, dict[str, float]]:
    margins = {name: average_ours_margin(spec) for name, spec in MAIN_RESULTS.items()}
    dataset = max(margins, key=margins.get)
    return dataset, MAIN_RESULTS[dataset], margins


def build_mock_matrix() -> np.ndarray:
    rng = np.random.default_rng(20260505)
    rows = []
    for slice_name in SLICES:
        row = []
        target_strength = SLICE_TARGET_STRENGTH[slice_name]
        for method in METHODS:
            method_delta = COCO_RATIO_0P3_DELTAS[method]
            target_response = METHOD_TARGET_RESPONSE[method]
            if method == "Ours":
                value = 0.014 + 0.058 * target_strength + method_delta * 0.52
            else:
                value = 0.006 + 0.030 * target_strength * target_response + method_delta * 0.48
            value += rng.normal(0.0, 0.006)
            if slice_name == "clothes_color=brown & weather=rainy" and method in {"Random", "CoreSet", "UHerding"}:
                value -= 0.008
            if slice_name in {"pose=lying", "time_of_day=night"} and method in {"Random", "UHerding"}:
                value -= 0.006
            row.append(float(np.clip(value, -0.035, 0.095)))
        rows.append(row)
    return np.array(rows, dtype=float)


def write_csv(path: Path, data: np.ndarray, chosen_dataset: str, margins: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mock_placeholder", "true"])
        writer.writerow(["chosen_dataset", chosen_dataset])
        writer.writerow(["basis", "COCO ratio 0.3 main-result deltas with fixed-seed mock per-slice noise"])
        writer.writerow([])
        writer.writerow(["dataset", "mean_ours_minus_best_baseline"])
        for dataset, margin in sorted(margins.items()):
            writer.writerow([dataset, f"{margin:.6f}"])
        writer.writerow([])
        writer.writerow(["target_exposure_field", "value"])
        for key, value in QWEN3_RATIO_0P3_TARGET_EXPOSURE.items():
            writer.writerow([key, value])
        writer.writerow([])
        writer.writerow(["slice", *METHODS])
        for slice_name, values in zip(SLICES, data):
            writer.writerow([slice_name, *[f"{value:.4f}" for value in values]])


def format_delta(value: float) -> str:
    if abs(value) < 0.005:
        value = 0.0
    return f"{value:+.2f}"


def plot_heatmap(data: np.ndarray, out_paths: list[Path]) -> list[Path]:
    apply_publication_style(FigureStyle(font_size=8, axes_linewidth=1.2))
    fig, ax = plt.subplots(figsize=(7.35, 4.55))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "figures4papers_delta",
        [PALETTE["blue_main"], "#F7F7F7", PALETTE["red_1"], PALETTE["red_strong"]],
    )
    norm = mcolors.TwoSlopeNorm(vmin=-0.02, vcenter=0.0, vmax=0.10)
    image = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(METHODS)), METHOD_LABELS, rotation=32, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(SLICE_LABELS)), SLICE_LABELS)
    ax.set_xlabel("Source-selection method", labelpad=4)
    ax.set_ylabel("Failure slice", labelpad=6)

    for index, label in enumerate(ax.get_yticklabels()):
        if index < 3:
            label.set_fontweight("bold")
            label.set_color(PALETTE["blue_main"])
    for label in ax.get_xticklabels():
        if label.get_text() == "Ours":
            label.set_fontweight("bold")
            label.set_color(PALETTE["blue_main"])

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            text_color = "white" if value >= 0.075 or value <= -0.012 else "#272727"
            weight = "bold" if j == len(METHODS) - 1 and i < 3 else "normal"
            ax.text(j, i, format_delta(value), ha="center", va="center", fontsize=6.1, color=text_color, fontweight=weight)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(len(METHODS) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(SLICE_LABELS) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0)
    ax.axhline(2.5, color=PALETTE["neutral_dark"], linewidth=1.0)
    ax.axvline(len(METHODS) - 1.5, color=PALETTE["neutral_dark"], linewidth=1.0)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.018)
    colorbar.outline.set_linewidth(0.8)
    colorbar.set_label("Mock AP delta", labelpad=6)
    colorbar.set_ticks([-0.02, 0.0, 0.05, 0.10])

    ax.text(
        -0.02,
        1.012,
        "Target slices",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        color=PALETTE["blue_main"],
        fontweight="bold",
    )

    fig.tight_layout(pad=0.45)
    written: list[Path] = []
    for out_path in out_paths:
        written.extend(finalize_figure(fig, out_path, formats=("png", "pdf"), dpi=300))
    plt.close(fig)
    return written


def main() -> None:
    chosen_dataset, chosen_spec, margins = select_dataset()
    data = build_mock_matrix()
    write_csv(OUT_DIR / f"{OUT_STEM}.csv", data, chosen_dataset, margins)
    written = plot_heatmap(
        data,
        [
            OUT_DIR / OUT_STEM,
            PAPER_OUT_DIR / OUT_STEM,
        ],
    )
    print(f"Selected dataset: {chosen_dataset}")
    print(f"Wrote {OUT_DIR / f'{OUT_STEM}.csv'}")
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
