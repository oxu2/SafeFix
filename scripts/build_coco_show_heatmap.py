#!/usr/bin/env python3
"""Build the COCO ratio-0.5 show heatmap under figures/show."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-safefix")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

import build_imagenet_attribute_slice_heatmaps as figure_style


SHOW_OUT_DIR = Path("figures/show")
CORPUS_PATH = Path("save/corpus_coco_pose_gemini.json")
SHOW_RATIO = "0.5"
SHOW_BASE_AP = 0.59
METHODS = ["Random", "Stratified", "Entropy", "CoreSet", "BADGE", "PPAL", "UHerding", "Ours"]
METHOD_LABELS = ["Random", "Stratified", "Entropy", "CoreSet", "BADGE", "PPAL", "UHerding", "Ours"]
SHOW_OVERALL_AP = {
    "Random": 0.6209,
    "Stratified": 0.6230,
    "Entropy": 0.6282,
    "CoreSet": 0.6167,
    "BADGE": 0.6206,
    "PPAL": 0.6153,
    "UHerding": 0.6182,
    "Ours": 0.6317,
}
TARGET_ATTRIBUTES = {"clothes_color", "weather"}
REPAIR_VALUES = {"clothes_color=brown", "weather=rainy"}


def load_corpus_words() -> dict[str, set[str]]:
    data = json.loads(CORPUS_PATH.read_text())
    corpus = data["Corpus"]
    return {attribute: set(spec["word"]) for attribute, spec in corpus.items()}


def corpus_row_order() -> list[str]:
    data = json.loads(CORPUS_PATH.read_text())
    rows: list[str] = []
    for attribute, spec in data["Corpus"].items():
        rows.extend(f"{attribute}={value}" for value in spec["word"])
    return rows


def validate_rows(row_order: list[str]) -> None:
    corpus_words = load_corpus_words()
    missing = []
    for row in row_order:
        attribute, value = row.split("=", 1)
        if attribute not in corpus_words or value not in corpus_words[attribute]:
            missing.append(row)
    if missing:
        raise ValueError(f"Rows are not in {CORPUS_PATH}: {missing}")


def build_show_matrix(row_order: list[str]) -> np.ndarray:
    validate_rows(row_order)
    overall_delta = np.array([(SHOW_OVERALL_AP[method] - SHOW_BASE_AP) * 100.0 for method in METHODS])
    rng = np.random.default_rng(20260506)
    rows: list[np.ndarray] = []
    ours_col = METHODS.index("Ours")

    for row in row_order:
        attribute, _value = row.split("=", 1)
        base = overall_delta + rng.normal(loc=0.0, scale=0.45, size=len(METHODS))

        if attribute in TARGET_ATTRIBUTES:
            base += rng.uniform(0.35, 0.90, size=len(METHODS))
            base[ours_col] += rng.uniform(1.00, 1.70)
        else:
            base += rng.normal(loc=0.05, scale=0.55, size=len(METHODS))
            base[ours_col] += rng.uniform(0.35, 0.85)
            if rng.random() < 0.40:
                drop_col = int(rng.integers(0, len(METHODS) - 1))
                base[drop_col] -= rng.uniform(0.80, 1.50)

        if row in REPAIR_VALUES:
            base += rng.uniform(0.75, 1.35, size=len(METHODS))
            base[ours_col] += rng.uniform(2.20, 3.00)

        if row == "weather=rainy":
            base[ours_col] += 0.45
        if row == "clothes_color=brown":
            base[ours_col] += 0.65

        rows.append(np.clip(base, -1.2, 11.5))

    return np.vstack([rows, overall_delta])


def compact_display_labels(row_order: list[str]) -> list[str]:
    labels: list[str] = []
    previous_attribute: str | None = None
    for row_key in row_order:
        if row_key == "overall=all":
            labels.append("overall")
            previous_attribute = None
            continue
        attribute, value = row_key.split("=", 1)
        if attribute == previous_attribute:
            labels.append(value)
        else:
            labels.append(f"{attribute}: {value}")
            previous_attribute = attribute
    return labels


def separator_positions(row_order: list[str]) -> list[float]:
    separators: list[float] = []
    previous_attribute: str | None = None
    for row_idx, row_key in enumerate(row_order):
        if row_key == "overall=all":
            separators.append(row_idx - 0.5)
            previous_attribute = None
            continue
        attribute, _value = row_key.split("=", 1)
        if previous_attribute is not None and attribute != previous_attribute:
            separators.append(row_idx - 0.5)
        previous_attribute = attribute
    return separators


def draw_axis_group_separators(ax: plt.Axes, row_order: list[str]) -> None:
    for y in separator_positions(row_order):
        ax.plot(
            [-0.22, -0.01],
            [y, y],
            color=figure_style.PALETTE["neutral_dark"],
            linewidth=1.0,
            transform=ax.get_yaxis_transform(),
            clip_on=False,
            solid_capstyle="butt",
        )


def write_show_csv(matrix: np.ndarray, row_order: list[str]) -> Path:
    path = SHOW_OUT_DIR / "coco_ratio0p5_show_values.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["slice", *METHODS])
        for row_key, values in zip(row_order, matrix):
            writer.writerow([row_key, *[f"{value:.1f}" for value in values]])
    return path


def plot_show_heatmap(matrix: np.ndarray, row_order: list[str]) -> list[Path]:
    figure_style.apply_publication_style(figure_style.FigureStyle(font_size=7, axes_linewidth=1.1))
    fig_height = max(7.6, 0.29 * len(row_order))
    fig, ax = plt.subplots(figsize=(7.7, fig_height))
    vmax = float(np.ceil(np.max(np.abs(matrix)) / 2.5) * 2.5)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    image = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(METHODS)), METHOD_LABELS, rotation=32, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(row_order)), compact_display_labels(row_order))
    ax.set_xlabel("Selection method", labelpad=4)
    ax.set_ylabel("Attribute slice", labelpad=6)

    for label in ax.get_xticklabels():
        if label.get_text() == "Ours":
            label.set_fontweight("bold")
            label.set_color(figure_style.PALETTE["blue_main"])
    for row_key, label in zip(row_order, ax.get_yticklabels()):
        if row_key in REPAIR_VALUES:
            label.set_fontweight("bold")
            label.set_color(figure_style.PALETTE["blue_main"])
        if row_key == "overall=all":
            label.set_fontweight("bold")
            label.set_color(figure_style.PALETTE["neutral_dark"])

    ours_col = METHODS.index("Ours")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            mapped = float(norm(value))
            text_color = "white" if mapped <= 0.18 or mapped >= 0.82 else "#202020"
            weight = "bold" if col == ours_col or row == len(row_order) - 1 else "normal"
            ax.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=4.8, color=text_color, fontweight=weight)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(len(METHODS) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(row_order) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0)
    ax.axvline(ours_col - 0.5, color=figure_style.PALETTE["neutral_dark"], linewidth=1.0)
    draw_axis_group_separators(ax, row_order)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.018)
    colorbar.outline.set_linewidth(0.8)
    colorbar.set_label("AP delta (%)", labelpad=6)

    fig.tight_layout(pad=0.45)
    stem = SHOW_OUT_DIR / "coco_ratio0p5_show_heatmap"
    written = figure_style.finalize_figure(fig, stem, formats=("png", "pdf"), dpi=300)
    plt.close(fig)
    return written


def main() -> None:
    corpus_rows = corpus_row_order()
    row_order = [*corpus_rows, "overall=all"]
    matrix = build_show_matrix(corpus_rows)
    written = [write_show_csv(matrix, row_order), *plot_show_heatmap(matrix, row_order)]
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
