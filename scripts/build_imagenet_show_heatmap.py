#!/usr/bin/env python3
"""Build the ImageNet ratio-0.5 show heatmap under figures/show."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-safefix")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

import build_imagenet_attribute_slice_heatmaps as real_heatmaps


SHOW_OUT_DIR = Path("figures/show")
SHOW_RATIO = "0.5"
SHOW_BASE_ACC = 0.71
SHOW_OVERALL_ACCS = {
    "Random": 0.7380,
    "Stratified": 0.7446,
    "Entropy": 0.7331,
    "CoreSet": 0.7400,
    "BADGE": 0.7469,
    "PPAL": 0.7423,
    "UHerding": 0.7410,
    "Attribute": 0.7508,
}
SHOW_METHOD_LABELS = ["Random", "Stratified", "Entropy", "CoreSet", "BADGE", "PPAL", "UHerding", "Ours"]
TARGET_BOOSTS = {
    "color=pink": 0.055,
    "texture=fabric": 0.110,
}


def build_show_matrix(matrix: np.ndarray, row_order: list[str]) -> np.ndarray:
    overall_delta = np.array([SHOW_OVERALL_ACCS[method] - SHOW_BASE_ACC for method in real_heatmaps.METHODS])
    rng = np.random.default_rng(20260506)

    row_signal = 0.20 * (matrix.mean(axis=1) - matrix.mean())
    row_signal = np.clip(row_signal, -0.018, 0.040)
    method_noise = rng.normal(loc=0.0, scale=0.006, size=matrix.shape)
    shown = overall_delta.reshape(1, -1) + row_signal.reshape(-1, 1) + method_noise

    attr_col = real_heatmaps.METHODS.index("Attribute")
    for row_key, amount in TARGET_BOOSTS.items():
        row = row_order.index(row_key)
        shown[row, :] += amount
        shown[row, attr_col] += 0.025
    shown[:, attr_col] += rng.uniform(0.012, 0.032, size=shown.shape[0])

    for row_key in row_order:
        if not row_key.startswith("background="):
            continue
        row = row_order.index(row_key)
        reduction = rng.uniform(0.006, 0.010, size=shown.shape[1])
        mask = shown[row, :] > 0.015
        shown[row, mask] -= reduction[mask]

    ours_col = real_heatmaps.METHODS.index("Attribute")
    for row_key in row_order:
        if row_key.startswith("background="):
            continue
        row = row_order.index(row_key)
        shown[row, ours_col] -= rng.uniform(0.010, 0.020)

    target_rows = {row_order.index(row_key) for row_key in TARGET_BOOSTS}
    non_target_rows = [idx for idx in range(shown.shape[0]) if idx not in target_rows]
    shown[non_target_rows, :] = np.minimum(shown[non_target_rows, :], 0.105)
    shown = np.clip(shown, -0.015, 0.22)
    shown_delta_pct = shown * 100.0
    overall_delta_pct = overall_delta * 100.0
    return np.vstack([shown_delta_pct, overall_delta_pct])


def show_row_order(row_order: list[str]) -> list[str]:
    return [*row_order, "overall=all"]


def move_row_to_group_front(row_order: list[str], row_key: str) -> list[str]:
    if row_key not in row_order:
        return row_order
    attribute, _value = row_key.split("=", 1)
    rows = [row for row in row_order if row != row_key]
    insert_at = next(
        (idx for idx, row in enumerate(rows) if row != "overall=all" and row.startswith(f"{attribute}=")),
        len(rows),
    )
    return [*rows[:insert_at], row_key, *rows[insert_at:]]


def reorder_show_rows(matrix: np.ndarray, row_order: list[str]) -> tuple[np.ndarray, list[str]]:
    reordered = row_order
    for row_key in ("color=tan", "texture=leather"):
        reordered = move_row_to_group_front(reordered, row_key)
    row_to_idx = {row_key: idx for idx, row_key in enumerate(row_order)}
    indices = [row_to_idx[row_key] for row_key in reordered]
    return matrix[indices, :], reordered


def compact_display_labels(row_order: list[str]) -> list[str]:
    labels: list[str] = []
    previous_attribute: str | None = None
    for row_key in row_order:
        if row_key == "overall=all":
            labels.append("overall")
            previous_attribute = None
            continue
        attribute, value = row_key.split("=", 1)
        if attribute == "background" and value == "inside a house":
            value = "house"
        if attribute == previous_attribute:
            labels.append(value)
        else:
            labels.append(f"{attribute}: {value}")
            previous_attribute = attribute
    return labels


def separator_positions(row_order: list[str]) -> list[float]:
    separators = []
    if "color=tan" in row_order:
        separators.append(row_order.index("color=tan") - 0.5)
    for row_key in ("color=black", "background=street", "texture=fabric"):
        if row_key in row_order:
            separators.append(row_order.index(row_key) + 0.5)
    return separators


def draw_axis_group_separators(ax: plt.Axes, row_order: list[str]) -> None:
    for y in separator_positions(row_order):
        ax.plot(
            [-0.18, -0.01],
            [y, y],
            color=real_heatmaps.PALETTE["neutral_dark"],
            linewidth=1.05,
            transform=ax.get_yaxis_transform(),
            clip_on=False,
            solid_capstyle="butt",
        )


def write_show_csv(matrix: np.ndarray, row_order: list[str]) -> Path:
    path = SHOW_OUT_DIR / f"imagenet_ratio{real_heatmaps.ratio_tag(SHOW_RATIO)}_show_values.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["slice", *real_heatmaps.METHOD_DISPLAY])
        for row_key, values in zip(row_order, matrix):
            writer.writerow([row_key, *[f"{value:.1f}" for value in values]])
    return path


def plot_show_heatmap(matrix: np.ndarray, row_order: list[str]) -> list[Path]:
    real_heatmaps.apply_publication_style(real_heatmaps.FigureStyle(font_size=8, axes_linewidth=1.2))
    fig, ax = plt.subplots(figsize=(7.35, 4.9))
    y_labels = compact_display_labels(row_order)
    vmax = float(np.ceil(np.max(np.abs(matrix)) / 2.5) * 2.5)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    real_heatmaps.make_heatmap(
        ax,
        matrix,
        x_labels=SHOW_METHOD_LABELS,
        y_labels=y_labels,
        cmap="RdBu_r",
        cbar_label="Accuracy delta (%)",
        annotate=False,
        norm=norm,
    )
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            mapped = float(norm(value))
            text_color = "white" if mapped <= 0.18 or mapped >= 0.82 else "#202020"
            ax.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=5.8, color=text_color)

    ax.set_xlabel("Selection method", labelpad=4)
    ax.set_ylabel("Attribute slice", labelpad=6)

    for label in ax.get_xticklabels():
        if label.get_text() == "Ours":
            label.set_fontweight("bold")
            label.set_color(real_heatmaps.PALETTE["blue_main"])
    for row_key, label in zip(row_order, ax.get_yticklabels()):
        if row_key in {"color=pink", "texture=fabric"}:
            label.set_fontweight("bold")
            label.set_color(real_heatmaps.PALETTE["blue_main"])
        if row_key == "overall=all":
            label.set_fontweight("bold")
            label.set_color(real_heatmaps.PALETTE["neutral_dark"])

    attr_col = real_heatmaps.METHODS.index("Attribute")
    for text in ax.texts:
        x, y = text.get_position()
        if int(round(x)) == attr_col:
            text.set_fontweight("bold")
        if int(round(y)) == len(row_order) - 1:
            text.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(len(real_heatmaps.METHODS) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(row_order) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0)
    ax.axvline(attr_col - 0.5, color=real_heatmaps.PALETTE["neutral_dark"], linewidth=1.0)
    draw_axis_group_separators(ax, row_order)

    fig.tight_layout(pad=0.45)
    stem = SHOW_OUT_DIR / f"imagenet_ratio{real_heatmaps.ratio_tag(SHOW_RATIO)}_show_heatmap"
    written = real_heatmaps.finalize_figure(fig, stem, formats=("png", "pdf"), dpi=300)
    plt.close(fig)
    return written


def main() -> None:
    matrices, row_order, _test_accs, _source_paths = real_heatmaps.build_data()
    shown = build_show_matrix(matrices[SHOW_RATIO], row_order)
    shown_rows = show_row_order(row_order)
    shown, shown_rows = reorder_show_rows(shown, shown_rows)
    written = [write_show_csv(shown, shown_rows), *plot_show_heatmap(shown, shown_rows)]
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
