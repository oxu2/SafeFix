#!/usr/bin/env python3
"""Build real ImageNet attribute-slice accuracy-delta heatmaps from logs."""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-safefix")

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


OUT_DIR = Path("figures/imagenet_attribute_slice_heatmaps")
PAPER_OUT_DIR = Path("nips26/figures/imagenet_attribute_slice_heatmaps")
RATIOS = ["0.1", "0.3", "0.5", "0.7", "0.9"]
METHODS = ["Random", "Stratified", "Entropy", "CoreSet", "BADGE", "PPAL", "UHerding", "Attribute"]
METHOD_DISPLAY = ["Random", "Stratified", "Entropy", "CoreSet", "BADGE", "PPAL", "UHerding", "Ours"]
METHOD_LABELS = ["Random", "Strat.", "Entropy", "CoreSet", "BADGE", "PPAL", "UHerd.", "Ours"]


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 8
    axes_linewidth: float = 1.2
    font_family: tuple[str, ...] = ("DejaVu Sans", "sans-serif")


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_light": "#F2C8C4",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "neutral_dark": "#4D4D4D",
}


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


def finalize_figure(fig: plt.Figure, out_stem: Path, formats: tuple[str, ...] = ("png", "pdf"), dpi: int = 300) -> list[Path]:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        path = out_stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        written.append(path)
    return written


def ratio_tag(ratio: str) -> str:
    return "0p" + ratio.split(".")[1]


def method_logs() -> dict[str, dict[str, Path]]:
    logs: dict[str, dict[str, Path]] = {}
    for ratio in RATIOS:
        rp = ratio_tag(ratio)
        logs[ratio] = {
            "Random": Path(f"logs/logs/imagenet/resume_lr5_epoch15_random_resnet_imagenet_1000_ratio{ratio}.log"),
            "Stratified": Path(f"logs/logs/imagenet/resume_lr5_epoch15_stratified_resnet_imagenet_1000_ratio{ratio}.log"),
            "Entropy": Path(f"logs/logs/imagenet/resume_lr5_epoch15_uncertainty_stratified_resnet_imagenet_1000_ratio{ratio}.log"),
            "CoreSet": Path(f"logs/logs/imagenet/resume_lr5_epoch15_coreset_resnet_imagenet_1000_ratio{ratio}.log"),
            "BADGE": Path(f"logs/imagenet/instructpix2pix_badge_resnet_imagenet_1000_ratio{rp}_gpu{badge_gpu(ratio)}_{badge_stamp(ratio)}.log"),
            "PPAL": Path(f"logs/imagenet/ppal_resnet_imagenet_1000_ratio{rp}_gpu3_20260502_185507.log"),
            "UHerding": Path(f"logs/imagenet/uherding_resnet_imagenet_1000_ratio{rp}_gpu1_20260502_140255.log"),
            "Attribute": attribute_log_for_ratio(ratio),
        }
    logs["0.9"]["BADGE"] = Path("logs/imagenet/badge_resnet_imagenet_1000_ratio0p9_gpu1_20260504_004831.log")
    return logs


def badge_gpu(ratio: str) -> str:
    return {"0.1": "1", "0.3": "1", "0.5": "2", "0.7": "3", "0.9": "1"}[ratio]


def badge_stamp(ratio: str) -> str:
    return "20260429_2130" if ratio == "0.9" else "20260429_065612"


def attribute_log_for_ratio(ratio: str) -> Path:
    return {
        "0.1": Path("logs/imagenet/controlnet_rpc8_no_r_resnet_imagenet_1000_ratio0p1_gpu3_20260501_231910.log"),
        "0.3": Path("logs/imagenet/controlnet_rpc8_noentropy_only_c_resnet_imagenet_1000_ratio0p3_gpu3_20260502_023905.log"),
        "0.5": Path("logs/imagenet/controlnet_rpc8_noentropy_none_resnet_imagenet_1000_ratio0p5_gpu3_20260502_023905.log"),
        "0.7": Path("logs/imagenet/controlnet_rpc_numeric_r003_p05_cinf_resnet_imagenet_1000_ratio0p7_gpu3_20260504_120159.log"),
        "0.9": Path("logs/imagenet/controlnet_rpc8_only_p_resnet_imagenet_1000_ratio0p9_gpu3_20260501_231910.log"),
    }[ratio]


def parse_log(path: Path) -> tuple[float, dict[str, tuple[float, float, float]]]:
    text = path.read_text(errors="ignore")
    if "Attribute-specific accuracy comparison:" not in text:
        raise ValueError(f"missing Attribute-specific accuracy comparison block: {path}")
    test_acc_match = re.findall(r"Model Test ACC:\s*([0-9.]+)", text)
    if not test_acc_match:
        raise ValueError(f"missing Model Test ACC: {path}")
    rows: dict[str, tuple[float, float, float]] = {}
    pattern = re.compile(r"^([^,\n]+),\s*([^,\n]+),\s*([0-9.]+)\s*->\s*([0-9.]+):\s*(\w+)", re.MULTILINE)
    for attr, value, before, after, _status in pattern.findall(text):
        before_f = float(before)
        after_f = float(after)
        key = f"{attr.strip()}={value.strip()}"
        rows[key] = (before_f, after_f, after_f - before_f)
    if not rows:
        raise ValueError(f"no parsed attribute rows: {path}")
    return float(test_acc_match[-1]), rows


def display_label(row_key: str) -> str:
    attr, value = row_key.split("=", 1)
    return f"{attr}: {value}"


def format_delta(value: float) -> str:
    if abs(value) < 0.005:
        value = 0.0
    return f"{value:+.2f}"


def make_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    cmap: str | mcolors.Colormap = "RdBu_r",
    cbar_label: str | None = None,
    annotate: bool = False,
    norm: mcolors.Normalize | None = None,
) -> matplotlib.image.AxesImage:
    image = ax.imshow(matrix, cmap=cmap, aspect="auto", norm=norm)
    ax.set_xticks(np.arange(len(x_labels)), x_labels, rotation=32, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(y_labels)), y_labels)
    if annotate:
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                if norm is not None:
                    mapped = float(norm(value))
                    text_color = "white" if mapped <= 0.18 or mapped >= 0.82 else "#202020"
                else:
                    text_color = "white" if value >= matrix.max() * 0.56 else "#202020"
                ax.text(col, row, format_delta(value), ha="center", va="center", fontsize=5.8, color=text_color)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.018)
    colorbar.outline.set_linewidth(0.8)
    if cbar_label is not None:
        colorbar.set_label(cbar_label, labelpad=6)
    return image


def build_data() -> tuple[dict[str, np.ndarray], list[str], dict[tuple[str, str], float], dict[tuple[str, str], Path]]:
    logs = method_logs()
    parsed: dict[tuple[str, str], dict[str, tuple[float, float, float]]] = {}
    test_accs: dict[tuple[str, str], float] = {}
    source_paths: dict[tuple[str, str], Path] = {}
    row_order: list[str] | None = None
    for ratio in RATIOS:
        for method in METHODS:
            path = logs[ratio][method]
            test_acc, rows = parse_log(path)
            parsed[(ratio, method)] = rows
            test_accs[(ratio, method)] = test_acc
            source_paths[(ratio, method)] = path
            if row_order is None:
                row_order = list(rows.keys())
            else:
                missing = set(row_order) - set(rows)
                if missing:
                    raise ValueError(f"{path} is missing rows: {sorted(missing)}")
    assert row_order is not None
    matrices: dict[str, np.ndarray] = {}
    for ratio in RATIOS:
        matrix = np.zeros((len(row_order), len(METHODS)), dtype=float)
        for col, method in enumerate(METHODS):
            rows = parsed[(ratio, method)]
            for row, key in enumerate(row_order):
                matrix[row, col] = rows[key][2]
        matrices[ratio] = matrix
    return matrices, row_order, test_accs, source_paths


def write_ratio_csv(
    ratio: str,
    matrix: np.ndarray,
    row_order: list[str],
    test_accs: dict[tuple[str, str], float],
    source_paths: dict[tuple[str, str], Path],
) -> Path:
    path = OUT_DIR / f"imagenet_ratio{ratio_tag(ratio)}_attribute_slice_delta_heatmap.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "ImageNet ControlNet / pinkcolor_fabrictexture"])
        writer.writerow(["ratio", ratio])
        writer.writerow(["metric", "attribute-specific accuracy delta (after - before)"])
        writer.writerow([])
        writer.writerow(["method", "test_acc", "source_log"])
        for method, display in zip(METHODS, METHOD_DISPLAY):
            writer.writerow([display, f"{test_accs[(ratio, method)]:.6f}", source_paths[(ratio, method)]])
        writer.writerow([])
        writer.writerow(["slice", *METHOD_DISPLAY])
        for row_key, values in zip(row_order, matrix):
            writer.writerow([row_key, *[f"{value:.6f}" for value in values]])
    return path


def write_summary_csv(
    matrices: dict[str, np.ndarray],
    row_order: list[str],
    test_accs: dict[tuple[str, str], float],
    source_paths: dict[tuple[str, str], Path],
) -> Path:
    path = OUT_DIR / "imagenet_attribute_slice_delta_heatmap_sources.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ratio", "method", "test_acc", "source_log"])
        for ratio in RATIOS:
            for method, display in zip(METHODS, METHOD_DISPLAY):
                writer.writerow([ratio, display, f"{test_accs[(ratio, method)]:.6f}", source_paths[(ratio, method)]])
        writer.writerow([])
        writer.writerow(["ratio", "slice", *METHOD_DISPLAY])
        for ratio in RATIOS:
            for row_key, values in zip(row_order, matrices[ratio]):
                writer.writerow([ratio, row_key, *[f"{value:.6f}" for value in values]])
    return path


def plot_heatmap(ratio: str, matrix: np.ndarray, row_order: list[str]) -> list[Path]:
    apply_publication_style(FigureStyle(font_size=8, axes_linewidth=1.2))
    fig, ax = plt.subplots(figsize=(7.35, 4.9))
    y_labels = [display_label(row) for row in row_order]
    vmax = 0.36
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    make_heatmap(
        ax,
        matrix,
        x_labels=METHOD_LABELS,
        y_labels=y_labels,
        cmap="RdBu_r",
        cbar_label="Accuracy delta",
        annotate=True,
        norm=norm,
    )

    ax.set_xlabel("Selection method", labelpad=4)
    ax.set_ylabel("Attribute slice", labelpad=6)
    ax.set_title(f"ImageNet ratio {ratio}: attribute-slice accuracy delta", loc="left", fontsize=9.5, fontweight="bold")

    for label in ax.get_xticklabels():
        if label.get_text() == "Ours":
            label.set_fontweight("bold")
            label.set_color(PALETTE["blue_main"])
    for label in ax.get_yticklabels():
        if label.get_text() == "color: green":
            label.set_fontweight("bold")
            label.set_color(PALETTE["blue_main"])

    attr_col = METHODS.index("Attribute")
    for text in ax.texts:
        x, _y = text.get_position()
        if int(round(x)) == attr_col:
            text.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(len(METHODS) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(row_order) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0)
    ax.axvline(attr_col - 0.5, color=PALETTE["neutral_dark"], linewidth=1.0)

    fig.tight_layout(pad=0.45)
    written: list[Path] = []
    stem = f"imagenet_ratio{ratio_tag(ratio)}_attribute_slice_delta_heatmap"
    written.extend(finalize_figure(fig, OUT_DIR / stem, formats=("png", "pdf"), dpi=300))
    written.extend(finalize_figure(fig, PAPER_OUT_DIR / stem, formats=("png", "pdf"), dpi=300))
    plt.close(fig)
    return written


def main() -> None:
    matrices, row_order, test_accs, source_paths = build_data()
    written: list[Path] = []
    csvs: list[Path] = []
    for ratio in RATIOS:
        csvs.append(write_ratio_csv(ratio, matrices[ratio], row_order, test_accs, source_paths))
        written.extend(plot_heatmap(ratio, matrices[ratio], row_order))
    csvs.append(write_summary_csv(matrices, row_order, test_accs, source_paths))
    for path in csvs + written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
