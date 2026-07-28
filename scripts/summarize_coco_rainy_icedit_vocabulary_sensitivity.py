#!/usr/bin/env python3
"""Validate and summarize the COCO rain-only ICEdit ratio sweep."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


RATIOS = (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
ERROR_PATTERNS = (
    "Traceback",
    "CUDA out of memory",
    "OutOfMemory",
    "Post-train slice analysis failed",
    "Required path not found",
    "No readable generated image",
)
SAMPLING_PATTERN = re.compile(
    r"Train sampling summary: original_candidates=(?P<candidates>\d+), "
    r"selected_original=(?P<selected_original>\d+) .*?"
    r"generated_available=(?P<generated_available>\d+), "
    r"generated_selected=(?P<generated_selected>\d+), "
    r"final_train=(?P<final_train>\d+)"
)


def ratio_tag(ratio: float) -> str:
    return f"{ratio:.1f}".replace(".", "p")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_rainy_slice(summary: Dict[str, Any], path: Path) -> Dict[str, Any]:
    matches = [
        row
        for row in summary.get("slices", [])
        if row.get("attribute") == "weather" and row.get("value") == "rainy"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one weather=rainy slice in {path}; found {len(matches)}")
    return matches[0]


def validate_ratio(root: Path, ratio: float) -> Dict[str, Any]:
    run_dir = root / f"ratio_{ratio_tag(ratio)}"
    model_dir = run_dir / "model"
    log_path = run_dir / "train.log"
    required = (
        log_path,
        model_dir / "history.json",
        model_dir / "val_metrics_best.json",
        model_dir / "test_metrics.json",
        model_dir / "slice_analysis/qwen3/val_slice_stats.json",
        model_dir / "slice_analysis/qwen3/test_slice_stats.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required files for ratio {ratio}: {missing}")

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    found_errors = [pattern for pattern in ERROR_PATTERNS if pattern in log_text]
    if found_errors:
        raise ValueError(f"Error markers for ratio {ratio}: {found_errors}")
    if "Training complete. Artifacts:" not in log_text:
        raise ValueError(f"Missing completion marker for ratio {ratio}")

    sampling_match = SAMPLING_PATTERN.search(log_text)
    if sampling_match is None:
        raise ValueError(f"Missing sampling summary for ratio {ratio}")
    counts = {key: int(value) for key, value in sampling_match.groupdict().items()}
    expected_original = int(round(20_000 * ratio))
    expected_counts = {
        "candidates": 20_000,
        "selected_original": expected_original,
        "generated_available": 1_314,
        "generated_selected": 1_000,
        "final_train": expected_original + 1_000,
    }
    if counts != expected_counts:
        raise ValueError(
            f"Unexpected sampling counts for ratio {ratio}: {counts}; expected {expected_counts}"
        )

    history = read_json(model_dir / "history.json")
    if len(history) != 1 or int(history[0].get("epoch", -1)) != 1:
        raise ValueError(f"Unexpected history for ratio {ratio}: {history}")

    val_metrics = read_json(model_dir / "val_metrics_best.json").get("metrics", {})
    test_metrics = read_json(model_dir / "test_metrics.json").get("metrics", {})
    val_slice_path = model_dir / "slice_analysis/qwen3/val_slice_stats.json"
    test_slice_path = model_dir / "slice_analysis/qwen3/test_slice_stats.json"
    val_summary = read_json(val_slice_path)
    test_summary = read_json(test_slice_path)

    for split, summary, metrics, expected_images in (
        ("val", val_summary, val_metrics, 1_173),
        ("test", test_summary, test_metrics, 1_173),
    ):
        if summary.get("attributes") != ["weather"]:
            raise ValueError(f"Unexpected {split} attributes for ratio {ratio}")
        overall = summary.get("overall", {})
        if int(overall.get("num_images", -1)) != expected_images:
            raise ValueError(f"Unexpected {split} image count for ratio {ratio}: {overall}")
        if abs(float(overall.get("AP", -1.0)) - float(metrics.get("AP", -2.0))) > 1e-12:
            raise ValueError(f"{split} overall AP mismatch for ratio {ratio}")

    val_rainy = find_rainy_slice(val_summary, val_slice_path)
    test_rainy = find_rainy_slice(test_summary, test_slice_path)
    if int(val_rainy.get("num_images", -1)) != 26:
        raise ValueError(f"Unexpected rainy val count for ratio {ratio}: {val_rainy}")
    if int(test_rainy.get("num_images", -1)) != 31:
        raise ValueError(f"Unexpected rainy test count for ratio {ratio}: {test_rainy}")

    return {
        "ratio": ratio,
        "counts": counts,
        "train_loss": float(history[0]["train_loss"]),
        "epoch_time_sec": float(history[0]["epoch_time_sec"]),
        "val": val_metrics,
        "test": test_metrics,
        "rainy_val": val_rainy,
        "rainy_test": test_rainy,
    }


def render_markdown(records: List[Dict[str, Any]], source_root: Path) -> str:
    lines = [
        "# COCO Rain-Only ICEdit Vocabulary-Sensitivity Summary",
        "",
        "Single-seed (`42`) diagnostic using Qwen3 train20k metadata, 1,000 "
        "rain-only ICEdit images, and weather-only AttriSelect scoring below "
        "ratio `1.0`; ratio `1.0` uses the full source set.",
        "",
        "| Ratio | Source | Generated | Val AP | Test AP | Rainy val AP (`n=26`) | Rainy test AP (`n=31`) | Rainy test delta |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        lines.append(
            "| `{ratio:.1f}` | `{source}` | `{generated}` | `{val:.4f}` | `{test:.4f}` | "
            "`{rainy_val:.4f}` | `{rainy_test:.4f}` | `{delta:+.4f}` |".format(
                ratio=record["ratio"],
                source=record["counts"]["selected_original"],
                generated=record["counts"]["generated_selected"],
                val=record["val"]["AP"],
                test=record["test"]["AP"],
                rainy_val=record["rainy_val"]["AP"],
                rainy_test=record["rainy_test"]["AP"],
                delta=record["rainy_test"]["ap_delta"],
            )
        )

    best_overall = max(records, key=lambda row: row["test"]["AP"])
    best_rainy = max(records, key=lambda row: row["rainy_test"]["AP"])
    lines.extend(
        [
            "",
            f"- Best overall test AP: ratio `{best_overall['ratio']:.1f}`, "
            f"`{best_overall['test']['AP']:.4f}`.",
            f"- Best rainy-slice test AP: ratio `{best_rainy['ratio']:.1f}`, "
            f"`{best_rainy['rainy_test']['AP']:.4f}`.",
            "- The rainy test slice has only 31 images, so its AP is a small-slice diagnostic.",
            "- This run changes both the selection vocabulary and the generated repair pool; it does not isolate vocabulary coverage alone.",
            f"- Validated source root: `{source_root}`.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Job artifact root containing ratio_* directories")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [validate_ratio(args.root, ratio) for ratio in RATIOS]
    payload = {
        "job_id": "308170",
        "seed": 42,
        "generator": "ICEdit",
        "generated_attribute": "rainy",
        "selection_attributes": ["weather"],
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(records, args.root), encoding="utf-8")
    print(f"Validated {len(records)} ratios")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
