#!/usr/bin/env python3
"""Summarize ImageNet10 one-shot selection and active-learning JSON results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_results(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    rows = []
    for path in sorted(root.rglob("*_result.json")):
        with path.open("r", encoding="utf-8") as handle:
            rows.append((path, json.load(handle)))
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.result_root
    data_rows: list[dict[str, Any]] = []
    al_rows: list[dict[str, Any]] = []
    for path, result in load_results(args.result_root):
        if result.get("selection_only"):
            continue
        common = {
            "method": result["method"],
            "seed": result["seed"],
            "result_json": str(path),
        }
        if result["experiment"] == "data_selection":
            data_rows.append(
                {
                    **common,
                    "ratio": result["ratio"],
                    "selected_count": result["selection"]["selected_count"],
                    "unique_class_attribute_tuples": result["selection"][
                        "unique_class_attribute_tuples"
                    ],
                    "test_accuracy": result["metrics"]["test_accuracy"],
                    "best_val_accuracy": result["metrics"]["best_val_accuracy"],
                    "macro_attribute_accuracy": result["attribute_test_metrics"][
                        "macro_attribute_accuracy_min_support"
                    ],
                    "selection_seconds": result["selection"]["selection_seconds"],
                    "training_seconds": result["metrics"]["training_seconds"],
                }
            )
        else:
            for round_result in result["round_results"]:
                al_rows.append(
                    {
                        **common,
                        "round": round_result["round"],
                        "labeled_ratio": round_result["labeled_ratio"],
                        "labeled_count": round_result["labeled_count"],
                        "unique_class_attribute_tuples": round_result["labeled_diagnostics"][
                            "unique_class_attribute_tuples"
                        ],
                        "test_accuracy": round_result["metrics"]["test_accuracy"],
                        "best_val_accuracy": round_result["metrics"]["best_val_accuracy"],
                        "macro_attribute_accuracy": round_result["attribute_test_metrics"][
                            "macro_attribute_accuracy_min_support"
                        ],
                        "training_seconds": round_result["metrics"]["training_seconds"],
                    }
                )

    data_rows.sort(key=lambda row: (row["ratio"], row["method"], row["seed"]))
    al_rows.sort(key=lambda row: (row["round"], row["method"], row["seed"]))
    data_fields = [
        "method",
        "ratio",
        "seed",
        "selected_count",
        "unique_class_attribute_tuples",
        "test_accuracy",
        "best_val_accuracy",
        "macro_attribute_accuracy",
        "selection_seconds",
        "training_seconds",
        "result_json",
    ]
    al_fields = [
        "method",
        "round",
        "labeled_ratio",
        "seed",
        "labeled_count",
        "unique_class_attribute_tuples",
        "test_accuracy",
        "best_val_accuracy",
        "macro_attribute_accuracy",
        "training_seconds",
        "result_json",
    ]
    write_csv(output_dir / "data_selection_summary.csv", data_fields, data_rows)
    write_csv(output_dir / "active_learning_summary.csv", al_fields, al_rows)
    print(f"data_selection_rows={len(data_rows)}")
    print(f"active_learning_rows={len(al_rows)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
