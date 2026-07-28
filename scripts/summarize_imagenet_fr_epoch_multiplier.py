#!/usr/bin/env python3
"""Validate and summarize the ImageNet10 single-seed FR epoch study."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


METHOD_ORDER = ("random", "entropy", "coreset", "badge", "attribute")
METHOD_NAMES = {
    "random": "Random",
    "entropy": "Entropy",
    "coreset": "CoreSet",
    "badge": "BADGE",
    "attribute": "AttriSelect",
}
EXPECTED_MILESTONES = (15, 30, 45, 60)
DEFAULT_BEFORE_ACCURACY = 0.7173


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Pilot artifact root containing task directories.")
    parser.add_argument("--output-json", help="Optional validated summary JSON path.")
    parser.add_argument("--output-md", help="Optional Markdown summary path.")
    parser.add_argument("--before-accuracy", type=float, default=DEFAULT_BEFORE_ACCURACY)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def compute_fix_rate(after_accuracy: float, before_accuracy: float) -> float:
    if not 0.0 <= before_accuracy < 1.0:
        raise ValueError("Before accuracy must be in [0, 1).")
    return 100.0 * (after_accuracy - before_accuracy) / (1.0 - before_accuracy)


def selected_generated_hash(manifest: Mapping[str, Any]) -> str:
    rows = [
        (str(row["path"]), int(row["label"]))
        for row in manifest.get("selected_generated", [])
    ]
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def metric_accuracy(metrics: Mapping[str, Any], key: str) -> float:
    value = (metrics.get(key) or {}).get("accuracy")
    if value is None:
        raise ValueError(f"Missing {key} accuracy")
    return float(value)


def collect_records(root: Path, before_accuracy: float) -> List[Dict[str, Any]]:
    result_paths = sorted(
        Path(path) for path in glob.glob(str(root / "**" / "*_result.json"), recursive=True)
    )
    records: List[Dict[str, Any]] = []
    shared_configuration = None
    generated_hash = None

    for result_path in result_paths:
        result = load_json(result_path)
        strategy = str(result.get("strategy"))
        if strategy not in METHOD_ORDER:
            continue

        manifest_paths = list(result_path.parent.glob("*_selection.json"))
        if len(manifest_paths) != 1:
            raise ValueError(
                f"Expected one selection manifest next to {result_path}, got {manifest_paths}"
            )
        manifest_path = manifest_paths[0]
        manifest = load_json(manifest_path)

        configuration = {
            "source_ratio": float(result.get("source_ratio")),
            "source_seed": int(result.get("source_seed")),
            "generated_seed": int(result.get("generated_seed")),
            "num_generated": int(result.get("num_generated")),
            "num_epochs": int(result.get("num_epochs")),
            "learning_rate": float(result.get("learning_rate")),
            "batch_size": int(result.get("batch_size")),
            "resume_model": str(result.get("resume_model")),
            "attribute_file": str(result.get("attribute_file")),
            "generated_image_dir": str(result.get("generated_image_dir")),
        }
        expected_configuration = {
            "source_ratio": 0.9,
            "source_seed": 42,
            "generated_seed": 42,
            "num_generated": 1000,
            "num_epochs": 60,
            "learning_rate": 1e-5,
            "batch_size": 64,
        }
        for key, expected_value in expected_configuration.items():
            if configuration[key] != expected_value:
                raise ValueError(
                    f"Unexpected {key}={configuration[key]!r} in {result_path}; "
                    f"expected {expected_value!r}"
                )
        if shared_configuration is None:
            shared_configuration = configuration
        else:
            for key in ("resume_model", "attribute_file", "generated_image_dir"):
                if configuration[key] != shared_configuration[key]:
                    raise ValueError(f"Shared configuration mismatch for {key} in {result_path}")

        if int(manifest.get("selected_source_count", -1)) != 9360:
            raise ValueError(f"Source budget mismatch in {manifest_path}")
        if int(manifest.get("selected_generated_count", -1)) != 1000:
            raise ValueError(f"Generated budget mismatch in {manifest_path}")
        current_generated_hash = selected_generated_hash(manifest)
        if generated_hash is None:
            generated_hash = current_generated_hash
        elif current_generated_hash != generated_hash:
            raise ValueError(f"Generated set differs in {manifest_path}")

        milestone_results = result.get("milestone_results") or []
        actual_milestones = tuple(int(row["epoch"]) for row in milestone_results)
        if actual_milestones != EXPECTED_MILESTONES:
            raise ValueError(
                f"Milestone mismatch in {result_path}: {actual_milestones}"
            )

        milestone_records = []
        for milestone in milestone_results:
            metrics = milestone.get("test_slice_metrics") or {}
            overall_count = int((metrics.get("overall") or {}).get("count", -1))
            target_count = int((metrics.get("target") or {}).get("count", -1))
            non_target_count = int((metrics.get("non_target") or {}).get("count", -1))
            if (overall_count, target_count, non_target_count) != (1300, 16, 1284):
                raise ValueError(
                    f"Unexpected test slice counts in {result_path} at epoch {milestone['epoch']}"
                )
            test_accuracy = float(milestone["test_accuracy"])
            milestone_records.append(
                {
                    "epoch": int(milestone["epoch"]),
                    "best_epoch": int(milestone["best_epoch"]),
                    "best_validation_accuracy": float(milestone["best_validation_accuracy"]),
                    "test_accuracy": test_accuracy,
                    "absolute_gain": test_accuracy - before_accuracy,
                    "absolute_gain_percentage_points": 100.0 * (test_accuracy - before_accuracy),
                    "fix_rate": compute_fix_rate(test_accuracy, before_accuracy),
                    "target_accuracy": metric_accuracy(metrics, "target"),
                    "target_count": target_count,
                    "non_target_accuracy": metric_accuracy(metrics, "non_target"),
                    "non_target_count": non_target_count,
                    "training_time_seconds": float(milestone["training_time_seconds"]),
                }
            )

        records.append(
            {
                "strategy": strategy,
                "method": METHOD_NAMES[strategy],
                "selection_time_seconds": float(result["selection_time_seconds"]),
                "milestones": milestone_records,
                "result_path": str(result_path),
                "manifest_path": str(manifest_path),
            }
        )

    records.sort(key=lambda row: METHOD_ORDER.index(row["strategy"]))
    return records


def validate_complete(records: Sequence[Mapping[str, Any]]) -> None:
    strategies = [str(record["strategy"]) for record in records]
    if len(strategies) != len(set(strategies)):
        raise ValueError(f"Duplicate method records found: {strategies}")
    if tuple(strategies) != METHOD_ORDER:
        raise ValueError(
            f"Incomplete method grid: expected={METHOD_ORDER}, actual={tuple(strategies)}"
        )


def add_rankings(records: Sequence[Dict[str, Any]]) -> None:
    for milestone_index, epoch in enumerate(EXPECTED_MILESTONES):
        ordered = sorted(
            records,
            key=lambda row: (
                -float(row["milestones"][milestone_index]["test_accuracy"]),
                METHOD_ORDER.index(str(row["strategy"])),
            ),
        )
        previous_accuracy = None
        competition_rank = 0
        for position, record in enumerate(ordered, start=1):
            milestone = record["milestones"][milestone_index]
            if int(milestone["epoch"]) != epoch:
                raise ValueError("Milestone ordering changed while assigning rankings")
            accuracy = float(milestone["test_accuracy"])
            if previous_accuracy is None or accuracy != previous_accuracy:
                competition_rank = position
                previous_accuracy = accuracy
            milestone["rank"] = competition_rank


def markdown_summary(records: Sequence[Mapping[str, Any]], before_accuracy: float) -> str:
    lines = [
        "# ImageNet10 FR Epoch-Multiplier Pilot (Seed 42)",
        "",
        "Single-seed diagnostic at source ratio 0.9. No significance claim is made.",
        "",
        f"Before-repair accuracy used for FR: `{before_accuracy:.4f}`.",
        "",
        "| Method | 15 epochs FR (%) | 30 epochs FR (%) | 45 epochs FR (%) | 60 epochs FR (%) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        values = [float(row["fix_rate"]) for row in record["milestones"]]
        lines.append(
            f"| {record['method']} | {values[0]:.2f} | {values[1]:.2f} | "
            f"{values[2]:.2f} | {values[3]:.2f} |"
        )

    lines.extend(
        [
            "",
            "| Method | Epoch | Best epoch | Test ACC | Gain (pp) | FR (%) | Target ACC (n=16) | Non-target ACC (n=1284) | Rank |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for record in records:
        for row in record["milestones"]:
            lines.append(
                f"| {record['method']} | {row['epoch']} | {row['best_epoch']} | "
                f"{row['test_accuracy']:.4f} | {row['absolute_gain_percentage_points']:.2f} | "
                f"{row['fix_rate']:.2f} | {row['target_accuracy']:.4f} | "
                f"{row['non_target_accuracy']:.4f} | {row['rank']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    records = collect_records(root, args.before_accuracy)
    if args.require_complete:
        validate_complete(records)
    add_rankings(records)

    summary = {
        "dataset": "imagenet10",
        "single_seed_diagnostic": True,
        "source_seed": 42,
        "generated_seed": 42,
        "source_ratio": 0.9,
        "before_accuracy": args.before_accuracy,
        "milestones": list(EXPECTED_MILESTONES),
        "record_count": len(records),
        "records": records,
    }
    if args.output_json:
        write_json_atomic(Path(args.output_json).expanduser(), summary)
    markdown = markdown_summary(records, args.before_accuracy)
    if args.output_md:
        output_md = Path(args.output_md).expanduser()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
