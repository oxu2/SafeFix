#!/usr/bin/env python3
"""Validate independent ImageNet10 repair-epoch runs with rescaled StepLR."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


METHOD_ORDER = ("random", "entropy", "coreset", "badge", "attribute")
METHOD_NAMES = {
    "random": "Random",
    "entropy": "Entropy",
    "coreset": "CoreSet",
    "badge": "BADGE",
    "attribute": "AttriSelect",
}
EPOCH_ORDER = (15, 30, 45, 60)
STEP_SIZE_BY_EPOCH = {15: 3, 30: 6, 45: 9, 60: 12}
DEFAULT_BEFORE_ACCURACY = 0.7173


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Corrected-mode artifact root.")
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    parser.add_argument("--before-accuracy", type=float, default=DEFAULT_BEFORE_ACCURACY)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHOD_ORDER,
        default=list(METHOD_ORDER),
        help="Methods required in the summary grid (default: all methods).",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def compute_fix_rate(after_accuracy: float, before_accuracy: float) -> float:
    if not 0.0 <= before_accuracy < 1.0:
        raise ValueError("Before accuracy must be in [0, 1).")
    return 100.0 * (after_accuracy - before_accuracy) / (1.0 - before_accuracy)


def rows_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    normalized = [(str(row["path"]), int(row["label"])) for row in rows]
    payload = json.dumps(normalized, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
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
    generated_reference = None
    source_hash_by_method: Dict[str, str] = {}
    shared_paths = None

    for result_path in result_paths:
        result = load_json(result_path)
        strategy = str(result.get("strategy"))
        if strategy not in METHOD_ORDER:
            continue
        epochs = int(result.get("num_epochs"))
        if epochs not in EPOCH_ORDER:
            continue

        manifest_paths = list(result_path.parent.glob("*_selection.json"))
        if len(manifest_paths) != 1:
            raise ValueError(f"Expected one selection manifest next to {result_path}")
        manifest_path = manifest_paths[0]
        manifest = load_json(manifest_path)

        expected = {
            "source_ratio": 0.9,
            "source_seed": 42,
            "generated_seed": 42,
            "num_generated": 1000,
            "learning_rate": 1e-5,
            "lr_step_size": STEP_SIZE_BY_EPOCH[epochs],
            "lr_gamma": 0.5,
            "batch_size": 64,
        }
        for key, expected_value in expected.items():
            actual_value = result.get(key)
            if isinstance(expected_value, float):
                actual_value = float(actual_value)
            else:
                actual_value = int(actual_value)
            if actual_value != expected_value:
                raise ValueError(
                    f"Unexpected {key}={actual_value!r} in {result_path}; expected {expected_value!r}"
                )

        paths = (
            str(result.get("resume_model")),
            str(result.get("attribute_file")),
            str(result.get("generated_image_dir")),
        )
        if shared_paths is None:
            shared_paths = paths
        elif paths != shared_paths:
            raise ValueError(f"Shared input path mismatch in {result_path}")

        if int(manifest.get("selected_source_count", -1)) != 9360:
            raise ValueError(f"Source budget mismatch in {manifest_path}")
        if int(manifest.get("selected_generated_count", -1)) != 1000:
            raise ValueError(f"Generated budget mismatch in {manifest_path}")

        generated_hash = rows_hash(manifest.get("selected_generated", []))
        if generated_reference is None:
            generated_reference = generated_hash
        elif generated_hash != generated_reference:
            raise ValueError(f"Generated set differs in {manifest_path}")

        source_hash = rows_hash(manifest.get("selected_source", []))
        reference_source_hash = source_hash_by_method.setdefault(strategy, source_hash)
        if source_hash != reference_source_hash:
            raise ValueError(
                f"Selected source subset changes across epoch budgets for {strategy}: {manifest_path}"
            )

        metrics = result.get("test_slice_metrics") or {}
        counts = tuple(
            int((metrics.get(key) or {}).get("count", -1))
            for key in ("overall", "target", "non_target")
        )
        if counts != (1300, 16, 1284):
            raise ValueError(f"Unexpected test counts in {result_path}: {counts}")
        overall_accuracy = metric_accuracy(metrics, "overall")

        records.append(
            {
                "strategy": strategy,
                "method": METHOD_NAMES[strategy],
                "epochs": epochs,
                "lr_step_size": int(result["lr_step_size"]),
                "lr_gamma": float(result["lr_gamma"]),
                "final_scheduled_lr": float(result["learning_rate"])
                * float(result["lr_gamma"]) ** (epochs // int(result["lr_step_size"])),
                "best_epoch": int(result["best_validation_epoch"]),
                "best_validation_accuracy": float(result["best_validation_accuracy"]),
                "test_accuracy": overall_accuracy,
                "absolute_gain": overall_accuracy - before_accuracy,
                "absolute_gain_percentage_points": 100.0 * (overall_accuracy - before_accuracy),
                "fix_rate": compute_fix_rate(overall_accuracy, before_accuracy),
                "target_accuracy": metric_accuracy(metrics, "target"),
                "non_target_accuracy": metric_accuracy(metrics, "non_target"),
                "selection_time_seconds": float(result["selection_time_seconds"]),
                "training_time_seconds": float(result["training_time_seconds"]),
                "result_path": str(result_path),
                "manifest_path": str(manifest_path),
            }
        )

    records.sort(key=lambda row: (METHOD_ORDER.index(row["strategy"]), EPOCH_ORDER.index(row["epochs"])))
    return records


def validate_complete(
    records: Sequence[Mapping[str, Any]], methods: Sequence[str] = METHOD_ORDER
) -> None:
    actual = [(str(row["strategy"]), int(row["epochs"])) for row in records]
    if len(actual) != len(set(actual)):
        raise ValueError("Duplicate method/epoch result records found")
    expected = [(method, epochs) for method in methods for epochs in EPOCH_ORDER]
    if actual != expected:
        raise ValueError(f"Incomplete corrected grid: expected={expected}, actual={actual}")


def add_rankings(records: Sequence[Dict[str, Any]]) -> None:
    for epochs in EPOCH_ORDER:
        epoch_records = [record for record in records if int(record["epochs"]) == epochs]
        ordered = sorted(
            epoch_records,
            key=lambda row: (-float(row["test_accuracy"]), METHOD_ORDER.index(str(row["strategy"]))),
        )
        previous_accuracy = None
        rank = 0
        for position, record in enumerate(ordered, start=1):
            accuracy = float(record["test_accuracy"])
            if previous_accuracy is None or accuracy != previous_accuracy:
                rank = position
                previous_accuracy = accuracy
            record["rank"] = rank


def markdown_summary(
    records: Sequence[Mapping[str, Any]],
    before_accuracy: float,
    methods: Sequence[str] = METHOD_ORDER,
) -> str:
    by_key = {(row["strategy"], row["epochs"]): row for row in records}
    lines = [
        "# ImageNet10 FR Epoch-Multiplier: Rescaled StepLR (Seed 42)",
        "",
        "Independent single-seed runs at source ratio 0.9. No significance claim is made.",
        "",
        "Schedules: `15/3`, `30/6`, `45/9`, and `60/12` for total epochs / StepLR interval; gamma is `0.5`.",
        f"Before-repair accuracy used for FR: `{before_accuracy:.4f}`.",
        "",
        "| Method | 15 epochs FR (%) | 30 epochs FR (%) | 45 epochs FR (%) | 60 epochs FR (%) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for strategy in methods:
        values = [float(by_key[(strategy, epochs)]["fix_rate"]) for epochs in EPOCH_ORDER]
        lines.append(
            f"| {METHOD_NAMES[strategy]} | {values[0]:.2f} | {values[1]:.2f} | {values[2]:.2f} | {values[3]:.2f} |"
        )
    lines.extend(
        [
            "",
            "| Method | Epochs | StepLR interval | Best epoch | Test ACC | Gain (pp) | FR (%) | Target ACC | Non-target ACC | Rank |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in records:
        lines.append(
            f"| {row['method']} | {row['epochs']} | {row['lr_step_size']} | {row['best_epoch']} | "
            f"{row['test_accuracy']:.4f} | {row['absolute_gain_percentage_points']:.2f} | "
            f"{row['fix_rate']:.2f} | {row['target_accuracy']:.4f} | "
            f"{row['non_target_accuracy']:.4f} | {row['rank']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    records = collect_records(root, args.before_accuracy)
    selected_methods = tuple(args.methods)
    records = [record for record in records if record["strategy"] in selected_methods]
    if args.require_complete:
        validate_complete(records, selected_methods)
    add_rankings(records)
    summary = {
        "dataset": "imagenet10",
        "single_seed_diagnostic": True,
        "source_seed": 42,
        "generated_seed": 42,
        "source_ratio": 0.9,
        "before_accuracy": args.before_accuracy,
        "epoch_order": list(EPOCH_ORDER),
        "step_size_by_epoch": {str(key): value for key, value in STEP_SIZE_BY_EPOCH.items()},
        "record_count": len(records),
        "records": records,
    }
    if args.output_json:
        write_json_atomic(Path(args.output_json).expanduser(), summary)
    markdown = markdown_summary(records, args.before_accuracy, selected_methods)
    if args.output_md:
        output_md = Path(args.output_md).expanduser()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
