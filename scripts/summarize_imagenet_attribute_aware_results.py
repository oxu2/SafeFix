#!/usr/bin/env python3
"""Validate and summarize ImageNet10 same-information baseline results."""

from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.attribute_aware_baselines import build_attribute_match_pools


METHOD_ORDER = (
    "target_slice_stratified",
    "attribute_filtered_random",
    "attribute_filtered_entropy",
    "attribute",
)
METHOD_NAMES = {
    "target_slice_stratified": "Target-Slice + Stratified",
    "attribute_filtered_random": "Attribute-Filtered Random",
    "attribute_filtered_entropy": "Attribute-Filtered Entropy",
    "attribute": "AttriSelect",
}
RATIO_ORDER = (0.1, 0.3, 0.5, 0.7, 0.9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Full-mode artifact root containing array task directories.")
    parser.add_argument("--attribute-file", required=True, help="ImageNet10 attribute metadata JSON.")
    parser.add_argument("--output-json", help="Optional validated summary JSON path.")
    parser.add_argument("--output-md", help="Optional Markdown table path.")
    parser.add_argument("--seed", type=int, default=42, help="Expected source seed.")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Require all four methods at all five source ratios.",
    )
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


def accuracy(metrics: Mapping[str, Any], key: str) -> float:
    value = (metrics.get(key) or {}).get("accuracy")
    if value is None:
        raise ValueError(f"Missing {key} accuracy in test_slice_metrics")
    return float(value)


def collect_records(
    root: Path,
    attribute_metadata: Mapping[str, Mapping[str, Any]],
    expected_seed: int,
) -> List[Dict[str, Any]]:
    result_paths = sorted(
        Path(path) for path in glob.glob(str(root / "**" / "*_result.json"), recursive=True)
    )
    records: List[Dict[str, Any]] = []
    generated_reference = None

    for result_path in result_paths:
        result = load_json(result_path)
        strategy = str(result.get("strategy"))
        ratio = float(result.get("source_ratio"))
        seed = int(result.get("source_seed"))
        if strategy not in METHOD_ORDER:
            continue
        if seed != expected_seed:
            raise ValueError(f"Unexpected seed {seed} in {result_path}; expected {expected_seed}")

        manifest_paths = list(result_path.parent.glob("*_selection.json"))
        if len(manifest_paths) != 1:
            raise ValueError(
                f"Expected exactly one selection manifest next to {result_path}, got {manifest_paths}"
            )
        manifest_path = manifest_paths[0]
        manifest = load_json(manifest_path)
        expected_source = int(10400 * ratio)
        if int(manifest.get("selected_source_count", -1)) != expected_source:
            raise ValueError(f"Source budget mismatch in {manifest_path}")
        if int(manifest.get("selected_generated_count", -1)) != 1000:
            raise ValueError(f"Generated budget mismatch in {manifest_path}")
        if int(manifest.get("generated_seed", -1)) != 42:
            raise ValueError(f"Generated seed mismatch in {manifest_path}")

        generated_rows = tuple(
            (str(row["path"]), int(row["label"]))
            for row in manifest["selected_generated"]
        )
        if generated_reference is None:
            generated_reference = generated_rows
        elif generated_rows != generated_reference:
            raise ValueError(f"Generated set differs in {manifest_path}")

        metrics = result.get("test_slice_metrics") or {}
        target_values = metrics.get("target_values") or {}
        source_rows = manifest["selected_source"]
        metadata_indices = [int(row["dataset_index"]) for row in source_rows]
        exact_indices, partial_indices, _ = build_attribute_match_pools(
            attribute_metadata=attribute_metadata,
            metadata_indices=metadata_indices,
            target_values=target_values,
        )
        class_counts = Counter(int(row["label"]) for row in source_rows)

        records.append(
            {
                "strategy": strategy,
                "method": METHOD_NAMES[strategy],
                "ratio": ratio,
                "seed": seed,
                "source_count": len(source_rows),
                "generated_count": len(generated_rows),
                "source_exact_target_count": len(exact_indices),
                "source_partial_target_count": len(partial_indices),
                "source_class_counts": {
                    str(label): count for label, count in sorted(class_counts.items())
                },
                "overall_accuracy": accuracy(metrics, "overall"),
                "target_accuracy": accuracy(metrics, "target"),
                "target_count": int(metrics["target"]["count"]),
                "non_target_accuracy": accuracy(metrics, "non_target"),
                "non_target_count": int(metrics["non_target"]["count"]),
                "selection_time_seconds": float(result["selection_time_seconds"]),
                "training_time_seconds": float(result["training_time_seconds"]),
                "wall_time_seconds": float(result["wall_time_seconds"]),
                "result_path": str(result_path),
                "manifest_path": str(manifest_path),
            }
        )

    records.sort(
        key=lambda row: (
            RATIO_ORDER.index(row["ratio"]),
            METHOD_ORDER.index(row["strategy"]),
        )
    )
    return records


def validate_complete(records: Sequence[Mapping[str, Any]]) -> None:
    keys = [(row["strategy"], row["ratio"]) for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate strategy/ratio result records found")
    expected = {(method, ratio) for method in METHOD_ORDER for ratio in RATIO_ORDER}
    actual = set(keys)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"Incomplete result grid; missing={missing}, extra={extra}")


def markdown_table(records: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Ratio | Method | Overall ACC | Target ACC | Non-target ACC | Source target | Selection (s) | Training (s) |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in records:
        lines.append(
            "| {ratio:.1f} | {method} | {overall_accuracy:.4f} | "
            "{target_accuracy:.4f} | {non_target_accuracy:.4f} | "
            "{source_exact_target_count}/{source_count} | "
            "{selection_time_seconds:.2f} | {training_time_seconds:.2f} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    attribute_metadata = load_json(Path(args.attribute_file).expanduser().resolve())
    records = collect_records(root, attribute_metadata, args.seed)
    if args.require_complete:
        validate_complete(records)

    summary = {
        "root": str(root),
        "seed": args.seed,
        "generated_seed": 42,
        "record_count": len(records),
        "records": records,
    }
    if args.output_json:
        write_json_atomic(Path(args.output_json).expanduser(), summary)
    table = markdown_table(records)
    if args.output_md:
        output_md = Path(args.output_md).expanduser()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(table, encoding="utf-8")
    print(table, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
