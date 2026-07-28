#!/usr/bin/env python3
"""Validate and aggregate the five-seed ImageNet10 attribute-aware baselines."""

from __future__ import annotations

import argparse
from collections import Counter
import glob
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
import sys
from typing import Any, Dict, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.attribute_aware_baselines import build_attribute_match_pools


METHOD_ORDER = (
    "target_slice_stratified",
    "attribute_filtered_random",
    "attribute_filtered_entropy",
)
METHOD_NAMES = {
    "target_slice_stratified": "Target-Slice + Stratified",
    "attribute_filtered_random": "Attribute-Filtered Random",
    "attribute_filtered_entropy": "Attribute-Filtered Entropy",
}
RATIO_ORDER = (0.1, 0.3, 0.5, 0.7, 0.9)
SEED_ORDER = (0, 42, 1234, 3407, 2024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="+",
        help="Artifact roots containing the seed-42 full run and remaining-seed run.",
    )
    parser.add_argument("--attribute-file", required=True, help="ImageNet10 attribute metadata JSON.")
    parser.add_argument("--output-json", help="Optional validated summary JSON path.")
    parser.add_argument("--output-md", help="Optional Markdown aggregate table path.")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Require three methods x five ratios x five seeds (75 records).",
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


def metric_accuracy(metrics: Mapping[str, Any], key: str) -> float:
    value = (metrics.get(key) or {}).get("accuracy")
    if value is None:
        raise ValueError(f"Missing {key} accuracy in test_slice_metrics")
    return float(value)


def collect_records(
    roots: Sequence[Path],
    attribute_metadata: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], str]:
    result_paths: List[Path] = []
    for root in roots:
        result_paths.extend(
            Path(path) for path in glob.glob(str(root / "**" / "*_result.json"), recursive=True)
        )
    result_paths = sorted(set(result_paths))

    records: List[Dict[str, Any]] = []
    generated_reference = None
    generated_hash_payload = ""

    for result_path in result_paths:
        result = load_json(result_path)
        strategy = str(result.get("strategy"))
        if strategy not in METHOD_ORDER:
            continue
        ratio = float(result.get("source_ratio"))
        seed = int(result.get("source_seed"))
        if ratio not in RATIO_ORDER:
            raise ValueError(f"Unexpected ratio {ratio} in {result_path}")
        if seed not in SEED_ORDER:
            raise ValueError(f"Unexpected seed {seed} in {result_path}")

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
        if int(manifest.get("source_seed", -1)) != seed:
            raise ValueError(f"Source seed mismatch in {manifest_path}")

        generated_rows = tuple(
            (str(row["path"]), int(row["label"]))
            for row in manifest["selected_generated"]
        )
        if generated_reference is None:
            generated_reference = generated_rows
            generated_hash_payload = json.dumps(generated_rows, separators=(",", ":"))
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
                "overall_accuracy": metric_accuracy(metrics, "overall"),
                "target_accuracy": metric_accuracy(metrics, "target"),
                "target_count": int(metrics["target"]["count"]),
                "non_target_accuracy": metric_accuracy(metrics, "non_target"),
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
            SEED_ORDER.index(row["seed"]),
        )
    )
    generated_hash = hashlib.sha256(generated_hash_payload.encode("utf-8")).hexdigest()
    return records, generated_hash


def validate_complete(records: Sequence[Mapping[str, Any]]) -> None:
    keys = [(row["strategy"], row["ratio"], row["seed"]) for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate strategy/ratio/seed result records found")
    expected = {
        (method, ratio, seed)
        for method in METHOD_ORDER
        for ratio in RATIO_ORDER
        for seed in SEED_ORDER
    }
    actual = set(keys)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"Incomplete result grid; missing={missing}, extra={extra}")


def aggregate_records(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    aggregates: List[Dict[str, Any]] = []
    metrics = (
        "overall_accuracy",
        "target_accuracy",
        "non_target_accuracy",
        "source_exact_target_count",
        "source_partial_target_count",
        "selection_time_seconds",
        "training_time_seconds",
    )
    for ratio in RATIO_ORDER:
        for strategy in METHOD_ORDER:
            rows = [
                row for row in records
                if row["ratio"] == ratio and row["strategy"] == strategy
            ]
            if not rows:
                continue
            aggregate: Dict[str, Any] = {
                "strategy": strategy,
                "method": METHOD_NAMES[strategy],
                "ratio": ratio,
                "seeds": [int(row["seed"]) for row in rows],
                "seed_count": len(rows),
                "source_count": int(rows[0]["source_count"]),
                "generated_count": int(rows[0]["generated_count"]),
                "target_count": int(rows[0]["target_count"]),
                "non_target_count": int(rows[0]["non_target_count"]),
            }
            for metric in metrics:
                values = [float(row[metric]) for row in rows]
                aggregate[f"{metric}_mean"] = mean(values)
                aggregate[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
            aggregates.append(aggregate)
    return aggregates


def markdown_table(aggregates: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Ratio | Method | Overall ACC | Target ACC | Non-target ACC | Exact source target | Selection (s) | Training (s) |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregates:
        lines.append(
            "| {ratio:.1f} | {method} | "
            "{overall_accuracy_mean:.4f} $\\pm$ {overall_accuracy_std:.4f} | "
            "{target_accuracy_mean:.4f} $\\pm$ {target_accuracy_std:.4f} | "
            "{non_target_accuracy_mean:.4f} $\\pm$ {non_target_accuracy_std:.4f} | "
            "{source_exact_target_count_mean:.1f} $\\pm$ {source_exact_target_count_std:.1f}/{source_count} | "
            "{selection_time_seconds_mean:.2f} | {training_time_seconds_mean:.2f} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    roots = [Path(root).expanduser().resolve() for root in args.roots]
    attribute_metadata = load_json(Path(args.attribute_file).expanduser().resolve())
    records, generated_hash = collect_records(roots, attribute_metadata)
    if args.require_complete:
        validate_complete(records)
    aggregates = aggregate_records(records)

    summary = {
        "roots": [str(root) for root in roots],
        "seeds": list(SEED_ORDER),
        "generated_seed": 42,
        "generated_set_sha256": generated_hash,
        "record_count": len(records),
        "records": records,
        "aggregates": aggregates,
    }
    if args.output_json:
        write_json_atomic(Path(args.output_json).expanduser(), summary)
    table = markdown_table(aggregates)
    if args.output_md:
        output_md = Path(args.output_md).expanduser()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(table, encoding="utf-8")
    print(table, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
