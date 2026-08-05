#!/usr/bin/env python3
"""Analyze seed-42 ImageNet10 same-information selection manifests.

The analysis is selection-only. It combines the matched array-306871
manifests with the seed-42 Generated-Complement manifests and reports
composition, categorical diversity, balance, generated-target redundancy,
subset overlap, and the downstream accuracy saved by the original runs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ATTRIBUTES = ("color", "background", "texture")
TARGET_VALUES = {"color": "pink", "texture": "fabric"}
RATIOS = (0.1, 0.3, 0.5, 0.7, 0.9)
CONSTRAINED_RATIOS = (0.02, 0.05)
METHOD_ORDER = (
    "target_slice_stratified",
    "attribute_filtered_random",
    "attribute_filtered_entropy",
    "generated_complement",
    "attribute",
)
METHOD_NAMES = {
    "target_slice_stratified": "Target-Slice + Stratified",
    "attribute_filtered_random": "Attribute-Filtered Random",
    "attribute_filtered_entropy": "Attribute-Filtered Entropy",
    "generated_complement": "Generated-Complement",
    "attribute": "AttriSelect",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matched-root",
        default="logs/juno/imagenet_attribute_aware/full",
        help="Local array-306871 artifact root.",
    )
    parser.add_argument(
        "--generated-complement-root",
        default="logs/juno/generated_complement_seed42",
        help="Local array-313018 artifact root.",
    )
    parser.add_argument(
        "--constrained-root",
        default="logs/juno/constrained_budget_seed42",
        help="Local ImageNet10 tasks 313516_0--9 for ratios 0.02/0.05.",
    )
    parser.add_argument(
        "--attribute-file",
        default="save/attribute10_qwen3_8B.json",
        help="ImageNet10 source attribute metadata.",
    )
    parser.add_argument(
        "--output-json",
        default="logs/juno/imagenet_seed42_selection_mechanism.json",
    )
    parser.add_argument(
        "--output-md",
        default="logs/juno/imagenet_seed42_selection_mechanism.md",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def normalized(value: Any) -> str:
    return str(value).strip().casefold()


def mismatch_probability(values: Sequence[str]) -> float:
    """Return mean pairwise categorical mismatch over unordered pairs."""
    count = len(values)
    if count < 2:
        return 0.0
    frequencies = Counter(values)
    matching_pairs = sum(value * (value - 1) for value in frequencies.values())
    return 1.0 - matching_pairs / (count * (count - 1))


def generated_redundancy(attribute_values: Mapping[str, Sequence[str]]) -> float:
    """Mean fraction of generated-supported target fields matched per sample."""
    count = len(next(iter(attribute_values.values())))
    if count == 0:
        return 0.0
    total = 0.0
    for index in range(count):
        matches = sum(
            attribute_values[field][index] == target
            for field, target in TARGET_VALUES.items()
        )
        total += matches / len(TARGET_VALUES)
    return total / count


def record_from_pair(
    manifest_path: Path,
    result_path: Path,
    attribute_metadata: Mapping[str, Mapping[str, Sequence[Any]]],
) -> Dict[str, Any]:
    manifest = load_json(manifest_path)
    result = load_json(result_path)
    strategy = str(manifest["strategy"])
    ratio = float(manifest["source_ratio"])
    seed = int(manifest["source_seed"])
    if seed != 42:
        raise ValueError(f"Expected seed 42, found {seed}: {manifest_path}")

    rows = manifest["selected_source"]
    selected_indices = [int(row["dataset_index"]) for row in rows]
    labels = [int(row["label"]) for row in rows]
    source_count = len(selected_indices)
    if source_count != int(manifest["selected_source_count"]):
        raise ValueError(f"Source-count mismatch: {manifest_path}")
    if len(set(selected_indices)) != source_count:
        raise ValueError(f"Duplicate source indices: {manifest_path}")

    attribute_values = {
        field: [
            normalized(attribute_metadata[field]["data"][index])
            for index in selected_indices
        ]
        for field in ATTRIBUTES
    }
    exact_count = sum(
        attribute_values["color"][index] == "pink"
        and attribute_values["texture"][index] == "fabric"
        for index in range(source_count)
    )
    single_count = sum(
        (attribute_values["color"][index] == "pink")
        ^ (attribute_values["texture"][index] == "fabric")
        for index in range(source_count)
    )
    background_count = source_count - exact_count - single_count
    class_counts = Counter(labels)
    per_class_counts = [class_counts.get(label, 0) for label in range(10)]
    unique_tuples = len(
        set(
            zip(
                labels,
                attribute_values["color"],
                attribute_values["background"],
                attribute_values["texture"],
            )
        )
    )
    attribute_distance = mean(
        mismatch_probability(attribute_values[field]) for field in ATTRIBUTES
    )
    test_metrics = result.get("test_slice_metrics") or {}
    overall_accuracy = float((test_metrics.get("overall") or {})["accuracy"])

    return {
        "strategy": strategy,
        "method": METHOD_NAMES[strategy],
        "ratio": ratio,
        "seed": seed,
        "source_count": source_count,
        "selected_indices": selected_indices,
        "exact_target_count": exact_count,
        "exact_target_fraction": exact_count / source_count,
        "single_factor_count": single_count,
        "single_factor_fraction": single_count / source_count,
        "background_count": background_count,
        "background_fraction": background_count / source_count,
        "classes_covered": len(class_counts),
        "class_imbalance": max(per_class_counts) - min(per_class_counts),
        "unique_class_attribute_tuples": unique_tuples,
        "generated_redundancy": generated_redundancy(attribute_values),
        "mean_pairwise_attribute_distance": attribute_distance,
        "overall_accuracy": overall_accuracy,
        "manifest_path": str(manifest_path),
        "result_path": str(result_path),
    }


def matched_pairs(root: Path) -> Iterable[Tuple[Path, Path]]:
    for result_name in glob.glob(str(root / "**" / "*_result.json"), recursive=True):
        result_path = Path(result_name)
        manifests = list(result_path.parent.glob("*_selection.json"))
        if len(manifests) != 1:
            raise ValueError(f"Expected one manifest beside {result_path}: {manifests}")
        yield manifests[0], result_path


def generated_complement_pairs(root: Path) -> Iterable[Tuple[Path, Path]]:
    for task_index in range(5):
        task_dir = root / f"313018_{task_index}"
        yield task_dir / "selection.json", task_dir / "result.json"


def constrained_pairs(root: Path) -> Iterable[Tuple[Path, Path]]:
    for task_index in range(10):
        task_dir = root / f"313516_{task_index}"
        yield task_dir / "selection.json", task_dir / "result.json"


def validate_grid(
    records: Sequence[Mapping[str, Any]], ratios: Sequence[float]
) -> None:
    keys = {(row["strategy"], row["ratio"]) for row in records}
    expected = {(method, ratio) for method in METHOD_ORDER for ratio in ratios}
    if keys != expected or len(records) != len(expected):
        raise ValueError(
            f"Incomplete mechanism grid: missing={sorted(expected - keys)}, "
            f"extra={sorted(keys - expected)}, records={len(records)}"
        )


def add_jaccard(records: List[Dict[str, Any]], ratios: Sequence[float]) -> None:
    for ratio in ratios:
        ratio_records = [row for row in records if row["ratio"] == ratio]
        attriselect = next(row for row in ratio_records if row["strategy"] == "attribute")
        reference = set(attriselect["selected_indices"])
        for row in ratio_records:
            selected = set(row["selected_indices"])
            row["jaccard_with_attriselect"] = len(selected & reference) / len(
                selected | reference
            )


def format_count_fraction(count: int, fraction: float) -> str:
    return f"{count} ({100.0 * fraction:.2f}%)"


def format_ratio(ratio: float) -> str:
    return f"{ratio:.2f}" if ratio < 0.1 else f"{ratio:.1f}"


def composition_table(records: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Ratio | Method | Exact Target # (%) | Single-Factor # (%) | Background # (%) | Unique Class--Attribute Tuples | Classes Covered | Overall ACC |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            "| {ratio} | {method} | {exact} | {single} | {background} | "
            "{tuples} | {classes} | {accuracy:.4f} |".format(
                ratio=format_ratio(row["ratio"]),
                method=row["method"],
                exact=format_count_fraction(
                    row["exact_target_count"], row["exact_target_fraction"]
                ),
                single=format_count_fraction(
                    row["single_factor_count"], row["single_factor_fraction"]
                ),
                background=format_count_fraction(
                    row["background_count"], row["background_fraction"]
                ),
                tuples=row["unique_class_attribute_tuples"],
                classes=row["classes_covered"],
                accuracy=row["overall_accuracy"],
            )
        )
    return "\n".join(lines)


def mechanism_table(records: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Ratio | Method | Class Imbalance $\\downarrow$ | Generated Redundancy $\\downarrow$ | Jaccard with AttriSelect | Mean Pairwise Attribute Distance $\\uparrow$ | Overall ACC |",
        "|---:|:---|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            "| {ratio} | {method} | {imbalance:d} | {redundancy:.4f} | "
            "{jaccard:.4f} | {distance:.4f} | {accuracy:.4f} |".format(
                ratio=format_ratio(row["ratio"]),
                method=row["method"],
                imbalance=row["class_imbalance"],
                redundancy=row["generated_redundancy"],
                jaccard=row["jaccard_with_attriselect"],
                distance=row["mean_pairwise_attribute_distance"],
                accuracy=row["overall_accuracy"],
            )
        )
    return "\n".join(lines)


def low_budget_summary(
    records: Sequence[Mapping[str, Any]], ratios: Sequence[float]
) -> List[Dict[str, Any]]:
    summaries = []
    for ratio in ratios:
        rows = [row for row in records if row["ratio"] == ratio]
        attriselect = next(row for row in rows if row["strategy"] == "attribute")
        controls = [row for row in rows if row["strategy"] != "attribute"]
        best_accuracy = max(controls, key=lambda row: row["overall_accuracy"])
        best_tuple = max(controls, key=lambda row: row["unique_class_attribute_tuples"])
        best_distance = max(
            controls, key=lambda row: row["mean_pairwise_attribute_distance"]
        )
        summaries.append(
            {
                "ratio": ratio,
                "attriselect_accuracy": attriselect["overall_accuracy"],
                "best_control": best_accuracy["method"],
                "best_control_accuracy": best_accuracy["overall_accuracy"],
                "accuracy_margin": attriselect["overall_accuracy"]
                - best_accuracy["overall_accuracy"],
                "tuple_gain_over_best_control": attriselect[
                    "unique_class_attribute_tuples"
                ]
                - best_tuple["unique_class_attribute_tuples"],
                "distance_gain_over_best_control": attriselect[
                    "mean_pairwise_attribute_distance"
                ]
                - best_distance["mean_pairwise_attribute_distance"],
            }
        )
    return summaries


def main() -> int:
    args = parse_args()
    attribute_metadata = load_json(Path(args.attribute_file))
    records = [
        record_from_pair(manifest, result, attribute_metadata)
        for manifest, result in matched_pairs(Path(args.matched_root))
    ]
    records.extend(
        record_from_pair(manifest, result, attribute_metadata)
        for manifest, result in generated_complement_pairs(
            Path(args.generated_complement_root)
        )
    )
    records.sort(
        key=lambda row: (RATIOS.index(row["ratio"]), METHOD_ORDER.index(row["strategy"]))
    )
    validate_grid(records, RATIOS)
    add_jaccard(records, RATIOS)

    constrained_records = [
        record_from_pair(manifest, result, attribute_metadata)
        for manifest, result in constrained_pairs(Path(args.constrained_root))
    ]
    constrained_records.sort(
        key=lambda row: (
            CONSTRAINED_RATIOS.index(row["ratio"]),
            METHOD_ORDER.index(row["strategy"]),
        )
    )
    validate_grid(constrained_records, CONSTRAINED_RATIOS)
    add_jaccard(constrained_records, CONSTRAINED_RATIOS)
    summaries = low_budget_summary(constrained_records, CONSTRAINED_RATIOS)

    output = {
        "seed": 42,
        "generated_seed": 42,
        "attribute_fields": list(ATTRIBUTES),
        "generated_supported_target_values": TARGET_VALUES,
        "generated_redundancy_definition": (
            "For each selected source sample, compute the fraction of the two "
            "generated-supported target fields (color=pink, texture=fabric) "
            "that match; average this fraction over the selected subset."
        ),
        "mean_pairwise_attribute_distance_definition": (
            "Mean categorical Hamming distance over color, background, and "
            "texture across all unordered selected-source pairs."
        ),
        "class_imbalance_definition": (
            "Difference between the largest and smallest selected counts "
            "across task-specific groups."
        ),
        "records": records,
        "constrained_records": constrained_records,
        "low_budget_summary": summaries,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    markdown = "\n\n".join(
        [
            "# ImageNet10 Seed-42 Selected-Set Mechanism Analysis",
            "## Composition",
            composition_table(records),
            "## Diversity, Balance, Redundancy, Overlap, and Downstream ACC",
            mechanism_table(records),
            "## More-Constrained Composition (Ratios 0.02/0.05)",
            composition_table(constrained_records),
            "## More-Constrained Diversity, Balance, Redundancy, Overlap, and Downstream ACC",
            mechanism_table(constrained_records),
            "## More-Constrained Summary\n\n"
            + "\n".join(
                "- Ratio {ratio}: AttriSelect ACC {attriselect_accuracy:.4f}; "
                "best control {best_control} {best_control_accuracy:.4f}; margin "
                "{accuracy_margin:+.4f}; tuple gain {tuple_gain_over_best_control:+d}; "
                "attribute-distance gain {distance_gain_over_best_control:+.4f}.".format(
                    **{**summary, "ratio": format_ratio(summary["ratio"])}
                )
                for summary in summaries
            ),
        ]
    ) + "\n"
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown)
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
