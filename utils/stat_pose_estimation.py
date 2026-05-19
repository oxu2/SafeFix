#!/usr/bin/env python3
"""
Slice-level statistics for COCO pose-estimation results.

This script evaluates COCO-format keypoint predictions on attribute-defined image
subsets. It is intended for outputs such as:
  - pose_estimation/train_pose_estimation.py eval/test result JSONs
  - attribute JSON/CSV files produced by utils/assign_attribute.py or utils/blip.py

Notes:
  - Slice metrics are computed at the image level: all person instances in a
    selected image contribute to the slice's COCO AP/AR.
  - When the attribute source is JSON, image ids are reconstructed from the
    canonical sorted valid-image ordering of the specified COCO split. This
    matches the ordering used by the attribute-assignment scripts.
"""

import argparse
import contextlib
import copy
import csv
import io
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from pose_estimation.dataset import (
    CocoPersonKeypointsDataset,
    build_test_dataset,
    build_train_val_test_datasets,
    default_coco_paths,
)


COCO_STAT_NAMES = [
    "AP",
    "AP50",
    "AP75",
    "AP_medium",
    "AP_large",
    "AR",
    "AR50",
    "AR75",
    "AR_medium",
    "AR_large",
]

CSV_METADATA_COLUMNS = {"index", "image_path", "image_id", "file_name", "split"}


@dataclass
class AttributeSource:
    values_by_attr: Dict[str, Dict[int, str]]
    words_by_attr: Dict[str, List[str]]
    image_ids: List[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute slice-level COCO keypoint metrics from attribute annotations."
    )
    parser.add_argument(
        "--predictions-json",
        type=Path,
        required=True,
        help="COCO-format keypoint predictions JSON (list of detections).",
    )
    parser.add_argument(
        "--attribute-file",
        type=Path,
        required=True,
        help="Attribute JSON/CSV aligned to a COCO split (or train+val combined).",
    )
    parser.add_argument(
        "--attribute-split",
        choices=["train", "val_full", "val", "test", "all"],
        default="val_full",
        help="Which COCO split ordering was used for --attribute-file.",
    )
    parser.add_argument(
        "--train-attribute-file",
        type=Path,
        default=None,
        help="Optional second attribute JSON/CSV for train-distribution comparison.",
    )
    parser.add_argument(
        "--train-attribute-split",
        choices=["train", "val_full", "val", "test", "all"],
        default="train",
        help="Which COCO split ordering was used for --train-attribute-file.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Optional corpus JSON to control attribute/value ordering.",
    )
    parser.add_argument(
        "--attributes",
        type=str,
        default=None,
        help="Comma-separated attribute names to evaluate. Defaults to all available attributes.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/data/oxx220000/coco",
        help="COCO root used to resolve default annotation/image paths.",
    )
    parser.add_argument(
        "--eval-annotation-file",
        type=str,
        default=None,
        help="Ground-truth COCO annotation JSON for evaluation. Defaults to val/test annotations from --data-root.",
    )
    parser.add_argument(
        "--eval-image-dir",
        type=str,
        default=None,
        help="Image directory for the eval split when reconstructing JSON ordering.",
    )
    parser.add_argument(
        "--eval-split",
        choices=["val", "test", "val_full", "train"],
        default="val",
        help="COCO split represented by --predictions-json.",
    )
    parser.add_argument(
        "--train-annotation-file",
        type=str,
        default=None,
        help="Optional train annotation JSON used for train-distribution reconstruction.",
    )
    parser.add_argument(
        "--train-image-dir",
        type=str,
        default=None,
        help="Optional train image dir used for train-distribution reconstruction.",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=25,
        help="Skip slices with fewer than this many images.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="How many lowest-AP slices to print.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["ap", "ap_delta", "images"],
        default="ap",
        help="Sort order for printed/saved slice rows.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults under logs/pose/pose_slice_stats/.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV output path. Defaults under logs/pose/pose_slice_stats/.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for the random val/test half split.",
    )
    parser.add_argument(
        "--acc-diff-threshold",
        type=float,
        default=0.005,
        help="Minimum AP drop vs. overall AP before a slice is flagged as a failure case.",
    )
    parser.add_argument(
        "--distribution-diff-threshold",
        type=float,
        default=0.15,
        help="Minimum eval-train share gap before a flagged slice is labeled distribution_shift.",
    )
    parser.add_argument(
        "--rare-label-threshold",
        type=float,
        default=1.1,
        help=(
            "Rare-slice multiplier against the uniform train-share baseline. "
            "Larger values mark more slices as rare."
        ),
    )
    parser.add_argument(
        "--train-max-samples",
        type=int,
        default=0,
        help="Optional cap on train images when computing train-side attribute distribution.",
    )
    return parser.parse_args()


def parse_attribute_names(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    values = [part.strip() for part in raw.split(",")]
    return [value for value in values if value]


def default_output_paths(predictions_json: Path, attribute_file: Path) -> Tuple[Path, Path]:
    base_dir = Path("logs/pose/pose_slice_stats")
    stem = f"{predictions_json.stem}__{attribute_file.stem}"
    return base_dir / f"{stem}.json", base_dir / f"{stem}.csv"


def split_suffix_token(split: str) -> Optional[str]:
    if split in {"val", "val_full"}:
        return "val"
    if split in {"train", "test", "all"}:
        return split
    return None


def candidate_attribute_paths(path: Path, split: str) -> List[Path]:
    candidates = [path]
    token = split_suffix_token(split)
    if token is None:
        return candidates

    stem = path.stem
    suffix = path.suffix
    tagged_stem = f"{stem}_{token}"
    token_suffix = f"_{token}"

    if stem.endswith(token_suffix):
        candidates.append(path.with_name(f"{stem[:-len(token_suffix)]}{suffix}"))
    else:
        candidates.append(path.with_name(f"{tagged_stem}{suffix}"))

    deduped: List[Path] = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def resolve_attribute_path(path: Optional[Path], split: str, label: str) -> Optional[Path]:
    if path is None:
        return None

    candidates = candidate_attribute_paths(path, split)
    for candidate in candidates:
        if candidate.exists():
            if candidate != path:
                print(f"{label}: using {candidate} (fallback for requested path {path})")
            return candidate
    return path


def resolve_split_paths(
    data_root: str,
    split: str,
    annotation_file: Optional[str] = None,
    image_dir: Optional[str] = None,
) -> Tuple[str, str]:
    paths = default_coco_paths(data_root)
    concrete_split = "val" if split == "val_full" else split
    if concrete_split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported concrete split: {split}")
    return (
        annotation_file or paths[f"{concrete_split}_annotations"],
        image_dir or paths[f"{concrete_split}_images"],
    )


def build_valid_image_ids(
    data_root: str,
    split: str,
    annotation_file: Optional[str] = None,
    image_dir: Optional[str] = None,
    seed: int = 42,
    max_train_samples: int = 0,
) -> List[int]:
    if split == "all":
        train_ids = build_valid_image_ids(
            data_root=data_root,
            split="train",
            annotation_file=annotation_file,
            image_dir=image_dir,
            seed=seed,
            max_train_samples=max_train_samples,
        )
        val_ids = build_valid_image_ids(data_root=data_root, split="val_full", seed=seed)
        return train_ids + val_ids

    if split == "train":
        resolved_annotation, resolved_image_dir = resolve_split_paths(
            data_root=data_root,
            split="train",
            annotation_file=annotation_file,
            image_dir=image_dir,
        )
        dataset = CocoPersonKeypointsDataset(
            image_dir=resolved_image_dir,
            annotation_file=resolved_annotation,
            train=False,
            hflip_prob=0.0,
            filter_empty=True,
        )
        image_ids = list(dataset.image_ids)
        if max_train_samples > 0:
            image_ids = image_ids[:max_train_samples]
        return image_ids

    if split == "val_full":
        resolved_annotation, resolved_image_dir = resolve_split_paths(
            data_root=data_root,
            split="val",
            annotation_file=annotation_file,
            image_dir=image_dir,
        )
        dataset = CocoPersonKeypointsDataset(
            image_dir=resolved_image_dir,
            annotation_file=resolved_annotation,
            train=False,
            hflip_prob=0.0,
            filter_empty=True,
        )
        return list(dataset.image_ids)

    if split == "val":
        _, val_dataset, _ = build_train_val_test_datasets(
            data_root=data_root,
            val_images=image_dir,
            val_annotations=annotation_file,
            hflip_prob=0.0,
            seed=seed,
            max_train_samples=0,
        )
        return list(val_dataset.image_ids)

    if split == "test":
        test_dataset = build_test_dataset(
            data_root=data_root,
            test_images=image_dir,
            test_annotations=annotation_file,
            seed=seed,
        )
        return list(test_dataset.image_ids)

    raise ValueError(f"Unsupported split: {split}")


def load_corpus(path: Optional[Path]) -> Dict[str, Dict]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "Corpus" in payload and isinstance(payload["Corpus"], dict):
        return payload["Corpus"]
    return payload


def load_attribute_source(
    path: Path,
    split: str,
    data_root: str,
    annotation_file: Optional[str] = None,
    image_dir: Optional[str] = None,
    seed: int = 42,
    max_train_samples: int = 0,
) -> AttributeSource:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_attribute_source_from_csv(path)
    if suffix != ".json":
        raise ValueError(f"Unsupported attribute file type: {path}")
    expected_ids = build_valid_image_ids(
        data_root=data_root,
        split=split,
        annotation_file=annotation_file,
        image_dir=image_dir,
        seed=seed,
        max_train_samples=max_train_samples,
    )
    return load_attribute_source_from_json(path, expected_ids)


def load_attribute_source_from_json(path: Path, expected_ids: Sequence[int]) -> AttributeSource:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if "Corpus" in payload:
        raise ValueError(f"{path} looks like a corpus definition, not attribute assignments.")

    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{path} does not contain attribute assignments.")

    attr_names = list(payload.keys())
    lengths = {}
    values_by_attr: Dict[str, Dict[int, str]] = {}
    words_by_attr: Dict[str, List[str]] = {}

    for attr_name in attr_names:
        attr_payload = payload[attr_name]
        if not isinstance(attr_payload, dict) or "data" not in attr_payload:
            raise ValueError(f"{path} -> '{attr_name}' is missing a 'data' field.")
        data = attr_payload["data"]
        if not isinstance(data, list):
            raise ValueError(f"{path} -> '{attr_name}.data' must be a list.")
        lengths[attr_name] = len(data)
        words_by_attr[attr_name] = list(attr_payload.get("word", []))

    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(f"{path} has inconsistent attribute lengths: {lengths}")

    row_count = unique_lengths.pop()
    if row_count > len(expected_ids):
        raise ValueError(
            f"{path} has {row_count} rows but the reconstructed split only has {len(expected_ids)} valid images."
        )

    image_ids = list(expected_ids[:row_count])
    for attr_name in attr_names:
        data = payload[attr_name]["data"]
        attr_values = {}
        for image_id, value in zip(image_ids, data):
            attr_values[int(image_id)] = str(value).strip() if value is not None else ""
        values_by_attr[attr_name] = attr_values

    return AttributeSource(
        values_by_attr=values_by_attr,
        words_by_attr=words_by_attr,
        image_ids=image_ids,
    )


def load_attribute_source_from_csv(path: Path) -> AttributeSource:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "image_id" not in reader.fieldnames:
            raise ValueError(f"{path} must contain an 'image_id' column.")

        attr_names = [name for name in reader.fieldnames if name not in CSV_METADATA_COLUMNS]
        image_ids: List[int] = []
        values_by_attr = {attr_name: {} for attr_name in attr_names}
        for row in reader:
            raw_image_id = (row.get("image_id") or "").strip()
            if not raw_image_id:
                continue
            image_id = int(raw_image_id)
            image_ids.append(image_id)
            for attr_name in attr_names:
                values_by_attr[attr_name][image_id] = (row.get(attr_name) or "").strip()

    return AttributeSource(
        values_by_attr=values_by_attr,
        words_by_attr={},
        image_ids=image_ids,
    )


def choose_attribute_names(
    source: AttributeSource,
    requested_names: Optional[Sequence[str]],
) -> List[str]:
    available = list(source.values_by_attr.keys())
    if requested_names is None:
        return available

    missing = [name for name in requested_names if name not in source.values_by_attr]
    if missing:
        raise ValueError(f"Requested attributes not found: {missing}")
    return list(requested_names)


def project_source_to_ids(
    source: AttributeSource,
    image_ids: Sequence[int],
    attr_names: Sequence[str],
) -> Dict[str, Dict[int, str]]:
    projected = {}
    for attr_name in attr_names:
        attr_lookup = source.values_by_attr[attr_name]
        projected[attr_name] = {
            int(image_id): attr_lookup[int(image_id)]
            for image_id in image_ids
            if int(image_id) in attr_lookup
        }
    return projected


def build_value_order(
    attr_name: str,
    eval_lookup: Dict[int, str],
    train_lookup: Optional[Dict[int, str]],
    corpus: Dict[str, Dict],
    source: AttributeSource,
) -> List[str]:
    ordered: List[str] = []
    seen = set()

    corpus_words = corpus.get(attr_name, {}).get("word", [])
    source_words = source.words_by_attr.get(attr_name, [])
    observed = []
    for lookup in (eval_lookup, train_lookup or {}):
        observed.extend(value for value in lookup.values() if value)

    for value in list(corpus_words) + list(source_words) + observed:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def load_predictions(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(
            f"{path} must be a COCO result list. Summary-only metric JSONs are not supported."
        )
    return payload


def build_coco_subset(coco_gt: COCO, image_ids: Sequence[int]) -> COCO:
    image_ids = sorted({int(image_id) for image_id in image_ids})
    ann_ids = coco_gt.getAnnIds(imgIds=image_ids)
    subset = COCO()
    subset.dataset = {
        "info": copy.deepcopy(coco_gt.dataset.get("info", {})),
        "licenses": copy.deepcopy(coco_gt.dataset.get("licenses", [])),
        "images": copy.deepcopy(coco_gt.loadImgs(image_ids)),
        "annotations": copy.deepcopy(coco_gt.loadAnns(ann_ids)),
        "categories": copy.deepcopy(coco_gt.loadCats(coco_gt.getCatIds())),
    }
    subset.createIndex()
    return subset


def empty_metrics() -> Dict[str, float]:
    return {name: 0.0 for name in COCO_STAT_NAMES}


def blank_metrics() -> Dict[str, Optional[float]]:
    return {name: None for name in COCO_STAT_NAMES}


def blank_overall(eval_image_count: int) -> Dict[str, Optional[float]]:
    overall = blank_metrics()
    overall["num_images"] = None
    overall["num_gt_annotations"] = None
    overall["num_predictions"] = None
    overall["coverage_images"] = 0
    overall["coverage_fraction"] = 0.0 if eval_image_count else None
    return overall


def evaluate_subset(
    coco_gt: COCO,
    predictions: Sequence[Dict],
    image_ids: Sequence[int],
) -> Dict[str, float]:
    unique_image_ids = sorted({int(image_id) for image_id in image_ids})
    image_id_set = set(unique_image_ids)
    subset_predictions = [row for row in predictions if int(row.get("image_id", -1)) in image_id_set]
    subset_gt = build_coco_subset(coco_gt, unique_image_ids)

    metrics = empty_metrics()
    metrics["num_images"] = len(unique_image_ids)
    metrics["num_gt_annotations"] = len(subset_gt.dataset.get("annotations", []))
    metrics["num_predictions"] = len(subset_predictions)

    if not unique_image_ids or metrics["num_gt_annotations"] == 0 or not subset_predictions:
        return metrics

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = subset_gt.loadRes(subset_predictions)

    coco_eval = COCOeval(subset_gt, coco_dt, "keypoints")
    coco_eval.params.imgIds = unique_image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.summarize()
    stats = coco_eval.stats
    for name, value in zip(COCO_STAT_NAMES, stats):
        metrics[name] = float(value)
    return metrics


def compute_distribution(lookup: Dict[int, str]) -> Tuple[Counter, int]:
    counter = Counter(value for value in lookup.values() if value)
    total = sum(counter.values())
    return counter, total


def nullable_sort_key(value: Optional[float], *, descending: bool = False) -> Tuple[int, float]:
    if value is None:
        return (1, 0.0)
    return (0, -float(value) if descending else float(value))


def sort_rows(rows: List[Dict], sort_by: str) -> List[Dict]:
    if sort_by == "images":
        return sorted(
            rows,
            key=lambda row: (
                *nullable_sort_key(row.get("num_images"), descending=True),
                row["attribute"],
                row["value"],
            ),
        )
    if sort_by == "ap_delta":
        return sorted(
            rows,
            key=lambda row: (
                *nullable_sort_key(row.get("ap_delta")),
                *nullable_sort_key(row.get("AP")),
                row["attribute"],
                row["value"],
            ),
        )
    return sorted(
        rows,
        key=lambda row: (
            *nullable_sort_key(row.get("AP")),
            row["attribute"],
            row["value"],
        ),
    )


def classify_failure_case(
    *,
    ap: float,
    overall_ap: float,
    acc_diff_threshold: float,
    train_share: Optional[float],
    eval_share: Optional[float],
    distribution_diff_threshold: float,
    rare_label_threshold: float,
    num_values: int,
) -> Tuple[bool, Optional[str]]:
    if ap >= overall_ap - acc_diff_threshold:
        return False, None

    if train_share is not None and eval_share is not None:
        uniform_share = 1.0 / max(1, num_values)
        if eval_share - train_share > distribution_diff_threshold:
            return True, "distribution_shift"
        if train_share < rare_label_threshold * uniform_share:
            return True, "rare_case"
    return True, "hard_case"


def print_summary(overall: Dict[str, float], rows: Sequence[Dict], top_k: int) -> None:
    if overall.get("AP") is None:
        coverage_images = overall.get("coverage_images")
        print(
            "Overall:",
            "unavailable",
            f"coverage_images={int(coverage_images) if coverage_images is not None else 'NA'}",
            "Eval-side metrics were left blank.",
        )
        print("")
    else:
        print(
            "Overall:",
            f"images={int(overall['num_images'])}",
            f"gt={int(overall['num_gt_annotations'])}",
            f"pred={int(overall['num_predictions'])}",
            f"AP={overall['AP']:.4f}",
            f"AP50={overall['AP50']:.4f}",
            f"AR={overall['AR']:.4f}",
        )
        print("")

    failure_rows = [row for row in rows if row.get("is_failure_case")]
    print(f"Failure cases (top {min(top_k, len(failure_rows))}):")
    for row in failure_rows[:top_k]:
        line = (
            f"{row['attribute']}={row['value']} | images={row['num_images']} "
            f"| gt={row['num_gt_annotations']} | pred={row['num_predictions']} "
            f"| AP={row['AP']:.4f} | delta={row['ap_delta']:+.4f}"
        )
        if row.get("failure_type"):
            line += f" | type={row['failure_type']}"
        train_share = row.get("train_share")
        eval_share = row.get("eval_share")
        if train_share is not None and eval_share is not None:
            line += (
                f" | train_share={train_share:.4f}"
                f" | eval_share={eval_share:.4f}"
                f" | gap={row['share_gap']:+.4f}"
            )
        print(line)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        fieldnames = [
            "attribute",
            "value",
            "num_images",
            "num_gt_annotations",
            "num_predictions",
            *COCO_STAT_NAMES,
            "ap_delta",
            "train_count",
            "train_share",
            "eval_count",
            "eval_share",
            "share_gap",
            "is_failure_case",
            "failure_type",
        ]
    else:
        fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ("" if value is None else value)
                    for key, value in row.items()
                }
            )


def write_unavailable_summary(
    args: argparse.Namespace,
    *,
    eval_annotation_file: str,
    reason: str,
) -> Dict[str, object]:
    summary = {
        "predictions_json": str(args.predictions_json),
        "attribute_file": str(args.attribute_file),
        "attribute_split": args.attribute_split,
        "train_attribute_file": str(args.train_attribute_file) if args.train_attribute_file else None,
        "train_attribute_split": args.train_attribute_split if args.train_attribute_file else None,
        "eval_annotation_file": str(eval_annotation_file),
        "eval_split": args.eval_split,
        "attributes": parse_attribute_names(args.attributes),
        "min_images": args.min_images,
        "seed": args.seed,
        "acc_diff_threshold": args.acc_diff_threshold,
        "distribution_diff_threshold": args.distribution_diff_threshold,
        "rare_label_threshold": args.rare_label_threshold,
        "train_max_samples": args.train_max_samples,
        "status": "NA",
        "reason": reason,
        "overall": None,
        "slices": [],
    }

    ensure_parent(args.output_json)
    ensure_parent(args.output_csv)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_csv(args.output_csv, [])

    print("")
    print(f"Slice analysis unavailable: {reason}")
    print(f"Saved JSON summary to {args.output_json}")
    print(f"Saved CSV summary to {args.output_csv}")
    return summary


def run_analysis(args: argparse.Namespace) -> Dict[str, object]:
    requested_attributes = parse_attribute_names(args.attributes)

    args.attribute_file = resolve_attribute_path(args.attribute_file, args.attribute_split, "Eval attribute file")
    args.train_attribute_file = resolve_attribute_path(
        args.train_attribute_file,
        args.train_attribute_split,
        "Train attribute file",
    )

    if args.output_json is None or args.output_csv is None:
        default_json, default_csv = default_output_paths(args.predictions_json, args.attribute_file)
        if args.output_json is None:
            args.output_json = default_json
        if args.output_csv is None:
            args.output_csv = default_csv

    eval_annotation_file, eval_image_dir = resolve_split_paths(
        data_root=args.data_root,
        split=args.eval_split,
        annotation_file=args.eval_annotation_file,
        image_dir=args.eval_image_dir,
    )
    eval_image_ids = build_valid_image_ids(
        data_root=args.data_root,
        split=args.eval_split,
        annotation_file=eval_annotation_file,
        image_dir=eval_image_dir,
        seed=args.seed,
    )

    if not args.attribute_file.exists():
        return write_unavailable_summary(
            args,
            eval_annotation_file=eval_annotation_file,
            reason=f"Missing eval attribute file: {args.attribute_file}",
        )

    if args.train_attribute_file is not None and not args.train_attribute_file.exists():
        print(
            f"Train attribute file missing: {args.train_attribute_file}. "
            "Train-side share fields will be NA."
        )
        args.train_attribute_file = None

    primary_source = load_attribute_source(
        path=args.attribute_file,
        split=args.attribute_split,
        data_root=args.data_root,
        annotation_file=args.eval_annotation_file if args.attribute_split == args.eval_split else None,
        image_dir=args.eval_image_dir if args.attribute_split == args.eval_split else None,
        seed=args.seed,
        max_train_samples=args.train_max_samples if args.attribute_split == "train" else 0,
    )
    attr_names = choose_attribute_names(primary_source, requested_attributes)

    primary_eval_lookup = project_source_to_ids(primary_source, eval_image_ids, attr_names)
    covered_eval_ids = [
        image_id
        for image_id in eval_image_ids
        if all(image_id in primary_eval_lookup[attr_name] for attr_name in attr_names)
    ]

    train_lookup: Optional[Dict[str, Dict[int, str]]] = None
    train_source: Optional[AttributeSource] = None
    train_image_ids: List[int] = []
    if args.train_attribute_file is not None:
        train_annotation_file, train_image_dir = resolve_split_paths(
            data_root=args.data_root,
            split="train",
            annotation_file=args.train_annotation_file,
            image_dir=args.train_image_dir,
        )
        train_image_ids = build_valid_image_ids(
            data_root=args.data_root,
            split="train",
            annotation_file=train_annotation_file,
            image_dir=train_image_dir,
            seed=args.seed,
            max_train_samples=args.train_max_samples,
        )
        train_source = load_attribute_source(
            path=args.train_attribute_file,
            split=args.train_attribute_split,
            data_root=args.data_root,
            annotation_file=args.train_annotation_file
            if args.train_attribute_split == "train"
            else None,
            image_dir=args.train_image_dir if args.train_attribute_split == "train" else None,
            seed=args.seed,
            max_train_samples=args.train_max_samples if args.train_attribute_split == "train" else 0,
        )
        train_lookup = project_source_to_ids(train_source, train_image_ids, attr_names)
    else:
        train_image_ids = build_valid_image_ids(
            data_root=args.data_root,
            split="train",
            seed=args.seed,
            max_train_samples=args.train_max_samples,
        )
        inferred_train_lookup = project_source_to_ids(primary_source, train_image_ids, attr_names)
        if any(inferred_train_lookup[attr_name] for attr_name in attr_names):
            train_lookup = inferred_train_lookup
            train_source = primary_source

    if not covered_eval_ids and train_lookup is None:
        return write_unavailable_summary(
            args,
            eval_annotation_file=eval_annotation_file,
            reason=(
                "No eval images are covered by the requested attribute file, and no "
                "train-side attribute distribution could be recovered."
            ),
        )

    corpus = load_corpus(args.corpus)
    predictions: Optional[List[Dict]] = None
    coco_gt: Optional[COCO] = None
    if covered_eval_ids:
        predictions = load_predictions(args.predictions_json)
        coco_gt = COCO(eval_annotation_file)
        overall: Dict[str, object] = evaluate_subset(
            coco_gt=coco_gt,
            predictions=predictions,
            image_ids=covered_eval_ids,
        )
        overall["coverage_images"] = len(covered_eval_ids)
        overall["coverage_fraction"] = (
            len(covered_eval_ids) / len(eval_image_ids) if eval_image_ids else 0.0
        )
    else:
        overall = blank_overall(len(eval_image_ids))
        print(
            "No eval images are covered by the requested attribute file. "
            "Eval-side metrics will be left blank, while train-side distributions "
            "will still be summarized when available."
        )

    slice_rows: List[Dict] = []
    for attr_name in attr_names:
        eval_lookup = {
            image_id: primary_eval_lookup[attr_name][image_id]
            for image_id in covered_eval_ids
            if primary_eval_lookup[attr_name].get(image_id)
        }
        train_attr_lookup = train_lookup[attr_name] if train_lookup is not None else None
        value_order = build_value_order(
            attr_name=attr_name,
            eval_lookup=eval_lookup,
            train_lookup=train_attr_lookup,
            corpus=corpus,
            source=train_source or primary_source,
        )

        eval_counter, eval_total = compute_distribution(eval_lookup)
        train_counter, train_total = compute_distribution(train_attr_lookup or {})

        for value in value_order:
            eval_count = int(eval_counter.get(value, 0))
            train_count = int(train_counter.get(value, 0)) if train_lookup is not None else None

            if eval_count == 0:
                if train_count is None or train_count < args.min_images:
                    continue
            elif eval_count < args.min_images:
                continue

            row = {
                "attribute": attr_name,
                "value": value,
                "num_images": None,
                "num_gt_annotations": None,
                "num_predictions": None,
            }
            for metric_name in COCO_STAT_NAMES:
                row[metric_name] = None
            row["ap_delta"] = None

            row["eval_count"] = None
            row["eval_share"] = None

            if train_lookup is not None and train_total:
                row["train_count"] = train_count
                row["train_share"] = round(train_count / train_total, 6)
            else:
                row["train_count"] = None
                row["train_share"] = None
            row["share_gap"] = None
            row["is_failure_case"] = None
            row["failure_type"] = None

            if eval_count > 0:
                value_image_ids = [
                    image_id
                    for image_id in covered_eval_ids
                    if primary_eval_lookup[attr_name].get(image_id) == value
                ]
                metrics = evaluate_subset(
                    coco_gt=coco_gt,
                    predictions=predictions,
                    image_ids=value_image_ids,
                )
                row["num_images"] = int(metrics["num_images"])
                row["num_gt_annotations"] = int(metrics["num_gt_annotations"])
                row["num_predictions"] = int(metrics["num_predictions"])
                for metric_name in COCO_STAT_NAMES:
                    row[metric_name] = round(float(metrics[metric_name]), 6)
                row["eval_count"] = eval_count
                row["eval_share"] = round(eval_count / eval_total, 6) if eval_total else None

                if row["train_share"] is not None and row["eval_share"] is not None:
                    row["share_gap"] = round(row["eval_share"] - row["train_share"], 6)

                if overall.get("AP") is not None:
                    row["ap_delta"] = round(float(metrics["AP"] - overall["AP"]), 6)
                    is_failure_case, failure_type = classify_failure_case(
                        ap=float(metrics["AP"]),
                        overall_ap=float(overall["AP"]),
                        acc_diff_threshold=args.acc_diff_threshold,
                        train_share=row["train_share"],
                        eval_share=row["eval_share"],
                        distribution_diff_threshold=args.distribution_diff_threshold,
                        rare_label_threshold=args.rare_label_threshold,
                        num_values=max(1, len(value_order)),
                    )
                    row["is_failure_case"] = is_failure_case
                    row["failure_type"] = failure_type

            slice_rows.append(row)

    sorted_rows = sort_rows(slice_rows, args.sort_by)
    print_summary(overall=overall, rows=sorted_rows, top_k=args.top_k)

    summary = {
        "predictions_json": str(args.predictions_json),
        "attribute_file": str(args.attribute_file),
        "attribute_split": args.attribute_split,
        "train_attribute_file": str(args.train_attribute_file) if args.train_attribute_file else None,
        "train_attribute_split": args.train_attribute_split if args.train_attribute_file else None,
        "eval_annotation_file": str(eval_annotation_file),
        "eval_split": args.eval_split,
        "attributes": attr_names,
        "min_images": args.min_images,
        "seed": args.seed,
        "acc_diff_threshold": args.acc_diff_threshold,
        "distribution_diff_threshold": args.distribution_diff_threshold,
        "rare_label_threshold": args.rare_label_threshold,
        "train_max_samples": args.train_max_samples,
        "overall": overall,
        "slices": sorted_rows,
    }
    if not covered_eval_ids:
        summary["status"] = "train_only"
        summary["reason"] = (
            "No eval images are covered by the requested attribute file. "
            "Eval-side fields were left blank; train-side distributions were "
            "still summarized when available."
        )

    ensure_parent(args.output_json)
    ensure_parent(args.output_csv)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_csv(args.output_csv, sorted_rows)

    print("")
    print(f"Saved JSON summary to {args.output_json}")
    print(f"Saved CSV summary to {args.output_csv}")
    return summary


def main() -> None:
    args = parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
