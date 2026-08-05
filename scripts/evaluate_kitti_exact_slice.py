#!/usr/bin/env python3
"""Evaluate a KITTI exact attribute conjunction and its complement with YOLO.val."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-weights", type=Path, required=True)
    parser.add_argument("--kitti-root", type=Path, required=True)
    parser.add_argument("--attribute-file", type=Path, required=True)
    parser.add_argument("--exact-slice", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def parse_assignments(raw: str) -> Dict[str, str]:
    assignments: Dict[str, str] = {}
    for part in raw.split(","):
        if "=" not in part:
            raise ValueError(f"Invalid exact-slice assignment: {part!r}")
        name, value = (token.strip() for token in part.split("=", 1))
        if not name or not value:
            raise ValueError(f"Invalid exact-slice assignment: {part!r}")
        assignments[name] = value
    return assignments


def load_attributes(path: Path, image_ids: list[str]) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    result: Dict[str, Dict[str, str]] = {}
    for name, info in payload.items():
        if not isinstance(info, dict) or not isinstance(info.get("data"), list):
            continue
        result[name] = {
            image_id: str(value).strip()
            for image_id, value in zip(image_ids, info["data"])
            if value is not None and str(value).strip()
        }
    return result


def metric_dict(results) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for key, value in getattr(results, "results_dict", {}).items():
        try:
            output[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return output


def stage_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Missing KITTI subset source file: {source}")
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_dataset_yaml(
    output_dir: Path,
    name: str,
    image_paths: list[Path],
    kitti_root: Path,
) -> Path:
    subset_root = output_dir / "datasets" / name
    image_dir = subset_root / "images" / "test"
    label_dir = subset_root / "labels" / "test"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for cache_path in subset_root.rglob("*.cache"):
        cache_path.unlink()
    for image_path in image_paths:
        label_path = kitti_root / "labels" / "test" / f"{image_path.stem}.txt"
        stage_file(image_path, image_dir / image_path.name)
        stage_file(label_path, label_dir / label_path.name)

    dataset_yaml = output_dir / f"{name}.yaml"
    dataset_yaml.write_text(
        (
            f"path: {subset_root.absolute()}\n"
            "train: images/test\n"
            "val: images/test\n"
            "test: images/test\n"
            "names: ['Car', 'Pedestrian']\n"
        ),
        encoding="utf-8",
    )
    return dataset_yaml


def main() -> None:
    args = parse_args()
    assignments = parse_assignments(args.exact_slice)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    test_images = sorted((args.kitti_root / "images" / "test").glob("*"))
    test_images = [path for path in test_images if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    image_ids = [path.stem for path in test_images]
    attributes = load_attributes(args.attribute_file, image_ids)
    missing = [name for name in assignments if name not in attributes]
    if missing:
        raise ValueError("Missing exact-slice attributes: " + ", ".join(missing))

    target_images = [
        path
        for path in test_images
        if all(attributes[name].get(path.stem) == value for name, value in assignments.items())
    ]
    target_ids = {path.stem for path in target_images}
    non_target_images = [path for path in test_images if path.stem not in target_ids]
    if not target_images or not non_target_images:
        raise ValueError(
            f"Invalid target/non-target support: {len(target_images)}/{len(non_target_images)}"
        )

    target_yaml = write_dataset_yaml(
        args.output_dir, "target", target_images, args.kitti_root
    )
    non_target_yaml = write_dataset_yaml(
        args.output_dir, "non_target", non_target_images, args.kitti_root
    )

    from ultralytics import YOLO

    model = YOLO(str(args.model_weights))
    common = {
        "split": "test",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": str(args.output_dir.resolve()),
        "plots": False,
        "verbose": False,
    }
    target_results = model.val(data=str(target_yaml), name="target_eval", **common)
    non_target_results = model.val(data=str(non_target_yaml), name="non_target_eval", **common)

    output = {
        "model_weights": str(args.model_weights.resolve()),
        "kitti_root": str(args.kitti_root.resolve()),
        "attribute_file": str(args.attribute_file.resolve()),
        "assignments": assignments,
        "target": {"num_images": len(target_images), "metrics": metric_dict(target_results)},
        "non_target": {
            "num_images": len(non_target_images),
            "metrics": metric_dict(non_target_results),
        },
    }
    output_path = args.output_dir / "official_exact_slice.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
