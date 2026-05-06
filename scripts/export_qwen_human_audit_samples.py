#!/usr/bin/env python3
"""Export reproducible COCO/KITTI Qwen assignment samples for human audit."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_ids(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "\n".join(str(row["image_id"]) for row in rows) + "\n",
        encoding="utf-8",
    )


def load_coco_file_names(annotation_path: Path) -> dict[str, str]:
    with annotation_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(int(item["id"])): str(item["file_name"]) for item in data["images"]}


def sample_rows(rows: list[dict[str, str]], n: int, seed: int) -> list[dict[str, str]]:
    if n > len(rows):
        raise ValueError(f"Cannot sample {n} rows from {len(rows)} rows.")
    rng = random.Random(seed)
    return rng.sample(rows, n)


def maybe_copy_image(source_path: Path, image_dir: Path, file_name: str, copy_images: bool) -> Path:
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if not copy_images:
        return source_path
    image_dir.mkdir(parents=True, exist_ok=True)
    dest_path = image_dir / file_name
    shutil.copy2(source_path, dest_path)
    return dest_path


def export_coco(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_csv_rows(args.coco_qwen_csv)
    id_to_file = load_coco_file_names(args.coco_annotation_json)
    sampled = sample_rows(rows, args.n, args.seed)
    out_rows = []
    for index, row in enumerate(sampled, start=1):
        image_id = str(int(row["image_id"]))
        file_name = id_to_file[image_id]
        source_image_path = args.coco_image_root / file_name
        audit_image_path = maybe_copy_image(
            source_image_path,
            args.out_dir / "coco_images",
            file_name,
            args.copy_images,
        )
        out_rows.append(
            {
                "sample_index": str(index),
                "dataset": "coco",
                "image_id": image_id,
                "file_name": file_name,
                "image_path": str(audit_image_path),
                "source_image_path": str(source_image_path),
                "qwen_target_clothes_color_brown": str(int(row["clothes_color"] == "brown")),
                "qwen_target_weather_rainy": str(int(row["weather"] == "rainy")),
                **row,
            }
        )
    fieldnames = [
        "sample_index",
        "dataset",
        "image_id",
        "file_name",
        "image_path",
        "source_image_path",
        "qwen_target_clothes_color_brown",
        "qwen_target_weather_rainy",
        "clothes_color",
        "clothes_type",
        "background_style",
        "time_of_day",
        "weather",
        "pose",
    ]
    write_csv(args.out_dir / "coco_qwen_random100_assignments.csv", out_rows, fieldnames)
    write_ids(args.out_dir / "coco_qwen_random100_image_ids.txt", out_rows)
    return out_rows


def export_kitti(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_csv_rows(args.kitti_qwen_csv)
    sampled = sample_rows(rows, args.n, args.seed + 1)
    out_rows = []
    for index, row in enumerate(sampled, start=1):
        image_id = row["image_id"]
        file_name = f"{image_id}.png"
        source_image_path = args.kitti_image_root / file_name
        audit_image_path = maybe_copy_image(
            source_image_path,
            args.out_dir / "kitti_images",
            file_name,
            args.copy_images,
        )
        out_rows.append(
            {
                "sample_index": str(index),
                "dataset": "kitti",
                "image_id": image_id,
                "file_name": file_name,
                "image_path": str(audit_image_path),
                "source_image_path": str(source_image_path),
                "qwen_target_color_red": str(int(row["color"] == "red")),
                "qwen_target_weather_cloudy": str(int(row["weather"] == "cloudy")),
                **row,
            }
        )
    fieldnames = [
        "sample_index",
        "dataset",
        "image_id",
        "file_name",
        "image_path",
        "source_image_path",
        "qwen_target_color_red",
        "qwen_target_weather_cloudy",
        "color",
        "weather",
        "background_style",
        "traffic_density",
        "lighting_condition",
        "road_condition",
    ]
    write_csv(args.out_dir / "kitti_qwen_random100_assignments.csv", out_rows, fieldnames)
    write_ids(args.out_dir / "kitti_qwen_random100_image_ids.txt", out_rows)
    return out_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "bxz/qwen_human_audit")
    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument("--copy-images", dest="copy_images", action="store_true")
    image_group.add_argument("--no-copy-images", dest="copy_images", action="store_false")
    parser.set_defaults(copy_images=True)
    parser.add_argument(
        "--coco-qwen-csv",
        type=Path,
        default=REPO_ROOT / "save/attribute_coco_pose_gemini_train20k_qwen3_8B.csv",
    )
    parser.add_argument(
        "--coco-annotation-json",
        type=Path,
        default=Path("/data/oxx220000/coco/annotations/person_keypoints_train2017.json"),
    )
    parser.add_argument("--coco-image-root", type=Path, default=Path("/data/oxx220000/coco/train2017"))
    parser.add_argument("--kitti-qwen-csv", type=Path, default=REPO_ROOT / "save/attribute_kitti_qwen3_8B.csv")
    parser.add_argument(
        "--kitti-image-root",
        type=Path,
        default=Path("/data/oxx220000/kitti_yolov8_first5000/images/train"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coco_rows = export_coco(args)
    kitti_rows = export_kitti(args)
    summary = {
        "seed": args.seed,
        "n_per_dataset": args.n,
        "coco_qwen_csv": str(args.coco_qwen_csv),
        "kitti_qwen_csv": str(args.kitti_qwen_csv),
        "copy_images": args.copy_images,
        "coco_rows": len(coco_rows),
        "kitti_rows": len(kitti_rows),
        "outputs": [
            str(args.out_dir / "coco_qwen_random100_assignments.csv"),
            str(args.out_dir / "coco_qwen_random100_image_ids.txt"),
            str(args.out_dir / "kitti_qwen_random100_assignments.csv"),
            str(args.out_dir / "kitti_qwen_random100_image_ids.txt"),
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "qwen_random100_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
