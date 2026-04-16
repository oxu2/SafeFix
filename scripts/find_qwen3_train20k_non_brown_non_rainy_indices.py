#!/usr/bin/env python3
"""
Find Qwen3 train20k image ids whose clothes color is not brown and weather is not rainy.

The default input is the canonical Qwen3 COCO pose train20k CSV:
    save/attribute_coco_pose_gemini_train20k_qwen3_8B.csv

The script writes:
  1. A plain-text file with one matched image_id per line.
  2. A CSV file with matched image_id plus the original CSV row_idx for reference.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find Qwen3 train20k image ids where clothes_color is not brown "
            "and weather is not rainy."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("save/attribute_coco_pose_gemini_train20k_qwen3_8B.csv"),
        help="Input Qwen3 attribute CSV.",
    )
    parser.add_argument(
        "--exclude-clothes-color",
        type=str,
        default="brown",
        help="Exclude rows with this clothes_color value.",
    )
    parser.add_argument(
        "--exclude-weather",
        type=str,
        default="rainy",
        help="Exclude rows with this weather value.",
    )
    parser.add_argument(
        "--row-index-base",
        type=int,
        choices=[0, 1],
        default=0,
        help="Whether the reference row_idx in the metadata CSV should be 0-based or 1-based.",
    )
    parser.add_argument(
        "--output-image-ids",
        type=Path,
        default=Path("save/qwen3_train20k_non_brown_non_rainy_image_ids.txt"),
        help="Output text file containing one matched image_id per line.",
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=Path("save/qwen3_train20k_non_brown_non_rainy_matches.csv"),
        help="Output CSV file containing row_idx, image_id, clothes_color, and weather.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    required_columns = {"image_id", "clothes_color", "weather"}
    matches = []

    with args.csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {args.csv}")

        missing = required_columns.difference(reader.fieldnames)
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"Missing required columns in {args.csv}: {missing_str}")

        for zero_based_idx, row in enumerate(reader):
            clothes_color = row["clothes_color"].strip()
            weather = row["weather"].strip()
            if clothes_color == args.exclude_clothes_color:
                continue
            if weather == args.exclude_weather:
                continue

            matches.append(
                {
                    "row_idx": zero_based_idx + args.row_index_base,
                    "image_id": row["image_id"].strip(),
                    "clothes_color": clothes_color,
                    "weather": weather,
                }
            )

    args.output_image_ids.parent.mkdir(parents=True, exist_ok=True)
    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)

    with args.output_image_ids.open("w", newline="") as handle:
        for item in matches:
            handle.write(f"{item['image_id']}\n")

    with args.output_metadata.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_idx", "image_id", "clothes_color", "weather"],
        )
        writer.writeheader()
        writer.writerows(matches)

    print(f"Input CSV: {args.csv}")
    print(f"Matched rows: {len(matches)}")
    print(f"Metadata row_idx base: {args.row_index_base}")
    print(f"Image id output: {args.output_image_ids}")
    print(f"Metadata output: {args.output_metadata}")
    if matches:
        preview = ", ".join(str(item["image_id"]) for item in matches[:10])
        print(f"First 10 image_ids: {preview}")


if __name__ == "__main__":
    main()
