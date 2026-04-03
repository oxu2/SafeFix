#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

KITTI_CLASSES = [
    "Car",
    "Van",
    "Truck",
    "Pedestrian",
    "Person_sitting",
    "Cyclist",
    "Tram",
    "Misc",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(KITTI_CLASSES)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare KITTI object detection data in YOLO format."
    )
    parser.add_argument("--kitti-root", type=Path, default=Path("/data/oxx220000/kitti_object"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/oxx220000/kitti_yolov8_first5000"),
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=5000,
        help="Use first N images (sorted by filename) from KITTI training split.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=500,
        help="Use next M images after subset for validation. Set 0 for no validation split.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of symlink/hardlink.",
    )
    return parser.parse_args()


def safe_link_or_copy(src: Path, dst: Path, copy_images: bool):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        dst.hardlink_to(src)
    except Exception:
        dst.symlink_to(src)


def convert_kitti_label_to_yolo(
    label_path: Path,
    image_size: Tuple[int, int],
) -> List[str]:
    width, height = image_size
    lines: List[str] = []
    with label_path.open("r", encoding="utf-8") as f:
        for raw in f:
            parts = raw.strip().split()
            if len(parts) < 8:
                continue
            cls_name = parts[0]
            if cls_name == "DontCare":
                continue
            if cls_name not in CLASS_TO_ID:
                continue

            left = float(parts[4])
            top = float(parts[5])
            right = float(parts[6])
            bottom = float(parts[7])

            x_center = ((left + right) / 2.0) / width
            y_center = ((top + bottom) / 2.0) / height
            box_w = (right - left) / width
            box_h = (bottom - top) / height

            # Clamp to [0, 1]
            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            box_w = min(max(box_w, 0.0), 1.0)
            box_h = min(max(box_h, 0.0), 1.0)

            class_id = CLASS_TO_ID[cls_name]
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
    return lines


def write_split(
    image_ids: List[str],
    src_image_dir: Path,
    src_label_dir: Path,
    dst_image_dir: Path,
    dst_label_dir: Path,
    copy_images: bool,
):
    dst_image_dir.mkdir(parents=True, exist_ok=True)
    dst_label_dir.mkdir(parents=True, exist_ok=True)

    for image_id in image_ids:
        src_image = src_image_dir / f"{image_id}.png"
        src_label = src_label_dir / f"{image_id}.txt"
        if not src_image.exists():
            continue

        with Image.open(src_image) as img:
            width, height = img.size

        yolo_lines = convert_kitti_label_to_yolo(src_label, (width, height))
        if len(yolo_lines) == 0:
            # Keep image without labels for background negatives.
            yolo_lines = []

        dst_image = dst_image_dir / src_image.name
        safe_link_or_copy(src_image, dst_image, copy_images=copy_images)
        dst_label = dst_label_dir / f"{image_id}.txt"
        with dst_label.open("w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))
            if yolo_lines:
                f.write("\n")


def write_dataset_yaml(output_root: Path, has_val: bool):
    yaml_path = output_root / "kitti_first5000.yaml"
    names = ", ".join([f"'{name}'" for name in KITTI_CLASSES])
    val_line = "val: images/val" if has_val else "val: images/train"
    content = "\n".join(
        [
            f"path: {output_root}",
            "train: images/train",
            val_line,
            f"names: [{names}]",
            "",
        ]
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def main():
    args = parse_args()

    src_image_dir = args.kitti_root / "training" / "image_2"
    src_label_dir = args.kitti_root / "training" / "label_2"
    if not src_image_dir.exists() or not src_label_dir.exists():
        raise FileNotFoundError(
            f"KITTI training folders not found under {args.kitti_root}. "
            "Expected training/image_2 and training/label_2."
        )

    image_ids = sorted([p.stem for p in src_image_dir.glob("*.png")])
    if len(image_ids) == 0:
        raise RuntimeError(f"No images found in {src_image_dir}")

    subset_size = min(args.subset_size, len(image_ids))
    train_ids = image_ids[:subset_size]
    remaining = image_ids[subset_size:]
    val_size = max(0, min(args.val_size, len(remaining)))
    val_ids = remaining[:val_size]

    images_train = args.output_root / "images" / "train"
    labels_train = args.output_root / "labels" / "train"
    images_val = args.output_root / "images" / "val"
    labels_val = args.output_root / "labels" / "val"

    write_split(
        image_ids=train_ids,
        src_image_dir=src_image_dir,
        src_label_dir=src_label_dir,
        dst_image_dir=images_train,
        dst_label_dir=labels_train,
        copy_images=args.copy_images,
    )

    if len(val_ids) > 0:
        write_split(
            image_ids=val_ids,
            src_image_dir=src_image_dir,
            src_label_dir=src_label_dir,
            dst_image_dir=images_val,
            dst_label_dir=labels_val,
            copy_images=args.copy_images,
        )

    yaml_path = write_dataset_yaml(args.output_root, has_val=len(val_ids) > 0)

    (args.output_root / "train_ids.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    if len(val_ids) > 0:
        (args.output_root / "val_ids.txt").write_text("\n".join(val_ids) + "\n", encoding="utf-8")

    summary: Dict[str, int] = {
        "total_images_in_kitti_train": len(image_ids),
        "subset_size_train": len(train_ids),
        "val_size": len(val_ids),
    }
    print("Prepared KITTI YOLO dataset:")
    for key, value in summary.items():
        print(f"- {key}: {value}")
    print(f"- dataset_yaml: {yaml_path}")
    print(f"- output_root: {args.output_root}")


if __name__ == "__main__":
    main()

