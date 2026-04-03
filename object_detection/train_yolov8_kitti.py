#!/usr/bin/env python3
import argparse
import datetime
from pathlib import Path
import time


def str2bool(value: str) -> bool:
    value = value.lower().strip()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on KITTI first-5000 subset.")
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=Path("/data/oxx220000/kitti_yolov8_first5000/kitti_first5000.yaml"),
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Pretrained weight path/name used when --pretrained true.",
    )
    parser.add_argument(
        "--model-cfg",
        type=str,
        default="yolov8n.yaml",
        help="Model config used when --pretrained false.",
    )
    parser.add_argument("--pretrained", type=str2bool, default=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=str, default="runs/kitti_yolov8")
    parser.add_argument("--name", type=str, default="first5000")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _format_duration(seconds: float) -> str:
    total = int(max(0, round(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _timestamp_str(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def main():
    overall_start_time = time.time()
    print("Current time:", datetime.datetime.now().strftime("%m%d%H%M%S"))
    print(f"Training start timestamp: {_timestamp_str(overall_start_time)}")

    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required. Install with: pip install ultralytics"
        ) from exc

    model_source = args.weights if args.pretrained else args.model_cfg
    model = YOLO(model_source)
    train_start_time = time.time()
    train_results = model.train(
        data=str(args.data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
        pretrained=args.pretrained,
    )
    train_elapsed = time.time() - train_start_time
    overall_elapsed = time.time() - overall_start_time
    end_timestamp = time.time()
    print(f"Total training time for {args.epochs} epochs: {train_elapsed:.2f} seconds")
    print(f"End-to-end wall time (setup + training): {overall_elapsed:.2f} seconds")
    print(f"Training end timestamp: {_timestamp_str(end_timestamp)}")
    print(f"Training elapsed (HH:MM:SS): {_format_duration(train_elapsed)}")
    print(f"End-to-end elapsed (HH:MM:SS): {_format_duration(overall_elapsed)}")
    print(train_results)


if __name__ == "__main__":
    main()
