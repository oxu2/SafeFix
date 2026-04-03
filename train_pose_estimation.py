#!/usr/bin/env python3
import argparse
import datetime
import json
import random
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from pose_estimation.dataset import build_datasets, collate_fn, default_coco_paths
from pose_estimation.engine import evaluate_coco_keypoints, load_checkpoint, save_checkpoint, train_one_epoch
from pose_estimation.modeling import create_keypointrcnn_model
from pose_estimation.visualize import draw_pose_predictions, image_to_tensor


def _format_duration(seconds: float) -> str:
    total = int(max(0, round(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _timestamp_str(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def parse_args():
    parser = argparse.ArgumentParser(description="COCO pose estimation with Keypoint R-CNN.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train model on COCO keypoints.")
    _add_common_data_args(train_parser)
    train_parser.add_argument("--epochs", type=int, default=12)
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument("--num-workers", type=int, default=4)
    train_parser.add_argument("--lr", type=float, default=0.005)
    train_parser.add_argument("--momentum", type=float, default=0.9)
    train_parser.add_argument("--weight-decay", type=float, default=5e-4)
    train_parser.add_argument("--lr-step-size", type=int, default=8)
    train_parser.add_argument("--lr-gamma", type=float, default=0.1)
    train_parser.add_argument("--hflip-prob", type=float, default=0.5)
    train_parser.add_argument("--eval-every", type=int, default=1)
    train_parser.add_argument("--output-dir", type=Path, default=Path("outputs/pose_keypointrcnn"))
    train_parser.add_argument("--resume", type=Path, default=None)
    train_parser.add_argument("--score-threshold", type=float, default=0.05)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--max-train-samples", type=int, default=0)
    train_parser.add_argument("--pretrained", type=str2bool, default=True)
    train_parser.add_argument("--device", type=str, default="cuda")

    eval_parser = subparsers.add_parser("eval", help="Evaluate model on COCO val keypoints.")
    _add_common_data_args(eval_parser)
    eval_parser.add_argument("--batch-size", type=int, default=2)
    eval_parser.add_argument("--num-workers", type=int, default=4)
    eval_parser.add_argument("--checkpoint", type=Path, required=False)
    eval_parser.add_argument("--score-threshold", type=float, default=0.05)
    eval_parser.add_argument("--output-json", type=Path, default=Path("outputs/pose_eval_results.json"))
    eval_parser.add_argument("--pretrained", type=str2bool, default=True)
    eval_parser.add_argument("--device", type=str, default="cuda")

    infer_parser = subparsers.add_parser("infer", help="Run inference on one image and draw pose.")
    infer_parser.add_argument("--image", type=Path, required=True)
    infer_parser.add_argument("--checkpoint", type=Path, default=None)
    infer_parser.add_argument("--output", type=Path, default=Path("outputs/pose_infer.jpg"))
    infer_parser.add_argument("--score-threshold", type=float, default=0.5)
    infer_parser.add_argument("--keypoint-threshold", type=float, default=0.1)
    infer_parser.add_argument("--pretrained", type=str2bool, default=True)
    infer_parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def _add_common_data_args(parser):
    parser.add_argument("--data-root", type=str, default="/data/oxx220000/coco")
    parser.add_argument("--train-images", type=str, default=None)
    parser.add_argument("--val-images", type=str, default=None)
    parser.add_argument("--train-annotations", type=str, default=None)
    parser.add_argument("--val-annotations", type=str, default=None)


def str2bool(value: str) -> bool:
    value = value.lower().strip()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def build_loaders(args) -> Dict[str, DataLoader]:
    train_dataset, val_dataset = build_datasets(
        data_root=args.data_root,
        train_images=args.train_images,
        val_images=args.val_images,
        train_annotations=args.train_annotations,
        val_annotations=args.val_annotations,
        hflip_prob=getattr(args, "hflip_prob", 0.5),
    )

    if getattr(args, "max_train_samples", 0) and args.max_train_samples > 0:
        train_dataset.image_ids = train_dataset.image_ids[: args.max_train_samples]

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return {"train": train_loader, "val": val_loader}


def run_train(args):
    overall_start_time = time.time()
    print("Current time:", datetime.datetime.now().strftime("%m%d%H%M%S"))
    print(f"Training start timestamp: {_timestamp_str(overall_start_time)}")

    set_seed(args.seed)
    device = resolve_device(args.device)
    loaders = build_loaders(args)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    model = create_keypointrcnn_model(num_keypoints=17, pretrained=args.pretrained).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma,
    )

    start_epoch = 0
    if args.resume is not None:
        start_epoch = load_checkpoint(
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            map_location="cpu",
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_ap = -1.0
    history = []
    total_train_time = 0.0
    total_eval_time = 0.0
    train_loop_start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        metrics = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            print_freq=50,
        )
        lr_scheduler.step()
        epoch_train_time = float(metrics.get("epoch_time_sec", time.time() - epoch_start_time))
        total_train_time += epoch_train_time

        eval_metrics = {}
        if (epoch + 1) % args.eval_every == 0:
            eval_start_time = time.time()
            eval_metrics = evaluate_coco_keypoints(
                model=model,
                data_loader=val_loader,
                coco_gt=val_loader.dataset.coco,
                device=device,
                score_threshold=args.score_threshold,
                result_json_path=args.output_dir / f"val_predictions_epoch_{epoch:03d}.json",
            )
            eval_duration = time.time() - eval_start_time
            total_eval_time += eval_duration
            eval_metrics["eval_time_sec"] = eval_duration
            if eval_metrics["AP"] > best_ap:
                best_ap = eval_metrics["AP"]
                save_checkpoint(
                    checkpoint_path=args.output_dir / "best.pth",
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    epoch=epoch,
                    extra={"best_ap": best_ap},
                )

        save_checkpoint(
            checkpoint_path=args.output_dir / "last.pth",
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
            extra={"best_ap": best_ap},
        )

        row = {"epoch": epoch, **metrics, **eval_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

    with (args.output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    end_timestamp = time.time()
    train_loop_elapsed = end_timestamp - train_loop_start_time
    overall_elapsed_time = end_timestamp - overall_start_time
    print(f"Total training time for {args.epochs} epochs: {train_loop_elapsed:.2f} seconds")
    print(f"Total training-only epoch time: {total_train_time:.2f} seconds")
    print(f"Total evaluation time: {total_eval_time:.2f} seconds")
    print(
        "End-to-end wall time (data prep + training + evaluation + setup): "
        f"{overall_elapsed_time:.2f} seconds"
    )
    print(f"Training end timestamp: {_timestamp_str(end_timestamp)}")
    print(f"Training elapsed (HH:MM:SS): {_format_duration(train_loop_elapsed)}")
    print(f"End-to-end elapsed (HH:MM:SS): {_format_duration(overall_elapsed_time)}")
    print(f"Training complete. Artifacts: {args.output_dir}")


def run_eval(args):
    device = resolve_device(args.device)
    paths = default_coco_paths(args.data_root)
    _, val_dataset = build_datasets(
        data_root=args.data_root,
        train_images=args.train_images,
        val_images=args.val_images,
        train_annotations=args.train_annotations,
        val_annotations=args.val_annotations,
        hflip_prob=0.0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = create_keypointrcnn_model(num_keypoints=17, pretrained=args.pretrained).to(device)
    if args.checkpoint is not None:
        load_checkpoint(
            checkpoint_path=args.checkpoint,
            model=model,
            optimizer=None,
            lr_scheduler=None,
            map_location="cpu",
        )
    else:
        print("No checkpoint specified; evaluating default pretrained model.")

    print(
        f"Evaluating on val set: images={args.val_images or paths['val_images']}, "
        f"annotations={args.val_annotations or paths['val_annotations']}"
    )
    metrics = evaluate_coco_keypoints(
        model=model,
        data_loader=val_loader,
        coco_gt=val_loader.dataset.coco,
        device=device,
        score_threshold=args.score_threshold,
        result_json_path=args.output_json,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


@torch.no_grad()
def run_infer(args):
    device = resolve_device(args.device)
    model = create_keypointrcnn_model(num_keypoints=17, pretrained=args.pretrained).to(device)
    if args.checkpoint is not None:
        load_checkpoint(
            checkpoint_path=args.checkpoint,
            model=model,
            optimizer=None,
            lr_scheduler=None,
            map_location="cpu",
        )
    model.eval()

    image = Image.open(args.image).convert("RGB")
    image_tensor = image_to_tensor(image).to(device)

    predictions = model([image_tensor])[0]
    canvas = draw_pose_predictions(
        image=image,
        prediction=predictions,
        score_threshold=args.score_threshold,
        keypoint_threshold=args.keypoint_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"Saved inference visualization to {args.output}")


def main():
    args = parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "eval":
        run_eval(args)
    elif args.command == "infer":
        run_infer(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
