#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader

from pose_estimation.dataset import (
    build_datasets,
    build_test_dataset,
    build_train_val_test_datasets,
    collate_fn,
    default_coco_paths,
)
from pose_estimation.engine import (
    evaluate_coco_keypoints,
    load_checkpoint,
    load_model_checkpoint,
    read_checkpoint_metadata,
    save_model_checkpoint,
    save_checkpoint,
    train_one_epoch,
)
from pose_estimation.modeling import create_keypointrcnn_model
from pose_estimation.visualize import draw_pose_predictions, image_to_tensor
from utils.stat_pose_estimation import run_analysis as run_pose_slice_analysis


REPO_ROOT = Path(__file__).resolve().parent
POSE_LOG_ROOT = Path("log_pose")
ASSIGNER_MODEL_MAP = {
    "qwen3": "Qwen/Qwen3-VL-8B-Instruct",
    "internvl3": "OpenGVLab/InternVL3-8B",
}
ANALYSIS_SOURCE_DEFAULTS = {
    "internvl3": {
        "attribute_file": Path("save/attribute_coco_pose_gemini_internvl3_8B.json"),
        "attribute_split": "val_full",
        "train_attribute_file": Path("save/attribute_coco_pose_gemini_train20k_internvl3_8B.json"),
        "train_attribute_split": "train",
    },
    "qwen3": {
        "attribute_file": Path("save/attribute_coco_pose_gemini_qwen3_8B.json"),
        "attribute_split": "val_full",
        "train_attribute_file": Path("save/attribute_coco_pose_gemini_train20k_qwen3_8B.json"),
        "train_attribute_split": "train",
    },
    "blip": {
        "attribute_file": Path("save/attribute_coco_pose_gemini_blip_val.json"),
        "attribute_split": "val_full",
        "train_attribute_file": Path("save/attribute_coco_pose_gemini_train20k_blip.json"),
        "train_attribute_split": "train",
    },
}


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
    train_parser.add_argument("--epochs", type=int, default=2)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--num-workers", type=int, default=4)
    train_parser.add_argument("--lr", type=float, default=0.005)
    train_parser.add_argument("--momentum", type=float, default=0.9)
    train_parser.add_argument("--weight-decay", type=float, default=5e-4)
    train_parser.add_argument("--lr-step-size", type=int, default=8)
    train_parser.add_argument("--lr-gamma", type=float, default=0.1)
    train_parser.add_argument("--hflip-prob", type=float, default=0.5)
    train_parser.add_argument("--eval-every", type=int, default=1)
    train_parser.add_argument("--output-dir", type=Path, default=POSE_LOG_ROOT / "pose_keypointrcnn")
    train_parser.add_argument("--resume", type=Path, default=None)
    train_parser.add_argument(
        "--resume-model",
        type=Path,
        default=None,
        help=(
            "Load only model weights from a checkpoint and continue training with a fresh "
            "optimizer / scheduler. If the checkpoint stores an epoch, training resumes "
            "from the next epoch index."
        ),
    )
    train_parser.add_argument("--score-threshold", type=float, default=0.05)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--max-train-samples", type=int, default=20000)
    train_parser.add_argument("--pretrained", type=str2bool, default=True)
    train_parser.add_argument("--device", type=str, default="cuda")
    train_parser.add_argument("--analyze-after-train", type=str2bool, default=True)
    train_parser.add_argument(
        "--assigner",
        "--analysis-source",
        dest="assigner",
        choices=["internvl3", "qwen3", "blip"],
        default="internvl3",
        help="Select the precomputed attribute source for post-train analysis.",
    )
    train_parser.add_argument("--analysis-attribute-file", type=Path, default=None)
    train_parser.add_argument(
        "--analysis-attribute-split",
        choices=["train", "val_full", "val", "test", "all"],
        default="val_full",
    )
    train_parser.add_argument("--analysis-train-attribute-file", type=Path, default=None)
    train_parser.add_argument(
        "--analysis-train-attribute-split",
        choices=["train", "val_full", "val", "test", "all"],
        default="train",
    )
    train_parser.add_argument(
        "--analysis-corpus",
        type=Path,
        default=Path("save/corpus_coco_pose_gemini.json"),
    )
    train_parser.add_argument("--analysis-attributes", type=str, default=None)
    train_parser.add_argument("--analysis-min-images", type=int, default=25)
    train_parser.add_argument("--analysis-top-k", type=int, default=20)
    train_parser.add_argument(
        "--analysis-sort-by",
        choices=["ap", "ap_delta", "images"],
        default="ap",
    )
    train_parser.add_argument("--analysis-acc-diff-threshold", type=float, default=0.005)
    train_parser.add_argument("--analysis-distribution-diff-threshold", type=float, default=0.15)
    train_parser.add_argument("--analysis-rare-label-threshold", type=float, default=1.1)
    train_parser.add_argument("--assign-after-train", type=str2bool, default=False)
    train_parser.add_argument(
        "--assigners",
        type=str,
        default="blip",
        help="Comma-separated assigners to run after training. Supported: blip,qwen3,internvl3",
    )
    train_parser.add_argument(
        "--assigner-splits",
        type=str,
        default="train",
        help="Comma-separated COCO splits for assigners. Supported: val,train,all",
    )
    train_parser.add_argument("--assigner-corpus", type=Path, default=None)
    train_parser.add_argument("--assigner-output-root", type=Path, default=None)
    train_parser.add_argument("--assigner-save-every", type=int, default=500)
    train_parser.add_argument("--assigner-max-memory-per-gpu", type=float, default=23.0)
    train_parser.add_argument("--assigner-verbose", type=str2bool, default=False)
    train_parser.add_argument(
        "--save-resume-checkpoint",
        type=str2bool,
        default=False,
        help=(
            "Also save a full optimizer/scheduler resume checkpoint as last_resume.pth. "
            "Default is False to reduce disk usage."
        ),
    )
    train_parser.add_argument(
        "--save-last-checkpoint",
        type=str2bool,
        default=False,
        help=(
            "Save a compact last.pth model checkpoint after each epoch. "
            "Default is False because best.pth is usually sufficient for this pose workflow."
        ),
    )

    eval_parser = subparsers.add_parser("eval", help="Evaluate model on COCO val keypoints.")
    _add_common_data_args(eval_parser)
    eval_parser.add_argument("--batch-size", type=int, default=2)
    eval_parser.add_argument("--num-workers", type=int, default=4)
    eval_parser.add_argument("--checkpoint", type=Path, required=False)
    eval_parser.add_argument("--score-threshold", type=float, default=0.05)
    eval_parser.add_argument("--output-json", type=Path, default=POSE_LOG_ROOT / "pose_eval_results.json")
    eval_parser.add_argument("--pretrained", type=str2bool, default=True)
    eval_parser.add_argument("--device", type=str, default="cuda")
    eval_parser.add_argument("--seed", type=int, default=42)

    test_parser = subparsers.add_parser(
        "test",
        help="Final test evaluation on the held-out half of valid val2017 images.",
    )
    _add_common_data_args(test_parser)
    test_parser.add_argument("--batch-size", type=int, default=2)
    test_parser.add_argument("--num-workers", type=int, default=4)
    test_parser.add_argument("--checkpoint", type=Path, required=True,
                             help="Path to the model checkpoint (e.g. best.pth).")
    test_parser.add_argument("--score-threshold", type=float, default=0.05)
    test_parser.add_argument("--output-json", type=Path, default=POSE_LOG_ROOT / "pose_test_results.json")
    test_parser.add_argument("--pretrained", type=str2bool, default=True)
    test_parser.add_argument("--device", type=str, default="cuda")
    test_parser.add_argument("--seed", type=int, default=42)

    infer_parser = subparsers.add_parser("infer", help="Run inference on one image and draw pose.")
    infer_parser.add_argument("--image", type=Path, required=True)
    infer_parser.add_argument("--checkpoint", type=Path, default=None)
    infer_parser.add_argument("--output", type=Path, default=POSE_LOG_ROOT / "pose_infer.jpg")
    infer_parser.add_argument("--score-threshold", type=float, default=0.5)
    infer_parser.add_argument("--keypoint-threshold", type=float, default=0.1)
    infer_parser.add_argument("--pretrained", type=str2bool, default=True)
    infer_parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def _add_common_data_args(parser):
    parser.add_argument("--data-root", type=str, default="/data/oxx220000/coco")
    parser.add_argument("--train-images", type=str, default=None)
    parser.add_argument("--val-images", type=str, default=None)
    parser.add_argument("--test-images", type=str, default=None)
    parser.add_argument("--train-annotations", type=str, default=None)
    parser.add_argument("--val-annotations", type=str, default=None)
    parser.add_argument("--test-annotations", type=str, default=None)


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
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_loader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def build_loaders(args) -> Dict[str, DataLoader]:
    train_dataset, val_dataset, test_dataset = build_train_val_test_datasets(
        data_root=args.data_root,
        train_images=args.train_images,
        val_images=args.val_images,
        train_annotations=args.train_annotations,
        val_annotations=args.val_annotations,
        hflip_prob=getattr(args, "hflip_prob", 0.5),
        seed=args.seed,
        max_train_samples=getattr(args, "max_train_samples", 0),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=build_loader_generator(args.seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=build_loader_generator(args.seed + 1),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=build_loader_generator(args.seed + 2),
    )

    print(
        f"Dataset sizes after split: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
    )
    return {"train": train_loader, "val": val_loader, "test": test_loader}


def parse_csv_choices(raw: str, allowed: Sequence[str], flag_name: str) -> List[str]:
    values = [value.strip().lower() for value in raw.split(",") if value.strip()]
    invalid = [value for value in values if value not in allowed]
    if invalid:
        raise ValueError(f"Invalid values for {flag_name}: {invalid}. Allowed: {sorted(allowed)}")
    deduped: List[str] = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def resolve_assigner_corpus(args) -> Path:
    if args.assigner_corpus is not None:
        return args.assigner_corpus
    if args.analysis_corpus is not None:
        return args.analysis_corpus
    return Path("save/corpus_coco_pose_gemini.json")


def resolve_assigner_output_root(args) -> Path:
    return args.assigner_output_root or (args.output_dir / "assigners")


def resolve_analysis_source_config(
    args,
    assigner_artifacts: Optional[Dict[str, Dict[str, Dict[str, Path]]]] = None,
) -> Dict[str, object]:
    if args.analysis_attribute_file is not None:
        return {
            "tag": "manual",
            "attribute_file": args.analysis_attribute_file,
            "attribute_split": args.analysis_attribute_split,
            "train_attribute_file": args.analysis_train_attribute_file,
            "train_attribute_split": args.analysis_train_attribute_split,
        }

    if assigner_artifacts and args.assigner in assigner_artifacts:
        outputs = assigner_artifacts[args.assigner]
        eval_source = outputs.get("val") or outputs.get("all")
        train_source = outputs.get("train") or outputs.get("all")
        if eval_source is not None:
            return {
                "tag": args.assigner,
                "attribute_file": eval_source["json"],
                "attribute_split": "val_full" if "val" in outputs else "all",
                "train_attribute_file": train_source["json"] if train_source is not None else None,
                "train_attribute_split": "train" if "train" in outputs else ("all" if "all" in outputs else "train"),
            }

    config = ANALYSIS_SOURCE_DEFAULTS[args.assigner]
    return {
        "tag": args.assigner,
        "attribute_file": config["attribute_file"],
        "attribute_split": config["attribute_split"],
        "train_attribute_file": config["train_attribute_file"],
        "train_attribute_split": config["train_attribute_split"],
    }


def run_command(command: Sequence[str]) -> None:
    print("Running command:")
    print(" ".join(str(part) for part in command))
    subprocess.run(
        [str(part) for part in command],
        check=True,
        cwd=str(REPO_ROOT),
    )


def run_single_assigner(args, *, assigner: str, split: str, corpus_path: Path) -> Dict[str, Path]:
    output_root = resolve_assigner_output_root(args)
    artifact_dir = output_root / assigner / split
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if assigner == "blip":
        command: List[object] = [
            sys.executable,
            "-u",
            "utils/blip.py",
            "--dataset",
            "coco_pose",
            "--coco_root",
            args.data_root,
            "--split",
            split,
            "--corpus",
            corpus_path,
            "--with_attribute_selection",
            "--output_labels",
            artifact_dir / "labels.npy",
            "--output_attributes",
            artifact_dir / "attributes.json",
            "--csv_output",
            artifact_dir / "attributes.csv",
            "--text_labels_output",
            artifact_dir / "text_labels.npy",
            "--train_idx_output",
            artifact_dir / "train_idx.npy",
            "--val_idx_output",
            artifact_dir / "val_idx.npy",
        ]
        if split == "train" and args.max_train_samples > 0:
            command.extend(["--stop_at", str(args.max_train_samples)])
    else:
        model_name = ASSIGNER_MODEL_MAP[assigner]
        command = [
            sys.executable,
            "-u",
            "utils/assign_attribute.py",
            "--dataset",
            "coco_pose",
            "--coco_root",
            args.data_root,
            "--split",
            split,
            "--corpus",
            corpus_path,
            "--model",
            model_name,
            "--output",
            artifact_dir / "attributes.json",
            "--csv-output",
            artifact_dir / "attributes.csv",
            "--partial",
            artifact_dir / "attributes_partial.json",
            "--save_every",
            str(args.assigner_save_every),
            "--max_memory_per_gpu",
            str(args.assigner_max_memory_per_gpu),
        ]
        if split == "train" and args.max_train_samples > 0:
            command.extend(["--stop_at", str(args.max_train_samples)])
        if args.assigner_verbose:
            command.append("--verbose")

    run_command(command)
    artifacts = {
        "json": artifact_dir / "attributes.json",
        "csv": artifact_dir / "attributes.csv",
        "dir": artifact_dir,
    }
    if assigner != "blip":
        artifacts["partial"] = artifact_dir / "attributes_partial.json"
    return artifacts


def maybe_run_assigners_after_train(args) -> Dict[str, Dict[str, Dict[str, Path]]]:
    if not args.assign_after_train:
        return {}

    assigners = parse_csv_choices(
        args.assigners,
        allowed={"blip", "qwen3", "internvl3"},
        flag_name="--assigners",
    )
    splits = parse_csv_choices(
        args.assigner_splits,
        allowed={"train", "val", "all"},
        flag_name="--assigner-splits",
    )
    corpus_path = resolve_assigner_corpus(args)

    artifacts: Dict[str, Dict[str, Dict[str, Path]]] = {}
    for assigner in assigners:
        artifacts[assigner] = {}
        for split in splits:
            print(f"Running assigner '{assigner}' on split '{split}'...")
            artifacts[assigner][split] = run_single_assigner(
                args,
                assigner=assigner,
                split=split,
                corpus_path=corpus_path,
            )
    return artifacts


def build_analysis_args(
    *,
    args,
    predictions_json: Path,
    eval_split: str,
    output_json: Path,
    output_csv: Path,
    attribute_file: Path,
    attribute_split: str,
    train_attribute_file: Optional[Path],
    train_attribute_split: str,
) -> argparse.Namespace:
    paths = default_coco_paths(args.data_root)
    if eval_split == "val":
        eval_annotation_file = args.val_annotations or paths["val_annotations"]
        eval_image_dir = args.val_images or paths["val_images"]
    elif eval_split == "test":
        eval_annotation_file = args.test_annotations or paths["test_annotations"]
        eval_image_dir = args.test_images or paths["test_images"]
    else:
        raise ValueError(f"Unsupported analysis split: {eval_split}")

    return argparse.Namespace(
        predictions_json=predictions_json,
        attribute_file=attribute_file,
        attribute_split=attribute_split,
        train_attribute_file=train_attribute_file,
        train_attribute_split=train_attribute_split,
        corpus=args.analysis_corpus or resolve_assigner_corpus(args),
        attributes=args.analysis_attributes,
        data_root=args.data_root,
        eval_annotation_file=eval_annotation_file,
        eval_image_dir=eval_image_dir,
        eval_split=eval_split,
        train_annotation_file=args.train_annotations,
        train_image_dir=args.train_images,
        min_images=args.analysis_min_images,
        top_k=args.analysis_top_k,
        sort_by=args.analysis_sort_by,
        output_json=output_json,
        output_csv=output_csv,
        seed=args.seed,
        acc_diff_threshold=args.analysis_acc_diff_threshold,
        distribution_diff_threshold=args.analysis_distribution_diff_threshold,
        rare_label_threshold=args.analysis_rare_label_threshold,
        train_max_samples=args.max_train_samples,
    )


def maybe_run_post_train_analysis(
    args,
    *,
    val_predictions_path: Optional[Path],
    test_predictions_path: Optional[Path],
    assigner_artifacts: Optional[Dict[str, Dict[str, Dict[str, Path]]]] = None,
) -> None:
    if not args.analyze_after_train:
        return

    analysis_dir = args.output_dir / "slice_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    source = resolve_analysis_source_config(args, assigner_artifacts=assigner_artifacts)
    source_dir = analysis_dir / source["tag"]
    source_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    if val_predictions_path is not None and val_predictions_path.exists():
        tasks.append(
            (
                "val",
                build_analysis_args(
                    args=args,
                    predictions_json=val_predictions_path,
                    eval_split="val",
                    output_json=source_dir / "val_slice_stats.json",
                    output_csv=source_dir / "val_slice_stats.csv",
                    attribute_file=source["attribute_file"],
                    attribute_split=source["attribute_split"],
                    train_attribute_file=source["train_attribute_file"],
                    train_attribute_split=source["train_attribute_split"],
                ),
            )
        )
    if test_predictions_path is not None and test_predictions_path.exists():
        tasks.append(
            (
                "test",
                build_analysis_args(
                    args=args,
                    predictions_json=test_predictions_path,
                    eval_split="test",
                    output_json=source_dir / "test_slice_stats.json",
                    output_csv=source_dir / "test_slice_stats.csv",
                    attribute_file=source["attribute_file"],
                    attribute_split=source["attribute_split"],
                    train_attribute_file=source["train_attribute_file"],
                    train_attribute_split=source["train_attribute_split"],
                ),
            )
        )

    for split_name, analysis_args in tasks:
        print(f"Running post-train slice analysis on {split_name} split using '{source['tag']}' attributes...")
        try:
            run_pose_slice_analysis(analysis_args)
        except Exception as exc:
            print(f"Post-train slice analysis failed on {split_name} using '{source['tag']}': {exc}")


def run_train(args):
    overall_start_time = time.time()
    print("Current time:", datetime.datetime.now().strftime("%m%d%H%M%S"))
    print(f"Training start timestamp: {_timestamp_str(overall_start_time)}")

    if args.resume is not None and args.resume_model is not None:
        raise ValueError("Use only one of --resume or --resume-model.")

    set_seed(args.seed)
    print(
        f"Deterministic settings: seed={args.seed}, cudnn.deterministic={cudnn.deterministic}, "
        f"cudnn.benchmark={cudnn.benchmark}, CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}"
    )
    device = resolve_device(args.device)
    loaders = build_loaders(args)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

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
    best_ap = -1.0
    if args.resume is not None:
        resume_metadata = read_checkpoint_metadata(
            checkpoint_path=args.resume,
            map_location="cpu",
        )
        if not resume_metadata.get("has_optimizer", False):
            raise ValueError(
                f"{args.resume} does not contain optimizer state. "
                "Use --resume-model for model-only continuation, or rerun training with "
                "--save-resume-checkpoint true and resume from last_resume.pth."
            )
        start_epoch = load_checkpoint(
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            map_location="cpu",
        )
        if resume_metadata.get("best_ap") is not None:
            best_ap = float(resume_metadata["best_ap"])
    elif args.resume_model is not None:
        resume_metadata = load_model_checkpoint(
            checkpoint_path=args.resume_model,
            model=model,
            map_location="cpu",
        )
        if resume_metadata.get("epoch") is not None:
            start_epoch = int(resume_metadata["epoch"]) + 1
        if resume_metadata.get("best_ap") is not None:
            best_ap = float(resume_metadata["best_ap"])

        print(
            "Using fresh optimizer / scheduler state after --resume-model. "
            f"Training will start from epoch index {start_epoch} with the current "
            f"optimizer hyperparameters (lr={args.lr}, momentum={args.momentum}, "
            f"weight_decay={args.weight_decay}, step_size={args.lr_step_size}, gamma={args.lr_gamma})."
        )

    if start_epoch >= args.epochs:
        print(
            f"Warning: start_epoch={start_epoch} is already >= args.epochs={args.epochs}. "
            "No additional training epochs will run unless you increase --epochs."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    total_train_time = 0.0
    total_eval_time = 0.0
    train_loop_start_time = time.time()
    best_checkpoint_path = args.output_dir / "best.pth"
    last_checkpoint_path = args.output_dir / "last.pth"
    last_resume_checkpoint_path = args.output_dir / "last_resume.pth"
    assigner_artifacts: Dict[str, Dict[str, Dict[str, Path]]] = {}

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
                save_model_checkpoint(
                    checkpoint_path=best_checkpoint_path,
                    model=model,
                    epoch=epoch,
                    extra={"best_ap": best_ap},
                )

        if args.save_last_checkpoint:
            save_model_checkpoint(
                checkpoint_path=last_checkpoint_path,
                model=model,
                epoch=epoch,
                extra={"best_ap": best_ap},
            )
        if args.save_resume_checkpoint:
            save_checkpoint(
                checkpoint_path=last_resume_checkpoint_path,
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

    final_eval_checkpoint: Optional[Path] = None
    if best_checkpoint_path.exists():
        final_eval_checkpoint = best_checkpoint_path
    elif last_checkpoint_path.exists():
        final_eval_checkpoint = last_checkpoint_path

    best_val_predictions_path: Optional[Path] = None
    test_predictions_path: Optional[Path] = None
    if final_eval_checkpoint is not None:
        load_model_checkpoint(
            checkpoint_path=final_eval_checkpoint,
            model=model,
            map_location="cpu",
        )
        best_val_predictions_path = args.output_dir / "val_predictions_best.json"
        best_val_metrics = evaluate_coco_keypoints(
            model=model,
            data_loader=val_loader,
            coco_gt=val_loader.dataset.coco,
            device=device,
            score_threshold=args.score_threshold,
            result_json_path=best_val_predictions_path,
        )
        with (args.output_dir / "val_metrics_best.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "checkpoint": str(final_eval_checkpoint),
                    "metrics": best_val_metrics,
                },
                f,
                indent=2,
            )
        print("Held-out val metrics (best checkpoint):")
        print(json.dumps(best_val_metrics, ensure_ascii=False, indent=2))

        test_predictions_path = args.output_dir / "test_predictions.json"
        test_metrics = evaluate_coco_keypoints(
            model=model,
            data_loader=test_loader,
            coco_gt=test_loader.dataset.coco,
            device=device,
            score_threshold=args.score_threshold,
            result_json_path=test_predictions_path,
        )
        with (args.output_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "checkpoint": str(final_eval_checkpoint),
                    "metrics": test_metrics,
                },
                f,
                indent=2,
            )
        print("Held-out test metrics:")
        print(json.dumps(test_metrics, ensure_ascii=False, indent=2))
        assigner_artifacts = maybe_run_assigners_after_train(args)
        maybe_run_post_train_analysis(
            args,
            val_predictions_path=best_val_predictions_path,
            test_predictions_path=test_predictions_path,
            assigner_artifacts=assigner_artifacts,
        )
    else:
        print(
            "No saved best/last model checkpoint was found after training. "
            "Skipping the post-train reload step and using the in-memory final model for evaluation."
        )
        best_val_predictions_path = args.output_dir / "val_predictions_best.json"
        best_val_metrics = evaluate_coco_keypoints(
            model=model,
            data_loader=val_loader,
            coco_gt=val_loader.dataset.coco,
            device=device,
            score_threshold=args.score_threshold,
            result_json_path=best_val_predictions_path,
        )
        with (args.output_dir / "val_metrics_best.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "checkpoint": None,
                    "metrics": best_val_metrics,
                },
                f,
                indent=2,
            )
        print("Held-out val metrics (final in-memory model):")
        print(json.dumps(best_val_metrics, ensure_ascii=False, indent=2))

        test_predictions_path = args.output_dir / "test_predictions.json"
        test_metrics = evaluate_coco_keypoints(
            model=model,
            data_loader=test_loader,
            coco_gt=test_loader.dataset.coco,
            device=device,
            score_threshold=args.score_threshold,
            result_json_path=test_predictions_path,
        )
        with (args.output_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "checkpoint": None,
                    "metrics": test_metrics,
                },
                f,
                indent=2,
            )
        print("Held-out test metrics:")
        print(json.dumps(test_metrics, ensure_ascii=False, indent=2))
        assigner_artifacts = maybe_run_assigners_after_train(args)
        maybe_run_post_train_analysis(
            args,
            val_predictions_path=best_val_predictions_path,
            test_predictions_path=test_predictions_path,
            assigner_artifacts=assigner_artifacts,
        )

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
        seed=args.seed,
        max_train_samples=0,
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
        f"Evaluating on held-out val split: images={args.val_images or paths['val_images']}, "
        f"annotations={args.val_annotations or paths['val_annotations']}, seed={args.seed}, "
        f"num_images={len(val_dataset)}"
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


def run_test(args):
    """Final test evaluation on the held-out half of valid val2017 images."""
    device = resolve_device(args.device)
    paths = default_coco_paths(args.data_root)

    test_dataset = build_test_dataset(
        data_root=args.data_root,
        test_images=args.test_images,
        test_annotations=args.test_annotations,
        seed=args.seed,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = create_keypointrcnn_model(num_keypoints=17, pretrained=args.pretrained).to(device)
    load_checkpoint(
        checkpoint_path=args.checkpoint,
        model=model,
        optimizer=None,
        lr_scheduler=None,
        map_location="cpu",
    )

    test_img_dir = args.test_images or paths["test_images"]
    test_ann_file = args.test_annotations or paths["test_annotations"]
    print(f"Testing on held-out test split: images={test_img_dir}, annotations={test_ann_file}, seed={args.seed}")
    print(f"Test set size: {len(test_dataset)} images")

    metrics = evaluate_coco_keypoints(
        model=model,
        data_loader=test_loader,
        coco_gt=test_loader.dataset.coco,
        device=device,
        score_threshold=args.score_threshold,
        result_json_path=args.output_json,
    )

    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    print(f"  AP:   {metrics['AP']:.4f}")
    print(f"  AP50: {metrics['AP50']:.4f}")
    print(f"  AP75: {metrics['AP75']:.4f}")
    print(f"  AR:   {metrics['AR']:.4f}")
    print("=" * 50)

    # Save test results to JSON
    result_path = args.output_json
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "test_images": test_img_dir,
                "test_annotations": test_ann_file,
                "num_test_images": len(test_dataset),
                "metrics": metrics,
            },
            f,
            indent=2,
        )
    print(f"Test results saved to {result_path}")


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
    elif args.command == "test":
        run_test(args)
    elif args.command == "infer":
        run_infer(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
