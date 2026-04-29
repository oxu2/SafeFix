#!/usr/bin/env python3
"""Run RTMPose Tiny COCO eval and pose slice analysis.

The MMPose evaluator writes COCO-format keypoint detections as
``<outfile_prefix>.keypoints.json``.  This wrapper keeps that output format so
the existing SafeFix slice-analysis code can compare Qwen3, InternVL3, and BLIP
attribute sources on the same predictions.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.stat_pose_estimation import run_analysis


DEFAULT_CHECKPOINT = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/"
    "rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth"
)

ATTRIBUTE_SOURCES: Dict[str, Dict[str, Path]] = {
    "qwen3": {
        "eval": Path("save/attribute_coco_pose_gemini_qwen3_8B.json"),
        "train": Path("save/attribute_coco_pose_gemini_train20k_qwen3_8B.json"),
    },
    "internvl3": {
        "eval": Path("save/attribute_coco_pose_gemini_internvl3_8B.json"),
        "train": Path("save/attribute_coco_pose_gemini_train20k_internvl3_8B.json"),
    },
    "blip": {
        "eval": Path("save/attribute_coco_pose_gemini_blip_val.json"),
        "train": Path("save/attribute_coco_pose_gemini_train20k_blip.json"),
    },
}


def parse_csv(raw: str) -> List[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("pose_estimation/rtmpose_assets/rtmpose_tiny_coco_local_eval.py"),
        help="MMPose config for RTMPose Tiny COCO evaluation.",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", default="/data/oxx220000/coco")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--analysis-sources",
        default="qwen3,internvl3,blip",
        help="Comma-separated attribute sources to compare.",
    )
    parser.add_argument(
        "--analysis-splits",
        default="val,test",
        help="Comma-separated SafeFix held-out splits to analyze.",
    )
    parser.add_argument("--analysis-min-images", type=int, default=25)
    parser.add_argument("--analysis-top-k", type=int, default=20)
    parser.add_argument("--analysis-acc-diff-threshold", type=float, default=0.02)
    parser.add_argument("--analysis-sort-by", choices=["ap", "ap_delta", "images"], default="ap")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("save/corpus_coco_pose_gemini.json"),
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Only run RTMPose evaluation and save predictions/metrics.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def update_mapping(mapping, **updates) -> None:
    for key, value in updates.items():
        mapping[key] = value


def configure_mmpose(args: argparse.Namespace, outfile_prefix: Path):
    try:
        from mmengine.config import Config
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing MMPose/MMEngine dependencies. Install mmpose, mmdet, and mmcv "
            "in the selected environment before running RTMPose eval."
        ) from exc

    cfg = Config.fromfile(args.config)
    cfg.work_dir = str(args.work_dir)
    cfg.load_from = args.checkpoint
    cfg.randomness = dict(seed=args.seed, deterministic=True)
    cfg.env_cfg = cfg.get(
        "env_cfg",
        dict(cudnn_benchmark=False, mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0)),
    )
    cfg.env_cfg["cudnn_benchmark"] = False

    dataset_updates = dict(
        type="CocoDataset",
        data_root=args.data_root,
        data_mode="topdown",
        ann_file="annotations/person_keypoints_val2017.json",
        data_prefix=dict(img="val2017/"),
        test_mode=True,
    )

    for loader_name in ("val_dataloader", "test_dataloader"):
        if loader_name not in cfg:
            continue
        loader = cfg[loader_name]
        loader["batch_size"] = args.batch_size
        loader["num_workers"] = args.num_workers
        loader["persistent_workers"] = args.num_workers > 0
        update_mapping(loader["dataset"], **dataset_updates)
        loader["dataset"].pop("bbox_file", None)

    evaluator_updates = dict(
        type="CocoMetric",
        ann_file=str(Path(args.data_root) / "annotations/person_keypoints_val2017.json"),
        outfile_prefix=str(outfile_prefix),
    )
    for evaluator_name in ("val_evaluator", "test_evaluator"):
        if evaluator_name in cfg:
            update_mapping(cfg[evaluator_name], **evaluator_updates)
            cfg[evaluator_name].pop("iou_type", None)
            cfg[evaluator_name].pop("use_area", None)

    return cfg


def run_rtmpose_eval(args: argparse.Namespace) -> Path:
    from mmengine.runner import Runner

    args.work_dir.mkdir(parents=True, exist_ok=True)
    outfile_prefix = args.work_dir / "rtmpose_tiny_coco_val"
    cfg = configure_mmpose(args, outfile_prefix)

    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    metrics_path = args.work_dir / "rtmpose_tiny_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    predictions_path = outfile_prefix.with_suffix(".keypoints.json")
    if not predictions_path.exists():
        raise FileNotFoundError(f"MMPose did not produce expected predictions: {predictions_path}")

    print(f"Saved RTMPose metrics to {metrics_path}")
    print(f"Saved RTMPose predictions to {predictions_path}")
    return predictions_path


def analysis_namespace(
    *,
    args: argparse.Namespace,
    source: str,
    split: str,
    predictions_path: Path,
) -> SimpleNamespace:
    source_paths = ATTRIBUTE_SOURCES[source]
    out_dir = args.work_dir / "slice_analysis" / source
    return SimpleNamespace(
        predictions_json=predictions_path,
        attribute_file=source_paths["eval"],
        attribute_split="val_full",
        train_attribute_file=source_paths["train"],
        train_attribute_split="train",
        corpus=args.corpus,
        attributes=None,
        data_root=args.data_root,
        eval_annotation_file=None,
        eval_image_dir=None,
        eval_split=split,
        train_annotation_file=None,
        train_image_dir=None,
        min_images=args.analysis_min_images,
        top_k=args.analysis_top_k,
        sort_by=args.analysis_sort_by,
        output_json=out_dir / f"{split}_slice_stats.json",
        output_csv=out_dir / f"{split}_slice_stats.csv",
        seed=args.seed,
        acc_diff_threshold=args.analysis_acc_diff_threshold,
        distribution_diff_threshold=0.15,
        rare_label_threshold=1.1,
        train_max_samples=0,
    )


def run_attribute_analyses(args: argparse.Namespace, predictions_path: Path) -> None:
    sources = parse_csv(args.analysis_sources)
    splits = parse_csv(args.analysis_splits)
    unknown = [source for source in sources if source not in ATTRIBUTE_SOURCES]
    if unknown:
        raise ValueError(f"Unsupported analysis sources: {unknown}")

    for source in sources:
        for split in splits:
            print(f"Running RTMPose slice analysis: source={source}, split={split}")
            run_analysis(
                analysis_namespace(
                    args=args,
                    source=source,
                    split=split,
                    predictions_path=predictions_path,
                )
            )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    predictions_path = run_rtmpose_eval(args)
    if not args.skip_analysis:
        run_attribute_analyses(args, predictions_path)


if __name__ == "__main__":
    main()
