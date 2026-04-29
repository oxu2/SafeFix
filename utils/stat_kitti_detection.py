#!/usr/bin/env python3
"""
Slice-level statistics for KITTI YOLOv8 object-detection results.

This script evaluates a trained YOLOv8 model on attribute-defined image slices.
It mirrors the analysis workflow of utils/stat_pose_estimation.py but targets
KITTI object detection with YOLO-format labels and the Ultralytics API.

The approach:
  1. Run YOLOv8 inference on the eval split and cache predictions.
  2. For each (attribute, value) slice, select images and compute mAP by
     filtering the cached per-image match results.
  3. Output JSON and CSV with per-slice metrics and failure flags.

Usage:
    python utils/stat_kitti_detection.py \
        --model-weights runs/kitti_yolov8/.../weights/best.pt \
        --attribute-file save/attribute_kitti_qwen3_8B.json save/attribute_kitti_blip.json \
        --split train
"""

import argparse
import csv
import glob
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_KITTI_ROOT = "/data/oxx220000/kitti_yolov8_first5000"
DEFAULT_CORPUS = "save/corpus_kitti_gemini_clean.json"
CLASS_NAMES = ["Car", "Pedestrian"]
IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute slice-level object-detection metrics for KITTI using attribute annotations."
    )
    parser.add_argument("--model-weights", type=Path, required=True,
                        help="Path to the trained YOLOv8 .pt weights file.")
    parser.add_argument("--attribute-file", type=Path, nargs="+", required=True,
                        help="One or more attribute JSON files.")
    parser.add_argument("--kitti-root", type=str, default=DEFAULT_KITTI_ROOT)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--train-attribute-file", type=Path, default=None,
                        help="Optional attribute JSON for train-distribution comparison.")
    parser.add_argument("--corpus", type=Path, default=Path(DEFAULT_CORPUS))
    parser.add_argument("--output-dir", type=Path, default=Path("log_kitti/kitti_slice_stats"))
    parser.add_argument("--min-images", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--acc-diff-threshold", type=float, default=0.02)
    parser.add_argument("--conf-threshold", type=float, default=0.001)
    parser.add_argument("--iou-threshold", type=float, default=0.6)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_corpus(path: Path) -> Dict[str, Dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "Corpus" in payload and isinstance(payload["Corpus"], dict):
        return payload["Corpus"]
    return payload


def collect_split_images(kitti_root: str, split: str) -> List[Path]:
    split_dir = os.path.join(kitti_root, "images", split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"KITTI split directory not found: {split_dir}")
    paths = sorted(glob.glob(os.path.join(split_dir, "*.*")))
    return [Path(p) for p in paths if p.lower().endswith((".jpg", ".jpeg", ".png"))]


def load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> List[Dict]:
    if not label_path.exists():
        return []
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append({"class_id": cls_id, "bbox": np.array([x1, y1, x2, y2])})
    return boxes


def load_attribute_data(attr_file: Path, image_ids: List[str]) -> Dict[str, Dict[str, str]]:
    with open(attr_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "Corpus" in payload:
        raise ValueError(f"{attr_file} looks like a corpus, not attribute assignments.")
    result = {}
    for attr_name, attr_info in payload.items():
        if not isinstance(attr_info, dict) or "data" not in attr_info:
            continue
        data = attr_info["data"]
        mapping = {}
        for i, value in enumerate(data):
            if i < len(image_ids):
                mapping[image_ids[i]] = str(value).strip() if value else ""
        result[attr_name] = mapping
    return result


def get_split_image_ids(kitti_root: str, split: str) -> List[str]:
    paths = collect_split_images(kitti_root, split)
    return [p.stem for p in paths]


# ---------------------------------------------------------------------------
# Vectorized per-image detection matching
# ---------------------------------------------------------------------------
def compute_iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """Vectorized IoU computation. pred_boxes: (M,4), gt_boxes: (N,4) in xyxy format."""
    x1 = np.maximum(pred_boxes[:, None, 0], gt_boxes[None, :, 0])
    y1 = np.maximum(pred_boxes[:, None, 1], gt_boxes[None, :, 1])
    x2 = np.minimum(pred_boxes[:, None, 2], gt_boxes[None, :, 2])
    y2 = np.minimum(pred_boxes[:, None, 3], gt_boxes[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    area_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union = area_pred[:, None] + area_gt[None, :] - inter
    return inter / np.maximum(union, 1e-8)


def match_detections_per_image(
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    pred_confs: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    iou_thresholds: np.ndarray,
) -> Dict:
    """
    Pre-compute detection matching for a single image across all IoU thresholds.
    Returns a dict with per-class, per-threshold TP/FP arrays sorted by confidence.
    """
    result = {
        "num_gt": len(gt_classes),
        "num_pred": len(pred_classes),
        "classes": {},
    }

    all_classes = set(pred_classes.tolist()) | set(gt_classes.tolist())

    for cls_id in all_classes:
        cls_pred_mask = pred_classes == cls_id
        cls_gt_mask = gt_classes == cls_id
        cls_pred_boxes = pred_boxes[cls_pred_mask]
        cls_pred_confs = pred_confs[cls_pred_mask]
        cls_gt_boxes = gt_boxes[cls_gt_mask]

        # Sort by confidence descending
        sort_idx = np.argsort(-cls_pred_confs)
        cls_pred_boxes = cls_pred_boxes[sort_idx]
        cls_pred_confs = cls_pred_confs[sort_idx]

        num_gt = len(cls_gt_boxes)
        num_pred = len(cls_pred_boxes)

        cls_result = {
            "confs": cls_pred_confs.tolist(),
            "num_gt": num_gt,
            "tp_per_iou": {},
        }

        if num_pred == 0 or num_gt == 0:
            for iou_t in iou_thresholds:
                cls_result["tp_per_iou"][f"{iou_t:.2f}"] = [0] * num_pred
        else:
            iou_matrix = compute_iou_matrix(cls_pred_boxes, cls_gt_boxes)  # (num_pred, num_gt)
            for iou_t in iou_thresholds:
                gt_matched = np.zeros(num_gt, dtype=bool)
                tp = []
                for pi in range(num_pred):
                    ious = iou_matrix[pi]
                    # Mask already-matched GTs
                    ious_masked = ious.copy()
                    ious_masked[gt_matched] = 0.0
                    best_gt = np.argmax(ious_masked)
                    if ious_masked[best_gt] >= iou_t:
                        tp.append(1)
                        gt_matched[best_gt] = True
                    else:
                        tp.append(0)
                cls_result["tp_per_iou"][f"{iou_t:.2f}"] = tp

        result["classes"][int(cls_id)] = cls_result

    return result


def compute_ap_from_matches(
    match_results: List[Dict],
    image_ids: List[str],
    all_match_data: Dict[str, Dict],
    num_classes: int = 2,
) -> Dict[str, float]:
    """Compute mAP from pre-computed match results for a subset of images."""
    if not image_ids:
        return {"mAP50": 0.0, "mAP50-95": 0.0, "precision": 0.0, "recall": 0.0,
                "num_images": 0, "num_gt": 0, "num_pred": 0}

    total_gt = 0
    total_pred = 0
    for img_id in image_ids:
        md = all_match_data.get(img_id)
        if md:
            total_gt += md["num_gt"]
            total_pred += md["num_pred"]

    aps_per_iou = {f"{iou_t:.2f}": [] for iou_t in IOU_THRESHOLDS}

    for cls_id in range(num_classes):
        # Collect all predictions for this class across selected images
        all_confs = []
        all_tps = {f"{iou_t:.2f}": [] for iou_t in IOU_THRESHOLDS}
        total_cls_gt = 0

        for img_id in image_ids:
            md = all_match_data.get(img_id)
            if md is None:
                continue
            cls_data = md["classes"].get(cls_id)
            if cls_data is None:
                continue
            all_confs.extend(cls_data["confs"])
            total_cls_gt += cls_data["num_gt"]
            for iou_key in all_tps:
                all_tps[iou_key].extend(cls_data["tp_per_iou"].get(iou_key, []))

        if not all_confs:
            for iou_key in aps_per_iou:
                aps_per_iou[iou_key].append(0.0 if total_cls_gt > 0 else 1.0)
            continue

        # Sort all predictions by confidence
        sort_idx = np.argsort(-np.array(all_confs))

        for iou_key in aps_per_iou:
            tp_arr = np.array(all_tps[iou_key])[sort_idx]
            tp_cumsum = np.cumsum(tp_arr).astype(float)
            fp_cumsum = np.cumsum(1 - tp_arr).astype(float)

            if total_cls_gt == 0:
                aps_per_iou[iou_key].append(1.0 if len(tp_arr) == 0 else 0.0)
                continue

            recalls = tp_cumsum / total_cls_gt
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

            # 101-point interpolation
            ap = 0.0
            for t in np.linspace(0, 1, 101):
                p = precisions[recalls >= t]
                ap += np.max(p) / 101.0 if len(p) > 0 else 0.0
            aps_per_iou[iou_key].append(ap)

    map50 = float(np.mean(aps_per_iou["0.50"])) if aps_per_iou["0.50"] else 0.0
    map50_95 = float(np.mean([np.mean(aps) for aps in aps_per_iou.values()])) if aps_per_iou else 0.0

    # Precision/Recall at IoU=0.5 (aggregate)
    total_tp = 0
    total_fp_pr = 0
    total_fn = 0
    for img_id in image_ids:
        md = all_match_data.get(img_id)
        if md is None:
            continue
        for cls_id in range(num_classes):
            cls_data = md["classes"].get(cls_id)
            if cls_data is None:
                continue
            tp_list = cls_data["tp_per_iou"].get("0.50", [])
            total_tp += sum(tp_list)
            total_fp_pr += len(tp_list) - sum(tp_list)
            total_fn += cls_data["num_gt"] - sum(tp_list)

    precision = total_tp / (total_tp + total_fp_pr) if (total_tp + total_fp_pr) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    return {
        "mAP50": map50,
        "mAP50-95": map50_95,
        "precision": precision,
        "recall": recall,
        "num_images": len(image_ids),
        "num_gt": total_gt,
        "num_pred": total_pred,
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run_analysis(args: argparse.Namespace) -> None:
    start_time = time.time()
    print(f"Loading YOLOv8 model from {args.model_weights}...")

    from ultralytics import YOLO
    model = YOLO(str(args.model_weights))

    # Collect eval split images
    eval_images = collect_split_images(args.kitti_root, args.split)
    eval_image_ids = [p.stem for p in eval_images]
    print(f"Eval split '{args.split}': {len(eval_images)} images")

    # Train image IDs (for attribute alignment)
    train_image_ids = get_split_image_ids(args.kitti_root, "train")
    print(f"Train image IDs for attribute alignment: {len(train_image_ids)}")

    # Phase 1: Run inference and precompute per-image match results
    print(f"\nPhase 1: Running inference and matching on {len(eval_images)} images...")
    from PIL import Image as PILImage

    all_match_data: Dict[str, Dict] = {}
    for i, img_path in enumerate(eval_images):
        img_id = img_path.stem

        # Load ground truth
        label_path = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
        pil_img = PILImage.open(img_path)
        img_w, img_h = pil_img.size
        gt_boxes_raw = load_yolo_labels(label_path, img_w, img_h)

        # Run inference
        results = model.predict(str(img_path), conf=args.conf_threshold,
                                iou=args.iou_threshold, imgsz=args.imgsz, verbose=False)
        pred_list = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    pred_list.append({
                        "class_id": int(box.cls.item()),
                        "bbox": box.xyxy[0].cpu().numpy(),
                        "conf": float(box.conf.item()),
                    })

        # Convert to numpy arrays
        if pred_list:
            pred_boxes = np.array([p["bbox"] for p in pred_list])
            pred_classes = np.array([p["class_id"] for p in pred_list])
            pred_confs = np.array([p["conf"] for p in pred_list])
        else:
            pred_boxes = np.empty((0, 4))
            pred_classes = np.empty(0, dtype=int)
            pred_confs = np.empty(0)

        if gt_boxes_raw:
            gt_boxes = np.array([g["bbox"] for g in gt_boxes_raw])
            gt_classes = np.array([g["class_id"] for g in gt_boxes_raw])
        else:
            gt_boxes = np.empty((0, 4))
            gt_classes = np.empty(0, dtype=int)

        match_result = match_detections_per_image(
            pred_boxes, pred_classes, pred_confs,
            gt_boxes, gt_classes, IOU_THRESHOLDS,
        )
        all_match_data[img_id] = match_result

        if (i + 1) % 500 == 0 or i == 0:
            print(f"  [{i+1}/{len(eval_images)}] images processed")

    inference_time = time.time() - start_time
    print(f"Inference + matching complete in {inference_time:.1f}s")

    # Compute overall metrics
    overall = compute_ap_from_matches([], eval_image_ids, all_match_data, num_classes=len(CLASS_NAMES))
    print(f"\nOverall {args.split} metrics:")
    print(f"  Images: {overall['num_images']}, GT: {overall['num_gt']}, Pred: {overall['num_pred']}")
    print(f"  mAP50: {overall['mAP50']:.4f}, mAP50-95: {overall['mAP50-95']:.4f}")
    print(f"  Precision: {overall['precision']:.4f}, Recall: {overall['recall']:.4f}")

    # Phase 2: Per-attribute-file slice analysis
    for attr_file in args.attribute_file:
        phase2_start = time.time()
        print(f"\n{'='*60}")
        print(f"Analyzing with attribute file: {attr_file}")
        print(f"{'='*60}")

        assigner_tag = attr_file.stem
        attr_data = load_attribute_data(attr_file, train_image_ids)
        attr_names = list(attr_data.keys())
        print(f"Attributes: {attr_names}")

        # Check coverage
        if attr_names:
            covered = sum(1 for img_id in eval_image_ids if img_id in attr_data.get(attr_names[0], {}))
            print(f"Attribute coverage on {args.split}: {covered}/{len(eval_image_ids)} images")
            if covered == 0:
                print(f"WARNING: No eval images have attributes. Skipping this attribute file.")
                continue

        corpus = load_corpus(args.corpus) if args.corpus.exists() else {}

        slice_rows = []
        for attr_name in attr_names:
            attr_lookup = attr_data.get(attr_name, {})

            # Group eval images by attribute value
            value_to_images = defaultdict(list)
            for img_id in eval_image_ids:
                value = attr_lookup.get(img_id, "")
                if value:
                    value_to_images[value].append(img_id)

            corpus_words = corpus.get(attr_name, {}).get("word", [])
            all_values = list(corpus_words)
            for v in sorted(value_to_images.keys()):
                if v not in all_values:
                    all_values.append(v)

            for value in all_values:
                slice_images = value_to_images.get(value, [])
                if len(slice_images) < args.min_images:
                    continue

                metrics = compute_ap_from_matches(
                    [], slice_images, all_match_data, num_classes=len(CLASS_NAMES)
                )

                map_delta = metrics["mAP50-95"] - overall["mAP50-95"]
                map50_delta = metrics["mAP50"] - overall["mAP50"]

                # Train distribution
                train_count = sum(1 for v in attr_data.get(attr_name, {}).values() if v == value)
                train_total = sum(1 for v in attr_data.get(attr_name, {}).values() if v)
                train_share = train_count / train_total if train_total > 0 else None

                eval_count = len(slice_images)
                eval_total = sum(len(imgs) for imgs in value_to_images.values())
                eval_share = eval_count / eval_total if eval_total > 0 else None

                is_failure = metrics["mAP50-95"] < overall["mAP50-95"] - args.acc_diff_threshold

                row = {
                    "attribute": attr_name,
                    "value": value,
                    "num_images": metrics["num_images"],
                    "num_gt": metrics["num_gt"],
                    "num_pred": metrics["num_pred"],
                    "mAP50": round(metrics["mAP50"], 4),
                    "mAP50-95": round(metrics["mAP50-95"], 4),
                    "precision": round(metrics["precision"], 4),
                    "recall": round(metrics["recall"], 4),
                    "mAP50_delta": round(map50_delta, 4),
                    "mAP50-95_delta": round(map_delta, 4),
                    "train_count": train_count,
                    "train_share": round(train_share, 4) if train_share is not None else None,
                    "eval_count": eval_count,
                    "eval_share": round(eval_share, 4) if eval_share is not None else None,
                    "is_failure_case": is_failure,
                }
                slice_rows.append(row)

        # Sort by mAP50-95 ascending (worst first)
        slice_rows.sort(key=lambda r: r["mAP50-95"])

        failure_rows = [r for r in slice_rows if r["is_failure_case"]]
        print(f"\nTotal slices evaluated: {len(slice_rows)}")
        print(f"Failure cases (mAP50-95 < overall - {args.acc_diff_threshold}): {len(failure_rows)}")

        if failure_rows:
            print(f"\nTop {min(args.top_k, len(failure_rows))} failure slices:")
            for row in failure_rows[:args.top_k]:
                print(
                    f"  {row['attribute']}={row['value']} | "
                    f"images={row['num_images']} | "
                    f"mAP50-95={row['mAP50-95']:.4f} | "
                    f"delta={row['mAP50-95_delta']:+.4f} | "
                    f"train_share={row['train_share']}"
                )

        print(f"\nAll slices (sorted by mAP50-95):")
        for row in slice_rows:
            flag = " ***FAILURE***" if row["is_failure_case"] else ""
            print(
                f"  {row['attribute']:20s}={row['value']:20s} | "
                f"imgs={row['num_images']:5d} | "
                f"mAP50={row['mAP50']:.4f} | "
                f"mAP50-95={row['mAP50-95']:.4f} | "
                f"delta={row['mAP50-95_delta']:+.4f}{flag}"
            )

        # Save results
        output_subdir = args.output_dir / assigner_tag
        output_subdir.mkdir(parents=True, exist_ok=True)
        json_path = output_subdir / f"{args.split}_slice_stats.json"
        csv_path = output_subdir / f"{args.split}_slice_stats.csv"

        summary = {
            "model_weights": str(args.model_weights),
            "attribute_file": str(attr_file),
            "kitti_root": args.kitti_root,
            "split": args.split,
            "overall": overall,
            "num_slices": len(slice_rows),
            "num_failures": len(failure_rows),
            "acc_diff_threshold": args.acc_diff_threshold,
            "slices": slice_rows,
        }
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"\nSaved JSON to {json_path}")

        if slice_rows:
            fieldnames = list(slice_rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in slice_rows:
                    writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
            print(f"Saved CSV to {csv_path}")

        phase2_time = time.time() - phase2_start
        print(f"Slice analysis for {assigner_tag}: {phase2_time:.1f}s")

    elapsed = time.time() - start_time
    print(f"\nTotal analysis time: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    args = parse_args()
    run_analysis(args)
