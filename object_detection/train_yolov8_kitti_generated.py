#!/usr/bin/env python3
import argparse
import ast
from collections import defaultdict
import datetime
import json
import os
import random
import shutil
from pathlib import Path
import time
from typing import Dict, List, Sequence


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def str2bool(value: str) -> bool:
    value = value.lower().strip()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a KITTI YOLOv8 training split from a sub-ratio of the "
            "first-5000 original train images plus a random generated-image "
            "subset, then fine-tune from an existing checkpoint."
        )
    )
    parser.add_argument(
        "--base-data-root",
        type=Path,
        default=Path("/data/oxx220000/kitti_yolov8_first5000"),
        help="Prepared KITTI YOLO dataset root with images/train and labels/train.",
    )
    parser.add_argument(
        "--generated-image-dir",
        type=Path,
        default=Path(
            "/data/oxx220000/kitti-weather-edits/"
            "generated_kitti1352_red_cloudy_icedit_upscaled"
        ),
        help="Directory of generated KITTI images. Stems must match source KITTI IDs.",
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path("/data/oxx220000/kitti_yolov8_ratio1p0_red_cloudy_gen1000_seed42"),
        help="Output YOLO dataset root for the selected original+generated train split.",
    )
    parser.add_argument(
        "--resume-model",
        type=Path,
        default=Path(
            "runs/kitti_yolov8/"
            "first5000_1240val_1241test_gpu0_bs16_e30_20260429_012232/"
            "weights/best.pt"
        ),
        help="YOLO checkpoint used as the fine-tuning initialization.",
    )
    parser.add_argument(
        "--sub-ratio",
        type=float,
        default=1.0,
        help="Ratio of the 5000 original KITTI train images to keep before adding generated images.",
    )
    parser.add_argument(
        "--selection-strategy",
        choices=["random", "stratified"],
        default="random",
        help=(
            "How to select original KITTI train images before adding generated images. "
            "For detection, stratified groups images by the set of YOLO classes present."
        ),
    )
    parser.add_argument("--num-added-img", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Fine-tuning epochs after loading --resume-model.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help="Explicit optimizer for resume fine-tuning. Avoid 'auto' if you want lr0 respected.",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=2e-4,
        help="Initial fine-tuning learning rate. Previous plain auto-AdamW used about 1.667e-3.",
    )
    parser.add_argument(
        "--lrf",
        type=float,
        default=0.1,
        help="Final LR fraction for the YOLO scheduler.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=float,
        default=1.0,
        help="Short warmup for resume fine-tuning.",
    )
    parser.add_argument("--project", type=str, default="runs/kitti_yolov8")
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Ultralytics run name. Defaults to a deterministic name from counts and seed.",
    )
    parser.add_argument("--plots", type=str2bool, default=False)
    parser.add_argument("--eval-test", type=str2bool, default=True)
    parser.add_argument(
        "--copy-images",
        type=str2bool,
        default=False,
        help="Copy images instead of hardlinking/symlinking into the prepared dataset.",
    )
    parser.add_argument(
        "--overwrite-prepared",
        type=str2bool,
        default=False,
        help="Delete and rebuild --prepared-root if it already exists.",
    )
    parser.add_argument(
        "--reuse-prepared",
        type=str2bool,
        default=False,
        help="Skip dataset preparation if --prepared-root already exists.",
    )
    parser.add_argument(
        "--prepare-only",
        type=str2bool,
        default=False,
        help="Only write the derived YOLO dataset and manifests; do not train.",
    )
    return parser.parse_args()


def _format_duration(seconds: float) -> str:
    total = int(max(0, round(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _timestamp_str(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _dataset_has_split(data_yaml: Path, split_name: str) -> bool:
    split_prefix = f"{split_name}:"
    for raw_line in data_yaml.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(split_prefix):
            value = line.split(":", 1)[1].strip()
            return len(value) > 0
    return False


def _read_class_names(base_yaml: Path) -> List[str]:
    for raw_line in base_yaml.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("names:"):
            value = line.split(":", 1)[1].strip()
            parsed = ast.literal_eval(value)
            if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
                raise ValueError(f"Unsupported names entry in {base_yaml}: {value}")
            return parsed
    raise ValueError(f"No names entry found in {base_yaml}")


def _write_dataset_yaml(output_root: Path, class_names: Sequence[str], has_test: bool) -> Path:
    names = ", ".join([f"'{name}'" for name in class_names])
    lines = [
        f"path: {output_root}",
        "train: images/train",
        "val: images/val",
    ]
    if has_test:
        lines.append("test: images/test")
    lines.extend([f"names: [{names}]", ""])
    yaml_path = output_root / "kitti_random_generated.yaml"
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    return yaml_path


def _safe_link_or_copy(src: Path, dst: Path, copy_images: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        dst.symlink_to(src)


def _copy_or_link_label(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if src.exists():
        shutil.copy2(src, dst)
    else:
        dst.write_text("", encoding="utf-8")


def _list_images(image_dir: Path) -> List[Path]:
    return sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    )


def _label_signature(label_path: Path) -> str:
    if not label_path.exists():
        return "background"
    class_ids = set()
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if not parts:
            continue
        class_ids.add(parts[0])
    if not class_ids:
        return "background"
    return "+".join(sorted(class_ids, key=lambda value: int(value) if value.isdigit() else value))


def _stratified_sample_ids(
    image_ids: Sequence[str],
    label_dir: Path,
    n_select: int,
    rng: random.Random,
) -> List[str]:
    strata: Dict[str, List[str]] = defaultdict(list)
    for image_id in image_ids:
        strata[_label_signature(label_dir / f"{image_id}.txt")].append(image_id)

    total = len(image_ids)
    allocations = {}
    fractions = []
    allocated = 0
    for stratum, ids in sorted(strata.items()):
        exact = n_select * (len(ids) / total)
        count = min(len(ids), int(exact))
        allocations[stratum] = count
        allocated += count
        fractions.append((exact - int(exact), stratum))

    for _, stratum in sorted(fractions, reverse=True):
        if allocated >= n_select:
            break
        if allocations[stratum] < len(strata[stratum]):
            allocations[stratum] += 1
            allocated += 1

    selected: List[str] = []
    for stratum, ids in sorted(strata.items()):
        count = allocations[stratum]
        if count <= 0:
            continue
        selected.extend(rng.sample(ids, count))

    if len(selected) < n_select:
        remaining = [image_id for image_id in image_ids if image_id not in set(selected)]
        selected.extend(rng.sample(remaining, n_select - len(selected)))

    return sorted(selected)


def _prepare_split_copy(
    image_ids: Sequence[str],
    src_image_dir: Path,
    src_label_dir: Path,
    dst_image_dir: Path,
    dst_label_dir: Path,
    copy_images: bool,
):
    for image_id in image_ids:
        src_image = None
        for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            candidate = src_image_dir / f"{image_id}{ext}"
            if candidate.exists():
                src_image = candidate
                break
        if src_image is None:
            raise FileNotFoundError(f"Image for id {image_id} not found in {src_image_dir}")
        src_label = src_label_dir / f"{image_id}.txt"
        _safe_link_or_copy(src_image, dst_image_dir / src_image.name, copy_images=copy_images)
        _copy_or_link_label(src_label, dst_label_dir / f"{image_id}.txt")


def _select_generated_candidates(generated_dir: Path, base_train_label_dir: Path) -> Dict[str, Path]:
    candidates: Dict[str, Path] = {}
    for image_path in _list_images(generated_dir):
        source_id = image_path.stem
        if (base_train_label_dir / f"{source_id}.txt").exists():
            candidates[source_id] = image_path
    return dict(sorted(candidates.items()))


def prepare_dataset(args) -> Path:
    base_root = args.base_data_root
    base_yaml = base_root / "kitti_first5000.yaml"
    base_train_images = base_root / "images" / "train"
    base_train_labels = base_root / "labels" / "train"
    base_val_images = base_root / "images" / "val"
    base_val_labels = base_root / "labels" / "val"
    base_test_images = base_root / "images" / "test"
    base_test_labels = base_root / "labels" / "test"

    for required in [base_yaml, base_train_images, base_train_labels, args.generated_image_dir]:
        if not required.exists():
            raise FileNotFoundError(f"Required path not found: {required}")

    if args.prepared_root.exists():
        if args.reuse_prepared:
            return args.prepared_root / "kitti_random_generated.yaml"
        if args.overwrite_prepared:
            shutil.rmtree(args.prepared_root)
        else:
            raise FileExistsError(
                f"{args.prepared_root} already exists. Use --reuse-prepared true "
                "or --overwrite-prepared true."
            )

    rng = random.Random(args.seed)
    class_names = _read_class_names(base_yaml)

    train_images = _list_images(base_train_images)
    train_ids = [p.stem for p in train_images]
    if args.sub_ratio <= 0.0 or args.sub_ratio > 1.0:
        raise ValueError(f"--sub-ratio must be in (0, 1], got {args.sub_ratio}")
    original_sample_count = int(len(train_ids) * args.sub_ratio)
    if original_sample_count <= 0:
        raise ValueError(
            f"--sub-ratio {args.sub_ratio} selected zero images from {len(train_ids)} candidates"
        )
    if original_sample_count < len(train_ids):
        if args.selection_strategy == "stratified":
            selected_original_ids = _stratified_sample_ids(
                train_ids,
                base_train_labels,
                original_sample_count,
                rng,
            )
        else:
            selected_original_ids = sorted(rng.sample(train_ids, original_sample_count))
    else:
        selected_original_ids = list(train_ids)

    generated_candidates = _select_generated_candidates(args.generated_image_dir, base_train_labels)
    if not generated_candidates:
        raise RuntimeError(
            f"No generated images in {args.generated_image_dir} have matching train labels "
            f"under {base_train_labels}."
        )
    generated_ids = list(generated_candidates)
    generated_sample_count = min(args.num_added_img, len(generated_ids))
    selected_generated_ids = sorted(rng.sample(generated_ids, generated_sample_count))

    train_image_out = args.prepared_root / "images" / "train"
    train_label_out = args.prepared_root / "labels" / "train"
    val_image_out = args.prepared_root / "images" / "val"
    val_label_out = args.prepared_root / "labels" / "val"
    test_image_out = args.prepared_root / "images" / "test"
    test_label_out = args.prepared_root / "labels" / "test"

    _prepare_split_copy(
        selected_original_ids,
        base_train_images,
        base_train_labels,
        train_image_out,
        train_label_out,
        copy_images=args.copy_images,
    )

    for source_id in selected_generated_ids:
        generated_image = generated_candidates[source_id]
        generated_stem = f"gen_red_cloudy_{source_id}"
        _safe_link_or_copy(
            generated_image,
            train_image_out / f"{generated_stem}{generated_image.suffix.lower()}",
            copy_images=args.copy_images,
        )
        _copy_or_link_label(base_train_labels / f"{source_id}.txt", train_label_out / f"{generated_stem}.txt")

    val_ids: List[str] = []
    if base_val_images.exists() and base_val_labels.exists():
        val_ids = [p.stem for p in _list_images(base_val_images)]
        _prepare_split_copy(
            val_ids,
            base_val_images,
            base_val_labels,
            val_image_out,
            val_label_out,
            copy_images=args.copy_images,
        )

    test_ids: List[str] = []
    if base_test_images.exists() and base_test_labels.exists():
        test_ids = [p.stem for p in _list_images(base_test_images)]
        _prepare_split_copy(
            test_ids,
            base_test_images,
            base_test_labels,
            test_image_out,
            test_label_out,
            copy_images=args.copy_images,
        )

    yaml_path = _write_dataset_yaml(args.prepared_root, class_names, has_test=bool(test_ids))
    (args.prepared_root / "selected_original_ids.txt").write_text(
        "\n".join(selected_original_ids) + "\n",
        encoding="utf-8",
    )
    (args.prepared_root / "selected_generated_source_ids.txt").write_text(
        "\n".join(selected_generated_ids) + "\n",
        encoding="utf-8",
    )
    summary = {
        "base_data_root": str(args.base_data_root),
        "generated_image_dir": str(args.generated_image_dir),
        "seed": args.seed,
        "sub_ratio": args.sub_ratio,
        "selection_strategy": args.selection_strategy,
        "original_candidates": len(train_ids),
        "selected_original": len(selected_original_ids),
        "generated_candidates_with_labels": len(generated_candidates),
        "selected_generated": len(selected_generated_ids),
        "final_train_images": len(selected_original_ids) + len(selected_generated_ids),
        "val_images": len(val_ids),
        "test_images": len(test_ids),
        "class_names": list(class_names),
        "resume_model": str(args.resume_model),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "warmup_epochs": args.warmup_epochs,
    }
    (args.prepared_root / "selection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Prepared KITTI YOLO generated-training dataset:")
    print(json.dumps(summary, indent=2))
    print(f"Dataset YAML: {yaml_path}")
    if len(selected_generated_ids) < args.num_added_img:
        print(
            f"Warning: requested {args.num_added_img} generated images, "
            f"but only {len(selected_generated_ids)} matched train labels."
        )
    return yaml_path


def train_yolov8(args, data_yaml: Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required. Use the KITTI fp environment described in object_detection/README.md."
        ) from exc

    if not args.resume_model.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_model}")

    run_name = args.name
    if run_name is None:
        ratio_tag = str(args.sub_ratio).replace(".", "p")
        run_name = (
            f"kitti_ratio{ratio_tag}_{args.selection_strategy}_red_cloudy_gen{args.num_added_img}"
            f"_seed{args.seed}"
        )

    model = YOLO(str(args.resume_model))
    train_start_time = time.time()
    train_results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=run_name,
        seed=args.seed,
        plots=args.plots,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        warmup_epochs=args.warmup_epochs,
    )
    train_elapsed = time.time() - train_start_time
    print(f"Total training time for {args.epochs} epochs: {train_elapsed:.2f} seconds")
    print(f"Training elapsed (HH:MM:SS): {_format_duration(train_elapsed)}")
    print(train_results)

    if args.eval_test and _dataset_has_split(data_yaml, "test"):
        save_dir = None
        if hasattr(model, "trainer") and model.trainer is not None:
            save_dir = Path(model.trainer.save_dir)

        best_weights = None
        if save_dir is not None:
            candidate = save_dir / "weights" / "best.pt"
            if candidate.exists():
                best_weights = candidate

        eval_model = YOLO(str(best_weights)) if best_weights is not None else model
        print("Running final evaluation on dataset test split...")
        test_start_time = time.time()
        test_results = eval_model.val(
            data=str(data_yaml),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=args.project,
            name=f"{run_name}_test",
        )
        test_elapsed = time.time() - test_start_time
        print(f"Test evaluation elapsed: {test_elapsed:.2f} seconds")
        print(f"Test evaluation elapsed (HH:MM:SS): {_format_duration(test_elapsed)}")
        print(test_results)
    elif args.eval_test:
        print("Dataset yaml has no test split configured; skipping final test evaluation.")


def main():
    overall_start_time = time.time()
    print("Current time:", datetime.datetime.now().strftime("%m%d%H%M%S"))
    print(f"Start timestamp: {_timestamp_str(overall_start_time)}")
    args = parse_args()

    data_yaml = prepare_dataset(args)
    if args.prepare_only:
        print("prepare-only mode enabled; skipping training.")
    else:
        train_yolov8(args, data_yaml)

    overall_elapsed = time.time() - overall_start_time
    end_timestamp = time.time()
    print(f"End timestamp: {_timestamp_str(end_timestamp)}")
    print(f"End-to-end elapsed (HH:MM:SS): {_format_duration(overall_elapsed)}")


if __name__ == "__main__":
    main()
