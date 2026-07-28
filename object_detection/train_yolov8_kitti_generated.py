#!/usr/bin/env python3
import argparse
import ast
from collections import defaultdict
import datetime
import json
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
from pathlib import Path
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
DEFAULT_GENERATED_ATTRIBUTE_COUNTS = {
    "color": {"red": 1.0},
    "weather": {"cloudy": 1.0},
}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    def __getattr__(self, name):
        return getattr(self.streams[0], name)


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
            "save/runs/kitti_yolov8/"
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
        choices=["random", "stratified", "attribute", "coreset", "badge", "uherding", "ppal", "entropy"],
        default="random",
        help=(
            "How to select original KITTI train images before adding generated images. "
            "For detection, stratified groups images by the set of YOLO classes present. "
            "Attribute, coreset, badge, uherding, and ppal use the shared active-learning selectors. "
            "Entropy uses a YOLO detection-confidence entropy proxy."
        ),
    )
    parser.add_argument(
        "--al-entropy-conf-thres",
        type=float,
        default=0.001,
        help="Confidence threshold for the YOLO entropy proxy prediction pass.",
    )
    parser.add_argument(
        "--al-entropy-iou",
        type=float,
        default=0.7,
        help="IoU threshold for the YOLO entropy proxy prediction pass.",
    )
    parser.add_argument(
        "--al-entropy-batch",
        type=int,
        default=16,
        help="Prediction batch size for entropy proxy selection.",
    )
    parser.add_argument(
        "--al-initial-labeled",
        type=int,
        default=400,
        help="Number of validation proxy embeddings used as the labeled set for coreset/BADGE/UHerding/PPAL.",
    )
    parser.add_argument(
        "--al-embedding-thumbnail-size",
        type=int,
        default=8,
        help="Thumbnail side length for lightweight KITTI image embeddings used by coreset/BADGE/UHerding/PPAL.",
    )
    parser.add_argument(
        "--al-fast-selection",
        type=str2bool,
        default=True,
        help=(
            "Use the approximate coreset/BADGE/UHerding/PPAL selection paths from utils.active_learning. "
            "Set false to use exact k-center, exact BADGE k-means++, exact UHerding, or exact PPAL when feasible."
        ),
    )
    parser.add_argument(
        "--al-ppal-temperature",
        type=float,
        default=1.0,
        help="Temperature for PPAL proxy centroid probabilities.",
    )
    parser.add_argument(
        "--al-ppal-uncertainty",
        choices=["binary_entropy", "entropy", "margin", "confidence"],
        default="binary_entropy",
        help="Proxy uncertainty measure used by PPAL.",
    )
    parser.add_argument(
        "--al-ppal-pool-expand-ratio",
        type=float,
        default=4.0,
        help="PPAL uncertainty pool size multiplier before diversity selection.",
    )
    parser.add_argument(
        "--al-ppal-class-weight-ub",
        type=float,
        default=0.2,
        help="Upper-bound scale for PPAL proxy class-quality weighting.",
    )
    parser.add_argument(
        "--al-ppal-class-weight-alpha",
        type=float,
        default=0.3,
        help="Shape parameter for PPAL proxy class-quality weighting.",
    )
    parser.add_argument(
        "--al-ppal-diversity-power",
        type=float,
        default=1.0,
        help="Power applied to PPAL diversity distance weights.",
    )
    parser.add_argument(
        "--al-uherding-temperature",
        type=float,
        default=1.0,
        help="Temperature for the proxy centroid probabilities used by UHerding.",
    )
    parser.add_argument(
        "--al-uherding-uncertainty",
        choices=["margin", "entropy", "confidence"],
        default="margin",
        help="Proxy uncertainty measure used by UHerding.",
    )
    parser.add_argument(
        "--al-uherding-kernel-scale",
        type=float,
        default=0.0,
        help=(
            "RBF kernel scale for UHerding. Non-positive values use a median-distance "
            "estimate from the candidate embeddings."
        ),
    )
    parser.add_argument(
        "--attribute-file",
        type=Path,
        default=Path("save/attribute_kitti_qwen3_8B.json"),
        help="KITTI train attribute metadata JSON used by --selection-strategy attribute.",
    )
    parser.add_argument(
        "--al-attribute-focus",
        type=str,
        default="color,weather",
        help="Comma-separated attribute keys for attribute selection.",
    )
    parser.add_argument(
        "--al-attribute-rarity-power",
        type=float,
        default=0.03,
        help="R component exponent for attribute selection.",
    )
    parser.add_argument(
        "--al-attribute-use-rarity",
        type=str2bool,
        default=True,
        help="Include the R component in attribute selection.",
    )
    parser.add_argument(
        "--al-attribute-uncertainty-boost",
        type=float,
        default=1.0,
        help="Bonus for uncertain metadata entries, if present.",
    )
    parser.add_argument(
        "--al-attribute-log-path",
        type=Path,
        default=None,
        help="Optional diagnostic log used for log-derived failure weights.",
    )
    parser.add_argument(
        "--al-attribute-log-acc-scale",
        type=float,
        default=0.0,
        help="Scale for log-derived failure weights.",
    )
    parser.add_argument(
        "--al-attribute-extra-counts",
        type=str,
        default=None,
        help=(
            "Extra generated attribute counts, e.g. 'color:red=1000;weather:cloudy=1000'. "
            "Defaults to red/cloudy counts from --num-added-img."
        ),
    )
    parser.add_argument(
        "--al-attribute-extra-count-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to extra generated attribute counts.",
    )
    parser.add_argument(
        "--al-attribute-frequency-penalty",
        type=float,
        default=5.0,
        help="P component weight for penalizing over-represented attribute values.",
    )
    parser.add_argument(
        "--al-attribute-class-balance",
        type=str,
        default="15",
        help="C component weight for label-set class balancing; accepts 'inf'.",
    )
    parser.add_argument(
        "--al-conditional-rarity",
        type=str2bool,
        default=False,
        help="Use class-conditional rarity instead of global rarity.",
    )
    parser.add_argument("--num-added-img", type=int, default=1000)
    parser.add_argument(
        "--require-exact-generated-count",
        type=str2bool,
        default=False,
        help=(
            "Fail when fewer than --num-added-img eligible generated images are available. "
            "The default preserves the existing best-effort behavior."
        ),
    )
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
    parser.add_argument("--project", type=str, default="save/runs/kitti_yolov8")
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
    parser.add_argument(
        "--write-log",
        type=str2bool,
        default=True,
        help="Write stdout/stderr to a persistent log file while also printing to console.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory for auto-generated logs when --log-file is not set.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional explicit log path. Overrides --log-dir.",
    )
    parser.add_argument(
        "--timing-json",
        type=Path,
        default=None,
        help="Optional machine-readable timing and cost-accounting JSON output path.",
    )
    return parser.parse_args()


def _make_run_name(args) -> str:
    if args.name is not None:
        return args.name
    ratio_tag = str(args.sub_ratio).replace(".", "p")
    return (
        f"kitti_ratio{ratio_tag}_{args.selection_strategy}_red_cloudy_gen{args.num_added_img}"
        f"_seed{args.seed}"
    )


def _format_duration(seconds: float) -> str:
    total = int(max(0, round(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _timestamp_str(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _stamp_str(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S")


def _setup_log(args, run_name: str, start_time: float):
    if not args.write_log:
        return None, None, None

    log_path = args.log_file
    if log_path is None:
        log_path = args.log_dir / f"{run_name}_{_stamp_str(start_time)}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_handle = log_path.open("w", encoding="utf-8", buffering=1)
    sys.stdout = Tee(original_stdout, log_handle)
    sys.stderr = Tee(original_stderr, log_handle)
    return log_path, log_handle, (original_stdout, original_stderr)


TIMING_STAGE_NAMES = (
    "setup_input_validation",
    "original_source_selection",
    "generated_pool_discovery_sampling",
    "dataset_materialization",
    "yolo_checkpoint_model_load",
    "model_train",
    "final_eval_checkpoint_load",
    "test_evaluation",
)


def _finish_active_stage(stage_seconds: Dict[str, float], stage_state: Dict[str, Any]):
    stage_name = stage_state.pop("name", None)
    stage_start = stage_state.pop("start", None)
    if stage_name is not None and stage_start is not None:
        stage_seconds[stage_name] += time.perf_counter() - float(stage_start)


def _switch_active_stage(
    stage_seconds: Dict[str, float],
    stage_state: Dict[str, Any],
    stage_name: str,
):
    _finish_active_stage(stage_seconds, stage_state)
    stage_state["name"] = stage_name
    stage_state["start"] = time.perf_counter()


def _resolved_path(path: Optional[Union[Path, str]]) -> Optional[str]:
    if path is None:
        return None
    return str(Path(path).expanduser().resolve())


def _collect_gpu_metadata() -> Dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"query": command, "available": False, "error": f"{type(exc).__name__}: {exc}"}

    if completed.returncode != 0:
        return {
            "query": command,
            "available": False,
            "returncode": completed.returncode,
            "error": completed.stderr.strip(),
        }

    devices = []
    for raw_line in completed.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",", 4)]
        if len(parts) == 5:
            devices.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "uuid": parts[2],
                    "driver_version": parts[3],
                    "memory_total_mib": parts[4],
                }
            )
        elif raw_line.strip():
            devices.append({"raw": raw_line.strip()})
    return {"query": command, "available": True, "devices": devices}


def _selection_parameters(args) -> Dict[str, Any]:
    return {
        "strategy": args.selection_strategy,
        "sub_ratio": args.sub_ratio,
        "seed": args.seed,
        "num_added_img": args.num_added_img,
        "require_exact_generated_count": args.require_exact_generated_count,
        "entropy": {
            "conf_thres": args.al_entropy_conf_thres,
            "iou": args.al_entropy_iou,
            "batch": args.al_entropy_batch,
        },
        "embedding_proxy": {
            "initial_labeled": args.al_initial_labeled,
            "thumbnail_size": args.al_embedding_thumbnail_size,
            "fast_selection": args.al_fast_selection,
        },
        "ppal": {
            "temperature": args.al_ppal_temperature,
            "uncertainty": args.al_ppal_uncertainty,
            "pool_expand_ratio": args.al_ppal_pool_expand_ratio,
            "class_weight_ub": args.al_ppal_class_weight_ub,
            "class_weight_alpha": args.al_ppal_class_weight_alpha,
            "diversity_power": args.al_ppal_diversity_power,
        },
        "uherding": {
            "temperature": args.al_uherding_temperature,
            "uncertainty": args.al_uherding_uncertainty,
            "kernel_scale": args.al_uherding_kernel_scale,
        },
        "attribute": {
            "file": _resolved_path(args.attribute_file),
            "focus": _parse_focus_attributes(args.al_attribute_focus),
            "rarity_power": args.al_attribute_rarity_power,
            "use_rarity": args.al_attribute_use_rarity,
            "uncertainty_boost": args.al_attribute_uncertainty_boost,
            "log_path": _resolved_path(args.al_attribute_log_path),
            "log_acc_scale": args.al_attribute_log_acc_scale,
            "extra_counts_raw": args.al_attribute_extra_counts,
            "default_extra_counts": DEFAULT_GENERATED_ATTRIBUTE_COUNTS,
            "extra_counts_default_count": args.num_added_img,
            "extra_count_scale": args.al_attribute_extra_count_scale,
            "frequency_penalty": args.al_attribute_frequency_penalty,
            "class_balance": args.al_attribute_class_balance,
            "conditional_rarity": args.al_conditional_rarity,
        },
    }


def _training_schedule(args) -> Dict[str, Any]:
    return {
        "prepare_only": args.prepare_only,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "warmup_epochs": args.warmup_epochs,
        "plots": args.plots,
        "eval_test": args.eval_test,
    }


def _initial_timing_record(args, run_name: str, start_wall_time: float) -> Dict[str, Any]:
    environment_keys = (
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "CONDA_DEFAULT_ENV",
        "CONDA_PREFIX",
        "VIRTUAL_ENV",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    )
    project_path = Path(args.project).expanduser().resolve()
    prepared_root = args.prepared_root.expanduser().resolve()
    return {
        "schema_version": 1,
        "status": "running",
        "error": None,
        "command": {
            "argv": list(sys.argv),
            "executable": sys.executable,
            "cwd": str(Path.cwd()),
            "script": str(Path(__file__).resolve()),
        },
        "timestamps": {
            "start_unix": start_wall_time,
            "start_iso": datetime.datetime.fromtimestamp(start_wall_time).astimezone().isoformat(),
            "end_unix": None,
            "end_iso": None,
        },
        "selection_parameters": _selection_parameters(args),
        "training_schedule": _training_schedule(args),
        "dataset_options": {
            "copy_images": args.copy_images,
            "overwrite_prepared": args.overwrite_prepared,
            "reuse_prepared": args.reuse_prepared,
        },
        "paths": {
            "base_data_root": _resolved_path(args.base_data_root),
            "base_dataset_yaml": _resolved_path(args.base_data_root / "kitti_first5000.yaml"),
            "generated_image_dir": _resolved_path(args.generated_image_dir),
            "prepared_root": str(prepared_root),
            "prepared_dataset_yaml": str(prepared_root / "kitti_random_generated.yaml"),
            "selection_summary": str(prepared_root / "selection_summary.json"),
            "selected_original_ids": str(prepared_root / "selected_original_ids.txt"),
            "selected_generated_source_ids": str(
                prepared_root / "selected_generated_source_ids.txt"
            ),
            "resume_checkpoint": _resolved_path(args.resume_model),
            "project": str(project_path),
            "requested_run_dir": str(project_path / run_name),
            "actual_run_dir": None,
            "best_checkpoint": None,
            "test_eval_dir": None,
            "log_file": None,
            "timing_json": _resolved_path(args.timing_json),
        },
        "provenance": {
            "run_name": run_name,
            "prepared_dataset_reused": False,
            "selection_summary_preserved": True,
        },
        "counts": {
            "requested_generated": args.num_added_img,
            "require_exact_generated_count": args.require_exact_generated_count,
        },
        "evaluation": {},
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "pid": os.getpid(),
            "variables": {key: os.environ.get(key) for key in environment_keys},
            "slurm": {
                key: value for key, value in sorted(os.environ.items()) if key.startswith("SLURM_")
            },
            "gpu": {},
        },
        "stage_seconds": {name: 0.0 for name in TIMING_STAGE_NAMES},
        "stage_seconds_sum": 0.0,
        "command_total_seconds": 0.0,
        "unattributed_seconds": 0.0,
    }


def _finalize_timing_record(
    record: Dict[str, Any],
    command_start: float,
    status: str,
    error: Optional[BaseException] = None,
):
    end_wall_time = time.time()
    total_seconds = time.perf_counter() - command_start
    stage_sum = sum(float(record["stage_seconds"].get(name, 0.0)) for name in TIMING_STAGE_NAMES)
    record["status"] = status
    record["timestamps"]["end_unix"] = end_wall_time
    record["timestamps"]["end_iso"] = datetime.datetime.fromtimestamp(end_wall_time).astimezone().isoformat()
    record["stage_seconds_sum"] = stage_sum
    record["command_total_seconds"] = total_seconds
    record["unattributed_seconds"] = max(0.0, total_seconds - stage_sum)
    if error is not None:
        record["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        }


def _write_timing_json(path: Optional[Path], record: Dict[str, Any]):
    if path is None:
        return
    output_path = path.expanduser().resolve()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
        temporary_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary_path, output_path)
    except Exception as exc:
        print(
            f"Warning: failed to write timing JSON {output_path}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )


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


def _resolve_image_path(image_dir: Path, image_id: str) -> Path:
    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        candidate = image_dir / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image for id {image_id} not found in {image_dir}")


class KittiSelectionProxyDataset:
    def __init__(self, metadata_indices: Sequence[int], class_labels: Sequence[int]):
        self.attribute_metadata_indices = list(metadata_indices)
        self.al_targets = list(class_labels)

    def __len__(self):
        return len(self.attribute_metadata_indices)


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


def _image_proxy_embedding(image_path: Path, thumbnail_size: int) -> np.ndarray:
    from PIL import Image

    side = max(int(thumbnail_size), 2)
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((side, side), Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0

    flat = arr.reshape(-1)
    means = arr.mean(axis=(0, 1))
    stds = arr.std(axis=(0, 1))
    mins = arr.min(axis=(0, 1))
    maxs = arr.max(axis=(0, 1))
    return np.concatenate([flat, means, stds, mins, maxs], axis=0).astype(np.float32)


def _extract_proxy_embeddings(
    image_ids: Sequence[str],
    image_dir: Path,
    thumbnail_size: int,
) -> np.ndarray:
    embeddings = []
    for idx, image_id in enumerate(image_ids):
        if idx > 0 and idx % 1000 == 0:
            print(f"  Extracted proxy embeddings for {idx}/{len(image_ids)} images...")
        embeddings.append(_image_proxy_embedding(_resolve_image_path(image_dir, image_id), thumbnail_size))
    if not embeddings:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(embeddings, axis=0)


def _label_signature_targets(image_ids: Sequence[str], label_dir: Path, signature_to_label=None):
    signatures = [_label_signature(label_dir / f"{image_id}.txt") for image_id in image_ids]
    if signature_to_label is None:
        signature_to_label = {
            signature: idx for idx, signature in enumerate(sorted(set(signatures)))
        }
    targets = np.array([signature_to_label[signature] for signature in signatures], dtype=np.int64)
    return targets, signature_to_label


def _select_embedding_ids(
    strategy_name: str,
    train_ids: Sequence[str],
    train_image_dir: Path,
    train_label_dir: Path,
    val_image_dir: Path,
    val_label_dir: Path,
    n_select: int,
    args,
) -> List[str]:
    from utils.active_learning import BadgeStrategy, CoreSetStrategy, PPALStrategy, UncertaintyHerdingStrategy

    print(
        f"KITTI {strategy_name} selection: extracting lightweight image embeddings "
        f"for {len(train_ids)} train candidates."
    )
    val_ids = [p.stem for p in _list_images(val_image_dir)] if val_image_dir.exists() else []
    if not val_ids:
        raise RuntimeError(f"{strategy_name} selection requires validation images as a labeled proxy.")

    all_signatures = {
        _label_signature(train_label_dir / f"{image_id}.txt") for image_id in train_ids
    }
    all_signatures.update(
        _label_signature(val_label_dir / f"{image_id}.txt") for image_id in val_ids
    )
    signature_to_label = {
        signature: idx for idx, signature in enumerate(sorted(all_signatures))
    }

    train_embeddings = _extract_proxy_embeddings(
        train_ids,
        train_image_dir,
        args.al_embedding_thumbnail_size,
    )
    train_targets, _ = _label_signature_targets(train_ids, train_label_dir, signature_to_label)

    print(f"KITTI {strategy_name} selection: extracting validation proxy embeddings from {len(val_ids)} images.")
    val_embeddings = _extract_proxy_embeddings(
        val_ids,
        val_image_dir,
        args.al_embedding_thumbnail_size,
    )
    val_targets, _ = _label_signature_targets(val_ids, val_label_dir, signature_to_label)
    initial_size = min(max(int(args.al_initial_labeled), 1), len(val_embeddings))
    labeled_embeddings = val_embeddings[:initial_size]
    labeled_targets = val_targets[:initial_size]

    if strategy_name == "coreset":
        strategy = CoreSetStrategy(
            embedding_dim=train_embeddings.shape[1],
            exact_kcenter_max_select=(1 if args.al_fast_selection else 4096),
            exact_kcenter_max_samples=(1 if args.al_fast_selection else 20000),
        )
        selected_indices = strategy.greedy_k_center(
            labeled_embeddings=labeled_embeddings,
            unlabeled_embeddings=train_embeddings,
            n_select=n_select,
            random_seed=args.seed,
        )
    elif strategy_name == "badge":
        strategy = BadgeStrategy(
            exact_kmeanspp_max_select=(1 if args.al_fast_selection else 512),
            exact_kmeanspp_max_samples=(1 if args.al_fast_selection else 20000),
        )
        selected_indices = strategy.select(
            labeled_embeddings=labeled_embeddings,
            unlabeled_embeddings=train_embeddings,
            labeled_targets=labeled_targets,
            n_select=n_select,
            random_seed=args.seed,
        )
        if strategy.last_proxy_stats:
            print("KITTI BADGE proxy stats:")
            print(json.dumps(strategy.last_proxy_stats, indent=2))
    elif strategy_name == "uherding":
        strategy = UncertaintyHerdingStrategy(
            temperature=args.al_uherding_temperature,
            uncertainty=args.al_uherding_uncertainty,
            kernel_scale=(
                args.al_uherding_kernel_scale
                if args.al_uherding_kernel_scale > 0.0
                else None
            ),
            exact_max_select=(1 if args.al_fast_selection else 512),
            exact_max_samples=(1 if args.al_fast_selection else 10000),
        )
        selected_indices = strategy.select(
            labeled_embeddings=labeled_embeddings,
            unlabeled_embeddings=train_embeddings,
            labeled_targets=labeled_targets,
            n_select=n_select,
            random_seed=args.seed,
        )
        if strategy.last_proxy_stats:
            print("KITTI UHerding proxy stats:")
            print(json.dumps(strategy.last_proxy_stats, indent=2))
    elif strategy_name == "ppal":
        strategy = PPALStrategy(
            temperature=args.al_ppal_temperature,
            uncertainty=args.al_ppal_uncertainty,
            pool_expand_ratio=args.al_ppal_pool_expand_ratio,
            class_weight_ub=args.al_ppal_class_weight_ub,
            class_weight_alpha=args.al_ppal_class_weight_alpha,
            diversity_power=args.al_ppal_diversity_power,
            exact_max_select=(1 if args.al_fast_selection else 512),
            exact_max_pool=(1 if args.al_fast_selection else 4096),
        )
        selected_indices = strategy.select(
            labeled_embeddings=labeled_embeddings,
            unlabeled_embeddings=train_embeddings,
            labeled_targets=labeled_targets,
            n_select=n_select,
            random_seed=args.seed,
        )
        if strategy.last_proxy_stats:
            print("KITTI PPAL proxy stats:")
            print(json.dumps(strategy.last_proxy_stats, indent=2))
    else:
        raise ValueError(f"Unsupported embedding strategy: {strategy_name}")

    selected_ids = [train_ids[int(idx)] for idx in selected_indices]
    unique_selected = sorted(set(selected_ids))
    if len(unique_selected) < n_select:
        selected_set = set(unique_selected)
        for idx in np.argsort(train_targets):
            image_id = train_ids[int(idx)]
            if image_id not in selected_set:
                unique_selected.append(image_id)
                selected_set.add(image_id)
            if len(unique_selected) >= n_select:
                break

    print(
        f"KITTI {strategy_name} selection completed: selected {len(unique_selected)} "
        f"images using proxy label signatures {signature_to_label}."
    )
    return sorted(unique_selected[:n_select])


def _binary_entropy(values: np.ndarray) -> np.ndarray:
    probs = np.clip(values.astype(np.float64), 1e-6, 1.0 - 1e-6)
    return -(probs * np.log(probs) + (1.0 - probs) * np.log(1.0 - probs))


def _batched(sequence: Sequence, batch_size: int):
    step = max(int(batch_size), 1)
    for start in range(0, len(sequence), step):
        yield start, sequence[start : start + step]


def _select_entropy_ids(
    image_ids: Sequence[str],
    image_dir: Path,
    n_select: int,
    args,
) -> List[str]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for KITTI entropy selection."
        ) from exc

    if not args.resume_model.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_model}")

    print(
        "KITTI entropy selection: scoring "
        f"{len(image_ids)} train candidates with {args.resume_model}."
    )
    model = YOLO(str(args.resume_model))
    image_paths = [_resolve_image_path(image_dir, image_id) for image_id in image_ids]
    scores: List[float] = []

    for start, batch_paths in _batched(image_paths, args.al_entropy_batch):
        if start > 0 and start % 1000 == 0:
            print(f"  Entropy-scored {start}/{len(image_paths)} images...")
        results = model.predict(
            source=[str(path) for path in batch_paths],
            imgsz=args.imgsz,
            device=args.device,
            batch=len(batch_paths),
            conf=args.al_entropy_conf_thres,
            iou=args.al_entropy_iou,
            verbose=False,
            stream=False,
        )
        for result in results:
            boxes = getattr(result, "boxes", None)
            conf_tensor = getattr(boxes, "conf", None) if boxes is not None else None
            if conf_tensor is None or len(conf_tensor) == 0:
                scores.append(0.0)
                continue
            conf = conf_tensor.detach().float().cpu().numpy()
            entropy = _binary_entropy(conf)
            scores.append(float(entropy.max()))

    if len(scores) != len(image_ids):
        raise RuntimeError(
            f"Entropy selection scored {len(scores)} images, expected {len(image_ids)}."
        )

    scores_array = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores_array, kind="mergesort")[:n_select]
    selected_ids = [image_ids[int(idx)] for idx in order]
    selected_scores = scores_array[order]
    print(
        "KITTI entropy selection completed: "
        f"selected {len(selected_ids)} images; score range "
        f"{selected_scores.min():.6f}-{selected_scores.max():.6f}, "
        f"mean {selected_scores.mean():.6f}."
    )
    return sorted(selected_ids)


def _parse_focus_attributes(raw_focus: str) -> List[str]:
    return [part.strip() for part in str(raw_focus or "").split(",") if part.strip()]


def _resolve_attribute_extra_counts(args) -> Dict[str, Dict[str, float]]:
    from utils.active_learning import parse_attribute_value_counts

    if args.al_attribute_extra_counts:
        extra_counts = parse_attribute_value_counts(
            args.al_attribute_extra_counts,
            default_count=args.num_added_img,
        )
    else:
        extra_counts = {
            attr_name: {
                value: float(multiplier) * float(args.num_added_img)
                for value, multiplier in value_map.items()
            }
            for attr_name, value_map in DEFAULT_GENERATED_ATTRIBUTE_COUNTS.items()
        }

    scale = max(float(args.al_attribute_extra_count_scale), 0.0)
    if scale != 1.0:
        extra_counts = {
            attr_name: {
                value: float(count) * scale
                for value, count in value_map.items()
            }
            for attr_name, value_map in extra_counts.items()
        }
    return extra_counts


def _select_attribute_ids(
    image_ids: Sequence[str],
    label_dir: Path,
    n_select: int,
    args,
) -> List[str]:
    from utils.active_learning import (
        AttributeFocusedStrategy,
        load_attribute_metadata,
        parse_attribute_priority_log,
        parse_float_with_infinity,
    )

    if not args.attribute_file.exists():
        raise FileNotFoundError(f"Attribute metadata file not found: {args.attribute_file}")

    metadata = load_attribute_metadata(str(args.attribute_file))
    metadata_lengths = [
        len(attr_info.get("data", []))
        for attr_info in metadata.values()
        if isinstance(attr_info, dict)
    ]
    if not metadata_lengths:
        raise ValueError(f"No attribute data arrays found in {args.attribute_file}")
    if min(metadata_lengths) < len(image_ids):
        raise ValueError(
            f"Attribute metadata {args.attribute_file} covers {min(metadata_lengths)} images, "
            f"but {len(image_ids)} KITTI train images are available."
        )

    focus_attributes = _parse_focus_attributes(args.al_attribute_focus)
    extra_counts = _resolve_attribute_extra_counts(args)

    signatures = [_label_signature(label_dir / f"{image_id}.txt") for image_id in image_ids]
    signature_to_label = {
        signature: idx for idx, signature in enumerate(sorted(set(signatures)))
    }
    proxy_labels = [signature_to_label[signature] for signature in signatures]
    proxy_dataset = KittiSelectionProxyDataset(
        metadata_indices=list(range(len(image_ids))),
        class_labels=proxy_labels,
    )

    value_weights = None
    if args.al_attribute_log_path is not None and args.al_attribute_log_path.exists():
        value_weights = parse_attribute_priority_log(str(args.al_attribute_log_path))

    class_balance = parse_float_with_infinity(args.al_attribute_class_balance, default=0.0)
    strategy = AttributeFocusedStrategy(
        attribute_metadata=metadata,
        focus_attributes=focus_attributes,
        rarity_power=args.al_attribute_rarity_power,
        uncertain_boost=args.al_attribute_uncertainty_boost,
        value_weights=value_weights,
        log_acc_scale=args.al_attribute_log_acc_scale,
        extra_counts=extra_counts,
        frequency_penalty=args.al_attribute_frequency_penalty,
        class_balance_weight=class_balance,
        conditional_rarity=args.al_conditional_rarity,
        use_rarity=args.al_attribute_use_rarity,
        use_entropy=False,
    )

    debug_attribute = focus_attributes[0] if focus_attributes else None
    print("KITTI attribute selection config:")
    print(
        json.dumps(
            {
                "attribute_file": str(args.attribute_file),
                "focus_attributes": focus_attributes,
                "rarity_power": args.al_attribute_rarity_power,
                "use_rarity": args.al_attribute_use_rarity,
                "frequency_penalty": args.al_attribute_frequency_penalty,
                "class_balance": args.al_attribute_class_balance,
                "conditional_rarity": args.al_conditional_rarity,
                "extra_counts": extra_counts,
                "label_signature_to_proxy_class": signature_to_label,
            },
            indent=2,
        )
    )

    selected_indices = strategy.select(
        proxy_dataset,
        n_select,
        random_seed=args.seed,
        debug_attribute=debug_attribute,
    )
    selected_ids = [image_ids[int(idx)] for idx in selected_indices]
    return sorted(selected_ids)


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


def prepare_dataset(
    args,
    stage_seconds: Dict[str, float],
    stage_state: Dict[str, Any],
    timing_counts: Dict[str, Any],
) -> Tuple[Path, Dict[str, Any]]:
    _switch_active_stage(stage_seconds, stage_state, "setup_input_validation")
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
            yaml_path = args.prepared_root / "kitti_random_generated.yaml"
            summary_path = args.prepared_root / "selection_summary.json"
            reused_summary: Dict[str, Any] = {}
            if summary_path.exists():
                try:
                    reused_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    print(
                        f"Warning: could not read reused selection summary {summary_path}: {exc}",
                        file=sys.stderr,
                    )
            reused_counts = {
                key: reused_summary[key]
                for key in (
                    "original_candidates",
                    "selected_original",
                    "generated_candidates_with_labels",
                    "selected_generated",
                    "final_train_images",
                    "val_images",
                    "test_images",
                )
                if key in reused_summary
            }
            reused_counts["requested_generated"] = args.num_added_img
            reused_counts["require_exact_generated_count"] = args.require_exact_generated_count
            generated_manifest = args.prepared_root / "selected_generated_source_ids.txt"
            if "selected_generated" not in reused_counts and generated_manifest.exists():
                reused_counts["selected_generated"] = sum(
                    1
                    for line in generated_manifest.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            if "selected_generated" in reused_counts:
                reused_counts["generated_count_exact"] = (
                    reused_counts["selected_generated"] == args.num_added_img
                )
            timing_counts.update(reused_counts)
            if (
                args.require_exact_generated_count
                and reused_counts.get("selected_generated", 0) < args.num_added_img
            ):
                raise RuntimeError(
                    f"Reused prepared dataset contains "
                    f"{reused_counts.get('selected_generated', 0)} generated images, "
                    f"but exactly {args.num_added_img} were requested."
                )
            return yaml_path, {
                "counts": reused_counts,
                "prepared_dataset_reused": True,
                "selection_summary": reused_summary,
            }
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
    timing_counts["original_candidates"] = len(train_ids)
    timing_counts["requested_original"] = original_sample_count
    _switch_active_stage(stage_seconds, stage_state, "original_source_selection")
    if original_sample_count < len(train_ids):
        if args.selection_strategy == "stratified":
            selected_original_ids = _stratified_sample_ids(
                train_ids,
                base_train_labels,
                original_sample_count,
                rng,
            )
        elif args.selection_strategy in {"coreset", "badge", "uherding", "ppal"}:
            selected_original_ids = _select_embedding_ids(
                args.selection_strategy,
                train_ids,
                base_train_images,
                base_train_labels,
                base_val_images,
                base_val_labels,
                original_sample_count,
                args,
            )
        elif args.selection_strategy == "entropy":
            selected_original_ids = _select_entropy_ids(
                train_ids,
                base_train_images,
                original_sample_count,
                args,
            )
        elif args.selection_strategy == "attribute":
            selected_original_ids = _select_attribute_ids(
                train_ids,
                base_train_labels,
                original_sample_count,
                args,
            )
        else:
            selected_original_ids = sorted(rng.sample(train_ids, original_sample_count))
    else:
        selected_original_ids = list(train_ids)
    timing_counts["selected_original"] = len(selected_original_ids)

    _switch_active_stage(stage_seconds, stage_state, "generated_pool_discovery_sampling")
    original_selection_time = stage_seconds["original_source_selection"]
    print(f"Total KITTI original selection time: {original_selection_time:.2f} seconds")

    generated_candidates = _select_generated_candidates(args.generated_image_dir, base_train_labels)
    timing_counts["generated_candidates_with_labels"] = len(generated_candidates)
    if not generated_candidates:
        timing_counts["selected_generated"] = 0
        timing_counts["generated_count_exact"] = args.num_added_img == 0
        raise RuntimeError(
            f"No generated images in {args.generated_image_dir} have matching train labels "
            f"under {base_train_labels}."
        )
    generated_ids = list(generated_candidates)
    generated_sample_count = min(args.num_added_img, len(generated_ids))
    selected_generated_ids = sorted(rng.sample(generated_ids, generated_sample_count))
    timing_counts["selected_generated"] = len(selected_generated_ids)
    timing_counts["generated_count_exact"] = len(selected_generated_ids) == args.num_added_img
    if args.require_exact_generated_count and len(selected_generated_ids) < args.num_added_img:
        raise RuntimeError(
            f"Requested exactly {args.num_added_img} generated images, but only "
            f"{len(selected_generated_ids)} eligible images in {args.generated_image_dir} "
            f"have matching train labels under {base_train_labels}."
        )

    _switch_active_stage(stage_seconds, stage_state, "dataset_materialization")
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
        "al_initial_labeled": (
            args.al_initial_labeled if args.selection_strategy in {"coreset", "badge", "uherding", "ppal"} else None
        ),
        "al_embedding_thumbnail_size": (
            args.al_embedding_thumbnail_size if args.selection_strategy in {"coreset", "badge", "uherding", "ppal"} else None
        ),
        "al_fast_selection": (
            args.al_fast_selection if args.selection_strategy in {"coreset", "badge", "uherding", "ppal"} else None
        ),
        "al_ppal_temperature": (
            args.al_ppal_temperature if args.selection_strategy == "ppal" else None
        ),
        "al_ppal_uncertainty": (
            args.al_ppal_uncertainty if args.selection_strategy == "ppal" else None
        ),
        "al_ppal_pool_expand_ratio": (
            args.al_ppal_pool_expand_ratio if args.selection_strategy == "ppal" else None
        ),
        "al_ppal_class_weight_ub": (
            args.al_ppal_class_weight_ub if args.selection_strategy == "ppal" else None
        ),
        "al_ppal_class_weight_alpha": (
            args.al_ppal_class_weight_alpha if args.selection_strategy == "ppal" else None
        ),
        "al_ppal_diversity_power": (
            args.al_ppal_diversity_power if args.selection_strategy == "ppal" else None
        ),
        "al_uherding_temperature": (
            args.al_uherding_temperature if args.selection_strategy == "uherding" else None
        ),
        "al_uherding_uncertainty": (
            args.al_uherding_uncertainty if args.selection_strategy == "uherding" else None
        ),
        "al_uherding_kernel_scale": (
            args.al_uherding_kernel_scale if args.selection_strategy == "uherding" else None
        ),
        "al_entropy_conf_thres": (
            args.al_entropy_conf_thres if args.selection_strategy == "entropy" else None
        ),
        "al_entropy_iou": args.al_entropy_iou if args.selection_strategy == "entropy" else None,
        "al_entropy_batch": args.al_entropy_batch if args.selection_strategy == "entropy" else None,
        "attribute_file": str(args.attribute_file) if args.selection_strategy == "attribute" else None,
        "al_attribute_focus": args.al_attribute_focus if args.selection_strategy == "attribute" else None,
        "al_attribute_rarity_power": (
            args.al_attribute_rarity_power if args.selection_strategy == "attribute" else None
        ),
        "al_attribute_frequency_penalty": (
            args.al_attribute_frequency_penalty if args.selection_strategy == "attribute" else None
        ),
        "al_attribute_class_balance": (
            args.al_attribute_class_balance if args.selection_strategy == "attribute" else None
        ),
        "al_conditional_rarity": (
            args.al_conditional_rarity if args.selection_strategy == "attribute" else None
        ),
        "al_attribute_extra_counts": (
            _resolve_attribute_extra_counts(args) if args.selection_strategy == "attribute" else None
        ),
        "original_candidates": len(train_ids),
        "selected_original": len(selected_original_ids),
        "original_selection_time_seconds": original_selection_time,
        "generated_candidates_with_labels": len(generated_candidates),
        "requested_generated": args.num_added_img,
        "selected_generated": len(selected_generated_ids),
        "require_exact_generated_count": args.require_exact_generated_count,
        "generated_count_exact": len(selected_generated_ids) == args.num_added_img,
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
    _finish_active_stage(stage_seconds, stage_state)

    print("Prepared KITTI YOLO generated-training dataset:")
    print(json.dumps(summary, indent=2))
    print(f"Dataset YAML: {yaml_path}")
    if not args.require_exact_generated_count and len(selected_generated_ids) < args.num_added_img:
        print(
            f"Warning: requested {args.num_added_img} generated images, "
            f"but only {len(selected_generated_ids)} matched train labels."
        )
    return yaml_path, {
        "counts": {
            "original_candidates": len(train_ids),
            "selected_original": len(selected_original_ids),
            "generated_candidates_with_labels": len(generated_candidates),
            "requested_generated": args.num_added_img,
            "selected_generated": len(selected_generated_ids),
            "require_exact_generated_count": args.require_exact_generated_count,
            "generated_count_exact": len(selected_generated_ids) == args.num_added_img,
            "final_train_images": len(selected_original_ids) + len(selected_generated_ids),
            "val_images": len(val_ids),
            "test_images": len(test_ids),
        },
        "prepared_dataset_reused": False,
        "selection_summary": summary,
    }


def train_yolov8(
    args,
    data_yaml: Path,
    stage_seconds: Dict[str, float],
) -> Dict[str, Any]:
    model_load_start = time.perf_counter()
    try:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required. Use the KITTI fp environment described in object_detection/README.md."
            ) from exc

        if not args.resume_model.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_model}")

        run_name = _make_run_name(args)
        model = YOLO(str(args.resume_model))
    finally:
        stage_seconds["yolo_checkpoint_model_load"] += time.perf_counter() - model_load_start
    output_details: Dict[str, Any] = {
        "actual_run_dir": None,
        "best_checkpoint": None,
        "test_eval_dir": None,
        "test_evaluation_performed": False,
        "test_metrics": {},
    }

    train_start_time = time.perf_counter()
    try:
        train_results = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(Path(args.project).resolve()),
            name=run_name,
            seed=args.seed,
            plots=args.plots,
            optimizer=args.optimizer,
            lr0=args.lr0,
            lrf=args.lrf,
            warmup_epochs=args.warmup_epochs,
        )
    finally:
        train_elapsed = time.perf_counter() - train_start_time
        stage_seconds["model_train"] += train_elapsed
    print(f"Total training time for {args.epochs} epochs: {train_elapsed:.2f} seconds")
    print(f"Training elapsed (HH:MM:SS): {_format_duration(train_elapsed)}")
    print(train_results)

    save_dir = None
    best_weights = None
    if hasattr(model, "trainer") and model.trainer is not None:
        save_dir = Path(model.trainer.save_dir).resolve()
        output_details["actual_run_dir"] = str(save_dir)
        candidate = save_dir / "weights" / "best.pt"
        if candidate.exists():
            best_weights = candidate.resolve()
            output_details["best_checkpoint"] = str(best_weights)

    if args.eval_test and _dataset_has_split(data_yaml, "test"):
        if best_weights is not None:
            eval_load_start = time.perf_counter()
            try:
                eval_model = YOLO(str(best_weights))
            finally:
                stage_seconds["final_eval_checkpoint_load"] += (
                    time.perf_counter() - eval_load_start
                )
        else:
            eval_model = model

        print("Running final evaluation on dataset test split...")
        test_start_time = time.perf_counter()
        try:
            test_results = eval_model.val(
                data=str(data_yaml),
                split="test",
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                project=str(Path(args.project).resolve()),
                name=f"{run_name}_test",
            )
        finally:
            test_elapsed = time.perf_counter() - test_start_time
            stage_seconds["test_evaluation"] += test_elapsed
        output_details["test_evaluation_performed"] = True
        raw_metrics = getattr(test_results, "results_dict", {})
        if isinstance(raw_metrics, dict):
            for key, value in raw_metrics.items():
                try:
                    output_details["test_metrics"][str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        test_save_dir = getattr(test_results, "save_dir", None)
        if test_save_dir is not None:
            output_details["test_eval_dir"] = _resolved_path(test_save_dir)
        print(f"Test evaluation elapsed: {test_elapsed:.2f} seconds")
        print(f"Test evaluation elapsed (HH:MM:SS): {_format_duration(test_elapsed)}")
        print(test_results)
    elif args.eval_test:
        print("Dataset yaml has no test split configured; skipping final test evaluation.")

    return output_details


def main():
    command_start = time.perf_counter()
    start_wall_time = time.time()
    args = parse_args()
    run_name = _make_run_name(args)
    timing_record = _initial_timing_record(args, run_name, start_wall_time)
    log_path = None
    log_handle = None
    original_streams = None

    try:
        log_path, log_handle, original_streams = _setup_log(args, run_name, start_wall_time)
        timing_record["paths"]["log_file"] = _resolved_path(log_path)
        timing_record["environment"]["gpu"] = _collect_gpu_metadata()
        if log_path is not None:
            print(f"Log file: {log_path}")
        print("Current time:", datetime.datetime.now().strftime("%m%d%H%M%S"))
        print(f"Start timestamp: {_timestamp_str(start_wall_time)}")
        timing_record["stage_seconds"]["setup_input_validation"] += (
            time.perf_counter() - command_start
        )

        dataset_stage_state: Dict[str, Any] = {}
        try:
            data_yaml, dataset_details = prepare_dataset(
                args,
                timing_record["stage_seconds"],
                dataset_stage_state,
                timing_record["counts"],
            )
        finally:
            _finish_active_stage(timing_record["stage_seconds"], dataset_stage_state)
        timing_record["counts"].update(dataset_details["counts"])
        timing_record["provenance"]["prepared_dataset_reused"] = dataset_details[
            "prepared_dataset_reused"
        ]
        timing_record["provenance"]["selection_summary_snapshot"] = dataset_details[
            "selection_summary"
        ]

        if args.prepare_only:
            print("prepare-only mode enabled; skipping training.")
        else:
            output_details = train_yolov8(args, data_yaml, timing_record["stage_seconds"])
            timing_record["paths"]["actual_run_dir"] = output_details["actual_run_dir"]
            timing_record["paths"]["best_checkpoint"] = output_details["best_checkpoint"]
            timing_record["paths"]["test_eval_dir"] = output_details["test_eval_dir"]
            timing_record["provenance"]["test_evaluation_performed"] = output_details[
                "test_evaluation_performed"
            ]
            timing_record["evaluation"] = output_details["test_metrics"]

        end_timestamp = time.time()
        print(f"End timestamp: {_timestamp_str(end_timestamp)}")
        _finalize_timing_record(timing_record, command_start, status="success")
        print(
            "End-to-end elapsed (HH:MM:SS): "
            f"{_format_duration(timing_record['command_total_seconds'])}"
        )
        _write_timing_json(args.timing_json, timing_record)
    except BaseException as exc:
        _finalize_timing_record(timing_record, command_start, status="error", error=exc)
        _write_timing_json(args.timing_json, timing_record)
        raise
    finally:
        if log_handle is not None:
            log_handle.flush()
            if original_streams is not None:
                sys.stdout, sys.stderr = original_streams
            log_handle.close()


if __name__ == "__main__":
    main()
