#!/usr/bin/env bash
#SBATCH --job-name=img-qwen-annotation
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch --qos=juno-pri scripts/juno_imagenet_source_annotation_timing.sbatch
  bash scripts/juno_imagenet_source_annotation_timing.sbatch --dry-run

Measures fresh Qwen3-VL assignment time for the exact 10,400-image ImageNet10
training pool. The job stages only symlinks in node-local temporary storage and
writes annotations plus machine-readable timing to the durable Juno worktree.

Options:
  --repo-dir PATH    Default: /work/$USER/safefix_rebuttal
  --python PATH      Default: /work/$USER/venvs/qwen3/bin/python
  --data-dir PATH    Default: scratch ImageNet10 ImageFolder
  --train-index PATH Default: repo ImageNet10 train index
  --output-root PATH Default: durable Juno log root
  --dry-run          Print resolved paths without running
EOF
}

REPO_DIR="/work/${USER}/safefix_rebuttal"
PYTHON_EXE="/work/${USER}/venvs/qwen3/bin/python"
DATA_DIR="/home/${USER}/scratch/safefix_cache/data/imagenet_subset"
TRAIN_INDEX=""
OUTPUT_ROOT=""
DRY_RUN=0

require_value() {
  [[ $# -ge 2 && -n "$2" ]] || { echo "Missing value for $1" >&2; exit 2; }
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir) require_value "$@"; REPO_DIR="$2"; shift 2 ;;
    --python) require_value "$@"; PYTHON_EXE="$2"; shift 2 ;;
    --data-dir) require_value "$@"; DATA_DIR="$2"; shift 2 ;;
    --train-index) require_value "$@"; TRAIN_INDEX="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$TRAIN_INDEX" ]] || TRAIN_INDEX="$REPO_DIR/save/train_idx_imagenet.npy"
[[ -n "$OUTPUT_ROOT" ]] || OUTPUT_ROOT="$REPO_DIR/logs/juno/imagenet_source_annotation"

JOB_KEY="${SLURM_JOB_ID:-dryrun}"
TASK_DIR="$OUTPUT_ROOT/$JOB_KEY"
LOG_FILE="$TASK_DIR/imagenet10_train_qwen3_annotation.log"
OUTPUT_JSON="$TASK_DIR/attribute10_train_qwen3_8B.json"
OUTPUT_CSV="$TASK_DIR/attribute10_train_qwen3_8B.csv"
PARTIAL_JSON="$TASK_DIR/attribute10_train_qwen3_8B_partial.json"
TIMING_JSON="$TASK_DIR/attribute10_train_qwen3_8B_timing.json"
NOTIFICATION_ENV_FILE="$REPO_DIR/.vscode/.env"
STAGE_DIR="${TMPDIR:-/tmp}/safefix_imagenet10_train_${JOB_KEY}"

print_resolved_task() {
  echo "repo_dir=$REPO_DIR"
  echo "python=$PYTHON_EXE"
  echo "data_dir=$DATA_DIR"
  echo "train_index=$TRAIN_INDEX"
  echo "stage_dir=$STAGE_DIR"
  echo "task_dir=$TASK_DIR"
  echo "timing_json=$TIMING_JSON"
}

if [[ "$DRY_RUN" -eq 1 ]]; then
  print_resolved_task
  exit 0
fi

[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo "Submit with sbatch, or pass --dry-run." >&2
  exit 2
}

for required_file in \
  "$PYTHON_EXE" \
  "$REPO_DIR/utils/assign_attribute.py" \
  "$REPO_DIR/save/corpus10.json" \
  "$TRAIN_INDEX"; do
  [[ -s "$required_file" ]] || { echo "Missing or empty file: $required_file" >&2; exit 1; }
done
[[ -d "$DATA_DIR" ]] || { echo "Missing data directory: $DATA_DIR" >&2; exit 1; }

mkdir -p "$TASK_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
print_resolved_task

if [[ -z "${NOTIFICATION_WEBHOOK:-}" && -f "$NOTIFICATION_ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$NOTIFICATION_ENV_FILE"
  set +a
fi

cleanup_and_notify() {
  local exit_code="$1"
  [[ "$exit_code" -ne 0 ]] || return 0
  [[ -n "${NOTIFICATION_WEBHOOK:-}" ]] || return 0
  python3 "$REPO_DIR/scripts/test_notification.py" \
    "❌ ImageNet source-annotation timing job $JOB_KEY failed with exit $exit_code: artifacts=$TASK_DIR" \
    || true
}
trap 'cleanup_and_notify $?' EXIT

echo "host=$(hostname)"
echo "slurm_job_id=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

[[ ! -e "$STAGE_DIR" ]] || {
  echo "Refusing to overwrite existing staging path: $STAGE_DIR" >&2
  exit 1
}
mkdir -p "$STAGE_DIR"
"$PYTHON_EXE" - "$DATA_DIR" "$TRAIN_INDEX" "$STAGE_DIR" <<'PY'
import os
import sys
from pathlib import Path

import numpy as np

data_dir = Path(sys.argv[1]).resolve()
train_index_path = Path(sys.argv[2]).resolve()
stage_dir = Path(sys.argv[3]).resolve()
extensions = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}

classes = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
samples = []
for class_name in classes:
    class_dir = data_dir / class_name
    for root, _, filenames in sorted(os.walk(class_dir, followlinks=True)):
        for filename in sorted(filenames):
            path = Path(root) / filename
            if path.suffix.lower() in extensions:
                samples.append((path.resolve(), class_name))

train_indices = np.asarray(np.load(train_index_path), dtype=np.int64)
if len(samples) != 13000:
    raise RuntimeError(f"Expected 13000 ImageNet10 images, found {len(samples)}")
if len(train_indices) != 10400 or len(np.unique(train_indices)) != 10400:
    raise RuntimeError(f"Expected 10400 unique training indices, found {len(train_indices)}")
if train_indices.min() < 0 or train_indices.max() >= len(samples):
    raise RuntimeError("Training index outside ImageFolder sample range")

for source_index in train_indices.tolist():
    source_path, class_name = samples[source_index]
    class_stage = stage_dir / class_name
    class_stage.mkdir(parents=True, exist_ok=True)
    target = class_stage / f"{source_index:05d}_{source_path.name}"
    target.symlink_to(source_path)

staged = sum(1 for path in stage_dir.rglob("*") if path.is_symlink())
if staged != 10400:
    raise RuntimeError(f"Expected 10400 staged symlinks, found {staged}")
print(f"staged_images={staged}")
PY

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="/work/${USER}/hf_cache"
export CUDA_VISIBLE_DEVICES=0

"$PYTHON_EXE" -u "$REPO_DIR/utils/assign_attribute.py" \
  --dataset folder \
  --root_dir "$STAGE_DIR" \
  --corpus "$REPO_DIR/save/corpus10.json" \
  --output "$OUTPUT_JSON" \
  --csv-output "$OUTPUT_CSV" \
  --partial "$PARTIAL_JSON" \
  --timing-json "$TIMING_JSON" \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --max_memory_per_gpu 120 \
  --save_every 500 \
  --no-env-reexec \
  --fail-on-inference-error

for output in "$OUTPUT_JSON" "$OUTPUT_CSV" "$TIMING_JSON"; do
  [[ -s "$output" ]] || { echo "Missing output: $output" >&2; exit 1; }
done

"$PYTHON_EXE" - "$TIMING_JSON" <<'PY'
import json
import sys

timing = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print("timing_summary", json.dumps({
    "status": timing.get("status"),
    "counts": timing.get("counts"),
    "stages_seconds": timing.get("stages_seconds"),
    "inference": timing.get("inference"),
}, sort_keys=True))
PY

echo "completed ImageNet10 source annotation timing job=$JOB_KEY artifacts=$TASK_DIR"
