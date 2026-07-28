#!/usr/bin/env bash
#SBATCH --job-name=safefix-kitti-cost
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: sbatch scripts/juno_kitti_cost_accounting.sbatch [options]
       bash scripts/juno_kitti_cost_accounting.sbatch --dry-run [options]

Run KITTI YOLOv8 repair with Random, Entropy, or AttriSelect source selection
and write cost_summary.json/.md/.tsv. AttriSelect includes source-attribute
annotation; Random and Entropy skip Qwen annotation.

For the paired Random/Ours comparison, submit one throttled array without an
explicit --selection-strategy:

  sbatch --array=0-1%1 scripts/juno_kitti_cost_accounting.sbatch \
    --mode full --ratio 0.1

Array task 0 runs Random and task 1 runs Ours. The %1 throttle uses one GPU
sequentially; change it to %2 only when two concurrent GPUs are intended.
Use `--array-methods entropy,ours` for an Entropy/Ours pair.
For a mixed method/ratio grid, pass equal-length `--array-methods` and
`--array-ratios` lists; each array task consumes the entries at its task ID.

Options:
  --mode smoke|full             Default: smoke
  --ratio FLOAT                 Override the mode ratio, for example 0.1.
  --selection-strategy METHOD   random|entropy|coreset|badge|attribute|ours
  --array-methods METHODS       Comma-separated array mapping. Default: random,ours
  --array-ratios RATIOS         Optional comma-separated per-task ratio mapping
  --skip-source-annotation      Reuse canonical metadata and exclude fresh Qwen assignment
  --repo-dir PATH               Default: /work/$USER/safefix_rebuttal
  --data-root PATH              Default: /work/$USER/data
  --hf-home PATH                Default: /work/$USER/hf_cache
  --qwen-python PATH            Default: /work/$USER/venvs/qwen3/bin/python
  --train-python PATH           Default: /work/$USER/venvs/kitti/bin/python
  --kitti-yolo-root PATH        KITTI first-5000 YOLO root
  --generated-image-dir PATH    Generated red-cloudy image pool
  --source-attribute-file PATH  Canonical existing source attribute JSON used by repair
  --corpus PATH                 KITTI attribute corpus JSON used by Qwen annotation
  --diagnosis-checkpoint PATH   YOLO diagnosis/base checkpoint used to initialize repair
  --scratch-root PATH           Default: /home/$USER/scratch/safefix_rebuttal/kitti_cost
  --model MODEL_ID              Default: Qwen/Qwen3-VL-8B-Instruct
  --qwen-max-memory-gb N        Default: 120
  --dry-run                     Print resolved settings and commands without executing
  --help                        Show this help

Mode settings:
  smoke: ratio 0.01; 10 generated; 1 epoch
  full:  ratio 0.9 unless overridden; 1000 generated; 10 epochs

Both modes use batch=16, imgsz=640, AdamW, lr0=2e-4, lrf=0.1,
warmup_epochs=1, seed=42, attribute R=0.03/P=5/C=15, and final test eval.
Repair-image generation is outside this job's accounting scope. The generated
red/cloudy counts are supplied directly, so generated-image attribute assignment
is recorded as not_applicable.
EOF
}

MODE="smoke"
RATIO_OVERRIDE=""
SELECTION_STRATEGY="attribute"
SELECTION_STRATEGY_EXPLICIT=0
ARRAY_METHODS="random,ours"
ARRAY_RATIOS=""
REPO_DIR="/work/${USER}/safefix_rebuttal"
DATA_ROOT="/work/${USER}/data"
HF_HOME_DIR="/work/${USER}/hf_cache"
QWEN_PYTHON="/work/${USER}/venvs/qwen3/bin/python"
TRAIN_PYTHON="${TRAIN_PYTHON:-/work/${USER}/venvs/kitti/bin/python}"
KITTI_YOLO_ROOT=""
GENERATED_IMAGE_DIR=""
SOURCE_ATTRIBUTE_FILE=""
CORPUS_FILE=""
DIAGNOSIS_CHECKPOINT=""
SCRATCH_BASE="/home/${USER}/scratch/safefix_rebuttal/kitti_cost"
MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
QWEN_MAX_MEMORY_GB="120"
SKIP_SOURCE_ANNOTATION=0
DRY_RUN=0

require_value() {
  [[ $# -ge 2 && -n "$2" ]] || { echo "Missing value for $1" >&2; exit 2; }
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) require_value "$@"; MODE="$2"; shift 2 ;;
    --ratio) require_value "$@"; RATIO_OVERRIDE="$2"; shift 2 ;;
    --selection-strategy) require_value "$@"; SELECTION_STRATEGY="$2"; SELECTION_STRATEGY_EXPLICIT=1; shift 2 ;;
    --array-methods) require_value "$@"; ARRAY_METHODS="$2"; shift 2 ;;
    --array-ratios) require_value "$@"; ARRAY_RATIOS="$2"; shift 2 ;;
    --repo-dir) require_value "$@"; REPO_DIR="$2"; shift 2 ;;
    --data-root) require_value "$@"; DATA_ROOT="$2"; shift 2 ;;
    --hf-home) require_value "$@"; HF_HOME_DIR="$2"; shift 2 ;;
    --qwen-python) require_value "$@"; QWEN_PYTHON="$2"; shift 2 ;;
    --train-python) require_value "$@"; TRAIN_PYTHON="$2"; shift 2 ;;
    --kitti-yolo-root) require_value "$@"; KITTI_YOLO_ROOT="$2"; shift 2 ;;
    --generated-image-dir) require_value "$@"; GENERATED_IMAGE_DIR="$2"; shift 2 ;;
    --source-attribute-file) require_value "$@"; SOURCE_ATTRIBUTE_FILE="$2"; shift 2 ;;
    --corpus) require_value "$@"; CORPUS_FILE="$2"; shift 2 ;;
    --diagnosis-checkpoint) require_value "$@"; DIAGNOSIS_CHECKPOINT="$2"; shift 2 ;;
    --scratch-root) require_value "$@"; SCRATCH_BASE="$2"; shift 2 ;;
    --model) require_value "$@"; MODEL_ID="$2"; shift 2 ;;
    --qwen-max-memory-gb) require_value "$@"; QWEN_MAX_MEMORY_GB="$2"; shift 2 ;;
    --skip-source-annotation) SKIP_SOURCE_ANNOTATION=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "$SELECTION_STRATEGY_EXPLICIT" -eq 0 ]]; then
  IFS=',' read -r -a ARRAY_METHOD_LIST <<< "$ARRAY_METHODS"
  if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#ARRAY_METHOD_LIST[@]} )); then
    echo "Array task ID $SLURM_ARRAY_TASK_ID is outside --array-methods $ARRAY_METHODS" >&2
    exit 2
  fi
  SELECTION_STRATEGY="${ARRAY_METHOD_LIST[$SLURM_ARRAY_TASK_ID]}"
fi

ARRAY_RATIO_OVERRIDE=""
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -n "$ARRAY_RATIOS" ]]; then
  IFS=',' read -r -a ARRAY_RATIO_LIST <<< "$ARRAY_RATIOS"
  IFS=',' read -r -a ARRAY_METHOD_LIST_FOR_RATIO_CHECK <<< "$ARRAY_METHODS"
  if (( ${#ARRAY_RATIO_LIST[@]} != ${#ARRAY_METHOD_LIST_FOR_RATIO_CHECK[@]} )); then
    echo "--array-methods and --array-ratios must have equal lengths" >&2
    exit 2
  fi
  if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#ARRAY_RATIO_LIST[@]} )); then
    echo "Array task ID $SLURM_ARRAY_TASK_ID is outside --array-ratios $ARRAY_RATIOS" >&2
    exit 2
  fi
  ARRAY_RATIO_OVERRIDE="${ARRAY_RATIO_LIST[$SLURM_ARRAY_TASK_ID]}"
fi

case "$MODE" in
  smoke)
    QWEN_STOP_AT=2
    SUB_RATIO="0.01"
    NUM_GENERATED=10
    EPOCHS=1
    ;;
  full)
    QWEN_STOP_AT=5000
    SUB_RATIO="0.9"
    NUM_GENERATED=1000
    EPOCHS=10
    ;;
  *) echo "--mode must be smoke or full, got: $MODE" >&2; exit 2 ;;
esac

case "${SELECTION_STRATEGY,,}" in
  random) SELECTION_STRATEGY="random" ;;
  entropy) SELECTION_STRATEGY="entropy" ;;
  coreset) SELECTION_STRATEGY="coreset" ;;
  badge) SELECTION_STRATEGY="badge" ;;
  attribute|ours) SELECTION_STRATEGY="attribute" ;;
  *) echo "--selection-strategy must be random, entropy, coreset, badge, attribute, or ours, got: $SELECTION_STRATEGY" >&2; exit 2 ;;
esac

if [[ -n "$RATIO_OVERRIDE" ]]; then
  SUB_RATIO="$RATIO_OVERRIDE"
fi
if [[ -n "$ARRAY_RATIO_OVERRIDE" ]]; then
  SUB_RATIO="$ARRAY_RATIO_OVERRIDE"
fi
if ! awk -v value="$SUB_RATIO" 'BEGIN { exit !(value > 0.0 && value <= 1.0) }'; then
  echo "--ratio must be in (0, 1], got: $SUB_RATIO" >&2
  exit 2
fi

RUN_ANNOTATION=0
if [[ "$SELECTION_STRATEGY" == "attribute" && "$SKIP_SOURCE_ANNOTATION" -eq 0 ]]; then
  RUN_ANNOTATION=1
fi

choose_existing() {
  local kind="$1"
  shift
  local candidate
  for candidate in "$@"; do
    [[ -n "$candidate" ]] || continue
    case "$kind" in
      dir) [[ -d "$candidate" ]] && { printf '%s\n' "$candidate"; return 0; } ;;
      file) [[ -f "$candidate" ]] && { printf '%s\n' "$candidate"; return 0; } ;;
      executable) [[ -x "$candidate" ]] && { printf '%s\n' "$candidate"; return 0; } ;;
    esac
  done
  return 1
}

resolve_or_first() {
  local explicit="$1"
  local kind="$2"
  shift 2
  if [[ -n "$explicit" ]]; then
    printf '%s\n' "$explicit"
    return 0
  fi
  choose_existing "$kind" "$@" 2>/dev/null || printf '%s\n' "$1"
}

KITTI_YOLO_ROOT=$(resolve_or_first "$KITTI_YOLO_ROOT" dir \
  "$DATA_ROOT/kitti_yolov8_first5000_1240val_1241test" \
  "$DATA_ROOT/kitti_yolov8_first5000" \
  "$DATA_ROOT/kitti/kitti_yolov8_first5000" \
  "$DATA_ROOT/kitti_object/kitti_yolov8_first5000" \
  "$REPO_DIR/data/kitti_yolov8_first5000")
GENERATED_IMAGE_DIR=$(resolve_or_first "$GENERATED_IMAGE_DIR" dir \
  "$DATA_ROOT/kitti-weather-edits/generated_kitti1352_red_cloudy_icedit_upscaled" \
  "$DATA_ROOT/kitti_weather_edits/generated_kitti1352_red_cloudy_icedit_upscaled" \
  "$DATA_ROOT/generated_kitti1352_red_cloudy_icedit_upscaled" \
  "$REPO_DIR/data/generated_kitti1352_red_cloudy_icedit_upscaled")
SOURCE_ATTRIBUTE_FILE=$(resolve_or_first "$SOURCE_ATTRIBUTE_FILE" file \
  "$REPO_DIR/save/attribute_kitti_qwen3_8B.json" \
  "$DATA_ROOT/kitti_metadata/attribute_kitti_qwen3_8B.json" \
  "$DATA_ROOT/attribute_kitti_qwen3_8B.json")
CORPUS_FILE=$(resolve_or_first "$CORPUS_FILE" file \
  "$REPO_DIR/save/corpus_kitti_gemini_clean.json" \
  "$REPO_DIR/save/corpus_kitti_object_gpt_clean.json" \
  "$DATA_ROOT/kitti_metadata/corpus_kitti_gemini_clean.json")
DIAGNOSIS_CHECKPOINT=$(resolve_or_first "$DIAGNOSIS_CHECKPOINT" file \
  "$REPO_DIR/save/runs/kitti_yolov8/first5000_1240val_1241test_gpu0_bs16_e30_20260429_012232/weights/best.pt" \
  "$DATA_ROOT/checkpoints/kitti_yolov8_first5000_best.pt" \
  "$DATA_ROOT/kitti_yolov8_first5000/weights/best.pt")

if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
  JOB_ID="${SLURM_JOB_ID:-dryrun-$$}"
fi
JOB_DIR="$REPO_DIR/logs/juno/kitti_cost/$JOB_ID"
SCRATCH_JOB_DIR="$SCRATCH_BASE/$JOB_ID"
PREPARED_ROOT="$SCRATCH_JOB_DIR/prepared/kitti_${MODE}_${SELECTION_STRATEGY}_ratio${SUB_RATIO//./p}_gen${NUM_GENERATED}"
RUN_PROJECT="$SCRATCH_JOB_DIR/runs"
RUN_NAME="kitti_cost_${MODE}_${SELECTION_STRATEGY}_ratio${SUB_RATIO//./p}_job${JOB_ID}"
ANNOTATION_OUTPUT="$JOB_DIR/source_attributes_qwen3_isolated.json"
ANNOTATION_CSV="$JOB_DIR/source_attributes_qwen3_isolated.csv"
ANNOTATION_PARTIAL="$JOB_DIR/source_attributes_qwen3_isolated.partial.json"
ANNOTATION_TIMING="$JOB_DIR/annotation_timing.json"
REPAIR_TIMING="$JOB_DIR/repair_timing.json"
ANNOTATION_LOG="$JOB_DIR/annotation.log"
REPAIR_LOG="$JOB_DIR/repair.log"
COMMAND_LOG="$JOB_DIR/commands.sh"
SUMMARY_JSON="$JOB_DIR/cost_summary.json"
SUMMARY_MD="$JOB_DIR/cost_summary.md"
SUMMARY_TSV="$JOB_DIR/cost_summary.tsv"
NOTIFICATION_ENV_FILE="$REPO_DIR/.vscode/.env"

ANNOTATION_CMD=(
  "$QWEN_PYTHON" -u "$REPO_DIR/utils/assign_attribute.py"
  --dataset kitti
  --kitti_root "$KITTI_YOLO_ROOT"
  --split train
  --corpus "$CORPUS_FILE"
  --model "$MODEL_ID"
  --output "$ANNOTATION_OUTPUT"
  --csv-output "$ANNOTATION_CSV"
  --partial "$ANNOTATION_PARTIAL"
  --stop_at "$QWEN_STOP_AT"
  --save_every 500
  --max_memory_per_gpu "$QWEN_MAX_MEMORY_GB"
  --no-env-reexec
  --fail-on-inference-error
  --timing-json "$ANNOTATION_TIMING"
)

REPAIR_CMD=(
  "$TRAIN_PYTHON" -u "$REPO_DIR/object_detection/train_yolov8_kitti_generated.py"
  --base-data-root "$KITTI_YOLO_ROOT"
  --generated-image-dir "$GENERATED_IMAGE_DIR"
  --prepared-root "$PREPARED_ROOT"
  --resume-model "$DIAGNOSIS_CHECKPOINT"
  --sub-ratio "$SUB_RATIO"
  --selection-strategy "$SELECTION_STRATEGY"
  --attribute-file "$SOURCE_ATTRIBUTE_FILE"
  --al-attribute-focus color,weather
  --al-attribute-rarity-power 0.03
  --al-attribute-use-rarity true
  --al-attribute-frequency-penalty 5
  --al-attribute-class-balance 15
  --al-attribute-extra-counts "color:red=${NUM_GENERATED};weather:cloudy=${NUM_GENERATED}"
  --num-added-img "$NUM_GENERATED"
  --require-exact-generated-count true
  --epochs "$EPOCHS"
  --batch 16
  --imgsz 640
  --optimizer AdamW
  --lr0 2e-4
  --lrf 0.1
  --warmup-epochs 1
  --seed 42
  --device 0
  --workers 4
  --project "$RUN_PROJECT"
  --name "$RUN_NAME"
  --plots false
  --eval-test true
  --copy-images false
  --overwrite-prepared true
  --write-log false
  --timing-json "$REPAIR_TIMING"
)

if [[ "$SELECTION_STRATEGY" == "coreset" || "$SELECTION_STRATEGY" == "badge" ]]; then
  REPAIR_CMD+=(
    --al-fast-selection true
    --al-initial-labeled 400
    --al-embedding-thumbnail-size 8
  )
fi

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf 'mode=%s\nratio=%s\nselection_strategy=%s\nrun_annotation=%s\nskip_source_annotation=%s\nrepo_dir=%s\ndata_root=%s\nhf_home=%s\n' \
    "$MODE" "$SUB_RATIO" "$SELECTION_STRATEGY" "$RUN_ANNOTATION" "$SKIP_SOURCE_ANNOTATION" "$REPO_DIR" "$DATA_ROOT" "$HF_HOME_DIR"
  printf 'qwen_python=%s\ntrain_python=%s\nkitti_yolo_root=%s\n' "$QWEN_PYTHON" "$TRAIN_PYTHON" "$KITTI_YOLO_ROOT"
  printf 'generated_image_dir=%s\nsource_attribute_file=%s\ncorpus=%s\n' "$GENERATED_IMAGE_DIR" "$SOURCE_ATTRIBUTE_FILE" "$CORPUS_FILE"
  printf 'diagnosis_checkpoint=%s\njob_dir=%s\nscratch_job_dir=%s\n' "$DIAGNOSIS_CHECKPOINT" "$JOB_DIR" "$SCRATCH_JOB_DIR"
  if [[ "$RUN_ANNOTATION" -eq 1 ]]; then
    printf '\nAnnotation command:\n'; print_command "${ANNOTATION_CMD[@]}"
  else
    printf '\nAnnotation command: skipped for %s selection\n' "$SELECTION_STRATEGY"
  fi
  printf '\nRepair command:\n'; print_command "${REPAIR_CMD[@]}"
  exit 0
fi

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Submit this script with sbatch, or use --dry-run." >&2; exit 2; }
mkdir -p "$JOB_DIR" "$SCRATCH_JOB_DIR/prepared" "$RUN_PROJECT"
exec > >(tee -a "$JOB_DIR/job.log") 2>&1

if [[ -z "${NOTIFICATION_WEBHOOK:-}" && -f "$NOTIFICATION_ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$NOTIFICATION_ENV_FILE"
  set +a
fi

cleanup_and_notify() {
  local exit_code="$1"
  [[ -n "${NOTIFICATION_WEBHOOK:-}" ]] || return 0
  [[ -f "$REPO_DIR/scripts/test_notification.py" ]] || return 0
  local message
  if [[ "$exit_code" -eq 0 && -s "$SUMMARY_JSON" ]]; then
    message=$(python3 - "$SUMMARY_JSON" "$JOB_DIR" <<'PY'
import json
import sys

summary_path, artifact_dir = sys.argv[1:]
with open(summary_path, encoding="utf-8") as handle:
    summary = json.load(handle)

seconds = summary["seconds"]
counts = summary["counts"]["repair"]
evaluation = summary.get("evaluation", {})
method = {
    "attribute": "Ours",
    "badge": "BADGE",
    "coreset": "CoreSet",
    "entropy": "Entropy",
    "random": "Random",
}.get(summary["selection_strategy"], summary["selection_strategy"])
annotation = seconds["source_attribute_annotation_total"]
repair = seconds["repair_total"]
total = seconds["included_end_to_end_total"]
map_50_95 = evaluation.get("metrics/mAP50-95(B)")
metric_text = f", mAP50-95=`{map_50_95:.4f}`" if map_50_95 is not None else ""
print(
    f"✅ *SafeFix KITTI cost task `{summary['job_id']}` completed*\n"
    f"📊 *Report:* method=`{method}`, ratio=`{summary['ratio']}`, total=`{total / 60:.2f} min`\n"
    f"⏱️ annotation=`{annotation:.2f}s`, repair=`{repair:.2f}s`\n"
    f"🧪 selected=`{counts['selected_original']} source + {counts['selected_generated']} generated`, "
    f"test=`{counts['test_images']} images`{metric_text}\n"
    f"📁 Artifacts: `{artifact_dir}`"
)
PY
)
  else
    printf -v message \
      '❌ *SafeFix KITTI cost task `%s` failed*\n🧪 method=`%s`, ratio=`%s`, exit=`%s`\n📁 Artifacts: `%s`' \
      "$JOB_ID" "$SELECTION_STRATEGY" "$SUB_RATIO" "$exit_code" "$JOB_DIR"
  fi
  python3 "$REPO_DIR/scripts/test_notification.py" "$message" || true
}
trap 'cleanup_and_notify $?' EXIT

fail() { echo "ERROR: $*" >&2; exit 1; }
require_file() { [[ -s "$1" ]] || fail "Missing or empty $2: $1"; }
require_dir() { [[ -d "$1" ]] || fail "Missing $2 directory: $1"; }

require_dir "$REPO_DIR" "repository"
require_file "$REPO_DIR/object_detection/train_yolov8_kitti_generated.py" "repair script"
require_dir "$KITTI_YOLO_ROOT" "KITTI YOLO root"
require_dir "$GENERATED_IMAGE_DIR" "generated red-cloudy pool"
require_file "$SOURCE_ATTRIBUTE_FILE" "canonical source attribute file"
require_file "$CORPUS_FILE" "KITTI corpus"
require_file "$DIAGNOSIS_CHECKPOINT" "diagnosis checkpoint"
if [[ "$RUN_ANNOTATION" -eq 1 ]]; then
  require_file "$REPO_DIR/utils/assign_attribute.py" "annotation script"
  require_dir "$HF_HOME_DIR" "Hugging Face cache"
  [[ -x "$QWEN_PYTHON" ]] || fail "Qwen Python is not executable: $QWEN_PYTHON"
fi
if [[ "$TRAIN_PYTHON" == */* ]]; then
  [[ -x "$TRAIN_PYTHON" ]] || fail "Training Python is not executable: $TRAIN_PYTHON"
else
  TRAIN_PYTHON=$(command -v "$TRAIN_PYTHON" || true)
  [[ -n "$TRAIN_PYTHON" ]] || fail "Training Python was not found on PATH"
  REPAIR_CMD[0]="$TRAIN_PYTHON"
fi

export HF_HOME="$HF_HOME_DIR"
export HF_HUB_CACHE="$HF_HOME_DIR/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME_DIR/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

if [[ "$RUN_ANNOTATION" -eq 1 ]]; then
"$QWEN_PYTHON" - <<'PY'
import importlib
for name in ("torch", "transformers", "PIL"):
    importlib.import_module(name)
import torch
if not torch.cuda.is_available():
    raise SystemExit("Qwen Python cannot access the allocated CUDA GPU")
PY
fi

"$TRAIN_PYTHON" - <<'PY'
import torch
import ultralytics
if not torch.cuda.is_available():
    raise SystemExit("Training Python imports torch+ultralytics but cannot access CUDA")
print(f"train_import_check torch={torch.__version__} ultralytics={ultralytics.__version__}")
PY

if [[ "$RUN_ANNOTATION" -eq 1 ]]; then
  MODEL_CACHE_DIR="$HF_HOME_DIR/hub/models--${MODEL_ID//\//--}"
  [[ -f "$MODEL_CACHE_DIR/refs/main" ]] || fail "Offline model ref is missing: $MODEL_CACHE_DIR/refs/main"
  MODEL_SHARD=$(find -L "$MODEL_CACHE_DIR/snapshots" -type f -name '*.safetensors' -size +100M -print -quit 2>/dev/null || true)
  [[ -n "$MODEL_SHARD" ]] || fail "Offline model cache has no substantial safetensors shard: $MODEL_CACHE_DIR"
fi

PREFLIGHT_JSON="$JOB_DIR/preflight.json"
"$TRAIN_PYTHON" - "$KITTI_YOLO_ROOT" "$GENERATED_IMAGE_DIR" "$SOURCE_ATTRIBUTE_FILE" "$CORPUS_FILE" "$NUM_GENERATED" "$PREFLIGHT_JSON" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
generated = Path(sys.argv[2]).resolve()
attribute_path = Path(sys.argv[3]).resolve()
corpus_path = Path(sys.argv[4]).resolve()
requested_generated = int(sys.argv[5])
output = Path(sys.argv[6]).resolve()
extensions = {".jpg", ".jpeg", ".png", ".bmp"}

def images(directory):
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in extensions)

def require(condition, message):
    if not condition:
        raise SystemExit(message)

yaml_path = root / "kitti_first5000.yaml"
train_images = images(root / "images" / "train")
val_images = images(root / "images" / "val")
test_images = images(root / "images" / "test")
train_labels = root / "labels" / "train"
require(yaml_path.is_file(), f"Missing KITTI YAML: {yaml_path}")
require(len(train_images) >= 5000, f"Expected at least 5000 KITTI train images, found {len(train_images)}")
require(val_images, "KITTI validation images are empty")
require(test_images, "KITTI test images are empty; final test evaluation is required")
missing_train_labels = [path.stem for path in train_images[:5000] if not (train_labels / f"{path.stem}.txt").is_file()]
require(not missing_train_labels, f"Missing train labels for {len(missing_train_labels)} source images; first={missing_train_labels[:5]}")
generated_images = images(generated)
generated_with_labels = [path for path in generated_images if (train_labels / f"{path.stem}.txt").is_file()]
require(
    len(generated_with_labels) >= requested_generated,
    f"Need {requested_generated} generated images with matching source labels, found {len(generated_with_labels)}",
)
with attribute_path.open(encoding="utf-8") as handle:
    attributes = json.load(handle)
require(isinstance(attributes, dict), "Source attribute JSON must be an object")
for key in ("color", "weather"):
    require(key in attributes, f"Source attribute JSON is missing '{key}'")
    values = attributes[key].get("data") if isinstance(attributes[key], dict) else None
    require(isinstance(values, list) and len(values) >= 5000, f"Attribute '{key}' covers fewer than 5000 images")
with corpus_path.open(encoding="utf-8") as handle:
    corpus_payload = json.load(handle)
corpus = corpus_payload.get("Corpus", corpus_payload) if isinstance(corpus_payload, dict) else None
require(isinstance(corpus, dict), "Corpus JSON must be an object or contain a Corpus object")
for key in ("color", "weather"):
    require(key in corpus, f"Corpus is missing '{key}'")
    words = corpus[key].get("word") if isinstance(corpus[key], dict) else None
    require(isinstance(words, list) and words, f"Corpus attribute '{key}' has no word list")
record = {
    "kitti_yaml": str(yaml_path),
    "source_train_images": len(train_images),
    "source_val_images": len(val_images),
    "source_test_images": len(test_images),
    "generated_images_total": len(generated_images),
    "generated_images_with_matching_labels": len(generated_with_labels),
    "generated_images_requested": requested_generated,
    "source_attribute_lengths": {
        key: len(value.get("data", []))
        for key, value in attributes.items()
        if isinstance(value, dict) and isinstance(value.get("data"), list)
    },
    "corpus_attributes": list(corpus),
}
output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(record, indent=2, sort_keys=True))
PY

cd "$REPO_DIR"
{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "job_id=$SLURM_JOB_ID"
  echo "job_name=${SLURM_JOB_NAME:-}"
  echo "partition=${SLURM_JOB_PARTITION:-}"
  echo "mode=$MODE"
  echo "ratio=$SUB_RATIO"
  echo "selection_strategy=$SELECTION_STRATEGY"
  echo "run_annotation=$RUN_ANNOTATION"
  echo "repo_dir=$REPO_DIR"
  echo "data_root=$DATA_ROOT"
  echo "hf_home=$HF_HOME"
  echo "model=$MODEL_ID"
  echo "qwen_python=$QWEN_PYTHON"
  echo "train_python=$TRAIN_PYTHON"
  echo "kitti_yolo_root=$KITTI_YOLO_ROOT"
  echo "generated_image_dir=$GENERATED_IMAGE_DIR"
  echo "source_attribute_file=$SOURCE_ATTRIBUTE_FILE"
  echo "corpus=$CORPUS_FILE"
  echo "diagnosis_checkpoint=$DIAGNOSIS_CHECKPOINT"
  echo "job_dir=$JOB_DIR"
  echo "scratch_job_dir=$SCRATCH_JOB_DIR"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || printf unknown)"
  echo "git_branch=$(git branch --show-current 2>/dev/null || printf unknown)"
  echo "git_status_begin"
  git status --short 2>/dev/null || true
  echo "git_status_end"
  echo "nvidia_smi_begin"
  nvidia-smi || true
  echo "nvidia_smi_end"
  echo "qwen_versions_begin"
  if [[ "$RUN_ANNOTATION" -eq 1 ]]; then
    "$QWEN_PYTHON" -c 'import platform, torch, transformers, PIL; print("python="+platform.python_version()); print("torch="+torch.__version__); print("cuda="+str(torch.version.cuda)); print("transformers="+transformers.__version__); print("Pillow="+PIL.__version__)'
  else
    echo "skipped_for_random_selection"
  fi
  echo "qwen_versions_end"
  echo "train_versions_begin"
  "$TRAIN_PYTHON" -c 'import platform, torch, ultralytics; print("python="+platform.python_version()); print("torch="+torch.__version__); print("cuda="+str(torch.version.cuda)); print("ultralytics="+ultralytics.__version__)'
  echo "train_versions_end"
  echo "sha256_begin"
  sha256sum \
    "$REPO_DIR/utils/assign_attribute.py" \
    "$REPO_DIR/object_detection/train_yolov8_kitti_generated.py" \
    "$KITTI_YOLO_ROOT/kitti_first5000.yaml" \
    "$SOURCE_ATTRIBUTE_FILE" \
    "$CORPUS_FILE" \
    "$DIAGNOSIS_CHECKPOINT"
  echo "sha256_end"
} | tee "$JOB_DIR/environment.log"

{
  echo '#!/usr/bin/env bash'
  echo '# Exact commands executed sequentially by this job.'
  if [[ "$RUN_ANNOTATION" -eq 1 ]]; then
    print_command "${ANNOTATION_CMD[@]}"
  else
    printf '# Source attribute annotation skipped for %s selection.\n' "$SELECTION_STRATEGY"
  fi
  print_command "${REPAIR_CMD[@]}"
} > "$COMMAND_LOG"
chmod 700 "$COMMAND_LOG"

if [[ "$RUN_ANNOTATION" -eq 1 ]]; then
  printf '\n=== Source attribute annotation (%s, ratio=%s, method=%s) ===\n' \
    "$MODE" "$SUB_RATIO" "$SELECTION_STRATEGY"
  printf 'The isolated Qwen output is measured but is not used by the repair command.\n'
  print_command "${ANNOTATION_CMD[@]}"
  "${ANNOTATION_CMD[@]}" 2>&1 | tee "$ANNOTATION_LOG"
  require_file "$ANNOTATION_TIMING" "annotation timing JSON"
  require_file "$ANNOTATION_OUTPUT" "isolated annotation output"
else
  printf '\n=== Source attribute annotation skipped for %s selection ===\n' "$SELECTION_STRATEGY"
  "$TRAIN_PYTHON" - "$ANNOTATION_TIMING" "$SELECTION_STRATEGY" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
selection_strategy = sys.argv[2]
reason = (
    "canonical source attribute metadata was reused; fresh Qwen assignment was intentionally excluded"
    if selection_strategy == "attribute"
    else "selected source strategy does not require Qwen attribute annotation"
)
record = {
    "status": "skipped",
    "reason": reason,
    "counts": {"images_total": 0, "images_pending": 0, "images_processed": 0, "attributes": 0},
    "stages_seconds": {"process_total": 0.0},
    "inference": {"error_count": 0, "fallback_count": 0},
}
output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  require_file "$ANNOTATION_TIMING" "skipped annotation timing JSON"
fi

printf '\n=== KITTI repair (%s, ratio=%s, method=%s) ===\n' \
  "$MODE" "$SUB_RATIO" "$SELECTION_STRATEGY"
if [[ "$SELECTION_STRATEGY" == "attribute" ]]; then
  printf 'Repair uses canonical source metadata: %s\n' "$SOURCE_ATTRIBUTE_FILE"
fi
print_command "${REPAIR_CMD[@]}"
"${REPAIR_CMD[@]}" 2>&1 | tee "$REPAIR_LOG"
require_file "$REPAIR_TIMING" "repair timing JSON"

"$TRAIN_PYTHON" - \
  "$ANNOTATION_TIMING" "$REPAIR_TIMING" "$PREFLIGHT_JSON" \
  "$SUMMARY_JSON" "$SUMMARY_MD" "$SUMMARY_TSV" \
  "$MODE" "$JOB_ID" "$SOURCE_ATTRIBUTE_FILE" "$ANNOTATION_OUTPUT" \
  "$SUB_RATIO" "$SELECTION_STRATEGY" "$RUN_ANNOTATION" <<'PY'
import csv
import json
import sys
from pathlib import Path

(
    annotation_path,
    repair_path,
    preflight_path,
    json_path,
    markdown_path,
    tsv_path,
    mode,
    job_id,
    canonical_attribute_path,
    isolated_attribute_path,
    sub_ratio_raw,
    selection_strategy,
    run_annotation_raw,
) = sys.argv[1:]

sub_ratio = float(sub_ratio_raw)
run_annotation = bool(int(run_annotation_raw))

def load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)

annotation = load(annotation_path)
repair = load(repair_path)
preflight = load(preflight_path)

expected = {
    "annotated": (2 if mode == "smoke" else 5000) if run_annotation else 0,
    "selected_original": int(5000 * sub_ratio),
    "selected_generated": 10 if mode == "smoke" else 1000,
}
expected_annotation_status = "success" if run_annotation else "skipped"
if annotation.get("status") != expected_annotation_status:
    raise SystemExit(
        f"Annotation timing status is {annotation.get('status')}, expected {expected_annotation_status}"
    )
if repair.get("status") != "success":
    raise SystemExit(f"Repair timing status is not success: {repair.get('status')}")
annotation_counts = annotation.get("counts", {})
if annotation_counts.get("images_processed") != expected["annotated"]:
    raise SystemExit(
        f"Expected {expected['annotated']} annotated images, got {annotation_counts.get('images_processed')}"
    )
inference = annotation.get("inference", {})
if run_annotation and (inference.get("error_count", 0) != 0 or inference.get("fallback_count", 0) != 0):
    raise SystemExit(f"Annotation used inference fallbacks: {inference}")
repair_counts = repair.get("counts", {})
for key in ("selected_original", "selected_generated"):
    if repair_counts.get(key) != expected[key]:
        raise SystemExit(f"Expected {key}={expected[key]}, got {repair_counts.get(key)}")
if repair_counts.get("final_train_images") != expected["selected_original"] + expected["selected_generated"]:
    raise SystemExit(f"Unexpected final training count: {repair_counts.get('final_train_images')}")
if repair_counts.get("val_images") != 1240 or repair_counts.get("test_images") != 1241:
    raise SystemExit(f"Unexpected KITTI validation/test counts: {repair_counts}")
if repair.get("provenance", {}).get("test_evaluation_performed") is not True:
    raise SystemExit("Final KITTI test evaluation was not performed")

annotation_stages = annotation.get("stages_seconds", {})
repair_stages = repair.get("stage_seconds", {})
if annotation_stages.get("process_total") is None:
    raise SystemExit("Annotation process_total timing is missing")
if repair.get("command_total_seconds") is None:
    raise SystemExit("Repair command_total_seconds timing is missing")
annotation_total = float(annotation_stages["process_total"])
repair_total = float(repair["command_total_seconds"])
combined_total = annotation_total + repair_total
summary = {
    "schema_version": 1,
    "job_id": job_id,
    "mode": mode,
    "ratio": sub_ratio,
    "selection_strategy": selection_strategy,
    "status": "success" if annotation.get("status") == expected_annotation_status and repair.get("status") == "success" else "incomplete",
    "scope": {
        "included": (
            ["source_attribute_annotation", "repair_selection_materialization_training_final_test_evaluation"]
            if run_annotation
            else ["repair_selection_materialization_training_final_test_evaluation"]
        ),
        "repair_image_generation": {
            "status": "excluded_by_user_scope",
            "seconds": None,
            "reason": "Generated red-cloudy images were provided as an existing input pool.",
        },
        "generated_image_attribute_assignment": {
            "status": "not_applicable",
            "seconds": None,
            "reason": "Known red/cloudy generated-image counts were supplied directly to the selector.",
        },
    },
    "attribute_provenance": {
        "annotation_status": expected_annotation_status,
        "annotation_output_isolated": str(Path(isolated_attribute_path).resolve()) if run_annotation else None,
        "repair_source_attribute_file": str(Path(canonical_attribute_path).resolve()) if selection_strategy == "attribute" else None,
        "repair_used_isolated_annotation_output": False,
    },
    "seconds": {
        "source_attribute_annotation_total": annotation_total,
        "repair_total": repair_total,
        "included_end_to_end_total": combined_total,
        "annotation_stages": annotation_stages,
        "repair_stages": repair_stages,
    },
    "counts": {
        "annotation": annotation.get("counts", {}),
        "repair": repair.get("counts", {}),
        "preflight": preflight,
    },
    "evaluation": repair.get("evaluation", {}),
    "inputs": {
        "annotation_timing_json": str(Path(annotation_path).resolve()),
        "repair_timing_json": str(Path(repair_path).resolve()),
        "preflight_json": str(Path(preflight_path).resolve()),
    },
}
Path(json_path).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

rows = [
    ("source_attribute_annotation_total", annotation_total, "included" if run_annotation else "not_applicable"),
]
for name, seconds in annotation_stages.items():
    if name != "process_total" and seconds is not None:
        rows.append((f"annotation.{name}", float(seconds), "included"))
rows.append(("repair_total", repair_total, "included"))
for name, seconds in repair_stages.items():
    rows.append((f"repair.{name}", float(seconds or 0.0), "included"))
rows.extend([
    ("repair_image_generation", "", "excluded_by_user_scope"),
    ("generated_image_attribute_assignment", "", "not_applicable"),
    ("included_end_to_end_total", combined_total, "included"),
])
with Path(tsv_path).open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(["stage", "seconds", "scope_status"])
    writer.writerows(rows)

minutes = combined_total / 60.0
evaluation = summary.get("evaluation", {})
map_50_95 = evaluation.get("metrics/mAP50-95(B)")
metric_line = f"- Final test mAP50-95: `{map_50_95:.4f}`\n" if map_50_95 is not None else ""
markdown = f"""# KITTI Cost Accounting\n\n- Job: `{job_id}`\n- Mode: `{mode}`\n- Ratio: `{sub_ratio}`\n- Selection strategy: `{selection_strategy}`\n- Status: `{summary['status']}`\n- Source attribute annotation: `{annotation_total:.2f}` seconds ({'included' if run_annotation else 'not applicable'})\n- Repair (selection through final test evaluation): `{repair_total:.2f}` seconds\n- Included end-to-end total: `{combined_total:.2f}` seconds (`{minutes:.2f}` minutes)\n{metric_line}- Repair-image generation: **excluded by user scope**\n- Generated-image attribute assignment: **not applicable** because known red/cloudy counts were supplied\n- Repair metadata: {'canonical existing source attribute file; isolated Qwen timing output was not used for selection' if selection_strategy == 'attribute' else f'Qwen source metadata not used by {selection_strategy} selection'}\n\n## Stage timings\n\n| Stage | Seconds | Scope status |\n|---|---:|---|\n"""
for stage, seconds, scope in rows:
    shown = "" if seconds == "" else f"{float(seconds):.2f}"
    markdown += f"| `{stage}` | {shown} | {scope} |\n"
Path(markdown_path).write_text(markdown, encoding="utf-8")
print(json.dumps({"included_end_to_end_seconds": combined_total, "summary": json_path}, indent=2))
PY

require_file "$SUMMARY_JSON" "cost summary JSON"
require_file "$SUMMARY_MD" "cost summary Markdown"
require_file "$SUMMARY_TSV" "cost summary TSV"
printf '\nKITTI cost-accounting job completed.\nArtifacts: %s\nScratch: %s\n' "$JOB_DIR" "$SCRATCH_JOB_DIR"
