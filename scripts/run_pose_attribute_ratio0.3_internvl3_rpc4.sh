#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_pose_attribute_ratio0.3_internvl3_rpc4.sh [--base-model PATH] [--python PATH] [--attribute-file PATH] [--extra-count-scale FLOAT] [--dry-run]

Launches four detached COCO pose resume-from-base attribute-AL runs at ratio=0.3 on GPUs 0/1/2/3
using the InternVL3 train20k attribute metadata.

Selected scenarios:
  - GPU 0: attrx333_r000_p0_c0
  - GPU 1: attrx333_r000_p0_c15
  - GPU 2: attrx333_r001_p05_c0
  - GPU 3: attrx333_r001_p05_c15

Rationale:
  - ratio=0.3 changes the generated-count scaling to x3.333, so the old ratio=0.1
    steered family no longer yields four meaningfully distinct selections
  - this sweep isolates the two useful axes at ratio=0.3:
    conservative vs weather-steered, and class-balance off vs on
EOF
}

PYTHON_EXE="/people/cs/o/oxx220000/.conda/envs/ad/bin/python"
BASE_MODEL="logs/pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530/best.pth"
ATTRIBUTE_FILE="save/attribute_coco_pose_gemini_train20k_internvl3_8B.json"
DATA_ROOT="/data/oxx220000/coco"
DRY_RUN=0

RATIO="0.3"
EPOCHS_TOTAL=2
GENERATED_ATTRIBUTE="rainy"
NUM_ADDED_IMG=1000
MAX_TRAIN_SAMPLES=20000
BATCH_SIZE=8
NUM_WORKERS=4
LR="0.005"
SEED=42
FOCUS="weather"
EXTRA_COUNT_SCALE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-model)
      BASE_MODEL="$2"
      shift 2
      ;;
    --python)
      PYTHON_EXE="$2"
      shift 2
      ;;
    --attribute-file)
      ATTRIBUTE_FILE="$2"
      shift 2
      ;;
    --extra-count-scale)
      EXTRA_COUNT_SCALE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -x "$PYTHON_EXE" ]]; then
  echo "Python executable not found: $PYTHON_EXE" >&2
  exit 1
fi

if [[ ! -f "$BASE_MODEL" ]]; then
  echo "Base model not found: $BASE_MODEL" >&2
  exit 1
fi

if [[ ! -f "$ATTRIBUTE_FILE" ]]; then
  echo "Attribute file not found: $ATTRIBUTE_FILE" >&2
  exit 1
fi

if [[ -z "$EXTRA_COUNT_SCALE" ]]; then
  EXTRA_COUNT_SCALE="$(awk -v ratio="$RATIO" 'BEGIN { printf "%.6g", 1.0 / ratio }')"
fi

mkdir -p logs/pose/pids

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

launch_one() {
  local gpu="$1"
  local name="$2"
  local r="$3"
  local p="$4"
  local c="$5"

  local log_file="logs/pose/pose_resume_internvl3_epoch${EPOCHS_TOTAL}_attribute_ratio${RATIO}_${name}_gpu${gpu}_${TIMESTAMP}.log"
  local pid_file="logs/pose/pids/pose_resume_internvl3_epoch${EPOCHS_TOTAL}_attribute_ratio${RATIO}_${name}_gpu${gpu}_${TIMESTAMP}.pid"
  local out_dir="logs/pose/pose_keypointrcnn_resume_internvl3_attribute_ratio${RATIO}_${name}_gpu${gpu}_${TIMESTAMP}"

  local cmd=(
    env
    MPLCONFIGDIR=/tmp/matplotlib-safefix
    CUDA_VISIBLE_DEVICES="${gpu}"
    "${PYTHON_EXE}" -u pose_estimation/train_pose_estimation.py train
    --data-root "${DATA_ROOT}"
    --pretrained true
    --device cuda
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --epochs "${EPOCHS_TOTAL}"
    --eval-every 1
    --lr "${LR}"
    --seed "${SEED}"
    --max-train-samples "${MAX_TRAIN_SAMPLES}"
    --sub-ratio "${RATIO}"
    --resume-model "${BASE_MODEL}"
    --attribute "${GENERATED_ATTRIBUTE}"
    --num-added-img "${NUM_ADDED_IMG}"
    --analyze-after-train false
    --assigner internvl3
    --output-dir "${out_dir}"
    --use-active-learning true
    --al-strategy attribute
    --al-attribute-file "${ATTRIBUTE_FILE}"
    --al-attribute-focus "${FOCUS}"
    --al-attribute-rarity-power "${r}"
    --al-attribute-uncertainty-boost 1.0
    --al-attribute-log-acc-scale 0.0
    --al-attribute-extra-count-scale "${EXTRA_COUNT_SCALE}"
    --al-attribute-frequency-penalty "${p}"
    --al-attribute-class-balance "${c}"
    --al-attribute-use-entropy false
  )

  echo "[launch] gpu=${gpu} name=${name} R=${r} P=${p} C=${c}"
  echo "         base=${BASE_MODEL}"
  echo "         attribute_file=${ATTRIBUTE_FILE}"
  echo "         focus=${FOCUS}"
  echo "         extra_scale=${EXTRA_COUNT_SCALE}"
  echo "         log=${log_file}"
  echo "         out=${out_dir}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '         cmd='
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  (
    setsid nohup "${cmd[@]}" > "${log_file}" 2>&1 < /dev/null &
    echo $! > "${pid_file}"
    echo "PID=$(cat "${pid_file}")"
    echo "LOG=${log_file}"
    echo "OUTDIR=${out_dir}"
    echo "PIDFILE=${pid_file}"
  )
}

launch_one 0 attrx333_r000_p0_c0 0.0 0.0 0
launch_one 1 attrx333_r000_p0_c15 0.0 0.0 15.0
launch_one 2 attrx333_r001_p05_c0 0.01 0.5 0
launch_one 3 attrx333_r001_p05_c15 0.01 0.5 15.0
