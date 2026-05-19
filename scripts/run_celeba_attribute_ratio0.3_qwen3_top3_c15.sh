#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_celeba_attribute_ratio0.3_qwen3_top3_c15.sh [--base-model PATH] [--python PATH] [--attribute-file PATH] [--extra-count-scale FLOAT] [--dry-run]

Launches three detached CelebA resume-from-base attribute-AL runs at ratio=0.3 on GPUs 0/1/2
using the Qwen3-generated train+val attribute metadata and a finite class-balance cap C=15.

Selected scenarios:
  - GPU 0: qwen3mixx333_r000_p0_c15
  - GPU 1: qwen3mixx333_r001_p05_c15
  - GPU 2: qwen3mixx333_r003_p05_c15
EOF
}

PYTHON_EXE="/people/cs/o/oxx220000/.conda/envs/ad/bin/python"
BASE_MODEL="save/weights/model_celeba_resnet_base.pth"
ATTRIBUTE_FILE="save/attribute_celeba_trainval_qwen3_8B.json"
DRY_RUN=0

ATTRIBUTES="redhair_brownskin_sademotion"
FOCUS="hair color,skin,emotion"
BASE_EXTRA_COUNTS="hair color:red=1000,skin:brown=1000,emotion:sad=1000"
RATIO="0.3"
NUM_EPOCHS=15
NUM_ADDED_IMG=1000
LR="1e-5"
BATCH_SIZE=64
WORKERS=4
SEED=42
EXTRA_COUNT_SCALE=""
CLASS_BALANCE="15.0"

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

mkdir -p logs/celeba/pids

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

launch_one() {
  local gpu="$1"
  local name="$2"
  local r="$3"
  local p="$4"

  local log_file="logs/celeba/attr_ratio${RATIO}_${name}_gpu${gpu}_${TIMESTAMP}.log"
  local pid_file="logs/celeba/pids/attr_ratio${RATIO}_${name}_gpu${gpu}_${TIMESTAMP}.pid"

  local cmd=(
    env
    MPLCONFIGDIR=/tmp/matplotlib-safefix
    CUDA_VISIBLE_DEVICES="${gpu}"
    "${PYTHON_EXE}" -u alpred.py
    --dataset celeba
    --model resnet
    --attributes "${ATTRIBUTES}"
    --gen_image_dir "data/${ATTRIBUTES}"
    --num_added_img "${NUM_ADDED_IMG}"
    --num_epochs "${NUM_EPOCHS}"
    --resume_model "${BASE_MODEL}"
    --freeze_backbone False
    --lr "${LR}"
    --batch_size "${BATCH_SIZE}"
    --workers "${WORKERS}"
    --seed "${SEED}"
    --gpu "${gpu}"
    --save_pred_model best
    --save_preds True
    --save_model False
    --verbose False
    --sub_ratio "${RATIO}"
    --use_active_learning True
    --al_strategy attribute
    --attribute_file "${ATTRIBUTE_FILE}"
    --al_attribute_focus "${FOCUS}"
    --al_attribute_rarity_power "${r}"
    --al_attribute_log_acc_scale 0.0
    --al_attribute_extra_counts "${BASE_EXTRA_COUNTS}"
    --al_attribute_extra_count_scale "${EXTRA_COUNT_SCALE}"
    --al_attribute_frequency_penalty "${p}"
    --al_attribute_class_balance "${CLASS_BALANCE}"
    --al_attribute_use_entropy False
  )

  echo "[launch] gpu=${gpu} name=${name} R=${r} P=${p} C=${CLASS_BALANCE}"
  echo "         attribute_file=${ATTRIBUTE_FILE}"
  echo "         extra_scale=${EXTRA_COUNT_SCALE}"
  echo "         log=${log_file}"

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
    echo "PIDFILE=${pid_file}"
  )
}

launch_one 0 qwen3mixx333_r000_p0_c15 0.0 0.0
launch_one 1 qwen3mixx333_r001_p05_c15 0.01 0.5
launch_one 2 qwen3mixx333_r003_p05_c15 0.03 0.5
