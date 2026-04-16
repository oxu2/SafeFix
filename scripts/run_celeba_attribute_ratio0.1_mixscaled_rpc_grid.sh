#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_celeba_attribute_ratio0.1_mixscaled_rpc_grid.sh [--base-model PATH] [--python PATH] [--extra-count-scale FLOAT] [--dry-run]

Launches an 8-scenario CelebA resume-from-base attribute-AL grid at ratio=0.1 on GPUs 0/1/2/3.

Key change vs the earlier probe:
  - keeps E=False
  - scales generated-image extra counts by 1 / sub_ratio by default
    so the selector sees a generated mix closer to the final training set

Scenarios:
  - mixselx10_e0_r000_p0_cinf
  - mixselx10_e0_r000_p05_cinf
  - mixselx10_e0_r001_p0_cinf
  - mixselx10_e0_r001_p05_cinf
  - mixselx10_e0_r001_p1_cinf
  - mixselx10_e0_r003_p05_cinf
  - mixselx10_e0_r01_p05_cinf
  - mixselx10_e0_r01_p05_c15
EOF
}

PYTHON_EXE="/people/cs/o/oxx220000/.conda/envs/ad/bin/python"
BASE_MODEL="weights/model_celeba_resnet_base.pth"
DRY_RUN=0

ATTRIBUTES="redhair_brownskin_sademotion"
FOCUS="hair color,skin,emotion"
BASE_EXTRA_COUNTS="hair color:red=1000,skin:brown=1000,emotion:sad=1000"
RATIO="0.1"
NUM_EPOCHS=15
NUM_ADDED_IMG=1000
LR="1e-5"
BATCH_SIZE=64
WORKERS=4
SEED=42
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

if [[ -z "$EXTRA_COUNT_SCALE" ]]; then
  EXTRA_COUNT_SCALE="$(awk -v ratio="$RATIO" 'BEGIN { printf "%.6g", 1.0 / ratio }')"
fi

mkdir -p logs_celeba/pids

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

run_one() {
  local gpu="$1"
  local name="$2"
  local r="$3"
  local p="$4"
  local c="$5"

  local log_file="logs_celeba/attr_ratio${RATIO}_${name}_gpu${gpu}_${TIMESTAMP}.log"

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
    --attribute_file save/attribute.json
    --al_attribute_focus "${FOCUS}"
    --al_attribute_rarity_power "${r}"
    --al_attribute_log_acc_scale 0.0
    --al_attribute_extra_counts "${BASE_EXTRA_COUNTS}"
    --al_attribute_extra_count_scale "${EXTRA_COUNT_SCALE}"
    --al_attribute_frequency_penalty "${p}"
    --al_attribute_class_balance "${c}"
    --al_attribute_use_entropy False
  )

  echo "[launch] gpu=${gpu} name=${name} R=${r} P=${p} C=${c} extra_scale=${EXTRA_COUNT_SCALE}"
  echo "         log=${log_file}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '         cmd='
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  "${cmd[@]}" > "${log_file}" 2>&1
  echo "[done] gpu=${gpu} name=${name}"
}

run_queue() {
  local gpu="$1"
  shift
  local -a specs=("$@")

  echo "[queue-start] gpu=${gpu} scenarios=${#specs[@]}"
  for spec in "${specs[@]}"; do
    IFS='|' read -r name r p c <<< "${spec}"
    run_one "${gpu}" "${name}" "${r}" "${p}" "${c}"
  done
  echo "[queue-done] gpu=${gpu}"
}

GPU0_SPECS=(
  "mixselx10_e0_r001_p0_cinf|0.01|0.0|inf"
  "mixselx10_e0_r001_p05_cinf|0.01|0.5|inf"
)
GPU1_SPECS=(
  "mixselx10_e0_r000_p0_cinf|0.0|0.0|inf"
  "mixselx10_e0_r000_p05_cinf|0.0|0.5|inf"
)
GPU2_SPECS=(
  "mixselx10_e0_r003_p05_cinf|0.03|0.5|inf"
  "mixselx10_e0_r001_p1_cinf|0.01|1.0|inf"
)
GPU3_SPECS=(
  "mixselx10_e0_r01_p05_cinf|0.1|0.5|inf"
  "mixselx10_e0_r01_p05_c15|0.1|0.5|15.0"
)

echo "[grid] ratio=${RATIO} extra_count_scale=${EXTRA_COUNT_SCALE} timestamp=${TIMESTAMP}"

run_queue 0 "${GPU0_SPECS[@]}" &
run_queue 1 "${GPU1_SPECS[@]}" &
run_queue 2 "${GPU2_SPECS[@]}" &
run_queue 3 "${GPU3_SPECS[@]}" &

wait
echo "[grid-done] timestamp=${TIMESTAMP}"
