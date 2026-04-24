#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_pose_attribute_rpc12_resume_missing.sh [options]

Relaunches only the unfinished runs from the interrupted InternVL3 RPC12 queue.

Already completed in the original 20260423_224514 queue and therefore skipped:
  - GPU 0 / ratio 0.1: all 3 runs
  - GPU 1 / ratio 0.3: attrx333_r001_p05_c15, attrx333_r003_p05_c8
  - GPU 2 / ratio 0.5: attrx2_r001_p05_c8
  - GPU 3 / ratio 0.7: attrx143_r001_p05_c8

This resume controller launches only these missing items:
  - GPU 1 / ratio 0.3:
    - attrx333_r003_p05_c15
  - GPU 2 / ratio 0.5:
    - attrx2_r003_p05_c15
    - attrx2_r003_p05_c8
  - GPU 3 / ratio 0.7:
    - attrx143_r003_p05_c15
    - attrx143_r003_p05_c8

Options:
  --base-model PATH         Resume checkpoint.
  --python PATH             Python executable.
  --attribute-file PATH     InternVL3 attribute metadata JSON.
  --data-root PATH          COCO root.
  --epochs-total N          Total epochs passed to train_pose_estimation.py. Default: 2
  --generated-attribute V   Generated attribute to append. Default: rainy
  --num-added-img N         Number of generated images. Default: 1000
  --max-train-samples N     Original train cap before sub-ratio. Default: 20000
  --batch-size N            Training batch size. Default: 8
  --num-workers N           Training workers. Default: 4
  --lr V                    Learning rate. Default: 0.005
  --seed N                  Random seed. Default: 42
  --focus NAME              Attribute focus key. Default: weather
  --detach-controller       Launch the controller itself under nohup/setsid and exit.
  --dry-run                 Print commands without launching.
EOF
}

PYTHON_EXE="/people/cs/o/oxx220000/.conda/envs/ad/bin/python"
BASE_MODEL="log_pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530/best.pth"
ATTRIBUTE_FILE="save/attribute_coco_pose_gemini_train20k_internvl3_8B.json"
DATA_ROOT="/data/oxx220000/coco"
EPOCHS_TOTAL=2
GENERATED_ATTRIBUTE="rainy"
NUM_ADDED_IMG=1000
MAX_TRAIN_SAMPLES=20000
BATCH_SIZE=8
NUM_WORKERS=4
LR="0.005"
SEED=42
FOCUS="weather"
DETACH_CONTROLLER=0
DRY_RUN=0

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
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --epochs-total)
      EPOCHS_TOTAL="$2"
      shift 2
      ;;
    --generated-attribute)
      GENERATED_ATTRIBUTE="$2"
      shift 2
      ;;
    --num-added-img)
      NUM_ADDED_IMG="$2"
      shift 2
      ;;
    --max-train-samples)
      MAX_TRAIN_SAMPLES="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --focus)
      FOCUS="$2"
      shift 2
      ;;
    --detach-controller)
      DETACH_CONTROLLER=1
      shift
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

mkdir -p log_pose/pids

if [[ "$DETACH_CONTROLLER" -eq 1 ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "--detach-controller cannot be combined with --dry-run" >&2
    exit 1
  fi

  local_ts="$(date +%Y%m%d_%H%M%S)"
  controller_log="log_pose/run_pose_attribute_rpc12_resume_missing_${local_ts}.log"
  controller_pid="log_pose/pids/run_pose_attribute_rpc12_resume_missing_${local_ts}.pid"

  detached_cmd=(
    bash "$0"
    --base-model "$BASE_MODEL"
    --python "$PYTHON_EXE"
    --attribute-file "$ATTRIBUTE_FILE"
    --data-root "$DATA_ROOT"
    --epochs-total "$EPOCHS_TOTAL"
    --generated-attribute "$GENERATED_ATTRIBUTE"
    --num-added-img "$NUM_ADDED_IMG"
    --max-train-samples "$MAX_TRAIN_SAMPLES"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --lr "$LR"
    --seed "$SEED"
    --focus "$FOCUS"
  )

  setsid nohup "${detached_cmd[@]}" > "$controller_log" 2>&1 < /dev/null &
  controller_pid_value=$!
  echo "$controller_pid_value" > "$controller_pid"
  echo "CONTROLLER_LOG=$controller_log"
  echo "CONTROLLER_PIDFILE=$controller_pid"
  echo "CONTROLLER_PID=$controller_pid_value"
  exit 0
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

declare -a GPUS=(1 2 3)
declare -a RATIOS=(0.3 0.5 0.7)

declare -a QUEUE_1=(
  "attrx333_r003_p05_c15|0.03|0.5|15.0"
)
declare -a QUEUE_2=(
  "attrx2_r003_p05_c15|0.03|0.5|15.0"
  "attrx2_r003_p05_c8|0.03|0.5|8.0"
)
declare -a QUEUE_3=(
  "attrx143_r003_p05_c15|0.03|0.5|15.0"
  "attrx143_r003_p05_c8|0.03|0.5|8.0"
)

LAUNCHED_PID=""

ratio_extra_scale() {
  local ratio="$1"
  awk -v ratio="$ratio" 'BEGIN { printf "%.6g", 1.0 / ratio }'
}

launch_one() {
  local gpu="$1"
  local ratio="$2"
  local name="$3"
  local r="$4"
  local p="$5"
  local c="$6"
  local extra_scale
  extra_scale="$(ratio_extra_scale "$ratio")"

  local log_file="log_pose/pose_resume_internvl3_epoch${EPOCHS_TOTAL}_attribute_ratio${ratio}_${name}_gpu${gpu}_${TIMESTAMP}.log"
  local pid_file="log_pose/pids/pose_resume_internvl3_epoch${EPOCHS_TOTAL}_attribute_ratio${ratio}_${name}_gpu${gpu}_${TIMESTAMP}.pid"
  local out_dir="log_pose/pose_keypointrcnn_resume_internvl3_attribute_ratio${ratio}_${name}_gpu${gpu}_${TIMESTAMP}"

  local cmd=(
    env
    MPLCONFIGDIR=/tmp/matplotlib-safefix
    CUDA_VISIBLE_DEVICES="${gpu}"
    "${PYTHON_EXE}" -u train_pose_estimation.py train
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
    --sub-ratio "${ratio}"
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
    --al-attribute-extra-count-scale "${extra_scale}"
    --al-attribute-frequency-penalty "${p}"
    --al-attribute-class-balance "${c}"
    --al-attribute-use-entropy false
  )

  echo "[launch] gpu=${gpu} ratio=${ratio} name=${name} R=${r} P=${p} C=${c}"
  echo "         base=${BASE_MODEL}"
  echo "         attribute_file=${ATTRIBUTE_FILE}"
  echo "         focus=${FOCUS}"
  echo "         extra_scale=${extra_scale}"
  echo "         log=${log_file}"
  echo "         out=${out_dir}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '         cmd='
    printf '%q ' "${cmd[@]}"
    printf '\n'
    LAUNCHED_PID=""
    return 0
  fi

  "${cmd[@]}" > "${log_file}" 2>&1 &
  LAUNCHED_PID="$!"
  echo "${LAUNCHED_PID}" > "${pid_file}"
  echo "         pid=${LAUNCHED_PID} pidfile=${pid_file}"
}

run_gpu_sequence() {
  local gpu="$1"
  local ratio="$2"
  local queue_name="$3"

  local -n queue_ref="$queue_name"

  echo "[pipeline] gpu=${gpu} ratio=${ratio} starting missing-run queue"

  local idx
  for idx in "${!queue_ref[@]}"; do
    local spec="${queue_ref[$idx]}"
    local name=""
    local r=""
    local p=""
    local c=""
    IFS='|' read -r name r p c <<< "${spec}"

    launch_one "${gpu}" "${ratio}" "${name}" "${r}" "${p}" "${c}"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      continue
    fi

    local run_pid="${LAUNCHED_PID}"
    echo "[wait] gpu=${gpu} ratio=${ratio} name=${name} pid=${run_pid}"
    if wait "${run_pid}"; then
      echo "[done] gpu=${gpu} ratio=${ratio} name=${name} pid=${run_pid}"
    else
      local status=$?
      echo "[fail] gpu=${gpu} ratio=${ratio} name=${name} pid=${run_pid} exit=${status}" >&2
      return "${status}"
    fi
  done

  echo "[pipeline] gpu=${gpu} ratio=${ratio} finished missing-run queue"
}

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run for missing RPC12 runs."
  run_gpu_sequence "${GPUS[0]}" "${RATIOS[0]}" QUEUE_1
  run_gpu_sequence "${GPUS[1]}" "${RATIOS[1]}" QUEUE_2
  run_gpu_sequence "${GPUS[2]}" "${RATIOS[2]}" QUEUE_3
  echo "Dry run complete."
  exit 0
fi

echo "Launching missing RPC12 runs."
echo "  timestamp:         ${TIMESTAMP}"
echo "  base model:        ${BASE_MODEL}"
echo "  attribute file:    ${ATTRIBUTE_FILE}"
echo "  generated attr:    ${GENERATED_ATTRIBUTE}"
echo "  num added img:     ${NUM_ADDED_IMG}"
echo "  max train samples: ${MAX_TRAIN_SAMPLES}"
echo "  focus:             ${FOCUS}"

CONTROLLER_PIDS=()
run_gpu_sequence "${GPUS[0]}" "${RATIOS[0]}" QUEUE_1 &
CONTROLLER_PIDS+=("$!")
run_gpu_sequence "${GPUS[1]}" "${RATIOS[1]}" QUEUE_2 &
CONTROLLER_PIDS+=("$!")
run_gpu_sequence "${GPUS[2]}" "${RATIOS[2]}" QUEUE_3 &
CONTROLLER_PIDS+=("$!")

overall_status=0
for controller_pid in "${CONTROLLER_PIDS[@]}"; do
  if ! wait "${controller_pid}"; then
    overall_status=1
  fi
done

echo "[done] missing RPC12 runs timestamp=${TIMESTAMP} overall_status=${overall_status}"
exit "${overall_status}"
