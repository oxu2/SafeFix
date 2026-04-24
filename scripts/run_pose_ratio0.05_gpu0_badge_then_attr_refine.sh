#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_pose_ratio0.05_gpu0_badge_then_attr_refine.sh [options]

Runs the missing COCO pose InternVL3 ratio=0.05 badge baseline and four
conservative attribute refinements sequentially on GPU 0:
  1. badge
  2. attribute attrx20_r000_p0_c0
  3. attribute attrx20_r000_p0_c8
  4. attribute attrx20_r000_p0_c30
  5. attribute attrx20_r000_p0_cinf

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
  --start-at NAME           Skip earlier queue entries and start at this name.
  --detach-controller       Launch the controller itself under nohup/setsid and exit.
  --dry-run                 Print commands without launching.
EOF
}

PYTHON_EXE="/people/cs/o/oxx220000/.conda/envs/ad/bin/python"
BASE_MODEL="log_pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530/best.pth"
ATTRIBUTE_FILE="save/attribute_coco_pose_gemini_train20k_internvl3_8B.json"
DATA_ROOT="/data/oxx220000/coco"
GPU=0
RATIO="0.05"
EPOCHS_TOTAL=2
GENERATED_ATTRIBUTE="rainy"
NUM_ADDED_IMG=1000
MAX_TRAIN_SAMPLES=20000
BATCH_SIZE=8
NUM_WORKERS=4
LR="0.005"
SEED=42
FOCUS="weather"
AL_INITIAL_LABELED=400
AL_FEATURE_BATCH_SIZE=8
START_AT=""
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
    --start-at)
      START_AT="$2"
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
  controller_log="log_pose/run_pose_ratio0.05_gpu0_badge_then_attr_refine_${local_ts}.log"
  controller_pid="log_pose/pids/run_pose_ratio0.05_gpu0_badge_then_attr_refine_${local_ts}.pid"

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
  if [[ -n "$START_AT" ]]; then
    detached_cmd+=(--start-at "$START_AT")
  fi

  setsid nohup "${detached_cmd[@]}" > "$controller_log" 2>&1 < /dev/null &
  controller_pid_value=$!
  echo "$controller_pid_value" > "$controller_pid"
  echo "CONTROLLER_LOG=$controller_log"
  echo "CONTROLLER_PIDFILE=$controller_pid"
  echo "CONTROLLER_PID=$controller_pid_value"
  exit 0
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXTRA_COUNT_SCALE="$(awk -v ratio="$RATIO" 'BEGIN { printf "%.6g", 1.0 / ratio }')"
LAUNCHED_PID=""

common_cmd_prefix() {
  COMMON_CMD=(
    env
    MPLCONFIGDIR=/tmp/matplotlib-safefix
    CUDA_VISIBLE_DEVICES="${GPU}"
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
    --sub-ratio "${RATIO}"
    --resume-model "${BASE_MODEL}"
    --attribute "${GENERATED_ATTRIBUTE}"
    --num-added-img "${NUM_ADDED_IMG}"
    --analyze-after-train false
    --assigner internvl3
  )
}

launch_and_wait() {
  local name="$1"
  local log_file="$2"
  local pid_file="$3"
  shift 3
  local cmd=("$@")

  echo "[launch] ${name}"
  echo "         log=${log_file}"
  echo "         pidfile=${pid_file}"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '         cmd='
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  "${cmd[@]}" > "$log_file" 2>&1 &
  LAUNCHED_PID="$!"
  echo "$LAUNCHED_PID" > "$pid_file"
  echo "         pid=${LAUNCHED_PID}"

  if wait "$LAUNCHED_PID"; then
    echo "[done] ${name} pid=${LAUNCHED_PID}"
  else
    local status=$?
    echo "[fail] ${name} pid=${LAUNCHED_PID} exit=${status}" >&2
    return "$status"
  fi
}

launch_badge() {
  local log_file="log_pose/pose_resume_internvl3_epoch${EPOCHS_TOTAL}_badge_ratio${RATIO}_gpu${GPU}_${TIMESTAMP}.log"
  local pid_file="log_pose/pids/pose_resume_internvl3_epoch${EPOCHS_TOTAL}_badge_ratio${RATIO}_gpu${GPU}_${TIMESTAMP}.pid"
  local out_dir="log_pose/pose_keypointrcnn_resume_internvl3_badge_ratio${RATIO}_gpu${GPU}_${TIMESTAMP}"

  common_cmd_prefix
  local cmd=(
    "${COMMON_CMD[@]}"
    --output-dir "${out_dir}"
    --use-active-learning true
    --al-strategy badge
    --al-initial-labeled "${AL_INITIAL_LABELED}"
    --al-feature-batch-size "${AL_FEATURE_BATCH_SIZE}"
  )

  echo "         out=${out_dir}"
  launch_and_wait "badge" "$log_file" "$pid_file" "${cmd[@]}"
}

launch_attribute() {
  local name="$1"
  local c="$2"
  local log_file="log_pose/pose_resume_internvl3_epoch${EPOCHS_TOTAL}_attribute_ratio${RATIO}_${name}_gpu${GPU}_${TIMESTAMP}.log"
  local pid_file="log_pose/pids/pose_resume_internvl3_epoch${EPOCHS_TOTAL}_attribute_ratio${RATIO}_${name}_gpu${GPU}_${TIMESTAMP}.pid"
  local out_dir="log_pose/pose_keypointrcnn_resume_internvl3_attribute_ratio${RATIO}_${name}_gpu${GPU}_${TIMESTAMP}"

  common_cmd_prefix
  local cmd=(
    "${COMMON_CMD[@]}"
    --output-dir "${out_dir}"
    --use-active-learning true
    --al-strategy attribute
    --al-attribute-file "${ATTRIBUTE_FILE}"
    --al-attribute-focus "${FOCUS}"
    --al-attribute-rarity-power 0.0
    --al-attribute-uncertainty-boost 1.0
    --al-attribute-log-acc-scale 0.0
    --al-attribute-extra-count-scale "${EXTRA_COUNT_SCALE}"
    --al-attribute-frequency-penalty 0.0
    --al-attribute-class-balance "${c}"
    --al-attribute-use-entropy false
  )

  echo "         out=${out_dir}"
  echo "         R=0.0 P=0.0 C=${c} extra_scale=${EXTRA_COUNT_SCALE}"
  launch_and_wait "attribute ${name}" "$log_file" "$pid_file" "${cmd[@]}"
}

echo "Launching ratio=${RATIO} badge + attribute refinement queue on GPU ${GPU}."
echo "  timestamp:         ${TIMESTAMP}"
echo "  base model:        ${BASE_MODEL}"
echo "  attribute file:    ${ATTRIBUTE_FILE}"
echo "  generated attr:    ${GENERATED_ATTRIBUTE}"
echo "  num added img:     ${NUM_ADDED_IMG}"
echo "  max train samples: ${MAX_TRAIN_SAMPLES}"
echo "  focus:             ${FOCUS}"
echo "  extra scale:       ${EXTRA_COUNT_SCALE}"
echo "  rationale:         refine around best attrx20_r000_p0_c15 by sweeping class balance only"
if [[ -n "$START_AT" ]]; then
  echo "  start at:          ${START_AT}"
fi

STARTED=0
if [[ -z "$START_AT" ]]; then
  STARTED=1
fi

run_entry() {
  local name="$1"
  shift

  if [[ "$STARTED" -eq 0 ]]; then
    if [[ "$name" == "$START_AT" ]]; then
      STARTED=1
    else
      echo "[skip] ${name} before start-at=${START_AT}"
      return 0
    fi
  fi

  "$@"
}

run_entry badge launch_badge
run_entry attrx20_r000_p0_c0 launch_attribute attrx20_r000_p0_c0 0
run_entry attrx20_r000_p0_c8 launch_attribute attrx20_r000_p0_c8 8.0
run_entry attrx20_r000_p0_c30 launch_attribute attrx20_r000_p0_c30 30.0
run_entry attrx20_r000_p0_cinf launch_attribute attrx20_r000_p0_cinf inf

if [[ "$STARTED" -eq 0 ]]; then
  echo "No queue entry matched --start-at ${START_AT}" >&2
  exit 1
fi

echo "[done] ratio=${RATIO} badge + attribute refinement queue timestamp=${TIMESTAMP}"
