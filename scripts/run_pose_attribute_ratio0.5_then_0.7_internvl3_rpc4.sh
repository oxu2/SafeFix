#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_pose_attribute_ratio0.5_then_0.7_internvl3_rpc4.sh [--base-model PATH] [--python PATH] [--attribute-file PATH] [--dry-run]

Launches the InternVL3 COCO pose attribute RPC4 sweep at ratio=0.5 on GPUs 0/1/2/3.
As soon as one GPU's ratio=0.5 run finishes, the controller launches that same GPU's
ratio=0.7 follow-up immediately without waiting for the other GPUs.
EOF
}

PYTHON_EXE="/people/cs/o/oxx220000/.conda/envs/ad/bin/python"
BASE_MODEL="log_pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530/best.pth"
ATTRIBUTE_FILE="save/attribute_coco_pose_gemini_train20k_internvl3_8B.json"
DATA_ROOT="/data/oxx220000/coco"
DRY_RUN=0

EPOCHS_TOTAL=2
GENERATED_ATTRIBUTE="rainy"
NUM_ADDED_IMG=1000
MAX_TRAIN_SAMPLES=20000
BATCH_SIZE=8
NUM_WORKERS=4
LR="0.005"
SEED=42
FOCUS="weather"

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

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

declare -a GPUS=(0 1 2 3)
declare -a NAMES_05=("attrx2_r000_p0_c0" "attrx2_r000_p0_c15" "attrx2_r001_p05_c0" "attrx2_r001_p05_c15")
declare -a NAMES_07=("attrx143_r000_p0_c0" "attrx143_r000_p0_c15" "attrx143_r001_p05_c0" "attrx143_r001_p05_c15")
declare -a RARITY=("0.0" "0.0" "0.01" "0.01")
declare -a PENALTY=("0.0" "0.0" "0.5" "0.5")
declare -a CLASS_BALANCE=("0" "15.0" "0" "15.0")

LAUNCHED_PID=""
LAUNCHED_LOG=""
LAUNCHED_OUTDIR=""
LAUNCHED_PIDFILE=""

launch_one() {
  local ratio="$1"
  local extra_scale="$2"
  local gpu="$3"
  local name="$4"
  local r="$5"
  local p="$6"
  local c="$7"

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

  echo "[launch] ratio=${ratio} gpu=${gpu} name=${name} R=${r} P=${p} C=${c}"
  echo "         extra_scale=${extra_scale}"
  echo "         log=${log_file}"
  echo "         out=${out_dir}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '         cmd='
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  "${cmd[@]}" > "${log_file}" 2>&1 &
  LAUNCHED_PID="$!"
  LAUNCHED_LOG="${log_file}"
  LAUNCHED_OUTDIR="${out_dir}"
  LAUNCHED_PIDFILE="${pid_file}"
  echo "${LAUNCHED_PID}" > "${pid_file}"
  echo "PID=${LAUNCHED_PID}"
  echo "PIDFILE=${pid_file}"
  echo "LOG=${log_file}"
  echo "OUTDIR=${out_dir}"
}

run_gpu_pipeline() {
  local idx="$1"
  local gpu="${GPUS[$idx]}"
  local name05="${NAMES_05[$idx]}"
  local name07="${NAMES_07[$idx]}"
  local r="${RARITY[$idx]}"
  local p="${PENALTY[$idx]}"
  local c="${CLASS_BALANCE[$idx]}"

  launch_one "0.5" "2" "${gpu}" "${name05}" "${r}" "${p}" "${c}"
  local pid05="${LAUNCHED_PID}"
  echo "[monitor] gpu=${gpu} waiting for ratio=0.5 pid=${pid05}"
  wait "${pid05}"
  local status05=$?
  echo "[monitor] gpu=${gpu} ratio=0.5 pid=${pid05} exit=${status05}"
  if [[ "${status05}" -ne 0 ]]; then
    echo "[monitor] gpu=${gpu} skipping ratio=0.7 because ratio=0.5 failed"
    return "${status05}"
  fi

  launch_one "0.7" "1.42857" "${gpu}" "${name07}" "${r}" "${p}" "${c}"
  local pid07="${LAUNCHED_PID}"
  echo "[monitor] gpu=${gpu} waiting for ratio=0.7 pid=${pid07}"
  wait "${pid07}"
  local status07=$?
  echo "[monitor] gpu=${gpu} ratio=0.7 pid=${pid07} exit=${status07}"
  return "${status07}"
}

if [[ "${DRY_RUN}" -eq 1 ]]; then
  for idx in "${!GPUS[@]}"; do
    gpu="${GPUS[$idx]}"
    name05="${NAMES_05[$idx]}"
    name07="${NAMES_07[$idx]}"
    r="${RARITY[$idx]}"
    p="${PENALTY[$idx]}"
    c="${CLASS_BALANCE[$idx]}"
    launch_one "0.5" "2" "${gpu}" "${name05}" "${r}" "${p}" "${c}"
    launch_one "0.7" "1.42857" "${gpu}" "${name07}" "${r}" "${p}" "${c}"
  done
  exit 0
fi

declare -a MONITORS=()
for idx in "${!GPUS[@]}"; do
  run_gpu_pipeline "${idx}" &
  MONITORS+=("$!")
done

overall_status=0
for pid in "${MONITORS[@]}"; do
  if ! wait "${pid}"; then
    overall_status=1
  fi
done

echo "[done] controller timestamp=${TIMESTAMP} overall_status=${overall_status}"
exit "${overall_status}"
