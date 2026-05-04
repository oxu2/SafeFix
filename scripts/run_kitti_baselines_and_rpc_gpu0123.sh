#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/people/cs/o/oxx220000/safefix"
PYTHON_EXE="${PYTHON_EXE:-python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
NUM_ADDED_IMG="${NUM_ADDED_IMG:-1000}"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-16}"
IMG_SIZE="${IMG_SIZE:-640}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
QUEUE_MODE="${QUEUE_MODE:-launch}"
RUN_BASELINES="${RUN_BASELINES:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

SCRIPT="${ROOT_DIR}/object_detection/train_yolov8_kitti_generated.py"
PROJECT="${ROOT_DIR}/runs/kitti_yolov8"
DATA_ROOT="/data/oxx220000/kitti_yolov8_first5000"
GEN_DIR="/data/oxx220000/kitti-weather-edits/generated_kitti1352_red_cloudy_icedit_upscaled"
RUN_LOG_DIR="${ROOT_DIR}/logs"
PID_FILE="${RUN_LOG_DIR}/kitti_baselines_rpc_gpu0123_${STAMP}.pids"

cd "${ROOT_DIR}"
mkdir -p "${RUN_LOG_DIR}"

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate fp

ratio_tag() {
  local ratio="$1"
  echo "${ratio//./p}"
}

print_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

completed_log_exists() {
  local ratio="$1"
  local tag="$2"
  local ratio_tag_value
  ratio_tag_value="$(ratio_tag "${ratio}")"
  local pattern="${RUN_LOG_DIR}/kitti_ratio${ratio_tag_value}_${tag}_red_cloudy_gen${NUM_ADDED_IMG}_e${EPOCHS}_gpu*.log"
  local log
  for log in ${pattern}; do
    [[ -e "${log}" ]] || continue
    if grep -q "End timestamp:" "${log}"; then
      echo "${log}"
      return 0
    fi
  done
  return 1
}

attribute_rpc_args() {
  local rpc="$1"
  case "${rpc}" in
    r000_p0_c15)
      echo "0.0 0.0 15"
      ;;
    r001_p5_c15)
      echo "0.01 5.0 15"
      ;;
    r003_p5_c8)
      echo "0.03 5.0 8"
      ;;
    r003_p5_c15)
      echo "0.03 5.0 15"
      ;;
    *)
      echo "Unknown KITTI attribute RPC tag: ${rpc}" >&2
      return 2
      ;;
  esac
}

run_one() {
  local gpu="$1"
  local ratio="$2"
  local strategy="$3"
  local tag="$4"
  local rpc="${5:-}"
  local ratio_tag_value
  ratio_tag_value="$(ratio_tag "${ratio}")"

  if [[ "${SKIP_COMPLETED}" == "1" ]]; then
    local existing_log
    if existing_log="$(completed_log_exists "${ratio}" "${tag}")"; then
      echo "[$(date '+%F %T')] SKIP completed gpu=${gpu} ratio=${ratio} strategy=${tag}"
      echo "existing_log=${existing_log}"
      return 0
    fi
  fi

  local prepared_root="/data/oxx220000/kitti_yolov8_ratio${ratio_tag_value}_${tag}_red_cloudy_gen${NUM_ADDED_IMG}_seed${SEED}"
  local run_name="kitti_ratio${ratio_tag_value}_${tag}_red_cloudy_gen${NUM_ADDED_IMG}_e${EPOCHS}_gpu${gpu}_${STAMP}"
  local log_file="${RUN_LOG_DIR}/${run_name}.log"

  local args=(
    "${PYTHON_EXE}" -u "${SCRIPT}"
    --base-data-root "${DATA_ROOT}"
    --generated-image-dir "${GEN_DIR}"
    --sub-ratio "${ratio}"
    --selection-strategy "${strategy}"
    --num-added-img "${NUM_ADDED_IMG}"
    --prepared-root "${prepared_root}"
    --overwrite-prepared true
    --epochs "${EPOCHS}"
    --batch "${BATCH}"
    --imgsz "${IMG_SIZE}"
    --device "${gpu}"
    --workers "${WORKERS}"
    --project "${PROJECT}"
    --name "${run_name}"
    --seed "${SEED}"
    --eval-test true
    --log-file "${log_file}"
  )

  if [[ "${strategy}" == "attribute" ]]; then
    read -r rarity penalty class_balance < <(attribute_rpc_args "${rpc}")
    args+=(
      --attribute-file "${ROOT_DIR}/save/attribute_kitti_qwen3_8B.json"
      --al-attribute-focus color,weather
      --al-attribute-rarity-power "${rarity}"
      --al-attribute-frequency-penalty "${penalty}"
      --al-attribute-class-balance "${class_balance}"
      --al-conditional-rarity false
    )
  fi

  if [[ "${strategy}" == "coreset" || "${strategy}" == "badge" || "${strategy}" == "uherding" || "${strategy}" == "ppal" ]]; then
    args+=(
      --al-fast-selection true
      --al-initial-labeled 400
      --al-embedding-thumbnail-size 8
    )
  fi

  if [[ "${strategy}" == "uherding" ]]; then
    args+=(
      --al-uherding-uncertainty margin
      --al-uherding-temperature 1.0
    )
  fi

  echo "[$(date '+%F %T')] START gpu=${gpu} ratio=${ratio} strategy=${tag}"
  echo "log=${log_file}"
  echo "prepared_root=${prepared_root}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    print_cmd "${args[@]}"
  else
    "${args[@]}"
  fi
  echo "[$(date '+%F %T')] END gpu=${gpu} ratio=${ratio} strategy=${tag}"
}

run_ratio_queue() {
  local gpu="$1"
  shift
  local ratios=("$@")
  local ratio
  for ratio in "${ratios[@]}"; do
    if [[ "${RUN_BASELINES}" == "1" ]]; then
      run_one "${gpu}" "${ratio}" random random
      run_one "${gpu}" "${ratio}" stratified stratified
      run_one "${gpu}" "${ratio}" entropy entropy
      run_one "${gpu}" "${ratio}" coreset coreset_fast
      run_one "${gpu}" "${ratio}" badge badge_fast
      run_one "${gpu}" "${ratio}" uherding uherding_fast
      run_one "${gpu}" "${ratio}" ppal ppal_fast
    fi
    run_one "${gpu}" "${ratio}" attribute attribute_r000_p0_c15 r000_p0_c15
    run_one "${gpu}" "${ratio}" attribute attribute_r001_p5_c15 r001_p5_c15
    run_one "${gpu}" "${ratio}" attribute attribute_r003_p5_c8 r003_p5_c8
    run_one "${gpu}" "${ratio}" attribute attribute_r003_p5_c15 r003_p5_c15
  done
}

case "${QUEUE_MODE}" in
  gpu0)
    run_ratio_queue 0 0.1 0.3
    exit 0
    ;;
  gpu1)
    run_ratio_queue 1 0.5
    exit 0
    ;;
  gpu2)
    run_ratio_queue 2 0.7
    exit 0
    ;;
  gpu3)
    run_ratio_queue 3 0.9
    exit 0
    ;;
esac

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run for stamp ${STAMP}"
  QUEUE_MODE=gpu0 bash "${BASH_SOURCE[0]}"
  QUEUE_MODE=gpu1 bash "${BASH_SOURCE[0]}"
  QUEUE_MODE=gpu2 bash "${BASH_SOURCE[0]}"
  QUEUE_MODE=gpu3 bash "${BASH_SOURCE[0]}"
  exit 0
fi

declare -A out_files=(
  [gpu0]="${RUN_LOG_DIR}/kitti_baselines_rpc_gpu0_${STAMP}.out"
  [gpu1]="${RUN_LOG_DIR}/kitti_baselines_rpc_gpu1_${STAMP}.out"
  [gpu2]="${RUN_LOG_DIR}/kitti_baselines_rpc_gpu2_${STAMP}.out"
  [gpu3]="${RUN_LOG_DIR}/kitti_baselines_rpc_gpu3_${STAMP}.out"
)

echo "Launching KITTI baseline/RPC queues with stamp ${STAMP}"
echo "PID file: ${PID_FILE}"

{
  echo "stamp=${STAMP}"
  for mode in gpu0 gpu1 gpu2 gpu3; do
    out_file="${out_files[${mode}]}"
    setsid nohup env \
      STAMP="${STAMP}" \
      QUEUE_MODE="${mode}" \
      PYTHON_EXE="${PYTHON_EXE}" \
      NUM_ADDED_IMG="${NUM_ADDED_IMG}" \
      EPOCHS="${EPOCHS}" \
      BATCH="${BATCH}" \
      IMG_SIZE="${IMG_SIZE}" \
      WORKERS="${WORKERS}" \
      SEED="${SEED}" \
      RUN_BASELINES="${RUN_BASELINES}" \
      SKIP_COMPLETED="${SKIP_COMPLETED}" \
      bash "${ROOT_DIR}/scripts/run_kitti_baselines_and_rpc_gpu0123.sh" \
      > "${out_file}" 2>&1 < /dev/null &
    pid=$!
    echo "${mode}_pid=${pid}"
    echo "${mode}_out=${out_file}"
    echo "${mode} launched: pid=${pid}, out=${out_file}"
  done
} | tee "${PID_FILE}"
