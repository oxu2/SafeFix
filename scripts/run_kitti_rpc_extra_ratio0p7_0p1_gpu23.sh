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
DRY_RUN="${DRY_RUN:-0}"

SCRIPT="${ROOT_DIR}/object_detection/train_yolov8_kitti_generated.py"
PROJECT="${ROOT_DIR}/runs/kitti_yolov8"
DATA_ROOT="/data/oxx220000/kitti_yolov8_first5000"
GEN_DIR="/data/oxx220000/kitti-weather-edits/generated_kitti1352_red_cloudy_icedit_upscaled"
ATTRIBUTE_FILE="${ROOT_DIR}/save/attribute_kitti_qwen3_8B.json"
RUN_LOG_DIR="${ROOT_DIR}/logs"
PID_FILE="${RUN_LOG_DIR}/kitti_rpc_extra_ratio0p7_0p1_gpu023_${STAMP}.pids"

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

rpc_args() {
  local rpc="$1"
  case "${rpc}" in
    r000_p0_c15) echo "0.0 0.0 15" ;;
    r000_p0_cinf) echo "0.0 0.0 inf" ;;
    r001_p5_cinf) echo "0.01 5.0 inf" ;;
    r003_p05_c15) echo "0.03 0.5 15" ;;
    r003_p05_cinf) echo "0.03 0.5 inf" ;;
    r003_p5_c15) echo "0.03 5.0 15" ;;
    r01_p5_c15) echo "0.1 5.0 15" ;;
    r08_p10_c4) echo "0.8 10.0 4" ;;
    *)
      echo "Unknown RPC tag: ${rpc}" >&2
      return 2
      ;;
  esac
}

run_one() {
  local gpu="$1"
  local ratio="$2"
  local rpc="$3"
  local seed="$4"
  local ratio_tag_value
  ratio_tag_value="$(ratio_tag "${ratio}")"

  local tag="attribute_${rpc}_seed${seed}"
  local prepared_root="/data/oxx220000/kitti_yolov8_ratio${ratio_tag_value}_${tag}_red_cloudy_gen${NUM_ADDED_IMG}"
  local run_name="kitti_ratio${ratio_tag_value}_${tag}_red_cloudy_gen${NUM_ADDED_IMG}_e${EPOCHS}_gpu${gpu}_${STAMP}"
  local log_file="${RUN_LOG_DIR}/${run_name}.log"
  local rarity penalty class_balance
  read -r rarity penalty class_balance < <(rpc_args "${rpc}")

  local args=(
    "${PYTHON_EXE}" -u "${SCRIPT}"
    --base-data-root "${DATA_ROOT}"
    --generated-image-dir "${GEN_DIR}"
    --sub-ratio "${ratio}"
    --selection-strategy attribute
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
    --seed "${seed}"
    --eval-test true
    --log-file "${log_file}"
    --attribute-file "${ATTRIBUTE_FILE}"
    --al-attribute-focus color,weather
    --al-attribute-rarity-power "${rarity}"
    --al-attribute-frequency-penalty "${penalty}"
    --al-attribute-class-balance "${class_balance}"
    --al-conditional-rarity false
  )

  echo "[$(date '+%F %T')] START gpu=${gpu} ratio=${ratio} rpc=${rpc} seed=${seed}"
  echo "log=${log_file}"
  echo "prepared_root=${prepared_root}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    print_cmd "${args[@]}"
  else
    "${args[@]}"
  fi
  echo "[$(date '+%F %T')] END gpu=${gpu} ratio=${ratio} rpc=${rpc} seed=${seed}"
}

run_gpu0_ratio0p7() {
  run_one 0 0.7 r001_p5_cinf 42
  run_one 0 0.7 r01_p5_c15 42
  run_one 0 0.7 r08_p10_c4 42
}

run_gpu2_ratio0p7() {
  run_one 2 0.7 r000_p0_cinf 42
  run_one 2 0.7 r003_p05_c15 42
  run_one 2 0.7 r003_p05_cinf 42
}

run_gpu3_ratio0p1() {
  run_one 3 0.1 r001_p5_cinf 42
  run_one 3 0.1 r01_p5_c15 42
  run_one 3 0.1 r08_p10_c4 42
  run_one 3 0.1 r000_p0_cinf 42
  run_one 3 0.1 r003_p05_c15 42
  run_one 3 0.1 r003_p05_cinf 42
}

case "${1:-}" in
  --gpu0)
    run_gpu0_ratio0p7
    exit 0
    ;;
  --gpu2)
    run_gpu2_ratio0p7
    exit 0
    ;;
  --gpu3)
    run_gpu3_ratio0p1
    exit 0
    ;;
  --dry-run)
    DRY_RUN=1 bash "${BASH_SOURCE[0]}" --gpu0
    DRY_RUN=1 bash "${BASH_SOURCE[0]}" --gpu2
    DRY_RUN=1 bash "${BASH_SOURCE[0]}" --gpu3
    exit 0
    ;;
esac

if [[ "${DRY_RUN}" == "1" ]]; then
  bash "${BASH_SOURCE[0]}" --dry-run
  exit 0
fi

gpu0_out="${RUN_LOG_DIR}/kitti_rpc_extra_ratio0p7_gpu0_${STAMP}.out"
gpu2_out="${RUN_LOG_DIR}/kitti_rpc_extra_ratio0p7_gpu2_${STAMP}.out"
gpu3_out="${RUN_LOG_DIR}/kitti_rpc_extra_ratio0p1_gpu3_${STAMP}.out"

setsid nohup env STAMP="${STAMP}" PYTHON_EXE="${PYTHON_EXE}" NUM_ADDED_IMG="${NUM_ADDED_IMG}" EPOCHS="${EPOCHS}" BATCH="${BATCH}" IMG_SIZE="${IMG_SIZE}" WORKERS="${WORKERS}" \
  bash "${ROOT_DIR}/scripts/run_kitti_rpc_extra_ratio0p7_0p1_gpu23.sh" --gpu0 > "${gpu0_out}" 2>&1 < /dev/null &
gpu0_pid=$!

setsid nohup env STAMP="${STAMP}" PYTHON_EXE="${PYTHON_EXE}" NUM_ADDED_IMG="${NUM_ADDED_IMG}" EPOCHS="${EPOCHS}" BATCH="${BATCH}" IMG_SIZE="${IMG_SIZE}" WORKERS="${WORKERS}" \
  bash "${ROOT_DIR}/scripts/run_kitti_rpc_extra_ratio0p7_0p1_gpu23.sh" --gpu2 > "${gpu2_out}" 2>&1 < /dev/null &
gpu2_pid=$!

setsid nohup env STAMP="${STAMP}" PYTHON_EXE="${PYTHON_EXE}" NUM_ADDED_IMG="${NUM_ADDED_IMG}" EPOCHS="${EPOCHS}" BATCH="${BATCH}" IMG_SIZE="${IMG_SIZE}" WORKERS="${WORKERS}" \
  bash "${ROOT_DIR}/scripts/run_kitti_rpc_extra_ratio0p7_0p1_gpu23.sh" --gpu3 > "${gpu3_out}" 2>&1 < /dev/null &
gpu3_pid=$!

{
  echo "stamp=${STAMP}"
  echo "gpu0_pid=${gpu0_pid}"
  echo "gpu0_out=${gpu0_out}"
  echo "gpu2_pid=${gpu2_pid}"
  echo "gpu2_out=${gpu2_out}"
  echo "gpu3_pid=${gpu3_pid}"
  echo "gpu3_out=${gpu3_out}"
} | tee "${PID_FILE}"
