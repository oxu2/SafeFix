#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/people/cs/o/oxx220000/safefix"
PYTHON_EXE="${PYTHON_EXE:-python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RATIOS=(0.1 0.3 0.5 0.7 0.9)
NUM_ADDED_IMG="${NUM_ADDED_IMG:-1000}"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-16}"
IMG_SIZE="${IMG_SIZE:-640}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
QUEUE_MODE="${QUEUE_MODE:-launch}"

lookup_gpu_uuid() {
  local index="$1"
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader | \
    awk -F',' -v idx="${index}" '
      $1 + 0 == idx {
        gsub(/^[ \t]+|[ \t]+$/, "", $2)
        print $2
        exit
      }
    '
}

GPU0_VISIBLE="${GPU0_VISIBLE:-$(lookup_gpu_uuid 0)}"
GPU3_VISIBLE="${GPU3_VISIBLE:-$(lookup_gpu_uuid 3)}"

SCRIPT="${ROOT_DIR}/object_detection/train_yolov8_kitti_generated.py"
PROJECT="${ROOT_DIR}/save/runs/kitti_yolov8"
DATA_ROOT="/data/oxx220000/kitti_yolov8_first5000"
GEN_DIR="/data/oxx220000/kitti-weather-edits/generated_kitti1352_red_cloudy_icedit_upscaled"
RUN_LOG_DIR="${ROOT_DIR}/logs"
PID_FILE="${RUN_LOG_DIR}/kitti_generated_al_ratio_sweep_gpu03_${STAMP}.pids"
GPU0_OUT="${RUN_LOG_DIR}/kitti_generated_al_ratio_sweep_gpu0_${STAMP}.out"
GPU3_OUT="${RUN_LOG_DIR}/kitti_generated_al_ratio_sweep_gpu3_${STAMP}.out"

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

run_one() {
  local gpu="$1"
  local ratio="$2"
  local strategy="$3"
  local tag="$4"
  local gpu_uuid
  local ratio_tag_value
  if [[ "${gpu}" == "0" ]]; then
    gpu_uuid="${GPU0_VISIBLE}"
  elif [[ "${gpu}" == "3" ]]; then
    gpu_uuid="${GPU3_VISIBLE}"
  else
    gpu_uuid="${gpu}"
  fi
  ratio_tag_value="$(ratio_tag "${ratio}")"
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
    args+=(
      --attribute-file "${ROOT_DIR}/save/attribute_kitti_qwen3_8B.json"
      --al-attribute-focus color,weather
      --al-attribute-rarity-power 0.03
      --al-attribute-frequency-penalty 5
      --al-attribute-class-balance 15
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
  echo "physical_gpu=${gpu}"
  echo "gpu_uuid=${gpu_uuid}"
  echo "ultralytics_device=${gpu}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    print_cmd "${args[@]}"
  else
    "${args[@]}"
  fi
  echo "[$(date '+%F %T')] END gpu=${gpu} ratio=${ratio} strategy=${tag}"
}

run_gpu0_queue() {
  for ratio in "${RATIOS[@]}"; do
    run_one 0 "${ratio}" random random
    run_one 0 "${ratio}" attribute attribute_r003_p5_c15
    run_one 0 "${ratio}" stratified stratified
  done
}

run_gpu3_queue() {
  for ratio in "${RATIOS[@]}"; do
    run_one 3 "${ratio}" coreset coreset_fast
    run_one 3 "${ratio}" badge badge_fast
    run_one 3 "${ratio}" uherding uherding_fast
    run_one 3 "${ratio}" ppal ppal_fast
  done
}

if [[ "${QUEUE_MODE}" == "gpu0" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run for stamp ${STAMP}"
    echo "GPU0 queue: random -> attribute -> stratified for ratios ${RATIOS[*]}"
  fi
  run_gpu0_queue
  exit 0
fi

if [[ "${QUEUE_MODE}" == "gpu3" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run for stamp ${STAMP}"
    echo "GPU3 queue: coreset_fast -> badge_fast -> uherding_fast -> ppal_fast for ratios ${RATIOS[*]}"
  fi
  run_gpu3_queue
  exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run for stamp ${STAMP}"
  echo "GPU0 queue: random -> attribute -> stratified for ratios ${RATIOS[*]}"
  run_gpu0_queue
  echo "GPU3 queue: coreset_fast -> badge_fast -> uherding_fast -> ppal_fast for ratios ${RATIOS[*]}"
  run_gpu3_queue
  exit 0
fi

echo "Launching KITTI generated AL ratio sweep with stamp ${STAMP}"
echo "GPU0 output: ${GPU0_OUT}"
echo "GPU3 output: ${GPU3_OUT}"
echo "PID file: ${PID_FILE}"

setsid nohup env \
  STAMP="${STAMP}" \
  QUEUE_MODE="gpu0" \
  GPU0_VISIBLE="${GPU0_VISIBLE}" \
  GPU3_VISIBLE="${GPU3_VISIBLE}" \
  PYTHON_EXE="${PYTHON_EXE}" \
  NUM_ADDED_IMG="${NUM_ADDED_IMG}" \
  EPOCHS="${EPOCHS}" \
  BATCH="${BATCH}" \
  IMG_SIZE="${IMG_SIZE}" \
  WORKERS="${WORKERS}" \
  SEED="${SEED}" \
  bash "${ROOT_DIR}/scripts/run_kitti_generated_al_ratio_sweep_gpu03.sh" \
  > "${GPU0_OUT}" 2>&1 < /dev/null &
gpu0_pid=$!
setsid nohup env \
  STAMP="${STAMP}" \
  QUEUE_MODE="gpu3" \
  GPU0_VISIBLE="${GPU0_VISIBLE}" \
  GPU3_VISIBLE="${GPU3_VISIBLE}" \
  PYTHON_EXE="${PYTHON_EXE}" \
  NUM_ADDED_IMG="${NUM_ADDED_IMG}" \
  EPOCHS="${EPOCHS}" \
  BATCH="${BATCH}" \
  IMG_SIZE="${IMG_SIZE}" \
  WORKERS="${WORKERS}" \
  SEED="${SEED}" \
  bash "${ROOT_DIR}/scripts/run_kitti_generated_al_ratio_sweep_gpu03.sh" \
  > "${GPU3_OUT}" 2>&1 < /dev/null &
gpu3_pid=$!

{
  echo "stamp=${STAMP}"
  echo "gpu0_pid=${gpu0_pid}"
  echo "gpu3_pid=${gpu3_pid}"
  echo "gpu0_visible=${GPU0_VISIBLE}"
  echo "gpu3_visible=${GPU3_VISIBLE}"
  echo "gpu0_out=${GPU0_OUT}"
  echo "gpu3_out=${GPU3_OUT}"
} > "${PID_FILE}"

echo "gpu0_pid=${gpu0_pid}"
echo "gpu3_pid=${gpu3_pid}"
echo "Wrote ${PID_FILE}"
