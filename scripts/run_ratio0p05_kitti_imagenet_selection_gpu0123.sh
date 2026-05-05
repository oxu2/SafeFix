#!/usr/bin/env bash
set -euo pipefail

ROOT="/people/cs/o/oxx220000/safefix"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-safefix}"

AD_PYTHON="${AD_PYTHON:-/people/cs/o/oxx220000/.conda/envs/ad/bin/python}"
FP_PYTHON="${FP_PYTHON:-/people/cs/o/oxx220000/.conda/envs/fp/bin/python}"

KITTI_BASE_ROOT="${KITTI_BASE_ROOT:-/data/oxx220000/kitti_yolov8_first5000}"
KITTI_GEN_DIR="${KITTI_GEN_DIR:-/data/oxx220000/kitti-weather-edits/generated_kitti1352_red_cloudy_icedit_upscaled}"
KITTI_PROJECT="${KITTI_PROJECT:-runs/kitti_yolov8}"
KITTI_NUM_ADDED="${KITTI_NUM_ADDED:-1000}"
KITTI_BATCH="${KITTI_BATCH:-16}"
KITTI_IMGSZ="${KITTI_IMGSZ:-640}"
KITTI_WORKERS="${KITTI_WORKERS:-4}"
SEED="${SEED:-42}"

cd "${ROOT}"
mkdir -p "${LOG_DIR}" "${MPLCONFIGDIR}" selection_exports

if [[ ! -x "${AD_PYTHON}" ]]; then
  AD_PYTHON="python"
fi
if [[ ! -x "${FP_PYTHON}" ]]; then
  FP_PYTHON="python"
fi

run_kitti_prepare() {
  local gpu="$1"
  local method="$2"
  local tag="${method}_fast"
  local prepared_root="/data/oxx220000/kitti_yolov8_ratio0p05_${tag}_red_cloudy_gen${KITTI_NUM_ADDED}_seed${SEED}_${STAMP}"
  local run_name="kitti_ratio0p05_${tag}_red_cloudy_gen${KITTI_NUM_ADDED}_prepareonly_gpu${gpu}_${STAMP}"
  local log_file="${LOG_DIR}/${run_name}.log"
  local -a args=(
    "${FP_PYTHON}" -u object_detection/train_yolov8_kitti_generated.py
    --base-data-root "${KITTI_BASE_ROOT}"
    --generated-image-dir "${KITTI_GEN_DIR}"
    --sub-ratio 0.05
    --selection-strategy "${method}"
    --num-added-img "${KITTI_NUM_ADDED}"
    --prepared-root "${prepared_root}"
    --overwrite-prepared true
    --prepare-only true
    --epochs 1
    --batch "${KITTI_BATCH}"
    --imgsz "${KITTI_IMGSZ}"
    --device 0
    --workers "${KITTI_WORKERS}"
    --project "${KITTI_PROJECT}"
    --name "${run_name}"
    --seed "${SEED}"
    --eval-test false
    --log-file "${log_file}"
    --al-fast-selection true
    --al-initial-labeled 400
    --al-embedding-thumbnail-size 8
  )

  if [[ "${method}" == "uherding" ]]; then
    args+=(--al-uherding-uncertainty margin --al-uherding-temperature 1.0)
  else
    args+=(
      --al-ppal-pool-expand-ratio 4.0
      --al-ppal-class-weight-ub 0.2
      --al-ppal-class-weight-alpha 0.3
      --al-ppal-diversity-power 1.0
    )
  fi

  echo "[$(date '+%F %T')] START kitti ratio=0.05 method=${method} gpu=${gpu} log=${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" MPLCONFIGDIR="${MPLCONFIGDIR}" "${args[@]}"
  echo "[$(date '+%F %T')] DONE  kitti ratio=0.05 method=${method} gpu=${gpu} prepared_root=${prepared_root}"
}

run_imagenet_export() {
  local gpu="$1"
  local method="$2"
  local log_file="${LOG_DIR}/selection_export_imagenet_ratio0p05_${method}_gpu${gpu}_${STAMP}.log"
  local out_file="selection_exports/ratio0p05/imagenet/${method}/selected_original_ids.txt"

  if [[ -s "${out_file}" ]]; then
    echo "[$(date '+%F %T')] SKIP  imagenet ratio=0.05 method=${method} existing=${out_file}"
    return 0
  fi

  echo "[$(date '+%F %T')] START imagenet ratio=0.05 method=${method} gpu=${gpu} log=${log_file}"
  MPLCONFIGDIR="${MPLCONFIGDIR}" "${AD_PYTHON}" scripts/export_selection_ids.py \
    --dataset imagenet \
    --ratio 0.05 \
    --method "${method}" \
    --gpu "${gpu}" \
    --out-root selection_exports \
    --batch-size 32 \
    --num-workers 4 > "${log_file}" 2>&1
  echo "[$(date '+%F %T')] DONE  imagenet ratio=0.05 method=${method} gpu=${gpu}"
}

worker_gpu0() {
  run_kitti_prepare 0 uherding
}

worker_gpu1() {
  run_kitti_prepare 1 ppal
}

worker_gpu2() {
  for method in random entropy badge uherding; do
    run_imagenet_export 2 "${method}"
  done
}

worker_gpu3() {
  for method in stratified coreset ppal ours; do
    run_imagenet_export 3 "${method}"
  done
}

if [[ "${1:-}" == "--worker" ]]; then
  "worker_gpu${2:?gpu required}"
  exit 0
fi

pid_file="${LOG_DIR}/ratio0p05_kitti_imagenet_selection_gpu0123_${STAMP}.pids"
: > "${pid_file}"

for gpu in 0 1 2 3; do
  out_log="${LOG_DIR}/ratio0p05_worker_gpu${gpu}_${STAMP}.out"
  setsid nohup bash "$0" --worker "${gpu}" > "${out_log}" 2>&1 < /dev/null &
  pid=$!
  echo "gpu${gpu} ${pid} ${out_log}" | tee -a "${pid_file}"
done

echo "PID_FILE=${pid_file}"
