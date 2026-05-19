#!/usr/bin/env bash
set -euo pipefail

ROOT="/people/cs/o/oxx220000/safefix"
WAIT_PID="${WAIT_PID:-936349}"
GPU="${GPU:-3}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RATIO="${RATIO:-0.05}"
RATIO_TAG="${RATIO//./p}"
NUM_ADDED_IMG="${NUM_ADDED_IMG:-1000}"
EPOCHS="${EPOCHS:-1}"
BATCH="${BATCH:-16}"
IMG_SIZE="${IMG_SIZE:-640}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-42}"

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  WAIT_PID=<pid> GPU=3 STAMP=<stamp> bash scripts/run_kitti_entropy_smoke_gpu3_after_queue.sh

Waits for WAIT_PID to exit, then runs a 1-epoch KITTI generated-image entropy
selection smoke on the requested physical GPU.
EOF
  exit 0
fi

cd "${ROOT}"
mkdir -p logs

echo "[$(date '+%F %T')] KITTI entropy smoke queued."
echo "wait_pid=${WAIT_PID}"
echo "gpu=${GPU}"
echo "stamp=${STAMP}"

if [[ "${WAIT_PID}" != "0" ]]; then
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 30
  done
fi

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate fp

RUN_NAME="kitti_ratio${RATIO_TAG}_entropy_red_cloudy_gen${NUM_ADDED_IMG}_e${EPOCHS}_gpu${GPU}_${STAMP}"
RUN_LOG="${ROOT}/logs/${RUN_NAME}.log"
PREPARED_ROOT="/data/oxx220000/kitti_yolov8_ratio${RATIO_TAG}_entropy_red_cloudy_gen${NUM_ADDED_IMG}_seed${SEED}"

echo "[$(date '+%F %T')] Starting KITTI entropy smoke."
echo "run_name=${RUN_NAME}"
echo "run_log=${RUN_LOG}"
echo "prepared_root=${PREPARED_ROOT}"
echo "physical_gpu=${GPU}"
echo "ultralytics_device=${GPU}"

python -u \
  object_detection/train_yolov8_kitti_generated.py \
  --base-data-root /data/oxx220000/kitti_yolov8_first5000 \
  --generated-image-dir /data/oxx220000/kitti-weather-edits/generated_kitti1352_red_cloudy_icedit_upscaled \
  --sub-ratio "${RATIO}" \
  --selection-strategy entropy \
  --num-added-img "${NUM_ADDED_IMG}" \
  --prepared-root "${PREPARED_ROOT}" \
  --overwrite-prepared true \
  --epochs "${EPOCHS}" \
  --batch "${BATCH}" \
  --imgsz "${IMG_SIZE}" \
  --device "${GPU}" \
  --workers "${WORKERS}" \
  --project "${ROOT}/save/runs/kitti_yolov8" \
  --name "${RUN_NAME}" \
  --seed "${SEED}" \
  --eval-test true \
  --log-file "${RUN_LOG}" \
  --al-entropy-conf-thres 0.001 \
  --al-entropy-batch 16

echo "[$(date '+%F %T')] KITTI entropy smoke finished."
