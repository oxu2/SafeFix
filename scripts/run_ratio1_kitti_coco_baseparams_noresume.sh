#!/usr/bin/env bash
set -euo pipefail

ROOT="/people/cs/o/oxx220000/safefix"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
KITTI_GPU="${KITTI_GPU:-0}"
COCO_GPU="${COCO_GPU:-1}"

cd "${ROOT}"
mkdir -p logs log_pose/pids

launch_kitti() {
  local prepared_root="/data/oxx220000/kitti_yolov8_ratio1p0_random_red_cloudy_gen1000_seed42_baseparams_noresume_${STAMP}"
  local run_name="kitti_ratio1p0_random_red_cloudy_gen1000_e30_baseparams_noresume_gpu${KITTI_GPU}_${STAMP}"
  local log="logs/${run_name}.log"
  local pidfile="logs/${run_name}.pid"

  setsid nohup bash -lc "
    set -euo pipefail
    cd '${ROOT}'
    source /usr/local/anaconda3/etc/profile.d/conda.sh
    conda activate fp
    export YOLO_CONFIG_DIR=/tmp/ultralytics
    export OMP_NUM_THREADS=4
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES='${KITTI_GPU}'
    python -u object_detection/train_yolov8_kitti_generated.py \
      --base-data-root /data/oxx220000/kitti_yolov8_first5000 \
      --generated-image-dir /data/oxx220000/kitti-weather-edits/generated_kitti1352_red_cloudy_icedit_upscaled \
      --sub-ratio 1.0 \
      --selection-strategy random \
      --num-added-img 1000 \
      --prepared-root '${prepared_root}' \
      --overwrite-prepared true \
      --prepare-only true \
      --seed 42 \
      --log-file '${log}.prepare'
    python -u object_detection/train_yolov8_kitti.py \
      --data-yaml '${prepared_root}/kitti_random_generated.yaml' \
      --pretrained true \
      --weights weights/yolov8n.pt \
      --epochs 30 \
      --batch 16 \
      --imgsz 640 \
      --device '${KITTI_GPU}' \
      --workers 4 \
      --plots false \
      --project '${ROOT}/runs/kitti_yolov8' \
      --name '${run_name}' \
      --seed 42 \
      --eval-test true
  " > "${log}" 2>&1 < /dev/null &
  local pid=$!
  echo "${pid}" > "${pidfile}"
  echo "KITTI_LOG=${log}"
  echo "KITTI_PID=${pid}"
  echo "KITTI_PIDFILE=${pidfile}"
  echo "KITTI_PREPARED_ROOT=${prepared_root}"
}

launch_coco() {
  local out_dir="log_pose/pose_keypointrcnn_brown_rainy_ratio1p0_gen1000_e2_baseparams_noresume_gpu${COCO_GPU}_${STAMP}"
  local log="log_pose/pose_qwen3_brown_rainy_ratio1p0_gen1000_e2_baseparams_noresume_gpu${COCO_GPU}_${STAMP}.log"
  local pidfile="log_pose/pids/pose_qwen3_brown_rainy_ratio1p0_gen1000_e2_baseparams_noresume_gpu${COCO_GPU}_${STAMP}.pid"

  setsid nohup bash -lc "
    set -euo pipefail
    cd '${ROOT}'
    source /usr/local/anaconda3/etc/profile.d/conda.sh
    conda activate ad
    export MPLCONFIGDIR=/tmp/matplotlib-safefix
    export CUDA_VISIBLE_DEVICES='${COCO_GPU}'
    python -u train_pose_estimation.py train \
      --data-root /data/oxx220000/coco \
      --pretrained true \
      --device cuda \
      --epochs 2 \
      --batch-size 8 \
      --num-workers 4 \
      --eval-every 1 \
      --lr 0.005 \
      --seed 42 \
      --max-train-samples 20000 \
      --sub-ratio 1.0 \
      --use-active-learning false \
      --attribute brown_rainy \
      --generated-image-dir /data/oxx220000/coco-pose-weather-edits/generated_coco_qwen3_1500_brown_rain_icedit \
      --num-added-img 1000 \
      --assigner qwen3 \
      --analyze-after-train false \
      --output-dir '${out_dir}'
  " > "${log}" 2>&1 < /dev/null &
  local pid=$!
  echo "${pid}" > "${pidfile}"
  echo "COCO_LOG=${log}"
  echo "COCO_PID=${pid}"
  echo "COCO_PIDFILE=${pidfile}"
  echo "COCO_OUT=${out_dir}"
}

echo "STAMP=${STAMP}"
launch_kitti
launch_coco
