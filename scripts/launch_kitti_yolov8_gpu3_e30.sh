#!/usr/bin/env bash
set -euo pipefail

cd /people/cs/o/oxx220000/safefix

mkdir -p logs
ts=$(date +%Y%m%d_%H%M%S)
name="first5000_1240val_1241test_gpu3_e30_${ts}"
log="logs/kitti_yolov8_${name}.log"
pidfile="logs/kitti_yolov8_${name}.pid"

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate fp

export YOLO_CONFIG_DIR=/tmp/ultralytics
export OMP_NUM_THREADS=4
# Pin the CUDA order so numeric CUDA_VISIBLE_DEVICES matches nvidia-smi.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3

setsid nohup python -u object_detection/train_yolov8_kitti.py \
  --data-yaml /data/oxx220000/kitti_yolov8_first5000/kitti_first5000.yaml \
  --pretrained true \
  --weights save/weights/yolov8n.pt \
  --epochs 30 \
  --batch 64 \
  --imgsz 640 \
  --device 3 \
  --workers 4 \
  --plots false \
  --project /people/cs/o/oxx220000/safefix/save/runs/kitti_yolov8 \
  --name "${name}" \
  > "${log}" 2>&1 < /dev/null &

pid=$!
echo "${pid}" > "${pidfile}"
echo "LOG=${log}"
echo "PIDFILE=${pidfile}"
echo "PID=${pid}"
