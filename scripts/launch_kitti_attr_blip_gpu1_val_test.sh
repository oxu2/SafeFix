#!/usr/bin/env bash
set -euo pipefail

cd /people/cs/o/oxx220000/safefix

mkdir -p logs
ts=$(date +%Y%m%d_%H%M%S)
log="logs/assign_attr_kitti_val_test_blip_gpu1_${ts}.log"
pidfile="logs/assign_attr_kitti_val_test_blip_gpu1_${ts}.pid"

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate ad

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export BLIP_VQA_BATCH_SIZE=32
export BLIP_ITM_BATCH_SIZE=64
export BLIP_ITC_BATCH_SIZE=64

setsid nohup bash -lc '
python -u utils/blip.py \
  --dataset kitti \
  --kitti_root /data/oxx220000/kitti_yolov8_first5000 \
  --split val \
  --corpus save/corpus_kitti_gemini_clean.json \
  --with_attribute_selection \
  --output_labels save/labels_kitti_val_blip.npy \
  --output_attributes save/attribute_kitti_blip_val.json \
  --csv_output save/attribute_kitti_blip_val.csv \
  --text_labels_output save/label_kitti_val_blip_txt.npy \
  --train_idx_output save/train_idx_kitti_val_blip.npy \
  --val_idx_output save/val_idx_kitti_val_blip.npy

python -u utils/blip.py \
  --dataset kitti \
  --kitti_root /data/oxx220000/kitti_yolov8_first5000 \
  --split test \
  --corpus save/corpus_kitti_gemini_clean.json \
  --with_attribute_selection \
  --output_labels save/labels_kitti_test_blip.npy \
  --output_attributes save/attribute_kitti_blip_test.json \
  --csv_output save/attribute_kitti_blip_test.csv \
  --text_labels_output save/label_kitti_test_blip_txt.npy \
  --train_idx_output save/train_idx_kitti_test_blip.npy \
  --val_idx_output save/val_idx_kitti_test_blip.npy
' > "${log}" 2>&1 < /dev/null &

pid=$!
echo "${pid}" > "${pidfile}"
echo "LOG=${log}"
echo "PIDFILE=${pidfile}"
echo "PID=${pid}"
