#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-ad}"
GPU_ID="${2:-3}"
EPOCHS="${3:-12}"

ts="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs outputs

LOGFILE="logs/pose_coco_pretrained_gpu${GPU_ID}_${ENV_NAME}_${ts}.log"
PIDFILE="logs/pose_coco_pretrained_gpu${GPU_ID}_${ENV_NAME}_${ts}.pid"
OUTDIR="outputs/pose_keypointrcnn_${ENV_NAME}_gpu${GPU_ID}_${ts}"

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" nohup python train_pose_estimation.py train \
  --data-root /data/oxx220000/coco \
  --pretrained true \
  --device cuda \
  --batch-size 2 \
  --num-workers 4 \
  --epochs "${EPOCHS}" \
  --output-dir "${OUTDIR}" > "${LOGFILE}" 2>&1 &

echo $! > "${PIDFILE}"
echo "LOG=${LOGFILE}"
echo "PIDFILE=${PIDFILE}"
echo "OUTDIR=${OUTDIR}"
echo "PID=$(cat "${PIDFILE}")"

