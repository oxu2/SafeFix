#!/bin/bash
# Run all 4 LTL baselines across GPUs 0-3.
# GPU 0: focal
# GPU 1: class_weight
# GPU 2: resample
# GPU 3: ldam
# Each GPU runs ImageNet first, then CelebA.

PYTHON="/people/cs/o/oxx220000/.conda/envs/ad/bin/python"
SCRIPT="predict_ltl.py"
LOGDIR="logs"
mkdir -p "$LOGDIR"
SUFFIX=$(date +%m%d_%H%M)

echo "Starting LTL baseline experiments at $(date)"

# --- GPU 0: Focal Loss ---
(
  echo "[GPU 0] focal - imagenet starting at $(date)"
  CUDA_VISIBLE_DEVICES=0 $PYTHON -u $SCRIPT \
    --dataset imagenet --model resnet --ltl_baseline focal --gpu 0 --batch_size 32 \
    > $LOGDIR/ltl_focal_imagenet_resnet_${SUFFIX}.log 2>&1
  echo "[GPU 0] focal - imagenet finished at $(date)"

  echo "[GPU 0] focal - celeba starting at $(date)"
  CUDA_VISIBLE_DEVICES=0 $PYTHON -u $SCRIPT \
    --dataset celeba --model resnet --ltl_baseline focal --gpu 0 \
    > $LOGDIR/ltl_focal_celeba_resnet_${SUFFIX}.log 2>&1
  echo "[GPU 0] focal - celeba finished at $(date)"
) &
PID_GPU0=$!

# --- GPU 1: Class-Balanced Loss ---
(
  echo "[GPU 1] class_weight - imagenet starting at $(date)"
  CUDA_VISIBLE_DEVICES=1 $PYTHON -u $SCRIPT \
    --dataset imagenet --model resnet --ltl_baseline class_weight --gpu 0 --batch_size 32 \
    > $LOGDIR/ltl_classweight_imagenet_resnet_${SUFFIX}.log 2>&1
  echo "[GPU 1] class_weight - imagenet finished at $(date)"

  echo "[GPU 1] class_weight - celeba starting at $(date)"
  CUDA_VISIBLE_DEVICES=1 $PYTHON -u $SCRIPT \
    --dataset celeba --model resnet --ltl_baseline class_weight --gpu 0 \
    > $LOGDIR/ltl_classweight_celeba_resnet_${SUFFIX}.log 2>&1
  echo "[GPU 1] class_weight - celeba finished at $(date)"
) &
PID_GPU1=$!

# --- GPU 2: Resample ---
(
  echo "[GPU 2] resample - imagenet starting at $(date)"
  CUDA_VISIBLE_DEVICES=2 $PYTHON -u $SCRIPT \
    --dataset imagenet --model resnet --ltl_baseline resample --gpu 0 --batch_size 32 \
    > $LOGDIR/ltl_resample_imagenet_resnet_${SUFFIX}.log 2>&1
  echo "[GPU 2] resample - imagenet finished at $(date)"

  echo "[GPU 2] resample - celeba starting at $(date)"
  CUDA_VISIBLE_DEVICES=2 $PYTHON -u $SCRIPT \
    --dataset celeba --model resnet --ltl_baseline resample --gpu 0 \
    > $LOGDIR/ltl_resample_celeba_resnet_${SUFFIX}.log 2>&1
  echo "[GPU 2] resample - celeba finished at $(date)"
) &
PID_GPU2=$!

# --- GPU 3: LDAM ---
(
  echo "[GPU 3] ldam - imagenet starting at $(date)"
  CUDA_VISIBLE_DEVICES=3 $PYTHON -u $SCRIPT \
    --dataset imagenet --model resnet --ltl_baseline ldam --gpu 0 --batch_size 32 \
    > $LOGDIR/ltl_ldam_imagenet_resnet_${SUFFIX}.log 2>&1
  echo "[GPU 3] ldam - imagenet finished at $(date)"

  echo "[GPU 3] ldam - celeba starting at $(date)"
  CUDA_VISIBLE_DEVICES=3 $PYTHON -u $SCRIPT \
    --dataset celeba --model resnet --ltl_baseline ldam --gpu 0 \
    > $LOGDIR/ltl_ldam_celeba_resnet_${SUFFIX}.log 2>&1
  echo "[GPU 3] ldam - celeba finished at $(date)"
) &
PID_GPU3=$!

echo "All jobs launched. PIDs: GPU0=$PID_GPU0, GPU1=$PID_GPU1, GPU2=$PID_GPU2, GPU3=$PID_GPU3"
echo "Monitor with: tail -f logs/ltl_*.log"

# Wait for all to finish
wait $PID_GPU0 $PID_GPU1 $PID_GPU2 $PID_GPU3
echo "All LTL experiments completed at $(date)"
