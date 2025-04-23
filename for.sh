#!/usr/bin/env bash
#
# Sequentially run predict.py for resnet, clip, and vit models
# Usage: chmod +x for.sh && ./for.sh

# Configuration
CUDA_DEVICES=1
ATTRIBUTES="redhair_brownskin"
NUM_ADDED_IMG=1000
# Optional: override epochs if you like
# NUM_EPOCHS=20
RUN_SCRIPT="run.sh"
PREDICT_SCRIPT="predict.py"
LOG_DIR="logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

for MODEL in resnet clip vit; do
  LOG_FILE="${LOG_DIR}/train_${MODEL}_${ATTRIBUTES}_${NUM_ADDED_IMG}imgs.log"
  echo "=== Starting ${MODEL} (logs -> ${LOG_FILE}) ==="

  CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} \
    bash "${RUN_SCRIPT}" "${PREDICT_SCRIPT}" \
      --attributes "${ATTRIBUTES}" \
      --model "${MODEL}" \
      --num_added_img "${NUM_ADDED_IMG}" \
      > "${LOG_FILE}" 2>&1

  if [ $? -ne 0 ]; then
    echo "!!! ${MODEL} failed (exit code $?). Check ${LOG_FILE} for details."
    exit 1
  fi

  echo "=== Finished ${MODEL} ==="
  echo
done

echo "All models completed successfully."