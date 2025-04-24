#!/usr/bin/env bash
#
# Sequentially run predict.py for resnet, clip, and vit models
# Usage: chmod +x for.sh && ./for.sh [-n num_added_img] [-a attributes] [-d cuda_devices]

# Configuration defaults
# Configuration defaults
CUDA_DEVICES=1
ATTRIBUTES="redhair_brownskin"
NUM_ADDED_IMG=1000

# Parse short options: -n for NUM_ADDED_IMG, -a for ATTRIBUTES, -d for CUDA_DEVICES
while getopts "n:a:d:" opt; do
  case $opt in
    n) NUM_ADDED_IMG="$OPTARG" ;;
    a) ATTRIBUTES="$OPTARG" ;;
    d) CUDA_DEVICES="$OPTARG" ;;
    *) echo "Usage: $(basename "$0") [-n num_added_img] [-a attributes] [-d cuda_devices]"; exit 1 ;;
  esac
done
shift $((OPTIND - 1))
# Optional: override epochs if you like
# NUM_EPOCHS=20
RUN_SCRIPT="time_run.sh"
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