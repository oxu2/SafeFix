#!/usr/bin/env bash
#
# Usage: chmod +x batch.sh && ./batch.sh -m MODEL -a ATTRIBUTES -d CUDA_DEVICES [-g]
# Example:
#   nohup ./batch.sh -m resnet -a redhair_brownskin -d 3 > batch.log 2>&1 &


CUDA_DEVICES=0
MODEL=""
ATTRIBUTES=""
GEN_ONLY=false

RUN_SCRIPT="time_run.sh"
PREDICT_SCRIPT="predict.py"

# -m model, -a attributes, -d cuda_devices, -g gen_only
while getopts "m:a:d:g" opt; do
  case $opt in
    m) MODEL="$OPTARG" ;;
    a) ATTRIBUTES="$OPTARG" ;;
    d) CUDA_DEVICES="$OPTARG" ;;
    g) GEN_ONLY=true ;;
    *)
      echo "Usage: $(basename "$0") -m model -a attributes -d cuda_devices [-g]"
      exit 1
      ;;
  esac
done
shift $((OPTIND -1))


if [[ -z "$MODEL" || -z "$ATTRIBUTES" ]]; then
  echo "Error: model (-m) and attributes (-a) are required."
  exit 1
fi

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"


for NUM_ADDED_IMG in 1000 2000 4000; do
  if [[ "${GEN_ONLY}" == true ]]; then
    SUFFIX="g_"
  else
    SUFFIX=""
  fi
  LOG_FILE="${LOG_DIR}/${SUFFIX}${MODEL}_${ATTRIBUTES}_${NUM_ADDED_IMG}imgs.log"
  echo "=== [${NUM_ADDED_IMG} imgs] Starting ${MODEL} -> ${LOG_FILE} ==="

  CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} \
    bash "${RUN_SCRIPT}" "${PREDICT_SCRIPT}" \
      --model "${MODEL}" \
      --attributes "${ATTRIBUTES}" \
      --num_added_img "${NUM_ADDED_IMG}" \
      --gen_only "${GEN_ONLY}" \
      > "${LOG_FILE}" 2>&1

  if [[ $? -ne 0 ]]; then
    echo "!!! [${NUM_ADDED_IMG} imgs] ${MODEL} failed (exit code $?). See ${LOG_FILE}"
    exit 1
  fi

  echo "=== [${NUM_ADDED_IMG} imgs] Finished ${MODEL} ==="
  echo
done

echo "All batches completed successfully."