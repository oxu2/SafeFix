#!/usr/bin/env bash
set -euo pipefail

ROOT="/people/cs/o/oxx220000/safefix"
PYTHON_EXE="${PYTHON_EXE:-/people/cs/o/oxx220000/.conda/envs/ad/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-save/selection_exports}"
LOG_DIR="${LOG_DIR:-logs}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-safefix}"

DATASETS=(imagenet celeba coco)
RATIOS=(${RATIOS_OVERRIDE:-0.1})
METHODS=(random stratified entropy coreset badge ppal uherding ours)
GPUS=(${GPUS_OVERRIDE:-0 1 2 3})

cd "${ROOT}"
mkdir -p "${LOG_DIR}" "${OUT_ROOT}" "${MPLCONFIGDIR}"

if [[ ! -x "${PYTHON_EXE}" ]]; then
  PYTHON_EXE="python"
fi

run_task() {
  local gpu="$1"
  local dataset="$2"
  local ratio="$3"
  local method="$4"
  local rtag="${ratio//./p}"
  local log="${LOG_DIR}/selection_export_${dataset}_ratio${rtag}_${method}_gpu${gpu}_${STAMP}.log"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START gpu=${gpu} dataset=${dataset} ratio=${ratio} method=${method} log=${log}"
  MPLCONFIGDIR="${MPLCONFIGDIR}" "${PYTHON_EXE}" scripts/export_selection_ids.py \
    --dataset "${dataset}" \
    --ratio "${ratio}" \
    --method "${method}" \
    --gpu "${gpu}" \
    --out-root "${OUT_ROOT}" \
    --batch-size 32 \
    --num-workers 4 > "${log}" 2>&1
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  gpu=${gpu} dataset=${dataset} ratio=${ratio} method=${method}"
}

worker() {
  local gpu="$1"
  local worker_idx="$2"
  local idx=0
  for dataset in "${DATASETS[@]}"; do
    for ratio in "${RATIOS[@]}"; do
      for method in "${METHODS[@]}"; do
        if (( idx % ${#GPUS[@]} == worker_idx )); then
          run_task "${gpu}" "${dataset}" "${ratio}" "${method}"
        fi
        idx=$((idx + 1))
      done
    done
  done
}

if [[ "${1:-}" == "--worker" ]]; then
  worker "${2:?gpu required}" "${3:?worker index required}"
  exit 0
fi

pid_file="${LOG_DIR}/selection_export_ratio0p1_0p3_gpu0123_${STAMP}.pids"
: > "${pid_file}"

for worker_idx in "${!GPUS[@]}"; do
  gpu="${GPUS[worker_idx]}"
  out_log="${LOG_DIR}/selection_export_worker_gpu${gpu}_${STAMP}.out"
  setsid nohup bash "$0" --worker "${gpu}" "${worker_idx}" > "${out_log}" 2>&1 < /dev/null &
  pid=$!
  echo "gpu${gpu} ${pid} ${out_log}" | tee -a "${pid_file}"
done

echo "PID_FILE=${pid_file}"
