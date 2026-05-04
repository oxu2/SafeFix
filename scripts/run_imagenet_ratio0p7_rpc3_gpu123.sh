#!/usr/bin/env bash

set -euo pipefail

ROOT="/people/cs/o/oxx220000/safefix"
PYTHON_EXE="${PYTHON_EXE:-/people/cs/o/oxx220000/.conda/envs/ad/bin/python}"
BASE_MODEL="${BASE_MODEL:-weights/model_imagenet_resnet_base.pth}"
ATTRIBUTE_FILE="${ATTRIBUTE_FILE:-save/attribute10_internvl3_8B.json}"
GEN_DIR="${GEN_DIR:-data/imagenet_pinkcolor_fabrictexture}"
DATA_DIR="${DATA_DIR:-data/imagenet_subset}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

cd "${ROOT}"
mkdir -p logs logs_imagenet

rpc_params() {
  local rpc="$1"
  case "${rpc}" in
    r003_p0_cinf) printf '0.03 0.0 inf\n' ;;
    r001_p05_cinf) printf '0.01 0.5 inf\n' ;;
    r003_p05_cinf) printf '0.03 0.5 inf\n' ;;
    *)
      echo "Unsupported RPC: ${rpc}" >&2
      return 1
      ;;
  esac
}

run_one() {
  local gpu="$1"
  local rpc="$2"
  local rarity penalty class_balance log
  read -r rarity penalty class_balance < <(rpc_params "${rpc}")
  log="logs_imagenet/controlnet_rpc_numeric_${rpc}_resnet_imagenet_1000_ratio0p7_gpu${gpu}_${STAMP}.log"

  echo "[run] gpu=${gpu} ratio=0.7 rpc=${rpc} R=${rarity} P=${penalty} C=${class_balance} log=${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_EXE}" -u alpred.py \
    --dataset imagenet \
    --model resnet \
    --num_added_img 1000 \
    --attributes pinkcolor_fabrictexture \
    --gen_image_dir "${GEN_DIR}" \
    --data_dir "${DATA_DIR}" \
    --num_epochs 15 \
    --lr 1e-5 \
    --verbose False \
    --resume_model "${BASE_MODEL}" \
    --sub_ratio 0.7 \
    --use_active_learning True \
    --al_strategy attribute \
    --attribute_file "${ATTRIBUTE_FILE}" \
    --al_attribute_focus color,background,texture \
    --al_attribute_rarity_power "${rarity}" \
    --al_attribute_frequency_penalty "${penalty}" \
    --al_attribute_class_balance "${class_balance}" \
    --al_attribute_use_entropy False \
    --skip_feasible True > "${log}" 2>&1
}

launch_one() {
  local gpu="$1"
  local rpc="$2"
  local out="logs/imagenet_ratio0p7_rpc3_${rpc}_gpu${gpu}_${STAMP}.out"
  local pidfile="logs/imagenet_ratio0p7_rpc3_${rpc}_gpu${gpu}_${STAMP}.pid"

  setsid bash "$0" --worker "${gpu}" "${rpc}" > "${out}" 2>&1 < /dev/null &
  echo "$!" > "${pidfile}"
  echo "gpu=${gpu} rpc=${rpc} pid=$(cat "${pidfile}") out=${out}"
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  run_one "$@"
  exit 0
fi

launch_one 1 r003_p0_cinf
launch_one 2 r001_p05_cinf
launch_one 3 r003_p05_cinf
