#!/usr/bin/env bash

set -euo pipefail

PYTHON_EXE="${PYTHON_EXE:-/people/cs/o/oxx220000/.conda/envs/ad/bin/python}"
BASE_MODEL="${BASE_MODEL:-save/weights/model_celeba_resnet_base.pth}"
ATTRIBUTE_FILE="${ATTRIBUTE_FILE:-save/attribute_celeba_trainval_qwen3_8B.json}"
ATTRIBUTES="${ATTRIBUTES:-redhair_brownskin_sademotion}"
GEN_IMAGE_DIR="${GEN_IMAGE_DIR:-data/${ATTRIBUTES}}"
STAMP="${STAMP:-20260504_024733}"
LAUNCH_STAMP="${LAUNCH_STAMP:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p logs/celeba/pids

rpc_params() {
  local rpc="$1"
  case "${rpc}" in
    r003_p05_c8) printf '0.03 0.5 8.0\n' ;;
    r002_p05_c15) printf '0.02 0.5 15.0\n' ;;
    r001_p05_c15) printf '0.01 0.5 15.0\n' ;;
    *)
      echo "Unsupported RPC: ${rpc}" >&2
      return 1
      ;;
  esac
}

run_attribute() {
  local gpu="$1"
  local rpc="$2"
  local rarity penalty class_balance log
  read -r rarity penalty class_balance < <(rpc_params "${rpc}")
  log="logs/celeba/mainfill_attribute_${rpc}_resnet_celeba_1000_ratio0p7_gpu${gpu}_${STAMP}.log"

  echo "[run] gpu=${gpu} ratio=0.7 attribute=${rpc} log=${log}"
  env MPLCONFIGDIR=/tmp/matplotlib-safefix CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_EXE}" -u alpred.py \
    --dataset celeba \
    --model resnet \
    --attributes "${ATTRIBUTES}" \
    --gen_image_dir "${GEN_IMAGE_DIR}" \
    --num_added_img 1000 \
    --num_epochs 15 \
    --resume_model "${BASE_MODEL}" \
    --freeze_backbone False \
    --lr 1e-5 \
    --batch_size 64 \
    --workers 4 \
    --seed 42 \
    --gpu "${gpu}" \
    --save_pred_model best \
    --save_preds True \
    --save_model False \
    --verbose False \
    --sub_ratio 0.7 \
    --use_active_learning True \
    --al_strategy attribute \
    --attribute_file "${ATTRIBUTE_FILE}" \
    --al_attribute_focus "hair color,skin,emotion" \
    --al_attribute_rarity_power "${rarity}" \
    --al_attribute_log_acc_scale 0.0 \
    --al_attribute_extra_counts "hair color:red=1000,skin:brown=1000,emotion:sad=1000" \
    --al_attribute_extra_count_scale 1.42857 \
    --al_attribute_frequency_penalty "${penalty}" \
    --al_attribute_class_balance "${class_balance}" \
    --al_attribute_use_entropy False > "${log}" 2>&1
}

launch_one() {
  local gpu="$1"
  local rpc="$2"
  local out="logs/celeba/mainfill_ratio0p7_refine3_${rpc}_gpu${gpu}_${LAUNCH_STAMP}.out"
  local pidfile="logs/celeba/pids/mainfill_ratio0p7_refine3_${rpc}_gpu${gpu}_${LAUNCH_STAMP}.pid"

  setsid bash "$0" --worker "${gpu}" "${rpc}" > "${out}" 2>&1 < /dev/null &
  echo "$!" > "${pidfile}"
  echo "gpu=${gpu} rpc=${rpc} pid=$(cat "${pidfile}") out=${out}"
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  run_attribute "$@"
  exit 0
fi

launch_one 1 r003_p05_c8
launch_one 2 r002_p05_c15
launch_one 3 r001_p05_c15
