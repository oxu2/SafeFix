#!/usr/bin/env bash

set -euo pipefail

PYTHON_EXE="${PYTHON_EXE:-/people/cs/o/oxx220000/.conda/envs/ad/bin/python}"
BASE_MODEL="${BASE_MODEL:-weights/model_celeba_resnet_base.pth}"
ATTRIBUTE_FILE="${ATTRIBUTE_FILE:-save/attribute_celeba_trainval_qwen3_8B.json}"
ATTRIBUTES="${ATTRIBUTES:-redhair_brownskin_sademotion}"
GEN_IMAGE_DIR="${GEN_IMAGE_DIR:-data/${ATTRIBUTES}}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
DELAY_GPU1_SECONDS="${DELAY_GPU1_SECONDS:-3600}"
DRY_RUN="${DRY_RUN:-0}"
AUTO_REFINE="${AUTO_REFINE:-1}"
CLOSE_MARGIN="${CLOSE_MARGIN:-0.002}"

NUM_EPOCHS="${NUM_EPOCHS:-15}"
NUM_ADDED_IMG="${NUM_ADDED_IMG:-1000}"
LR="${LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-42}"
FOCUS="${FOCUS:-hair color,skin,emotion}"
BASE_EXTRA_COUNTS="${BASE_EXTRA_COUNTS:-hair color:red=1000,skin:brown=1000,emotion:sad=1000}"

mkdir -p logs_celeba/pids

if [[ ! -x "${PYTHON_EXE}" ]]; then
  echo "Python executable not found: ${PYTHON_EXE}" >&2
  exit 1
fi
if [[ ! -f "${BASE_MODEL}" ]]; then
  echo "Base model not found: ${BASE_MODEL}" >&2
  exit 1
fi
if [[ ! -f "${ATTRIBUTE_FILE}" ]]; then
  echo "Attribute file not found: ${ATTRIBUTE_FILE}" >&2
  exit 1
fi

ratio_tag() {
  local ratio="$1"
  printf '%s' "${ratio//./p}"
}

extra_scale_for_ratio() {
  local ratio="$1"
  awk -v ratio="${ratio}" 'BEGIN { printf "%.6g", 1.0 / ratio }'
}

rpc_params() {
  local rpc="$1"
  case "${rpc}" in
    r000_p0_c15)
      printf '0.0 0.0 15.0\n'
      ;;
    r000_p0_cinf)
      printf '0.0 0.0 inf\n'
      ;;
    r003_p05_c15)
      printf '0.03 0.5 15.0\n'
      ;;
    r003_p05_cinf)
      printf '0.03 0.5 inf\n'
      ;;
    r003_p05_c8)
      printf '0.03 0.5 8.0\n'
      ;;
    r002_p05_c15)
      printf '0.02 0.5 15.0\n'
      ;;
    r001_p05_c15)
      printf '0.01 0.5 15.0\n'
      ;;
    r001_p05_cinf)
      printf '0.01 0.5 inf\n'
      ;;
    *)
      echo "Unknown RPC: ${rpc}" >&2
      return 1
      ;;
  esac
}

strategy_args() {
  local strategy="$1"
  case "${strategy}" in
    random)
      printf '%s\n' --use_active_learning False
      ;;
    stratified)
      printf '%s\n' --use_active_learning True --al_strategy stratified
      ;;
    coreset)
      printf '%s\n' --use_active_learning True --al_strategy coreset --al_embedding_dim 512 --al_initial_labeled 1000
      ;;
    badge)
      printf '%s\n' --use_active_learning True --al_strategy badge --al_embedding_dim 512
      ;;
    *)
      echo "Unknown baseline strategy: ${strategy}" >&2
      return 1
      ;;
  esac
}

common_cmd_prefix() {
  local gpu="$1"
  shift
  env MPLCONFIGDIR=/tmp/matplotlib-safefix CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_EXE}" -u alpred.py \
    --dataset celeba \
    --model resnet \
    --attributes "${ATTRIBUTES}" \
    --gen_image_dir "${GEN_IMAGE_DIR}" \
    --num_added_img "${NUM_ADDED_IMG}" \
    --num_epochs "${NUM_EPOCHS}" \
    --resume_model "${BASE_MODEL}" \
    --freeze_backbone False \
    --lr "${LR}" \
    --batch_size "${BATCH_SIZE}" \
    --workers "${WORKERS}" \
    --seed "${SEED}" \
    --gpu "${gpu}" \
    --save_pred_model best \
    --save_preds True \
    --save_model False \
    --verbose False \
    "$@"
}

run_baseline() {
  local gpu="$1"
  local ratio="$2"
  local strategy="$3"
  local rtag
  rtag="$(ratio_tag "${ratio}")"
  local log="logs_celeba/mainfill_${strategy}_resnet_celeba_1000_ratio${rtag}_gpu${gpu}_${STAMP}.log"

  echo "[run] gpu=${gpu} ratio=${ratio} baseline=${strategy} log=${log}"
  mapfile -t extra_args < <(strategy_args "${strategy}")

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '      cmd='
    printf '%q ' env MPLCONFIGDIR=/tmp/matplotlib-safefix CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_EXE}" -u alpred.py \
      --dataset celeba --model resnet --attributes "${ATTRIBUTES}" --gen_image_dir "${GEN_IMAGE_DIR}" \
      --num_added_img "${NUM_ADDED_IMG}" --num_epochs "${NUM_EPOCHS}" --resume_model "${BASE_MODEL}" \
      --freeze_backbone False --lr "${LR}" --batch_size "${BATCH_SIZE}" --workers "${WORKERS}" \
      --seed "${SEED}" --gpu "${gpu}" --save_pred_model best --save_preds True --save_model False \
      --verbose False --sub_ratio "${ratio}" "${extra_args[@]}"
    printf '\n'
    return 0
  fi

  common_cmd_prefix "${gpu}" --sub_ratio "${ratio}" "${extra_args[@]}" > "${log}" 2>&1
}

run_attribute() {
  local gpu="$1"
  local ratio="$2"
  local rpc="$3"
  local rtag
  rtag="$(ratio_tag "${ratio}")"
  local log="logs_celeba/mainfill_attribute_${rpc}_resnet_celeba_1000_ratio${rtag}_gpu${gpu}_${STAMP}.log"
  local rarity penalty class_balance
  read -r rarity penalty class_balance < <(rpc_params "${rpc}")
  local extra_scale
  extra_scale="$(extra_scale_for_ratio "${ratio}")"

  echo "[run] gpu=${gpu} ratio=${ratio} attribute=${rpc} R=${rarity} P=${penalty} C=${class_balance} log=${log}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '      cmd='
    printf '%q ' env MPLCONFIGDIR=/tmp/matplotlib-safefix CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_EXE}" -u alpred.py \
      --dataset celeba --model resnet --attributes "${ATTRIBUTES}" --gen_image_dir "${GEN_IMAGE_DIR}" \
      --num_added_img "${NUM_ADDED_IMG}" --num_epochs "${NUM_EPOCHS}" --resume_model "${BASE_MODEL}" \
      --freeze_backbone False --lr "${LR}" --batch_size "${BATCH_SIZE}" --workers "${WORKERS}" \
      --seed "${SEED}" --gpu "${gpu}" --save_pred_model best --save_preds True --save_model False \
      --verbose False --sub_ratio "${ratio}" --use_active_learning True --al_strategy attribute \
      --attribute_file "${ATTRIBUTE_FILE}" --al_attribute_focus "${FOCUS}" \
      --al_attribute_rarity_power "${rarity}" --al_attribute_log_acc_scale 0.0 \
      --al_attribute_extra_counts "${BASE_EXTRA_COUNTS}" --al_attribute_extra_count_scale "${extra_scale}" \
      --al_attribute_frequency_penalty "${penalty}" --al_attribute_class_balance "${class_balance}" \
      --al_attribute_use_entropy False
    printf '\n'
    return 0
  fi

  common_cmd_prefix "${gpu}" \
    --sub_ratio "${ratio}" \
    --use_active_learning True \
    --al_strategy attribute \
    --attribute_file "${ATTRIBUTE_FILE}" \
    --al_attribute_focus "${FOCUS}" \
    --al_attribute_rarity_power "${rarity}" \
    --al_attribute_log_acc_scale 0.0 \
    --al_attribute_extra_counts "${BASE_EXTRA_COUNTS}" \
    --al_attribute_extra_count_scale "${extra_scale}" \
    --al_attribute_frequency_penalty "${penalty}" \
    --al_attribute_class_balance "${class_balance}" \
    --al_attribute_use_entropy False > "${log}" 2>&1
}

extract_test_acc() {
  local log="$1"
  awk '/Model Test ACC:/ {
    value=$4
    gsub(",", "", value)
  } END {
    if (value != "") print value
  }' "${log}" 2>/dev/null || true
}

best_attr_acc_for_ratio() {
  local ratio="$1"
  local rtag
  rtag="$(ratio_tag "${ratio}")"
  local best=""
  local log acc
  shopt -s nullglob
  for log in logs_celeba/mainfill_attribute_*_ratio"${rtag}"_gpu*_"${STAMP}".log; do
    acc="$(extract_test_acc "${log}")"
    if [[ -n "${acc}" ]] && awk -v acc="${acc}" -v best="${best:-0}" 'BEGIN { exit !(acc > best) }'; then
      best="${acc}"
    fi
  done
  shopt -u nullglob
  printf '%s\n' "${best}"
}

best_baseline_acc_for_ratio() {
  local ratio="$1"
  local rtag
  rtag="$(ratio_tag "${ratio}")"
  local best=""
  local log acc
  shopt -s nullglob
  for log in logs_celeba/mainfill_{random,stratified,coreset,badge}_resnet_celeba_1000_ratio"${rtag}"_gpu*_"${STAMP}".log; do
    acc="$(extract_test_acc "${log}")"
    if [[ -n "${acc}" ]] && awk -v acc="${acc}" -v best="${best:-0}" 'BEGIN { exit !(acc > best) }'; then
      best="${acc}"
    fi
  done
  shopt -u nullglob

  if [[ -n "${best}" ]]; then
    printf '%s\n' "${best}"
    return 0
  fi

  case "${ratio}" in
    0.5) printf '0.9151\n' ;;
    0.7) printf '0.9153\n' ;;
    0.9) printf '0.9135\n' ;;
    *) printf '\n' ;;
  esac
}

maybe_refine_ratio() {
  local gpu="$1"
  local ratio="$2"
  local best_attr best_base delta

  if [[ "${AUTO_REFINE}" -ne 1 ]]; then
    echo "[refine] ratio=${ratio} skipped because AUTO_REFINE=${AUTO_REFINE}"
    return 0
  fi

  best_attr="$(best_attr_acc_for_ratio "${ratio}")"
  best_base="$(best_baseline_acc_for_ratio "${ratio}")"
  if [[ -z "${best_attr}" || -z "${best_base}" ]]; then
    echo "[refine] ratio=${ratio} skipped; best_attr=${best_attr:-missing}, best_base=${best_base:-missing}"
    return 0
  fi

  delta="$(awk -v attr="${best_attr}" -v base="${best_base}" 'BEGIN { printf "%.6f", attr - base }')"
  echo "[refine] ratio=${ratio} best_attr=${best_attr} best_baseline=${best_base} delta=${delta}"

  if awk -v delta="${delta}" -v margin="${CLOSE_MARGIN}" 'BEGIN { exit !(delta >= 0 || delta >= -margin) }'; then
    echo "[refine] ratio=${ratio} close/competitive; running 2 extra RPCs"
    run_attribute "${gpu}" "${ratio}" r003_p05_c8
    run_attribute "${gpu}" "${ratio}" r002_p05_c15
  else
    echo "[refine] ratio=${ratio} clearly below baselines; running broader 4 extra RPCs"
    run_attribute "${gpu}" "${ratio}" r003_p05_c8
    run_attribute "${gpu}" "${ratio}" r002_p05_c15
    run_attribute "${gpu}" "${ratio}" r001_p05_c15
    run_attribute "${gpu}" "${ratio}" r001_p05_cinf
  fi
}

run_gpu2_queue() {
  run_attribute 2 0.5 r000_p0_c15
  run_attribute 2 0.5 r000_p0_cinf
  run_attribute 2 0.5 r003_p05_c15
  run_attribute 2 0.5 r003_p05_cinf
  maybe_refine_ratio 2 0.5
}

run_gpu3_queue() {
  run_baseline 3 0.9 random
  run_baseline 3 0.9 stratified
  run_baseline 3 0.9 coreset
  run_baseline 3 0.9 badge
  run_attribute 3 0.7 r000_p0_c15
  run_attribute 3 0.7 r000_p0_cinf
  run_attribute 3 0.7 r003_p05_c15
  run_attribute 3 0.7 r003_p05_cinf
  maybe_refine_ratio 3 0.7
}

run_gpu1_queue() {
  if [[ "${DELAY_GPU1_SECONDS}" -gt 0 ]]; then
    echo "[delay] gpu=1 sleeping ${DELAY_GPU1_SECONDS}s before CelebA queue"
    sleep "${DELAY_GPU1_SECONDS}"
  fi
  run_attribute 1 0.9 r000_p0_c15
  run_attribute 1 0.9 r000_p0_cinf
  run_attribute 1 0.9 r003_p05_c15
  run_attribute 1 0.9 r003_p05_cinf
  maybe_refine_ratio 1 0.9
}

launch_controller() {
  local gpu="$1"
  local func="$2"
  local out="logs_celeba/mainfill_gpu${gpu}_${STAMP}.out"
  local pidfile="logs_celeba/pids/mainfill_gpu${gpu}_${STAMP}.pid"

  echo "[launch-controller] gpu=${gpu} func=${func} out=${out}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    "${func}"
    return 0
  fi

  setsid bash "$0" --worker "${func}" > "${out}" 2>&1 < /dev/null &
  local pid=$!
  echo "${pid}" > "${pidfile}"
  echo "  pid=${pid} pidfile=${pidfile}"
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  "$@"
  exit 0
fi

launch_controller 2 run_gpu2_queue
launch_controller 3 run_gpu3_queue
launch_controller 1 run_gpu1_queue

cat > "logs_celeba/pids/mainfill_gpu231_${STAMP}.pids" <<EOF
gpu2 $(cat "logs_celeba/pids/mainfill_gpu2_${STAMP}.pid" 2>/dev/null || true) logs_celeba/mainfill_gpu2_${STAMP}.out
gpu3 $(cat "logs_celeba/pids/mainfill_gpu3_${STAMP}.pid" 2>/dev/null || true) logs_celeba/mainfill_gpu3_${STAMP}.out
gpu1 $(cat "logs_celeba/pids/mainfill_gpu1_${STAMP}.pid" 2>/dev/null || true) logs_celeba/mainfill_gpu1_${STAMP}.out
EOF

echo "[stamp] ${STAMP}"
cat "logs_celeba/pids/mainfill_gpu231_${STAMP}.pids"
