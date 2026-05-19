#!/usr/bin/env bash
set -euo pipefail

ROOT="/people/cs/o/oxx220000/safefix"
PYTHON_EXE="${PYTHON_EXE:-/people/cs/o/oxx220000/.conda/envs/ad/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
GPU="${GPU:-3}"

RATIOS="${RATIOS:-0.1 0.3 0.5 0.7 0.9}"
TAGS="${TAGS:-none only_r only_p only_c no_r no_p no_c anchor}"

BASE_MODEL="${BASE_MODEL:-save/weights/model_imagenet_resnet_base.pth}"
ATTRIBUTE_FILE="${ATTRIBUTE_FILE:-save/attribute10_internvl3_8B.json}"
GEN_DIR="${GEN_DIR:-data/imagenet_pinkcolor_fabrictexture}"
DATA_DIR="${DATA_DIR:-data/imagenet_subset}"
LOG_DIR="${LOG_DIR:-logs/imagenet}"
PID_DIR="${PID_DIR:-logs}"

cd "${ROOT}"
mkdir -p "${LOG_DIR}" "${PID_DIR}"

if [[ ! -x "${PYTHON_EXE}" ]]; then
  PYTHON_EXE="python3"
fi

for required in "${BASE_MODEL}" "${ATTRIBUTE_FILE}" "${GEN_DIR}" "${DATA_DIR}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required path not found: ${required}" >&2
    exit 1
  fi
done

ratio_tag() {
  local ratio="$1"
  echo "${ratio//./p}"
}

add_rpc_overrides() {
  local tag="$1"
  case "${tag}" in
    anchor)
      ;;
    none)
      echo "--al_attribute_use_rarity False --al_attribute_frequency_penalty 0.0 --al_attribute_class_balance 0"
      ;;
    no_r)
      echo "--al_attribute_use_rarity False"
      ;;
    no_p)
      echo "--al_attribute_frequency_penalty 0.0"
      ;;
    no_c)
      echo "--al_attribute_class_balance 0"
      ;;
    only_r)
      echo "--al_attribute_frequency_penalty 0.0 --al_attribute_class_balance 0"
      ;;
    only_p)
      echo "--al_attribute_use_rarity False --al_attribute_class_balance 0"
      ;;
    only_c)
      echo "--al_attribute_use_rarity False --al_attribute_frequency_penalty 0.0"
      ;;
    *)
      echo "Unknown RPC tag: ${tag}" >&2
      return 2
      ;;
  esac
}

run_one() {
  local gpu="$1"
  local ratio="$2"
  local tag="$3"
  local rtag
  rtag="$(ratio_tag "${ratio}")"
  local log="${LOG_DIR}/controlnet_rpc8_noentropy_${tag}_resnet_imagenet_1000_ratio${rtag}_gpu${gpu}_${STAMP}.log"

  local -a args=(
    -u alpred.py
    --dataset imagenet
    --model resnet
    --num_added_img 1000
    --attributes pinkcolor_fabrictexture
    --gen_image_dir "${GEN_DIR}"
    --data_dir "${DATA_DIR}"
    --num_epochs 15
    --lr 1e-5
    --verbose False
    --resume_model "${BASE_MODEL}"
    --sub_ratio "${ratio}"
    --use_active_learning True
    --al_strategy attribute
    --attribute_file "${ATTRIBUTE_FILE}"
    --al_attribute_focus color,background,texture
    --al_attribute_rarity_power 0.01
    --al_attribute_frequency_penalty 5.0
    --al_attribute_class_balance inf
    --al_attribute_use_entropy False
    --skip_feasible True
  )

  read -r -a overrides <<< "$(add_rpc_overrides "${tag}")"
  args+=("${overrides[@]}")

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] dataset=imagenet gpu=${gpu} ratio=${ratio} tag=${tag} entropy=false log=${log}"
  if [[ -s "${log}" ]] && grep -q "Model Test ACC:" "${log}"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] already completed; skipping ${log}"
    return 0
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'cmd=CUDA_VISIBLE_DEVICES=%q %q ' "${gpu}" "${PYTHON_EXE}"
    printf '%q ' "${args[@]}"
    printf '\n'
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_EXE}" "${args[@]}" > "${log}" 2>&1
}

run_worker() {
  local gpu="$1"
  local ratio tag
  for ratio in ${RATIOS}; do
    for tag in ${TAGS}; do
      run_one "${gpu}" "${ratio}" "${tag}"
    done
  done
}

if [[ "${1:-}" == "--worker" ]]; then
  run_worker "${GPU}"
  exit 0
fi

out_log="${PID_DIR}/imagenet_controlnet_rpc8_noentropy_gpu${GPU}_${STAMP}.out"
pid_file="${PID_DIR}/imagenet_controlnet_rpc8_noentropy_gpu${GPU}_${STAMP}.pid"

echo "Launching ImageNet ControlNet RPC8 no-entropy sweep on GPU ${GPU}, stamp=${STAMP}"
echo "Ratios: ${RATIOS}"
echo "Tags: ${TAGS}"
echo "Worker output: ${out_log}"

if [[ "${DRY_RUN}" == "1" ]]; then
  bash "$0" --worker
  exit 0
fi

setsid nohup bash "$0" --worker > "${out_log}" 2>&1 < /dev/null &
pid=$!
printf '%s %s\n' "${pid}" "${out_log}" > "${pid_file}"
cat "${pid_file}"
