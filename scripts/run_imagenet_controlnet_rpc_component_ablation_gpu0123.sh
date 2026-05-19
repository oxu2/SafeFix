#!/usr/bin/env bash
set -euo pipefail

ROOT="/people/cs/o/oxx220000/safefix"
PYTHON_EXE="${PYTHON_EXE:-/people/cs/o/oxx220000/.conda/envs/ad/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RATIO="${RATIO:-0.7}"
RATIO_TAG="${RATIO//./p}"
BASE_MODEL="${BASE_MODEL:-save/weights/model_imagenet_resnet_base.pth}"
ATTRIBUTE_FILE="${ATTRIBUTE_FILE:-save/attribute10_internvl3_8B.json}"
GEN_DIR="${GEN_DIR:-data/imagenet_pinkcolor_fabrictexture}"
DATA_DIR="${DATA_DIR:-data/imagenet_subset}"
LOG_DIR="logs/imagenet"
PID_DIR="logs"
DRY_RUN="${DRY_RUN:-0}"

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

run_one() {
  local gpu="$1"
  local tag="$2"
  local log="${LOG_DIR}/controlnet_rpc_ablation_${tag}_resnet_imagenet_1000_ratio${RATIO_TAG}_gpu${gpu}_${STAMP}.log"

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
    --sub_ratio "${RATIO}"
    --use_active_learning True
    --al_strategy attribute
    --attribute_file "${ATTRIBUTE_FILE}"
    --al_attribute_focus color,background,texture
    --al_attribute_rarity_power 0.01
    --al_attribute_frequency_penalty 5.0
    --al_attribute_class_balance inf
    --al_attribute_use_entropy True
    --skip_feasible True
  )

  case "${tag}" in
    anchor)
      ;;
    no_r)
      args+=(--al_attribute_use_rarity False)
      ;;
    no_p)
      args+=(--al_attribute_frequency_penalty 0.0)
      ;;
    no_c)
      args+=(--al_attribute_class_balance 0)
      ;;
    no_u)
      args+=(--al_attribute_uncertainty_boost 0.0 --al_attribute_use_entropy False)
      ;;
    only_r)
      args+=(--al_attribute_frequency_penalty 0.0 --al_attribute_class_balance 0)
      ;;
    only_p)
      args+=(--al_attribute_use_rarity False --al_attribute_class_balance 0)
      ;;
    only_c)
      args+=(--al_attribute_use_rarity False --al_attribute_frequency_penalty 0.0)
      ;;
    *)
      echo "Unknown ablation tag: ${tag}" >&2
      return 2
      ;;
  esac

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] gpu=${gpu} tag=${tag} log=${log}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'cmd=CUDA_VISIBLE_DEVICES=%q %q ' "${gpu}" "${PYTHON_EXE}"
    printf '%q ' "${args[@]}"
    printf '\n'
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_EXE}" "${args[@]}" > "${log}" 2>&1
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  worker_gpu="$1"
  shift
  for tag in "$@"; do
    run_one "${worker_gpu}" "${tag}"
  done
  exit 0
fi

echo "Launching ImageNet ControlNet RPC component ablation, stamp=${STAMP}, ratio=${RATIO}"

setsid nohup bash "$0" --worker 0 anchor no_u > "${PID_DIR}/imagenet_controlnet_rpc_ablation_gpu0_${STAMP}.out" 2>&1 < /dev/null &
pid0=$!
setsid nohup bash "$0" --worker 1 no_r > "${PID_DIR}/imagenet_controlnet_rpc_ablation_no_r_gpu1_${STAMP}.out" 2>&1 < /dev/null &
pid1=$!
setsid nohup bash "$0" --worker 2 no_p > "${PID_DIR}/imagenet_controlnet_rpc_ablation_no_p_gpu2_${STAMP}.out" 2>&1 < /dev/null &
pid2=$!
setsid nohup bash "$0" --worker 3 no_c > "${PID_DIR}/imagenet_controlnet_rpc_ablation_no_c_gpu3_${STAMP}.out" 2>&1 < /dev/null &
pid3=$!

pid_file="${PID_DIR}/imagenet_controlnet_rpc_ablation_gpu0123_${STAMP}.pids"
cat > "${pid_file}" <<EOF
gpu0 ${pid0} ${PID_DIR}/imagenet_controlnet_rpc_ablation_gpu0_${STAMP}.out ${LOG_DIR}/controlnet_rpc_ablation_anchor_resnet_imagenet_1000_ratio${RATIO_TAG}_gpu0_${STAMP}.log then ${LOG_DIR}/controlnet_rpc_ablation_no_u_resnet_imagenet_1000_ratio${RATIO_TAG}_gpu0_${STAMP}.log
gpu1 ${pid1} ${PID_DIR}/imagenet_controlnet_rpc_ablation_no_r_gpu1_${STAMP}.out ${LOG_DIR}/controlnet_rpc_ablation_no_r_resnet_imagenet_1000_ratio${RATIO_TAG}_gpu1_${STAMP}.log
gpu2 ${pid2} ${PID_DIR}/imagenet_controlnet_rpc_ablation_no_p_gpu2_${STAMP}.out ${LOG_DIR}/controlnet_rpc_ablation_no_p_resnet_imagenet_1000_ratio${RATIO_TAG}_gpu2_${STAMP}.log
gpu3 ${pid3} ${PID_DIR}/imagenet_controlnet_rpc_ablation_no_c_gpu3_${STAMP}.out ${LOG_DIR}/controlnet_rpc_ablation_no_c_resnet_imagenet_1000_ratio${RATIO_TAG}_gpu3_${STAMP}.log
EOF

cat "${pid_file}"
