#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXE="${PYTHON_EXE:-/people/cs/o/oxx220000/.conda/envs/ad/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RATIO="0.7"
RATIO_TAG="0p7"
BASE_MODEL="${BASE_MODEL:-save/weights/model_imagenet_resnet_base.pth}"
ATTRIBUTE_FILE="${ATTRIBUTE_FILE:-save/attribute10_internvl3_8B.json}"
LOG_DIR="logs/imagenet"
PID_DIR="logs"

mkdir -p "${LOG_DIR}" "${PID_DIR}"

COMMON_ARGS=(
  --dataset imagenet
  --model resnet
  --num_added_img 1000
  --attributes pinkcolor_fabrictexture
  --gen_image_dir data/imagenet_pinkcolor_fabrictexture
  --num_epochs 15
  --lr 1e-5
  --verbose False
  --resume_model "${BASE_MODEL}"
  --sub_ratio "${RATIO}"
  --al_strategy attribute
  --al_attribute_focus color,background,texture
  --attribute_file "${ATTRIBUTE_FILE}"
  --al_attribute_use_entropy True
)

run_one() {
  local gpu="$1"
  local tag="$2"
  local rarity="$3"
  local penalty="$4"
  local class_balance="$5"
  local log="${LOG_DIR}/internvl3_ratio${RATIO_TAG}_${tag}_resnet_imagenet_1000_gpu${gpu}_${STAMP}.log"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu}: starting ${tag} (R=${rarity}, P=${penalty}, C=${class_balance})"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_EXE}" -u alpred.py \
    "${COMMON_ARGS[@]}" \
    --al_attribute_rarity_power "${rarity}" \
    --al_attribute_frequency_penalty "${penalty}" \
    --al_attribute_class_balance "${class_balance}" \
    > "${log}" 2>&1
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${gpu}: finished ${tag}; log=${log}"
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  run_one "$@"
  exit 0
fi

setsid nohup bash "$0" --worker 0 r000_p0_c15 0.0 0.0 15.0 > "${PID_DIR}/imagenet_internvl3_ratio${RATIO_TAG}_rpc4_gpu0_${STAMP}.out" 2>&1 < /dev/null &
pid0=$!
setsid nohup bash "$0" --worker 1 r001_p5_cinf 0.01 5.0 inf > "${PID_DIR}/imagenet_internvl3_ratio${RATIO_TAG}_rpc4_gpu1_${STAMP}.out" 2>&1 < /dev/null &
pid1=$!
setsid nohup bash "$0" --worker 2 r01_p5_c15 0.1 5.0 15.0 > "${PID_DIR}/imagenet_internvl3_ratio${RATIO_TAG}_rpc4_gpu2_${STAMP}.out" 2>&1 < /dev/null &
pid2=$!
setsid nohup bash "$0" --worker 3 r08_p10_c4 0.8 10.0 4.0 > "${PID_DIR}/imagenet_internvl3_ratio${RATIO_TAG}_rpc4_gpu3_${STAMP}.out" 2>&1 < /dev/null &
pid3=$!

pid_file="${PID_DIR}/imagenet_internvl3_ratio${RATIO_TAG}_rpc4_gpu0123_${STAMP}.pids"
cat > "${pid_file}" <<EOF
gpu0 ${pid0} ${PID_DIR}/imagenet_internvl3_ratio${RATIO_TAG}_rpc4_gpu0_${STAMP}.out ${LOG_DIR}/internvl3_ratio${RATIO_TAG}_r000_p0_c15_resnet_imagenet_1000_gpu0_${STAMP}.log
gpu1 ${pid1} ${PID_DIR}/imagenet_internvl3_ratio${RATIO_TAG}_rpc4_gpu1_${STAMP}.out ${LOG_DIR}/internvl3_ratio${RATIO_TAG}_r001_p5_cinf_resnet_imagenet_1000_gpu1_${STAMP}.log
gpu2 ${pid2} ${PID_DIR}/imagenet_internvl3_ratio${RATIO_TAG}_rpc4_gpu2_${STAMP}.out ${LOG_DIR}/internvl3_ratio${RATIO_TAG}_r01_p5_c15_resnet_imagenet_1000_gpu2_${STAMP}.log
gpu3 ${pid3} ${PID_DIR}/imagenet_internvl3_ratio${RATIO_TAG}_rpc4_gpu3_${STAMP}.out ${LOG_DIR}/internvl3_ratio${RATIO_TAG}_r08_p10_c4_resnet_imagenet_1000_gpu3_${STAMP}.log
EOF

echo "Launched ImageNet InternVL3 ratio ${RATIO} RPC4 runs with stamp ${STAMP}"
cat "${pid_file}"
