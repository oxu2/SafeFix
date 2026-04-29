#!/usr/bin/env bash
set -euo pipefail

ROOT="/people/cs/o/oxx220000/safefix"
PYTHON_EXE="/people/cs/o/oxx220000/.conda/envs/ad/bin/python"
BASE_MODEL="save/models/model_imagenet_resnet_base.pth"
DATA_DIR="data/imagenet_subset"
GEN_DIR="data/pinkcolor_fabrictexture_instruct_pix2pix_seed42_1k"
NUM_EPOCHS=15
NUM_ADDED=1000
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
WORKER_MODE=False

if [[ "${1:-}" == "--worker" ]]; then
  WORKER_MODE=True
  shift
  WORKER_GPU="$1"
  shift
  WORKER_RATIOS=("$@")
fi

cd "${ROOT}"
mkdir -p logs_imagenet logs

if [[ ! -x "${PYTHON_EXE}" ]]; then
  PYTHON_EXE="python3"
fi

if [[ ! -f "${BASE_MODEL}" ]]; then
  echo "Missing base model: ${BASE_MODEL}" >&2
  exit 1
fi

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "Missing ImageNet source data: ${DATA_DIR}" >&2
  exit 1
fi

if [[ ! -d "${GEN_DIR}" ]]; then
  echo "Missing generated image directory: ${GEN_DIR}" >&2
  exit 1
fi

run_one() {
  local gpu="$1"
  local ratio="$2"
  local method="$3"
  local rpc_tag="${4:-}"
  local ratio_tag="${ratio/./p}"
  local method_tag="${method}${rpc_tag:+_${rpc_tag}}"
  local log="logs_imagenet/instructpix2pix_${method_tag}_resnet_imagenet_1000_ratio${ratio_tag}_gpu${gpu}_${STAMP}.log"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] gpu=${gpu} ratio=${ratio} method=${method} log=${log}"
  if [[ -s "${log}" ]] && grep -q "Model Test ACC:" "${log}"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] gpu=${gpu} ratio=${ratio} method=${method_tag} already completed; skipping"
    return 0
  fi

  local -a args=(
    -u alpred.py
    --dataset imagenet
    --model resnet
    --num_added_img "${NUM_ADDED}"
    --attributes pinkcolor_fabrictexture
    --num_epochs "${NUM_EPOCHS}"
    --verbose True
    --resume_model "${BASE_MODEL}"
    --sub_ratio "${ratio}"
    --data_dir "${DATA_DIR}"
    --gen_image_dir "${GEN_DIR}"
    --skip_feasible True
  )

  case "${method}" in
    random)
      args+=(--use_active_learning False)
      ;;
    stratified)
      args+=(--al_strategy stratified)
      ;;
    coreset)
      args+=(--al_strategy coreset --al_initial_labeled 400 --al_embedding_dim 256)
      ;;
    badge)
      args+=(--al_strategy badge --al_initial_labeled 400 --al_embedding_dim 256)
      ;;
    attribute)
      case "${rpc_tag}" in
        r000_p0_c15)
          args+=(
            --al_attribute_rarity_power 0.0
            --al_attribute_frequency_penalty 0.0
            --al_attribute_class_balance 15.0
          )
          ;;
        r001_p5_cinf)
          args+=(
            --al_attribute_rarity_power 0.01
            --al_attribute_frequency_penalty 5.0
            --al_attribute_class_balance inf
          )
          ;;
        r01_p5_c15)
          args+=(
            --al_attribute_rarity_power 0.1
            --al_attribute_frequency_penalty 5.0
            --al_attribute_class_balance 15.0
          )
          ;;
        r08_p10_c4)
          args+=(
            --al_attribute_rarity_power 0.8
            --al_attribute_frequency_penalty 10.0
            --al_attribute_class_balance 4.0
          )
          ;;
        *)
          echo "Unknown or missing attribute RPC tag: ${rpc_tag}" >&2
          return 2
          ;;
      esac
      args+=(
        --al_strategy attribute
        --al_attribute_focus color,background,texture
        --al_attribute_use_entropy False
      )
      ;;
    *)
      echo "Unknown method: ${method}" >&2
      return 2
      ;;
  esac

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_EXE}" "${args[@]}" > "${log}" 2>&1
}

run_queue() {
  local gpu="$1"
  shift
  local ratios=("$@")
  local methods=(random stratified coreset badge attribute)
  local attribute_rpcs=(r000_p0_c15 r001_p5_cinf r01_p5_c15 r08_p10_c4)

  for ratio in "${ratios[@]}"; do
    for method in "${methods[@]}"; do
      if [[ "${method}" == "attribute" ]]; then
        for rpc in "${attribute_rpcs[@]}"; do
          run_one "${gpu}" "${ratio}" "${method}" "${rpc}"
        done
      else
        run_one "${gpu}" "${ratio}" "${method}"
      fi
    done
  done
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching ImageNet instruct-pix2pix local queues, stamp=${STAMP}"
echo "Source data: ${DATA_DIR}"
echo "Generated:   ${GEN_DIR}"

if [[ "${WORKER_MODE}" == "True" ]]; then
  run_queue "${WORKER_GPU}" "${WORKER_RATIOS[@]}"
  exit 0
fi

setsid nohup bash "$0" --worker 1 0.1 0.3 > "logs/imagenet_instructpix2pix_queue_gpu1_${STAMP}.log" 2>&1 < /dev/null &
pid1=$!
setsid nohup bash "$0" --worker 2 0.5 > "logs/imagenet_instructpix2pix_queue_gpu2_${STAMP}.log" 2>&1 < /dev/null &
pid2=$!
setsid nohup bash "$0" --worker 3 0.7 > "logs/imagenet_instructpix2pix_queue_gpu3_${STAMP}.log" 2>&1 < /dev/null &
pid3=$!

cat > "logs/imagenet_instructpix2pix_queues_${STAMP}.pids" <<EOF
gpu1 ${pid1} logs/imagenet_instructpix2pix_queue_gpu1_${STAMP}.log
gpu2 ${pid2} logs/imagenet_instructpix2pix_queue_gpu2_${STAMP}.log
gpu3 ${pid3} logs/imagenet_instructpix2pix_queue_gpu3_${STAMP}.log
EOF

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Queues launched:"
cat "logs/imagenet_instructpix2pix_queues_${STAMP}.pids"
