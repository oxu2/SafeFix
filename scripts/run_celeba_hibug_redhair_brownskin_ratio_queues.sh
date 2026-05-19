#!/usr/bin/env bash
set -euo pipefail

ROOT="/people/cs/o/oxx220000/safefix"
PYTHON_EXE="/people/cs/o/oxx220000/.conda/envs/ad/bin/python"
BASE_MODEL="save/weights/model_celeba_resnet_base.pth"
ATTRIBUTE_FILE="save/attribute_celeba_trainval_qwen3_8B.json"
ATTRIBUTES="redhair_brownskin"
GEN_DIR="data/hibug_redhair_brownskin_seed42_1k"
NUM_EPOCHS=15
NUM_ADDED=1000
BATCH_SIZE=64
WORKERS=4
SEED=42
LR="1e-5"
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
mkdir -p logs/celeba logs/celeba/pids logs

if [[ ! -x "${PYTHON_EXE}" ]]; then
  PYTHON_EXE="python3"
fi

if [[ ! -f "${BASE_MODEL}" ]]; then
  echo "Missing base model: ${BASE_MODEL}" >&2
  exit 1
fi

if [[ ! -f "${ATTRIBUTE_FILE}" ]]; then
  echo "Missing attribute file: ${ATTRIBUTE_FILE}" >&2
  exit 1
fi

if [[ ! -d "${GEN_DIR}" ]]; then
  echo "Missing generated image directory: ${GEN_DIR}" >&2
  exit 1
fi

ratio_tag() {
  local ratio="$1"
  printf '%s\n' "${ratio/./p}"
}

extra_scale_for_ratio() {
  local ratio="$1"
  awk -v ratio="${ratio}" 'BEGIN { printf "%.6g", 1.0 / ratio }'
}

run_one() {
  local gpu="$1"
  local ratio="$2"
  local method="$3"
  local rpc_tag="${4:-}"
  local rtag
  rtag="$(ratio_tag "${ratio}")"
  local method_tag="${method}${rpc_tag:+_${rpc_tag}}"
  local log="logs/celeba/hibug_redhair_brownskin_${method_tag}_resnet_celeba_1000_ratio${rtag}_gpu${gpu}_${STAMP}.log"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] gpu=${gpu} ratio=${ratio} method=${method_tag} log=${log}"
  if [[ -s "${log}" ]] && grep -q "Model Test ACC:" "${log}"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] gpu=${gpu} ratio=${ratio} method=${method_tag} already completed; skipping"
    return 0
  fi

  local -a args=(
    -u alpred.py
    --dataset celeba
    --model resnet
    --attributes "${ATTRIBUTES}"
    --gen_image_dir "${GEN_DIR}"
    --num_added_img "${NUM_ADDED}"
    --num_epochs "${NUM_EPOCHS}"
    --resume_model "${BASE_MODEL}"
    --freeze_backbone False
    --lr "${LR}"
    --batch_size "${BATCH_SIZE}"
    --workers "${WORKERS}"
    --seed "${SEED}"
    --gpu "${gpu}"
    --save_pred_model best
    --save_preds True
    --save_model False
    --verbose False
    --sub_ratio "${ratio}"
    --skip_feasible True
  )

  case "${method}" in
    random)
      args+=(--use_active_learning False)
      ;;
    stratified)
      args+=(--use_active_learning True --al_strategy stratified)
      ;;
    coreset)
      args+=(
        --use_active_learning True
        --al_strategy coreset
        --al_embedding_dim 512
        --al_initial_labeled 1000
      )
      ;;
    badge)
      args+=(
        --use_active_learning True
        --al_strategy badge
        --al_embedding_dim 512
      )
      ;;
    attribute)
      case "${rpc_tag}" in
        r003_p05_c15)
          args+=(--al_attribute_rarity_power 0.03 --al_attribute_frequency_penalty 0.5 --al_attribute_class_balance 15.0)
          ;;
        r000_p0_c15)
          args+=(--al_attribute_rarity_power 0.0 --al_attribute_frequency_penalty 0.0 --al_attribute_class_balance 15.0)
          ;;
        r000_p0_cinf)
          args+=(--al_attribute_rarity_power 0.0 --al_attribute_frequency_penalty 0.0 --al_attribute_class_balance inf)
          ;;
        r003_p05_cinf)
          args+=(--al_attribute_rarity_power 0.03 --al_attribute_frequency_penalty 0.5 --al_attribute_class_balance inf)
          ;;
        *)
          echo "Unknown or missing attribute RPC tag: ${rpc_tag}" >&2
          return 2
          ;;
      esac
      args+=(
        --use_active_learning True
        --al_strategy attribute
        --attribute_file "${ATTRIBUTE_FILE}"
        --al_attribute_focus "hair color,skin"
        --al_attribute_log_acc_scale 0.0
        --al_attribute_extra_counts "hair color:red=1000,skin:brown=1000"
        --al_attribute_extra_count_scale "$(extra_scale_for_ratio "${ratio}")"
        --al_attribute_use_entropy False
      )
      ;;
    *)
      echo "Unknown method: ${method}" >&2
      return 2
      ;;
  esac

  env MPLCONFIGDIR=/tmp/matplotlib-safefix CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_EXE}" "${args[@]}" > "${log}" 2>&1
}

run_queue() {
  local gpu="$1"
  shift
  local ratios=("$@")
  local methods=(random stratified coreset badge attribute)
  local attribute_rpcs=(r003_p05_c15 r000_p0_c15 r000_p0_cinf r003_p05_cinf)

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

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching CelebA HiBUG redhair_brownskin local queues, stamp=${STAMP}"
echo "Generated:      ${GEN_DIR}"
echo "Base model:     ${BASE_MODEL}"
echo "Attribute file: ${ATTRIBUTE_FILE}"
echo "Attributes:     ${ATTRIBUTES}"

if [[ "${WORKER_MODE}" == "True" ]]; then
  run_queue "${WORKER_GPU}" "${WORKER_RATIOS[@]}"
  exit 0
fi

setsid nohup bash "$0" --worker 1 0.1 0.3 > "logs/celeba/hibug_redhair_brownskin_queue_gpu1_${STAMP}.log" 2>&1 < /dev/null &
pid1=$!
setsid nohup bash "$0" --worker 2 0.5 > "logs/celeba/hibug_redhair_brownskin_queue_gpu2_${STAMP}.log" 2>&1 < /dev/null &
pid2=$!
setsid nohup bash "$0" --worker 3 0.7 > "logs/celeba/hibug_redhair_brownskin_queue_gpu3_${STAMP}.log" 2>&1 < /dev/null &
pid3=$!

cat > "logs/celeba/hibug_redhair_brownskin_queues_${STAMP}.pids" <<EOF
gpu1 ${pid1} logs/celeba/hibug_redhair_brownskin_queue_gpu1_${STAMP}.log
gpu2 ${pid2} logs/celeba/hibug_redhair_brownskin_queue_gpu2_${STAMP}.log
gpu3 ${pid3} logs/celeba/hibug_redhair_brownskin_queue_gpu3_${STAMP}.log
EOF

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Queues launched:"
cat "logs/celeba/hibug_redhair_brownskin_queues_${STAMP}.pids"
