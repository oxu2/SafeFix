#!/usr/bin/env bash
set -euo pipefail

ROOT="/people/cs/o/oxx220000/safefix"
PYTHON_EXE="/people/cs/o/oxx220000/.conda/envs/ad/bin/python"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RATIO="0.9"
RATIO_TAG="0p9"

IMAGENET_BASE_MODEL="save/models/model_imagenet_resnet_base.pth"
IMAGENET_DATA_DIR="data/imagenet_subset"
IMAGENET_GEN_DIR="data/pinkcolor_fabrictexture_instruct_pix2pix_seed42_1k"

CELEBA_BASE_MODEL="save/weights/model_celeba_resnet_base.pth"
CELEBA_ATTRIBUTE_FILE="save/attribute_celeba_trainval_qwen3_8B.json"
CELEBA_ATTRIBUTES="redhair_brownskin_sademotion"
CELEBA_GEN_DIR="data/redhair_brownskin_sademotion_instruct_pix2pix_seed42_1k"

cd "${ROOT}"
mkdir -p logs logs/imagenet logs/celeba logs/celeba/pids

if [[ ! -x "${PYTHON_EXE}" ]]; then
  PYTHON_EXE="python3"
fi

for path in \
  "${IMAGENET_BASE_MODEL}" \
  "${CELEBA_BASE_MODEL}" \
  "${CELEBA_ATTRIBUTE_FILE}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
done

for path in \
  "${IMAGENET_DATA_DIR}" \
  "${IMAGENET_GEN_DIR}" \
  "${CELEBA_GEN_DIR}"; do
  if [[ ! -d "${path}" ]]; then
    echo "Missing required directory: ${path}" >&2
    exit 1
  fi
done

extra_scale_for_ratio() {
  awk -v ratio="${RATIO}" 'BEGIN { printf "%.6g", 1.0 / ratio }'
}

run_imagenet_one() {
  local gpu="$1"
  local method="$2"
  local rpc_tag="${3:-}"
  local method_tag="${method}${rpc_tag:+_${rpc_tag}}"
  local log="logs/imagenet/instructpix2pix_${method_tag}_resnet_imagenet_1000_ratio${RATIO_TAG}_gpu${gpu}_${STAMP}.log"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] imagenet gpu=${gpu} ratio=${RATIO} method=${method_tag} log=${log}"
  if [[ -s "${log}" ]] && grep -q "Model Test ACC:" "${log}"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] imagenet ${method_tag} already completed; skipping"
    return 0
  fi

  local -a args=(
    -u alpred.py
    --dataset imagenet
    --model resnet
    --num_added_img 1000
    --attributes pinkcolor_fabrictexture
    --num_epochs 15
    --verbose True
    --resume_model "${IMAGENET_BASE_MODEL}"
    --sub_ratio "${RATIO}"
    --data_dir "${IMAGENET_DATA_DIR}"
    --gen_image_dir "${IMAGENET_GEN_DIR}"
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
          args+=(--al_attribute_rarity_power 0.0 --al_attribute_frequency_penalty 0.0 --al_attribute_class_balance 15.0)
          ;;
        r001_p5_cinf)
          args+=(--al_attribute_rarity_power 0.01 --al_attribute_frequency_penalty 5.0 --al_attribute_class_balance inf)
          ;;
        r01_p5_c15)
          args+=(--al_attribute_rarity_power 0.1 --al_attribute_frequency_penalty 5.0 --al_attribute_class_balance 15.0)
          ;;
        r08_p10_c4)
          args+=(--al_attribute_rarity_power 0.8 --al_attribute_frequency_penalty 10.0 --al_attribute_class_balance 4.0)
          ;;
        *)
          echo "Unknown ImageNet RPC tag: ${rpc_tag}" >&2
          return 2
          ;;
      esac
      args+=(--al_strategy attribute --al_attribute_focus color,background,texture --al_attribute_use_entropy False)
      ;;
    *)
      echo "Unknown ImageNet method: ${method}" >&2
      return 2
      ;;
  esac

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_EXE}" "${args[@]}" > "${log}" 2>&1
}

run_celeba_one() {
  local gpu="$1"
  local method="$2"
  local rpc_tag="${3:-}"
  local method_tag="${method}${rpc_tag:+_${rpc_tag}}"
  local log="logs/celeba/instructpix2pix_${method_tag}_resnet_celeba_1000_ratio${RATIO_TAG}_gpu${gpu}_${STAMP}.log"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] celeba gpu=${gpu} ratio=${RATIO} method=${method_tag} log=${log}"
  if [[ -s "${log}" ]] && grep -q "Model Test ACC:" "${log}"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] celeba ${method_tag} already completed; skipping"
    return 0
  fi

  local -a args=(
    -u alpred.py
    --dataset celeba
    --model resnet
    --attributes "${CELEBA_ATTRIBUTES}"
    --gen_image_dir "${CELEBA_GEN_DIR}"
    --num_added_img 1000
    --num_epochs 15
    --resume_model "${CELEBA_BASE_MODEL}"
    --freeze_backbone False
    --lr 1e-5
    --batch_size 64
    --workers 4
    --seed 42
    --gpu "${gpu}"
    --save_pred_model best
    --save_preds True
    --save_model False
    --verbose False
    --sub_ratio "${RATIO}"
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
      args+=(--use_active_learning True --al_strategy coreset --al_embedding_dim 512 --al_initial_labeled 1000)
      ;;
    badge)
      args+=(--use_active_learning True --al_strategy badge --al_embedding_dim 512)
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
          echo "Unknown CelebA RPC tag: ${rpc_tag}" >&2
          return 2
          ;;
      esac
      args+=(
        --use_active_learning True
        --al_strategy attribute
        --attribute_file "${CELEBA_ATTRIBUTE_FILE}"
        --al_attribute_focus "hair color,skin,emotion"
        --al_attribute_log_acc_scale 0.0
        --al_attribute_extra_counts "hair color:red=1000,skin:brown=1000,emotion:sad=1000"
        --al_attribute_extra_count_scale "$(extra_scale_for_ratio)"
        --al_attribute_use_entropy False
      )
      ;;
    *)
      echo "Unknown CelebA method: ${method}" >&2
      return 2
      ;;
  esac

  env MPLCONFIGDIR=/tmp/matplotlib-safefix CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_EXE}" "${args[@]}" > "${log}" 2>&1
}

run_gpu1_imagenet() {
  run_imagenet_one 1 random
  run_imagenet_one 1 stratified
  run_imagenet_one 1 coreset
  run_imagenet_one 1 badge
  run_imagenet_one 1 attribute r000_p0_c15
  run_imagenet_one 1 attribute r001_p5_cinf
  run_imagenet_one 1 attribute r01_p5_c15
  run_imagenet_one 1 attribute r08_p10_c4
}

run_gpu2_celeba_baselines() {
  run_celeba_one 2 random
  run_celeba_one 2 stratified
  run_celeba_one 2 coreset
  run_celeba_one 2 badge
}

run_gpu3_celeba_attributes() {
  run_celeba_one 3 attribute r003_p05_c15
  run_celeba_one 3 attribute r000_p0_c15
  run_celeba_one 3 attribute r000_p0_cinf
  run_celeba_one 3 attribute r003_p05_cinf
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  return 0
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching missing ratio=0.9 instruct-pix2pix jobs, stamp=${STAMP}"

setsid nohup bash -c 'source "$1"; run_gpu1_imagenet' _ "$0" > "logs/ratio0p9_missing_gpu1_imagenet_${STAMP}.log" 2>&1 < /dev/null &
pid1=$!
setsid nohup bash -c 'source "$1"; run_gpu2_celeba_baselines' _ "$0" > "logs/ratio0p9_missing_gpu2_celeba_baselines_${STAMP}.log" 2>&1 < /dev/null &
pid2=$!
setsid nohup bash -c 'source "$1"; run_gpu3_celeba_attributes' _ "$0" > "logs/ratio0p9_missing_gpu3_celeba_attributes_${STAMP}.log" 2>&1 < /dev/null &
pid3=$!

cat > "logs/ratio0p9_missing_gpu123_${STAMP}.pids" <<EOF
gpu1 ${pid1} logs/ratio0p9_missing_gpu1_imagenet_${STAMP}.log
gpu2 ${pid2} logs/ratio0p9_missing_gpu2_celeba_baselines_${STAMP}.log
gpu3 ${pid3} logs/ratio0p9_missing_gpu3_celeba_attributes_${STAMP}.log
EOF

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Queues launched:"
cat "logs/ratio0p9_missing_gpu123_${STAMP}.pids"
