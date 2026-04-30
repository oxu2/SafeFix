#!/usr/bin/env bash
set -euo pipefail

ROOT="/people/cs/o/oxx220000/safefix"
PYTHON_EXE="/people/cs/o/oxx220000/.conda/envs/ad/bin/python"
BASE_MODEL="weights/model_celeba_resnet_base.pth"
ATTRIBUTE_FILE="save/attribute_celeba_trainval_qwen3_8B.json"
ATTRIBUTES="redhair_brownskin_sademotion"
GEN_DIR="data/redhair_brownskin_sademotion_instruct_pix2pix_seed42_1k"
NUM_EPOCHS=15
NUM_ADDED=1000
BATCH_SIZE=64
WORKERS=4
SEED=42
LR="1e-5"
GPU=0
SOURCE_STAMP="20260429_071551"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

cd "${ROOT}"
mkdir -p logs_celeba logs_celeba/pids

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

has_completed_source_log() {
  local ratio="$1"
  local method_tag="$2"
  local rtag
  rtag="$(ratio_tag "${ratio}")"
  local candidate

  while IFS= read -r candidate; do
    if grep -q "Model Test ACC:" "${candidate}"; then
      return 0
    fi
  done < <(compgen -G "logs_celeba/instructpix2pix_${method_tag}_resnet_celeba_1000_ratio${rtag}_gpu*_${SOURCE_STAMP}.log" || true)
  return 1
}

run_one() {
  local ratio="$1"
  local method="$2"
  local rpc_tag="${3:-}"
  local rtag
  rtag="$(ratio_tag "${ratio}")"
  local method_tag="${method}${rpc_tag:+_${rpc_tag}}"
  local log="logs_celeba/instructpix2pix_${method_tag}_resnet_celeba_1000_ratio${rtag}_gpu${GPU}_resume_${STAMP}.log"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] gpu=${GPU} ratio=${ratio} method=${method_tag} log=${log}"
  if has_completed_source_log "${ratio}" "${method_tag}"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] source stamp already completed ratio=${ratio} method=${method_tag}; skipping"
    return 0
  fi
  if [[ -s "${log}" ]] && grep -q "Model Test ACC:" "${log}"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] resume log already completed ratio=${ratio} method=${method_tag}; skipping"
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
    --gpu "${GPU}"
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
          echo "Unknown or missing attribute RPC tag: ${rpc_tag}" >&2
          return 2
          ;;
      esac
      args+=(
        --use_active_learning True
        --al_strategy attribute
        --attribute_file "${ATTRIBUTE_FILE}"
        --al_attribute_focus "hair color,skin,emotion"
        --al_attribute_log_acc_scale 0.0
        --al_attribute_extra_counts "hair color:red=1000,skin:brown=1000,emotion:sad=1000"
        --al_attribute_extra_count_scale "$(extra_scale_for_ratio "${ratio}")"
        --al_attribute_use_entropy False
      )
      ;;
    *)
      echo "Unknown method: ${method}" >&2
      return 2
      ;;
  esac

  env MPLCONFIGDIR=/tmp/matplotlib-safefix CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_EXE}" "${args[@]}" > "${log}" 2>&1
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting CelebA instruct-pix2pix GPU ${GPU} missing-row resume, stamp=${STAMP}"
echo "Source stamp: ${SOURCE_STAMP}"

run_one 0.3 coreset
run_one 0.3 badge
run_one 0.3 attribute r003_p05_c15
run_one 0.3 attribute r000_p0_c15
run_one 0.3 attribute r000_p0_cinf
run_one 0.3 attribute r003_p05_cinf

run_one 0.5 attribute r003_p05_c15
run_one 0.5 attribute r000_p0_c15
run_one 0.5 attribute r000_p0_cinf
run_one 0.5 attribute r003_p05_cinf

run_one 0.7 badge
run_one 0.7 attribute r003_p05_c15
run_one 0.7 attribute r000_p0_c15
run_one 0.7 attribute r000_p0_cinf
run_one 0.7 attribute r003_p05_cinf

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed CelebA instruct-pix2pix GPU ${GPU} missing-row resume"
