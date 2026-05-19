#!/usr/bin/env bash
set -euo pipefail

ROOT="/people/cs/o/oxx220000/safefix"
PYTHON_EXE="${PYTHON_EXE:-/people/cs/o/oxx220000/.conda/envs/ad/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

RATIOS="${RATIOS:-0.1 0.3 0.5 0.7 0.9}"
TAGS="${TAGS:-none only_r only_p only_c no_r no_p no_c anchor}"

IMAGENET_BASE_MODEL="${IMAGENET_BASE_MODEL:-save/weights/model_imagenet_resnet_base.pth}"
IMAGENET_ATTRIBUTE_FILE="${IMAGENET_ATTRIBUTE_FILE:-save/attribute10_internvl3_8B.json}"
IMAGENET_GEN_DIR="${IMAGENET_GEN_DIR:-data/imagenet_pinkcolor_fabrictexture}"
IMAGENET_DATA_DIR="${IMAGENET_DATA_DIR:-data/imagenet_subset}"
IMAGENET_LOG_DIR="${IMAGENET_LOG_DIR:-logs/imagenet}"

COCO_ROOT="${COCO_ROOT:-/data/oxx220000/coco}"
COCO_GEN_DIR="${COCO_GEN_DIR:-/data/oxx220000/coco-pose-weather-edits/generated_coco_qwen3_1500_brown_rain_icedit}"
COCO_BASE_MODEL="${COCO_BASE_MODEL:-logs/pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530/best.pth}"
COCO_ATTRIBUTE_FILE="${COCO_ATTRIBUTE_FILE:-save/attribute_coco_pose_gemini_train20k_qwen3_8B.json}"
COCO_LOG_DIR="${COCO_LOG_DIR:-logs/pose}"

PID_DIR="${PID_DIR:-logs}"

cd "${ROOT}"
mkdir -p "${IMAGENET_LOG_DIR}" "${COCO_LOG_DIR}" "${PID_DIR}"

if [[ ! -x "${PYTHON_EXE}" ]]; then
  PYTHON_EXE="python3"
fi

for required in \
  "${IMAGENET_BASE_MODEL}" \
  "${IMAGENET_ATTRIBUTE_FILE}" \
  "${IMAGENET_GEN_DIR}" \
  "${IMAGENET_DATA_DIR}" \
  "${COCO_ROOT}" \
  "${COCO_GEN_DIR}" \
  "${COCO_BASE_MODEL}" \
  "${COCO_ATTRIBUTE_FILE}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required path not found: ${required}" >&2
    exit 1
  fi
done

ratio_tag() {
  local ratio="$1"
  echo "${ratio//./p}"
}

add_rpc_overrides_imagenet() {
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
      echo "Unknown ImageNet RPC tag: ${tag}" >&2
      return 2
      ;;
  esac
}

add_rpc_overrides_coco() {
  local tag="$1"
  case "${tag}" in
    anchor)
      ;;
    none)
      echo "--al-attribute-use-rarity false --al-attribute-frequency-penalty 0.0 --al-attribute-class-balance 0"
      ;;
    no_r)
      echo "--al-attribute-use-rarity false"
      ;;
    no_p)
      echo "--al-attribute-frequency-penalty 0.0"
      ;;
    no_c)
      echo "--al-attribute-class-balance 0"
      ;;
    only_r)
      echo "--al-attribute-frequency-penalty 0.0 --al-attribute-class-balance 0"
      ;;
    only_p)
      echo "--al-attribute-use-rarity false --al-attribute-class-balance 0"
      ;;
    only_c)
      echo "--al-attribute-use-rarity false --al-attribute-frequency-penalty 0.0"
      ;;
    *)
      echo "Unknown COCO RPC tag: ${tag}" >&2
      return 2
      ;;
  esac
}

run_imagenet_one() {
  local gpu="$1"
  local ratio="$2"
  local tag="$3"
  local rtag
  rtag="$(ratio_tag "${ratio}")"
  local log="${IMAGENET_LOG_DIR}/controlnet_rpc8_${tag}_resnet_imagenet_1000_ratio${rtag}_gpu${gpu}_${STAMP}.log"

  local -a args=(
    -u alpred.py
    --dataset imagenet
    --model resnet
    --num_added_img 1000
    --attributes pinkcolor_fabrictexture
    --gen_image_dir "${IMAGENET_GEN_DIR}"
    --data_dir "${IMAGENET_DATA_DIR}"
    --num_epochs 15
    --lr 1e-5
    --verbose False
    --resume_model "${IMAGENET_BASE_MODEL}"
    --sub_ratio "${ratio}"
    --use_active_learning True
    --al_strategy attribute
    --attribute_file "${IMAGENET_ATTRIBUTE_FILE}"
    --al_attribute_focus color,background,texture
    --al_attribute_rarity_power 0.01
    --al_attribute_frequency_penalty 5.0
    --al_attribute_class_balance inf
    --al_attribute_use_entropy True
    --skip_feasible True
  )

  read -r -a overrides <<< "$(add_rpc_overrides_imagenet "${tag}")"
  args+=("${overrides[@]}")

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] dataset=imagenet gpu=${gpu} ratio=${ratio} tag=${tag} log=${log}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'cmd=CUDA_VISIBLE_DEVICES=%q %q ' "${gpu}" "${PYTHON_EXE}"
    printf '%q ' "${args[@]}"
    printf '\n'
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_EXE}" "${args[@]}" > "${log}" 2>&1
}

run_coco_one() {
  local gpu="$1"
  local ratio="$2"
  local tag="$3"
  local rtag
  rtag="$(ratio_tag "${ratio}")"
  local out_dir="${COCO_LOG_DIR}/pose_keypointrcnn_resume_qwen3_brown_rainy_rpc8_${tag}_ratio${rtag}_gpu${gpu}_${STAMP}"
  local log="${COCO_LOG_DIR}/pose_qwen3_brown_rainy_rpc8_${tag}_ratio${rtag}_gpu${gpu}_${STAMP}.log"
  local extra_scale
  extra_scale="$(awk -v ratio="${ratio}" 'BEGIN { printf "%.6g", 1.0 / ratio }')"

  local -a args=(
    -u pose_estimation/train_pose_estimation.py train
    --data-root "${COCO_ROOT}"
    --pretrained true
    --device cuda
    --batch-size 8
    --num-workers 4
    --epochs 2
    --eval-every 1
    --lr 0.005
    --seed 42
    --max-train-samples 20000
    --sub-ratio "${ratio}"
    --resume-model "${COCO_BASE_MODEL}"
    --attribute brown_rainy
    --generated-image-dir "${COCO_GEN_DIR}"
    --num-added-img 1000
    --analyze-after-train false
    --assigner qwen3
    --output-dir "${out_dir}"
    --use-active-learning true
    --al-strategy attribute
    --al-attribute-file "${COCO_ATTRIBUTE_FILE}"
    --al-attribute-focus clothes_color,weather
    --al-attribute-rarity-power 0.03
    --al-attribute-uncertainty-boost 1.0
    --al-attribute-log-acc-scale 0.0
    --al-attribute-extra-counts "clothes_color:brown=1000,weather:rainy=1000"
    --al-attribute-extra-count-scale "${extra_scale}"
    --al-attribute-frequency-penalty 0.5
    --al-attribute-class-balance 15.0
    --al-attribute-use-entropy false
  )

  read -r -a overrides <<< "$(add_rpc_overrides_coco "${tag}")"
  args+=("${overrides[@]}")

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] dataset=coco gpu=${gpu} ratio=${ratio} tag=${tag} log=${log}"
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
  shift
  local item dataset ratio tag
  for item in "$@"; do
    IFS=: read -r dataset ratio tag <<< "${item}"
    case "${dataset}" in
      imagenet)
        run_imagenet_one "${gpu}" "${ratio}" "${tag}"
        ;;
      coco)
        run_coco_one "${gpu}" "${ratio}" "${tag}"
        ;;
      *)
        echo "Unknown dataset in queue item: ${item}" >&2
        return 2
        ;;
    esac
  done
}

append_grid() {
  local -n queue_ref="$1"
  local dataset="$2"
  shift 2
  local ratio tag
  for ratio in "$@"; do
    for tag in ${TAGS}; do
      queue_ref+=("${dataset}:${ratio}:${tag}")
    done
  done
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  worker_gpu="$1"
  shift
  run_worker "${worker_gpu}" "$@"
  exit 0
fi

declare -a gpu1_queue=()
declare -a gpu2_queue=()
declare -a gpu3_queue=()

# COCO is much larger and slower, so split it by approximate cost
# proportional to selected original samples. GPU 1 gets 0.9+0.3+0.1
# (cost 29), GPU 2 gets 0.7+0.5 (cost 26).
append_grid gpu1_queue coco 0.9 0.3 0.1
append_grid gpu2_queue coco 0.7 0.5

# ImageNet ControlNet rows are small and fast; keep them on one serial queue.
append_grid gpu3_queue imagenet ${RATIOS}

pid_file="${PID_DIR}/rpc8_ratio_sweep_gpu123_${STAMP}.pids"

echo "Launching RPC 8-combo ratio sweep on GPUs 1/2/3, stamp=${STAMP}"
echo "Tags: ${TAGS}"
echo "Ratios: ${RATIOS}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY_RUN] gpu1 queue length: ${#gpu1_queue[@]}"
  bash "$0" --worker 1 "${gpu1_queue[@]}"
  echo "[DRY_RUN] gpu2 queue length: ${#gpu2_queue[@]}"
  bash "$0" --worker 2 "${gpu2_queue[@]}"
  echo "[DRY_RUN] gpu3 queue length: ${#gpu3_queue[@]}"
  bash "$0" --worker 3 "${gpu3_queue[@]}"
  exit 0
fi

setsid nohup bash "$0" --worker 1 "${gpu1_queue[@]}" > "${PID_DIR}/rpc8_ratio_sweep_gpu1_${STAMP}.out" 2>&1 < /dev/null &
pid1=$!
setsid nohup bash "$0" --worker 2 "${gpu2_queue[@]}" > "${PID_DIR}/rpc8_ratio_sweep_gpu2_${STAMP}.out" 2>&1 < /dev/null &
pid2=$!
setsid nohup bash "$0" --worker 3 "${gpu3_queue[@]}" > "${PID_DIR}/rpc8_ratio_sweep_gpu3_${STAMP}.out" 2>&1 < /dev/null &
pid3=$!

cat > "${pid_file}" <<EOF
gpu1 ${pid1} ${PID_DIR}/rpc8_ratio_sweep_gpu1_${STAMP}.out coco ratios 0.9 0.3 0.1
gpu2 ${pid2} ${PID_DIR}/rpc8_ratio_sweep_gpu2_${STAMP}.out coco ratios 0.7 0.5
gpu3 ${pid3} ${PID_DIR}/rpc8_ratio_sweep_gpu3_${STAMP}.out imagenet ratios ${RATIOS}
EOF

cat "${pid_file}"
