#!/usr/bin/env bash
set -euo pipefail

ROOT="/people/cs/o/oxx220000/safefix"
PYTHON_EXE="${PYTHON_EXE:-/people/cs/o/oxx220000/.conda/envs/ad/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RATIO="${RATIO:-0.9}"
RATIO_TAG="${RATIO//./p}"
COCO_ROOT="${COCO_ROOT:-/data/oxx220000/coco}"
GEN_DIR="${GEN_DIR:-/data/oxx220000/coco-pose-weather-edits/generated_coco_qwen3_1500_brown_rain_icedit}"
BASE_MODEL="${BASE_MODEL:-log_pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530/best.pth}"
ATTRIBUTE_FILE="${ATTRIBUTE_FILE:-save/attribute_coco_pose_gemini_train20k_qwen3_8B.json}"
LOG_DIR="log_pose"
PID_DIR="log_pose/pids"
DRY_RUN="${DRY_RUN:-0}"
INCLUDE_ANCHOR="${INCLUDE_ANCHOR:-0}"

cd "${ROOT}"
mkdir -p "${LOG_DIR}" "${PID_DIR}"

if [[ ! -x "${PYTHON_EXE}" ]]; then
  PYTHON_EXE="python3"
fi

for required in "${COCO_ROOT}" "${GEN_DIR}" "${BASE_MODEL}" "${ATTRIBUTE_FILE}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required path not found: ${required}" >&2
    exit 1
  fi
done

run_one() {
  local gpu="$1"
  local tag="$2"
  local out_dir="${LOG_DIR}/pose_keypointrcnn_resume_qwen3_brown_rainy_rpc_ablation_${tag}_ratio${RATIO_TAG}_gpu${gpu}_${STAMP}"
  local log="${LOG_DIR}/pose_qwen3_brown_rainy_rpc_ablation_${tag}_ratio${RATIO_TAG}_gpu${gpu}_${STAMP}.log"

  local -a args=(
    -u train_pose_estimation.py train
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
    --sub-ratio "${RATIO}"
    --resume-model "${BASE_MODEL}"
    --attribute brown_rainy
    --generated-image-dir "${GEN_DIR}"
    --num-added-img 1000
    --analyze-after-train false
    --assigner qwen3
    --output-dir "${out_dir}"
    --use-active-learning true
    --al-strategy attribute
    --al-attribute-file "${ATTRIBUTE_FILE}"
    --al-attribute-focus clothes_color,weather
    --al-attribute-rarity-power 0.03
    --al-attribute-uncertainty-boost 1.0
    --al-attribute-log-acc-scale 0.0
    --al-attribute-extra-counts "clothes_color:brown=1000,weather:rainy=1000"
    --al-attribute-extra-count-scale "$(awk -v ratio="${RATIO}" 'BEGIN { printf "%.6g", 1.0 / ratio }')"
    --al-attribute-frequency-penalty 0.5
    --al-attribute-class-balance 15.0
    --al-attribute-use-entropy false
  )

  case "${tag}" in
    anchor)
      ;;
    no_r)
      args+=(--al-attribute-use-rarity false)
      ;;
    no_p)
      args+=(--al-attribute-frequency-penalty 0.0)
      ;;
    no_c)
      args+=(--al-attribute-class-balance 0)
      ;;
    only_r)
      args+=(--al-attribute-frequency-penalty 0.0 --al-attribute-class-balance 0)
      ;;
    only_p)
      args+=(--al-attribute-use-rarity false --al-attribute-class-balance 0)
      ;;
    only_c)
      args+=(--al-attribute-use-rarity false --al-attribute-frequency-penalty 0.0)
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
  run_one "$@"
  exit 0
fi

echo "Launching COCO brown-rainy RPC component ablation, stamp=${STAMP}, ratio=${RATIO}"

declare -a records=()
if [[ "${INCLUDE_ANCHOR}" == "1" ]]; then
  records+=("0 anchor")
fi
records+=("1 no_r" "2 no_p" "3 no_c")

pid_file="${PID_DIR}/pose_brown_rainy_rpc_ablation_gpu0123_${STAMP}.pids"
: > "${pid_file}"

for record in "${records[@]}"; do
  read -r gpu tag <<< "${record}"
  out="${LOG_DIR}/pose_qwen3_brown_rainy_rpc_ablation_${tag}_gpu${gpu}_${STAMP}.out"
  setsid nohup bash "$0" --worker "${gpu}" "${tag}" > "${out}" 2>&1 < /dev/null &
  pid=$!
  echo "gpu${gpu} ${pid} ${out} ${LOG_DIR}/pose_qwen3_brown_rainy_rpc_ablation_${tag}_ratio${RATIO_TAG}_gpu${gpu}_${STAMP}.log" >> "${pid_file}"
done

cat "${pid_file}"
