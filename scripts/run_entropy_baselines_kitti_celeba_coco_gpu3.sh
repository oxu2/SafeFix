#!/usr/bin/env bash
set -euo pipefail

ROOT="/people/cs/o/oxx220000/safefix"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
GPU="${GPU:-3}"
RATIOS="${RATIOS:-0.1 0.3 0.5 0.7 0.9}"
DATASETS="${DATASETS:-kitti celeba coco}"
DRY_RUN="${DRY_RUN:-0}"

FP_PYTHON="${FP_PYTHON:-/people/cs/o/oxx220000/.conda/envs/fp/bin/python}"
AD_PYTHON="${AD_PYTHON:-/people/cs/o/oxx220000/.conda/envs/ad/bin/python}"

KITTI_BASE_ROOT="${KITTI_BASE_ROOT:-/data/oxx220000/kitti_yolov8_first5000}"
KITTI_GEN_DIR="${KITTI_GEN_DIR:-/data/oxx220000/kitti-weather-edits/generated_kitti1352_red_cloudy_icedit_upscaled}"
KITTI_PROJECT="${KITTI_PROJECT:-${ROOT}/save/runs/kitti_yolov8}"
KITTI_NUM_ADDED="${KITTI_NUM_ADDED:-1000}"
KITTI_EPOCHS="${KITTI_EPOCHS:-10}"
KITTI_BATCH="${KITTI_BATCH:-16}"
KITTI_IMGSZ="${KITTI_IMGSZ:-640}"
KITTI_WORKERS="${KITTI_WORKERS:-4}"

CELEBA_BASE_MODEL="${CELEBA_BASE_MODEL:-save/weights/model_celeba_resnet_base.pth}"
CELEBA_ATTRIBUTES="${CELEBA_ATTRIBUTES:-redhair_brownskin_sademotion}"
CELEBA_GEN_DIR="${CELEBA_GEN_DIR:-data/redhair_brownskin_sademotion}"
CELEBA_NUM_ADDED="${CELEBA_NUM_ADDED:-1000}"
CELEBA_EPOCHS="${CELEBA_EPOCHS:-15}"
CELEBA_BATCH="${CELEBA_BATCH:-64}"
CELEBA_WORKERS="${CELEBA_WORKERS:-4}"
CELEBA_LR="${CELEBA_LR:-1e-5}"

COCO_ROOT="${COCO_ROOT:-/data/oxx220000/coco}"
COCO_GEN_DIR="${COCO_GEN_DIR:-/data/oxx220000/coco-pose-weather-edits/generated_coco_qwen3_1500_brown_rain_icedit}"
COCO_BASE_MODEL="${COCO_BASE_MODEL:-logs/pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530/best.pth}"
COCO_NUM_ADDED="${COCO_NUM_ADDED:-1000}"
COCO_EPOCHS="${COCO_EPOCHS:-2}"
COCO_BATCH="${COCO_BATCH:-8}"
COCO_WORKERS="${COCO_WORKERS:-4}"
COCO_LR="${COCO_LR:-0.005}"
COCO_MAX_TRAIN="${COCO_MAX_TRAIN:-20000}"

SEED="${SEED:-42}"

cd "${ROOT}"
mkdir -p logs logs/celeba logs/pose

if [[ ! -x "${FP_PYTHON}" ]]; then
  FP_PYTHON="python"
fi
if [[ ! -x "${AD_PYTHON}" ]]; then
  AD_PYTHON="python3"
fi

for required in \
  "${KITTI_BASE_ROOT}" \
  "${KITTI_GEN_DIR}" \
  "${CELEBA_BASE_MODEL}" \
  "${CELEBA_GEN_DIR}" \
  "${COCO_ROOT}" \
  "${COCO_GEN_DIR}" \
  "${COCO_BASE_MODEL}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required path not found: ${required}" >&2
    exit 1
  fi
done

ratio_tag() {
  local ratio="$1"
  echo "${ratio//./p}"
}

print_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

run_kitti_one() {
  local ratio="$1"
  local rtag
  rtag="$(ratio_tag "${ratio}")"
  local prepared_root="/data/oxx220000/kitti_yolov8_ratio${rtag}_entropy_red_cloudy_gen${KITTI_NUM_ADDED}_seed${SEED}"
  local run_name="kitti_ratio${rtag}_entropy_red_cloudy_gen${KITTI_NUM_ADDED}_e${KITTI_EPOCHS}_gpu${GPU}_${STAMP}"
  local log_file="logs/${run_name}.log"

  local -a args=(
    "${FP_PYTHON}" -u object_detection/train_yolov8_kitti_generated.py
    --base-data-root "${KITTI_BASE_ROOT}"
    --generated-image-dir "${KITTI_GEN_DIR}"
    --sub-ratio "${ratio}"
    --selection-strategy entropy
    --num-added-img "${KITTI_NUM_ADDED}"
    --prepared-root "${prepared_root}"
    --overwrite-prepared true
    --epochs "${KITTI_EPOCHS}"
    --batch "${KITTI_BATCH}"
    --imgsz "${KITTI_IMGSZ}"
    --device "${GPU}"
    --workers "${KITTI_WORKERS}"
    --project "${KITTI_PROJECT}"
    --name "${run_name}"
    --seed "${SEED}"
    --eval-test true
    --log-file "${log_file}"
    --al-entropy-conf-thres 0.001
    --al-entropy-batch 16
  )

  echo "[$(date '+%F %T')] START dataset=kitti ratio=${ratio} gpu=${GPU} log=${log_file}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    print_cmd "${args[@]}"
  else
    "${args[@]}"
  fi
  echo "[$(date '+%F %T')] END dataset=kitti ratio=${ratio} gpu=${GPU}"
}

run_celeba_one() {
  local ratio="$1"
  local rtag
  rtag="$(ratio_tag "${ratio}")"
  local log_file="logs/celeba/controlnet_entropy_${CELEBA_ATTRIBUTES}_resnet_celeba_${CELEBA_NUM_ADDED}_ratio${rtag}_gpu${GPU}_${STAMP}.log"

  local -a args=(
    "${AD_PYTHON}" -u alpred.py
    --dataset celeba
    --model resnet
    --attributes "${CELEBA_ATTRIBUTES}"
    --gen_image_dir "${CELEBA_GEN_DIR}"
    --num_added_img "${CELEBA_NUM_ADDED}"
    --num_epochs "${CELEBA_EPOCHS}"
    --resume_model "${CELEBA_BASE_MODEL}"
    --freeze_backbone False
    --lr "${CELEBA_LR}"
    --batch_size "${CELEBA_BATCH}"
    --workers "${CELEBA_WORKERS}"
    --seed "${SEED}"
    --gpu "${GPU}"
    --save_pred_model best
    --save_preds True
    --save_model False
    --verbose False
    --sub_ratio "${ratio}"
    --skip_feasible True
    --use_active_learning True
    --al_strategy entropy
  )

  echo "[$(date '+%F %T')] START dataset=celeba ratio=${ratio} gpu=${GPU} log=${log_file}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPU}"
    print_cmd "${args[@]}"
  else
    CUDA_VISIBLE_DEVICES="${GPU}" "${args[@]}" > "${log_file}" 2>&1
  fi
  echo "[$(date '+%F %T')] END dataset=celeba ratio=${ratio} gpu=${GPU}"
}

run_coco_one() {
  local ratio="$1"
  local rtag
  rtag="$(ratio_tag "${ratio}")"
  local out_dir="logs/pose/pose_keypointrcnn_resume_qwen3_brown_rainy_entropy_ratio${rtag}_gpu${GPU}_${STAMP}"
  local log_file="logs/pose/pose_qwen3_brown_rainy_entropy_ratio${rtag}_gpu${GPU}_${STAMP}.log"

  local -a args=(
    "${AD_PYTHON}" -u pose_estimation/train_pose_estimation.py train
    --data-root "${COCO_ROOT}"
    --pretrained true
    --device cuda
    --batch-size "${COCO_BATCH}"
    --num-workers "${COCO_WORKERS}"
    --epochs "${COCO_EPOCHS}"
    --eval-every 1
    --lr "${COCO_LR}"
    --seed "${SEED}"
    --max-train-samples "${COCO_MAX_TRAIN}"
    --sub-ratio "${ratio}"
    --resume-model "${COCO_BASE_MODEL}"
    --attribute brown_rainy
    --generated-image-dir "${COCO_GEN_DIR}"
    --num-added-img "${COCO_NUM_ADDED}"
    --analyze-after-train false
    --assigner qwen3
    --output-dir "${out_dir}"
    --use-active-learning true
    --al-strategy entropy
    --al-entropy-batch-size 2
  )

  echo "[$(date '+%F %T')] START dataset=coco ratio=${ratio} gpu=${GPU} log=${log_file}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPU}"
    print_cmd "${args[@]}"
  else
    CUDA_VISIBLE_DEVICES="${GPU}" "${args[@]}" > "${log_file}" 2>&1
  fi
  echo "[$(date '+%F %T')] END dataset=coco ratio=${ratio} gpu=${GPU}"
}

run_worker() {
  local dataset ratio
  for dataset in ${DATASETS}; do
    for ratio in ${RATIOS}; do
      case "${dataset}" in
        kitti)
          run_kitti_one "${ratio}"
          ;;
        celeba)
          run_celeba_one "${ratio}"
          ;;
        coco)
          run_coco_one "${ratio}"
          ;;
        *)
          echo "Unknown dataset: ${dataset}" >&2
          return 2
          ;;
      esac
    done
  done
}

if [[ "${1:-}" == "--worker" ]]; then
  run_worker
  exit 0
fi

out_log="logs/entropy_baselines_kitti_celeba_coco_gpu${GPU}_${STAMP}.out"
pid_file="logs/entropy_baselines_kitti_celeba_coco_gpu${GPU}_${STAMP}.pid"

echo "Launching entropy baselines on GPU ${GPU}, stamp=${STAMP}"
echo "Datasets: ${DATASETS}"
echo "Ratios: ${RATIOS}"
echo "Worker output: ${out_log}"

export STAMP GPU RATIOS DATASETS DRY_RUN
export FP_PYTHON AD_PYTHON
export KITTI_BASE_ROOT KITTI_GEN_DIR KITTI_PROJECT KITTI_NUM_ADDED KITTI_EPOCHS KITTI_BATCH KITTI_IMGSZ KITTI_WORKERS
export CELEBA_BASE_MODEL CELEBA_ATTRIBUTES CELEBA_GEN_DIR CELEBA_NUM_ADDED CELEBA_EPOCHS CELEBA_BATCH CELEBA_WORKERS CELEBA_LR
export COCO_ROOT COCO_GEN_DIR COCO_BASE_MODEL COCO_NUM_ADDED COCO_EPOCHS COCO_BATCH COCO_WORKERS COCO_LR COCO_MAX_TRAIN
export SEED

if [[ "${DRY_RUN}" == "1" ]]; then
  bash "$0" --worker
  exit 0
fi

setsid nohup bash "$0" --worker > "${out_log}" 2>&1 < /dev/null &
pid=$!
printf '%s %s\n' "${pid}" "${out_log}" > "${pid_file}"
cat "${pid_file}"
