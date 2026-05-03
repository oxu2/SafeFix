#!/usr/bin/env bash
set -euo pipefail

ROOT="/people/cs/o/oxx220000/safefix"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RATIOS="${RATIOS:-0.1 0.3 0.5 0.7 0.9}"
DRY_RUN="${DRY_RUN:-0}"

FP_PYTHON="${FP_PYTHON:-/people/cs/o/oxx220000/.conda/envs/fp/bin/python}"
AD_PYTHON="${AD_PYTHON:-/people/cs/o/oxx220000/.conda/envs/ad/bin/python}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_safefix}"

KITTI_BASE_ROOT="${KITTI_BASE_ROOT:-/data/oxx220000/kitti_yolov8_first5000}"
KITTI_GEN_DIR="${KITTI_GEN_DIR:-/data/oxx220000/kitti-weather-edits/generated_kitti1352_red_cloudy_icedit_upscaled}"
KITTI_PROJECT="${KITTI_PROJECT:-${ROOT}/runs/kitti_yolov8}"
KITTI_NUM_ADDED="${KITTI_NUM_ADDED:-1000}"
KITTI_EPOCHS="${KITTI_EPOCHS:-10}"
KITTI_BATCH="${KITTI_BATCH:-16}"
KITTI_IMGSZ="${KITTI_IMGSZ:-640}"
KITTI_WORKERS="${KITTI_WORKERS:-4}"

IMAGENET_BASE_MODEL="${IMAGENET_BASE_MODEL:-weights/model_imagenet_resnet_base.pth}"
IMAGENET_GEN_DIR="${IMAGENET_GEN_DIR:-data/imagenet_pinkcolor_fabrictexture}"
IMAGENET_DATA_DIR="${IMAGENET_DATA_DIR:-data/imagenet_subset}"
IMAGENET_NUM_ADDED="${IMAGENET_NUM_ADDED:-1000}"
IMAGENET_EPOCHS="${IMAGENET_EPOCHS:-15}"
IMAGENET_BATCH="${IMAGENET_BATCH:-64}"
IMAGENET_WORKERS="${IMAGENET_WORKERS:-4}"
IMAGENET_LR="${IMAGENET_LR:-1e-5}"

COCO_ROOT="${COCO_ROOT:-/data/oxx220000/coco}"
COCO_GEN_DIR="${COCO_GEN_DIR:-/data/oxx220000/coco-pose-weather-edits/generated_coco_qwen3_1500_brown_rain_icedit}"
COCO_BASE_MODEL="${COCO_BASE_MODEL:-log_pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530/best.pth}"
COCO_NUM_ADDED="${COCO_NUM_ADDED:-1000}"
COCO_EPOCHS="${COCO_EPOCHS:-2}"
COCO_BATCH="${COCO_BATCH:-8}"
COCO_WORKERS="${COCO_WORKERS:-4}"
COCO_LR="${COCO_LR:-0.005}"
COCO_MAX_TRAIN="${COCO_MAX_TRAIN:-20000}"

PPAL_POOL_EXPAND_RATIO="${PPAL_POOL_EXPAND_RATIO:-4.0}"
PPAL_CLASS_WEIGHT_UB="${PPAL_CLASS_WEIGHT_UB:-0.2}"
PPAL_CLASS_WEIGHT_ALPHA="${PPAL_CLASS_WEIGHT_ALPHA:-0.3}"
PPAL_DIVERSITY_POWER="${PPAL_DIVERSITY_POWER:-1.0}"
SEED="${SEED:-42}"

cd "${ROOT}"
mkdir -p logs logs_imagenet log_pose

ratio_tag() {
  local ratio="$1"
  echo "${ratio//./p}"
}

print_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

run_imagenet_one() {
  local gpu="$1"
  local ratio="$2"
  local rtag
  rtag="$(ratio_tag "${ratio}")"
  local log_file="logs_imagenet/ppal_resnet_imagenet_${IMAGENET_NUM_ADDED}_ratio${rtag}_gpu${gpu}_${STAMP}.log"

  local -a args=(
    "${AD_PYTHON}" -u alpred.py
    --dataset imagenet
    --model resnet
    --num_added_img "${IMAGENET_NUM_ADDED}"
    --attributes pinkcolor_fabrictexture
    --gen_image_dir "${IMAGENET_GEN_DIR}"
    --data_dir "${IMAGENET_DATA_DIR}"
    --num_epochs "${IMAGENET_EPOCHS}"
    --lr "${IMAGENET_LR}"
    --verbose False
    --resume_model "${IMAGENET_BASE_MODEL}"
    --sub_ratio "${ratio}"
    --use_active_learning True
    --al_strategy ppal
    --al_initial_labeled 400
    --al_ppal_pool_expand_ratio "${PPAL_POOL_EXPAND_RATIO}"
    --al_ppal_class_weight_ub "${PPAL_CLASS_WEIGHT_UB}"
    --al_ppal_class_weight_alpha "${PPAL_CLASS_WEIGHT_ALPHA}"
    --al_ppal_diversity_power "${PPAL_DIVERSITY_POWER}"
    --batch_size "${IMAGENET_BATCH}"
    --workers "${IMAGENET_WORKERS}"
    --seed "${SEED}"
    --gpu "${gpu}"
    --skip_feasible True
  )

  echo "[$(date '+%F %T')] START dataset=imagenet ratio=${ratio} gpu=${gpu} log=${log_file}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q MPLCONFIGDIR=%q ' "${gpu}" "${MPLCONFIGDIR}"
    print_cmd "${args[@]}"
  else
    CUDA_VISIBLE_DEVICES="${gpu}" MPLCONFIGDIR="${MPLCONFIGDIR}" "${args[@]}" > "${log_file}" 2>&1
  fi
  echo "[$(date '+%F %T')] END dataset=imagenet ratio=${ratio} gpu=${gpu}"
}

run_kitti_one() {
  local gpu="$1"
  local ratio="$2"
  local rtag
  rtag="$(ratio_tag "${ratio}")"
  local prepared_root="/data/oxx220000/kitti_yolov8_ratio${rtag}_ppal_fast_red_cloudy_gen${KITTI_NUM_ADDED}_seed${SEED}_${STAMP}"
  local run_name="kitti_ratio${rtag}_ppal_fast_red_cloudy_gen${KITTI_NUM_ADDED}_e${KITTI_EPOCHS}_gpu${gpu}_${STAMP}"
  local log_file="logs/${run_name}.log"

  local -a args=(
    "${FP_PYTHON}" -u object_detection/train_yolov8_kitti_generated.py
    --base-data-root "${KITTI_BASE_ROOT}"
    --generated-image-dir "${KITTI_GEN_DIR}"
    --sub-ratio "${ratio}"
    --selection-strategy ppal
    --num-added-img "${KITTI_NUM_ADDED}"
    --prepared-root "${prepared_root}"
    --overwrite-prepared true
    --epochs "${KITTI_EPOCHS}"
    --batch "${KITTI_BATCH}"
    --imgsz "${KITTI_IMGSZ}"
    --device 0
    --workers "${KITTI_WORKERS}"
    --project "${KITTI_PROJECT}"
    --name "${run_name}"
    --seed "${SEED}"
    --eval-test true
    --log-file "${log_file}"
    --al-fast-selection true
    --al-initial-labeled 400
    --al-embedding-thumbnail-size 8
    --al-ppal-pool-expand-ratio "${PPAL_POOL_EXPAND_RATIO}"
    --al-ppal-class-weight-ub "${PPAL_CLASS_WEIGHT_UB}"
    --al-ppal-class-weight-alpha "${PPAL_CLASS_WEIGHT_ALPHA}"
    --al-ppal-diversity-power "${PPAL_DIVERSITY_POWER}"
  )

  echo "[$(date '+%F %T')] START dataset=kitti ratio=${ratio} gpu=${gpu} log=${log_file}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q MPLCONFIGDIR=%q ' "${gpu}" "${MPLCONFIGDIR}"
    print_cmd "${args[@]}"
  else
    CUDA_VISIBLE_DEVICES="${gpu}" MPLCONFIGDIR="${MPLCONFIGDIR}" "${args[@]}"
  fi
  echo "[$(date '+%F %T')] END dataset=kitti ratio=${ratio} gpu=${gpu}"
}

run_coco_one() {
  local gpu="$1"
  local ratio="$2"
  local rtag
  rtag="$(ratio_tag "${ratio}")"
  local out_dir="log_pose/pose_keypointrcnn_resume_qwen3_brown_rainy_ppal_ratio${rtag}_gpu${gpu}_${STAMP}"
  local log_file="log_pose/pose_qwen3_brown_rainy_ppal_ratio${rtag}_gpu${gpu}_${STAMP}.log"

  local -a args=(
    "${AD_PYTHON}" -u train_pose_estimation.py train
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
    --al-strategy ppal
    --al-initial-labeled 400
    --al-feature-batch-size 16
    --al-ppal-pool-expand-ratio "${PPAL_POOL_EXPAND_RATIO}"
    --al-ppal-class-weight-ub "${PPAL_CLASS_WEIGHT_UB}"
    --al-ppal-class-weight-alpha "${PPAL_CLASS_WEIGHT_ALPHA}"
    --al-ppal-diversity-power "${PPAL_DIVERSITY_POWER}"
  )

  echo "[$(date '+%F %T')] START dataset=coco ratio=${ratio} gpu=${gpu} log=${log_file}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q MPLCONFIGDIR=%q ' "${gpu}" "${MPLCONFIGDIR}"
    print_cmd "${args[@]}"
  else
    CUDA_VISIBLE_DEVICES="${gpu}" MPLCONFIGDIR="${MPLCONFIGDIR}" "${args[@]}" > "${log_file}" 2>&1
  fi
  echo "[$(date '+%F %T')] END dataset=coco ratio=${ratio} gpu=${gpu}"
}

run_gpu1_worker() {
  local ratio
  for ratio in ${RATIOS}; do
    run_imagenet_one 1 "${ratio}"
  done
  for ratio in ${RATIOS}; do
    run_kitti_one 1 "${ratio}"
  done
}

run_gpu2_worker() {
  local ratio
  for ratio in ${RATIOS}; do
    run_coco_one 2 "${ratio}"
  done
}

case "${1:-launch}" in
  gpu1-worker)
    run_gpu1_worker
    ;;
  gpu2-worker)
    run_gpu2_worker
    ;;
  launch)
    gpu1_out="logs/ppal_gpu1_imagenet_then_kitti_${STAMP}.out"
    gpu2_out="log_pose/ppal_gpu2_coco_${STAMP}.out"
    pid_file="logs/ppal_baselines_gpu12_${STAMP}.pids"

    echo "Launching PPAL baselines, stamp=${STAMP}"
    echo "GPU1: ImageNet ratios then KITTI ratios -> ${gpu1_out}"
    echo "GPU2: COCO ratios -> ${gpu2_out}"

    if [[ "${DRY_RUN}" == "1" ]]; then
      "$0" gpu1-worker
      "$0" gpu2-worker
      exit 0
    fi

    setsid nohup env STAMP="${STAMP}" RATIOS="${RATIOS}" bash "$0" gpu1-worker > "${gpu1_out}" 2>&1 < /dev/null &
    gpu1_pid=$!
    setsid nohup env STAMP="${STAMP}" RATIOS="${RATIOS}" bash "$0" gpu2-worker > "${gpu2_out}" 2>&1 < /dev/null &
    gpu2_pid=$!

    {
      echo "stamp=${STAMP}"
      echo "gpu1_pid=${gpu1_pid}"
      echo "gpu1_out=${gpu1_out}"
      echo "gpu2_pid=${gpu2_pid}"
      echo "gpu2_out=${gpu2_out}"
    } > "${pid_file}"
    cat "${pid_file}"
    ;;
  *)
    echo "Unknown mode: ${1:-}" >&2
    exit 2
    ;;
esac
