#!/usr/bin/env bash
set -euo pipefail

ROOT="/people/cs/o/oxx220000/safefix"
PYTHON_EXE="${PYTHON_EXE:-/people/cs/o/oxx220000/.conda/envs/ad/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RATIO="1.0"
RATIO_TAG="1p0"
export STAMP

CELEBA_BASE_MODEL="weights/model_celeba_resnet_base.pth"
IMAGENET_BASE_MODEL="weights/model_imagenet_resnet_base.pth"
COCO_POSE_BASE_MODEL="log_pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530/best.pth"
COCO_ROOT="/data/oxx220000/coco"
COCO_GEN_DIR="/data/oxx220000/coco-pose-weather-edits/generated_coco_qwen3_1500_brown_rain_icedit"

cd "${ROOT}"
mkdir -p logs logs_celeba logs_imagenet log_pose

if [[ ! -x "${PYTHON_EXE}" ]]; then
  PYTHON_EXE="python3"
fi

run_celeba_full() {
  local gpu="$1"
  local label="$2"
  local attrs="$3"
  local gen_dir="$4"
  local log="logs_celeba/controlnet_${label}_resnet_celeba_1000_ratio${RATIO_TAG}_gpu${gpu}_${STAMP}.log"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] celeba gpu=${gpu} label=${label} attrs=${attrs} log=${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_EXE}" -u alpred.py \
    --dataset celeba \
    --model resnet \
    --attributes "${attrs}" \
    --gen_image_dir "${gen_dir}" \
    --num_added_img 1000 \
    --num_epochs 15 \
    --resume_model "${CELEBA_BASE_MODEL}" \
    --freeze_backbone False \
    --lr 1e-5 \
    --batch_size 64 \
    --workers 4 \
    --seed 42 \
    --gpu "${gpu}" \
    --save_pred_model best \
    --save_preds True \
    --save_model False \
    --verbose False \
    --sub_ratio "${RATIO}" \
    --use_active_learning False \
    --skip_feasible True \
    > "${log}" 2>&1
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] celeba finished label=${label}; log=${log}"
}

run_imagenet_full() {
  local gpu="3"
  local log="logs_imagenet/controlnet_pinkcolor_fabrictexture_resnet_imagenet_1000_ratio${RATIO_TAG}_gpu${gpu}_${STAMP}.log"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] imagenet gpu=${gpu} log=${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_EXE}" -u alpred.py \
    --dataset imagenet \
    --model resnet \
    --num_added_img 1000 \
    --attributes pinkcolor_fabrictexture \
    --gen_image_dir data/imagenet_pinkcolor_fabrictexture \
    --num_epochs 15 \
    --lr 1e-5 \
    --verbose False \
    --resume_model "${IMAGENET_BASE_MODEL}" \
    --sub_ratio "${RATIO}" \
    --use_active_learning False \
    > "${log}" 2>&1
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] imagenet finished; log=${log}"
}

run_coco_pose_full() {
  local gpu="3"
  local out_dir="log_pose/pose_keypointrcnn_resume_qwen3_brown_rainy_full_ratio${RATIO_TAG}_gpu${gpu}_${STAMP}"
  local log="log_pose/pose_qwen3_brown_rainy_full_ratio${RATIO_TAG}_gpu${gpu}_${STAMP}.log"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] coco-pose gpu=${gpu} out_dir=${out_dir} log=${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_EXE}" -u train_pose_estimation.py train \
    --data-root "${COCO_ROOT}" \
    --device cuda \
    --epochs 2 \
    --batch-size 8 \
    --num-workers 4 \
    --eval-every 1 \
    --lr 0.005 \
    --seed 42 \
    --max-train-samples 20000 \
    --sub-ratio "${RATIO}" \
    --use-active-learning False \
    --resume-model "${COCO_POSE_BASE_MODEL}" \
    --attribute brown_rainy \
    --generated-image-dir "${COCO_GEN_DIR}" \
    --num-added-img 1000 \
    --assigner qwen3 \
    --analyze-after-train false \
    --output-dir "${out_dir}" \
    > "${log}" 2>&1
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] coco-pose finished; log=${log}"
}

run_gpu3_queue() {
  run_imagenet_full
  run_coco_pose_full
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  return 0
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching ratio=1.0 full ControlNet queues, stamp=${STAMP}"

setsid nohup bash -c 'source "$1"; run_celeba_full 0 redhair_brownskin_sademotion redhair_brownskin_sademotion data/redhair_brownskin_sademotion' _ "$0" > "logs/controlnet_ratio${RATIO_TAG}_celeba_gpu0_${STAMP}.out" 2>&1 < /dev/null &
pid0=$!
setsid nohup bash -c 'source "$1"; run_celeba_full 1 redhair_brownskin_unfiltered redhair_brownskin data/redhair_brownskin_unfiltered' _ "$0" > "logs/controlnet_ratio${RATIO_TAG}_celeba_gpu1_${STAMP}.out" 2>&1 < /dev/null &
pid1=$!
setsid nohup bash -c 'source "$1"; run_celeba_full 2 redhair_brownskin redhair_brownskin data/redhair_brownskin' _ "$0" > "logs/controlnet_ratio${RATIO_TAG}_celeba_gpu2_${STAMP}.out" 2>&1 < /dev/null &
pid2=$!
setsid nohup bash -c 'source "$1"; run_gpu3_queue' _ "$0" > "logs/controlnet_ratio${RATIO_TAG}_imagenet_then_coco_pose_gpu3_${STAMP}.out" 2>&1 < /dev/null &
pid3=$!

pid_file="logs/controlnet_ratio${RATIO_TAG}_gpu0123_${STAMP}.pids"
cat > "${pid_file}" <<EOF
gpu0 ${pid0} logs/controlnet_ratio${RATIO_TAG}_celeba_gpu0_${STAMP}.out logs_celeba/controlnet_redhair_brownskin_sademotion_resnet_celeba_1000_ratio${RATIO_TAG}_gpu0_${STAMP}.log
gpu1 ${pid1} logs/controlnet_ratio${RATIO_TAG}_celeba_gpu1_${STAMP}.out logs_celeba/controlnet_redhair_brownskin_unfiltered_resnet_celeba_1000_ratio${RATIO_TAG}_gpu1_${STAMP}.log
gpu2 ${pid2} logs/controlnet_ratio${RATIO_TAG}_celeba_gpu2_${STAMP}.out logs_celeba/controlnet_redhair_brownskin_resnet_celeba_1000_ratio${RATIO_TAG}_gpu2_${STAMP}.log
gpu3 ${pid3} logs/controlnet_ratio${RATIO_TAG}_imagenet_then_coco_pose_gpu3_${STAMP}.out logs_imagenet/controlnet_pinkcolor_fabrictexture_resnet_imagenet_1000_ratio${RATIO_TAG}_gpu3_${STAMP}.log then log_pose/pose_qwen3_brown_rainy_full_ratio${RATIO_TAG}_gpu3_${STAMP}.log
EOF

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Queues launched:"
cat "${pid_file}"
