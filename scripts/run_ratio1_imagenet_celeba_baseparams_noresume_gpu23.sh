#!/usr/bin/env bash
set -euo pipefail

ROOT="/people/cs/o/oxx220000/safefix"
PYTHON_EXE="${PYTHON_EXE:-/people/cs/o/oxx220000/.conda/envs/ad/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
IMAGENET_GPU="${IMAGENET_GPU:-2}"
CELEBA_GPU="${CELEBA_GPU:-3}"

cd "${ROOT}"
mkdir -p logs logs/imagenet logs/celeba

if [[ ! -x "${PYTHON_EXE}" ]]; then
  PYTHON_EXE="python3"
fi

launch_imagenet() {
  local log="logs/imagenet/controlnet_pinkcolor_fabrictexture_resnet_imagenet_1000_ratio1p0_e30_baseparams_noresume_gpu${IMAGENET_GPU}_${STAMP}.log"
  local pidfile="logs/imagenet/controlnet_pinkcolor_fabrictexture_resnet_imagenet_1000_ratio1p0_e30_baseparams_noresume_gpu${IMAGENET_GPU}_${STAMP}.pid"

  setsid nohup bash -lc "
    set -euo pipefail
    cd '${ROOT}'
    export CUDA_VISIBLE_DEVICES='${IMAGENET_GPU}'
    '${PYTHON_EXE}' -u alpred.py \
      --dataset imagenet \
      --model resnet \
      --num_added_img 1000 \
      --attributes pinkcolor_fabrictexture \
      --gen_image_dir data/imagenet_pinkcolor_fabrictexture \
      --num_epochs 30 \
      --lr 1e-3 \
      --batch_size 64 \
      --workers 4 \
      --seed 42 \
      --gpu '${IMAGENET_GPU}' \
      --save_pred_model best \
      --save_preds True \
      --save_model False \
      --verbose False \
      --sub_ratio 1.0 \
      --use_active_learning False
  " > "${log}" 2>&1 < /dev/null &
  local pid=$!
  echo "${pid}" > "${pidfile}"
  echo "IMAGENET_LOG=${log}"
  echo "IMAGENET_PID=${pid}"
  echo "IMAGENET_PIDFILE=${pidfile}"
}

launch_celeba() {
  local log="logs/celeba/controlnet_redhair_brownskin_sademotion_resnet_celeba_1000_ratio1p0_e20_baseparams_noresume_gpu${CELEBA_GPU}_${STAMP}.log"
  local pidfile="logs/celeba/controlnet_redhair_brownskin_sademotion_resnet_celeba_1000_ratio1p0_e20_baseparams_noresume_gpu${CELEBA_GPU}_${STAMP}.pid"

  setsid nohup bash -lc "
    set -euo pipefail
    cd '${ROOT}'
    export CUDA_VISIBLE_DEVICES='${CELEBA_GPU}'
    '${PYTHON_EXE}' -u alpred.py \
      --dataset celeba \
      --model resnet \
      --attributes redhair_brownskin_sademotion \
      --gen_image_dir data/redhair_brownskin_sademotion \
      --num_added_img 1000 \
      --num_epochs 20 \
      --lr 1e-5 \
      --batch_size 64 \
      --workers 4 \
      --seed 42 \
      --gpu '${CELEBA_GPU}' \
      --save_pred_model best \
      --save_preds True \
      --save_model False \
      --verbose False \
      --sub_ratio 1.0 \
      --use_active_learning False \
      --skip_feasible True
  " > "${log}" 2>&1 < /dev/null &
  local pid=$!
  echo "${pid}" > "${pidfile}"
  echo "CELEBA_LOG=${log}"
  echo "CELEBA_PID=${pid}"
  echo "CELEBA_PIDFILE=${pidfile}"
}

echo "STAMP=${STAMP}"
launch_imagenet
launch_celeba
