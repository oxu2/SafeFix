#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_celeba_attribute_ratio0.1_rpc.sh [--base-model PATH] [--python PATH] [--dry-run]

Launches three CelebA resume-from-base attribute-AL runs at ratio=0.1 on GPUs 0/1/2.
The sweep keeps entropy disabled (E=False) and varies only R/P/C:
  - GPU 0: baseline_def_e0_r001_p5_cinf
  - GPU 1: soft_e0_r01_p5_c15
  - GPU 2: aggro_e0_r08_p10_c4
EOF
}

PYTHON_EXE="/people/cs/o/oxx220000/.conda/envs/ad/bin/python"
BASE_MODEL="weights/model_celeba_resnet_base.pth"
DRY_RUN=0

ATTRIBUTES="redhair_brownskin_sademotion"
FOCUS="hair color,skin,emotion"
EXTRA_COUNTS="hair color:red=1000,skin:brown=1000,emotion:sad=1000"
RATIO="0.1"
NUM_EPOCHS=15
NUM_ADDED_IMG=1000
LR="1e-5"
BATCH_SIZE=64
WORKERS=4
SEED=42

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-model)
      BASE_MODEL="$2"
      shift 2
      ;;
    --python)
      PYTHON_EXE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -x "$PYTHON_EXE" ]]; then
  echo "Python executable not found: $PYTHON_EXE" >&2
  exit 1
fi

if [[ ! -f "$BASE_MODEL" ]]; then
  echo "Base model not found: $BASE_MODEL" >&2
  exit 1
fi

mkdir -p logs_celeba/pids

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

launch_one() {
  local gpu="$1"
  local name="$2"
  local r="$3"
  local p="$4"
  local c="$5"

  local log_file="logs_celeba/attr_ratio${RATIO}_${name}_gpu${gpu}_${TIMESTAMP}.log"
  local pid_file="logs_celeba/pids/attr_ratio${RATIO}_${name}_gpu${gpu}_${TIMESTAMP}.pid"

  local cmd=(
    env
    MPLCONFIGDIR=/tmp/matplotlib-safefix
    CUDA_VISIBLE_DEVICES="${gpu}"
    "${PYTHON_EXE}" -u alpred.py
    --dataset celeba
    --model resnet
    --attributes "${ATTRIBUTES}"
    --gen_image_dir "data/${ATTRIBUTES}"
    --num_added_img "${NUM_ADDED_IMG}"
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
    --sub_ratio "${RATIO}"
    --use_active_learning True
    --al_strategy attribute
    --attribute_file save/attribute.json
    --al_attribute_focus "${FOCUS}"
    --al_attribute_rarity_power "${r}"
    --al_attribute_log_acc_scale 0.0
    --al_attribute_extra_counts "${EXTRA_COUNTS}"
    --al_attribute_frequency_penalty "${p}"
    --al_attribute_class_balance "${c}"
    --al_attribute_use_entropy False
  )

  echo "[launch] gpu=${gpu} name=${name} R=${r} P=${p} C=${c}"
  echo "         log=${log_file}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '         cmd='
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  (
    setsid nohup "${cmd[@]}" > "${log_file}" 2>&1 < /dev/null &
    echo $! > "${pid_file}"
    echo "PID=$(cat "${pid_file}")"
    echo "LOG=${log_file}"
    echo "PIDFILE=${pid_file}"
  )
}

launch_one 0 baseline_def_e0_r001_p5_cinf 0.01 5.0 inf
launch_one 1 soft_e0_r01_p5_c15 0.1 5.0 15.0
launch_one 2 aggro_e0_r08_p10_c4 0.8 10.0 4.0
