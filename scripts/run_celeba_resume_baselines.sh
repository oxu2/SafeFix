#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_celeba_resume_baselines.sh <strategy> [--base-model PATH] [--num-epochs N] [--python PATH] [--dry-run]

Strategies:
  all
  random
  stratified
  coreset
  badge
  hierarchical

What this script does:
  - launches CelebA ResNet resume-from-base runs with 1000 generated images
  - pins ratio 0.1 / 0.3 / 0.5 / 0.7 to GPU 0 / 1 / 2 / 3
  - in `all` mode, each GPU runs `random -> stratified -> coreset -> badge -> hierarchical` sequentially for its own ratio
  - writes logs to logs/celeba/
  - writes PID files to logs/celeba/pids/

Examples:
  ./scripts/run_celeba_resume_baselines.sh all
  ./scripts/run_celeba_resume_baselines.sh random
  ./scripts/run_celeba_resume_baselines.sh badge --base-model save/models/model_celeba_resnet_redhair_brownskin_sademotion_0imgs_0416024000.pth
  ./scripts/run_celeba_resume_baselines.sh coreset --dry-run
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

STRATEGY="$1"
shift

BASE_MODEL=""
NUM_EPOCHS=15
PYTHON_EXE="/people/cs/o/oxx220000/.conda/envs/ad/bin/python"
DRY_RUN=0

ATTRIBUTES="redhair_brownskin_sademotion"
GEN_IMAGE_DIR="data/${ATTRIBUTES}"
NUM_ADDED_IMG=1000
BATCH_SIZE=64
WORKERS=4
SEED=42
LR="1e-5"

RATIOS=(0.1 0.3 0.5 0.7)
GPUS=(0 1 2 3)
ALL_STRATEGIES=(random stratified coreset badge hierarchical)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-model)
      BASE_MODEL="$2"
      shift 2
      ;;
    --num-epochs)
      NUM_EPOCHS="$2"
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

case "$STRATEGY" in
  all|random|stratified|coreset|badge|hierarchical)
    ;;
  *)
    echo "Unsupported strategy: $STRATEGY" >&2
    usage
    exit 1
    ;;
esac

if [[ ! -x "$PYTHON_EXE" ]]; then
  echo "Python executable not found: $PYTHON_EXE" >&2
  exit 1
fi

resolve_latest_base_model() {
  local latest=""
  local candidate=""

  shopt -s nullglob
  for candidate in save/models/model_celeba_resnet_${ATTRIBUTES}_0imgs_*.pth; do
    if [[ -z "$latest" || "$candidate" -nt "$latest" ]]; then
      latest="$candidate"
    fi
  done
  shopt -u nullglob

  if [[ -z "$latest" ]]; then
    echo "" >&2
    return 1
  fi

  printf '%s\n' "$latest"
}

if [[ -z "$BASE_MODEL" ]]; then
  if ! BASE_MODEL="$(resolve_latest_base_model)"; then
    echo "Could not auto-discover a CelebA base checkpoint." >&2
    echo "Expected something like save/models/model_celeba_resnet_${ATTRIBUTES}_0imgs_*.pth" >&2
    exit 1
  fi
fi

if [[ ! -f "$BASE_MODEL" ]]; then
  echo "Base model not found: $BASE_MODEL" >&2
  exit 1
fi

mkdir -p logs/celeba/pids

build_strategy_args() {
  local strategy="$1"
  STRATEGY_ARGS=()

  case "$strategy" in
    random)
      STRATEGY_ARGS=(
        --use_active_learning False
      )
      ;;
    stratified)
      STRATEGY_ARGS=(
        --use_active_learning True
        --al_strategy stratified
      )
      ;;
    coreset)
      STRATEGY_ARGS=(
        --use_active_learning True
        --al_strategy coreset
        --al_embedding_dim 512
        --al_initial_labeled 1000
      )
      ;;
    badge)
      STRATEGY_ARGS=(
        --use_active_learning True
        --al_strategy badge
        --al_embedding_dim 512
      )
      ;;
    hierarchical)
      STRATEGY_ARGS=(
        --use_active_learning True
        --al_strategy hierarchical
        --al_embedding_dim 512
        --al_round1_ratio 0.8
        --al_alpha 0.5
        --al_beta 0.35
        --al_gamma 0.15
        --al_lambda_div 0.3
        --al_min_slice_quota 20
        --al_max_slice_quota 500
      )
      ;;
  esac
}

launch_ratio() {
  local strategy="$1"
  local gpu="$2"
  local ratio="$3"

  local logfile="logs/celeba/resume_lr1e-5_epoch${NUM_EPOCHS}_${strategy}_resnet_celeba_1000_ratio${ratio}.log"
  local pidfile="logs/celeba/pids/resume_lr1e-5_epoch${NUM_EPOCHS}_${strategy}_ratio${ratio}.pid"

  local cmd=(
    env
    MPLCONFIGDIR=/tmp/matplotlib-safefix
    CUDA_VISIBLE_DEVICES="${gpu}"
    "${PYTHON_EXE}" -u alpred.py
    --dataset celeba
    --model resnet
    --attributes "${ATTRIBUTES}"
    --gen_image_dir "${GEN_IMAGE_DIR}"
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
    --sub_ratio "${ratio}"
    "${STRATEGY_ARGS[@]}"
  )

  echo "[launch] strategy=${strategy} gpu=${gpu} ratio=${ratio}"
  echo "         log=${logfile}"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '         cmd='
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  setsid nohup "${cmd[@]}" > "${logfile}" 2>&1 < /dev/null &
  local pid=$!
  echo "${pid}" > "${pidfile}"
  echo "         pid=${pid} pidfile=${pidfile}"
  LAUNCHED_PIDS+=("${pid}")
}

launch_strategy_group() {
  local strategy="$1"

  LAUNCHED_PIDS=()
  build_strategy_args "${strategy}"

  echo "Launching CelebA resume sweep"
  echo "  strategy:   ${strategy}"
  echo "  base model: ${BASE_MODEL}"
  echo "  epochs:     ${NUM_EPOCHS}"
  echo "  attrs:      ${ATTRIBUTES}"

  for idx in "${!RATIOS[@]}"; do
    launch_ratio "${strategy}" "${GPUS[$idx]}" "${RATIOS[$idx]}"
  done
}

wait_for_strategy_group() {
  local strategy="$1"
  local failed=0

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run for ${strategy} complete."
    return 0
  fi

  echo "[wait] strategy=${strategy} waiting for ${#LAUNCHED_PIDS[@]} launched jobs"
  for pid in "${LAUNCHED_PIDS[@]}"; do
    if wait "${pid}"; then
      echo "[wait] strategy=${strategy} pid=${pid} completed successfully"
    else
      local status=$?
      echo "[wait] strategy=${strategy} pid=${pid} exited with status ${status}" >&2
      failed=1
    fi
  done

  if [[ "${failed}" -ne 0 ]]; then
    echo "Strategy ${strategy} completed with failures." >&2
    return 1
  fi

  echo "Strategy ${strategy} completed successfully."
}

run_gpu_sequence() {
  local gpu="$1"
  local ratio="$2"

  echo "[pipeline] gpu=${gpu} ratio=${ratio} starting strategy sequence"

  for strategy_name in "${ALL_STRATEGIES[@]}"; do
    launch_strategy_group_single_ratio "${strategy_name}" "${gpu}" "${ratio}"
    wait_for_strategy_group "${strategy_name}"
  done

  echo "[pipeline] gpu=${gpu} ratio=${ratio} finished strategy sequence"
}

launch_strategy_group_single_ratio() {
  local strategy="$1"
  local gpu="$2"
  local ratio="$3"

  LAUNCHED_PIDS=()
  build_strategy_args "${strategy}"

  echo "Launching CelebA resume sweep"
  echo "  strategy:   ${strategy}"
  echo "  base model: ${BASE_MODEL}"
  echo "  epochs:     ${NUM_EPOCHS}"
  echo "  attrs:      ${ATTRIBUTES}"
  echo "  gpu:        ${gpu}"
  echo "  ratio:      ${ratio}"

  launch_ratio "${strategy}" "${gpu}" "${ratio}"
}

if [[ "${STRATEGY}" == "all" ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run for per-GPU sequential baseline queues."
    for idx in "${!RATIOS[@]}"; do
      run_gpu_sequence "${GPUS[$idx]}" "${RATIOS[$idx]}"
    done
  else
    echo "Launching per-GPU sequential baseline queues."
    CONTROLLER_PIDS=()
    for idx in "${!RATIOS[@]}"; do
      run_gpu_sequence "${GPUS[$idx]}" "${RATIOS[$idx]}" &
      CONTROLLER_PIDS+=("$!")
    done

    for controller_pid in "${CONTROLLER_PIDS[@]}"; do
      wait "${controller_pid}"
    done
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run complete for all strategies."
  else
    echo "All strategies completed."
  fi
else
  launch_strategy_group "${STRATEGY}"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run complete."
  else
    echo "All ${STRATEGY} launches submitted."
  fi
fi
