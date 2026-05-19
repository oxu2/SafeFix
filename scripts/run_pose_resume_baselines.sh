#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_pose_resume_baselines.sh <strategy> [options]

Strategies:
  all
  random
  stratified
  coreset
  badge

What this script does:
  - launches COCO pose resume-from-base runs with generated rainy images
  - pins ratio 0.1 / 0.3 / 0.5 / 0.7 to GPU 0 / 1 / 2 / 3
  - in `all` mode, each GPU runs `random -> stratified -> coreset -> badge` sequentially
    for its own ratio
  - writes logs to logs/pose/
  - writes PID files to logs/pose/pids/

Important epoch semantics:
  - this script uses `pose_estimation/train_pose_estimation.py train --resume-model`
  - `--epochs-total` is the total epoch count seen by the trainer, not "extra epochs"
  - the current canonical qwen3 / internvl3 best checkpoints are saved at epoch 0
  - therefore the default `--epochs-total 2` means "run one additional epoch"

Options:
  --base-model PATH         Explicit base checkpoint to resume from.
  --base-source NAME        Auto-resolve canonical base checkpoint for `qwen3` or `internvl3`.
                            Default: qwen3
  --epochs-total N          Total epoch count passed to `pose_estimation/train_pose_estimation.py`. Default: 2
  --python PATH             Python executable. Default: /people/cs/o/oxx220000/.conda/envs/ad/bin/python
  --data-root PATH          COCO root. Default: /data/oxx220000/coco
  --assigner NAME           Analysis source for post-train slice analysis (`qwen3`, `internvl3`, `blip`).
                            Default: same as --base-source
  --generated-attribute V   Generated attribute to append. Default: rainy
  --num-added-img N         Number of generated images. Default: 1000
  --max-train-samples N     Original train cap before sub-ratio. Default: 20000
  --batch-size N            Training batch size. Default: 8
  --num-workers N           Training workers. Default: 4
  --lr V                    Learning rate for fresh optimizer under --resume-model. Default: 0.005
  --seed N                  Random seed. Default: 42
  --al-initial-labeled N    Proxy labeled count for coreset/badge. Default: 400
  --al-feature-batch-size N Feature batch size for coreset/badge. Default: 8
  --analyze-after-train B   Whether to run post-train slice analysis. Default: false
  --delay-gpu-seconds MAP   Optional comma-separated per-GPU startup delays in seconds.
                            Example: 3:4200 or 1:600,3:4200
  --detach-controller       Launch the top-level controller itself under nohup/setsid and exit.
  --dry-run                 Print commands without launching.

Examples:
  ./scripts/run_pose_resume_baselines.sh all
  ./scripts/run_pose_resume_baselines.sh stratified --dry-run
  ./scripts/run_pose_resume_baselines.sh badge --base-source internvl3
  ./scripts/run_pose_resume_baselines.sh random --base-model logs/pose/pose_keypointrcnn_train20k_qwen3_bs8_e2_rare1p1_20260415_144429/best.pth
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

STRATEGY="$1"
shift

BASE_MODEL=""
BASE_SOURCE="qwen3"
EPOCHS_TOTAL=2
PYTHON_EXE="/people/cs/o/oxx220000/.conda/envs/ad/bin/python"
DATA_ROOT="/data/oxx220000/coco"
ANALYSIS_SOURCE=""
GENERATED_ATTRIBUTE="rainy"
NUM_ADDED_IMG=1000
MAX_TRAIN_SAMPLES=20000
BATCH_SIZE=8
NUM_WORKERS=4
LR="0.005"
SEED=42
AL_INITIAL_LABELED=400
AL_FEATURE_BATCH_SIZE=8
ANALYZE_AFTER_TRAIN="false"
DELAY_GPU_SECONDS_RAW=""
DETACH_CONTROLLER=0
DRY_RUN=0

RATIOS=(0.1 0.3 0.5 0.7)
GPUS=(0 1 2 3)
ALL_STRATEGIES=(random stratified coreset badge)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-model)
      BASE_MODEL="$2"
      shift 2
      ;;
    --base-source)
      BASE_SOURCE="$2"
      shift 2
      ;;
    --epochs-total)
      EPOCHS_TOTAL="$2"
      shift 2
      ;;
    --python)
      PYTHON_EXE="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --assigner|--analysis-source)
      ANALYSIS_SOURCE="$2"
      shift 2
      ;;
    --generated-attribute)
      GENERATED_ATTRIBUTE="$2"
      shift 2
      ;;
    --num-added-img)
      NUM_ADDED_IMG="$2"
      shift 2
      ;;
    --max-train-samples)
      MAX_TRAIN_SAMPLES="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --al-initial-labeled)
      AL_INITIAL_LABELED="$2"
      shift 2
      ;;
    --al-feature-batch-size)
      AL_FEATURE_BATCH_SIZE="$2"
      shift 2
      ;;
    --analyze-after-train)
      ANALYZE_AFTER_TRAIN="$2"
      shift 2
      ;;
    --delay-gpu-seconds)
      DELAY_GPU_SECONDS_RAW="$2"
      shift 2
      ;;
    --detach-controller)
      DETACH_CONTROLLER=1
      shift
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
  all|random|stratified|coreset|badge)
    ;;
  *)
    echo "Unsupported strategy: $STRATEGY" >&2
    usage
    exit 1
    ;;
esac

case "$BASE_SOURCE" in
  qwen3|internvl3)
    ;;
  *)
    echo "Unsupported --base-source: $BASE_SOURCE (expected qwen3 or internvl3)" >&2
    exit 1
    ;;
esac

if [[ -z "$ANALYSIS_SOURCE" ]]; then
  ANALYSIS_SOURCE="$BASE_SOURCE"
fi

case "$ANALYSIS_SOURCE" in
  qwen3|internvl3|blip)
    ;;
  *)
    echo "Unsupported --assigner/--analysis-source: $ANALYSIS_SOURCE" >&2
    exit 1
    ;;
esac

if [[ ! -x "$PYTHON_EXE" ]]; then
  echo "Python executable not found: $PYTHON_EXE" >&2
  exit 1
fi

latest_matching_best() {
  local pattern="$1"
  local latest=""
  local candidate=""

  shopt -s nullglob
  for candidate in $pattern; do
    if [[ -z "$latest" || "$candidate" -nt "$latest" ]]; then
      latest="$candidate"
    fi
  done
  shopt -u nullglob

  if [[ -z "$latest" ]]; then
    return 1
  fi

  printf '%s\n' "$latest"
}

resolve_default_base_model() {
  local preferred=""
  local fallback=""

  case "$BASE_SOURCE" in
    qwen3)
      preferred="logs/pose/pose_keypointrcnn_train20k_qwen3_bs8_e2_rare1p1_20260415_144429/best.pth"
      fallback="logs/pose/pose_keypointrcnn_train20k_qwen3*/best.pth"
      ;;
    internvl3)
      preferred="logs/pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530/best.pth"
      fallback="logs/pose/pose_keypointrcnn_train20k_internvl3*/best.pth"
      ;;
  esac

  if [[ -f "$preferred" ]]; then
    printf '%s\n' "$preferred"
    return 0
  fi

  if latest_matching_best "$fallback"; then
    return 0
  fi

  return 1
}

if [[ -z "$BASE_MODEL" ]]; then
  if ! BASE_MODEL="$(resolve_default_base_model)"; then
    echo "Could not auto-discover a pose base checkpoint for source '$BASE_SOURCE'." >&2
    echo "Pass --base-model explicitly." >&2
    exit 1
  fi
fi

if [[ ! -f "$BASE_MODEL" ]]; then
  echo "Base model not found: $BASE_MODEL" >&2
  exit 1
fi

mkdir -p logs/pose/pids

if [[ "$DETACH_CONTROLLER" -eq 1 ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "--detach-controller cannot be combined with --dry-run" >&2
    exit 1
  fi

  local_ts="$(date +%Y%m%d_%H%M%S)"
  controller_log="logs/pose/run_pose_resume_${BASE_SOURCE}_${STRATEGY}_${local_ts}.log"
  controller_pid="logs/pose/pids/run_pose_resume_${BASE_SOURCE}_${STRATEGY}_${local_ts}.pid"

  detached_cmd=(
    bash "$0" "$STRATEGY"
    --base-model "$BASE_MODEL"
    --base-source "$BASE_SOURCE"
    --epochs-total "$EPOCHS_TOTAL"
    --python "$PYTHON_EXE"
    --data-root "$DATA_ROOT"
    --assigner "$ANALYSIS_SOURCE"
    --generated-attribute "$GENERATED_ATTRIBUTE"
    --num-added-img "$NUM_ADDED_IMG"
    --max-train-samples "$MAX_TRAIN_SAMPLES"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --lr "$LR"
    --seed "$SEED"
    --al-initial-labeled "$AL_INITIAL_LABELED"
    --al-feature-batch-size "$AL_FEATURE_BATCH_SIZE"
    --analyze-after-train "$ANALYZE_AFTER_TRAIN"
  )

  if [[ -n "$DELAY_GPU_SECONDS_RAW" ]]; then
    detached_cmd+=(--delay-gpu-seconds "$DELAY_GPU_SECONDS_RAW")
  fi

  setsid nohup "${detached_cmd[@]}" > "$controller_log" 2>&1 < /dev/null &
  controller_pid_value=$!
  echo "$controller_pid_value" > "$controller_pid"
  echo "CONTROLLER_LOG=$controller_log"
  echo "CONTROLLER_PIDFILE=$controller_pid"
  echo "CONTROLLER_PID=$controller_pid_value"
  exit 0
fi

declare -A GPU_START_DELAYS=()

if [[ -n "$DELAY_GPU_SECONDS_RAW" ]]; then
  IFS=',' read -r -a delay_entries <<< "$DELAY_GPU_SECONDS_RAW"
  for entry in "${delay_entries[@]}"; do
    if [[ "$entry" != *:* ]]; then
      echo "Invalid --delay-gpu-seconds entry: $entry (expected GPU:SECONDS)" >&2
      exit 1
    fi
    gpu_id="${entry%%:*}"
    delay_sec="${entry##*:}"
    if ! [[ "$gpu_id" =~ ^[0-9]+$ ]]; then
      echo "Invalid GPU id in --delay-gpu-seconds: $gpu_id" >&2
      exit 1
    fi
    if ! [[ "$delay_sec" =~ ^[0-9]+$ ]]; then
      echo "Invalid delay seconds in --delay-gpu-seconds: $delay_sec" >&2
      exit 1
    fi
    GPU_START_DELAYS["$gpu_id"]="$delay_sec"
  done
fi

build_strategy_args() {
  local strategy="$1"
  STRATEGY_ARGS=()

  case "$strategy" in
    random)
      STRATEGY_ARGS=(
        --use-active-learning false
      )
      ;;
    stratified)
      STRATEGY_ARGS=(
        --use-active-learning true
        --al-strategy stratified
      )
      ;;
    coreset)
      STRATEGY_ARGS=(
        --use-active-learning true
        --al-strategy coreset
        --al-initial-labeled "${AL_INITIAL_LABELED}"
        --al-feature-batch-size "${AL_FEATURE_BATCH_SIZE}"
      )
      ;;
    badge)
      STRATEGY_ARGS=(
        --use-active-learning true
        --al-strategy badge
        --al-initial-labeled "${AL_INITIAL_LABELED}"
        --al-feature-batch-size "${AL_FEATURE_BATCH_SIZE}"
      )
      ;;
  esac
}

launch_ratio() {
  local strategy="$1"
  local gpu="$2"
  local ratio="$3"
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"

  local logfile="logs/pose/pose_resume_${BASE_SOURCE}_epoch${EPOCHS_TOTAL}_${strategy}_ratio${ratio}_gpu${gpu}_${ts}.log"
  local pidfile="logs/pose/pids/pose_resume_${BASE_SOURCE}_epoch${EPOCHS_TOTAL}_${strategy}_ratio${ratio}_gpu${gpu}_${ts}.pid"
  local outdir="logs/pose/pose_keypointrcnn_resume_${BASE_SOURCE}_${strategy}_ratio${ratio}_gpu${gpu}_${ts}"
  local delay_seconds="${GPU_START_DELAYS[$gpu]:-0}"

  local cmd=(
    env
    MPLCONFIGDIR=/tmp/matplotlib-safefix
    CUDA_VISIBLE_DEVICES="${gpu}"
    "${PYTHON_EXE}" -u pose_estimation/train_pose_estimation.py train
    --data-root "${DATA_ROOT}"
    --pretrained true
    --device cuda
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --epochs "${EPOCHS_TOTAL}"
    --eval-every 1
    --lr "${LR}"
    --seed "${SEED}"
    --max-train-samples "${MAX_TRAIN_SAMPLES}"
    --sub-ratio "${ratio}"
    --resume-model "${BASE_MODEL}"
    --attribute "${GENERATED_ATTRIBUTE}"
    --num-added-img "${NUM_ADDED_IMG}"
    --analyze-after-train "${ANALYZE_AFTER_TRAIN}"
    --assigner "${ANALYSIS_SOURCE}"
    --output-dir "${outdir}"
    "${STRATEGY_ARGS[@]}"
  )

  echo "[launch] strategy=${strategy} gpu=${gpu} ratio=${ratio}"
  echo "         base=${BASE_MODEL}"
  echo "         log=${logfile}"
  echo "         out=${outdir}"
  if [[ "$delay_seconds" -gt 0 ]]; then
    echo "         delayed_start_sec=${delay_seconds}"
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '         cmd='
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  if [[ "$delay_seconds" -gt 0 ]]; then
    setsid nohup bash -lc "sleep ${delay_seconds}; exec $(printf '%q ' "${cmd[@]}")" > "${logfile}" 2>&1 < /dev/null &
  else
    setsid nohup "${cmd[@]}" > "${logfile}" 2>&1 < /dev/null &
  fi
  local pid=$!
  echo "${pid}" > "${pidfile}"
  echo "         pid=${pid} pidfile=${pidfile}"
  LAUNCHED_PIDS+=("${pid}")
}

launch_strategy_group() {
  local strategy="$1"

  LAUNCHED_PIDS=()
  build_strategy_args "${strategy}"

  echo "Launching pose resume sweep"
  echo "  strategy:           ${strategy}"
  echo "  base source:        ${BASE_SOURCE}"
  echo "  base model:         ${BASE_MODEL}"
  echo "  epochs total:       ${EPOCHS_TOTAL}"
  echo "  generated attr:     ${GENERATED_ATTRIBUTE}"
  echo "  num added img:      ${NUM_ADDED_IMG}"
  echo "  max train samples:  ${MAX_TRAIN_SAMPLES}"
  echo "  analysis source:    ${ANALYSIS_SOURCE}"
  echo "  analyze after run:  ${ANALYZE_AFTER_TRAIN}"

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

launch_strategy_group_single_ratio() {
  local strategy="$1"
  local gpu="$2"
  local ratio="$3"

  LAUNCHED_PIDS=()
  build_strategy_args "${strategy}"

  echo "Launching pose resume sweep"
  echo "  strategy:     ${strategy}"
  echo "  base source:  ${BASE_SOURCE}"
  echo "  base model:   ${BASE_MODEL}"
  echo "  epochs total: ${EPOCHS_TOTAL}"
  echo "  analysis src: ${ANALYSIS_SOURCE}"
  echo "  gpu:          ${gpu}"
  echo "  ratio:        ${ratio}"

  launch_ratio "${strategy}" "${gpu}" "${ratio}"
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

if [[ "${STRATEGY}" == "all" ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run for per-GPU sequential pose baseline queues."
    for idx in "${!RATIOS[@]}"; do
      run_gpu_sequence "${GPUS[$idx]}" "${RATIOS[$idx]}"
    done
  else
    echo "Launching per-GPU sequential pose baseline queues."
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
