#!/usr/bin/env bash
#SBATCH --job-name=br-base-ratios
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch scripts/juno_pose_brown_rainy_baselines_ratio_sweep.sbatch [options]

Runs Qwen3 brown-rainy COCO pose baselines in ratio order:
  0.05 -> 0.1 -> 0.3 -> 0.5 -> 0.7 -> 0.9

For each ratio, four methods start concurrently on four allocated GPUs:
  random, stratified, coreset, badge

Options:
  --repo-dir PATH          Repo path on Juno. Default: $PWD or SLURM_SUBMIT_DIR
  --conda-env PATH         Conda environment prefix. Default: /groups/yguo/oxx220000/conda/envs/ad
  --data-root PATH         COCO root. Default: ~/scratch/safefix_cache/data/coco
  --generated-dir PATH     Brown-rainy generated image dir.
  --base-model PATH        Resume checkpoint path.
  --batch-size N           Training batch size. Default: 8
  --num-workers N          DataLoader workers per run. Default: 4
  --epochs-total N         Total epochs passed to pose_estimation/train_pose_estimation.py. Default: 2
  --dry-run                Print per-run commands without training.
EOF
}

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
CONDA_ENV="/groups/yguo/oxx220000/conda/envs/ad"
DATA_ROOT="$HOME/scratch/safefix_cache/data/coco"
GENERATED_DIR="$HOME/scratch/safefix_cache/data/coco-pose-weather-edits/generated_coco_qwen3_1500_brown_rain_icedit"
BASE_MODEL="$HOME/scratch/safefix_cache/logs/pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530/best.pth"
BATCH_SIZE=8
NUM_WORKERS=4
EPOCHS_TOTAL=2
LR="0.005"
SEED=42
NUM_ADDED_IMG=1000
MAX_TRAIN_SAMPLES=20000
AL_INITIAL_LABELED=400
AL_FEATURE_BATCH_SIZE=8
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --generated-dir)
      GENERATED_DIR="$2"
      shift 2
      ;;
    --base-model)
      BASE_MODEL="$2"
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
    --epochs-total)
      EPOCHS_TOTAL="$2"
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

cd "$REPO_DIR"

for required in \
  "$CONDA_ENV/bin/python" \
  "$DATA_ROOT" \
  "$GENERATED_DIR" \
  "$BASE_MODEL"; do
  if [[ ! -e "$required" ]]; then
    echo "Required path not found: $required" >&2
    exit 1
  fi
done

mkdir -p "$REPO_DIR/logs/pose/pids"

PYTHON_EXE="$CONDA_ENV/bin/python"
RATIOS=(0.05 0.1 0.3 0.5 0.7 0.9)
STRATEGIES=(random stratified coreset badge)
GPUS=(0 1 2 3)

build_strategy_args() {
  local strategy="$1"
  STRATEGY_ARGS=()

  case "$strategy" in
    random)
      STRATEGY_ARGS=(--use-active-learning false)
      ;;
    stratified)
      STRATEGY_ARGS=(--use-active-learning true --al-strategy stratified)
      ;;
    coreset)
      STRATEGY_ARGS=(
        --use-active-learning true
        --al-strategy coreset
        --al-initial-labeled "$AL_INITIAL_LABELED"
        --al-feature-batch-size "$AL_FEATURE_BATCH_SIZE"
      )
      ;;
    badge)
      STRATEGY_ARGS=(
        --use-active-learning true
        --al-strategy badge
        --al-initial-labeled "$AL_INITIAL_LABELED"
        --al-feature-batch-size "$AL_FEATURE_BATCH_SIZE"
      )
      ;;
    *)
      echo "Unsupported strategy: $strategy" >&2
      exit 1
      ;;
  esac
}

launch_one() {
  local ratio="$1"
  local strategy="$2"
  local gpu="$3"
  local ts="$4"

  build_strategy_args "$strategy"

  local ratio_label="${ratio//./p}"
  local out_dir="$REPO_DIR/logs/pose/pose_keypointrcnn_resume_qwen3_brown_rainy_${strategy}_ratio${ratio_label}_gpu${gpu}_${ts}"
  local log_file="$REPO_DIR/logs/pose/pose_resume_qwen3_brown_rainy_${strategy}_ratio${ratio_label}_gpu${gpu}_${ts}.log"
  local pid_file="$REPO_DIR/logs/pose/pids/pose_resume_qwen3_brown_rainy_${strategy}_ratio${ratio_label}_gpu${gpu}_${ts}.pid"

  local cmd=(
    env
    MPLCONFIGDIR=/tmp/matplotlib-safefix
    CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON_EXE" -u pose_estimation/train_pose_estimation.py train
    --data-root "$DATA_ROOT"
    --pretrained true
    --device cuda
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --epochs "$EPOCHS_TOTAL"
    --eval-every 1
    --lr "$LR"
    --seed "$SEED"
    --max-train-samples "$MAX_TRAIN_SAMPLES"
    --sub-ratio "$ratio"
    --resume-model "$BASE_MODEL"
    --attribute brown_rainy
    --generated-image-dir "$GENERATED_DIR"
    --num-added-img "$NUM_ADDED_IMG"
    --analyze-after-train false
    --assigner qwen3
    --output-dir "$out_dir"
    "${STRATEGY_ARGS[@]}"
  )

  echo "[launch] ratio=${ratio} strategy=${strategy} gpu=${gpu}"
  echo "         log=${log_file}"
  echo "         out=${out_dir}"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '         cmd='
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  "${cmd[@]}" > "$log_file" 2>&1 &
  local pid=$!
  echo "$pid" > "$pid_file"
  echo "         pid=${pid} pidfile=${pid_file}"
  RUN_PIDS+=("$pid")
}

wait_for_ratio() {
  local ratio="$1"
  local failed=0

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] ratio=${ratio} complete"
    return 0
  fi

  echo "[wait] ratio=${ratio} waiting for ${#RUN_PIDS[@]} methods"
  for pid in "${RUN_PIDS[@]}"; do
    if wait "$pid"; then
      echo "[wait] ratio=${ratio} pid=${pid} completed"
    else
      local status=$?
      echo "[wait] ratio=${ratio} pid=${pid} failed with status ${status}" >&2
      failed=1
    fi
  done

  if [[ "$failed" -ne 0 ]]; then
    echo "Ratio ${ratio} had failed methods." >&2
    return 1
  fi

  echo "[done] ratio=${ratio} all methods completed"
}

echo "Running brown-rainy baseline ratio sweep on host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual}"
echo "repo_dir=$REPO_DIR"
echo "data_root=$DATA_ROOT"
echo "generated_dir=$GENERATED_DIR"
echo "base_model=$BASE_MODEL"
echo "batch_size=$BATCH_SIZE"
echo "num_workers=$NUM_WORKERS"
echo "epochs_total=$EPOCHS_TOTAL"
echo "dry_run=$DRY_RUN"
echo "ratios=${RATIOS[*]}"
echo "strategies=${STRATEGIES[*]}"

for ratio in "${RATIOS[@]}"; do
  ts="$(date +%Y%m%d_%H%M%S)"
  RUN_PIDS=()

  echo
  echo "=== START ratio=${ratio} $(date) ==="
  for idx in "${!STRATEGIES[@]}"; do
    launch_one "$ratio" "${STRATEGIES[$idx]}" "${GPUS[$idx]}" "$ts"
  done
  wait_for_ratio "$ratio"
  echo "=== END ratio=${ratio} $(date) ==="
done

echo "All brown-rainy baseline ratio sweeps completed."
