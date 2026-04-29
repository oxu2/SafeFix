#!/usr/bin/env bash
#SBATCH --job-name=br-attr-miss
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch scripts/juno_pose_brown_rainy_attribute_missing_ratio_pairs.sbatch [options]

Runs missing Qwen3 brown-rainy COCO pose attribute-RPC rows on one H100.
Default missing ratios are:
  0.7 0.9

Runs a flat queue with at most two concurrent jobs. As soon as either job
finishes, the next queued job starts. The default queue interleaves ratios, so
ratio=0.7 and ratio=0.9 start together for the same RPC setting.

For each ratio, runs four RPC settings:
  r=0.0,  p=0.0, c=15.0
  r=0.01, p=0.5, c=15.0
  r=0.03, p=0.5, c=15.0
  r=0.03, p=0.5, c=8.0

For ratio=0.7 only, also runs two nonzero-R/P probes inspired by the report's
strong ratio=0.7 attribute rows:
  r=0.01, p=0.5, c=0.0
  r=0.03, p=0.5, c=0.0

Options:
  --repo-dir PATH          Repo path on Juno. Default: $PWD or SLURM_SUBMIT_DIR
  --conda-env PATH         Conda environment prefix. Default: /groups/yguo/oxx220000/conda/envs/ad
  --data-root PATH         COCO root. Default: ~/scratch/safefix_cache/data/coco
  --generated-dir PATH     Brown-rainy generated image dir.
  --base-model PATH        Resume checkpoint path.
  --attribute-file PATH    Qwen3 train20k attribute metadata JSON.
  --ratios "R ..."         Space-separated ratios. Default: "0.7 0.9"
  --batch-size N           Training batch size. Default: 8
  --num-workers N          DataLoader workers per run. Default: 4
  --epochs-total N         Total epochs passed to train_pose_estimation.py. Default: 2
  --max-parallel N         Max concurrent training processes. Default: 2
  --rerun-existing         Rerun rows that already have test_metrics.json.
  --dry-run                Print per-run commands without training.
EOF
}

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
CONDA_ENV="/groups/yguo/oxx220000/conda/envs/ad"
DATA_ROOT="$HOME/scratch/safefix_cache/data/coco"
GENERATED_DIR="$HOME/scratch/safefix_cache/data/coco-pose-weather-edits/generated_coco_qwen3_1500_brown_rain_icedit"
BASE_MODEL="$HOME/scratch/safefix_cache/log_pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530/best.pth"
ATTRIBUTE_FILE=""
RATIOS_STRING="0.7 0.9"
BATCH_SIZE=8
NUM_WORKERS=4
EPOCHS_TOTAL=2
MAX_PARALLEL=2
LR="0.005"
SEED=42
NUM_ADDED_IMG=1000
MAX_TRAIN_SAMPLES=20000
RERUN_EXISTING=0
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
    --attribute-file)
      ATTRIBUTE_FILE="$2"
      shift 2
      ;;
    --ratios)
      RATIOS_STRING="$2"
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
    --max-parallel)
      MAX_PARALLEL="$2"
      shift 2
      ;;
    --rerun-existing)
      RERUN_EXISTING=1
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

cd "$REPO_DIR"

if [[ -z "$ATTRIBUTE_FILE" ]]; then
  ATTRIBUTE_FILE="$REPO_DIR/save/attribute_coco_pose_gemini_train20k_qwen3_8B.json"
fi

for required in \
  "$CONDA_ENV/bin/python" \
  "$DATA_ROOT" \
  "$GENERATED_DIR" \
  "$BASE_MODEL" \
  "$ATTRIBUTE_FILE"; do
  if [[ ! -e "$required" ]]; then
    echo "Required path not found: $required" >&2
    exit 1
  fi
done

mkdir -p "$REPO_DIR/log_pose/pids"

PYTHON_EXE="$CONDA_ENV/bin/python"
read -r -a RATIOS <<< "$RATIOS_STRING"

declare -a RPC_RECORDS=(
  "0.0|0.0|15.0"
  "0.01|0.5|15.0"
  "0.03|0.5|15.0"
  "0.03|0.5|8.0"
)

declare -a EXTRA_RATIO_07_RECORDS=(
  "0.01|0.5|0.0"
  "0.03|0.5|0.0"
)

ratio_label() {
  local value="$1"
  echo "${value//./p}"
}

run_name() {
  local ratio="$1"
  local rarity="$2"
  local penalty="$3"
  local balance="$4"
  local name="qwen3_brown_rainy_r${ratio}_r${rarity}_p${penalty}_c${balance}"
  echo "${name//./p}"
}

has_existing_result() {
  local name="$1"
  compgen -G "$REPO_DIR/log_pose/pose_keypointrcnn_resume_${name}_gpu0_*/test_metrics.json" >/dev/null
}

launch_one() {
  local ratio="$1"
  local rarity="$2"
  local penalty="$3"
  local balance="$4"
  local ts="$5"

  local extra_count_scale
  extra_count_scale="$(awk -v ratio="$ratio" 'BEGIN { printf "%.6g", 1.0 / ratio }')"

  local name
  name="$(run_name "$ratio" "$rarity" "$penalty" "$balance")"

  if [[ "$RERUN_EXISTING" -eq 0 ]] && has_existing_result "$name"; then
    echo "[skip] existing test_metrics.json found for ${name}"
    LAUNCHED_PID=""
    return 0
  fi

  local out_dir="$REPO_DIR/log_pose/pose_keypointrcnn_resume_${name}_gpu0_${ts}"
  local log_file="$REPO_DIR/log_pose/pose_resume_${name}_gpu0_${ts}.log"
  local pid_file="$REPO_DIR/log_pose/pids/pose_resume_${name}_gpu0_${ts}.pid"

  local cmd=(
    env
    MPLCONFIGDIR=/tmp/matplotlib-safefix
    CUDA_VISIBLE_DEVICES=0
    "$PYTHON_EXE" -u train_pose_estimation.py train
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
    --use-active-learning true
    --al-strategy attribute
    --al-attribute-file "$ATTRIBUTE_FILE"
    --al-attribute-focus clothes_color,weather
    --al-attribute-rarity-power "$rarity"
    --al-attribute-uncertainty-boost 1.0
    --al-attribute-log-acc-scale 0.0
    --al-attribute-extra-counts "clothes_color:brown=${NUM_ADDED_IMG},weather:rainy=${NUM_ADDED_IMG}"
    --al-attribute-extra-count-scale "$extra_count_scale"
    --al-attribute-frequency-penalty "$penalty"
    --al-attribute-class-balance "$balance"
    --al-attribute-use-entropy false
  )

  echo "[launch] ratio=${ratio} name=${name} R=${rarity} P=${penalty} C=${balance}"
  echo "         extra_count_scale=${extra_count_scale}"
  echo "         log=${log_file}"
  echo "         out=${out_dir}"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '         cmd='
    printf '%q ' "${cmd[@]}"
    printf '\n'
    LAUNCHED_PID=""
    return 0
  fi

  "${cmd[@]}" > "$log_file" 2>&1 &
  local pid=$!
  echo "$pid" > "$pid_file"
  echo "         pid=${pid} pidfile=${pid_file}"
  LAUNCHED_PID="$pid"
}

remove_active_at() {
  local remove_idx="$1"
  local new_pids=()
  local new_labels=()
  local i
  for i in "${!ACTIVE_PIDS[@]}"; do
    if [[ "$i" -ne "$remove_idx" ]]; then
      new_pids+=("${ACTIVE_PIDS[$i]}")
      new_labels+=("${ACTIVE_LABELS[$i]}")
    fi
  done
  ACTIVE_PIDS=("${new_pids[@]}")
  ACTIVE_LABELS=("${new_labels[@]}")
}

wait_for_one_active() {
  local finished_pid=""
  local status=0
  local idx=0
  local label="unknown"

  if wait -n -p finished_pid "${ACTIVE_PIDS[@]}"; then
    status=0
  else
    status=$?
    OVERALL_STATUS=1
  fi

  for idx in "${!ACTIVE_PIDS[@]}"; do
    if [[ "${ACTIVE_PIDS[$idx]}" == "$finished_pid" ]]; then
      label="${ACTIVE_LABELS[$idx]}"
      remove_active_at "$idx"
      break
    fi
  done

  if [[ "$status" -eq 0 ]]; then
    echo "[done] ${label} pid=${finished_pid} completed"
  else
    echo "[done] ${label} pid=${finished_pid} failed with status ${status}" >&2
  fi
}

echo "Running missing brown-rainy attribute ratio/RPC pairs on host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual}"
echo "repo_dir=$REPO_DIR"
echo "data_root=$DATA_ROOT"
echo "generated_dir=$GENERATED_DIR"
echo "base_model=$BASE_MODEL"
echo "attribute_file=$ATTRIBUTE_FILE"
echo "batch_size=$BATCH_SIZE"
echo "num_workers=$NUM_WORKERS"
echo "epochs_total=$EPOCHS_TOTAL"
echo "max_parallel=$MAX_PARALLEL"
echo "ratios=${RATIOS[*]}"
echo "rerun_existing=$RERUN_EXISTING"
echo "dry_run=$DRY_RUN"

declare -a TASKS=()
for record in "${RPC_RECORDS[@]}"; do
  for ratio in "${RATIOS[@]}"; do
    TASKS+=("${ratio}|${record}")
  done
done
for record in "${EXTRA_RATIO_07_RECORDS[@]}"; do
  TASKS+=("0.7|${record}")
done

echo "queued_tasks=${#TASKS[@]}"
for task in "${TASKS[@]}"; do
  echo "  ${task}"
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  for task in "${TASKS[@]}"; do
    IFS='|' read -r ratio rarity penalty balance <<< "$task"
    launch_one "$ratio" "$rarity" "$penalty" "$balance" "$(date +%Y%m%d_%H%M%S)"
  done
  exit 0
fi

declare -a ACTIVE_PIDS=()
declare -a ACTIVE_LABELS=()
OVERALL_STATUS=0

for task in "${TASKS[@]}"; do
  while [[ "${#ACTIVE_PIDS[@]}" -ge "$MAX_PARALLEL" ]]; do
    wait_for_one_active
  done

  IFS='|' read -r ratio rarity penalty balance <<< "$task"
  ts="$(date +%Y%m%d_%H%M%S)"
  LAUNCHED_PID=""
  launch_one "$ratio" "$rarity" "$penalty" "$balance" "$ts"
  if [[ -n "$LAUNCHED_PID" ]]; then
    ACTIVE_PIDS+=("$LAUNCHED_PID")
    ACTIVE_LABELS+=("ratio=${ratio} R=${rarity} P=${penalty} C=${balance}")
  fi
done

while [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; do
  wait_for_one_active
done

echo "[all done] overall_status=${OVERALL_STATUS} $(date)"
exit "$OVERALL_STATUS"
