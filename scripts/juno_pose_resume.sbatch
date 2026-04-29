#!/usr/bin/env bash
#SBATCH --job-name=safefix-pose
#SBATCH --partition=h100-2.47gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch scripts/juno_pose_resume.sbatch [options]

Options:
  --queue NAME            Queue to run: ratio0.9 or ratio0.05. Default: ratio0.9
  --start-at NAME         Queue entry to start from.
  --repo-dir PATH         Repo path on Juno. Default: $PWD
  --conda-env PATH        Conda environment prefix. Default: /groups/yguo/oxx220000/conda/envs/ad
  --data-root PATH        COCO root on persistent storage. Default: ~/work/data/coco
  --generated-dir PATH    Generated rainy image dir on persistent storage.
  --base-model PATH       Resume checkpoint path.
  --attribute-file PATH   Attribute metadata JSON.
  --partition NAME        Informational only after sbatch has started.
  --batch-size N          Training batch size. Default: 8
  --num-workers N         DataLoader workers. Default: 4
  --no-scratch-copy       Use persistent paths directly instead of scratch copies.
EOF
}

QUEUE="ratio0.9"
START_AT=""
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
CONDA_ENV="/groups/yguo/oxx220000/conda/envs/ad"
DATA_ROOT="$HOME/work/data/coco"
GENERATED_DIR="$HOME/work/data/coco-pose-weather-edits/generated_coco_train1500_rain"
BASE_MODEL=""
ATTRIBUTE_FILE=""
BATCH_SIZE=8
NUM_WORKERS=4
USE_SCRATCH=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --queue)
      QUEUE="$2"
      shift 2
      ;;
    --start-at)
      START_AT="$2"
      shift 2
      ;;
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
    --partition)
      echo "Note: --partition must be passed to sbatch before the script path to affect scheduling." >&2
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
    --no-scratch-copy)
      USE_SCRATCH=0
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

if [[ -z "$BASE_MODEL" ]]; then
  BASE_MODEL="$REPO_DIR/log_pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530/best.pth"
fi

if [[ -z "$ATTRIBUTE_FILE" ]]; then
  ATTRIBUTE_FILE="$REPO_DIR/save/attribute_coco_pose_gemini_train20k_internvl3_8B.json"
fi

PYTHON_EXE="$CONDA_ENV/bin/python"
if [[ ! -x "$PYTHON_EXE" ]]; then
  echo "Python executable not found: $PYTHON_EXE" >&2
  exit 1
fi

for required in "$DATA_ROOT" "$GENERATED_DIR" "$BASE_MODEL" "$ATTRIBUTE_FILE"; do
  if [[ ! -e "$required" ]]; then
    echo "Required path not found: $required" >&2
    exit 1
  fi
done

RUN_DATA_ROOT="$DATA_ROOT"
RUN_GENERATED_DIR="$GENERATED_DIR"
SCRATCH_RUN_DIR=""

if [[ "$USE_SCRATCH" -eq 1 ]]; then
  SCRATCH_BASE="${SCRATCH:-$HOME/scratch}"
  SCRATCH_RUN_DIR="$SCRATCH_BASE/safefix_${SLURM_JOB_ID:-manual}_$QUEUE"
  mkdir -p "$SCRATCH_RUN_DIR"

  echo "Copying COCO and generated images to scratch: $SCRATCH_RUN_DIR"
  rsync -a "$DATA_ROOT/" "$SCRATCH_RUN_DIR/coco/"
  mkdir -p "$SCRATCH_RUN_DIR/coco-pose-weather-edits"
  rsync -a "$GENERATED_DIR/" "$SCRATCH_RUN_DIR/coco-pose-weather-edits/generated_coco_train1500_rain/"

  RUN_DATA_ROOT="$SCRATCH_RUN_DIR/coco"
  RUN_GENERATED_DIR="$SCRATCH_RUN_DIR/coco-pose-weather-edits/generated_coco_train1500_rain"
fi

mkdir -p "$REPO_DIR/log_pose/pids"

if [[ "$QUEUE" == "ratio0.9" ]]; then
  QUEUE_SCRIPT="$REPO_DIR/scripts/run_pose_ratio0.9_gpu1_internvl3_queue.sh"
elif [[ "$QUEUE" == "ratio0.05" ]]; then
  QUEUE_SCRIPT="$REPO_DIR/scripts/run_pose_ratio0.05_gpu0_badge_then_attr_refine.sh"
else
  echo "Unknown queue: $QUEUE" >&2
  usage
  exit 1
fi

cmd=(
  bash "$QUEUE_SCRIPT"
  --python "$PYTHON_EXE"
  --base-model "$BASE_MODEL"
  --attribute-file "$ATTRIBUTE_FILE"
  --data-root "$RUN_DATA_ROOT"
  --generated-image-dir "$RUN_GENERATED_DIR"
  --gpu 0
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
)

if [[ -n "$START_AT" ]]; then
  cmd+=(--start-at "$START_AT")
fi

echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Command:"
printf '  %q' "${cmd[@]}"
printf '\n'

export SAFE_FIX_GENERATED_IMAGE_DIR="$RUN_GENERATED_DIR"
"${cmd[@]}"

if [[ "$USE_SCRATCH" -eq 1 ]]; then
  echo "Scratch run directory retained for inspection: $SCRATCH_RUN_DIR"
fi
