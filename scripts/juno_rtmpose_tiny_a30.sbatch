#!/usr/bin/env bash
#SBATCH --job-name=rtmpose-tiny-a30
#SBATCH --partition=a30
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
  sbatch scripts/juno_rtmpose_tiny_a30.sbatch [options]

Runs RTMPose Tiny COCO val inference on Juno A30, saves COCO-format keypoint
predictions, then compares qwen3/internvl3/blip pose attribute slice analyses.

Options:
  --repo-dir PATH          Repo path on Juno. Default: $PWD
  --python PATH            Python executable. Default: /groups/yguo/oxx220000/venvs/rtmpose/bin/python
  --bootstrap-deps         Create/update the default RTMPose venv before running.
  --data-root PATH         COCO root. Default: ~/scratch/safefix_cache/data/coco
  --config PATH            MMPose config. Default: pose_estimation/rtmpose_assets/rtmpose_tiny_coco_juno_eval.py
  --checkpoint PATH_OR_URL RTMPose Tiny checkpoint. Default: official OpenMMLab RTMPose-t COCO checkpoint URL.
  --batch-size N           Eval batch size. Default: 32
  --num-workers N          DataLoader workers. Default: 4
  --seed N                 Deterministic seed. Default: 42
  --skip-analysis          Only run RTMPose eval, not attribute slice analysis.
  --dry-run                Print resolved command and exit.
EOF
}

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
DEFAULT_VENV="/groups/yguo/oxx220000/venvs/rtmpose"
PYTHON_EXE="$DEFAULT_VENV/bin/python"
BASE_PYTHON="/groups/yguo/oxx220000/conda/envs/sf/bin/python"
BOOTSTRAP_DEPS=0
DATA_ROOT="$HOME/scratch/safefix_cache/data/coco"
CONFIG="pose_estimation/rtmpose_assets/rtmpose_tiny_coco_juno_eval.py"
CHECKPOINT="https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth"
BATCH_SIZE=32
NUM_WORKERS=4
SEED=42
SKIP_ANALYSIS=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_EXE="$2"
      shift 2
      ;;
    --bootstrap-deps)
      BOOTSTRAP_DEPS=1
      shift
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
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
    --seed)
      SEED="$2"
      shift 2
      ;;
    --skip-analysis)
      SKIP_ANALYSIS=1
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

if [[ "$BOOTSTRAP_DEPS" -eq 1 && "$DRY_RUN" -eq 0 ]]; then
  if [[ ! -x "$BASE_PYTHON" ]]; then
    echo "Base Python for venv bootstrap not found: $BASE_PYTHON" >&2
    exit 1
  fi
  if [[ ! -x "$DEFAULT_VENV/bin/python" ]]; then
    mkdir -p "$(dirname "$DEFAULT_VENV")"
    "$BASE_PYTHON" -m venv --system-site-packages "$DEFAULT_VENV"
  fi
  "$DEFAULT_VENV/bin/python" -m pip install --upgrade pip openmim
  "$DEFAULT_VENV/bin/python" -m mim install "mmcv>=2.0.0,<2.3.0" "mmdet>=3.0.0,<3.4.0" "mmpose>=1.3.2,<1.4.0"
  PYTHON_EXE="$DEFAULT_VENV/bin/python"
fi

if [[ ! -x "$PYTHON_EXE" ]]; then
  echo "Python executable not found: $PYTHON_EXE" >&2
  echo "Rerun with --bootstrap-deps or pass --python PATH for an env containing mmpose/mmdet/mmcv." >&2
  exit 1
fi

for required in "$DATA_ROOT" "$CONFIG"; do
  if [[ ! -e "$required" ]]; then
    echo "Required path not found: $required" >&2
    exit 1
  fi
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_DIR/logs/pose/rtmpose_tiny_coco_val_seed${SEED}_a30_${TIMESTAMP}"
LOG_FILE="$REPO_DIR/logs/pose/rtmpose_tiny_coco_val_seed${SEED}_a30_${TIMESTAMP}.log"
mkdir -p "$OUT_DIR"

cmd=(
  env
  MPLCONFIGDIR=/tmp/matplotlib-safefix
  CUDA_VISIBLE_DEVICES=0
  "$PYTHON_EXE" -u utils/run_rtmpose_eval_and_analysis.py
  --config "$CONFIG"
  --checkpoint "$CHECKPOINT"
  --data-root "$DATA_ROOT"
  --work-dir "$OUT_DIR"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --seed "$SEED"
)

if [[ "$SKIP_ANALYSIS" -eq 1 ]]; then
  cmd+=(--skip-analysis)
fi

echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual}"
echo "python=$PYTHON_EXE"
echo "data_root=$DATA_ROOT"
echo "config=$CONFIG"
echo "checkpoint=$CHECKPOINT"
echo "seed=$SEED"
echo "log=$LOG_FILE"
echo "out=$OUT_DIR"
echo "Command:"
printf '  %q' "${cmd[@]}"
printf '\n'

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

"${cmd[@]}" > "$LOG_FILE" 2>&1
