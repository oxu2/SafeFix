#!/usr/bin/env bash
#SBATCH --job-name=celeba-ip2p-smoke
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch scripts/juno_celeba_instructpix2pix_smoke.sbatch [options]

Runs a one-epoch CelebA alpred.py smoke test on a Juno H100 using the
redhair_brownskin_sademotion_instruct_pix2pix_seed42_1k generated subset.

The job stages a runnable repo copy under scratch and runs from that scratch
copy so alpred.py reads/writes local run artifacts under scratch. The full
CelebA source and generated subset are expected to already live in the Juno
scratch cache.

Options:
  --repo-dir PATH       Persistent repo path. Default: $PWD or SLURM_SUBMIT_DIR
  --conda-env PATH      Conda env prefix. Default: /groups/yguo/oxx220000/conda/envs/ad
  --data-dir PATH       CelebA source image dir in scratch.
  --gen-dir PATH        Generated image dir in scratch.
  --base-model PATH     Resume checkpoint in persistent repo.
  --attribute-file PATH Qwen3 CelebA metadata JSON.
  --ratio FLOAT         Original-data sub-ratio. Default: 0.1
  --dry-run             Print resolved command and exit before training.
EOF
}

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
CONDA_ENV="/groups/yguo/oxx220000/conda/envs/ad"
DATA_DIR="$HOME/scratch/safefix_cache/data/img_align_celeba"
GEN_DIR="$HOME/scratch/safefix_cache/data/redhair_brownskin_sademotion_instruct_pix2pix_seed42_1k"
BASE_MODEL=""
ATTRIBUTE_FILE=""
RATIO="0.1"
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
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --gen-dir)
      GEN_DIR="$2"
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
    --ratio)
      RATIO="$2"
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

if [[ -z "$BASE_MODEL" ]]; then
  BASE_MODEL="$REPO_DIR/weights/model_celeba_resnet_base.pth"
fi

if [[ -z "$ATTRIBUTE_FILE" ]]; then
  ATTRIBUTE_FILE="$REPO_DIR/save/attribute_celeba_trainval_qwen3_8B.json"
fi

PYTHON_EXE="$CONDA_ENV/bin/python"

for required in "$PYTHON_EXE" "$REPO_DIR/alpred.py" "$DATA_DIR" "$GEN_DIR" "$BASE_MODEL" "$ATTRIBUTE_FILE"; do
  if [[ ! -e "$required" ]]; then
    echo "Required path not found: $required" >&2
    exit 1
  fi
done

SCRATCH_BASE="${SCRATCH:-$HOME/scratch}"
JOB_ID="${SLURM_JOB_ID:-manual}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$SCRATCH_BASE/safefix_runs/celeba_ip2p_smoke_${JOB_ID}_${TIMESTAMP}"
RUN_REPO="$RUN_ROOT/safefix"
PERSIST_LOG_DIR="$REPO_DIR/logs_celeba"

mkdir -p "$RUN_ROOT" "$PERSIST_LOG_DIR"

echo "Running CelebA instruct-pix2pix smoke test on host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual}"
echo "repo_dir=$REPO_DIR"
echo "run_root=$RUN_ROOT"
echo "run_repo=$RUN_REPO"
echo "data_dir=$DATA_DIR"
echo "gen_dir=$GEN_DIR"
echo "base_model=$BASE_MODEL"
echo "attribute_file=$ATTRIBUTE_FILE"
echo "ratio=$RATIO"
echo "dry_run=$DRY_RUN"

if [[ "$DRY_RUN" -eq 0 ]]; then
  echo "Staging repo-side code, metadata, checkpoint, and attribute files into scratch."
  mkdir -p "$RUN_REPO"
  rsync -a \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude 'log_pose/' \
    --exclude 'logs/' \
    --exclude 'logs_celeba/' \
    --exclude 'logs_imagenet/' \
    --exclude 'runs/' \
    --exclude 'outputs/' \
    --exclude 'data/img_align_celeba/' \
    --exclude 'data/redhair_brownskin_sademotion_instruct_pix2pix/' \
    --exclude 'data/redhair_brownskin_sademotion_instruct_pix2pix_seed42_1k/' \
    --exclude 'data/hibug_redhair_brownskin/' \
    --exclude 'data/hibug_redhair_brownskin_seed42_1k/' \
    --exclude 'data/pinkcolor_fabrictexture_instruct_pix2pix/' \
    --exclude 'data/pinkcolor_fabrictexture_instruct_pix2pix_seed42_1k/' \
    "$REPO_DIR/" "$RUN_REPO/"
else
  RUN_REPO="$REPO_DIR"
fi

cd "$RUN_REPO"
mkdir -p logs_celeba save

LOG_FILE="$RUN_REPO/logs_celeba/juno_smoke_instructpix2pix_random_resnet_celeba_1000_ratio${RATIO//./p}_${TIMESTAMP}.log"

cmd=(
  env
  MPLCONFIGDIR="$RUN_ROOT/matplotlib"
  CUDA_VISIBLE_DEVICES=0
  "$PYTHON_EXE" -u alpred.py
  --dataset celeba
  --model resnet
  --attributes redhair_brownskin_sademotion
  --data_dir "$DATA_DIR"
  --gen_image_dir "$GEN_DIR"
  --num_added_img 1000
  --num_epochs 1
  --resume_model "$RUN_REPO/weights/model_celeba_resnet_base.pth"
  --freeze_backbone False
  --lr 1e-5
  --batch_size 64
  --workers 4
  --seed 42
  --gpu 0
  --save_pred_model best
  --save_preds False
  --save_model False
  --verbose True
  --sub_ratio "$RATIO"
  --skip_feasible True
  --use_active_learning False
)

printf 'Resolved command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

set +e
"${cmd[@]}" > "$LOG_FILE" 2>&1
status=$?
set -e

echo "alpred exit status: $status"
echo "scratch log: $LOG_FILE"

rsync -a "$RUN_REPO/logs_celeba/" "$PERSIST_LOG_DIR/"

if [[ "$status" -ne 0 ]]; then
  echo "Smoke test failed; copied logs to $PERSIST_LOG_DIR" >&2
  exit "$status"
fi

echo "Smoke test completed; copied logs to $PERSIST_LOG_DIR"
