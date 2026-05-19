#!/usr/bin/env bash
#SBATCH --job-name=br-rpc03-05-seq
#SBATCH --partition=h100
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
  sbatch scripts/juno_pose_brown_rainy_rpc03_05_sweep.sbatch [options]

Runs the remaining Qwen3 brown-rainy COCO pose RPC grid for ratios 0.3 and 0.5
sequentially inside one H100 SLURM allocation. The already completed
ratio=0.3, R=0.0, P=0.0, C=15.0 row is intentionally skipped.

Options:
  --repo-dir PATH          Repo path on Juno. Default: $PWD or SLURM_SUBMIT_DIR
  --conda-env PATH         Conda environment prefix. Default: /groups/yguo/oxx220000/conda/envs/ad
  --data-root PATH         COCO root. Default: ~/scratch/safefix_cache/data/coco
  --generated-dir PATH     Brown-rainy generated image dir.
  --base-model PATH        Resume checkpoint path.
  --batch-size N           Training batch size. Default: 8
  --num-workers N          DataLoader workers. Default: 4
  --dry-run                Print all per-run commands without training.
EOF
}

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
CONDA_ENV="/groups/yguo/oxx220000/conda/envs/ad"
DATA_ROOT="$HOME/scratch/safefix_cache/data/coco"
GENERATED_DIR="$HOME/scratch/safefix_cache/data/coco-pose-weather-edits/generated_coco_qwen3_1500_brown_rain_icedit"
BASE_MODEL="$HOME/scratch/safefix_cache/logs/pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530/best.pth"
BATCH_SIZE=8
NUM_WORKERS=4
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
  "$BASE_MODEL" \
  "$REPO_DIR/scripts/juno_pose_brown_rainy_attribute.sbatch"; do
  if [[ ! -e "$required" ]]; then
    echo "Required path not found: $required" >&2
    exit 1
  fi
done

declare -a RUNS=(
  "0.3|0.01|0.5|15.0|r03-r001-p05-c15"
  "0.3|0.03|0.5|15.0|r03-r003-p05-c15"
  "0.3|0.03|0.5|8.0|r03-r003-p05-c8"
  "0.5|0.0|0.0|15.0|r05-r000-p0-c15"
  "0.5|0.01|0.5|15.0|r05-r001-p05-c15"
  "0.5|0.03|0.5|15.0|r05-r003-p05-c15"
  "0.5|0.03|0.5|8.0|r05-r003-p05-c8"
)

echo "Running remaining sequential brown-rainy RPC sweep on host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual}"
echo "repo_dir=$REPO_DIR"
echo "data_root=$DATA_ROOT"
echo "generated_dir=$GENERATED_DIR"
echo "base_model=$BASE_MODEL"
echo "batch_size=$BATCH_SIZE"
echo "num_workers=$NUM_WORKERS"
echo "dry_run=$DRY_RUN"

for record in "${RUNS[@]}"; do
  IFS='|' read -r ratio rarity penalty balance label <<< "$record"

  cmd=(
    bash scripts/juno_pose_brown_rainy_attribute.sbatch
    --repo-dir "$REPO_DIR"
    --conda-env "$CONDA_ENV"
    --data-root "$DATA_ROOT"
    --generated-dir "$GENERATED_DIR"
    --base-model "$BASE_MODEL"
    --ratio "$ratio"
    --rarity-power "$rarity"
    --frequency-penalty "$penalty"
    --class-balance "$balance"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --no-scratch-copy
  )

  if [[ "$DRY_RUN" -eq 1 ]]; then
    cmd+=(--dry-run)
  fi

  echo
  echo "=== START ${label} ratio=${ratio} R=${rarity} P=${penalty} C=${balance} $(date) ==="
  printf '  %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
  echo "=== END ${label} $(date) ==="
done
