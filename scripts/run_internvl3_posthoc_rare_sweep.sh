#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <train-pid> <run-root> [rare-threshold ...]" >&2
  exit 1
fi

train_pid="$1"
shift
run_root="$1"
shift

if [ "$#" -gt 0 ]; then
  thresholds=("$@")
else
  thresholds=(0.97 0.95 0.93)
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="/people/cs/o/oxx220000/.conda/envs/ad/bin/python"
val_predictions="$run_root/val_predictions_best.json"
test_predictions="$run_root/test_predictions.json"
eval_attribute_file="save/attribute_coco_pose_gemini_internvl3_8B.json"
train_attribute_file="save/attribute_coco_pose_gemini_train20k_internvl3_8B.json"
corpus_file="save/corpus_coco_pose_gemini.json"

while kill -0 "$train_pid" 2>/dev/null; do
  sleep 60
done

if [ ! -f "$val_predictions" ] || [ ! -f "$test_predictions" ]; then
  echo "Missing prediction JSON after training exit under $run_root" >&2
  exit 1
fi

for thr in "${thresholds[@]}"; do
  tag="${thr//./p}"
  outdir="$run_root/slice_analysis/internvl3_rare${tag}"
  mkdir -p "$outdir"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running val analysis at rare threshold $thr"
  "$python_bin" -u utils/stat_pose_estimation.py \
    --predictions-json "$val_predictions" \
    --attribute-file "$eval_attribute_file" \
    --attribute-split val_full \
    --train-attribute-file "$train_attribute_file" \
    --train-attribute-split train \
    --corpus "$corpus_file" \
    --eval-split val \
    --train-max-samples 20000 \
    --rare-label-threshold "$thr" \
    --output-json "$outdir/val_slice_stats.json" \
    --output-csv "$outdir/val_slice_stats.csv"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running test analysis at rare threshold $thr"
  "$python_bin" -u utils/stat_pose_estimation.py \
    --predictions-json "$test_predictions" \
    --attribute-file "$eval_attribute_file" \
    --attribute-split val_full \
    --train-attribute-file "$train_attribute_file" \
    --train-attribute-split train \
    --corpus "$corpus_file" \
    --eval-split test \
    --train-max-samples 20000 \
    --rare-label-threshold "$thr" \
    --output-json "$outdir/test_slice_stats.json" \
    --output-csv "$outdir/test_slice_stats.csv"
done
