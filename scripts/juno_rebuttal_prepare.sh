#!/usr/bin/env bash
# Inspect (and, only when requested, hydrate) the Juno assets needed for the
# NeurIPS rebuttal experiments.  This script never downloads a model unless
# --download-model is passed explicitly.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/juno_rebuttal_prepare.sh [options]

Options:
  --repo-dir PATH       Juno rebuttal worktree. Default: /work/$USER/safefix_rebuttal
  --data-root PATH      Juno data source. Default: /work/$USER/data
  --hf-home PATH        Persistent Hugging Face cache. Default: /work/$USER/hf_cache
  --model MODEL_ID      Hugging Face model. Default: Qwen/Qwen3-VL-8B-Instruct
  --download-model      Download the model only if it is absent or incomplete.
  --help                Show this message.

The script only inventories data.  CelebA and ImageNet are license-controlled,
so it intentionally does not attempt to fetch them automatically.  COCO and
KITTI should be downloaded with the repository download helpers if missing.
EOF
}

REPO_DIR="/work/${USER}/safefix_rebuttal"
DATA_ROOT="/work/${USER}/data"
HF_HOME_DIR="/work/${USER}/hf_cache"
MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
DOWNLOAD_MODEL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --hf-home) HF_HOME_DIR="$2"; shift 2 ;;
    --model) MODEL_ID="$2"; shift 2 ;;
    --download-model) DOWNLOAD_MODEL=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

model_cache_name="models--${MODEL_ID//\//--}"
model_cache_dir="$HF_HOME_DIR/hub/$model_cache_name"

model_is_complete() {
  [[ -f "$model_cache_dir/refs/main" ]] &&
    find -L "$model_cache_dir/snapshots" -type f -name '*.safetensors' -size +100M -print -quit 2>/dev/null | grep -q .
}

if ! model_is_complete && [[ "$DOWNLOAD_MODEL" -eq 1 ]]; then
  mkdir -p "$HF_HOME_DIR"
  export HF_HOME="$HF_HOME_DIR"
  if command -v hf >/dev/null 2>&1; then
    hf download "$MODEL_ID"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$MODEL_ID"
  else
    echo "Neither 'hf' nor 'huggingface-cli' is available; install huggingface_hub first." >&2
    exit 1
  fi
fi

status() {
  local label="$1"
  local path="$2"
  if [[ -e "$path" ]]; then
    printf 'PRESENT\t%s\t%s\n' "$label" "$path"
  else
    printf 'MISSING\t%s\t%s\n' "$label" "$path"
  fi
}

image_status() {
  local label="$1"
  local path="$2"
  local image
  image=$(find -L "$path" -maxdepth 3 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -print -quit 2>/dev/null || true)
  if [[ -n "$image" ]]; then
    printf 'PRESENT\t%s\t%s\n' "$label" "$path"
  else
    printf 'EMPTY_OR_MISSING\t%s\t%s\n' "$label" "$path"
  fi
}

echo "Juno rebuttal asset inventory"
echo "repo_dir=$REPO_DIR"
echo "data_root=$DATA_ROOT"
echo "hf_home=$HF_HOME_DIR"
echo "model=$MODEL_ID"
if model_is_complete; then
  echo "model_cache=READY ($model_cache_dir)"
else
  echo "model_cache=MISSING_OR_INCOMPLETE ($model_cache_dir)"
fi

printf '\nSTATUS\tASSET\tPATH\n'
status "rebuttal worktree" "$REPO_DIR"
image_status "COCO train2017" "$DATA_ROOT/coco/train2017"
image_status "COCO val2017" "$DATA_ROOT/coco/val2017"
status "COCO keypoint annotations" "$DATA_ROOT/coco/annotations/person_keypoints_train2017.json"
image_status "COCO generated brown-rainy set" "$DATA_ROOT/coco-pose-weather-edits/generated_coco_qwen3_1500_brown_rain"
image_status "COCO ICEdit set" "$DATA_ROOT/coco-pose-weather-edits/generated_coco_qwen3_1500_brown_rain_icedit"
image_status "KITTI source" "$DATA_ROOT/kitti_object/training"
image_status "KITTI YOLO split" "$DATA_ROOT/kitti_yolov8_first5000"
image_status "CelebA source" "$DATA_ROOT/img_align_celeba"
image_status "CelebA generated set" "$DATA_ROOT/redhair_brownskin_sademotion_instruct_pix2pix_seed42_1k"
image_status "ImageNet10 source" "$DATA_ROOT/imagenet_subset"
image_status "ImageNet10 generated set" "$DATA_ROOT/pinkcolor_fabrictexture_instruct_pix2pix_seed42_1k"

printf '\nExpected metadata (after code sync)\n'
for metadata in \
  save/attribute_coco_pose_gemini_train20k_qwen3_8B.json \
  save/attribute_coco_pose_gemini_train20k_internvl3_8B.json \
  save/attribute_coco_pose_gemini_train20k_blip.json \
  save/attribute_kitti_qwen3_8B.json \
  save/attribute_celeba_trainval_qwen3_8B.json \
  save/attribute10_qwen3_8B.json; do
  status "$metadata" "$REPO_DIR/$metadata"
done

printf '\nRecommended first experiment: COCO attribute-source ablation\n'
printf '%s\n' 'It reuses the cached source/generated images and the existing Qwen3-VL, InternVL3, and BLIP metadata.'
