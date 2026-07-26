#!/usr/bin/env bash
# Submit with an explicit Python environment, for example:
# PYTHON_EXE=/work/oxx220000/venvs/qwen3/bin/python \
#   sbatch scripts/juno_qwen3_rebuttal_smoke.sbatch
# The environment must provide torch, transformers (with Qwen3-VL support),
# accelerate, Pillow, and qwen-vl-utils.
#SBATCH --job-name=safefix-qwen3-smoke
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-/work/${USER}/safefix_rebuttal}"
PYTHON_EXE="${PYTHON_EXE:-}"
HF_HOME_DIR="${HF_HOME:-/work/${USER}/hf_cache}"
DATA_ROOT="${SAFEFIX_DATA_ROOT:-/work/${USER}/data}"
MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
NOTIFICATION_ENV_FILE="$REPO_DIR/.vscode/.env"

if [[ -z "${NOTIFICATION_WEBHOOK:-}" && -f "$NOTIFICATION_ENV_FILE" ]]; then
  set -a
  source "$NOTIFICATION_ENV_FILE"
  set +a
fi

cleanup_and_notify() {
  local exit_code="$1"
  [[ -n "${NOTIFICATION_WEBHOOK:-}" ]] || return 0
  [[ -f "$REPO_DIR/scripts/test_notification.py" ]] || return 0

  local message
  if [[ "$exit_code" -eq 0 ]]; then
    message="✅ SafeFix job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) completed on ${SLURM_JOB_PARTITION}."
  else
    message="❌ SafeFix job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) failed with exit ${exit_code} on ${SLURM_JOB_PARTITION}."
  fi
  python3 "$REPO_DIR/scripts/test_notification.py" "$message" || true
}
trap 'cleanup_and_notify $?' EXIT

if [[ -z "$PYTHON_EXE" || ! -x "$PYTHON_EXE" ]]; then
  echo "Set PYTHON_EXE to the Juno Qwen3 environment before submission." >&2
  exit 2
fi

for required in "$REPO_DIR" "$DATA_ROOT/coco/train2017" "$HF_HOME_DIR"; do
  [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 1; }
done

mkdir -p "$REPO_DIR/logs/juno"
export HF_HOME="$HF_HOME_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export SAFEFIX_DATA_ROOT="$DATA_ROOT"
export QWEN_MODEL_ID="$MODEL_ID"

cd "$REPO_DIR"
echo "host=$(hostname)"
echo "python=$PYTHON_EXE"
echo "model=$MODEL_ID"
echo "hf_home=$HF_HOME"
echo "data_root=$DATA_ROOT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

"$PYTHON_EXE" - <<'PY'
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")
data_root = Path(os.environ["SAFEFIX_DATA_ROOT"])
image_candidates = (
    data_root / "coco" / "train2017",
    data_root / "kitti_object" / "training" / "image_2",
    data_root / "imagenet_subset",
)
image_path = None
for candidate_root in image_candidates:
    if not candidate_root.is_dir():
        continue
    image_path = next(
        (
            path
            for pattern in ("*.jpg", "*.jpeg", "*.png")
            for path in candidate_root.rglob(pattern)
        ),
        None,
    )
    if image_path is not None:
        break
if image_path is None:
    raise RuntimeError(f"No smoke-test image was found under {data_root}.")
print(f"smoke_image={image_path}")

print(f"torch={torch.__version__}, cuda={torch.version.cuda}, gpu={torch.cuda.get_device_name(0)}")
processor = AutoProcessor.from_pretrained(model_id, local_files_only=True, use_fast=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)
image = Image.open(image_path).convert("RGB")
image.thumbnail((640, 640))
messages = [{"role": "user", "content": [
    {"type": "image", "url": f"file://{image_path}"},
    {"type": "text", "text": "Reply with exactly one word: person or no-person."},
]}]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt").to(model.device)
generated = model.generate(**inputs, max_new_tokens=4, do_sample=False)
new_tokens = generated[:, inputs.input_ids.shape[1]:]
print("smoke_response=" + processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip())
print("Qwen3-VL offline smoke test passed.")
PY
