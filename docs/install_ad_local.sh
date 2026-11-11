#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_root="$(conda info --base)"
ad_python="$conda_root/envs/ad/bin/python"
if [[ ! -x "$ad_python" ]]; then
    conda create -n ad python=3.9 pip -y
fi
"$ad_python" -m pip install uv
"$ad_python" -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 numpy==1.26.4 --index-url https://download.pytorch.org/whl/cu121
"$conda_root/envs/ad/bin/uv" --config-file "$repo/docs/ad_uv.toml" pip install --python "$ad_python" -r "$repo/docs/ad_requirements_local.txt"
"$ad_python" "$repo/docs/ad_patch_chumpy.py"
"$ad_python" -m pip check
