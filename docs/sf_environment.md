# Juno `sf` Environment

This note records the clean Juno environment intended for future SafeFix
reproduction. It is separate from the historical `ad` environment, which now
works on H100 but still contains older and experiment-specific packages.

## Location Recommendation

Use the PI group path for the main Juno training environment:

```text
/groups/yguo/oxx220000/conda/envs/sf
```

Do not put the main shared training environment under `~/`. A home-directory
environment is acceptable for small personal tests, but it is a poor default for
this project because it is more likely to hit home quota limits, is less clear
for shared reproduction, and is easier to break when scripts run from SLURM.
The group path is the better stable target for batch jobs and future GitHub
reproduction notes.

## Created Environment

The current Juno environment is:

- Path: `/groups/yguo/oxx220000/conda/envs/sf`
- Name convention: `sf` means SafeFix
- Python: `3.10`
- GPU stack:
  - `torch==2.4.1+cu121`
  - `torchvision==0.19.1+cu121`
  - `torchaudio==2.4.1+cu121`
- H100 validation job: `185362`
  - CUDA available: `True`
  - device: `NVIDIA H100 80GB HBM3`
  - capability: `(9, 0)`
  - CUDA tensor matmul succeeded

The package snapshots are stored in:

- `docs/sf_pip_freeze.txt`
- `docs/sf_conda_export.txt`

The current Juno `ad` snapshots, after the H100 PyTorch upgrade, are stored in:

- `docs/juno_ad_current_pip_freeze.txt`
- `docs/juno_ad_current_conda_export.txt`

## Recreate Commands

Run these on Juno:

```bash
module load miniconda

conda create -p /groups/yguo/oxx220000/conda/envs/sf python=3.10 pip -y

/groups/yguo/oxx220000/conda/envs/sf/bin/python -m pip install \
  --upgrade --no-cache-dir \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121

/groups/yguo/oxx220000/conda/envs/sf/bin/python -m pip install \
  --upgrade --no-cache-dir \
  numpy==1.26.4 \
  pillow==11.0.0 \
  pycocotools==2.0.11 \
  scipy==1.13.1 \
  scikit-learn==1.5.2 \
  matplotlib==3.9.4 \
  tqdm==4.66.6 \
  transformers==4.46.3 \
  safetensors==0.4.5 \
  accelerate==1.1.1 \
  timm==1.0.12 \
  pandas==2.2.3 \
  opencv-python-headless==4.10.0.84 \
  tensorboard==2.18.0 \
  wandb==0.18.7 \
  json-tricks==3.17.3
```

## Train-Pose Dependency Surface

For `train_pose_estimation.py train`, the direct and relevant transitive
third-party dependencies are:

- `torch`
- `torchvision`
- `numpy`
- `Pillow`
- `pycocotools`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `transformers`
- `timm`
- `tqdm`

Useful adjacent packages included for reproducibility and normal experiment
work:

- `pandas`
- `opencv-python-headless`
- `tensorboard`
- `wandb`
- `accelerate`
- `safetensors`
- `json-tricks`

The repo import and syntax validation passed with `sf` for:

```text
train_pose_estimation.py
utils/imports.py
utils/active_learning.py
utils/hierarchical_al.py
utils/stat_pose_estimation.py
pose_estimation/dataset.py
pose_estimation/engine.py
pose_estimation/modeling.py
pose_estimation/visualize.py
```

The command below also passed on Juno:

```bash
cd ~/work/safefix
/groups/yguo/oxx220000/conda/envs/sf/bin/python train_pose_estimation.py train --help
```
