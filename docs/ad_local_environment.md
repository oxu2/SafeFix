# Local ad environment (2026-11-11)

Host: `csr-93596`, Ubuntu 22.04 x86_64; six NVIDIA A100 80GB and two RTX A6000 GPUs, driver 560.35.05. Environment: `/home/rxl190028/miniconda3/envs/ad`, Python 3.9.25.

## Sources and reproduction

`../ad_requirements.txt` preserves the user-provided UTD pip-freeze snapshot verbatim, including nonportable paths and conflicting historical dependencies. Install from `ad_requirements_local.txt`, not the original freeze. All changes and deferred packages are enumerated in `ad_local_adjustments.json`. The installed package snapshot is `ad_local_pip_freeze.txt`.

```bash
bash docs/install_ad_local.sh
conda activate ad
python -m pip check
```

The installer expects Conda and reuses an existing environment named ad. For exact reproduction use a clean ad environment: installing into a populated environment can retain unrelated packages. `ad_uv.toml` provides the undeclared pip/NumPy build requirements of chumpy.

## Compatibility decisions

- Use PyTorch 2.1.0+cu121, torchvision 0.16.0+cu121 and torchaudio 2.1.0+cu121 together. The selected CUDA 12.1 runtime works with driver 560; its compiled architectures include sm_80 and sm_86. The official MMCV 2.1.0 CPython 3.9 CUDA 12.1 / torch 2.1 wheel avoids rebuilding native extensions. Do not mix the old CUDA 11/12 runtime pins from the original snapshot.
- Retain NumPy 1.26.4. Transformers 4.37.2 and tokenizers 0.15.2 satisfy sentence-transformers 2.7.0 and the retained Hugging Face Hub. Upgrade torchdata to 0.7.0, timm to 0.9.16, PyArrow to 14.0.2, requests to 2.32.3 and CMake to 3.31.6; adjust tensorboardX to 2.6 and tensorflow-metadata to 1.15.0 to resolve shared dependencies.
- Keep both OpenCV distributions required by retained dependencies at 4.10.0.84 to avoid the original mismatched cv2 versions.
- `ad_patch_chumpy.py` applies a small idempotent source patch to chumpy 0.70: replace removed NumPy scalar aliases with their original Python built-in equivalents. This patch must be reapplied if chumpy is reinstalled; pip freeze alone does not capture it.
- Defer custom editable adapter-transformers and Apex because their source trees are absent. Standard Transformers is installed; the missing custom implementation is not silently replaced.
- Defer adatest/checklist, clip-retrieval/img2dataset, salesforce-lavis, openmim/openxlab, keras-nightly/tensorflow-estimator due to legacy build behavior or conflicting constraints. For example, retrieval/downloader pins require fsspec 2022.1.0, and LAVIS requires Transformers 4.26.1 while sentence-transformers requires at least 4.34.0. These need separate compatible environments or deliberate package updates.
- Defer decord 0.6.0 because its wheel metadata declares CPython 3.6 ABI, causing pip check to fail on Python 3.9. AV and torchvision video support are installed. clip-anytorch supplies the clip module, so do not install another overlapping clip distribution.

## Verification and editor integration

`pip check` reports no broken requirements. Imports passed for PyTorch/vision/audio, Transformers, Accelerate, Diffusers, timm, NumPy, pandas, SciPy, scikit-learn, OpenCV, Matplotlib, CLIP/open_clip, sentence-transformers, MMCV including native ops, MMDetection, MMPose, FAISS, HDF5, PyArrow, OpenAI, ipykernel and SHAP. Chumpy imports and arithmetic passed after patching.

CPU checks passed for native MMCV NMS, MMPose API import, ResNet18 forward, CLIP/ViT imports and chumpy arithmetic. CUDA is available and enumerates eight GPUs. No CUDA compute test or model download was performed because all GPUs had existing workloads; no training/inference quality or full project regression result is claimed. Some older packages emit pkg_resources deprecation warnings.

Registered Jupyter kernel `ad` with display name `Python (ad)`. Updated `python.defaultInterpreterPath` in `~/.vscode/settings.json` and `~/.vscode-server/data/Machine/settings.json`, preserving timestamped backups. Existing workspaces may retain an explicit selection; use Python: Select Interpreter to select the environment if needed.

Installation logs: `/tmp/ad-conda-create.log`, `/tmp/ad-install-torch.log`, `/tmp/ad-install-packages.log`, `/tmp/ad-install-cmake.log`.

The requested remote was fetched from `https://github.com/oxu2/safefix.git`. Its main branch has unrelated history to the existing master checkout, so it was not forcibly merged. Installation uses the user-provided snapshot and preserves the existing worktree.

Official binary references: [PyTorch version pairs](https://pytorch.org/get-started/previous-versions/), [MMCV CUDA 12.1 / torch 2.1 wheels](https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html).
