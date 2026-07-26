# Juno Migration Notes

This note records what must move from the UTD server to Juno for the current
SafeFix COCO pose experiments.

## Juno Rules Relevant To This Repo

- Keep the repository, metadata, and normal logs under `~/work`. Put the conda
  environment under the PI group path `/groups/yguo/oxx220000/conda/envs/ad`.
- Use `~/scratch` for heavy batch-job I/O. Copy COCO and generated images from
  scratch at the start of a SLURM job, run there, then write logs/results under
  `~/work/safefix/logs/pose/`.
- Run GPU work through SLURM, not direct `nohup` from the login node.
- Current Juno GPU partitions observed with `sinfo` on 2026-04-28:
  - `h100`: physical H100, 80 GB or 94 GB VRAM.
  - `a30`: physical A30, 24 GB VRAM.
  - `a30-2.12gb`: half A30 virtual GPU, 12 GB VRAM.
  - `a30-4.6gb`: quarter A30 virtual GPU, 6 GB VRAM.


## Juno Scheduler And Multi-Run Workflow

The Juno orientation user limit should be treated as an upper bound, not a
concurrency guarantee. The reported policy is up to `4` running jobs and `8`
submitted jobs per user, but a job only starts when all scheduler checks pass:

1. `Resources`: a suitable GPU is physically available in the requested
   partition.
2. `Priority`: the job is high enough in the queue after fairshare, job age,
   and job size are considered.
3. `Limits`: the user is still under the running-job cap.

Therefore, four independent `--gres=gpu:1` jobs may run at once only if
resources and priority also allow it. `PENDING (Resources)` means the requested
partition does not currently have a matching free GPU for that job. `PENDING
(Priority)` means the scheduler is holding the job behind higher-priority work,
even if the user is below the nominal job cap.

For independent single-GPU experiment sweeps:

- Default all experiment submissions in this repo to `h100`.
- Do not use A30 or virtual-A30 partitions by default. Use them only when the
  user explicitly asks for A30 or when a specific small probe has already shown
  that the workload fits the target A30 partition's GPU VRAM and host-memory
  profile. The old A30 probe job `185433` failed with CUDA OOM at
  `--batch-size 8`, so that specific batch size should not be reused on A30
  without rechecking memory.
- Prefer smaller batches of submissions, for example `2` to `4` jobs at a time,
  instead of filling all `8` submitted slots when the queue is congested.
- Monitor both partitions before and after submission, then move only jobs that
  are safe for the target GPU.

### Short PyTorch Sweep Policy From Juno

Sasmita's guidance on 2026-04-29 for short independent PyTorch sweeps:

- For a single-node, one-GPU case such as `8` independent experiments that each
  take about `10` to `30` minutes and fit in one H100's GPU memory, Juno
  recommends using a **Parallel Launcher**.
- For a multi-GPU case such as `8` independent experiments that should run
  concurrently across about `2` to `3` H100 GPUs, Juno also recommends using a
  **Launcher** rather than many separate manual submissions.
- The stated reason is HPC efficiency: Launcher is the preferred Juno workflow
  for handling multiple short independent experiments flexibly without filling
  the queue with many small jobs.
- Treat SLURM job arrays as a reasonable fallback only when Launcher is
  unavailable or when a specific experiment layout is easier to express as an
  array. For the cases above, Juno's direct recommendation is Launcher.
- Before using Launcher for real GPU sweeps, run a small GPU-node smoke test
  that confirms the Launcher job can see CUDA and can run a tiny PyTorch CUDA
  operation on the allocated GPU node.

Useful queue checks:

```bash
sinfo -p h100,a30
squeue --me -o "%i %P %j %T %M %D %R"
sacct -j <jobid> --format=JobID,JobName%32,Partition,State,ExitCode,Elapsed,NodeList
```

To check which H100 nodes still have unallocated GPUs, compare each node's
configured GPU TRES with allocated GPU TRES:

```bash
sinfo -p h100 -N -o "%N %t %G %C %m"
squeue -p h100 -t RUNNING -o "%i %u %P %j %T %M %N %b"
scontrol show node g-04-02 | egrep "NodeName=|Gres=|State=|CfgTRES=|AllocTRES="
scontrol show node g-05-01 | egrep "NodeName=|Gres=|State=|CfgTRES=|AllocTRES="
scontrol show node g-06-01 | egrep "NodeName=|Gres=|State=|CfgTRES=|AllocTRES="
```

Interpretation:

- `CfgTRES` is the node capacity, for example `gres/gpu=4`.
- `AllocTRES` is what SLURM has already allocated, for example
  `gres/gpu=3`.
- A node with `CfgTRES gres/gpu=4` and `AllocTRES gres/gpu=3` has one nominally
  unallocated GPU.
- A nominally unallocated GPU may still not be schedulable for a new job if CPU,
  host memory, priority, or planned allocations block it. Check `AllocTRES`
  memory and the node `State` as well as GPU count. Treat old `AllocTRES`
  memory observations as historical only; the live node state is what matters.

To inspect why a pending job is not starting and what it requested:

```bash
scontrol show job <jobid> | egrep "JobId=|JobState=|Reason=|Partition=|NumNodes=|NumCPUs=|ReqTRES=|AllocTRES=|TRESPerNode=|MinMemory"
```

## Current Run Dependencies Not Covered By Git

The following are required beyond a normal `git clone`:

### CelebA `alpred.py` Resume Runs

For CelebA resume-from-base active-learning runs with `alpred.py`, the current
Juno state checked on 2026-04-29 is:

- Full CelebA source images are copied and verified:
  - path:
    `/home/oxx220000/scratch/safefix_cache/data/img_align_celeba`
  - file count: `202599`
  - size: `1.7G`
- CelebA red-hair brown-skin sad-emotion instruct-pix2pix seed-42 1k generated
  subset is copied and verified:
  - image path:
    `/home/oxx220000/scratch/safefix_cache/data/redhair_brownskin_sademotion_instruct_pix2pix_seed42_1k`
  - manifest:
    `/home/oxx220000/scratch/safefix_cache/data/redhair_brownskin_sademotion_instruct_pix2pix_seed42_1k_files.txt`
  - image count: `1000`
  - manifest lines: `1000`
- HiBUG red-hair brown-skin seed-42 1k generated subset is also copied and
  verified:
  - path:
    `/home/oxx220000/scratch/safefix_cache/data/hibug_redhair_brownskin_seed42_1k`
  - image count: `1000`
- Repo-local CelebA metadata needed by `alpred.py` has been copied to Juno:
  - path: `/home/oxx220000/work/safefix/data/celebA`
  - size: about `101M`
  - includes at least:
    `list_attr_celeba.txt`, `train_idx.npy`, `labels.npy`,
    `predictions.npy`, and the attribute index text files.
- Qwen3 CelebA attribute metadata is present:
  - path:
    `/home/oxx220000/work/safefix/save/attribute_celeba_trainval_qwen3_8B.json`
  - SHA-256:
    `152d2d415ac8c5de12bd64f6cae4515b77a1e9ece158604d12068b4baa6af6ab`
- The CelebA resume checkpoint is present:
  - path:
    `/home/oxx220000/work/safefix/save/weights/model_celeba_resnet_base.pth`
  - size: `2.7M`
  - SHA-256:
    `dafdd2481587604df6e32f614ad9c42df8c165c6a6bce9e568bddd6814916dc0`

Important path caveat:

- The generated CelebA subset is in Juno scratch, not under
  `/home/oxx220000/work/safefix/data/`.
- `alpred.py` can discover the scratch generated directory by default when
  `--gen_image_dir` is omitted, but local queue scripts that hardcode
  `GEN_DIR="data/redhair_brownskin_sademotion_instruct_pix2pix_seed42_1k"`
  will fail on Juno unless that path is overridden, patched to the scratch
  path, or replaced with a Juno-specific launcher.
- For direct Juno runs, use:

```bash
--data_dir /home/oxx220000/scratch/safefix_cache/data/img_align_celeba \
--gen_image_dir /home/oxx220000/scratch/safefix_cache/data/redhair_brownskin_sademotion_instruct_pix2pix_seed42_1k \
--resume_model save/weights/model_celeba_resnet_base.pth \
--attribute_file save/attribute_celeba_trainval_qwen3_8B.json
```

The original base-run log is not currently present on Juno. This is not a
blocker for loading `save/weights/model_celeba_resnet_base.pth`; it only means helper
code may not recover the base test accuracy from a local log.

One-epoch GPU smoke test:

- Script:
  `scripts/juno_celeba_instructpix2pix_smoke.sbatch`
- Purpose:
  verify that CelebA `alpred.py` resume training works on a Juno H100 while
  using the scratch cached
  `redhair_brownskin_sademotion_instruct_pix2pix_seed42_1k` generated subset.
- The script stages a runnable repo copy under
  `$HOME/scratch/safefix_save/runs/celeba_ip2p_smoke_${SLURM_JOB_ID}_${TIMESTAMP}`
  and runs from that scratch copy so run-time `save/` and `logs/celeba/` writes
  happen under scratch.
- Submitted on 2026-04-29 as SLURM job `185848`; initial state was
  `PENDING (Resources)` on `h100`.

### COCO Pose Runs

The following are required beyond a normal `git clone`:

1. COCO 2017 data:
   - Juno path: `~/work/data/coco`
   - Download on Juno from the official COCO URLs in `DATASET_DOWNLOAD_REPRO.md`.
   - Size observed on UTD: about `39G`
   - Required structure:
     - `train2017/`
     - `val2017/`
     - `annotations/person_keypoints_train2017.json`
     - `annotations/person_keypoints_val2017.json`

2. COCO pose generated rainy images:
   - Juno path:
     `~/work/data/coco-pose-weather-edits/generated_coco_train1500_rain`
   - Download on Juno from Hugging Face dataset
     `BZNR3/coco-pose-weather-edits`.
   - Size observed on UTD: about `1.0G`
   - Used by `pose_estimation/train_pose_estimation.py` when `--attribute rainy` is set.

3. COCO pose generated brown-rainy images:
   - Juno path:
     `~/work/data/coco-pose-weather-edits/generated_coco_qwen3_1500_brown_rain_icedit`
   - Download on Juno from Hugging Face dataset
     `BZNR3/coco-pose-weather-edits`.
   - Current Juno check: `1500` files, about `58M`.
   - Current experiment policy: keep this full directory intact, but append
     only `1000` generated images per run, sampled deterministically with
     `--seed 42`.
   - Used by `pose_estimation/train_pose_estimation.py` when `--attribute brown_rainy` is set.
   - Attribute selection for this setting uses exact metadata keys and values:
     - `clothes_color=brown`
     - `weather=rainy`

4. InternVL3 pose attribute metadata:
   - `save/attribute_coco_pose_gemini_train20k_internvl3_8B.json`
   - Also keep the paired CSV:
     `save/attribute_coco_pose_gemini_train20k_internvl3_8B.csv`
   - These are under `save/`, which is mostly gitignored except top-level CSV
     and JSON files. Verify they exist after transfer.

5. Qwen3 pose attribute metadata for brown-rainy selection:
   - `save/attribute_coco_pose_gemini_train20k_qwen3_8B.json`
   - Also keep the paired CSV:
     `save/attribute_coco_pose_gemini_train20k_qwen3_8B.csv`
   - The train20k JSON contains the relevant values:
     - `clothes_color`: includes `brown`
     - `weather`: includes `rainy`

6. Resume checkpoint, only if you want to resume from the current UTD run:
   - `logs/pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530/best.pth`
   - `.gitignore` ignores `*.pth` and most of `logs/pose/`, so this checkpoint
     will not move with Git.
   - Current migration policy: do not copy `logs/pose` output folders or
     checkpoints by default. If the checkpoint is not copied, pass another
     `--base-model` path on Juno or rerun the base model there first.

7. Existing experiment logs:
   - Top-level `logs/pose/*.log` are unignored by `.gitignore`, but nested
     output directories under `logs/pose/pose_keypointrcnn_*` are ignored.
   - Copy only log files for this migration. Do not copy nested output folders,
     PID files, checkpoints, metrics JSONs, or predictions unless a specific run
     must be resumed or inspected.

Do not copy `.opencode/`, `.vscode/`, `__pycache__/`, `.pytest_cache/`, or
PID files. They are local tooling or stale process state.

## Recommended Juno Layout

On Juno:

```bash
mkdir -p ~/work ~/scratch ~/work/data ~/work/artifacts
cd ~/work
git clone <repo-url> safefix
```

Place the non-git data here:

```text
~/work/data/coco
~/work/data/coco-pose-weather-edits/generated_coco_train1500_rain
~/work/data/coco-pose-weather-edits/generated_coco_qwen3_1500_brown_rain_icedit
~/work/safefix/save/attribute_coco_pose_gemini_train20k_internvl3_8B.json
~/work/safefix/save/attribute_coco_pose_gemini_train20k_qwen3_8B.json
```

## Transfer Commands From UTD Server

Code should move by Git when possible:

```bash
ssh juno 'cd ~/work/safefix && git pull --ff-only'
```

For ad hoc local edits that are not committed yet, sync only the changed file or
directory with `rsync`. Prefer `~/work/safefix` in examples so the command is
portable across Juno users. If a full path is needed, replace `<netid>` with the
target Juno username, for example `/work/<netid>/safefix`.

One Python file:

```bash
rsync -av utils/my_new_eval.py \
  juno:~/work/safefix/utils/my_new_eval.py
```

One script:

```bash
rsync -av scripts/my_job.sbatch \
  juno:~/work/safefix/scripts/my_job.sbatch
```

One directory, preserving relative contents:

```bash
rsync -av scripts/ \
  juno:~/work/safefix/scripts/
```

For a broader code sync, exclude heavy outputs and generated artifacts:

```bash
rsync -av \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'logs/pose/' \
  --exclude 'save/runs/' \
  --exclude 'outputs/' \
  ./ juno:~/work/safefix/
```

After syncing, verify on Juno before submitting:

```bash
ssh juno 'cd ~/work/safefix && ls -l utils/my_new_eval.py'
ssh juno 'cd ~/work/safefix && bash -n scripts/my_job.sbatch'
```

Use Juno-local downloads for public datasets. COCO:

```bash
ssh juno 'mkdir -p ~/work/data/coco && cd ~/work/data/coco && nohup bash -lc '"'"'
wget -c --no-check-certificate \
  https://images.cocodataset.org/zips/train2017.zip \
  https://images.cocodataset.org/zips/val2017.zip \
  https://images.cocodataset.org/zips/test2017.zip \
  https://images.cocodataset.org/annotations/annotations_trainval2017.zip \
  https://images.cocodataset.org/annotations/image_info_test2017.zip &&
unzip -q -n train2017.zip &&
unzip -q -n val2017.zip &&
unzip -q -n test2017.zip &&
unzip -q -n annotations_trainval2017.zip &&
unzip -q -n image_info_test2017.zip
'"'"' > download_coco2017_juno.log 2>&1 & echo $!'
```

Generated rainy images from Hugging Face:

```bash
ssh juno
module load miniconda
conda activate /groups/yguo/oxx220000/conda/envs/ad
hf auth login
mkdir -p ~/work/data/coco-pose-weather-edits
hf download BZNR3/coco-pose-weather-edits \
  --repo-type dataset \
  --local-dir ~/work/data/coco-pose-weather-edits \
  --max-workers 8
```

The brown-rainy generated image directory should then exist at:

```bash
ls ~/work/data/coco-pose-weather-edits/generated_coco_qwen3_1500_brown_rain_icedit
find ~/work/data/coco-pose-weather-edits/generated_coco_qwen3_1500_brown_rain_icedit -maxdepth 1 -type f | wc -l
```

Use rsync only for selected logs:

```bash
rsync -aP --include '*/' --include '*.log' --exclude '*' \
  /people/cs/o/oxx220000/safefix/logs/pose/ \
  juno:~/work/safefix/logs/pose/
```

If you later decide that the current resume checkpoint is required, copy only
that checkpoint, not the full `logs/pose` output tree:

```bash
rsync -aP \
  /people/cs/o/oxx220000/safefix/logs/pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530/best.pth \
  juno:~/work/safefix/logs/pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530/best.pth
```

## Conda Environment

The current UTD server environment is:

- Python `3.9.20`
- PyTorch `1.13.1+cu117`
- torchvision `0.14.1+cu117`

For ongoing Juno work, prefer the clean SafeFix environment:

```text
/groups/yguo/oxx220000/conda/envs/sf
```

Do not use `~/` for the main shared training environment. A home-directory
environment is acceptable for small personal tests, but the group path is better
for SLURM jobs, quota stability, and later reproduction notes. See
`docs/sf_environment.md` for the exact package list, recreation commands, and
H100 validation.

Historical `ad` setup:

```bash
module load miniconda
conda create -p /groups/yguo/oxx220000/conda/envs/ad python=3.9 -y
conda activate /groups/yguo/oxx220000/conda/envs/ad
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117
pip install numpy==1.26.4 pillow pycocotools scipy scikit-learn matplotlib tqdm
```

Current Juno `ad` state after the April 28 sync:

- Environment path:
  `/groups/yguo/oxx220000/conda/envs/ad`
- Current H100-ready GPU stack after the April 28 upgrade:
  - `torch==2.4.1+cu121`
  - `torchvision==0.19.1+cu121`
  - `torchaudio==2.4.1+cu121`
  - validation H100 smoke job reported:
    - CUDA available: `True`
    - device: `NVIDIA H100 80GB HBM3`
    - capability: `(9, 0)`
- Added / aligned safefix runtime packages for the current pose workflow:
  - `transformers==4.26.1`
  - `tokenizers==0.13.3`
  - `huggingface-hub==0.26.2`
  - `safetensors==0.4.5`
  - `accelerate==1.1.1`
  - `timm==0.4.12`
  - `pandas==1.5.3`
  - `opencv-python-headless==4.7.0.72`
  - `opencv-python==4.10.0.84`
  - `openai==1.55.0`
  - `diffusers==0.31.0`
  - `qwen-vl-utils==0.0.10`
  - `lpips==0.1.4`
  - `pytorch-fid==0.3.0`
  - `wandb==0.12.21`
  - `tensorboard==2.18.0`
  - `scikit-image==0.24.0`
  - `albumentations==1.4.0`
  - `pycocoevalcap==1.2`
  - `xtcocotools==1.14.3`
  - `json-tricks==3.17.3`
  - `mmengine==0.10.7`
- Validation commands that passed:

```bash
cd ~/work/safefix
/groups/yguo/oxx220000/conda/envs/ad/bin/python -c 'from utils.imports import *; print("utils.imports ok")'
/groups/yguo/oxx220000/conda/envs/ad/bin/python -m py_compile pose_estimation/train_pose_estimation.py utils/imports.py utils/active_learning.py
/groups/yguo/oxx220000/conda/envs/ad/bin/python pose_estimation/train_pose_estimation.py train --help
bash scripts/juno_pose_brown_rainy_attribute.sbatch --ratio 0.05 --dry-run --no-scratch-copy
```

This is intentionally close to the active local environment for the safefix
pose workflow, but it is not a byte-for-byte clone. Nonportable local editable,
file, and git installs were skipped. `sentence-transformers==2.7.0` was also
skipped because it conflicts with the active local `transformers==4.26.1`
version.

The current `ad` package snapshots are stored in:

- `docs/juno_ad_current_pip_freeze.txt`
- `docs/juno_ad_current_conda_export.txt`

If exact reproduction becomes necessary, install `conda-pack` on the UTD server
and pack the current environment:

```bash
conda install -n base -c conda-forge conda-pack
conda-pack -p /people/cs/o/oxx220000/.conda/envs/ad -o ad-conda-env.tar.gz
rsync -aP ad-conda-env.tar.gz <netid>@juno.utdallas.edu:~/work/
```

Then unpack on Juno:

```bash
mkdir -p /groups/yguo/oxx220000/conda/envs/ad
tar -xzf ~/work/ad-conda-env.tar.gz -C /groups/yguo/oxx220000/conda/envs/ad
/groups/yguo/oxx220000/conda/envs/ad/bin/conda-unpack
```

Packing is more faithful, but rebuilding is cleaner if Juno compilers/CUDA
libraries differ.

## Submit The Current Pose Queue On Juno

From `~/work/safefix` on Juno:

```bash
sbatch scripts/juno_pose_resume.sbatch \
  --queue ratio0.9 \
  --base-model /path/to/base_or_resume_checkpoint.pth \
  --start-at attrx111_r0001_p01_c15
```

For the unfinished ratio `0.05` refinement queue:

```bash
sbatch scripts/juno_pose_resume.sbatch \
  --queue ratio0.05 \
  --base-model /path/to/base_or_resume_checkpoint.pth \
  --start-at attrx20_r000_p0_c8
```

For the brown-rainy attribute-selection run, use the dedicated wrapper. It
selects original images with `clothes_color,weather` focus and accounts for the
generated set as `clothes_color:brown` plus `weather:rainy`:

```bash
sbatch scripts/juno_pose_brown_rainy_attribute.sbatch \
  --base-model ~/work/safefix/logs/pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530/best.pth
```

If the Juno scratch cache already exists, prefer reading it directly and do not
copy the dataset again for each job. The cache is currently:

```text
~/scratch/safefix_cache/data/coco
~/scratch/safefix_cache/data/coco-pose-weather-edits/generated_coco_qwen3_1500_brown_rain_icedit
~/scratch/safefix_cache/logs/pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530/best.pth
```

Use `--no-scratch-copy` with those paths:

```bash
sbatch scripts/juno_pose_brown_rainy_attribute.sbatch \
  --ratio 0.1 \
  --data-root ~/scratch/safefix_cache/data/coco \
  --generated-dir ~/scratch/safefix_cache/data/coco-pose-weather-edits/generated_coco_qwen3_1500_brown_rain_icedit \
  --base-model ~/scratch/safefix_cache/logs/pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530/best.pth \
  --no-scratch-copy
```

For the remaining `ratio=0.3` and `ratio=0.5` brown-rainy RPC grid, prefer one
sequential H100 allocation instead of several independent pending jobs:

```bash
sbatch scripts/juno_pose_brown_rainy_rpc03_05_sweep.sbatch
```

The sequential wrapper runs these seven missing settings in order. The
`ratio=0.3`, `R=0.0`, `P=0.0`, `C=15.0` row already completed as Juno job
`185438`, so it is not rerun by default.

| Ratio | R | P | C |
| --- | --- | --- | --- |
| `0.3` | `0.01` | `0.5` | `15.0` |
| `0.3` | `0.03` | `0.5` | `15.0` |
| `0.3` | `0.03` | `0.5` | `8.0` |
| `0.5` | `0.0` | `0.0` | `15.0` |
| `0.5` | `0.01` | `0.5` | `15.0` |
| `0.5` | `0.03` | `0.5` | `15.0` |
| `0.5` | `0.03` | `0.5` | `8.0` |

The resolved defaults are:

```text
data root:       ~/work/data/coco
generated dir:   ~/work/data/coco-pose-weather-edits/generated_coco_qwen3_1500_brown_rain_icedit
attribute file:  ~/work/safefix/save/attribute_coco_pose_gemini_train20k_qwen3_8B.json
focus:           clothes_color,weather
extra counts:    clothes_color:brown=1000,weather:rainy=1000
generated attr:  brown_rainy
sub-ratio:       0.9
num added img:   1000
```

Useful SLURM commands:

```bash
squeue --me
scancel <jobid>
tail -f slurm-<jobid>.out
```

The wrapper defaults are Juno paths, but can be overridden:

```bash
sbatch scripts/juno_pose_resume.sbatch \
  --queue ratio0.9 \
  --conda-env /groups/yguo/oxx220000/conda/envs/ad \
  --data-root ~/work/data/coco \
  --generated-dir ~/work/data/coco-pose-weather-edits/generated_coco_train1500_rain \
  --partition h100 \
  --batch-size 8
```
