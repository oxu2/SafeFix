# Juno Quickstart For `NetID`

This note is a shareable quickstart for using Juno, syncing project code, and
checking H100 availability. Replace `NetID` with your Juno username.

## Juno Rules

- Keep repositories, metadata, and normal logs under `~/work`.
- Use `~/scratch` for heavy batch-job I/O. For large training jobs, copy data
  from scratch at the start of a SLURM job, run there, then write logs/results
  under `~/work`.
- Put long-lived conda environments under the PI group path, for example
  `/groups/yguo/NetID/conda/envs/<env-name>` if that group layout is available
  for your account.
- Run GPU work through SLURM. Do not start long GPU jobs directly with `nohup`
  from the login node.
- Current Juno GPU partitions observed with `sinfo`:
  - `h100`: physical H100, 80 GB or 94 GB VRAM.
  - `a30`: physical A30, 24 GB VRAM.
  - `a30-2.12gb`: half A30 virtual GPU, 12 GB VRAM.
  - `a30-4.6gb`: quarter A30 virtual GPU, 6 GB VRAM.

## Current H100 Layout

Snapshot from Juno at `2026-04-29 02:16 CDT`:

| Node | H100 Type | Total GPUs | Allocated GPUs | Nominal Free GPUs | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `g-04-02` | `nvidia_h100_80gb_hbm3` | `4` | `4` | `0` | 80 GB H100 node; fully allocated. |
| `g-05-01` | `nvidia_h100_nvl` | `1` | `1` | `0` | H100 NVL node; fully allocated. |
| `g-06-01` | `nvidia_h100_nvl` | `2` | `1` | `1` | SLURM shows one nominal free GPU, but allocated memory was `500G / 500G`, so a new job may still stay pending. |

Interpret the table as a scheduling hint, not a guarantee. A job can stay
`PENDING (Resources)` even when one GPU looks free if CPU, memory, node state,
priority, or reservations prevent placement.

Check the current H100 layout yourself:

```bash
sinfo -p h100 -N -o "%N %t %G %C %m"
squeue -p h100 -t RUNNING -o "%i %P %j %T %M %N %b"

for n in $(sinfo -p h100 -N -h -o "%N"); do
  echo "=== $n ==="
  scontrol show node "$n" | grep -E "NodeName=|Gres=|State=|CfgTRES=|AllocTRES="
done
```

## Scheduler Basics

The Juno orientation user limit should be treated as an upper bound, not a
concurrency guarantee. The reported policy is up to `4` running jobs and `8`
submitted jobs per user, but a job only starts when all scheduler checks pass:

1. `Resources`: a suitable GPU is physically available in the requested
   partition.
2. `Priority`: the job is high enough in the queue after fairshare, job age,
   and job size are considered.
3. `Limits`: the user is still under the running-job cap.

Useful queue checks:

```bash
sinfo -p h100,a30
squeue --me -o "%i %P %j %T %M %D %R"
sacct -j <jobid> --format=JobID,JobName%32,Partition,State,ExitCode,Elapsed,NodeList
```

Inspect why a pending job is not starting:

```bash
scontrol show job <jobid> | grep -E "JobId=|JobState=|Reason=|Partition=|NumNodes=|NumCPUs=|ReqTRES=|AllocTRES=|TRESPerNode=|MinMemory"
```

For independent single-GPU experiment sweeps:

- Keep H100-only or H100-sensitive jobs on `h100` when the workload needs H100
  memory or speed.
- Prefer smaller batches of submissions, for example `2` to `4` jobs at a
  time, instead of filling all `8` submitted slots when the queue is congested.
- Monitor the queue before and after submission.

## Recommended Layout

On Juno:

```bash
mkdir -p /work/NetID ~/work ~/scratch ~/work/data ~/work/artifacts
cd /work/NetID
git clone <repo-url> project
```

Common paths:

```text
/work/NetID/project
/work/NetID/data
/home/NetID/scratch
```

If you prefer home-relative paths:

```bash
mkdir -p ~/work ~/scratch ~/work/data ~/work/artifacts
cd ~/work
git clone <repo-url> project
```

## Sync Local Code To Juno

For normal committed code, use Git:

```bash
ssh juno 'cd /work/NetID/project && git pull --ff-only'
```

For local files that are not committed yet, use `rsync`.

One Python file:

```bash
rsync -av utils/my_new_eval.py \
  juno:/work/NetID/project/utils/my_new_eval.py
```

One SLURM script:

```bash
rsync -av scripts/my_job.sbatch \
  juno:/work/NetID/project/scripts/my_job.sbatch
```

One directory:

```bash
rsync -av scripts/ \
  juno:/work/NetID/project/scripts/
```

Broader code sync, excluding heavy generated outputs:

```bash
rsync -av \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'log_pose/' \
  --exclude 'runs/' \
  --exclude 'outputs/' \
  ./ juno:/work/NetID/project/
```

Verify after syncing:

```bash
ssh juno 'cd /work/NetID/project && ls -l utils/my_new_eval.py'
ssh juno 'cd /work/NetID/project && bash -n scripts/my_job.sbatch'
```

## Download Public Data On Juno

Use Juno-local downloads for public datasets instead of copying large datasets
through the login node.

Example COCO 2017 download:

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

Example Hugging Face dataset download:

```bash
ssh juno
module load miniconda
conda activate /groups/yguo/NetID/conda/envs/<env-name>
hf auth login
mkdir -p ~/work/data/<dataset-dir>
hf download <owner>/<dataset> \
  --repo-type dataset \
  --local-dir ~/work/data/<dataset-dir> \
  --max-workers 8
```

## Conda Environment

Create or activate a personal environment:

```bash
module load miniconda
conda create -p /groups/yguo/NetID/conda/envs/<env-name> python=3.10 -y
conda activate /groups/yguo/NetID/conda/envs/<env-name>
python -m pip install -U pip
```

Install PyTorch according to the CUDA version needed by your project. Then
install project packages from the repository requirements or setup file.

Quick validation:

```bash
python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_capability(0))
PY
```

The CUDA validation should be run inside a SLURM GPU allocation, not on the
login node.

## Submit A Job

Submit from the repo directory on Juno:

```bash
ssh juno
cd /work/NetID/project
sbatch scripts/my_job.sbatch
```

Minimal H100 job template:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=my-h100-job
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

module load miniconda
conda activate /groups/yguo/NetID/conda/envs/<env-name>

cd /work/NetID/project
nvidia-smi
python train.py
```

Monitor:

```bash
squeue -u "$USER" -o "%i %P %j %T %M %D %R"
tail -f slurm-<jobid>.out
scancel <jobid>
```

## Use `nvitop` Or `gpustat`

`nvitop` and `gpustat` must run on a GPU compute node. They will not work on a
login node if NVIDIA Management Library (NVML) is unavailable.

Install in your personal conda environment if needed:

```bash
python -m pip install -U nvitop gpustat
```

From a GPU node:

```bash
nvitop
gpustat --no-color
```

From the login node, if direct SSH to the compute node is allowed:

```bash
ssh g-04-02 nvitop
ssh g-04-02 'gpustat --no-color'
```

## Copy Back Selected Results

Prefer copying logs and small result files instead of entire output trees:

```bash
rsync -aP --include '*/' --include '*.log' --exclude '*' \
  juno:/work/NetID/project/logs/ \
  ./logs/
```

Copy a specific checkpoint only when it is needed:

```bash
rsync -aP \
  juno:/work/NetID/project/runs/<run-name>/weights/best.pt \
  runs/<run-name>/weights/best.pt
```

Do not copy editor folders, Python caches, `.pytest_cache/`, PID files, or
large generated outputs unless you explicitly need them.
