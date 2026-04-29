# Juno Quickstart For `cxz220015`



This note is a shareable, user-specific quickstart for syncing local project
code to Juno and checking GPU usage. It avoids private paths from other users.


## Juno Rules Relevant To This Repo

- Keep the repository, metadata, and normal logs under `~/work`. Put the conda
  environment under the PI group path `/groups/你老板的netid或者姓`.
- Use `~/scratch` for heavy batch-job I/O. Copy COCO and generated images from
  scratch at the start of a SLURM job, run there, then write logs/results under
  `~/work/your_username/`.
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

- Keep H100-only or H100-sensitive jobs on `h100` when the workload needs H100
  memory or speed.
- For COCO training and COCO pose resume jobs in this repo, use `h100` only.
  Do not plan COCO runs on `a30` by default because memory is the limiting
  resource. The A30 probe job `185433` failed with CUDA OOM at
  `--batch-size 8`, and the current project policy is to reserve A30 only for
  smaller non-COCO work or explicit memory-fit probes.
- Prefer smaller batches of submissions, for example `2` to `4` jobs at a time,
  instead of filling all `8` submitted slots when the queue is congested.
- Monitor both partitions before and after submission, then move only jobs that
  are safe for the target GPU.

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
  memory, priority, or planned allocations block it. Check `AllocTRES` memory
  and the node `State` as well as GPU count.

To inspect why a pending job is not starting and what it requested:

```bash
scontrol show job <jobid> | egrep "JobId=|JobState=|Reason=|Partition=|NumNodes=|NumCPUs=|ReqTRES=|AllocTRES=|TRESPerNode=|MinMemory"
```

## Assumed Juno Layout

Use this layout on Juno:

```bash
mkdir -p /work/cxz220015
cd /work/cxz220015
git clone <repo-url> project
```

Repository path:

```text
/work/cxz220015/project
```

Optional data and scratch paths:

```text
/work/cxz220015/data
/home/cxz220015/scratch
```

## Sync Local Code To Juno

For normal committed code, use Git:

```bash
ssh juno 'cd /work/cxz220015/project && git pull --ff-only'
```

For local files that are not committed yet, use `rsync`.

One Python file:

```bash
rsync -av utils/my_new_eval.py \
  juno:/work/cxz220015/project/utils/my_new_eval.py
```

One SLURM script:

```bash
rsync -av scripts/my_job.sbatch \
  juno:/work/cxz220015/project/scripts/my_job.sbatch
```

One directory:

```bash
rsync -av scripts/ \
  juno:/work/cxz220015/project/scripts/
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
  ./ juno:/work/cxz220015/project/
```

Verify after syncing:

```bash
ssh juno 'cd /work/cxz220015/project && ls -l utils/my_new_eval.py'
ssh juno 'cd /work/cxz220015/project && bash -n scripts/my_job.sbatch'
```

## Check SLURM Queue And GPU Availability

User queue:

```bash
squeue -u "$USER" -o "%i %P %j %T %M %D %R"
```

GPU node allocation:

```bash
for n in $(sinfo -p h100,a30 -N -h -o "%N"); do
  echo "=== $n ==="
  scontrol show node "$n" | grep -E "NodeName=|CfgTRES=|AllocTRES="
done
```

Interpretation:

- `CfgTRES gres/gpu=N` is total GPU count on the node.
- `AllocTRES gres/gpu=M` is allocated GPU count.
- Nominal free GPUs are approximately `N - M`.
- Also check `mem=` in `AllocTRES`; a GPU can be nominally free while the node
  memory is too full for a new job.

Pending job detail:

```bash
scontrol show job <jobid> | grep -E "JobId=|JobState=|Reason=|Partition=|ReqTRES=|AllocTRES=|TRESPerNode=|MinMemory"
```

## Use `nvitop` Or `gpustat`

`nvitop` and `gpustat` must run on a GPU compute node. They will not work on a
login node if NVIDIA Management Library (NVML) is unavailable.

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

If these commands are missing in the active environment, install them in a
personal conda environment:

```bash
python -m pip install -U nvitop gpustat
```

## Submit A Job

Submit from the repo directory on Juno:

```bash
ssh juno
cd /work/cxz220015/project
sbatch scripts/my_job.sbatch
```

Monitor:

```bash
squeue -u "$USER" -o "%i %P %j %T %M %D %R"
tail -f slurm-<jobid>.out
```
