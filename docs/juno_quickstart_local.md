# Juno Quickstart

This file is the canonical Juno workflow note for Ouyang Xu. It consolidates
the reusable and project-specific guidance previously kept in AtlasRepair and
SafeFix.

## Authority And Update Policy

- Use the [official Juno User Guide](https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/)
  as the primary source for cluster-wide user policy and supported workflows.
  This note was reconciled against the relevant operational guide pages on
  August 10, 2026. The orientation slides remain a useful secondary overview.
- Read this file for all Juno storage, environment, sync, SLURM, queueing,
  notification, monitoring, service, backup, and artifact workflows.
- Update this file when any reusable Juno rule or project-specific Juno profile
  changes. Do not maintain separate operational copies in individual repos.
- The AtlasRepair and SafeFix repo-local `docs/juno_quickstart.md` files are
  convenience symlinks to this file and are not independent sources of truth.
- Keep changing experiment state in the relevant repo journal, task record,
  `.handoff.md`, or `report.md`, not here. Examples include job IDs, current
  queue positions, run metrics, temporary blockers, and one-off output paths.
- Replace `NetID`, `<repo-name>`, and other placeholders before running shared
  examples.
- Treat live `scontrol`, `sacctmgr`, `sinfo`, `module avail`, and quota output
  as authoritative when they disagree with a static guide page.

## Core Rules

- For AtlasRepair production datasets, separate T2I generation from VLM
  evaluation. Generation arrays must not start a VLM or use VLM outcomes for
  traversal/pruning. Start a separate evaluator array only after all generation
  tasks and image/manifest/checksum coverage gates pass. Run HSE or other
  adaptive-pruning replay from the complete evaluated fixed-K artifact.
- Keep repositories, metadata, normal logs, durable model caches, and final
  artifacts under `/work/NetID` or `~/work`.
- Use `~/scratch` for heavy batch-job I/O, decoded datasets, generated images,
  and other large or repeatedly accessed intermediates.
- For assets around 1 GB or larger, keep a durable source or manifest in
  `/work` when needed, but point active jobs at a scratch mirror when this
  materially reduces I/O cost.
- Run GPU workloads and long-running services through SLURM. Do not run them
  directly on a login node with `nohup`.
- Request memory explicitly. Juno applies a 64 GB default job memory limit when
  neither `--mem` nor `--mem-per-cpu` is supplied.
- Default new GPU experiments to `h200`. Use `h100` as the secondary fallback
  or when explicitly requested. Do not default to A30 or virtual-A30
  partitions.
- If the user explicitly specifies a GPU or partition, launch on that target
  without first checking availability unless the user asks for a check.
- Prefer one `sbatch` script and one throttled SLURM array for shards that share
  the same code, environment, model, and resource shape.
- For multiple independent one-GPU tasks, use one array whose tasks each
  request `--gres=gpu:1`. For this user's Juno GPU workloads, default to
  `--qos=juno-pri` and `ArrayTaskThrottle=8` (for example,
  `--array=0-7%8`). Recheck the live association and QOS limits before each
  submission; fall back to the normal QOS and `%4` when `juno-pri` is not
  available. Do not impose a lower default such as `%1` or `%2` unless the
  user or shared-state safety requires it; let the QOS enforce its
  account-wide concurrent-job limit.
  Array concurrency controls how many independent one-GPU jobs run at once; it
  does not make one training run use multiple GPUs.
- The live `juno-pri` QOS currently reports `MaxJobsPU=8` and
  `MaxSubmitPU=150`. Its availability and limits are cluster-controlled, so
  a current `sacctmgr show qos`/association check takes precedence over this
  remembered snapshot.
- Every experiment must use the local workstation watcher
  `scripts/job_notifier.py slurm --host juno <jobid>` when the repo provides
  this helper. The watcher queries Juno through SSH but sends Slack requests
  locally, so Juno does not need webhook or external-network access.
- Merge-only and other post-processing jobs are exempt from this watcher rule;
  do not start Slack notifications for them. The exemption does not apply to
  the generation/evaluation array whose outputs are later merged.
- Routine successful-completion notifications are silent by default. The
  array watcher should report live count changes such as fewer `PENDING` tasks,
  while the in-job trap and terminal watcher should send notifications only
  for abnormal states such as `FAILED`, `CANCELLED`, `TIMEOUT`,
  `OUT_OF_MEMORY`, `NODE_FAIL`, or `PREEMPTED`. Opt in to a success message
  only when a particular workflow explicitly needs it.
- Download public datasets and model assets directly on Juno when practical.
- Never put webhook URLs or other secrets in tracked scripts, docs, journals,
  logs, or command histories.

### Cross-account H200 execution

- For this user's Juno H200 work, prefer submitting through `bxz200006` when
  the live association, QoS, ACLs, storage, and local notification path have
  been verified; fall back to `oxx220000` when any of those checks fail. This
  is a scheduling preference, not a guarantee that `bxz200006` will start
  sooner: compare live `squeue`, `sprio`, fair-share, reservations, and
  resource availability before switching.
- Independent arrays may run under both accounts at the same time when they
  do not share writers and the live account limits permit it. Do not run two
  accounts against the same shard, JSONL, checkpoint, manifest, or merge
  directory concurrently. Preserve run identity, shard index/count, and all
  linked artifacts when moving or resuming work across accounts.
- Keep the `oxx220000` repository, source artifacts, configs, and any
  intentionally shared model cache readable by `bxz200006` through explicit
  ACLs. If the scratch filesystem rejects user ACLs, use an account-local HF
  cache instead of broadening scratch permissions. Give `bxz200006` write
  permission only on the output/run roots that the submitted job owns; do not
  make the entire `/work/oxx220000` tree or secrets writable. Verify ACLs and
  read/write access before each cross-account submission.

### Comma-valued submission environment

1. **Scope / trigger.** This applies whenever `sbatch --export` would carry a
   list value containing commas, such as `SEEDS=0,1,2,3`. SLURM uses commas to
   delimit the `--export` entries themselves.
2. **Signature.** Prefer `sbatch --export=ALL,KEY=value script.sbatch` only
   when `value` contains no commas. For a fixed list, omit the key and let the
   tracked wrapper supply its default. Otherwise encode the list with a
   delimiter that is not part of the `--export` grammar and decode it inside
   the wrapper.
3. **Contract.** A list-valued wrapper variable must arrive as one value, and
   the job must record or validate the resulting element count before calling
   the experiment CLI. Do not infer correctness from an accepted `sbatch`
   command.
4. **Validation / error matrix.** If a submitted list is truncated, cancel
   the affected new jobs before material computation, preserve completed
   upstream artifacts, and resubmit with the wrapper default or safe encoding.
   If a terminal artifact records the wrong list length, treat that artifact
   as an incomplete diagnostic rather than production evidence.
5. **Good / base / bad cases.** Good: omit `REPLAY_SEEDS` and use the tracked
   `0--19` wrapper default. Base: export a scalar such as
   `EXPECTED_NODES=18854`. Bad: pass
   `--export=ALL,REPLAY_SEEDS=0,1,...,19`, which sets the intended value to
   only `0` under the comma-delimited grammar.
6. **Tests required.** Run `bash -n` on the wrapper, verify its default list,
   and inspect the first completed artifact or manifest for the exact element
   count before accepting the full run.
7. **Wrong vs correct.** Wrong:
   `sbatch --export=ALL,REPLAY_SEEDS=0,1,2 job.sbatch`. Correct:
   `sbatch --export=ALL job.sbatch` when `job.sbatch` owns the reviewed seed
   default.

## Live Cluster Checks

Partition specifications, availability, limits, and priority are dynamic.
Recheck them rather than treating an old snapshot as current:

```bash
sinfo -p h200,h100,a30 -N -o "%N %P %t %G %C %m"
squeue --me -o "%i %P %j %T %M %D %R"
scontrol show partition h200
sacctmgr show qos format=Name,MaxTRES,MaxJobsPU,MaxSubmitPU
sacctmgr show qos juno-pri \
  format=Name,Priority,MaxTRES,MaxJobsPU,MaxSubmitPU

for node in $(sinfo -p h200,h100 -N -h -o "%N" | sort -u); do
  echo "=== ${node} ==="
  scontrol show node "${node}" | \
    grep -E "NodeName=|Gres=|State=|CfgTRES=|AllocTRES="
done

mfsgetquota -H ~
mfsgetquota -H ~/work
module avail miniconda apptainer cuda
```

Observed hardware profiles include physical H200 NVL GPUs with 141 GB VRAM,
physical H100 GPUs with 80 GB or 94 GB VRAM, physical A30 GPUs with 24 GB
VRAM, and virtual H100/A30 slices. Treat these as orientation only and verify
with `sinfo` and `scontrol`.

The [Juno Orientation Summer 2026 v14](https://bpb-us-e2.wpmucdn.com/sites.utdallas.edu/dist/4/1507/files/2026/07/3_Juno_Orientation_v14.pdf)
describes 26 H200 nodes with two physical H200 NVL GPUs per node and a default
limit of four running jobs per user. Live checks on July 27, 2026 confirmed 52
H200 GPUs, `node=8`, `MaxJobsPU=4`, and `MaxSubmitPU=100` under the `juno` QoS.
The FAQ currently shows an older eight-submitted-job value, so use
`sacctmgr show qos` for the active limit. More than four concurrent jobs
normally requires HPC staff to raise the limit temporarily for a workload
that shows efficient scaling.

### Current `juno-pri` default and concurrency scaling

For this user's throughput-sensitive Juno GPU workloads, `juno-pri` is the
default QOS when a live association check confirms that it is usable. The
current QOS reports `MaxJobsPU=8`, `MaxSubmitPU=150`, `Priority=200000`, and
`OverPartQOS`; use `%8` for independent arrays by default.

When scaling beyond 8 concurrent GPU tasks:

1. **Dual-account priority scaling (recommended)**: Submit independent array
   shards across `bxz200006` and `oxx220000` with `--qos=juno-pri --array=...%8`
   on each account. This achieves up to 16 concurrent GPU tasks (8 + 8) while
   retaining the high `Priority=200000` queue status.
2. **Opportunistic `normal` QoS backfill**: In `sacctmgr show qos`, the
   `normal` QOS has no hard `MaxJobsPU` limit set. When `h200` has surplus idle
   nodes and no higher-priority jobs competing, submitting arrays under the
   default `normal` QOS with `%12` or `%16` allows Slurm to backfill and run
   10 to 16 tasks concurrently. However, `normal` has `Priority=0`, meaning
   it will yield immediately to priority workloads during congestion.
3. **Avoid `large` QoS for GPU workloads**: Although `large` reports
   `MaxJobsPU=150`, its `Priority=0` offers no scheduling advantage over
   `normal`. Historical accounting data confirms that `large` is intended for
   multi-node CPU/MPI jobs rather than GPU workloads. Use `normal` or
   `juno-pri` instead.

Despite the product name, Juno's H200 NVL cards have no NVLink bridges. The
two GPUs in each H200 node communicate over PCIe Gen 5.0 across CPU sockets;
H200 nodes communicate over 400 Gb InfiniBand. Only H100 node `g-04-02` is
documented as having NVLink. Verify topology inside the allocation with
`nvidia-smi topo -m` before tuning a multi-GPU workload.

## Recommended Layout

```bash
mkdir -p /work/NetID ~/work ~/scratch \
  /work/NetID/data /work/NetID/artifacts /work/NetID/backups \
  /work/NetID/venvs /work/NetID/hf_cache

cd /work/NetID
git clone <repo-url> <repo-name>
```

Common paths:

```text
/work/NetID/<repo-name>
/work/NetID/data
/work/NetID/artifacts
/work/NetID/backups
/work/NetID/venvs
/work/NetID/hf_cache
/home/NetID/scratch
```

Check quota before large transfers:

```bash
mfsgetquota -H ~
mfsgetquota -H ~/work
du -sh "$HOME/scratch"
```

## Storage And Cache Policy

- Home (`~`) is backed up daily and has a documented 50 GB quota. Keep shell
  configuration, SSH keys, and small files there.
- Work (`~/work`, normally `/work/NetID`) is backed up daily and has a
  documented 1 TB quota. Keep active repositories, environments, durable
  datasets, checkpoints, logs, and results there.
- Group storage (`/groups/<pi-name>`) is backed up daily, shared between Juno
  and Ganymede 2, and starts around 1 TB with increases available by request.
- Store durable code, checkpoints, logs, manifests, and result summaries in
  `/work` or `~/work`.
- Store temporary generated images, extracted datasets, and high-I/O run state
  in `~/scratch`.
- Scratch has a documented 30 TB soft limit, a seven-day grace period after
  exceeding it, no backups or recovery, and automatic purge for files not
  accessed in 30 days. Juno HPC Support confirmed in Incident `815829` on
  August 10, 2026 that the inactivity purge is active, access-time based,
  unrecoverable, and has no advance warning or grace period. A later HPC notice
  changed the retention period from 45 to 30 days, effective with the September
  data purge cycle during the last week of the month. The seven-day grace period
  above applies only to exceeding the storage soft limit; it does not apply to
  inactivity purge. At the hard limit, new writes or transfers may be
  corrupted. Never keep the only copy of an input, checkpoint, or result in
  scratch.
- Listing a directory or checking metadata does not reset file access time.
  Reading, writing, or running a program against a file may do so. Never use
  `touch`, synthetic reads, or any other timestamp-refresh technique to bypass
  the retention rule; HPC warned that such activity may result in access
  suspension. Monitor age, keep durable sources in `/work`, and stage files
  back to scratch only when a real workload needs them.
- AtlasRepair provides a local daily Slack monitor. It warns at 25 days,
  marks files critical at the 30-day purge threshold, and reports scan
  failures. Before each report, its companion archiver moves only aged
  top-level `atlasrepair/logs/*.out` files whose complete `sacct` record is
  terminal. Each move copies to durable storage, verifies SHA-256, appends a
  manifest, and removes the scratch source only after copy verification:

```bash
scripts/start_juno_scratch_age_monitor.sh juno
tmux has-session -t scratch_age_monitor
tail -n 50 logs/watchers/scratch_age.log
```

  The durable destination is
  `/work/oxx220000/atlasrepair/logs/scratch_archive/`; its
  `archive_manifest.jsonl` records the source, destination, job state, size,
  and checksum. The age reporter itself never deletes files or refreshes
  access times. The archiver excludes subdirectories, symlinks, watcher logs,
  `.lock`, `.incomplete`, checkpoints, caches, and any file without a fully
  terminal accounting record. Archived files remain in `/work` unless a real
  workload needs them; never move them back on a timer to reset retention.
  Other alerts remain decision points: rebuild disposable caches, stage
  actively needed files from durable storage, or copy unique data to `/work`
  or backed-up group storage before the purge boundary.
- Use a per-job scratch directory and copy durable outputs back before exit:

```bash
RUN_DIR="$HOME/scratch/<repo-name>/${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

# For very small-file-heavy temporary I/O, use node-local storage.
LOCAL_RUN_DIR="$TMPDIR/<repo-name>"
mkdir -p "$LOCAL_RUN_DIR"
```

- Node-local `$TMPDIR` is wiped when the allocation ends. Copy anything needed
  after the job to `~/scratch` or `~/work` before exit.
- Copy only small final artifacts back to durable storage by default. Copy
  complete image trees only when visual inspection or archival requires them.
- When generated images are needed as a backup, keep the active copy under
  `~/scratch` for job I/O and create an explicit snapshot under
  `/work/NetID/backups/<repo-or-experiment>/snapshots/<timestamp>/`. A scratch
  directory is never the only backup: it has no recovery guarantee and files
  may be purged after 30 days. For image collections that approach the `/work`
  quota, use backed-up group storage under `/groups/<pi-name>` instead.
- Use explicit cache variables in batch scripts:

```bash
export HF_HOME=/work/NetID/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HOME/scratch/hf_datasets"
```

- Make offline model access mandatory only after verifying that the persistent
  cache is complete.

## Python Environments

Juno currently provides `miniconda/24.11.1`, but existing projects may use
plain virtual environments or a fully qualified interpreter instead. Verify
the exact interpreter and imports used by every job; do not silently switch an
established project environment. The current account has used system Python
`/usr/bin/python3` 3.9.21 and project virtual environments under
`/work/oxx220000/venvs/`.

Load modules explicitly in job scripts, preferably with versions. Do not put
project module loads in `.bashrc`:

```bash
module purge
module load miniconda/24.11.1
module load cuda/12.6
module list
```

Create Conda environments under `~/work` or `/groups`, not home. Prefer an
`environment.yml` recipe for collaboration; if a group shares one installed
environment, designate one maintainer and treat it as read-only for everyone
else. Do not keep the only copy of an environment in purgeable scratch.

Generic virtual environment setup:

```bash
/usr/bin/python3 -m venv --system-site-packages \
  /work/NetID/venvs/<env-name>
/work/NetID/venvs/<env-name>/bin/python -m pip install -U pip
/work/NetID/venvs/<env-name>/bin/python -m pip install -r requirements.txt
```

Use a full interpreter path in scripts and validate imports before submission:

```bash
PYTHON_EXE=/work/NetID/venvs/<env-name>/bin/python
"$PYTHON_EXE" -c \
  "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

Do not run concurrent `pip install --user` processes. Parallel writes to the
same user site-packages tree can leave it partially installed.

Run CUDA validation inside a SLURM GPU allocation, not on the login node:

```bash
"$PYTHON_EXE" - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_capability(0))
PY
```

Juno's guide recommends CUDA 12.6 for H100/H200 and CUDA 12.4 for A30, but the
project's PyTorch build and model stack remain the compatibility authority.
Check `nvcc --version`, `torch.version.cuda`, and a real CUDA operation inside
the requested partition.

## Containers

Juno runs containers with Apptainer, not a Docker daemon. The current module is
`apptainer/1.3.4`. Build privileged images off-cluster, transfer the immutable
`.sif`, and run it through SLURM:

```bash
module purge
module load apptainer/1.3.4
apptainer exec --nv \
  --bind "$HOME/scratch:/workspace" \
  /work/NetID/containers/<image>.sif \
  python /workspace/train.py
```

Use `--nv` for NVIDIA access and explicit `--bind` mappings for data. Pin image
tags, version-control definition files, and set `APPTAINER_CACHEDIR` outside
home.

## Sync Code To Juno

For committed code, use Git:

```bash
ssh juno 'cd /work/NetID/<repo-name> && git pull --ff-only'
```

For focused uncommitted changes, sync only the required files:

```bash
rsync -av scripts/my_job.sbatch \
  juno:/work/NetID/<repo-name>/scripts/my_job.sbatch
```

For a broader code sync, exclude generated data and secrets:

```bash
rsync -av \
  --exclude '.git/' \
  --exclude '.vscode/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache/' \
  --exclude 'outputs/' \
  --exclude 'logs/' \
  --exclude 'save/runs/' \
  --exclude '.trellis/.runtime/' \
  ./ juno:/work/NetID/<repo-name>/
```

Keep `.vscode/.env` on the local workstation for Slack notifications. Do not
sync the webhook to Juno for the standard local-watcher workflow. If a legacy
job explicitly requires the secret on Juno, sync it separately and protect it
immediately:

```bash
ssh juno 'mkdir -p /work/NetID/<repo-name>/.vscode'
rsync -av .vscode/.env \
  juno:/work/NetID/<repo-name>/.vscode/.env
ssh juno 'chmod 600 /work/NetID/<repo-name>/.vscode/.env'
```

Verify the remote state:

```bash
ssh juno 'cd /work/NetID/<repo-name> && git status --short'
ssh juno 'cd /work/NetID/<repo-name> && bash -n scripts/my_job.sbatch'
```

## Login And Interactive Tools

- Connect with `ssh NetID@juno.utdallas.edu`; Ed25519 SSH keys are the
  recommended authentication method. Keep `~/.ssh` at mode `700`, private keys
  and `authorized_keys` at `600`, and public keys at `644`.
- If `ssh juno` or `ssh NetID@juno.utdallas.edu` cannot connect or times out,
  try the Juno login IP addresses `10.247.252.78` and `10.247.252.79`, for
  example `ssh NetID@10.247.252.78` or `ssh NetID@10.247.252.79`.
- Off-campus Open OnDemand access requires the UT Dallas VPN. The portal is
  `https://juno-ood.hpcre.utdallas.edu/`.
- Use Open OnDemand for JupyterLab, RStudio, VSCode, and most GUI sessions.
  These sessions must request a compute node and end when the work is done.
- Do not point VSCode Remote-SSH at a normal login node. Use Open OnDemand, or
  the dedicated `juno-vscode.utdallas.edu` host while it remains in testing;
  submit actual computation from its terminal through SLURM.
- For a shell on a compute node, use the documented two-step flow:

```bash
salloc -p normal -N 1 -n 1 -c 4 --mem=20GB -t 02:00:00
srun --pty bash
```

- Jupyter through an SSH tunnel is acceptable when a controlled Conda
  environment is required, but the Jupyter server itself must run on the
  allocated compute node. Open OnDemand is the default for simpler use.

## Slack Notifications

Store the webhook only in the local repo worktree's git-ignored
`.vscode/.env`:

```bash
NOTIFICATION_WEBHOOK=https://hooks.slack.com/services/...
```

Source it locally, submit the job through SSH, and run the watcher locally:

```bash
cd /local/path/to/<repo-name>
set -a
source .vscode/.env
set +a

JOB_ID=$(ssh juno \
  'cd /work/NetID/<repo-name> && sbatch --parsable scripts/my_job.sbatch')
mkdir -p logs/watchers
scripts/start_juno_notifier.sh "$JOB_ID" juno
```

Batch scripts may use an `EXIT` trap for process cleanup, but the standard
workflow does not call Slack from the trap and does not export the webhook to
Juno. Scheduler terminal state from local `sacct` monitoring is authoritative:

```bash
cleanup() {
  local exit_code="$1"

  if [[ -n "${server_pid:-}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi

  return "${exit_code}"
}
trap 'cleanup $?' EXIT
```

`scripts/job_notifier.py` sends the initial aggregate state and one aggregate
message for the first `PENDING` to `RUNNING` transition. Intermediate count
changes are silent. When the job leaves the queue, the parent watcher must
always send one final aggregate terminal summary. This includes an all-success
message such as
`✅ Array 307993 finished: 16 COMPLETED` and an abnormal summary containing the
terminal-state counts when any task fails, times out, or is cancelled.

The watcher does not send separate Slack messages for individual array task
IDs.

### Personal queue drain alert (AtlasRepair)

AtlasRepair provides a persistent local watcher for unused personal queue
capacity. It queries only the current Juno user's expanded array tasks and
sends one Slack alert when `RUNNING <= 3` and `PENDING = 0` remain true for ten
continuous minutes:

```bash
scripts/start_juno_low_running_watcher.sh juno
tmux has-session -t juno_low_running_watcher
tail -f logs/watchers/juno_low_running.log
```

The timer resets on any pending task, more than three running tasks, a failed
SSH/SLURM query, or malformed scheduler output. After an alert, the watcher
stays silent until the condition becomes false and later qualifies again.
This queue-level watcher complements rather than replaces per-job terminal
watchers.

### Temporary H200 capacity alert

During the temporary eight-job allocation window, the local capacity watcher
tracks the current user's aggregate running GPU allocation in the `h200`
partition. Through August 10, 2026 (America/Chicago), it sends one Slack alert
for every observed decrease that reaches `7` GPUs or fewer, such as `8 -> 7`
and then `7 -> 6`. The watcher still polls every 30 seconds, but it does not
wait for a five-minute sustain period. Increases do not send notifications.

Start it from the local workstation with:

```bash
scripts/start_juno_gpu_notifier.sh juno
```

The watcher exits and stops sending capacity alerts at midnight on August 11,
2026 (America/Chicago). This temporary capacity watcher is separate from the
per-job array watchers and does not replace them.

Do not query only `squeue -j <array-jobid>` when building live summaries.
Running array tasks may have distinct allocation job IDs and disappear from
that filtered view while pending rows remain. Query the current user's
expanded array queue with `squeue --array --me`, include `%F` (`ArrayJobID`) in
the output format, and filter records by the parent array ID. Report every
visible state count, so a transition says `2 PENDING, 1 RUNNING`, not only
`2 PENDING`.

Do not add per-task Slack calls to array-task traps. The local parent watcher
reports live aggregate transitions and the authoritative final state.

Treat `sacct` as authoritative. An in-job notification can be misleading when
a timeout or scheduler termination changes how shell cleanup runs.

## Submit A Job

Minimum single-GPU template:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=my-job
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

PYTHON_EXE=/work/NetID/venvs/<env-name>/bin/python
cd /work/NetID/<repo-name>

export HF_HOME=/work/NetID/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"

nvidia-smi
"$PYTHON_EXE" -m <module> --config <config.yaml>
```

Before submission, run the smallest useful checks:

```bash
bash -n scripts/my_job.sbatch
"$PYTHON_EXE" -m py_compile path/to/changed_script.py
```

For long-running or paper-relevant jobs, record:

- task and purpose;
- Git commit or `git status --short`;
- input dataset, split, prompt, config, and generated-data root;
- model, checkpoint, Python interpreter, and cache path;
- partition, GPU count, CPU count, memory, and walltime;
- SLURM job ID, log path, output path, and expected completion signal;
- notification watcher PID or log path;
- caveats required for interpretation or resumption.

## Arrays, Sequential Loops, And Backfill

Each independent `sbatch` submission enters the queue separately. When shards
share their script, environment, model, and resource shape, use one throttled
array, defaulting to `%8` under `juno-pri` for this user's Juno GPU work:

```bash
sbatch --qos=juno-pri --array=0-7%8 scripts/my_sweep.sbatch
```

For independent GPU tasks, choose the throttle as follows:

- `%1`: the user requests serial execution, tasks share mutable state, or one
  allocation with a sequential loop avoids substantial repeated startup cost;
- `%4`: fallback under the normal QOS or a deliberate conservative control;
- `%8`: default under the live `juno-pri` QOS when independent tasks and live
  resources permit it (maximum concurrent jobs per account under `juno-pri`);
- `%12` to `%16`: opportunistic backfill under `normal` QOS when the cluster has
  idle H200 nodes and no priority contention, or achieved across dual accounts
  by running `%8` on `bxz200006` and `%8` on `oxx220000` concurrently under
  `juno-pri`;
- above `%16`: do not assume it will run. Confirm the active QoS limit and
  obtain an explicit HPC allocation before requesting more concurrency.

Keep `#SBATCH --gres=gpu:1` in the script. For example, `%8` allows up to eight
separate one-GPU tasks to run concurrently when the QOS and resources admit
them; it does not turn each task into an eight-GPU training job. SLURM may place
those tasks on the same dual-GPU node or on different nodes. Check the queue
after submission:

```bash
squeue --array -j <array-jobid> \
  --format="%.18i %.9P %.24j %.8T %.10M %.20N %R"
```

Interpretation:

- the requested number of `RUNNING` array tasks means that many one-GPU tasks
  were allocated, regardless of whether SLURM placed them on the same or
  different nodes;
- one `RUNNING` task and remaining tasks pending for `Resources` means only one
  GPU is currently allocatable;
- pending tasks with `JobArrayTaskLimit` confirm that the requested throttle is
  enforcing the concurrency cap;
- do not replace the array with many independent `sbatch` submissions merely
  because the second task does not start immediately.

Do not convert a small or medium single-GPU training run to Distributed Data
Parallel (DDP) merely to use more GPUs. A multi-GPU run needs explicit software
support and a scaling benchmark; communication overhead, changed global batch
size, and optimizer-schedule changes can remove the speed benefit or alter the
experiment protocol. Prefer concurrent one-GPU array tasks when runs are
independent.

Inside the script:

```bash
SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
RUN_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SHARD_ID}"
SCRATCH_RUN_DIR="$HOME/scratch/<repo-name>/<run-kind>/${RUN_ID}"
FINAL_RUN_DIR="/work/NetID/<repo-name>/outputs/<run-kind>/${RUN_ID}"
```

Use a sequential loop inside one allocation when all work fits safely within
one walltime and repeated model or service startup is a large fraction of the
cost:

```bash
for shard_id in $(seq 0 7); do
  "$PYTHON_EXE" train.py --shard-id "$shard_id"
done
```

Use Launcher or GNU Parallel when one allocation must execute many short,
serial CPU tasks. Prefer a SLURM array when each item is itself a sizeable job
with useful per-task scheduler state. Do not submit hundreds of separate
`sbatch` jobs when one array or one Launcher allocation expresses the same
workload.

Request realistic short walltimes so backfill can use transient gaps. Estimate
from a smoke run or completed shard and add a `1.5x` to `2x` buffer for startup,
data staging, and transient I/O. Avoid a default `48:00:00` request when the
work can be split or resumed safely.

## Multi-GPU PyTorch On H200

- Start with one GPU and verify correctness, VRAM, throughput, and GPU
  utilization before adding distributed machinery.
- One H200 node has two GPUs with no NVLink. For DDP, request one process per
  GPU and expect less than ideal 2x scaling because intra-node communication
  crosses PCIe and CPU sockets.
- Keep the per-GPU micro-batch explicit. Record whether global batch size and
  the learning-rate schedule changed when comparing one-GPU and multi-GPU
  runs.
- Use `srun --kill-on-bad-exit=1 --cpu-bind=cores` for SLURM-launched ranks.
  `torchrun --standalone` is suitable for a controlled single-node scaling
  benchmark inside an existing allocation.
- For H200 multi-node DDP/FSDP, the official guide recommends starting with:

```bash
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
export MASTER_PORT=29500
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0
export NCCL_NET_GDR_LEVEL=SYS
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_SOCKET_IFNAME='^lo,docker0'
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

- Treat these NCCL values as a benchmark starting point, not universal truth.
  Confirm `NET/IB` rather than `NET/Socket` in the first log. A/B test
  `NCCL_NET_GDR_LEVEL=SYS` against `LOC`; cross-socket GPUDirect RDMA can be
  slower than host staging for some workloads.
- Track aggregate throughput, throughput per GPU, step time, Model FLOPs
  Utilization (MFU), and peak VRAM. Increase GPU count only while the scaling
  result justifies the extra allocation.

## Monitoring And Queue Diagnosis

```bash
squeue --me -o "%i %P %j %T %M %D %R"
sacct -j <jobid> \
  --format=JobID,JobName%32,Partition,State,ExitCode,Elapsed,MaxRSS,NodeList
tail -f logs/<job-name>-<jobid>.out
```

For arrays:

```bash
squeue -j <array-jobid>
sacct -j <array-jobid> \
  --format=JobID,JobName%32,Partition,State,ExitCode,Elapsed,NodeList
```

Diagnose a pending job:

```bash
scontrol show job <jobid> | \
  grep -E "JobId=|JobState=|Reason=|Partition=|Priority=|StartTime=|ReqTRES=|AllocTRES=|MinMemory"
sprio -j <jobid>
squeue --start -j <jobid>
sshare -U -u "$USER"
```

Start-time estimates are dynamic and can change with higher-priority
submissions, reservations, fairshare, and resource state.

After a completed job, use Juno's `jobstats` module to right-size later CPU,
memory, and walltime requests:

```bash
module load jobstats
jobstats <jobid>
```

Check the top-level `sacct` state as well as `.batch` and `srun` step rows.
`MaxRSS` is normally attached to a step rather than the top-level summary.
Treat `sacct` as authoritative for `COMPLETED`, `FAILED`, `TIMEOUT`,
`CANCELLED`, and `OUT_OF_MEMORY`.

`nvitop` and `gpustat` require a GPU compute node with NVML access:

```bash
nvidia-smi
nvidia-smi topo -m
nvitop
gpustat --no-color
```

## Long-Running Services

Treat persistent VLM, T2I, or evaluation services as experiment state. Record
the command, model/checkpoint, cache, host, port, GPU allocation, log, health
check, and dependent jobs. Launch services inside SLURM allocations and stop
them as soon as dependent work finishes.

Before submitting a dependent job, verify the endpoint from a compute node
with equivalent network visibility.

## Download Public Data And Models

Prefer Juno-local downloads:

```bash
ssh juno 'mkdir -p /work/NetID/data /work/NetID/hf_cache'
```

Hugging Face example:

```bash
PYTHON_EXE=/work/NetID/venvs/<env-name>/bin/python
"$PYTHON_EXE" -m huggingface_hub.commands.huggingface_cli login
"$PYTHON_EXE" -m huggingface_hub.commands.huggingface_cli download \
  <owner>/<dataset> \
  --repo-type dataset \
  --local-dir /work/NetID/data/<dataset-dir> \
  --max-workers 8
```

## Snapshot Backups

Use `/work`, not `~/scratch`, for persistent snapshots.

First snapshot:

```bash
src=/path/to/source
name=$(basename "$src")
host=$(hostname -s)
stamp=$(date +%F_%H%M%S)
dest_base=/work/NetID/backups/${host}/${name}

ssh juno "mkdir -p ${dest_base}/snapshots/${stamp}"
rsync -aHAX --numeric-ids --delete --info=progress2 \
  "${src}/" "juno:${dest_base}/snapshots/${stamp}/"
ssh juno "cd ${dest_base} && ln -sfn snapshots/${stamp} latest"
```

Incremental snapshot:

```bash
src=/path/to/source
name=$(basename "$src")
host=$(hostname -s)
stamp=$(date +%F_%H%M%S)
dest_base=/work/NetID/backups/${host}/${name}

ssh juno "mkdir -p ${dest_base}/snapshots/${stamp}"
rsync -aHAX --numeric-ids --delete --info=progress2 \
  --link-dest="${dest_base}/latest" \
  "${src}/" "juno:${dest_base}/snapshots/${stamp}/"
ssh juno "cd ${dest_base} && ln -sfn snapshots/${stamp} latest"
```

Keep the trailing slash in `"${src}/"`. `--delete` affects only the new dated
snapshot; earlier snapshots remain available. Verify size, file count, and the
`latest` symlink after completion.

For a generated-image backup, set `src` to the run's `images/` directory (or to
the complete run directory if the manifest and images should remain together),
and use a stable experiment-specific `dest_base`, for example:

```bash
src="$HOME/scratch/atlasrepair/dsg_eval/<run-id>/images"
dest_base=/work/NetID/backups/atlasrepair/dsg/<run-id>
stamp=$(date +%F_%H%M%S)

ssh juno "mkdir -p '${dest_base}/snapshots/${stamp}' && \
  rsync -aHAX --numeric-ids --delete --info=progress2 \
    '${src}/' '${dest_base}/snapshots/${stamp}/' && \
  cd '${dest_base}' && ln -sfn 'snapshots/${stamp}' latest"
```

For later snapshots, add
`--link-dest="${dest_base}/latest"` to the remote `rsync` command.

The repository's normal durable result merge may still omit the `images/`
tree; the backup snapshot is the separate archival copy.

## Copy Back Results

Prefer logs and small result files over complete output trees:

```bash
rsync -aP --include '*/' --include '*.log' --exclude '*' \
  juno:/work/NetID/<repo-name>/logs/ ./logs/

rsync -aP \
  juno:/work/NetID/<repo-name>/outputs/<run-name>/summary.json \
  outputs/<run-name>/summary.json
```

Do not copy editor directories, Python caches, PID files, or large generated
image trees unless they are explicitly needed.

## Official Guide Map

The internal pages reviewed for this note are grouped below so future updates
can go directly to the relevant official source:

- Getting started: [account request](https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/getting-started/account-request/),
  [login](https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/getting-started/login/),
  [SSH keys](https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/getting-started/ssh-keys/),
  [storage](https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/getting-started/storage/),
  [scratch](https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/getting-started/scratch-space/),
  [Saturn](https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/getting-started/saturn/), and
  [hardware](https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/getting-started/hardware/).
- Working on Juno: [Linux commands](https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/working-on-juno/linux-commands/),
  [modules](https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/working-on-juno/modules/), and
  [software](https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/working-on-juno/software/).
- Running programs: [SLURM](https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/running-programs/slurm/),
  [monitoring](https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/running-programs/advanced-slurm/),
  [common programs](https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/running-programs/common-programs/),
  [parallelism](https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/running-programs/parallelism/), and
  [Launcher](https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/running-programs/launcher/).
- GUI and tools: [Open OnDemand](https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/gui-and-tools/open-ondemand/),
  [X11](https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/gui-and-tools/gui-programs/),
  [JupyterLab](https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/gui-and-tools/jupyter/),
  [RStudio](https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/gui-and-tools/rstudio/), and
  [VSCode](https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/gui-and-tools/vscode/).
- GPU and AI/ML: [GPU overview](https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/ai-and-ml/),
  [PyTorch training](https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/ai-and-ml/pytorch-training/),
  [GPU performance](https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/ai-and-ml/gpu-performance/), and
  [AlphaFold 3](https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/ai-and-ml/alphafold3/).
- Advanced: [containers](https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/advanced/containers/),
  [Miniconda](https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/advanced/miniconda/), and
  [Python optimization](https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/advanced/python-optimization/).
- Support: [getting help](https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/support/getting-help/),
  and [FAQ](https://utdallas-hpc-juno-ug.readthedocs-hosted.com/en/latest/support/faq/).

## Project Profile: AtlasRepair

Repository and storage:

```text
Local repo: /people/cs/o/oxx220000/atlasrepair
Juno repo:  /work/oxx220000/atlasrepair
Juno env:   /work/oxx220000/venvs/atlas
HF cache:   /work/oxx220000/hf_cache
Job HF cache: /scratch/juno/oxx220000/hf_cache
Scratch:    /home/oxx220000/scratch/atlasrepair
Webhook:    /work/oxx220000/atlasrepair/.vscode/.env
```

Runtime profile:

- The working environment uses Python 3.11 with the interpreter
  `/work/oxx220000/venvs/atlas/bin/python3`.
- Triton builds may need
  `CPATH=/work/oxx220000/python3.11-headers/include/python3.11`.
- Verified core packages include PyTorch 2.8.0+cu126, torchvision 0.23.0+cu126,
  vLLM 0.11.0, Transformers 4.57.6, Diffusers 0.38.0, FastAPI 0.115.14,
  scikit-learn 1.9.0, and NumPy 2.2.6.
- Do not run parallel package installation processes against this environment.
- Current model assets include `Qwen/Qwen3-VL-8B-Instruct` and
  `stabilityai/sdxl-turbo` in the persistent Hugging Face cache.

Prepared commands:

```bash
# CPU-only search smoke
bash examples/run_demo.sh

# Qwen3-VL canaries
sbatch scripts/juno_qwen3_vl8b_canary.sbatch
sbatch scripts/juno_qwen3_vl32b_canary.sbatch

# Sequential real-model search smoke or shard array
sbatch scripts/juno_failureatlas_sequential_smoke.sbatch
sbatch --array=0-5%2 \
  --export=ALL,PROMPT_FILE=outputs/search_smoke/prompt_sample_full.jsonl,LIMIT_NODES=1000,IMAGES_PER_NODE=8,SYNC_IMAGES_TO_WORK=0 \
  scripts/juno_failureatlas_sequential_smoke.sbatch

# Standard search driver
sbatch --export=ALL,CONFIG=configs/default.yaml,TASK_NAME=my_run \
  scripts/juno_run_search.sbatch
```

AtlasRepair-specific rules:

- New AtlasRepair compute jobs and Hugging Face staging wrappers use
  `/scratch/juno/oxx220000/hf_cache` by default for model I/O and retain a
  synchronized source copy under `/work/oxx220000/hf_cache`. Do not delete the
  `/work` copy when cleaning scratch. If a model is not present in scratch,
  run `scripts/juno_stage_hf_cache.sh` before submitting the job; do not let
  multiple jobs initialize the same new cache concurrently.
- Keep heavy search images under
  `$HOME/scratch/atlasrepair/search_smoke/<run_id>/`; copy only summaries and
  logs to `/work` unless `SYNC_IMAGES_TO_WORK=1` is intentionally requested.
- From the local workstation, use
  `scripts/start_juno_scratch_age_monitor.sh juno` to run the daily scratch
  age report. It warns through the local webhook for files at least 25 days
  old and marks files at least 30 days old as critical. The same daily loop
  first runs `scripts/juno_archive_terminal_logs.py --execute` under the strict
  terminal-log allowlist documented above. All other file types are advisory
  only and are never moved or deleted automatically.
- For CPU-only AtlasRepair replay and analysis jobs that are serial, have low
  CPU utilization, and do not have safe checkpoints, prefer a generous
  `#SBATCH --time=48:00:00` walltime (or the longest limit allowed by the
  selected partition). Avoid timing out these jobs merely to improve
  backfill; use a shorter limit only when the work is safely resumable or has
  been deliberately split into an array.
- Configure `configs/t2i_server.yaml`, `configs/vlm_client.yaml`, and
  `configs/default.yaml` before a real search.
- T2I and VLM services must be launched inside SLURM allocations and recorded
  with their endpoints and dependent jobs.
- Record active job state and continuation details in the repo journal,
  Trellis task when explicitly used, and `.handoff.md`. Preserve stable
  human-facing results in `report.md`.
- Load `NOTIFICATION_WEBHOOK` from the repo's `.vscode/.env` for every local or
  Juno experiment.

## Project Profile: SafeFix

Repository and storage:

```text
Local repo:       /people/cs/o/oxx220000/safefix
Juno worktree:    /work/oxx220000/safefix_rebuttal
Legacy worktree:  /work/oxx220000/safefix
Juno env:         /work/oxx220000/venvs/qwen3
HF cache:         /work/oxx220000/hf_cache
Dataset scratch:  /home/oxx220000/scratch/safefix_cache/data
Run scratch:      /home/oxx220000/scratch/safefix_rebuttal
Webhook:          /work/oxx220000/safefix_rebuttal/.vscode/.env
```

Runtime profile:

- Use `/work/oxx220000/venvs/qwen3/bin/python` for Qwen3-VL validation.
- The environment reuses the installed CUDA PyTorch stack; verify imports with
  the exact interpreter before submitting arrays.
- Observed packages include PyTorch 2.8.0+cu128, Transformers 4.57.6, and
  Accelerate 1.10.1.
- `Qwen/Qwen3-VL-8B-Instruct` is staged in the persistent cache. Inventory
  existing assets before downloading anything again.
- The rebuttal worktree is the current Juno target. Do not overwrite the
  legacy `/work/oxx220000/safefix` worktree during sync.

Prepared commands:

```bash
# Inventory rebuttal models, datasets, generated images, and metadata
bash scripts/juno_rebuttal_prepare.sh

# Offline single-image Qwen3-VL validation
PYTHON_EXE=/work/oxx220000/venvs/qwen3/bin/python \
  sbatch scripts/juno_qwen3_rebuttal_smoke.sbatch
```

SafeFix-specific rules:

- Default new GPU experiments to H200 unless the user requests another target.
- SafeFix GPU experiments default to `juno-pri` with `%8` array throttling when
  the live association and QOS check confirms availability. Use normal QOS and
  `%4` as the fallback when the priority QOS is unavailable or when a
  conservative control is requested.
- Record dataset split, generated-data root, attribute metadata, model,
  checkpoint, interpreter, cache, resources, job ID, logs, outputs, and
  completion criteria for paper-relevant jobs.
- Keep changing rebuttal experiment state in
  `.trellis/workspace/oxu/journal-1.md` and `.handoff.md`; keep stable results
  in `report.md` or the authoritative manuscript files.
- Use `scripts/job_notifier.py` and `scripts/test_notification.py` with the
  git-ignored `.vscode/.env` for every experiment.
