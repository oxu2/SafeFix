#!/usr/bin/env python3
"""Unified job notifier for SLURM (Juno) and local jobs.

Usage:
    set -a
    source .vscode/.env
    set +a

    # Watch a SLURM job on Juno
    nohup python3 scripts/job_notifier.py slurm <JOB_ID> &

    # Watch Juno over SSH while sending Slack from this machine
    nohup python3 scripts/job_notifier.py slurm --host juno <JOB_ID> &

    # Watch aggregate H200 usage for the current user
    nohup python3 scripts/job_notifier.py gpu-capacity --host juno &

    # Run a local command and notify (command omitted from notifications by default)
    python3 scripts/job_notifier.py local --name "My Job" -- python3 train.py

    # Include the command in the start notification when useful
    python3 scripts/job_notifier.py local --name "My Job" --include-command -- python3 train.py

    # Label a command that executes on a remote host
    python3 scripts/job_notifier.py local --name "ZY Job" --host-label zy -- ssh zy command
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

WEBHOOK_URL = os.environ.get("NOTIFICATION_WEBHOOK", "")


def send_notification(
    message: str,
    *,
    attachments: list[dict] | None = None,
) -> bool:
    """Send to Slack/Discord webhook. No-op if WEBHOOK_URL is empty."""
    if not WEBHOOK_URL:
        return False

    if "discord.com" in WEBHOOK_URL or "discordapp.com" in WEBHOOK_URL:
        payload = {"content": message}
    else:
        payload = {"text": message}
        if attachments:
            payload["attachments"] = attachments

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        WEBHOOK_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.getcode() in (200, 204)
    except (OSError, TimeoutError, ValueError, urllib.error.URLError) as error:
        print(f"[notify] notification error: {error}", file=sys.stderr)
        return False


# --- Local Command Wrapper ---


def format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def display_hostname() -> str:
    """Return a concise hostname for user-facing notifications."""
    try:
        hostname = os.uname().nodename
    except AttributeError:
        import socket

        hostname = socket.gethostname()

    if hostname.startswith("csr-") and hostname.endswith(".utdallas.edu"):
        return "utd"
    return hostname


def run_local(
    name: str,
    cmd: list[str],
    include_command: bool = False,
    host_label: str | None = None,
) -> None:
    if not cmd:
        print("Error: No command provided.", file=sys.stderr)
        sys.exit(1)

    hostname = host_label or display_hostname()

    if not WEBHOOK_URL:
        print("[notify] WARNING: NOTIFICATION_WEBHOOK not set. Outputs to stdout.")

    start_msg = f"🚀 *{name}* STARTED on `{hostname}`"
    if include_command:
        start_msg += f"\nCommand: `{' '.join(cmd)}`"
    print(f"[notify] {start_msg}")
    send_notification(start_msg)

    start_time = time.time()
    exit_code = 1
    try:
        process = subprocess.Popen(cmd)
        process.wait()
        exit_code = process.returncode
    except KeyboardInterrupt:
        print("\n[notify] Job interrupted by user.")
        exit_code = 130
    except (OSError, subprocess.SubprocessError) as error:
        print(f"\n[notify] Failed to run command: {error}")
        exit_code = 1
    finally:
        end_time = time.time()
        duration = format_duration(end_time - start_time)

        if exit_code == 0:
            end_msg = (
                f"✅ *{name}* COMPLETED successfully on `{hostname}`\n"
                f"Duration: `{duration}`"
            )
        elif exit_code in {130, -2}:
            end_msg = f"🛑 *{name}* CANCELLED on `{hostname}`\nDuration: `{duration}`"
        else:
            end_msg = (
                f"❌ *{name}* FAILED with exit code {exit_code} on `{hostname}`\n"
                f"Duration: `{duration}`"
            )

        print(f"[notify] {end_msg}")
        send_notification(end_msg)
        sys.exit(exit_code)


# --- SLURM Watcher ---

POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "30"))
ACCOUNTING_GRACE_POLLS = int(os.environ.get("ACCOUNTING_GRACE_POLLS", "4"))
GPU_CAPACITY_DISABLE_AT = datetime(
    2026, 8, 11, tzinfo=ZoneInfo("America/Chicago")
)
GPU_CAPACITY_THRESHOLD = 7
# QoS → MaxJobsPU concurrent-running threshold for the "all slots
# occupied" notification.  juno-pri allows 8; other QoS defaults to 4.
QOS_CONCURRENCY: dict[str, int] = {
    "juno-pri": 8,
}
QOS_CONCURRENCY_DEFAULT = 4


def concurrency_limit_for_qos(qos: str) -> int:
    """Return the concurrent-running notification threshold for a QoS."""
    return QOS_CONCURRENCY.get(qos, QOS_CONCURRENCY_DEFAULT)

TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
}

STATE_EMOJI = {
    "PENDING": "⏳",
    "RUNNING": "🚀",
    "COMPLETED": "✅",
    "FAILED": "❌",
    "CANCELLED": "🛑",
    "TIMEOUT": "⏰",
    "OUT_OF_MEMORY": "💥",
    "NODE_FAIL": "🔧",
    "PREEMPTED": "⚡",
}

STATE_ORDER = (
    "PENDING",
    "CONFIGURING",
    "RUNNING",
    "COMPLETING",
    "SUSPENDED",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "UNKNOWN",
)

SLURM_USER_LABELS = {"bxz200006": "bxz", "rxl190028": "rxl"}


def display_slurm_user(user: str) -> str:
    """Return the concise user label used in notifications."""
    normalized = str(user).strip()
    return SLURM_USER_LABELS.get(normalized, normalized or "unknown")


@dataclass(frozen=True)
class SlurmRecord:
    job_id: str
    array_job_id: str
    name: str
    state: str
    partition: str = ""
    elapsed: str = ""
    nodelist: str = ""
    reason: str = ""
    exit_code: str = ""
    user: str = ""


@dataclass(frozen=True)
class SlurmSnapshot:
    job_id: str
    name: str
    partition: str
    state_counts: tuple[tuple[str, int], ...]
    task_count: int
    completed_task_count: int
    unsuccessful_task_count: int
    all_terminal: bool
    is_array: bool
    nodes: tuple[str, ...]
    user: str = ""
    user_cancelled: bool = False

    @property
    def signature(self) -> tuple[tuple[str, int], ...]:
        return self.state_counts

    @property
    def summary(self) -> str:
        parts = []
        for state, count in self.state_counts:
            if state == "CANCELLED" and self.user_cancelled:
                parts.append(f"{count} CANCELLED (user killed)")
            else:
                parts.append(f"{count} {state}")
        return ", ".join(parts)

    @property
    def overall_state(self) -> str:
        counts = dict(self.state_counts)
        for state in (
            "FAILED",
            "OUT_OF_MEMORY",
            "TIMEOUT",
            "NODE_FAIL",
            "CANCELLED",
            "PREEMPTED",
            "BOOT_FAIL",
            "DEADLINE",
        ):
            if counts.get(state, 0):
                return state
        if self.all_terminal and self.task_count:
            return "COMPLETED"
        if counts.get("RUNNING", 0) or counts.get("COMPLETING", 0):
            return "RUNNING"
        if counts.get("PENDING", 0) or counts.get("CONFIGURING", 0):
            return "PENDING"
        if counts.get("SUSPENDED", 0):
            return "SUSPENDED"
        return "UNKNOWN"


@dataclass(frozen=True)
class GpuUsageSnapshot:
    """Current running GPU allocation for the user's jobs in one partition."""

    used_gpus: int
    active_jobs: int
    nodes: tuple[str, ...]


def normalized_state(state: str) -> str:
    """Normalize scheduler decorations without hiding abnormal outcomes."""
    normalized = str(state).strip().upper()
    if normalized.startswith("CANCELLED"):
        return "CANCELLED"
    if not normalized:
        return "UNKNOWN"
    return normalized.split()[0].rstrip("+")


def _array_task_index(record_id: str, parent_job_id: str) -> int | None:
    match = re.fullmatch(rf"{re.escape(parent_job_id)}_\[?(\d+)\]?", record_id)
    return int(match.group(1)) if match else None


def _expand_compact_array_indices(
    record_id: str,
    parent_job_id: str,
) -> tuple[int, ...]:
    """Expand an squeue compact array ID such as ``300_[0-7%4]``."""
    match = re.fullmatch(
        rf"{re.escape(parent_job_id)}_\[([^\]]+)\]",
        record_id,
    )
    if not match:
        return ()

    expression = match.group(1).split("%", maxsplit=1)[0]
    indices: set[int] = set()
    try:
        for component in expression.split(","):
            range_match = re.fullmatch(r"(\d+)-(\d+)(?::(\d+))?", component)
            if range_match:
                start, stop = int(range_match.group(1)), int(range_match.group(2))
                step = int(range_match.group(3) or 1)
                if step <= 0 or stop < start:
                    return ()
                indices.update(range(start, stop + 1, step))
            elif component.isdigit():
                indices.add(int(component))
            else:
                return ()
    except ValueError:
        return ()
    return tuple(sorted(indices))


def parse_squeue_output(output: str, job_id: str) -> list[SlurmRecord]:
    """Parse expanded squeue array rows for one parent allocation."""
    records: list[SlurmRecord] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split("|")]
        fields.extend([""] * (9 - len(fields)))
        record = SlurmRecord(
            job_id=fields[0],
            array_job_id=fields[1],
            name=fields[2],
            state=fields[3],
            partition=fields[4],
            elapsed=fields[5],
            nodelist=fields[6],
            reason=fields[7],
            user=fields[8],
        )
        if record.job_id == job_id or record.array_job_id == job_id:
            records.append(record)
    return records


def parse_sacct_output(output: str, job_id: str) -> list[SlurmRecord]:
    """Parse allocation-only sacct rows, excluding batch/extern steps."""
    records: list[SlurmRecord] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split("|")]
        fields.extend([""] * (8 - len(fields)))
        record_id = fields[0]
        if "." in record_id:
            continue
        if record_id != job_id and not record_id.startswith(f"{job_id}_"):
            continue
        records.append(
            SlurmRecord(
                job_id=record_id,
                array_job_id=(job_id if record_id.startswith(f"{job_id}_") else ""),
                name=fields[1],
                state=fields[2],
                exit_code=fields[3],
                elapsed=fields[4],
                nodelist=fields[5],
                partition=fields[6],
                user=fields[7],
            )
        )
    return records


def remote_command(command: list[str], host: str | None) -> list[str]:
    """Return a scheduler command for local execution or SSH execution."""
    if not host:
        return command
    if host.startswith("-"):
        raise ValueError("SSH host must not start with '-'.")
    return ["ssh", "--", host, shlex.join(command)]


def parse_gpu_usage_output(output: str) -> GpuUsageSnapshot:
    """Parse expanded ``squeue`` rows into an aggregate GPU allocation."""
    used_gpus = 0
    active_jobs = 0
    nodes: set[str] = set()
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split("|")]
        fields.extend([""] * (5 - len(fields)))
        _job_id, state, tres, nodelist, _name = fields[:5]
        if normalized_state(state) not in {"RUNNING", "COMPLETING"}:
            continue
        matches = re.findall(
            r"(?:^|/)gpu(?::[^,:=]+)?[:=](\d+)(?:$|,)", tres
        )
        if not matches:
            raise ValueError(f"could not parse GPU allocation {tres!r}")
        used_gpus += sum(int(match) for match in matches)
        active_jobs += 1
        if nodelist not in {"", "None", "(null)", "N/A"}:
            nodes.add(nodelist)
    return GpuUsageSnapshot(
        used_gpus=used_gpus,
        active_jobs=active_jobs,
        nodes=tuple(sorted(nodes)),
    )


def get_gpu_usage_snapshot(
    partition: str = "h200",
    *,
    host: str | None = None,
) -> GpuUsageSnapshot | None:
    """Read aggregate running GPU usage for the current user."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", partition):
        raise ValueError("partition must contain only letters, digits, '.', '_' or '-'")
    try:
        result = subprocess.run(
            remote_command(
                [
                    "squeue",
                    "--array",
                    "--me",
                    f"--partition={partition}",
                    "--states=RUNNING,COMPLETING",
                    "--noheader",
                    "--format=%i|%T|%b|%N|%j",
                ],
                host,
            ),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        print(f"[watcher] GPU usage query error: {error}")
        return None
    if result.returncode != 0:
        print(f"[watcher] GPU usage query failed: {result.stderr.strip()}")
        return None
    try:
        return parse_gpu_usage_output(result.stdout)
    except ValueError as error:
        print(f"[watcher] GPU usage parse failed: {error}")
        return None


def gpu_capacity_threshold(now: datetime | None = None) -> int | None:
    """Return 7 through Aug. 10, then disable capacity alerts."""
    current = now or datetime.now(GPU_CAPACITY_DISABLE_AT.tzinfo)
    if current.tzinfo is None:
        current = current.replace(tzinfo=GPU_CAPACITY_DISABLE_AT.tzinfo)
    if current >= GPU_CAPACITY_DISABLE_AT:
        return None
    return GPU_CAPACITY_THRESHOLD


def _gpu_capacity_message(
    previous_used_gpus: int,
    usage: GpuUsageSnapshot,
    *,
    partition: str,
) -> str:
    node_text = ", ".join(usage.nodes[:6]) if usage.nodes else "none"
    return (
        f"📉 `{partition}` GPU usage decreased: `{previous_used_gpus}` → "
        f"`{usage.used_gpus}` GPU(s) in use; "
        f"{usage.active_jobs} active job(s), node(s): `{node_text}`"
    )


def run_gpu_capacity(
    *,
    host: str | None = None,
    partition: str = "h200",
    now_fn=None,
    max_polls: int | None = None,
) -> int:
    """Notify on each observed GPU decrease through the temporary cutoff."""
    now_fn = now_fn or (lambda: datetime.now(GPU_CAPACITY_DISABLE_AT.tzinfo))
    previous_used_gpus: int | None = None
    polls = 0

    print(
        f"[watcher] Watching aggregate {partition} GPU usage through "
        f"SSH host {host or 'local'} (poll every {POLL_INTERVAL}s; "
        "notify on each decrease)"
    )
    if not WEBHOOK_URL:
        print("[watcher] WARNING: NOTIFICATION_WEBHOOK not set.")

    while True:
        usage = get_gpu_usage_snapshot(partition, host=host)
        if usage is None:
            print("[watcher] GPU usage unavailable; preserving alert state.")
        else:
            threshold = gpu_capacity_threshold(now_fn())
            if threshold is None:
                print(
                    "[watcher] Aggregate GPU capacity notifications are disabled "
                    "after 2026-08-10 America/Chicago."
                )
                return 0

            print(
                f"[watcher] {partition}: {usage.used_gpus} GPU(s) in use, "
                f"{usage.active_jobs} active job(s); alert threshold <={threshold}."
            )
            if previous_used_gpus is not None:
                decreased_below_threshold = (
                    usage.used_gpus < previous_used_gpus
                    and usage.used_gpus <= threshold
                )
                if decreased_below_threshold:
                    message = _gpu_capacity_message(
                        previous_used_gpus,
                        usage,
                        partition=partition,
                    )
                    print(f"[watcher] {message}")
                    send_notification(message)
            previous_used_gpus = usage.used_gpus

        polls += 1
        if max_polls is not None and polls >= max_polls:
            return 0
        time.sleep(POLL_INTERVAL)


def get_squeue_records(
    job_id: str,
    *,
    host: str | None = None,
) -> list[SlurmRecord]:
    """Read all live array-task records for one parent job ID."""
    try:
        result = subprocess.run(
            remote_command(
                [
                    "squeue",
                    "--array",
                    "-j",
                    job_id,
                    "--noheader",
                    "--format=%i|%F|%j|%T|%P|%M|%N|%r|%u",
                ],
                host,
            ),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        print(f"[watcher] squeue error: {error}")
        return []
    if result.returncode != 0:
        print(f"[watcher] squeue failed: {result.stderr.strip()}")
        return []
    return parse_squeue_output(result.stdout, job_id)


def get_sacct_records(
    job_id: str,
    *,
    host: str | None = None,
) -> list[SlurmRecord]:
    """Read parent and array-task allocation records from SLURM accounting."""
    try:
        result = subprocess.run(
            remote_command(
                [
                    "sacct",
                    "-j",
                    job_id,
                    "--noheader",
                    "--parsable2",
                    "--allocations",
                    "--format=JobID,JobName,State,ExitCode,Elapsed,NodeList,Partition,User",
                ],
                host,
            ),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        print(f"[watcher] sacct error: {error}")
        return []
    if result.returncode != 0:
        print(f"[watcher] sacct failed: {result.stderr.strip()}")
        return []
    return parse_sacct_output(result.stdout, job_id)


def build_slurm_snapshot(
    job_id: str,
    *,
    queue_records: list[SlurmRecord],
    accounting_records: list[SlurmRecord],
) -> SlurmSnapshot:
    """Merge active and terminal array tasks without counting the parent row."""
    accounting_children = [
        record
        for record in accounting_records
        if _array_task_index(record.job_id, job_id) is not None
    ]
    compact_accounting_records = [
        (record, _expand_compact_array_indices(record.job_id, job_id))
        for record in accounting_records
    ]
    compact_accounting_records = [
        (record, indices) for record, indices in compact_accounting_records if indices
    ]
    compact_queue_records = [
        (record, _expand_compact_array_indices(record.job_id, job_id))
        for record in queue_records
    ]
    compact_queue_records = [
        (record, indices) for record, indices in compact_queue_records if indices
    ]
    numeric_queue_children = [
        record
        for record in queue_records
        if _array_task_index(record.job_id, job_id) is not None
    ]
    queue_is_array = bool(compact_queue_records or numeric_queue_children)
    accounting_is_array = bool(compact_accounting_records or accounting_children)

    records_by_task: dict[str, SlurmRecord] = {}
    for compact_record, indices in compact_queue_records:
        for index in indices:
            virtual_id = f"{job_id}_{index}"
            records_by_task[virtual_id] = SlurmRecord(
                job_id=virtual_id,
                array_job_id=job_id,
                name=compact_record.name,
                state=compact_record.state,
                partition=compact_record.partition,
                elapsed=compact_record.elapsed,
                nodelist=compact_record.nodelist,
                reason=compact_record.reason,
                exit_code=compact_record.exit_code,
                user=compact_record.user,
            )

    for compact_record, indices in compact_accounting_records:
        for index in indices:
            virtual_id = f"{job_id}_{index}"
            records_by_task[virtual_id] = SlurmRecord(
                job_id=virtual_id,
                array_job_id=job_id,
                name=compact_record.name,
                state=compact_record.state,
                partition=compact_record.partition,
                elapsed=compact_record.elapsed,
                nodelist=compact_record.nodelist,
                reason=compact_record.reason,
                exit_code=compact_record.exit_code,
                user=compact_record.user,
            )

    if accounting_children:
        records_by_task.update(
            {record.job_id: record for record in accounting_children}
        )
    elif not (queue_is_array or accounting_is_array):
        records_by_task.update(
            {
                record.job_id: record
                for record in accounting_records
                if record.job_id == job_id
            }
        )
    for record in queue_records:
        if (queue_is_array and record.job_id == job_id) or (
            _expand_compact_array_indices(record.job_id, job_id)
        ):
            continue
        records_by_task[record.job_id] = record

    records = list(records_by_task.values())
    state_counter = Counter(normalized_state(record.state) for record in records)
    ordered_states = [state for state in STATE_ORDER if state in state_counter]
    ordered_states.extend(
        sorted(state for state in state_counter if state not in STATE_ORDER)
    )
    state_counts = tuple((state, state_counter[state]) for state in ordered_states)
    task_count = len(records)
    completed_task_count = state_counter.get("COMPLETED", 0)
    all_terminal = bool(records) and all(
        normalized_state(record.state) in TERMINAL_STATES for record in records
    )
    unsuccessful_task_count = sum(
        count
        for state, count in state_counter.items()
        if state in TERMINAL_STATES and state != "COMPLETED"
    )
    name = next((record.name for record in records if record.name), "")
    partition = next(
        (record.partition for record in records if record.partition),
        "unknown",
    )
    nodes = tuple(
        sorted(
            {
                record.nodelist
                for record in records
                if record.nodelist not in {"", "None", "(null)", "N/A"}
            }
        )
    )
    user = next((record.user for record in records if record.user), "unknown")
    user_cancelled = any(
        "CANCELLED BY" in record.state.upper() for record in records
    ) or any(
        "CANCELLED BY" in record.state.upper()
        for record in accounting_records + queue_records
    )
    return SlurmSnapshot(
        job_id=job_id,
        name=name,
        partition=partition,
        state_counts=state_counts,
        task_count=task_count,
        completed_task_count=completed_task_count,
        unsuccessful_task_count=unsuccessful_task_count,
        all_terminal=all_terminal,
        is_array=(
            queue_is_array
            or accounting_is_array
            or bool(accounting_children)
            or any(
                _array_task_index(record.job_id, job_id) is not None
                for record in records
            )
        ),
        nodes=nodes,
        user=user,
        user_cancelled=user_cancelled,
    )


def get_slurm_snapshot(
    job_id: str,
    *,
    host: str | None = None,
) -> SlurmSnapshot:
    return build_slurm_snapshot(
        job_id,
        queue_records=get_squeue_records(job_id, host=host),
        accounting_records=get_sacct_records(job_id, host=host),
    )


def _snapshot_message(
    snapshot: SlurmSnapshot,
    *,
    previous_summary: str | None,
) -> str:
    emoji = STATE_EMOJI.get(snapshot.overall_state, "❓")
    noun = "Array" if snapshot.is_array else "Job"
    label = f"{noun} `{snapshot.job_id}`"
    if snapshot.name:
        label += f" ({snapshot.name})"
    label += f" for `{display_slurm_user(snapshot.user)}`"
    if snapshot.partition and snapshot.partition.lower() != "h200":
        execution = f" on partition `{snapshot.partition}`"
    else:
        execution = ""
    if previous_summary is None:
        message = f"{emoji} {label}{execution}: `{snapshot.summary}`"
    else:
        message = (
            f"{emoji} {label}{execution}: "
            f"`{previous_summary}` → `{snapshot.summary}`"
        )
    if snapshot.all_terminal:
        message += (
            f"; completed {snapshot.completed_task_count}/{snapshot.task_count}, "
            f"unsuccessful {snapshot.unsuccessful_task_count}"
        )
    elif snapshot.nodes:
        visible_nodes = ", ".join(snapshot.nodes[:4])
        if len(snapshot.nodes) > 4:
            visible_nodes += f", +{len(snapshot.nodes) - 4} more"
        message += f" on node(s) `{visible_nodes}`"
    return message


def run_slurm(
    job_id: str,
    *,
    host: str | None = None,
    concurrency_threshold: int = 8,
) -> int:
    """Watch a parent job or full array until every known task is terminal."""
    if not str(job_id).isdigit():
        print("Error: SLURM watcher requires a numeric parent job ID.", file=sys.stderr)
        return 2
    scheduler = f" through SSH host {host}" if host else ""
    print(
        f"[watcher] Watching SLURM job {job_id}{scheduler} "
        f"(poll every {POLL_INTERVAL}s)"
    )
    if not WEBHOOK_URL:
        print("[watcher] WARNING: NOTIFICATION_WEBHOOK not set.")

    previous_signature: tuple[tuple[str, int], ...] | None = None
    previous_summary: str | None = None
    running_notification_sent = False
    all_running_notification_sent = False
    maximum_task_count = 0
    last_known_user = "unknown"
    last_known_is_array = False
    last_known_user_cancelled = False
    last_known_any_cancelled = False
    incomplete_accounting_polls = 0
    while True:
        snapshot = get_slurm_snapshot(job_id, host=host)
        maximum_task_count = max(maximum_task_count, snapshot.task_count)
        if snapshot.user and snapshot.user != "unknown":
            last_known_user = snapshot.user
        last_known_is_array = (
            last_known_is_array or snapshot.is_array or maximum_task_count > 1
        )
        # Accumulate cancellation evidence from every snapshot (including
        # partial ones during the grace period) so that prior observations
        # survive even if later sacct queries return empty results.
        if snapshot.user_cancelled:
            last_known_user_cancelled = True
        if snapshot.user_cancelled or any(
            state == "CANCELLED" for state, _ in snapshot.state_counts
        ):
            last_known_any_cancelled = True
        accounting_is_incomplete = (
            snapshot.task_count == 0 or snapshot.task_count < maximum_task_count
        )
        if accounting_is_incomplete:
            incomplete_accounting_polls += 1
            print(
                "[watcher] Waiting for complete squeue/sacct task records "
                f"({incomplete_accounting_polls}/{ACCOUNTING_GRACE_POLLS})."
            )
            if incomplete_accounting_polls >= ACCOUNTING_GRACE_POLLS:
                if not snapshot.all_terminal and snapshot.task_count > 0:
                    # Array was pruned/shrunk while active tasks are still running or pending.
                    maximum_task_count = snapshot.task_count
                    incomplete_accounting_polls = 0
                else:
                    noun = "Array" if last_known_is_array else "Job"
                    user_label = display_slurm_user(last_known_user)
                    # One final sacct attempt to catch late-arriving records.
                    acc_records = get_sacct_records(job_id, host=host)
                    user_cancelled = last_known_user_cancelled or any(
                        "CANCELLED BY" in r.state.upper() for r in acc_records
                    )
                    any_cancelled = last_known_any_cancelled or user_cancelled or any(
                        normalized_state(r.state) == "CANCELLED" for r in acc_records
                    )
                    if any_cancelled:
                        cancellation_label = " (user killed)" if user_cancelled else ""
                        message = (
                            f"🛑 {noun} `{job_id}` for `{user_label}`: "
                            f"CANCELLED{cancellation_label}"
                        )
                    else:
                        message = f"❓ {noun} `{job_id}` for `{user_label}` disappeared"
                    print(f"[watcher] {message}")
                    send_notification(message)
                    return 1
            time.sleep(POLL_INTERVAL)
            if incomplete_accounting_polls != 0:
                continue
        incomplete_accounting_polls = 0

        counts = dict(snapshot.state_counts)
        running_count = counts.get("RUNNING", 0) + counts.get("COMPLETING", 0)
        has_running_task = bool(running_count)
        previous_counts = dict(previous_signature or ())
        previously_had_pending_task = bool(
            previous_counts.get("PENDING", 0)
            or previous_counts.get("CONFIGURING", 0)
        )
        is_first_snapshot = previous_signature is None
        first_running_transition = (
            not running_notification_sent
            and has_running_task
            and (is_first_snapshot or previously_had_pending_task)
        )
        # Notify when running tasks reach the QoS concurrency limit,
        # but only if this did not coincide with the first-running notification.
        all_tasks_running_transition = (
            running_count >= concurrency_threshold
            and not all_running_notification_sent
            and not first_running_transition
        )
        should_notify = (
            is_first_snapshot
            or first_running_transition
            or all_tasks_running_transition
            or snapshot.all_terminal
        )

        if should_notify:
            message = _snapshot_message(
                snapshot,
                previous_summary=(
                    previous_summary if not is_first_snapshot else None
                ),
            )
            print(f"[watcher] {message}")
            send_notification(message)
        if first_running_transition:
            running_notification_sent = True
            # If concurrency is already at the threshold on the first
            # running snapshot, suppress the redundant notification.
            if running_count >= concurrency_threshold:
                all_running_notification_sent = True
        if all_tasks_running_transition:
            all_running_notification_sent = True

        previous_signature = snapshot.signature
        previous_summary = snapshot.summary

        if snapshot.all_terminal:
            print("[watcher] All allocation records are terminal. Done.")
            return 0 if snapshot.unsuccessful_task_count == 0 else 1
        time.sleep(POLL_INTERVAL)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified job notifier for SLURM (Juno) and local jobs."
    )
    subparsers = parser.add_subparsers(
        dest="mode",
        help="Mode: slurm or local",
        required=True,
    )

    slurm_parser = subparsers.add_parser("slurm", help="Watch a SLURM job")
    slurm_parser.add_argument(
        "--host",
        help="Query SLURM over SSH while sending notifications locally",
    )
    slurm_parser.add_argument(
        "--qos",
        default="juno-pri",
        help=(
            "QoS name to determine the concurrency notification threshold "
            "(juno-pri=8, others=4). Default: juno-pri"
        ),
    )
    slurm_parser.add_argument("job_id", help="The parent SLURM Job ID to watch")

    capacity_parser = subparsers.add_parser(
        "gpu-capacity", help="Watch aggregate GPU usage for the current user"
    )
    capacity_parser.add_argument(
        "--host",
        help="Query SLURM over SSH while sending notifications locally",
    )
    capacity_parser.add_argument("--partition", default="h200")

    local_parser = subparsers.add_parser("local", help="Run a local command and notify")
    local_parser.add_argument("--name", default="Local Job", help="Name of the job")
    local_parser.add_argument(
        "--include-command",
        action="store_true",
        help="Include the command in the start notification",
    )
    local_parser.add_argument(
        "--host-label",
        help="Execution-host label shown in notifications (for example, zy)",
    )
    local_parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="Command to run"
    )

    args = parser.parse_args()

    if args.mode == "slurm":
        return run_slurm(
            args.job_id,
            host=args.host,
            concurrency_threshold=concurrency_limit_for_qos(args.qos),
        )
    if args.mode == "gpu-capacity":
        return run_gpu_capacity(
            host=args.host,
            partition=args.partition,
        )
    if args.mode == "local":
        cmd = args.command
        if cmd and cmd[0] == "--":
            cmd = cmd[1:]
        run_local(
            args.name,
            cmd,
            include_command=args.include_command,
            host_label=args.host_label,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
