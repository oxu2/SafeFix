#!/usr/bin/env python3
"""Watch a SLURM job and send Slack or Discord state-transition notifications.

Usage:
    export NOTIFICATION_WEBHOOK="https://hooks.slack.com/..."
    nohup python3 scripts/job_notifier.py slurm <JOB_ID> &
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from typing import Dict, List


WEBHOOK_URL = os.environ.get("NOTIFICATION_WEBHOOK", "")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "30"))
TERMINAL_STATES = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
    "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "SUSPENDED",
}
ABNORMAL_TERMINAL_STATES = {
    "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
    "PREEMPTED", "BOOT_FAIL", "DEADLINE", "SUSPENDED",
}
STATE_EMOJI = {
    "PENDING": "⏳", "RUNNING": "🚀", "COMPLETED": "✅", "FAILED": "❌",
    "CANCELLED": "🛑", "TIMEOUT": "⏰", "OUT_OF_MEMORY": "💥",
    "NODE_FAIL": "🔧", "PREEMPTED": "⚡",
}


def send_notification(message: str) -> bool:
    """Send one notification, or no-op when a webhook is not configured."""
    if not WEBHOOK_URL:
        return False
    payload = (
        {"content": message}
        if "discord.com" in WEBHOOK_URL or "discordapp.com" in WEBHOOK_URL
        else {"text": message}
    )
    request = urllib.request.Request(
        WEBHOOK_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.getcode() in (200, 204)
    except Exception as error:
        print(f"[watcher] notification error: {error}", file=sys.stderr)
        return False


def get_job_infos(job_id: str) -> List[Dict[str, str]]:
    """Read live SLURM records, including one row per array task."""
    try:
        result = subprocess.run(
            ["squeue", "--array", "-j", job_id, "--noheader", "--format=%i|%j|%T|%P|%M|%N|%r"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception as error:
        print(f"[watcher] squeue error: {error}")
        return []

    if not result.stdout.strip():
        return []
    keys = ("job_id", "name", "state", "partition", "time", "nodelist", "reason")
    records = []
    for line in result.stdout.strip().splitlines():
        fields = line.split("|")
        records.append({
            key: fields[index].strip() if index < len(fields) else ""
            for index, key in enumerate(keys)
        })
    return records


def get_sacct_states(job_id: str) -> List[str]:
    """Read terminal allocation states, preferring array-task records."""
    try:
        result = subprocess.run(
            ["sacct", "-j", job_id, "--noheader", "--parsable2", "--format=JobID,State", "--allocations"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        rows = []
        for line in result.stdout.strip().splitlines():
            fields = line.split("|", 1)
            if len(fields) == 2:
                rows.append((fields[0].strip(), fields[1].strip()))
        task_states = [state for record_id, state in rows if record_id.startswith(f"{job_id}_")]
        if task_states:
            return task_states
        parent_states = [state for record_id, state in rows if record_id == job_id]
        return parent_states or ["UNKNOWN"]
    except Exception:
        return ["UNKNOWN"]


def normalized_state(state: str) -> str:
    return "CANCELLED" if state.startswith("CANCELLED") else state


def state_summary(states: List[str]) -> str:
    counts = Counter(normalized_state(state) for state in states)
    return ", ".join(f"{count} {state}" for state, count in sorted(counts.items()))


def summary_emoji(states: List[str]) -> str:
    normalized = [normalized_state(state) for state in states]
    for state in ("FAILED", "OUT_OF_MEMORY", "TIMEOUT", "NODE_FAIL", "CANCELLED", "PREEMPTED"):
        if state in normalized:
            return STATE_EMOJI.get(state, "❓")
    if normalized and all(state == "COMPLETED" for state in normalized):
        return "✅"
    if "RUNNING" in normalized:
        return "🚀"
    if "PENDING" in normalized:
        return "⏳"
    return "❓"


def watch_slurm(job_id: str) -> None:
    """Poll a job until it reaches a terminal state."""
    print(f"[watcher] Watching SLURM job {job_id} every {POLL_INTERVAL}s")
    if not WEBHOOK_URL:
        print("[watcher] WARNING: NOTIFICATION_WEBHOOK is not set.")

    previous_signature = None
    while True:
        infos = get_job_infos(job_id)
        if not infos:
            final_states = get_sacct_states(job_id)
            final_signature = tuple(sorted(normalized_state(state) for state in final_states))
            if previous_signature != final_signature:
                noun = "Array" if len(final_states) > 1 else "Job"
                message = f"{summary_emoji(final_states)} {noun} `{job_id}` finished: {state_summary(final_states)}"
                print(f"[watcher] {message}")
                if any(
                    normalized_state(state) in ABNORMAL_TERMINAL_STATES
                    for state in final_states
                ):
                    send_notification(message)
                else:
                    print("[watcher] Successful terminal notification suppressed by default.")
            break

        states = [info["state"] for info in infos]
        signature = tuple(sorted(normalized_state(state) for state in states))
        if signature != previous_signature:
            if len(infos) == 1 and "_" not in infos[0]["job_id"]:
                info = infos[0]
                prefix = "initial state" if previous_signature is None else "state"
                message = f"{summary_emoji(states)} Job `{job_id}` ({info['name']}): {prefix} `{info['state']}` on `{info['partition']}`"
                if info["nodelist"]:
                    message += f" node `{info['nodelist']}`"
                if info["reason"] and info["reason"] != "None":
                    message += f" ({info['reason']})"
            else:
                name = infos[0]["name"] if infos else ""
                partition = infos[0]["partition"] if infos else "unknown"
                message = (
                    f"{summary_emoji(states)} Array `{job_id}` ({name}): "
                    f"{state_summary(states)} on `{partition}`"
                )
            print(f"[watcher] {message}")
            send_notification(message)
            previous_signature = signature

        if states and all(normalized_state(state) in TERMINAL_STATES for state in states):
            break
        time.sleep(POLL_INTERVAL)
    print("[watcher] Done.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch a Juno SLURM job and notify on state changes.")
    parser.add_argument("mode", choices=("slurm",))
    parser.add_argument("job_id")
    arguments = parser.parse_args()
    watch_slurm(arguments.job_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
