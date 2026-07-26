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
from typing import Dict


WEBHOOK_URL = os.environ.get("NOTIFICATION_WEBHOOK", "")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "30"))
TERMINAL_STATES = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
    "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "SUSPENDED",
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


def get_job_info(job_id: str) -> Dict[str, str]:
    """Read live SLURM state from squeue, returning an empty mapping if absent."""
    try:
        result = subprocess.run(
            ["squeue", "-j", job_id, "--noheader", "--format=%i|%j|%T|%P|%M|%N|%r"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception as error:
        print(f"[watcher] squeue error: {error}")
        return {}

    fields = result.stdout.strip().split("|")
    if not result.stdout.strip():
        return {}
    keys = ("job_id", "name", "state", "partition", "time", "nodelist", "reason")
    return {key: fields[index].strip() if index < len(fields) else "" for index, key in enumerate(keys)}


def get_sacct_state(job_id: str) -> str:
    """Read the terminal state after SLURM no longer lists the job in squeue."""
    try:
        result = subprocess.run(
            ["sacct", "-j", job_id, "--noheader", "--parsable2", "--format=State", "--allocations"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return result.stdout.strip().split("\n")[0] if result.stdout.strip() else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def normalized_state(state: str) -> str:
    return "CANCELLED" if state.startswith("CANCELLED") else state


def watch_slurm(job_id: str) -> None:
    """Poll a job until it reaches a terminal state."""
    print(f"[watcher] Watching SLURM job {job_id} every {POLL_INTERVAL}s")
    if not WEBHOOK_URL:
        print("[watcher] WARNING: NOTIFICATION_WEBHOOK is not set.")

    previous_state = None
    while True:
        info = get_job_info(job_id)
        if not info:
            final_state = get_sacct_state(job_id)
            if previous_state and final_state != previous_state:
                message = f"{STATE_EMOJI.get(normalized_state(final_state), '❓')} Job `{job_id}`: `{previous_state}` → `{final_state}`"
                print(f"[watcher] {message}")
                send_notification(message)
            break

        state = info["state"]
        if state != previous_state:
            prefix = "initial state" if previous_state is None else f"{previous_state} →"
            message = f"{STATE_EMOJI.get(normalized_state(state), '❓')} Job `{job_id}` ({info['name']}): {prefix} `{state}` on `{info['partition']}`"
            if info["nodelist"]:
                message += f" node `{info['nodelist']}`"
            if info["reason"] and info["reason"] != "None":
                message += f" ({info['reason']})"
            print(f"[watcher] {message}")
            send_notification(message)
            previous_state = state

        if normalized_state(state) in TERMINAL_STATES:
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
