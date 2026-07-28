#!/usr/bin/env python3
"""Send a Slack or Discord webhook notification using only the standard library."""

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


KITTI_JOB_MESSAGE = re.compile(
    r"^SafeFix KITTI job (?P<job_id>\S+) "
    r"\((?P<mode>[^,]+), ratio=(?P<ratio>[^,]+), method=(?P<method>[^)]+)\) "
    r"(?P<state>completed|failed with exit (?P<exit_code>\d+)) on (?P<partition>\S+)\. "
    r"Artifacts: (?P<artifacts>.+)$"
)

ABNORMAL_COMPLETION_TERMS = (
    "failed",
    "cancelled",
    "canceled",
    "timeout",
    "out of memory",
    "out_of_memory",
    "node_fail",
    "preempted",
)


def is_success_completion_message(message: str) -> bool:
    """Return whether a message is a routine successful-completion notice."""
    normalized = message.strip().lower()
    if any(term in normalized for term in ABNORMAL_COMPLETION_TERMS):
        return False
    return "completed" in normalized or normalized.startswith("✅")


def enrich_kitti_job_message(message: str) -> str:
    """Turn a KITTI EXIT-trap message into a compact emoji-rich report."""
    match = KITTI_JOB_MESSAGE.match(message.strip())
    if match is None:
        return message

    fields = match.groupdict()
    artifact_dir = Path(fields["artifacts"])
    method = {
        "attribute": "Ours",
        "badge": "BADGE",
        "coreset": "CoreSet",
        "entropy": "Entropy",
        "random": "Random",
    }.get(fields["method"], fields["method"])
    if fields["state"] != "completed":
        return (
            f"❌ *SafeFix KITTI cost task `{fields['job_id']}` failed*\n"
            f"🧪 method=`{method}`, ratio=`{fields['ratio']}`, "
            f"exit=`{fields['exit_code']}`\n"
            f"📁 Artifacts: `{artifact_dir}`"
        )

    try:
        summary = json.loads((artifact_dir / "cost_summary.json").read_text(encoding="utf-8"))
        repair_timing_path = artifact_dir / "repair_timing.json"
        repair_timing = (
            json.loads(repair_timing_path.read_text(encoding="utf-8"))
            if repair_timing_path.is_file()
            else {}
        )
        seconds = summary["seconds"]
        counts = summary["counts"]["repair"]
        evaluation = summary.get("evaluation") or repair_timing.get("evaluation", {})
        map_50_95 = evaluation.get("metrics/mAP50-95(B)")
        metric_text = f", mAP50-95=`{float(map_50_95):.4f}`" if map_50_95 is not None else ""
        return (
            f"✅ *SafeFix KITTI cost task `{fields['job_id']}` completed*\n"
            f"📊 *Report:* method=`{method}`, ratio=`{summary['ratio']}`, "
            f"total=`{seconds['included_end_to_end_total'] / 60:.2f} min`\n"
            f"⏱️ annotation=`{seconds['source_attribute_annotation_total']:.2f}s`, "
            f"repair=`{seconds['repair_total']:.2f}s`\n"
            f"🧪 selected=`{counts['selected_original']} source + "
            f"{counts['selected_generated']} generated`, "
            f"test=`{counts['test_images']} images`{metric_text}\n"
            f"📁 Artifacts: `{artifact_dir}`"
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return f"✅ {message}"


def send_notification(webhook_url: str, message: str) -> bool:
    """Send one message and return whether the webhook accepted it."""
    if not webhook_url:
        print("Error: Webhook URL is empty.")
        return False

    if "discord.com" in webhook_url or "discordapp.com" in webhook_url:
        payload = {"content": message}
    else:
        payload = {"text": message}

    request = urllib.request.Request(
        webhook_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.getcode() in (200, 204):
                print("Notification sent successfully.")
                return True
            print(f"Failed to send notification. Response code: {response.getcode()}")
            return False
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="ignore")
        print(f"HTTP Error: {error.code} - {detail}")
    except Exception as error:
        print(f"Error occurred: {error}")
    return False


def main() -> int:
    webhook = os.environ.get("NOTIFICATION_WEBHOOK", "")
    arguments = sys.argv[1:]
    allow_success = os.environ.get("NOTIFY_SUCCESSES", "0") == "1"
    if "--allow-success" in arguments:
        arguments.remove("--allow-success")
        allow_success = True
    if arguments and arguments[0].startswith("http"):
        webhook = arguments.pop(0)
    message = " ".join(arguments) if arguments else "Test notification from Juno cluster."

    if is_success_completion_message(message) and not allow_success:
        print("Routine successful-completion notification suppressed by default.")
        return 0

    message = enrich_kitti_job_message(message)

    if not webhook:
        print("Error: Set NOTIFICATION_WEBHOOK or pass the webhook URL as the first argument.")
        return 1

    return 0 if send_notification(webhook, message) else 1


if __name__ == "__main__":
    raise SystemExit(main())
