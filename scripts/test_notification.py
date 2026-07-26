#!/usr/bin/env python3
"""Send a Slack or Discord webhook notification using only the standard library."""

import json
import os
import sys
import urllib.error
import urllib.request


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
    if arguments and arguments[0].startswith("http"):
        webhook = arguments.pop(0)
    message = " ".join(arguments) if arguments else "Test notification from Juno cluster."

    if not webhook:
        print("Error: Set NOTIFICATION_WEBHOOK or pass the webhook URL as the first argument.")
        return 1

    return 0 if send_notification(webhook, message) else 1


if __name__ == "__main__":
    raise SystemExit(main())
