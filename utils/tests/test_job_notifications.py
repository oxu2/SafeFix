"""Regression tests for parent-array and per-task notification policy."""

import importlib.util
import os
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_script_module(name: str, relative_path: str):
    """Load a repository script as a module without requiring a package."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class JobNotificationPolicyTest(unittest.TestCase):
    def test_live_array_query_keeps_pending_and_allocated_running_tasks(self):
        notifier = load_script_module("job_notifier_query_test", "scripts/job_notifier.py")
        queue_output = "\n".join(
            [
                "309101_2|309101|img-general-select|PENDING|h200|0:00||Resources",
                "309101_1|309101|img-general-select|RUNNING|h200|1:00|g-08-01|None",
                "999999_0|999999|other-job|RUNNING|h200|1:00|g-08-02|None",
            ]
        )
        completed_process = mock.Mock(stdout=queue_output)

        with mock.patch.object(
            notifier.subprocess,
            "run",
            return_value=completed_process,
        ) as run_mock:
            infos = notifier.get_job_infos("309101")

        self.assertEqual([info["state"] for info in infos], ["PENDING", "RUNNING"])
        self.assertIn("--me", run_mock.call_args.args[0])
        self.assertNotIn("-j", run_mock.call_args.args[0])

    def test_parent_array_success_sends_final_aggregate(self):
        notifier = load_script_module("job_notifier_test", "scripts/job_notifier.py")
        sent_messages = []

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(notifier, "get_job_infos", return_value=[])
            )
            stack.enter_context(
                mock.patch.object(
                    notifier,
                    "get_sacct_states",
                    return_value=["COMPLETED"] * 16,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    notifier,
                    "send_notification",
                    side_effect=lambda message: sent_messages.append(message) or True,
                )
            )
            notifier.watch_slurm("307993")

        self.assertEqual(
            sent_messages,
            ["✅ Array `307993` finished: 16 COMPLETED"],
        )

    def test_per_task_success_is_suppressed(self):
        notification = load_script_module(
            "test_notification_success_test",
            "scripts/test_notification.py",
        )
        sent_messages = []
        arguments = [
            "test_notification.py",
            "✅ ImageNet general-selection task 308789_1 completed",
        ]

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(sys, "argv", arguments))
            stack.enter_context(
                mock.patch.object(
                    notification,
                    "send_notification",
                    side_effect=lambda webhook, message: sent_messages.append(message) or True,
                )
            )
            self.assertEqual(notification.main(), 0)

        self.assertEqual(sent_messages, [])

    def test_per_task_failure_is_sent(self):
        notification = load_script_module(
            "test_notification_failure_test",
            "scripts/test_notification.py",
        )
        sent_messages = []
        arguments = [
            "test_notification.py",
            "❌ ImageNet general-selection task 308789_1 failed with exit 1",
        ]

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(sys, "argv", arguments))
            stack.enter_context(
                mock.patch.dict(
                    os.environ,
                    {"NOTIFICATION_WEBHOOK": "https://example.com/webhook"},
                )
            )
            stack.enter_context(
                mock.patch.object(
                    notification,
                    "send_notification",
                    side_effect=lambda webhook, message: sent_messages.append(message) or True,
                )
            )
            self.assertEqual(notification.main(), 0)

        self.assertEqual(
            sent_messages,
            ["❌ ImageNet general-selection task 308789_1 failed with exit 1"],
        )


if __name__ == "__main__":
    unittest.main()
