#!/usr/bin/env python3
"""Summarize selection export wall-clock durations from worker logs."""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path


LINE_RE = re.compile(
    r"\[(?P<ts>[^\]]+)\] (?P<state>START|DONE)\s+gpu=(?P<gpu>\d+) "
    r"dataset=(?P<dataset>\w+) ratio=(?P<ratio>[0-9.]+) method=(?P<method>\w+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    starts = {}
    rows = []
    for path in sorted(args.log_dir.glob(f"selection_export_worker_gpu*_{args.stamp}.out")):
        for line in path.read_text(encoding="utf-8").splitlines():
            match = LINE_RE.search(line)
            if not match:
                continue
            item = match.groupdict()
            key = (item["dataset"], item["ratio"], item["method"], item["gpu"])
            ts = datetime.strptime(item["ts"], "%Y-%m-%d %H:%M:%S")
            if item["state"] == "START":
                starts[key] = ts
            elif key in starts:
                elapsed = (ts - starts[key]).total_seconds()
                rows.append(
                    {
                        "dataset": item["dataset"],
                        "ratio": item["ratio"],
                        "method": item["method"],
                        "gpu": item["gpu"],
                        "start_time": starts[key].isoformat(sep=" "),
                        "end_time": ts.isoformat(sep=" "),
                        "wall_time_sec": f"{elapsed:.3f}",
                        "worker_log": str(path),
                    }
                )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "ratio", "method", "gpu", "start_time", "end_time", "wall_time_sec", "worker_log"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
