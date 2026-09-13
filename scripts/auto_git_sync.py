#!/usr/bin/env python3
"""Poll a Git worktree and commit/push once enough files have changed."""

import argparse
import datetime as dt
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("auto_git_sync")


def run(args, cwd, check=False):
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=check)


def repo_root(path=None):
    cwd = Path(path).resolve() if path else None
    result = run(["git", "rev-parse", "--show-toplevel"], cwd)
    if result.returncode:
        log.error("Not inside a Git repository: %s", result.stderr.strip())
        sys.exit(1)
    return Path(result.stdout.strip())


def changed(root):
    result = run(["git", "status", "--porcelain"], root)
    if result.returncode:
        log.warning("git status failed: %s", result.stderr.strip())
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def conflicted(root):
    git_dir = root / ".git"
    return any((git_dir / name).exists() for name in (
        "MERGE_HEAD", "REBASE_HEAD", "rebase-apply", "rebase-merge",
        "CHERRY_PICK_HEAD",
    ))


def sync(root, branch, files, no_push=False):
    if conflicted(root):
        log.warning("Merge/rebase conflict detected; skipping auto-sync.")
        return False
    count = len(files)
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = "auto: sync %d changed files (%s)\n\nFiles:\n%s" % (
        count, stamp, "\n".join("- " + line for line in files[:20])
    )
    try:
        run(["git", "add", "-A"], root, check=True)
        run(["git", "commit", "-m", message], root, check=True)
        if no_push:
            return True
        pull = run(["git", "pull", "--rebase", "origin", branch], root)
        if pull.returncode:
            log.warning("git pull --rebase failed: %s", (pull.stderr or pull.stdout).strip())
            if conflicted(root):
                run(["git", "rebase", "--abort"], root)
                return False
        push = run(["git", "push", "origin", branch], root)
        if push.returncode:
            log.error("git push failed: %s", (push.stderr or push.stdout).strip())
            return False
        log.info("Committed and pushed %d files to origin/%s.", count, branch)
        return True
    except subprocess.CalledProcessError as error:
        log.error("Git command failed: %s", error)
        return False


def daemon(root, threshold, interval, debounce, branch, no_push):
    running = True

    def stop(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    log.info("Started: threshold=%d interval=%ds debounce=%ds branch=%s",
             threshold, interval, debounce, branch)
    while running:
        files = changed(root)
        if len(files) >= threshold:
            log.info("Threshold reached: %d files; waiting %ds.", len(files), debounce)
            time.sleep(debounce)
            files = changed(root)
            if len(files) >= threshold:
                sync(root, branch, files, no_push)
        for _ in range(interval):
            if not running:
                break
            time.sleep(1)
    log.info("Stopped.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", "-r", default="", help="Path to repository")
    parser.add_argument("--threshold", "-t", type=int, default=2)
    parser.add_argument("--interval", "-i", type=int, default=30)
    parser.add_argument("--debounce", type=int, default=10)
    parser.add_argument("--branch", default="")
    parser.add_argument("--daemon", "-d", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()
    root = repo_root(args.repo)
    branch = args.branch or run(["git", "branch", "--show-current"], root).stdout.strip() or "master"
    if args.daemon:
        daemon(root, args.threshold, args.interval, args.debounce, branch, args.no_push)
    else:
        files = changed(root)
        log.info("Changed files: %d (threshold: %d)", len(files), args.threshold)
        if len(files) >= args.threshold:
            sync(root, branch, files, args.no_push)


if __name__ == "__main__":
    main()
