#!/usr/bin/env bash
# Manage automatic Git commit/pull-rebase/push daemons for the workspace repositories.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

# List of target repositories: name|path|branch|threshold|interval
TARGETS=(
    "safefix|$REPO_ROOT|master|1|30"
    "27iclr|$REPO_ROOT/27iclr|main|1|30"
    "27cvpr|$REPO_ROOT/27cvpr|main|1|30"
    "camera_tmlr|$REPO_ROOT/Camera-TMLR-26-SafeFix|main|1|30"
)

get_pid() {
    local pid_file="$1"
    local repo="$2"
    if [ -f "$pid_file" ]; then
        local pid
        pid=$(sed -n '1p' "$pid_file" 2>/dev/null || true)
        local command_line
        command_line=$(tr '\0' ' ' 2>/dev/null < "/proc/${pid:-missing}/cmdline" || true)
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null &&
            [[ "$command_line" == *"$SCRIPT_DIR/auto_git_sync.py --repo $repo "* ]]; then
            echo "$pid"
            return 0
        fi
    fi
    return 1
}

get_log_file() {
    local name="$1"
    if [ "$name" = "safefix" ]; then
        echo "$LOG_DIR/auto_sync.log"
    else
        echo "$LOG_DIR/auto_sync_${name}.log"
    fi
}

start_target() {
    local name="$1"
    local repo="$2"
    local branch="$3"
    local threshold="$4"
    local interval="$5"
    local pid_file="$repo/.git/auto_sync.pid"
    local log_file
    log_file=$(get_log_file "$name")

    if [ ! -d "$repo/.git" ]; then
        echo "[AutoSync:$name] Directory $repo is not a Git repo, skipping."
        return 0
    fi

    if pid=$(get_pid "$pid_file" "$repo"); then
        echo "[AutoSync:$name] Daemon is already running (PID: $pid, repo: $repo)."
        return 0
    fi

    rm -f "$pid_file"
    setsid nohup python3 "$SCRIPT_DIR/auto_git_sync.py" \
        --repo "$repo" \
        --branch "$branch" \
        --threshold "$threshold" \
        --interval "$interval" \
        --daemon \
        >> "$log_file" 2>&1 < /dev/null &
    local pid=$!
    echo "$pid" > "$pid_file"
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
        echo "[AutoSync:$name] Started (PID: $pid, repo: $repo, branch: $branch, threshold: $threshold)."
    else
        rm -f "$pid_file"
        echo "[AutoSync:$name] Failed to start. Recent log ($log_file):"
        tail -n 20 "$log_file"
        return 1
    fi
}

stop_target() {
    local name="$1"
    local repo="$2"
    local pid_file="$repo/.git/auto_sync.pid"

    if pid=$(get_pid "$pid_file" "$repo"); then
        kill -TERM "$pid" 2>/dev/null || true
        for _ in {1..10}; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.5
        done
        kill -KILL "$pid" 2>/dev/null || true
        echo "[AutoSync:$name] Stopped (PID: $pid)."
    else
        echo "[AutoSync:$name] Daemon is not running."
    fi
    rm -f "$pid_file"
}

status_target() {
    local name="$1"
    local repo="$2"
    local pid_file="$repo/.git/auto_sync.pid"
    local log_file
    log_file=$(get_log_file "$name")

    if pid=$(get_pid "$pid_file" "$repo"); then
        echo "[AutoSync:$name] RUNNING (PID: $pid, repo: $repo)."
    else
        echo "[AutoSync:$name] STOPPED (repo: $repo)."
    fi
    if [ -f "$log_file" ]; then
        echo "  Recent log ($log_file):"
        tail -n 5 "$log_file" | sed 's/^/    /'
    fi
}

filter_targets() {
    local filter="${1:-all}"
    for target in "${TARGETS[@]}"; do
        IFS="|" read -r name repo branch threshold interval <<< "$target"
        if [ "$filter" = "all" ] || [ "$filter" = "$name" ]; then
            echo "$name|$repo|$branch|$threshold|$interval"
        fi
    done
}

ACTION="${1:-status}"
FILTER="${2:-all}"

case "$ACTION" in
    start)
        while IFS="|" read -r name repo branch threshold interval; do
            start_target "$name" "$repo" "$branch" "$threshold" "$interval"
        done < <(filter_targets "$FILTER")
        ;;
    stop)
        while IFS="|" read -r name repo branch threshold interval; do
            stop_target "$name" "$repo"
        done < <(filter_targets "$FILTER")
        ;;
    restart)
        while IFS="|" read -r name repo branch threshold interval; do
            stop_target "$name" "$repo"
            start_target "$name" "$repo" "$branch" "$threshold" "$interval"
        done < <(filter_targets "$FILTER")
        ;;
    status)
        while IFS="|" read -r name repo branch threshold interval; do
            status_target "$name" "$repo"
        done < <(filter_targets "$FILTER")
        ;;
    run)
        repo="${2:-$REPO_ROOT}"
        exec python3 "$SCRIPT_DIR/auto_git_sync.py" --repo "$repo" --daemon --threshold 2 --interval 30
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|run} [target_name]"
        exit 1
        ;;
esac
