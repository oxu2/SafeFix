#!/usr/bin/env bash
# Manage the SafeFix automatic Git commit/pull-rebase/push daemon.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$REPO_ROOT/.git/auto_sync.pid"
LOG_DIR="$REPO_ROOT/logs"
LOG_FILE="$LOG_DIR/auto_sync.log"
mkdir -p "$LOG_DIR"

get_pid() {
    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(sed -n '1p' "$PID_FILE" 2>/dev/null || true)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
            return 0
        fi
    fi
    return 1
}

start_daemon() {
    if pid=$(get_pid); then
        echo "[AutoSync] Daemon is already running (PID: $pid)."
        return
    fi
    rm -f "$PID_FILE"
    setsid nohup python3 "$SCRIPT_DIR/auto_git_sync.py" --daemon --threshold 2 --interval 30 \
        >> "$LOG_FILE" 2>&1 < /dev/null &
    pid=$!
    echo "$pid" > "$PID_FILE"
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
        echo "[AutoSync] Started (PID: $pid)."
    else
        rm -f "$PID_FILE"
        tail -n 20 "$LOG_FILE"
        return 1
    fi
}

stop_daemon() {
    if pid=$(get_pid); then
        kill -TERM "$pid" 2>/dev/null || true
        for _ in {1..10}; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.5
        done
        kill -KILL "$pid" 2>/dev/null || true
        echo "[AutoSync] Stopped (PID: $pid)."
    else
        echo "[AutoSync] Daemon is not running."
    fi
    rm -f "$PID_FILE"
}

case "${1:-status}" in
    start) start_daemon ;;
    stop) stop_daemon ;;
    restart) stop_daemon; start_daemon ;;
    status)
        if pid=$(get_pid); then echo "[AutoSync] RUNNING (PID: $pid)."; else echo "[AutoSync] STOPPED."; fi
        [ -f "$LOG_FILE" ] && tail -n 10 "$LOG_FILE"
        ;;
    run) exec python3 "$SCRIPT_DIR/auto_git_sync.py" --daemon --threshold 2 --interval 30 ;;
    *) echo "Usage: $0 {start|stop|restart|status|run}"; exit 1 ;;
esac
