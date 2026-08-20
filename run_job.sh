#!/usr/bin/env bash
# Run the UC1 pipeline as a detached background job that survives SSH
# disconnects, with commands to track its progress.
#
#   ./run_job.sh start [main.py args...]    # e.g. ./run_job.sh start --force
#   ./run_job.sh status                     # running? + per-step status + latest log lines
#   ./run_job.sh follow                     # live log stream (Ctrl-C detaches; job keeps running)
#   ./run_job.sh stop                       # stop the job (completed steps stay completed)
#
# The job is started with setsid+nohup, so closing the terminal or losing
# the SSH connection does not kill it. Progress is visible three ways:
# logs.txt (timestamped, written live by the pipeline), pipeline_status.json
# (per-step completion), and each step's files appearing under output/.
set -euo pipefail
cd "$(dirname "$0")"

PID_FILE=".pipeline_job.pid"
CONSOLE_LOG="job_console.log"   # catches crashes that happen BEFORE logs.txt teeing starts

activate_venv() {
  if [ -z "${VIRTUAL_ENV:-}" ]; then
    for v in .testVenv .synthenv .venv venv; do
      if [ -f "$v/bin/activate" ]; then
        # shellcheck disable=SC1090
        source "$v/bin/activate"
        echo "Activated virtualenv: $v"
        return 0
      fi
    done
    echo "⚠️  No virtualenv active or found (.testVenv/.synthenv/.venv/venv) -- using $(command -v python3 || true)"
  fi
  return 0
}

py() {
  if command -v python >/dev/null 2>&1; then python "$@"; else python3 "$@"; fi
}

running_pid() {
  [ -f "$PID_FILE" ] || return 1
  local pid
  pid=$(cat "$PID_FILE")
  kill -0 "$pid" 2>/dev/null || return 1
  echo "$pid"
}

case "${1:-}" in
  start)
    shift || true
    if pid=$(running_pid); then
      echo "A pipeline job is already running (PID $pid). Use ./run_job.sh status."
      exit 1
    fi
    activate_venv
    echo "Running preflight before detaching..."
    if ! py main.py --preflight; then
      echo "❌ Preflight FAILED -- not starting the job. Fix the items above first."
      exit 1
    fi
    # logs.txt is opened in overwrite mode by the pipeline; keep the
    # previous run's log instead of losing it.
    if [ -f logs.txt ]; then
      mkdir -p logs
      archived="logs/logs-$(date +%Y%m%d-%H%M%S).txt"
      mv logs.txt "$archived"
      echo "Previous logs.txt archived to $archived"
    fi
    PYBIN=$(command -v python || command -v python3)
    setsid nohup "$PYBIN" main.py "$@" > "$CONSOLE_LOG" 2>&1 < /dev/null &
    echo $! > "$PID_FILE"
    sleep 3
    if pid=$(running_pid); then
      echo ""
      echo "✅ Pipeline job started (PID $pid) with args: ${*:-<none>}"
      echo "   It is now safe to disconnect from the terminal."
      echo "   Track it with:"
      echo "     ./run_job.sh status    # step completion + latest log lines"
      echo "     ./run_job.sh follow    # live log stream (Ctrl-C detaches, job keeps running)"
    else
      echo "❌ Job exited within 3 seconds -- startup failure. Output was:"
      tail -n 40 "$CONSOLE_LOG"
      rm -f "$PID_FILE"
      exit 1
    fi
    ;;

  status)
    if pid=$(running_pid); then
      echo "🟢 RUNNING (PID $pid, elapsed $(ps -o etime= -p "$pid" | tr -d ' '))"
    else
      echo "⚪ NOT RUNNING (finished, stopped, or never started)"
    fi
    echo ""
    activate_venv >/dev/null 2>&1
    py main.py --status 2>/dev/null || echo "(could not read step status)"
    if [ -f logs.txt ]; then
      echo ""
      echo "--- last 15 log lines (logs.txt) ---"
      tail -n 15 logs.txt
    fi
    ;;

  follow)
    [ -f logs.txt ] || { echo "No logs.txt yet -- has the job started?"; exit 1; }
    exec tail -n 30 -f logs.txt
    ;;

  stop)
    if pid=$(running_pid); then
      echo "Stopping pipeline job (PID $pid)..."
      kill -- "-$pid" 2>/dev/null || kill "$pid"
      rm -f "$PID_FILE"
      echo "Stopped. pipeline_status.json keeps completed steps; the interrupted"
      echo "step is marked failed, so a plain restart resumes from it."
    else
      echo "No running job."
      rm -f "$PID_FILE"
    fi
    ;;

  *)
    echo "Usage: $0 {start [main.py args...] | status | follow | stop}"
    echo ""
    echo "  start --force        full 12-hour paper run (reruns every step)"
    echo "  start                resume: runs only steps not yet completed"
    echo "  status               is it running + per-step status + log tail"
    echo "  follow               stream the timestamped log live"
    echo "  stop                 stop the job without losing completed steps"
    exit 1
    ;;
esac
