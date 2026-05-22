#!/usr/bin/env bash
# Kill all running thesis experiments launched by train_all_thesis_experiments.sh.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$REPO_ROOT/logs/train_pids"

if [[ ! -f "$PID_FILE" ]]; then
    echo "No running experiments found ($PID_FILE does not exist)."
    exit 0
fi

echo "Killing experiments from $PID_FILE..."
while IFS= read -r pid; do
    if kill "$pid" 2>/dev/null; then
        echo "  Killed $pid"
    else
        echo "  $pid already gone"
    fi
done < "$PID_FILE"

rm -f "$PID_FILE"
echo "Done."
