#!/usr/bin/env bash
# Run all three thesis experiments (TD3, DDPG, PPO) in parallel.
# Logs go to logs/<algo>.log; PIDs are printed so you can track or kill them.
#
# Usage:
#   bash scripts/train_all_thesis_experiments.sh
#
# To override config values for all runs (e.g. shorter training):
#   EXTRA_ARGS="training.max_steps=100000" bash scripts/train_all_thesis_experiments.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

EXTRA_ARGS="${EXTRA_ARGS:-}"

PID_FILE="$LOG_DIR/train_pids"
PIDS=()

_kill_all() {
    echo ""
    echo "Killing all experiments..."
    if [[ -f "$PID_FILE" ]]; then
        while IFS= read -r pid; do
            kill "$pid" 2>/dev/null && echo "  Killed $pid" || true
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    fi
    exit 1
}
trap _kill_all INT TERM

for ALGO in td3 ddpg ppo; do
    case "$ALGO" in
        td3)  SCENARIO="pooled/td3_hft_lob_state_space_pooled_streaming_selected_dsr" ;;
        ddpg) SCENARIO="pooled/ddpg_hft_lob_state_space_pooled_streaming_selected_dsr" ;;
        ppo)  SCENARIO="pooled/ppo_hft_lob_state_space_pooled_streaming_selected_dsr" ;;
    esac
    LOG_FILE="$LOG_DIR/${ALGO}.log"

    echo "Starting $ALGO → $LOG_FILE"

    if [[ -n "$EXTRA_ARGS" ]]; then
        uv run python "$REPO_ROOT/src/cli.py" train \
            -c "$SCENARIO" \
            --config-override "$EXTRA_ARGS" \
            > "$LOG_FILE" 2>&1 &
    else
        uv run python "$REPO_ROOT/src/cli.py" train \
            -c "$SCENARIO" \
            > "$LOG_FILE" 2>&1 &
    fi

    PIDS+=($!)
    echo "  PID: $!"
done

printf '%s\n' "${PIDS[@]}" > "$PID_FILE"

echo ""
echo "All three experiments running. PIDs: ${PIDS[*]}"
echo "To kill all: bash scripts/kill_experiments.sh"
echo ""

if command -v multitail &>/dev/null; then
    echo "Monitor logs in a new terminal:"
    echo "  multitail -s 3 $LOG_DIR/td3.log $LOG_DIR/ddpg.log $LOG_DIR/ppo.log"
else
    echo "Monitor logs in a new terminal:"
    echo "  tail -f $LOG_DIR/td3.log $LOG_DIR/ddpg.log $LOG_DIR/ppo.log"
fi
echo ""
echo "Waiting for all to finish..."

FAILED=0
for i in "${!PIDS[@]}"; do
    ALGO=$(echo "td3 ddpg ppo" | tr ' ' '\n' | sed -n "$((i+1))p")
    if wait "${PIDS[$i]}"; then
        echo "$ALGO finished successfully"
    else
        echo "$ALGO FAILED (exit code $?)" >&2
        FAILED=1
    fi
done

rm -f "$PID_FILE"

if [[ $FAILED -eq 1 ]]; then
    echo "One or more experiments failed. Check logs in $LOG_DIR/" >&2
    exit 1
fi

echo ""
echo "All experiments completed. Logs in $LOG_DIR/"
