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

declare -A SCENARIOS=(
    [td3]="pooled/td3_hft_lob_state_space_pooled_streaming_selected_dsr"
    [ddpg]="pooled/ddpg_hft_lob_state_space_pooled_streaming_selected_dsr"
    [ppo]="pooled/ppo_hft_lob_state_space_pooled_streaming_selected_dsr"
)

PIDS=()

for ALGO in td3 ddpg ppo; do
    SCENARIO="${SCENARIOS[$ALGO]}"
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
    echo "  PID: ${PIDS[-1]}"
done

echo ""
echo "All three experiments running. PIDs: ${PIDS[*]}"
echo "Tail logs with:"
echo "  tail -f $LOG_DIR/td3.log $LOG_DIR/ddpg.log $LOG_DIR/ppo.log"
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

if [[ $FAILED -eq 1 ]]; then
    echo "One or more experiments failed. Check logs in $LOG_DIR/" >&2
    exit 1
fi

echo ""
echo "All experiments completed. Logs in $LOG_DIR/"
