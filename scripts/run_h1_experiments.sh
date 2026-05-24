#!/usr/bin/env bash
# H1: Train all agents (TD3, DDPG, PPO, Random) and evaluate them against benchmarks.
#
# H1: A TD3-based agent using order-book-derived HFT features can achieve
# stronger out-of-sample risk-adjusted performance than selected benchmark
# strategies under the assumptions of the simulated environment.
#
# Usage:
#   bash scripts/run_h1_experiments.sh
#   bash scripts/run_h1_experiments.sh --skip-train   # evaluate only (checkpoints must exist)
#   EXTRA_TRAIN_ARGS="training.max_steps=50000" bash scripts/run_h1_experiments.sh
#
# After completion, view results:
#   uv run python scripts/h1_performance_report.py

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

SKIP_TRAIN=0
for arg in "$@"; do
    [[ "$arg" == "--skip-train" ]] && SKIP_TRAIN=1
done

EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

# ---------------------------------------------------------------------------
# Scenarios: (scenario_name, log_dir_name)
# log_dir_name must match what src/configs/h1_performance.yaml declares.
# ---------------------------------------------------------------------------
declare -a SCENARIOS=(
    "pooled/td3_hft_lob_state_space_pooled_streaming_selected"
    "pooled/ddpg_hft_lob_state_space_pooled_streaming_selected_dsr"
    "pooled/ppo_hft_lob_state_space_pooled_streaming_selected_dsr"
    "pooled/random_hft_lob_state_space_pooled_streaming_5k"
)

# ---------------------------------------------------------------------------
# Step 1: Train
# ---------------------------------------------------------------------------
if [[ $SKIP_TRAIN -eq 0 ]]; then
    echo "=== H1: Training ==="
    for SCENARIO in "${SCENARIOS[@]}"; do
        LOG_NAME="${SCENARIO##*/}"
        LOG_FILE="$LOG_DIR/${LOG_NAME}_train.log"
        echo "Training $SCENARIO  →  $LOG_FILE"
        uv run python "$REPO_ROOT/src/cli.py" train \
            -c "$SCENARIO" \
            ${EXTRA_TRAIN_ARGS:+--config-override "$EXTRA_TRAIN_ARGS"} \
            2>&1 | tee "$LOG_FILE"
        echo "  done."
    done
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 2: Evaluate (metrics + benchmarks → results.json in log_dir)
# ---------------------------------------------------------------------------
echo "=== H1: Evaluating ==="
for SCENARIO in "${SCENARIOS[@]}"; do
    LOG_NAME="${SCENARIO##*/}"
    OUTPUT_DIR="$LOG_DIR/$LOG_NAME"
    LOG_FILE="$LOG_DIR/${LOG_NAME}_eval.log"
    echo "Evaluating $SCENARIO  →  $OUTPUT_DIR"
    uv run python "$REPO_ROOT/src/cli.py" evaluate \
        -c "$SCENARIO" \
        --output-dir "$OUTPUT_DIR" \
        --only metrics \
        --only benchmarks \
        2>&1 | tee "$LOG_FILE"
    echo "  done."
done
echo ""

# ---------------------------------------------------------------------------
# Step 3: Report
# ---------------------------------------------------------------------------
echo "=== H1: Report ==="
uv run python "$REPO_ROOT/scripts/h1_performance_report.py"
