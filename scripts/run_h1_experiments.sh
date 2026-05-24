#!/usr/bin/env bash
# H1: Train all agents (TD3, DDPG, PPO, Random) and evaluate them against benchmarks.
#
# H1: A TD3-based agent using order-book-derived HFT features can achieve
# stronger out-of-sample risk-adjusted performance than selected benchmark
# strategies under the assumptions of the simulated environment.
#
# Usage:
#   bash scripts/run_h1_experiments.sh
#   bash scripts/run_h1_experiments.sh --skip-train          # evaluate only (checkpoints must exist)
#   bash scripts/run_h1_experiments.sh --skip-eval           # train only (quick smoke run)
#   bash scripts/run_h1_experiments.sh --parallel            # train all agents concurrently
#   bash scripts/run_h1_experiments.sh --verbose / -v        # enable debug logging
#   bash scripts/run_h1_experiments.sh --skip-train --parallel
#   EXTRA_TRAIN_ARGS="training.max_steps=50000" bash scripts/run_h1_experiments.sh
#   EXTRA_EVAL_ARGS="evaluation.eval_steps=500" bash scripts/run_h1_experiments.sh
#
# After completion, view results:
#   uv run python scripts/h1_performance_report.py

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

SKIP_TRAIN=0
SKIP_EVAL=0
PARALLEL=0
VERBOSE=0
for arg in "$@"; do
    [[ "$arg" == "--skip-train" ]] && SKIP_TRAIN=1
    [[ "$arg" == "--skip-eval"  ]] && SKIP_EVAL=1
    [[ "$arg" == "--parallel"   ]] && PARALLEL=1
    [[ "$arg" == "--verbose"    ]] && VERBOSE=1
    [[ "$arg" == "-v"           ]] && VERBOSE=1
done

VERBOSE_FLAG=""
[[ $VERBOSE -eq 1 ]] && VERBOSE_FLAG="--verbose"

EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"

# Build repeated --config-override flags from a space-separated override string.
# e.g. "evaluation.eval_fraction=null evaluation.eval_steps=500"
#   → --config-override evaluation.eval_fraction=null --config-override evaluation.eval_steps=500
_override_flags() {
    local args="$1"
    local flags=()
    for kv in $args; do
        flags+=(--config-override "$kv")
    done
    echo "${flags[@]}"
}

# Print a multitail (or tail -f) hint for monitoring parallel log files.
_watch_hint() {
    local label="$1"; shift
    local logs=("$@")
    echo ""
    if command -v multitail &>/dev/null; then
        echo "Monitor $label logs:"
        echo "  multitail -s ${#logs[@]} ${logs[*]}"
    else
        echo "Monitor $label logs (install multitail for split-pane view):"
        echo "  tail -f ${logs[*]}"
    fi
    echo ""
}

# ---------------------------------------------------------------------------
# Scenarios: (scenario_name, log_dir_name)
# log_dir_name must match what src/configs/h1_performance.yaml declares.
# ---------------------------------------------------------------------------
declare -a SCENARIOS=(
    "pooled/td3_hft_lob_state_space_pooled_streaming_selected_dsr"
    "pooled/ddpg_hft_lob_state_space_pooled_streaming_selected_dsr"
    "pooled/ppo_hft_lob_state_space_pooled_streaming_selected_dsr"
    "pooled/random_hft_lob_state_space_pooled_streaming_selected_dsr"
)

# ---------------------------------------------------------------------------
# Step 1: Train
# ---------------------------------------------------------------------------
if [[ $SKIP_TRAIN -eq 0 ]]; then
    echo "=== H1: Training ==="
    if [[ $PARALLEL -eq 1 ]]; then
        declare -a TRAIN_PIDS=() TRAIN_LOGS=()
        for SCENARIO in "${SCENARIOS[@]}"; do
            LOG_NAME="${SCENARIO##*/}"
            LOG_FILE="$LOG_DIR/${LOG_NAME}_train.log"
            TRAIN_LOGS+=("$LOG_FILE")
            echo "Training $SCENARIO  →  $LOG_FILE  (background)"
            NO_COLOR=1 uv run python "$REPO_ROOT/src/cli.py" train \
                -c "$SCENARIO" \
                ${EXTRA_TRAIN_ARGS:+$(_override_flags "$EXTRA_TRAIN_ARGS")} \
                ${VERBOSE_FLAG:+"$VERBOSE_FLAG"} \
                >"$LOG_FILE" 2>&1 &
            TRAIN_PIDS+=($!)
        done
        _watch_hint "training" "${TRAIN_LOGS[@]}"
        echo "Waiting for ${#TRAIN_PIDS[@]} training jobs..."
        TRAIN_FAILED=0
        for PID in "${TRAIN_PIDS[@]}"; do
            wait "$PID" || { echo "  training job PID=$PID failed"; TRAIN_FAILED=1; }
        done
        [[ $TRAIN_FAILED -eq 1 ]] && { echo "One or more training jobs failed."; exit 1; }
    else
        for SCENARIO in "${SCENARIOS[@]}"; do
            LOG_NAME="${SCENARIO##*/}"
            LOG_FILE="$LOG_DIR/${LOG_NAME}_train.log"
            echo "Training $SCENARIO  →  $LOG_FILE"
            uv run python "$REPO_ROOT/src/cli.py" train \
                -c "$SCENARIO" \
                ${EXTRA_TRAIN_ARGS:+$(_override_flags "$EXTRA_TRAIN_ARGS")} \
                ${VERBOSE_FLAG:+"$VERBOSE_FLAG"} \
                2>&1 | tee "$LOG_FILE"
            echo "  done."
        done
    fi
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 2: Evaluate (metrics + benchmarks → results.json in log_dir)
# ---------------------------------------------------------------------------
if [[ $SKIP_EVAL -eq 1 ]]; then
    echo "=== H1: Skipping evaluate (--skip-eval) ==="
else
echo "=== H1: Evaluating ==="
if [[ $PARALLEL -eq 1 ]]; then
    declare -a EVAL_PIDS=() EVAL_LOGS=()
    for SCENARIO in "${SCENARIOS[@]}"; do
        LOG_NAME="${SCENARIO##*/}"
        OUTPUT_DIR="$LOG_DIR/$LOG_NAME"
        LOG_FILE="$LOG_DIR/${LOG_NAME}_eval.log"
        EVAL_LOGS+=("$LOG_FILE")
        echo "Evaluating $SCENARIO  →  $OUTPUT_DIR  (background)"
        NO_COLOR=1 uv run python "$REPO_ROOT/src/cli.py" evaluate \
            -c "$SCENARIO" \
            --output-dir "$OUTPUT_DIR" \
            --only metrics \
            --only benchmarks \
            --only plots \
            ${EXTRA_EVAL_ARGS:+$(_override_flags "$EXTRA_EVAL_ARGS")} \
            ${VERBOSE_FLAG:+"$VERBOSE_FLAG"} \
            >"$LOG_FILE" 2>&1 &
        EVAL_PIDS+=($!)
    done
    _watch_hint "evaluation" "${EVAL_LOGS[@]}"
    echo "Waiting for ${#EVAL_PIDS[@]} evaluation jobs..."
    EVAL_FAILED=0
    for PID in "${EVAL_PIDS[@]}"; do
        wait "$PID" || { echo "  evaluation job PID=$PID failed"; EVAL_FAILED=1; }
    done
    [[ $EVAL_FAILED -eq 1 ]] && { echo "One or more evaluation jobs failed."; exit 1; }
else
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
            --only plots \
            ${EXTRA_EVAL_ARGS:+$(_override_flags "$EXTRA_EVAL_ARGS")} \
            ${VERBOSE_FLAG:+"$VERBOSE_FLAG"} \
            2>&1 | tee "$LOG_FILE"
        echo "  done."
    done
fi
echo ""

# ---------------------------------------------------------------------------
# Step 3: Report
# ---------------------------------------------------------------------------
echo "=== H1: Report ==="
uv run python "$REPO_ROOT/scripts/h1_performance_report.py"

# ---------------------------------------------------------------------------
# Step 4: Export to thesis snapshot
# Writes thesis/qmd/results/{experiment_name}/latest_finished/ so Quarto
# chapters can render metrics and plots without querying the MLflow database.
# ---------------------------------------------------------------------------
echo "=== H1: Export to thesis ==="
for SCENARIO in "${SCENARIOS[@]}"; do
    echo "  Exporting $SCENARIO ..."
    uv run python "$REPO_ROOT/scripts/export_eval_to_thesis.py" \
        --scenario "$SCENARIO"
done
echo "Thesis snapshots updated."
fi  # end SKIP_EVAL gate
