#!/usr/bin/env bash
# H3: Sensitivity analysis — train all axis variants and compare.
#
# Three sensitivity axes, each holding the others constant:
#   Feature specification  — minimal (3), selected (10, baseline), full (33)
#   Reward design          — log return (baseline) vs Differential Sharpe (DSR)
#   Transaction cost       — 0 bp (baseline), 0.01 bp, 0.1 bp, 1 bp
#
# The "selected / log return / 0 bp" scenario is the shared baseline across
# all three axes and is trained only once.
#
# Usage:
#   bash scripts/run_h3_experiments.sh
#   bash scripts/run_h3_experiments.sh --skip-train          # evaluate only
#   bash scripts/run_h3_experiments.sh --skip-eval           # train only (quick smoke run)
#   bash scripts/run_h3_experiments.sh --parallel            # train all variants concurrently
#   bash scripts/run_h3_experiments.sh --verbose / -v        # enable debug logging
#   bash scripts/run_h3_experiments.sh --skip-train --parallel
#   EXTRA_TRAIN_ARGS="training.max_steps=50000" bash scripts/run_h3_experiments.sh
#   EXTRA_EVAL_ARGS="evaluation.eval_steps=500" bash scripts/run_h3_experiments.sh
#   EXTRA_TRAIN_ARGS="logging.debug_plots=true" EXTRA_EVAL_ARGS="logging.debug_plots=true" bash scripts/run_h3_experiments.sh
#
# After completion, view results:
#   uv run python scripts/h3_sensitivity_report.py

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

_override_flags() {
    local args="$1"
    local flags=()
    for kv in $args; do
        flags+=(--config-override "$kv")
    done
    echo "${flags[@]}"
}

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
# Scenarios: must produce log dirs matching src/configs/h3_sensitivity.yaml
# Baseline (selected / log return / 0 bp) is listed once; it is shared across
# all three axes.
# ---------------------------------------------------------------------------
declare -a SCENARIOS=(
    # Feature axis
    "pooled/td3_h3_features_minimal"
    "pooled/td3_hft_lob_state_space_pooled_streaming_selected"   # baseline
    "pooled/td3_h3_features_full"
    # Reward axis (baseline already above)
    "pooled/td3_hft_lob_state_space_pooled_streaming_selected_dsr"
    # Transaction cost axis (baseline already above)
    "pooled/td3_h3_fees_1e6"
    "pooled/td3_h3_fees_1e5"
    "pooled/td3_h3_fees_1e4"
)

# Build a deduplicated ordered list (preserves first occurrence).
declare -a UNIQUE_SCENARIOS=()
declare -A _SEEN=()
for _s in "${SCENARIOS[@]}"; do
    _key="${_s##*/}"
    if [[ -z "${_SEEN[$_key]+_}" ]]; then
        _SEEN[$_key]=1
        UNIQUE_SCENARIOS+=("$_s")
    fi
done

# ---------------------------------------------------------------------------
# Step 1: Train  (deduplicated — selected baseline trained only once)
# ---------------------------------------------------------------------------
if [[ $SKIP_TRAIN -eq 0 ]]; then
    echo "=== H3: Training ==="
    if [[ $PARALLEL -eq 1 ]]; then
        declare -a TRAIN_PIDS=() TRAIN_LOGS=()
        for SCENARIO in "${UNIQUE_SCENARIOS[@]}"; do
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
        for SCENARIO in "${UNIQUE_SCENARIOS[@]}"; do
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
# Step 2: Evaluate  (deduplicated — baseline evaluated only once)
# ---------------------------------------------------------------------------
if [[ $SKIP_EVAL -eq 1 ]]; then
    echo "=== H3: Skipping evaluate (--skip-eval) ==="
else
echo "=== H3: Evaluating ==="
if [[ $PARALLEL -eq 1 ]]; then
    declare -a EVAL_PIDS=() EVAL_LOGS=()
    for SCENARIO in "${UNIQUE_SCENARIOS[@]}"; do
        LOG_NAME="${SCENARIO##*/}"
        OUTPUT_DIR="$LOG_DIR/$LOG_NAME"
        LOG_FILE="$LOG_DIR/${LOG_NAME}_eval.log"
        EVAL_LOGS+=("$LOG_FILE")
        echo "Evaluating $SCENARIO  →  $OUTPUT_DIR  (background)"
        NO_COLOR=1 uv run python "$REPO_ROOT/src/cli.py" evaluate \
            -c "$SCENARIO" \
            --output-dir "$OUTPUT_DIR" \
            --only metrics \
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
    for SCENARIO in "${UNIQUE_SCENARIOS[@]}"; do
        LOG_NAME="${SCENARIO##*/}"
        OUTPUT_DIR="$LOG_DIR/$LOG_NAME"
        LOG_FILE="$LOG_DIR/${LOG_NAME}_eval.log"
        echo "Evaluating $SCENARIO  →  $OUTPUT_DIR"
        uv run python "$REPO_ROOT/src/cli.py" evaluate \
            -c "$SCENARIO" \
            --output-dir "$OUTPUT_DIR" \
            --only metrics \
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
echo "=== H3: Report ==="
uv run python "$REPO_ROOT/scripts/h3_sensitivity_report.py"

# ---------------------------------------------------------------------------
# Step 4: Export to thesis snapshot
# ---------------------------------------------------------------------------
echo "=== H3: Export to thesis ==="
for SCENARIO in "${UNIQUE_SCENARIOS[@]}"; do
    echo "  Exporting $SCENARIO ..."
    uv run python "$REPO_ROOT/scripts/export_eval_to_thesis.py" \
        --scenario "$SCENARIO"
done
echo "Thesis snapshots updated."
fi  # end SKIP_EVAL gate
