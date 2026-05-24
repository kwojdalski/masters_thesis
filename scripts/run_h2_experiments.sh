#!/usr/bin/env bash
# H2: Train three feature-specification variants (minimal, selected, full) and compare.
#
# H2: Extending the state representation from basic snapshot features to a
# broader microstructure-aware feature set leads to improved out-of-sample
# performance under identical training and evaluation conditions.
#
# Variants (all use TD3, same data, same reward, same transaction cost):
#   minimal   — 3 features  (spread_bps, book_pressure_l0, microprice)
#   selected  — 10 features (IC-selected; this is also the H1/H3 baseline)
#   full      — 33 features (all LOB microstructure features)
#
# Usage:
#   bash scripts/run_h2_experiments.sh
#   bash scripts/run_h2_experiments.sh --skip-train          # evaluate only
#   bash scripts/run_h2_experiments.sh --parallel            # train all variants concurrently
#   bash scripts/run_h2_experiments.sh --skip-train --parallel
#   EXTRA_TRAIN_ARGS="training.max_steps=50000" bash scripts/run_h2_experiments.sh
#
# After completion, view results:
#   uv run python scripts/h2_feature_sensitivity_report.py

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

SKIP_TRAIN=0
PARALLEL=0
for arg in "$@"; do
    [[ "$arg" == "--skip-train" ]] && SKIP_TRAIN=1
    [[ "$arg" == "--parallel"   ]] && PARALLEL=1
done

EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

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
# Scenarios: must produce log dirs matching src/configs/h2_feature_sensitivity.yaml
# ---------------------------------------------------------------------------
declare -a SCENARIOS=(
    "pooled/td3_h3_features_minimal"
    "pooled/td3_hft_lob_state_space_pooled_streaming_selected"  # shared baseline
    "pooled/td3_h3_features_full"
)

# ---------------------------------------------------------------------------
# Step 1: Train
# ---------------------------------------------------------------------------
if [[ $SKIP_TRAIN -eq 0 ]]; then
    echo "=== H2: Training ==="
    if [[ $PARALLEL -eq 1 ]]; then
        declare -a TRAIN_PIDS=() TRAIN_LOGS=()
        for SCENARIO in "${SCENARIOS[@]}"; do
            LOG_NAME="${SCENARIO##*/}"
            LOG_FILE="$LOG_DIR/${LOG_NAME}_train.log"
            TRAIN_LOGS+=("$LOG_FILE")
            echo "Training $SCENARIO  →  $LOG_FILE  (background)"
            uv run python "$REPO_ROOT/src/cli.py" train \
                -c "$SCENARIO" \
                ${EXTRA_TRAIN_ARGS:+--config-override "$EXTRA_TRAIN_ARGS"} \
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
                ${EXTRA_TRAIN_ARGS:+--config-override "$EXTRA_TRAIN_ARGS"} \
                2>&1 | tee "$LOG_FILE"
            echo "  done."
        done
    fi
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 2: Evaluate
# ---------------------------------------------------------------------------
echo "=== H2: Evaluating ==="
if [[ $PARALLEL -eq 1 ]]; then
    declare -a EVAL_PIDS=() EVAL_LOGS=()
    for SCENARIO in "${SCENARIOS[@]}"; do
        LOG_NAME="${SCENARIO##*/}"
        OUTPUT_DIR="$LOG_DIR/$LOG_NAME"
        LOG_FILE="$LOG_DIR/${LOG_NAME}_eval.log"
        EVAL_LOGS+=("$LOG_FILE")
        echo "Evaluating $SCENARIO  →  $OUTPUT_DIR  (background)"
        uv run python "$REPO_ROOT/src/cli.py" evaluate \
            -c "$SCENARIO" \
            --output-dir "$OUTPUT_DIR" \
            --only metrics \
            --only plots \
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
            --only plots \
            2>&1 | tee "$LOG_FILE"
        echo "  done."
    done
fi
echo ""

# ---------------------------------------------------------------------------
# Step 3: Report
# ---------------------------------------------------------------------------
echo "=== H2: Report ==="
uv run python "$REPO_ROOT/scripts/h2_feature_sensitivity_report.py"

# ---------------------------------------------------------------------------
# Step 4: Export to thesis snapshot
# ---------------------------------------------------------------------------
echo "=== H2: Export to thesis ==="
for SCENARIO in "${SCENARIOS[@]}"; do
    echo "  Exporting $SCENARIO ..."
    uv run python "$REPO_ROOT/scripts/export_eval_to_thesis.py" \
        --scenario "$SCENARIO"
done
echo "Thesis snapshots updated."
