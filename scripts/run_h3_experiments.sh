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
#   bash scripts/run_h3_experiments.sh --skip-train   # evaluate only
#   EXTRA_TRAIN_ARGS="training.max_steps=50000" bash scripts/run_h3_experiments.sh
#
# After completion, view results:
#   uv run python scripts/h3_sensitivity_report.py

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

# ---------------------------------------------------------------------------
# Step 1: Train  (deduplicated — selected baseline trained only once)
# ---------------------------------------------------------------------------
if [[ $SKIP_TRAIN -eq 0 ]]; then
    echo "=== H3: Training ==="
    declare -A TRAINED=()
    for SCENARIO in "${SCENARIOS[@]}"; do
        LOG_NAME="${SCENARIO##*/}"
        if [[ -n "${TRAINED[$LOG_NAME]+_}" ]]; then
            echo "Skipping $SCENARIO  (already trained this session)"
            continue
        fi
        TRAINED[$LOG_NAME]=1
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
# Step 2: Evaluate  (deduplicated — baseline evaluated only once)
# ---------------------------------------------------------------------------
echo "=== H3: Evaluating ==="
declare -A EVALUATED=()
for SCENARIO in "${SCENARIOS[@]}"; do
    LOG_NAME="${SCENARIO##*/}"
    if [[ -n "${EVALUATED[$LOG_NAME]+_}" ]]; then
        echo "Skipping eval $SCENARIO  (already evaluated this session)"
        continue
    fi
    EVALUATED[$LOG_NAME]=1
    OUTPUT_DIR="$LOG_DIR/$LOG_NAME"
    LOG_FILE="$LOG_DIR/${LOG_NAME}_eval.log"
    echo "Evaluating $SCENARIO  →  $OUTPUT_DIR"
    uv run python "$REPO_ROOT/src/cli.py" evaluate \
        -c "$SCENARIO" \
        --output-dir "$OUTPUT_DIR" \
        --only metrics \
        2>&1 | tee "$LOG_FILE"
    echo "  done."
done
echo ""

# ---------------------------------------------------------------------------
# Step 3: Report
# ---------------------------------------------------------------------------
echo "=== H3: Report ==="
uv run python "$REPO_ROOT/scripts/h3_sensitivity_report.py"
