#!/usr/bin/env bash
# H4: Learning progression check — run multiple short trials to determine if
# the algorithm can learn and outperform a dummy strategy.
#
# Runs N trials of the same algorithm with limited steps (default 200k),
# then merges results to analyze learning progression and baseline comparison.
#
# Usage:
#   bash scripts/run_h4_experiments.sh
#   bash scripts/run_h4_experiments.sh --scenario <scenario_name>
#   bash scripts/run_h4_experiments.sh --trials 5 --steps 200000
#   EXTRA_TRAIN_ARGS="training.max_steps=100000" bash scripts/run_h4_experiments.sh
#   bash scripts/run_h4_experiments.sh --skip-train          # analyze only
#   bash scripts/run_h4_experiments.sh --verbose / -v        # enable debug logging
#   bash scripts/run_h4_experiments.sh --skip-guardrails     # skip pre-flight guardrails check

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

# Defaults
SCENARIO="${SCENARIO:-pooled/td3_hft_lob_state_space_pooled_streaming_selected_dsr}"
N_TRIALS="${N_TRIALS:-5}"
STEPS="${STEPS:-200000}"

SKIP_TRAIN=0
SKIP_EVAL=0
VERBOSE=0
SKIP_GUARDRAILS=0
for arg in "$@"; do
    [[ "$arg" == "--skip-train" ]] && SKIP_TRAIN=1
    [[ "$arg" == "--skip-eval"  ]] && SKIP_EVAL=1
    [[ "$arg" == "--verbose"    ]] && VERBOSE=1
    [[ "$arg" == "-v"           ]] && VERBOSE=1
    [[ "$arg" == "--skip-guardrails" ]] && SKIP_GUARDRAILS=1
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

# ---------------------------------------------------------------------------
# Step 0: Pre-flight guardrails check
# ---------------------------------------------------------------------------
if [[ $SKIP_GUARDRAILS -eq 0 ]]; then
    echo "=== Pre-flight: Checking guardrails for $SCENARIO ==="
    local LOG_NAME="${SCENARIO##*/}"
    local LOG_FILE="$LOG_DIR/${LOG_NAME}_guardrails.log"
    if uv run python "$REPO_ROOT/src/cli.py" validate guardrails \
        -c "$SCENARIO" \
        ${VERBOSE_FLAG:+"$VERBOSE_FLAG"} \
        >"$LOG_FILE" 2>&1; then
        echo "  [PASS] $SCENARIO"
    else
        echo "  [FAIL] $SCENARIO (see $LOG_FILE)"
        echo ""
        echo "Fix the guardrail issues or run with --skip-guardrails to proceed anyway."
        exit 1
    fi
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 1: Train (multiple trials)
# ---------------------------------------------------------------------------
if [[ $SKIP_TRAIN -eq 0 ]]; then
    echo "=== H4: Training $N_TRIALS trials of $SCENARIO (max_steps=$STEPS) ==="
    TRAIN_ARGS=(
        -c "$SCENARIO"
        --trials "$N_TRIALS"
        --config-override "training.max_steps=$STEPS"
        ${EXTRA_TRAIN_ARGS:+$(_override_flags "$EXTRA_TRAIN_ARGS")}
        ${VERBOSE_FLAG:+"$VERBOSE_FLAG"}
    )
    uv run python "$REPO_ROOT/src/cli.py" train "${TRAIN_ARGS[@]}"
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 2: Evaluate all trials
# ---------------------------------------------------------------------------
if [[ $SKIP_EVAL -eq 1 ]]; then
    echo "=== H4: Skipping evaluate (--skip-eval) ==="
else
    echo "=== H4: Evaluating all $N_TRIALS trials ==="

    # Find all trial checkpoints from MLflow or logs
    # For simplicity, we rely on the CLI to auto-discover latest checkpoints
    EVAL_ARGS=(
        -c "$SCENARIO"
        --config-override "training.max_steps=$STEPS"
        ${EXTRA_EVAL_ARGS:+$(_override_flags "$EXTRA_EVAL_ARGS")}
        ${VERBOSE_FLAG:+"$VERBOSE_FLAG"}
    )
    uv run python "$REPO_ROOT/src/cli.py" evaluate "${EVAL_ARGS[@]}"
    echo ""

    # ---------------------------------------------------------------------------
    # Step 3: Report
    # ---------------------------------------------------------------------------
    echo "=== H4: Learning progression report ==="
    uv run python "$REPO_ROOT/scripts/h4_learning_progression_report.py" \
        --scenario "$SCENARIO" \
        --n-trials "$N_TRIALS" \
        --max-steps "$STEPS"

    # ---------------------------------------------------------------------------
    # Step 4: Export to thesis snapshot
    # ---------------------------------------------------------------------------
    echo "=== H4: Export to thesis ==="
    uv run python "$REPO_ROOT/scripts/export_eval_to_thesis.py" \
        --scenario "$SCENARIO"
    echo "Thesis snapshots updated."
fi