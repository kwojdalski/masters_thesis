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
#   # Per-trial and per-sweep wall-clock budgets (seconds); a trial that hits
#   # its budget stops cleanly with whatever it trained so far, the sweep
#   # stops launching further trials once its own budget is exceeded:
#   EXTRA_TRAIN_ARGS="training.max_train_seconds=600 max_total_seconds=3000" bash scripts/run_h4_experiments.sh
#   bash scripts/run_h4_experiments.sh --skip-train          # analyze only
#   bash scripts/run_h4_experiments.sh --verbose / -v        # enable debug logging
#   bash scripts/run_h4_experiments.sh --skip-guardrails     # skip pre-flight guardrails check
#
# Evaluation overhead (issue #362): each trial overrides evaluation.eval_fraction
# to 0.05, training.temp_eval.max_steps to 5000, and evaluation.skip_final_eval
# to true, since the scenario's own defaults (full-validation periodic evals,
# an unconditional post-training train+val+test pass) are tuned for a single
# final reported run, not N short exploratory trials, and previously added far
# more rollout steps than the nominal training budget. To restore the
# scenario's own eval settings for a specific run, override them back via
# EXTRA_TRAIN_ARGS, e.g.:
#   EXTRA_TRAIN_ARGS="evaluation.eval_fraction=1.0 evaluation.skip_final_eval=false" \
#       bash scripts/run_h4_experiments.sh

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
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenario) SCENARIO="$2"; shift 2 ;;
        --trials)   N_TRIALS="$2"; shift 2 ;;
        --steps)    STEPS="$2"; shift 2 ;;
        --skip-train)      SKIP_TRAIN=1; shift ;;
        --skip-eval)       SKIP_EVAL=1; shift ;;
        --verbose|-v)      VERBOSE=1; shift ;;
        --skip-guardrails) SKIP_GUARDRAILS=1; shift ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
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
    LOG_NAME="${SCENARIO##*/}"
    LOG_FILE="$LOG_DIR/${LOG_NAME}_guardrails.log"
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
        # H4 is an exploratory learning-progression sweep, not the final
        # reported evaluation -- the scenario's own eval_fraction=1.0 and
        # temp_eval defaults (full-validation-length rollouts) are tuned for
        # a single final run, not N short trials. Without these overrides,
        # periodic mid-training evals and the unconditional post-training
        # evaluate_all_splits() pass add far more rollout steps than the
        # nominal training budget (see issue #362). skip_final_eval is safe
        # here because Step 2 below runs its own separate `evaluate` pass.
        --config-override "evaluation.eval_fraction=0.05"
        --config-override "training.temp_eval.max_steps=5000"
        --config-override "evaluation.skip_final_eval=true"
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
