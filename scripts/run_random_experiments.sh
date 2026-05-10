#!/bin/bash
"""
Run multiple TD3 experiments with different random seeds for statistical analysis.

Usage:
    bash scripts/run_random_experiments.sh <num_trials> <config_name>

Example:
    bash scripts/run_random_experiments.sh 5 pooled/td3_hft_lob_state_space_pooled_streaming
"""

NUM_TRIALS=$1
CONFIG_NAME=$2

if [ -z "$NUM_TRIALS" ] || [ -z "$CONFIG_NAME" ]; then
    echo "Usage: bash scripts/run_random_experiments.sh <num_trials> <config_name>"
    echo "Example: bash scripts/run_random_experiments.sh 5 pooled/td3_hft_lob_state_space_pooled_streaming"
    exit 1
fi

echo "🎲 Running $NUM_TRIALS experiments with different random seeds..."
echo "📊 Config: $CONFIG_NAME"
echo ""

# Create output directory
OUTPUT_DIR="experiments/random_seeds_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Run experiments with different seeds
for i in $(seq 1 $NUM_TRIALS); do
    # Generate random seed
    RANDOM_SEED=$RANDOM

    echo "🚀 Trial $i/$NUM_TRIALS with seed=$RANDOM_SEED"
    echo "   Started: $(date)"

    # Run experiment
    uv run python src/cli.py train \
        --config "$CONFIG_NAME" \
        -o seed=$RANDOM_SEED \
        -o data.train_size=50000 \
        -o data.validation_size=10000 \
        -o data.test_size=10000 \
        2>&1 | tee "$OUTPUT_DIR/trial_${i}_seed_${RANDOM_SEED}.log"

    echo "   Finished: $(date)"
    echo ""

    # Small delay between runs
    sleep 2
done

echo ""
echo "✅ All $NUM_TRIALS experiments completed!"
echo "📁 Results saved to: $OUTPUT_DIR"
echo ""
echo "📊 Next steps:"
echo "   1. Check MLflow for all runs: uv run python -m mlflow ui"
echo "   2. Analyze performance variance across seeds"
echo "   3. Plot confidence intervals for metrics"
