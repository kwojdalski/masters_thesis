#!/bin/bash
# Run all scenarios with specified trials and steps

set -e  # Exit on error

# Default values
TRIALS=1
STEPS=1000
FILTER=""

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --trials)
            TRIALS="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --filter)
            FILTER="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--trials N] [--steps N] [--filter PATTERN]"
            echo "  --trials N      Number of trials per scenario (default: 1)"
            echo "  --steps N       Number of steps per trial (default: 1000)"
            echo "  --filter PATTERN  Only run scenarios matching grep pattern"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

ALL_SCENARIOS=(default sine_wave sine_wave_ppo_no_trend sine_wave_ppo_trend mean_reversion upward_trend_ddpg upward_trend_ppo white_noise_no_trend_ppo)

# Filter scenarios if pattern provided
if [[ -n "$FILTER" ]]; then
    SCENARIOS=()
    for scenario in "${ALL_SCENARIOS[@]}"; do
        if echo "$scenario" | grep -q "$FILTER"; then
            SCENARIOS+=("$scenario")
        fi
    done
    echo "Running filtered scenarios (pattern: '$FILTER') with $TRIALS trials and $STEPS steps each..."
else
    SCENARIOS=("${ALL_SCENARIOS[@]}")
    echo "Running all scenarios with $TRIALS trials and $STEPS steps each..."
fi

if [[ ${#SCENARIOS[@]} -eq 0 ]]; then
    echo "No scenarios match filter pattern: '$FILTER'"
    exit 1
fi

FAILED_SCENARIOS=()

for scenario in "${SCENARIOS[@]}"; do
    echo "Running scenario: $scenario"
    
    if ! python src/cli.py train --scenario "$scenario" --trials "$TRIALS" --max-steps "$STEPS" --generate-data; then
        echo "ERROR: Scenario $scenario failed!"
        FAILED_SCENARIOS+=("$scenario")
        continue
    fi
    
    echo "✓ Scenario $scenario completed successfully"
done

echo ""
if [ ${#FAILED_SCENARIOS[@]} -eq 0 ]; then
    echo "✓ All scenarios completed successfully!"
else
    echo "❌ Failed scenarios: ${FAILED_SCENARIOS[*]}"
    exit 1
fi
