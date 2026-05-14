#!/bin/bash
# Run all scenarios with specified trials

set -e

TRIALS=1
FILTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --trials)
            TRIALS="$2"
            shift 2
            ;;
        --filter)
            FILTER="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--trials N] [--filter PATTERN]"
            echo "  --trials N        Number of trials per scenario (default: 1)"
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

ALL_SCENARIOS=(
    # BTC
    btc/td3_tradingenv
    btc/td3_tradingenv_reduced_features
    btc/ppo_tradingenv_all_features
    btc/ppo_tradingenv_all_features_log_return

    # Pooled HFT
    pooled/td3_hft_lob_state_space_pooled_streaming_selected
    pooled/ddpg_hft_lob_state_space_pooled_streaming
    pooled/ppo_hft_lob_state_space_pooled_streaming

    # Sine wave
    sine_wave/td3_no_trend_tradingenv
    sine_wave/td3_tradingenv_all_features
    sine_wave/td3_tradingenv_ohlcv_price_action
    sine_wave/ppo_no_trend_tradingenv
    sine_wave/ppo_no_trend_continuous
    sine_wave/ppo_tradingenv_ohlcv_only

    # Synthetic
    synthetic/upward_trend_td3_tradingenv
    synthetic/upward_trend_ddpg_tradingenv
)

if [[ -n "$FILTER" ]]; then
    SCENARIOS=()
    for scenario in "${ALL_SCENARIOS[@]}"; do
        if echo "$scenario" | grep -q "$FILTER"; then
            SCENARIOS+=("$scenario")
        fi
    done
    echo "Running filtered scenarios (pattern: '$FILTER') with $TRIALS trials each..."
else
    SCENARIOS=("${ALL_SCENARIOS[@]}")
    echo "Running all scenarios with $TRIALS trials each..."
fi

if [[ ${#SCENARIOS[@]} -eq 0 ]]; then
    echo "No scenarios match filter pattern: '$FILTER'"
    exit 1
fi

FAILED_SCENARIOS=()

for scenario in "${SCENARIOS[@]}"; do
    echo "Running scenario: $scenario"
    if ! uv run python src/cli.py train -c "$scenario" --trials "$TRIALS"; then
        echo "ERROR: Scenario $scenario failed!"
        FAILED_SCENARIOS+=("$scenario")
        continue
    fi
    echo "Scenario $scenario completed successfully"
done

echo ""
if [ ${#FAILED_SCENARIOS[@]} -eq 0 ]; then
    echo "All scenarios completed successfully."
else
    echo "Failed scenarios: ${FAILED_SCENARIOS[*]}"
    exit 1
fi
