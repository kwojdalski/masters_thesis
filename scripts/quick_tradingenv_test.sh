#!/bin/bash
# Quick test of tradingenv backend integration

echo "Testing TradingEnv backend integration..."
echo ""

# Activate virtual environment
source .venv/bin/activate

# Set short training run for testing
export PYTHONPATH="src:$PYTHONPATH"

# Create temporary test config with minimal steps
cat > /tmp/test_tradingenv.yaml << 'EOF'
experiment_name: "tradingenv_quick_test"
seed: 42

data:
  data_path: "./data/raw/synthetic/sine_wave_ppo_no_trend.parquet"
  download_data: false
  train_size: 100
  no_features: false

env:
  name: "TEST_PORTFOLIO"
  backend: "tradingenv"
  trading_fees: 0.001
  price_columns: ["close"]

training:
  algorithm: "PPO"
  max_steps: 200  # Very short for testing
  eval_interval: 100
  log_interval: 50

logging:
  log_dir: "logs/tradingenv_quick_test"
  log_level: "INFO"
EOF

# Run quick test
python src/trading_rl/train_trading_agent.py /tmp/test_tradingenv.yaml

echo ""
echo "Test complete! If it ran without errors, tradingenv backend is working."
echo "Check logs/tradingenv_quick_test/ for results."
