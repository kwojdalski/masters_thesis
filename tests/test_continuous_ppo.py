"""Test integration for Continuous PPO with TradingEnv backend."""

import os
from pathlib import Path

import pytest

from trading_rl.config import ExperimentConfig
from trading_rl.train_trading_agent import run_experiment_from_config


@pytest.mark.integration
def test_continuous_ppo_sine_wave():
    """Test full training loop for Continuous PPO on sine wave data."""

    # Path to the new config file
    config_path = "src/configs/scenarios/sine_wave_ppo_no_trend_continuous.yaml"

    # Verify config exists
    assert os.path.exists(config_path), f"Config file not found: {config_path}"

    # Load config to check settings before running
    config = ExperimentConfig.from_yaml(config_path)
    assert config.env.backend == "tradingenv"
    assert config.training.algorithm == "PPO"

    # Run the experiment (1 trial)
    try:
        experiment_name = run_experiment_from_config(config_path, n_trials=1)
        assert experiment_name == "sine_wave_ppo_no_trend_continuous"

        # Verify checkpoint was created
        log_dir = Path(config.logging.log_dir)
        checkpoint_path = log_dir / f"{experiment_name}_checkpoint.pt"
        assert checkpoint_path.exists(), "Checkpoint file was not created"

    except Exception as e:
        pytest.fail(f"Continuous PPO experiment failed with error: {e!s}")


if __name__ == "__main__":
    test_continuous_ppo_sine_wave()
