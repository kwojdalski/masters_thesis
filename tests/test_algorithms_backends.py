import os
from pathlib import Path

import pytest
from rich.console import Console

from cli.commands import TrainingCommand, TrainingParams
from logger import configure_logging


@pytest.mark.slow
def test_ppo_anytrading_forex():
    """
    Test PPO algorithm with gym_anytrading.forex backend (discrete actions).

    Reproduces the CLI command:
    LOG_LEVEL=DEBUG python src/cli.py train \\
      --config src/configs/scenarios/upward_trend_ppo.yaml \\
      --max-steps 500

    Backend: gym_anytrading.forex (discrete: 2 actions [short=0, long=1])
    Algorithm: PPO (works with both discrete and continuous)
    Data: Synthetic upward trend pattern
    """
    # Configure Logging
    os.environ["LOG_LEVEL"] = "DEBUG"

    configure_logging(component="test_ppo_anytrading", level="DEBUG")

    # Setup Parameters
    config_path = Path("src/configs/scenarios/upward_trend_ppo.yaml")

    params = TrainingParams(
        config_file=config_path,
        max_steps=600,  # Divisible by frames_per_batch (200)
        actor_lr=0.0003,
    )

    # Initialize Command
    console = Console(force_terminal=True)
    cmd = TrainingCommand(console)

    # Execute
    try:
        cmd.execute(params)
    except Exception as e:
        pytest.fail(f"PPO + anytrading.forex training failed with error: {e}")


@pytest.mark.slow
def test_ppo_tradingenv():
    """
    Test PPO algorithm with tradingenv backend (continuous portfolio weights).

    Reproduces the CLI command:
    LOG_LEVEL=DEBUG python src/cli.py train \\
      --config src/configs/scenarios/tradingenv_ppo_example.yaml \\
      --max-steps 500

    Backend: tradingenv (continuous portfolio allocation)
    Algorithm: PPO (works with both discrete and continuous)
    Data: Synthetic sine wave pattern
    """
    # Configure Logging
    os.environ["LOG_LEVEL"] = "DEBUG"
    configure_logging(component="test_ppo_tradingenv", level="DEBUG")

    # Setup Parameters
    config_path = Path("src/configs/scenarios/sine_wave_ppo_no_trend_tradingenv.yaml")

    params = TrainingParams(
        config_file=config_path,
        max_steps=512,  # Divisible by frames_per_batch (256)
        actor_lr=0.0003,
    )

    # Initialize Command
    console = Console(force_terminal=True)
    cmd = TrainingCommand(console)

    # Execute
    try:
        cmd.execute(params)
    except Exception as e:
        pytest.fail(f"PPO + tradingenv training failed with error: {e}")


@pytest.mark.slow
def test_ddpg_continuous():
    """
    Test DDPG algorithm with gym_trading_env.continuous backend.

    Reproduces the CLI command:
    LOG_LEVEL=DEBUG python src/cli.py train \\
      --config src/configs/scenarios/sine_wave.yaml \\
      --max-steps 500

    Backend: gym_trading_env.continuous (continuous actions mapped to discrete positions)
    Algorithm: DDPG (requires continuous action space)
    Data: Synthetic sine wave pattern

    Note: Original config uses gym_anytrading.forex but DDPG requires continuous
    action space. The AlgorithmicEnvironmentBuilder should auto-resolve to
    gym_trading_env.continuous or raise an error if incompatible.
    """
    # Configure Logging
    os.environ["LOG_LEVEL"] = "DEBUG"
    configure_logging(component="test_ddpg_continuous", level="DEBUG")

    # Setup Parameters
    # Note: Using sine_wave.yaml as base, but backend incompatibility
    # should be handled by the builder (auto-resolution or error)
    config_path = Path("src/configs/scenarios/sine_wave.yaml")

    params = TrainingParams(
        config_file=config_path,
        max_steps=600,  # Divisible by frames_per_batch (200)
        actor_lr=0.0005,
    )

    # Initialize Command
    console = Console(force_terminal=True)
    cmd = TrainingCommand(console)

    # Execute
    try:
        cmd.execute(params)
    except Exception as e:
        pytest.fail(f"DDPG + continuous training failed with error: {e}")


@pytest.mark.slow
def test_td3_continuous():
    """
    Test TD3 algorithm with gym_trading_env.continuous backend.

    Reproduces the CLI command:
    LOG_LEVEL=DEBUG python src/cli.py train \\
      --config src/configs/scenarios/sine_wave_td3_no_trend.yaml \\
      --max-steps 500

    Backend: gym_trading_env.continuous (continuous actions)
    Algorithm: TD3 (requires continuous action space)
    Data: Synthetic sine wave pattern with no trend
    """
    # Configure Logging
    os.environ["LOG_LEVEL"] = "DEBUG"
    configure_logging(component="test_td3_continuous", level="DEBUG")

    # Setup Parameters
    config_path = Path("src/configs/scenarios/sine_wave_td3_no_trend.yaml")

    params = TrainingParams(
        config_file=config_path,
        max_steps=600,  # Divisible by frames_per_batch (200)
        actor_lr=0.0003,
    )

    # Initialize Command
    console = Console(force_terminal=True)
    cmd = TrainingCommand(console)

    # Execute
    try:
        cmd.execute(params)
    except Exception as e:
        pytest.fail(f"TD3 + continuous training failed with error: {e}")


if __name__ == "__main__":
    # Allows running tests individually for debugging
    # Example: python tests/test_algorithms_backends.py
    import sys

    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name == "ppo_anytrading":
            test_ppo_anytrading_forex()
        elif test_name == "ppo_tradingenv":
            test_ppo_tradingenv()
        elif test_name == "ddpg":
            test_ddpg_continuous()
        elif test_name == "td3":
            test_td3_continuous()
        else:
            print(f"Unknown test: {test_name}")
            print("Available tests: ppo_anytrading, ppo_tradingenv, ddpg, td3")
    else:
        # Run all tests
        print("Running all tests...")
        test_ppo_anytrading_forex()
        test_ppo_tradingenv()
        test_ddpg_continuous()
        test_td3_continuous()
