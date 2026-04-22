"""Shared string-valued enums for the trading_rl package.

Using StrEnum means enum members compare equal to their string values, so
YAML-loaded strings keep working in comparisons without any conversion:

    config.training.algorithm.upper() == Algorithm.PPO  # True when value is "PPO"
    reward_type == RewardType.LOG_RETURN                 # True when value is "log_return"
"""

from enum import StrEnum


class Algorithm(StrEnum):
    """Supported RL training algorithms."""

    PPO = "PPO"
    TD3 = "TD3"
    DDPG = "DDPG"


class RewardType(StrEnum):
    """Reward function variants for the trading environment."""

    LOG_RETURN = "log_return"
    DIFFERENTIAL_SHARPE = "differential_sharpe"


class EnvMode(StrEnum):
    """Experiment mode controlling which features are eligible."""

    MFT = "mft"
    HFT = "hft"
