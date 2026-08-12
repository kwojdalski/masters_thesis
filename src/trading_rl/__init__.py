"""Trading RL package with modular components for DDPG, PPO, and TD3 trading."""

from trading_rl.config import (
    DEFAULT_INITIAL_PORTFOLIO_VALUE,
    DataConfig,
    EnvConfig,
    ExperimentConfig,
    LoggingConfig,
    NetworkConfig,
    TrainingConfig,
)
from trading_rl.data import (
    PrepareDataConfig,
    download_trading_data,
    load_trading_data,
    prepare_data,
)
from trading_rl.envs import AlgorithmicEnvironmentBuilder, EnvBuildParams
from trading_rl.rewards import reward_function

from .models import (
    DiscreteNet,
    count_parameters,
    create_actor,
    create_ddpg_actor,
    create_ppo_actor,
    create_ppo_value_network,
    create_td3_actor,
    create_td3_qvalue_network,
    create_value_network,
)
from .train_trading_agent import (
    run_experiment_from_config,
    run_multiple_experiments,
    run_single_experiment,
    setup_mlflow_experiment,
)
from .training import DDPGTrainer, PPOTrainer, TD3Trainer

# Backwards-compatible convenience function for environment creation
_env_builder = AlgorithmicEnvironmentBuilder()


def create_environment(df, config):
    return _env_builder.create(df, EnvBuildParams.from_config(config))


__all__ = [
    # Config
    "DEFAULT_INITIAL_PORTFOLIO_VALUE",
    "AlgorithmicEnvironmentBuilder",
    # Training
    "DDPGTrainer",
    "DataConfig",
    # Models
    "DiscreteNet",
    "EnvConfig",
    "ExperimentConfig",
    "LoggingConfig",
    "NetworkConfig",
    "PPOTrainer",
    # Data
    "PrepareDataConfig",
    "TD3Trainer",
    "TrainingConfig",
    "count_parameters",
    "create_actor",
    "create_ddpg_actor",
    "create_environment",
    "create_ppo_actor",
    "create_ppo_value_network",
    "create_td3_actor",
    "create_td3_qvalue_network",
    "create_value_network",
    "download_trading_data",
    "load_trading_data",
    "prepare_data",
    "reward_function",
    "run_experiment_from_config",
    "run_multiple_experiments",
    "run_single_experiment",
    "setup_mlflow_experiment",
]
