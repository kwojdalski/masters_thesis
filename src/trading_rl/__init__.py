"""Trading RL package with modular components for DDPG trading."""

from trading_rl.config import (
    DataConfig,
    EnvConfig,
    ExperimentConfig,
    LoggingConfig,
    NetworkConfig,
    TrainingConfig,
)
from trading_rl.data_utils import (
    create_features,
    download_trading_data,
    load_trading_data,
    prepare_data,
    reward_function,
)

from .models import DiscreteNet, count_parameters, create_actor, create_value_network
from .train_trading_agent import (
    OptunaTrainingCallback,
    create_environment,
    evaluate_agent,
    run_multiple_experiments,
    run_single_experiment,
    set_seed,
    setup_logging,
    visualize_training,
)
from .training import DDPGTrainer

__all__ = [
    # Training
    "DDPGTrainer",
    # Config
    "DataConfig",
    # Models
    "DiscreteNet",
    "EnvConfig",
    "ExperimentConfig",
    "LoggingConfig",
    "NetworkConfig",
    "TrainingConfig",
    "count_parameters",
    "create_actor",
    # Data
    "create_environment",
    "create_features",
    "create_value_network",
    "download_trading_data",
    "evaluate_agent",
    "load_trading_data",
    "prepare_data",
    "reward_function",
    "run_multiple_experiments",
    "run_single_experiment",
    "set_seed",
    "setup_logging",
    "visualize_training",
    # Optuna integration
    "OptunaTrainingCallback",
]
