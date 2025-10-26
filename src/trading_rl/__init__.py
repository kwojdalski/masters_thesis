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

from .models import (
    DiscreteNet,
    count_parameters,
    create_actor,
    create_value_network,
    create_ppo_actor,
    create_ppo_value_network,
    create_ddpg_actor,
)
from .train_trading_agent import (
    MLflowTrainingCallback,
    create_environment,
    evaluate_agent,
    run_experiment_from_config,
    run_multiple_experiments,
    run_single_experiment,
    set_seed,
    setup_logging,
    setup_mlflow_experiment,
    visualize_training,
)
from .training import DDPGTrainer, PPOTrainer

__all__ = [
    # Training
    "DDPGTrainer",
    "PPOTrainer",
    # Config
    "DataConfig",
    "EnvConfig",
    "ExperimentConfig",
    "LoggingConfig",
    "NetworkConfig",
    "TrainingConfig",
    # Models
    "DiscreteNet",
    "count_parameters",
    "create_actor",
    "create_value_network",
    "create_ppo_actor",
    "create_ppo_value_network",
    "create_ddpg_actor",
    # Data
    "create_environment",
    "create_features",
    "download_trading_data",
    "evaluate_agent",
    "load_trading_data",
    "prepare_data",
    "reward_function",
    "run_experiment_from_config",
    "run_multiple_experiments",
    "run_single_experiment",
    "set_seed",
    "setup_logging",
    "setup_mlflow_experiment",
    "visualize_training",
    # MLflow integration
    "MLflowTrainingCallback",
]
