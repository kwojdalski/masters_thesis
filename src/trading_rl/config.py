"""Configuration for trading RL experiments."""

import datetime
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""

    data_path: str = "./data/raw/binance/binance-BTCUSDT-1h.parquet"
    download_data: bool = False
    exchange_names: list[str] = field(default_factory=lambda: ["binance"])
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT"])
    timeframe: str = "1h"
    data_dir: str = "data"
    download_since: datetime.datetime = field(
        default_factory=lambda: datetime.datetime(2025, 4, 27, tzinfo=datetime.UTC)
    )
    train_size: int = 1000


@dataclass
class EnvConfig:
    """Trading environment configuration."""

    name: str = "BTCUSD"
    positions: list[int] = field(default_factory=lambda: [-1, 0, 1])
    trading_fees: float = 0.0  # 0.01% = 0.0001
    borrow_interest_rate: float = 0.0  # 0.0003% = 0.000003


@dataclass
class NetworkConfig:
    """Neural network architecture configuration."""

    # Actor network
    actor_hidden_dims: list[int] = field(default_factory=lambda: [64, 32])

    # Value network
    value_hidden_dims: list[int] = field(default_factory=lambda: [64, 32, 16])


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""

    # Optimization
    actor_lr: float = 1e-4
    value_lr: float = 1e-3
    value_weight_decay: float = 1e-2

    # Training loop
    max_training_steps: int = 5_000_000
    init_rand_steps: int = 50
    frames_per_batch: int = 200
    optim_steps_per_batch: int = 50
    sample_size: int = 50

    # Replay buffer
    buffer_size: int = 100_000

    # Target network
    tau: float = 0.001

    # Loss function
    loss_function: str = "l2"

    # Evaluation
    eval_interval: int = 1000
    eval_steps: int = 500
    log_interval: int = 1000


@dataclass
class LoggingConfig:
    """Logging configuration."""

    log_dir: str = "logs"
    log_file: str = "trading_env_debug.log"
    log_level: str = "DEBUG"
    tensorboard_dir: str = "runs"


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Reproducibility
    seed: int = 42
    device: str = "cpu"  # or "cuda"

    # Experiment metadata
    experiment_name: str = "ddpg_trading"

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Create directories if they don't exist
        Path(self.logging.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logging.tensorboard_dir).mkdir(parents=True, exist_ok=True)
