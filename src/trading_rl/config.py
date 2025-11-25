"""Configuration for trading RL experiments."""

import datetime
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""

    data_path: str = "data/bitfinex2-BTCUSDT-1m.pkl"
    download_data: bool = True
    exchange_names: list[str] = field(default_factory=lambda: ["bitfinex2"])
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT"])
    timeframe: str = "1m"
    data_dir: str = "data"
    download_since: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC)
        - datetime.timedelta(days=1)
    )
    train_size: int = 1000
    no_features: bool = False


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

    # Algorithm selection
    algorithm: str = "PPO"  # "PPO", "DDPG", or "TD3"

    # Optimization
    actor_lr: float = 1e-4
    value_lr: float = 1e-3
    value_weight_decay: float = 1e-2

    # Training loop
    max_steps: int = 5_000  # _000
    init_rand_steps: int = 50
    frames_per_batch: int = 200
    optim_steps_per_batch: int = 50
    sample_size: int = 50

    # Replay buffer
    buffer_size: int = 100_000

    # DDPG-specific parameters
    tau: float = 0.001  # Target network update rate

    # PPO-specific parameters
    clip_epsilon: float = 0.2  # PPO clipping parameter
    entropy_bonus: float = 0.01  # Entropy bonus coefficient
    vf_coef: float = 0.5  # Value function loss coefficient
    ppo_epochs: int = 4  # Number of PPO epochs per batch

    # TD3-specific parameters
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    delay_actor: bool = True
    delay_qvalue: bool = True

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
    log_level: str = "INFO"
    tensorboard_dir: str = "runs"


@dataclass
class TrackingConfig:
    """MLflow tracking backend configuration."""

    tracking_uri: str = "sqlite:///mlflow.db"


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    # Reproducibility
    seed: int = 42
    device: str = "cpu"  # or "cuda"

    # Experiment metadata
    experiment_name: str = "ddpg_trading"

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ExperimentConfig":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            ExperimentConfig instance loaded from YAML
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ExperimentConfig":
        """Create config from dictionary.

        Args:
            config_dict: Dictionary with configuration values

        Returns:
            ExperimentConfig instance
        """
        # Create default config
        config = cls()

        # Update with provided values
        if "experiment_name" in config_dict:
            config.experiment_name = config_dict["experiment_name"]
        if "seed" in config_dict and config_dict["seed"] is not None:
            config.seed = config_dict["seed"]
        if "device" in config_dict:
            config.device = config_dict["device"]

        # Update data config
        if "data" in config_dict:
            data_dict = config_dict["data"]
            for key, value in data_dict.items():
                if hasattr(config.data, key):
                    # Handle datetime parsing
                    if key == "download_since" and isinstance(value, str):
                        setattr(
                            config.data,
                            key,
                            datetime.datetime.fromisoformat(
                                value.replace("Z", "+00:00")
                            ),
                        )
                    else:
                        setattr(config.data, key, value)

        # Update environment config
        if "environment" in config_dict:
            env_dict = config_dict["environment"]
            for key, value in env_dict.items():
                if hasattr(config.env, key):
                    setattr(config.env, key, value)

        # Update network config
        if "network" in config_dict:
            net_dict = config_dict["network"]
            for key, value in net_dict.items():
                if hasattr(config.network, key):
                    setattr(config.network, key, value)

        # Update training config
        if "training" in config_dict:
            train_dict = config_dict["training"]
            for key, value in train_dict.items():
                if key == "max_training_steps":
                    setattr(config.training, "max_steps", value)
                    continue
                if hasattr(config.training, key):
                    setattr(config.training, key, value)

        # Update logging config
        if "logging" in config_dict:
            log_dict = config_dict["logging"]
            for key, value in log_dict.items():
                if hasattr(config.logging, key):
                    setattr(config.logging, key, value)

        if "tracking" in config_dict:
            tracking_dict = config_dict["tracking"]
            for key, value in tracking_dict.items():
                if hasattr(config.tracking, key):
                    setattr(config.tracking, key, value)

        return config

    def to_yaml(self, yaml_path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            yaml_path: Path where to save the YAML file
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.to_dict()

        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the config
        """
        return {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "device": self.device,
            "data": {
                "data_path": self.data.data_path,
                "download_data": self.data.download_data,
                "exchange_names": self.data.exchange_names,
                "symbols": self.data.symbols,
                "timeframe": self.data.timeframe,
                "data_dir": self.data.data_dir,
                "download_since": self.data.download_since.isoformat(),
                "train_size": self.data.train_size,
            },
            "environment": {
                "name": self.env.name,
                "positions": self.env.positions,
                "trading_fees": self.env.trading_fees,
                "borrow_interest_rate": self.env.borrow_interest_rate,
            },
            "network": {
                "actor_hidden_dims": self.network.actor_hidden_dims,
                "value_hidden_dims": self.network.value_hidden_dims,
            },
            "training": {
                "actor_lr": self.training.actor_lr,
                "value_lr": self.training.value_lr,
                "value_weight_decay": self.training.value_weight_decay,
                "max_steps": self.training.max_steps,
                "init_rand_steps": self.training.init_rand_steps,
                "frames_per_batch": self.training.frames_per_batch,
                "optim_steps_per_batch": self.training.optim_steps_per_batch,
                "sample_size": self.training.sample_size,
                "buffer_size": self.training.buffer_size,
                "tau": self.training.tau,
                "loss_function": self.training.loss_function,
                "eval_steps": self.training.eval_steps,
                "eval_interval": self.training.eval_interval,
                "log_interval": self.training.log_interval,
            },
            "logging": {
                "log_dir": self.logging.log_dir,
                "log_file": self.logging.log_file,
                "log_level": self.logging.log_level,
                "tensorboard_dir": self.logging.tensorboard_dir,
            },
        }

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Create directories if they don't exist
        Path(self.logging.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logging.tensorboard_dir).mkdir(parents=True, exist_ok=True)
