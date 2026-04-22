"""Configuration for trading RL experiments."""

import datetime
from dataclasses import dataclass, field
from pathlib import Path

import yaml

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - optional dependency
    OmegaConf = None


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
    validation_size: int | None = None
    feature_config: str | None = None  # Path to feature config YAML


DEFAULT_INITIAL_PORTFOLIO_VALUE: float = 10000.0


@dataclass
class EnvConfig:
    """Trading environment configuration."""

    name: str = "BTCUSD"
    mode: str = "mft"  # Feature regime mode: "mft" (medium-frequency) or "hft" (high-frequency)
    positions: list[int] = field(default_factory=lambda: [-1, 0, 1])
    trading_fees: float = 0.0  # 0.01% = 0.0001
    borrow_interest_rate: float = 0.0  # 0.0003% = 0.000003
    initial_portfolio_value: float = DEFAULT_INITIAL_PORTFOLIO_VALUE  # Starting portfolio value for logging
    backend: str = "gym_anytrading.forex"  # Backend type: gym_trading_env.discrete, gym_trading_env.continuous, gym_anytrading.forex, gym_anytrading.stocks, tradingenv

    # TradingEnv-specific configuration (optional)
    price_column: str | None = (
        None  # Column to use as asset price (for tradingenv backend)
    )
    feature_columns: list[str] | None = (
        None  # Columns to use as features/observations (for tradingenv backend)
    )
    include_position_feature: bool = (
        False  # Append runtime feature_position from TradingEnv broker state
    )

    # Reward function configuration (all backends)
    reward_type: str = "log_return"  # Reward type: "log_return" or "differential_sharpe"
    reward_eta: float = 0.01  # Learning rate for DSR exponential moving averages (only used when reward_type="differential_sharpe")


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
    actor_weight_decay: float = 1e-4
    # Training loop
    max_steps: int = 5_000  # _000
    init_rand_steps: int = 50
    frames_per_batch: int = 200
    optim_steps_per_batch: int = 50
    sample_size: int = 50
    checkpoint_interval: int = 0

    # Replay buffer
    buffer_size: int = 100_000
    save_buffer: bool = (
        False  # Save replay buffer in checkpoint (increases file size significantly)
    )

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
    policy_delay: int = 2
    exploration_noise_std: float = 0.2
    delay_actor: bool = True
    delay_qvalue: bool = True

    # Loss function
    loss_function: str = "l2"

    # Evaluation
    eval_interval: int = 1000
    eval_steps: int = 500
    log_interval: int = 1000
    temp_eval_interval: int | None = None  # Run temporary evaluation every N steps (None = disabled)


@dataclass
class LoggingConfig:
    """Logging configuration."""

    log_dir: str = "logs"
    log_file: str = "trading_env_debug.log"
    log_level: str = "INFO"
    tensorboard_dir: str = "runs"
    save_plots: bool = False  # Save training plots to disk in addition to MLflow


@dataclass
class TrackingConfig:
    """MLflow tracking backend configuration."""

    tracking_uri: str = "sqlite:///mlflow.db"


@dataclass
class ExplainabilityConfig:
    """Post-training explainability configuration."""

    enabled: bool = False
    n_steps: int = 500
    methods: list[str] = field(default_factory=lambda: ["permutation", "integrated_gradients"])
    temp_explainability_interval: int | None = None  # Run temporary explainability every N steps (None = disabled)


@dataclass
class StatisticalTestingConfig:
    """Statistical significance testing configuration for equity curves."""

    enabled: bool = False
    log_to_research_artifacts: bool = False  # Log compact research artifact bundle to MLflow
    research_artifact_subdir: str = "research_artifacts/statistical_tests"

    # Comparison baselines (can enable multiple)
    compare_to_buy_and_hold: bool = True  # Compare to buy-and-hold benchmark
    compare_to_short_and_hold: bool = False  # Compare to short-and-hold benchmark
    compare_to_twap: bool = False  # Compare to TWAP execution baseline
    compare_to_vwap: bool = False  # Compare to VWAP execution baseline
    compare_to_random: bool = True  # Compare to random action baseline

    # Statistical tests to perform (can enable multiple)
    tests: list[str] = field(default_factory=lambda: [
        "t_test",  # T-test for mean returns
        "sharpe_bootstrap",  # Bootstrap test for Sharpe ratio
        "sortino_bootstrap",  # Bootstrap test for Sortino ratio (downside risk only)
        "mann_whitney",  # Mann-Whitney U test (non-parametric)
        "permutation_test"  # Permutation test (distribution-free)
    ])

    # Test parameters
    n_bootstrap_samples: int = 10000  # Number of bootstrap samples
    n_permutations: int = 10000  # Number of permutations for permutation test
    n_random_trials: int = 100  # Number of random baseline trials
    confidence_level: float = 0.95  # Confidence level (e.g., 0.95 for 95% CI)
    random_seed: int | None = None  # Seed for reproducible random baseline


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    statistical_testing: StatisticalTestingConfig = field(default_factory=StatisticalTestingConfig)

    # Reproducibility
    seed: int | None = None
    device: str = "cpu"  # or "cuda"

    # Experiment metadata
    experiment_name: str = "ddpg_trading"

    def __post_init__(self):
        """Validate configuration after initialization."""
        errors = []

        # Training hyperparameter validation
        if self.training.actor_lr <= 0:
            errors.append(
                f"training.actor_lr must be > 0, got {self.training.actor_lr}"
            )
        if self.training.value_lr <= 0:
            errors.append(
                f"training.value_lr must be > 0, got {self.training.value_lr}"
            )
        if self.training.max_steps <= 0:
            errors.append(
                f"training.max_steps must be > 0, got {self.training.max_steps}"
            )
        if self.training.init_rand_steps < 0:
            errors.append(
                f"training.init_rand_steps must be >= 0, got {self.training.init_rand_steps}"
            )
        if self.training.max_steps <= self.training.init_rand_steps:
            errors.append(
                f"training.max_steps ({self.training.max_steps}) must be > "
                f"init_rand_steps ({self.training.init_rand_steps})"
            )
        if self.training.frames_per_batch <= 0:
            errors.append(
                f"training.frames_per_batch must be > 0, got {self.training.frames_per_batch}"
            )
        if self.training.buffer_size <= 0:
            errors.append(
                f"training.buffer_size must be > 0, got {self.training.buffer_size}"
            )
        if self.training.eval_steps <= 0:
            errors.append(
                f"training.eval_steps must be > 0, got {self.training.eval_steps}"
            )
        if self.training.checkpoint_interval < 0:
            errors.append(
                f"training.checkpoint_interval must be >= 0, got {self.training.checkpoint_interval}"
            )

        # PPO-specific validation
        if self.training.algorithm.upper() == "PPO":
            if not (0 < self.training.clip_epsilon < 1):
                errors.append(
                    f"training.clip_epsilon must be in (0, 1), got {self.training.clip_epsilon}"
                )
            if self.training.entropy_bonus < 0:
                errors.append(
                    f"training.entropy_bonus must be >= 0, got {self.training.entropy_bonus}"
                )
            if self.training.ppo_epochs <= 0:
                errors.append(
                    f"training.ppo_epochs must be > 0, got {self.training.ppo_epochs}"
                )

        # DDPG/TD3-specific validation
        if self.training.algorithm.upper() in ["DDPG", "TD3"]:
            if not (0 < self.training.tau <= 1):
                errors.append(
                    f"training.tau must be in (0, 1], got {self.training.tau}"
                )

        # Data configuration validation
        if self.data.train_size <= 0:
            errors.append(f"data.train_size must be > 0, got {self.data.train_size}")
        if self.data.validation_size is not None and self.data.validation_size < 0:
            errors.append(
                "data.validation_size must be >= 0 when provided, "
                f"got {self.data.validation_size}"
            )
        if self.data.train_size < self.training.frames_per_batch:
            errors.append(
                f"data.train_size ({self.data.train_size}) must be >= "
                f"frames_per_batch ({self.training.frames_per_batch})"
            )

        # Environment configuration validation
        if self.env.trading_fees < 0:
            errors.append(
                f"env.trading_fees must be >= 0, got {self.env.trading_fees}"
            )
        if self.env.borrow_interest_rate < 0:
            errors.append(
                f"env.borrow_interest_rate must be >= 0, got {self.env.borrow_interest_rate}"
            )
        if self.env.initial_portfolio_value <= 0:
            errors.append(
                f"env.initial_portfolio_value must be > 0, got {self.env.initial_portfolio_value}"
            )
        if not self.env.positions:
            errors.append("env.positions must not be empty")
        if str(self.env.mode).lower() not in {"mft", "hft"}:
            errors.append(
                f"env.mode must be 'mft' or 'hft', got '{self.env.mode}'"
            )
        if (
            self.env.price_column is not None
            and (
                not isinstance(self.env.price_column, str)
                or not self.env.price_column.strip()
            )
        ):
            errors.append(
                "env.price_column must be a non-empty string when provided."
            )

        # Reward configuration validation
        if self.env.reward_type not in ["log_return", "differential_sharpe"]:
            errors.append(
                f"env.reward_type must be 'log_return' or 'differential_sharpe', "
                f"got '{self.env.reward_type}'"
            )
        if self.env.reward_type == "differential_sharpe" and self.env.reward_eta <= 0:
            errors.append(
                f"env.reward_eta must be > 0 when using differential_sharpe, "
                f"got {self.env.reward_eta}"
            )

        # Network configuration validation
        if not self.network.actor_hidden_dims:
            errors.append("network.actor_hidden_dims must not be empty")
        if not self.network.value_hidden_dims:
            errors.append("network.value_hidden_dims must not be empty")
        if any(dim <= 0 for dim in self.network.actor_hidden_dims):
            errors.append("network.actor_hidden_dims must contain only positive integers")
        if any(dim <= 0 for dim in self.network.value_hidden_dims):
            errors.append("network.value_hidden_dims must contain only positive integers")

        # Algorithm validation
        valid_algorithms = ["PPO", "DDPG", "TD3"]
        if self.training.algorithm.upper() not in valid_algorithms:
            errors.append(
                f"training.algorithm must be one of {valid_algorithms}, "
                f"got '{self.training.algorithm}'"
            )

        if errors:
            raise ValueError(
                "Configuration validation failed:\n  - " + "\n  - ".join(errors)
            )

        # Create directories if they don't exist
        Path(self.logging.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logging.tensorboard_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(
        cls, yaml_path: str | Path, overrides: list[str] | None = None
    ) -> "ExperimentConfig":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file
            overrides: Optional OmegaConf dotlist overrides

        Returns:
            ExperimentConfig instance loaded from YAML
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        overrides = overrides or []
        if OmegaConf is not None:
            cfg = OmegaConf.load(str(yaml_path))
            if overrides:
                cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
            config_dict = OmegaConf.to_container(cfg, resolve=True)
        else:
            if overrides:
                raise ImportError(
                    "OmegaConf is required for config overrides. "
                    "Install it or remove overrides."
                )
            with open(yaml_path) as f:
                config_dict = yaml.safe_load(f)

        config_dict = config_dict or {}
        if not isinstance(config_dict, dict):
            raise ValueError(
                f"Expected top-level config mapping in {yaml_path}, got {type(config_dict).__name__}"
            )

        # Build a unique name that includes the group subfolder when present.
        # e.g. src/configs/scenarios/sine_wave/ppo_no_trend.yaml -> sine_wave_ppo_no_trend
        parts = yaml_path.parts
        try:
            scenarios_idx = list(parts).index("scenarios")
            rel_parts = parts[scenarios_idx + 1:]
            derived_name = "_".join(Path(*rel_parts).with_suffix("").parts) if rel_parts else yaml_path.stem
        except ValueError:
            derived_name = yaml_path.stem

        if "experiment_name" not in config_dict:
            config_dict["experiment_name"] = derived_name

        log_dict = config_dict.get("logging")
        if log_dict is None:
            config_dict["logging"] = {"log_dir": str(Path("logs") / derived_name)}
        elif isinstance(log_dict, dict) and "log_dir" not in log_dict:
            log_dict["log_dir"] = str(Path("logs") / derived_name)

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
        if "environment" in config_dict or "env" in config_dict:
            env_dict = config_dict.get("environment", config_dict.get("env", {}))
            for key, value in env_dict.items():
                if key == "price_columns":
                    # Backward compatibility: accept legacy list-style key and
                    # map to canonical single-column env.price_column.
                    if isinstance(value, list) and value:
                        config.env.price_column = str(value[0])
                    elif isinstance(value, str):
                        config.env.price_column = value
                    continue
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
                    config.training.max_steps = value
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

        if "explainability" in config_dict:
            exp_dict = config_dict["explainability"]
            for key, value in exp_dict.items():
                if hasattr(config.explainability, key):
                    setattr(config.explainability, key, value)

        if "statistical_testing" in config_dict:
            stat_dict = config_dict["statistical_testing"]
            for key, value in stat_dict.items():
                if hasattr(config.statistical_testing, key):
                    setattr(config.statistical_testing, key, value)

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
                "validation_size": self.data.validation_size,
            },
            "environment": {
                "name": self.env.name,
                "mode": self.env.mode,
                "positions": self.env.positions,
                "trading_fees": self.env.trading_fees,
                "borrow_interest_rate": self.env.borrow_interest_rate,
                "initial_portfolio_value": self.env.initial_portfolio_value,
                "backend": self.env.backend,
                # TradingEnv-specific fields (only include if set)
                **(
                    {"price_column": self.env.price_column}
                    if self.env.price_column is not None
                    else {}
                ),
                **(
                    {"feature_columns": self.env.feature_columns}
                    if self.env.feature_columns is not None
                    else {}
                ),
                "reward_type": self.env.reward_type,
                "reward_eta": self.env.reward_eta,
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
                "checkpoint_interval": self.training.checkpoint_interval,
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
            "explainability": {
                "enabled": self.explainability.enabled,
                "n_steps": self.explainability.n_steps,
                "methods": self.explainability.methods,
            },
            "statistical_testing": {
                "enabled": self.statistical_testing.enabled,
                "log_to_research_artifacts": self.statistical_testing.log_to_research_artifacts,
                "research_artifact_subdir": self.statistical_testing.research_artifact_subdir,
                "compare_to_buy_and_hold": self.statistical_testing.compare_to_buy_and_hold,
                "compare_to_short_and_hold": self.statistical_testing.compare_to_short_and_hold,
                "compare_to_twap": self.statistical_testing.compare_to_twap,
                "compare_to_vwap": self.statistical_testing.compare_to_vwap,
                "compare_to_random": self.statistical_testing.compare_to_random,
                "tests": self.statistical_testing.tests,
                "n_bootstrap_samples": self.statistical_testing.n_bootstrap_samples,
                "n_permutations": self.statistical_testing.n_permutations,
                "n_random_trials": self.statistical_testing.n_random_trials,
                "confidence_level": self.statistical_testing.confidence_level,
                "random_seed": self.statistical_testing.random_seed,
            },
        }
