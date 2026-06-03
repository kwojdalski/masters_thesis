"""Configuration for trading RL experiments."""

import datetime
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import yaml

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - optional dependency
    OmegaConf = None

from trading_rl.constants import ActionPenaltyType, Algorithm, BenchmarkName, EnvBackend, EnvMode, EvalSymbolSelection, ExplainabilityMethod, LossFunction, MetricName, RewardType, SplitName, StatisticalTest, TradePosition

_COMMAND_STEMS = {"train", "evaluate", "experiment"}


def _derive_experiment_name(path: Path) -> str:
    """Derive a unique experiment name from a config file or scenario path.

    Walks up to the 'scenarios/' segment so that, e.g.
    ``src/configs/scenarios/btc/td3_tradingenv/train.yaml`` becomes
    ``btc_td3_tradingenv``.
    """
    parts = path.parts
    try:
        scenarios_idx = list(parts).index("scenarios")
        rel_parts = list(parts[scenarios_idx + 1:])
        if rel_parts and Path(rel_parts[-1]).stem in _COMMAND_STEMS:
            rel_parts = rel_parts[:-1]
        return "_".join(Path(*rel_parts).with_suffix("").parts) if rel_parts else path.stem
    except ValueError:
        return path.stem


def _apply_derived_defaults(config_dict: dict, derived_name: str) -> None:
    """Fill in experiment_name, env.name, and log_dir if not already set."""
    if "experiment_name" not in config_dict:
        config_dict["experiment_name"] = derived_name

    env_dict = config_dict.setdefault("env", {})
    if isinstance(env_dict, dict) and not env_dict.get("name"):
        env_dict["name"] = derived_name.upper()

    log_dict = config_dict.get("logging")
    if log_dict is None:
        config_dict["logging"] = {"log_dir": str(Path("logs") / derived_name)}
    elif isinstance(log_dict, dict) and "log_dir" not in log_dict:
        log_dict["log_dir"] = str(Path("logs") / derived_name)


# HFT-appropriate metric set: excludes CAGR and Calmar, which annualise
# intra-session tick returns and produce misleading values on sub-day horizons.
_HFT_DEFAULT_METRICS: list[MetricName] = [
    MetricName.TOTAL_RETURN,
    MetricName.ANNUALIZED_VOLATILITY,
    MetricName.SHARPE_RATIO,
    MetricName.SORTINO_RATIO,
    MetricName.OMEGA_RATIO,
    MetricName.MAX_DRAWDOWN,
    MetricName.AVERAGE_DRAWDOWN,
    MetricName.MAX_DRAWDOWN_DURATION,
    MetricName.RECOVERY_TIME_FROM_MAX_DRAWDOWN,
    MetricName.VAR_95,
    MetricName.CVAR_95,
    MetricName.DOWNSIDE_DEVIATION,
    MetricName.RETURN_SKEWNESS,
    MetricName.RETURN_KURTOSIS,
    MetricName.WIN_RATE,
    MetricName.LOSE_RATE,
    MetricName.PROFIT_FACTOR,
    MetricName.PAYOFF_RATIO,
    MetricName.EXPECTANCY_PER_PERIOD,
    MetricName.TURNOVER,
    MetricName.AVERAGE_HOLDING_PERIOD,
    MetricName.PCT_LONG,
    MetricName.PCT_SHORT,
    MetricName.BETA,
    MetricName.ALPHA,
    MetricName.INFORMATION_RATIO,
    MetricName.TRACKING_ERROR,
]


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
    test_size: int | None = None
    feature_config: str | None = None  # Path to feature config YAML
    feature_groups: str | None = None  # Path to feature groups YAML
    # Multi-symbol pooled training: list of parquet paths processed independently
    # (per-symbol normalization) then concatenated. When set, data_path is ignored.
    data_paths: list[str] | None = None
    # Lazy loading: use LazyDataFrame instead of in-memory DataFrames
    # to reduce memory usage. Set prepared_data_dir to enable caching.
    lazy_load: bool = False
    prepared_data_dir: str | None = None  # Directory for cached prepared splits
    # Memmap streaming: save per-symbol numpy memmap files so the training
    # environment loads only one episode window at a time. Set this directory
    # to enable; StreamingTradingEnv is used automatically when files are found.
    memmap_dir: str | None = None
    # Feature transformation cache: stores per-symbol train/val/test parquets
    # keyed by a content hash. Acts as a per-symbol checkpoint — completed symbols
    # are skipped on restart. Set to None to disable.
    feature_cache_dir: str | None = ".cache/feature_transformation"
    # Per-day validation split: when set, each path in data_paths is used in
    # full as a training unit (no internal train/val split).  These paths supply
    # the validation (first half) and test (second half) data instead.
    # Expected convention: one val file per symbol, e.g. the last trading day.
    val_data_paths: list[str] | None = None
    # Drop rows where none of the first N LOB levels changed. Set to e.g. 5 for
    # MBP-10 data when features only consume levels 0-4, eliminating deep-book-only
    # update events that carry no signal for the configured feature set.
    filter_lob_levels: int | None = None
    # When True, feature_selection.yaml in the scenario dir overrides feature_columns.
    automated_selection: bool = False
    # Which symbol to use as the representative sample in streaming/memmap mode.
    # "first"   — always index 0 (deterministic, default)
    # "random"  — uniform random pick each run
    # "rotated" — cycles 0, 1, 2, … across runs (counter stored in memmap_dir)
    eval_symbol_selection: EvalSymbolSelection = EvalSymbolSelection.FIRST
    # Number of parallel worker processes for per-symbol feature engineering.
    # 0 = auto (min(n_symbols, cpu_count)), 1 = sequential (useful for debugging).
    n_workers: int = 0
    # Drop first N rows from the training split after feature engineering.
    # Running scalers (RunningMeanStd) produce valid-but-extreme values for the
    # first N ticks before their statistics converge; discarding those rows
    # prevents gradient spikes during early training episodes.
    warmup_rows: int = 300


DEFAULT_INITIAL_PORTFOLIO_VALUE: float = 10000.0


@dataclass
class EnvConfig:
    """Trading environment configuration."""

    name: str = ""
    mode: str = EnvMode.MFT  # Feature regime mode: "mft" (medium-frequency) or "hft" (high-frequency)
    positions: list[int] = field(default_factory=lambda: list(TradePosition))
    trading_fees: float = 0.0  # 0.01% = 0.0001
    borrow_interest_rate: float = 0.0  # 0.0003% = 0.000003
    initial_portfolio_value: float = DEFAULT_INITIAL_PORTFOLIO_VALUE  # Starting portfolio value for logging
    backend: str = EnvBackend.GYM_ANYTRADING_FOREX  # Backend type: gym_trading_env.discrete, gym_trading_env.continuous, gym_anytrading.forex, gym_anytrading.stocks, tradingenv

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

    # Streaming episode length: rows per episode when StreamingTradingEnv is used.
    # Each reset() loads exactly this many rows from a random position in a random
    # per-symbol memmap file, so peak memory is episode_length × features × 4 bytes.
    streaming_episode_length: int = 10_000

    # Thresholds for ContinuousToDiscreteAction (gym_trading_env.continuous backend).
    # The actor's continuous output is bucketed into positions[0/1/2] based on these
    # two boundaries: output < lower → positions[0], lower..upper → positions[1],
    # output > upper → positions[2].
    continuous_action_thresholds: list[float] = field(default_factory=lambda: [-0.33, 0.33])

    # Observation clipping: clip normalized features to [-obs_clip, obs_clip] before
    # passing to the network. Eliminates gradient spikes from session-open cold-start
    # periods where running normalization stats haven't converged yet.
    obs_clip: float | None = 5.0

    # Reward function configuration (all backends)
    reward_type: str = RewardType.LOG_RETURN  # Reward type: "log_return" or "differential_sharpe"
    reward_eta: float = 0.01  # Learning rate for DSR exponential moving averages (only used when reward_type="differential_sharpe")
    reward_scale: float = 1.0  # Multiplicative scale applied to the reward after clipping (e.g. 1000.0 to amplify DSR signals)

    # Action penalty: subtracted from the reward at each step to discourage bang-bang (±1) behavior.
    # Off by default (lambda=0). Penalty types:
    #   "quadratic"        → lambda * action^2
    #   "absolute"         → lambda * abs(action)
    #   "change_quadratic" → lambda * (action_t - action_{t-1})^2  (models transaction costs)
    action_penalty_lambda: float = 0.0
    action_penalty_type: str = ActionPenaltyType.QUADRATIC

    # Execution price model: "mid" (default) fills at the mid-price column;
    # "bid_ask" fills buys at ask_px_00 and sells at bid_px_00 (spread cost).
    # Override the column names only if your data uses non-standard names.
    execution_price: str = "mid"
    bid_column: str = "bid_px_00"
    ask_column: str = "ask_px_00"

    # Latency simulation (streaming/memmap environments only).
    # Both default to 0 (disabled).  The two are additive: at each episode
    # reset the combined tick offset = obs_latency_ticks + exec_latency_ticks
    # is applied between the feature rows and the execution price rows.
    # Use obs_latency_ticks to model stale market-data feed latency and
    # exec_latency_ticks to model order-submission / fill delay.
    # For stochastic (distribution-based) latency inject LatencyModel instances
    # directly into StreamingTradingEnvXY instead of using these integer fields.
    obs_latency_ticks: int = 0
    exec_latency_ticks: int = 0
    obs_latency_us: float = 0.0
    exec_latency_us: float = 0.0


@dataclass
class NetworkConfig:
    """Neural network architecture configuration."""

    # Actor network
    actor_hidden_dims: list[int] = field(default_factory=lambda: [64, 32])

    # Value network
    value_hidden_dims: list[int] = field(default_factory=lambda: [64, 32, 16])

    # Recurrent policy (RecurrentPPO): number of GRU layers stacked on top of each other.
    # actor_hidden_dims[0] is used as the GRU hidden size.
    gru_num_layers: int = 1


@dataclass
class TD3Config:
    """TD3-specific hyperparameters."""

    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    exploration_noise_std: float = 0.2
    delay_actor: bool = True
    delay_qvalue: bool = True


@dataclass
class PPOConfig:
    """PPO-specific hyperparameters."""

    clip_epsilon: float = 0.2
    entropy_bonus: float = 0.01
    vf_coef: float = 0.5
    epochs: int = 4


@dataclass
class SACConfig:
    """SAC-specific hyperparameters."""

    alpha_lr: float = 3e-4
    initial_alpha: float = 0.2
    target_entropy: float | None = None


@dataclass
class TempEvalConfig:
    """Mid-training periodic evaluation configuration."""

    interval: int | None = None  # Run every N training steps (None = disabled)
    splits: list[SplitName] = field(default_factory=lambda: [SplitName.TRAIN, SplitName.VAL])
    max_steps: int = 200000      # Cap rollout length for each periodic eval pass
    log_data: bool = False       # Log per-step rollout parquet artifact

    # Dense mode — shorten cadence when training metrics show sustained improvement
    dense_enabled: bool = False
    dense_min_evals: int = 3     # Evaluations required before density adaptation activates
    dense_window: int = 3        # Comparison window (evaluations per side)
    dense_improvement_threshold: float = 0.05  # Min relative improvement to enter dense mode
    dense_interval: int | None = None  # Cadence in dense mode (None → interval // 2)


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""

    # Algorithm selection
    algorithm: str = Algorithm.PPO  # "PPO", "DDPG", "TD3", or "SAC"

    # Optimization
    actor_lr: float = 1e-4
    value_lr: float = 1e-3
    value_weight_decay: float = 1e-2
    actor_weight_decay: float = 1e-4
    max_grad_norm: float = 1.0  # Gradient clipping threshold; set to 0 to disable

    # Training loop
    max_steps: int = 10_000
    max_train_seconds: int | None = None  # wall-clock budget in seconds; None = unlimited
    init_rand_steps: int = 5000
    frames_per_batch: int = 200
    optim_steps_per_batch: int = 50
    sample_size: int = 50
    checkpoint_interval: int = 0

    # Replay buffer
    buffer_size: int = 100_000
    save_buffer: bool = False  # Save replay buffer in checkpoint (increases file size significantly)

    # Soft target-network update rate — used by DDPG, TD3, SAC
    tau: float = 0.001

    # Algorithm-specific sub-configs
    td3: TD3Config = field(default_factory=TD3Config)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    sac: SACConfig = field(default_factory=SACConfig)

    # Pre-flight guardrail checks
    skip_guardrails: bool = False
    skip_guardrail_prompts: bool = False  # suppress the y/N prompt; warnings still logged

    # Loss function
    loss_function: LossFunction = LossFunction.L2

    # Evaluation cadence (training-loop concerns)
    eval_interval: int = 1000
    log_interval: int = 1000
    temp_eval: TempEvalConfig = field(default_factory=TempEvalConfig)

    max_plot_points: int | None = 50_000  # Cap the number of plotted points per series; None = plot all
    show_allocation_ma: bool = True  # Overlay moving-average line on Portfolio Allocation plot
    allocation_ma_window: int = 500  # Rolling window size for the allocation MA

    # Runtime early stopping
    es_stale_policy_min_ratio: float = 0.0  # Min mean position-change ratio; 0 = disabled
    es_stale_policy_window: int = 20        # Rolling window in completed episodes
    es_saturation_max_rate: float = 0.0    # Max mean extreme-position rate; 0 = disabled
    es_saturation_window: int = 20         # Rolling window in completed episodes


@dataclass
class EvaluationConfig:
    """Evaluation configuration shared by training and the evaluate CLI."""

    eval_steps: int = 500
    eval_fraction: float | None = None  # Fraction of val data; overrides eval_steps when set
    log_data: bool = True  # Log per-step rollout parquet (action, returns) as an MLflow artifact
    eval_plots: list[str] = field(
        default_factory=lambda: ["rewards", "positions", "portfolio_value"]
    )  # Which plots to generate: any subset of "rewards", "positions", "portfolio_value"
    per_symbol_eval: bool = False  # After pooled evaluation, run evaluate_split per symbol using val_data_paths with the pooled-fitted pipeline

    def resolve_eval_steps(self, val_len: int) -> int:
        """Return the number of eval steps, honouring eval_fraction when set."""
        if self.eval_fraction is not None:
            return max(1, int(val_len * self.eval_fraction))
        return self.eval_steps


@dataclass
class LoggingConfig:
    """Logging configuration."""

    log_dir: str = "logs"
    log_file: str = "trading_env_debug.log"
    log_level: str = "INFO"
    tensorboard_dir: str = "runs"
    log_to_file: bool = False  # Write logs to log_dir/log_file (disabled by default)
    save_plots: bool = False  # Save training plots to disk in addition to MLflow
    log_oracle_alignment_plot: bool = False  # Scatter plot of feature_future_close_vel vs next-step log return
    debug_plots: bool = False  # Stamp build hash, date, size, and font in bottom-right corner of every evaluation plot
    log_data_overviews: bool = False  # Log raw_data_overview and transformed_data_overview artifacts to MLflow


@dataclass
class TrackingConfig:
    """MLflow tracking backend configuration."""

    tracking_uri: str = "sqlite:///mlflow.db"


@dataclass
class ExplainabilityConfig:
    """Post-training explainability configuration."""

    enabled: bool = False
    n_steps: int = 500
    methods: list[ExplainabilityMethod] = field(default_factory=lambda: [ExplainabilityMethod.PERMUTATION, ExplainabilityMethod.INTEGRATED_GRADIENTS])
    temp_explainability_interval: int | None = None  # Run temporary explainability every N steps (None = disabled)


@dataclass
class ProfilingConfig:
    """Wall-clock profiling of pipeline stages, printed as a table at experiment end.

    level 0: disabled
    level 1: coarse stages (runtime_build, training, evaluation per split)
    level 2: fine-grained stages within each coarse stage (data load, env build, rollout, plots, etc.)
    """

    level: int = 2


@dataclass
class BenchmarksConfig:
    """Which baseline strategies to compute returns for during evaluation.

    Use ``enabled`` to list the strategies by name.  Random-action baseline
    parameters (``n_random_trials``, ``random_seed``) remain separate because
    they control how the random baseline is computed, not just whether it runs.
    """

    enabled: list[BenchmarkName] = field(
        default_factory=lambda: [BenchmarkName.BUY_AND_HOLD, BenchmarkName.RANDOM_ACTIONS]
    )
    n_random_trials: int = 10
    random_seed: int | None = None

    @property
    def enabled_set(self) -> frozenset[BenchmarkName]:
        """Frozenset of enabled BenchmarkName values for fast membership tests."""
        return frozenset(self.enabled)

    @property
    def is_random(self) -> bool:
        """True when RANDOM_ACTIONS is in the enabled list."""
        return BenchmarkName.RANDOM_ACTIONS in self.enabled_set


@dataclass
class MetricsConfig:
    """Controls which financial metrics are included in evaluation reports.

    The default set omits annualised-return metrics (CAGR, Calmar) because
    this project evaluates HFT strategies on intra-session tick data where
    annualisation produces misleading values.  Set ``enabled`` to
    ``list(MetricName)`` if you need all 29 metrics.
    """

    enabled: list[MetricName] = field(default_factory=lambda: list(_HFT_DEFAULT_METRICS))

    @property
    def enabled_set(self) -> frozenset[str]:
        """Return enabled metric names as a frozenset of strings for fast lookup."""
        return frozenset(self.enabled)


@dataclass
class StatisticalTestingConfig:
    """Statistical significance testing configuration for equity curves."""

    enabled: bool = False
    log_to_research_artifacts: bool = False
    research_artifact_subdir: str = "research_artifacts/statistical_tests"

    # Statistical tests to perform
    tests: list[StatisticalTest] = field(default_factory=lambda: [
        StatisticalTest.T_TEST,
        StatisticalTest.SHARPE_BOOTSTRAP,
        StatisticalTest.SORTINO_BOOTSTRAP,
        StatisticalTest.MANN_WHITNEY,
        StatisticalTest.PERMUTATION_TEST,
    ])

    # Test parameters
    n_bootstrap_samples: int = 10000
    n_permutations: int = 10000
    confidence_level: float = 0.95


def _validate_experiment_config(cfg: "ExperimentConfig") -> None:
    """Collect all validation errors for an ExperimentConfig and raise on failure."""
    errors: list[str] = []

    # Training hyperparameters
    if cfg.training.actor_lr <= 0:
        errors.append(f"training.actor_lr must be > 0, got {cfg.training.actor_lr}")
    if cfg.training.value_lr <= 0:
        errors.append(f"training.value_lr must be > 0, got {cfg.training.value_lr}")
    if cfg.training.max_steps <= 0:
        errors.append(f"training.max_steps must be > 0, got {cfg.training.max_steps}")
    if cfg.training.init_rand_steps < 0:
        errors.append(f"training.init_rand_steps must be >= 0, got {cfg.training.init_rand_steps}")
    if cfg.training.max_steps <= cfg.training.init_rand_steps:
        errors.append(
            f"training.max_steps ({cfg.training.max_steps}) must be > "
            f"init_rand_steps ({cfg.training.init_rand_steps})"
        )
    if cfg.training.frames_per_batch <= 0:
        errors.append(f"training.frames_per_batch must be > 0, got {cfg.training.frames_per_batch}")
    if cfg.training.buffer_size <= 0:
        errors.append(f"training.buffer_size must be > 0, got {cfg.training.buffer_size}")
    if cfg.evaluation.eval_steps <= 0:
        errors.append(f"evaluation.eval_steps must be > 0, got {cfg.evaluation.eval_steps}")
    if cfg.evaluation.eval_fraction is not None and not (0.0 < cfg.evaluation.eval_fraction <= 1.0):
        errors.append(f"evaluation.eval_fraction must be in (0, 1], got {cfg.evaluation.eval_fraction}")
    if cfg.training.checkpoint_interval < 0:
        errors.append(f"training.checkpoint_interval must be >= 0, got {cfg.training.checkpoint_interval}")

    # Algorithm-specific
    if cfg.training.algorithm.upper() == Algorithm.PPO:
        if not (0 < cfg.training.ppo.clip_epsilon < 1):
            errors.append(f"training.ppo.clip_epsilon must be in (0, 1), got {cfg.training.ppo.clip_epsilon}")
        if cfg.training.ppo.entropy_bonus < 0:
            errors.append(f"training.ppo.entropy_bonus must be >= 0, got {cfg.training.ppo.entropy_bonus}")
        if cfg.training.ppo.epochs <= 0:
            errors.append(f"training.ppo.epochs must be > 0, got {cfg.training.ppo.epochs}")
    if cfg.training.algorithm.upper() in {Algorithm.DDPG, Algorithm.TD3}:
        if not (0 < cfg.training.tau <= 1):
            errors.append(f"training.tau must be in (0, 1], got {cfg.training.tau}")

    # Data
    if cfg.data.train_size <= 0:
        errors.append(f"data.train_size must be > 0, got {cfg.data.train_size}")
    if cfg.data.validation_size is not None and cfg.data.validation_size < 0:
        errors.append(f"data.validation_size must be >= 0 when provided, got {cfg.data.validation_size}")
    if cfg.data.train_size < cfg.training.frames_per_batch:
        errors.append(
            f"data.train_size ({cfg.data.train_size}) must be >= "
            f"frames_per_batch ({cfg.training.frames_per_batch})"
        )

    # Environment
    if cfg.env.trading_fees < 0:
        errors.append(f"env.trading_fees must be >= 0, got {cfg.env.trading_fees}")
    if cfg.env.borrow_interest_rate < 0:
        errors.append(f"env.borrow_interest_rate must be >= 0, got {cfg.env.borrow_interest_rate}")
    if cfg.env.initial_portfolio_value <= 0:
        errors.append(f"env.initial_portfolio_value must be > 0, got {cfg.env.initial_portfolio_value}")
    if not cfg.env.positions:
        errors.append("env.positions must not be empty")
    if str(cfg.env.mode).lower() not in set(EnvMode):
        errors.append(f"env.mode must be one of {list(EnvMode)}, got '{cfg.env.mode}'")
    if (
        cfg.env.price_column is not None
        and (not isinstance(cfg.env.price_column, str) or not cfg.env.price_column.strip())
    ):
        errors.append("env.price_column must be a non-empty string when provided.")
    if cfg.env.reward_type not in set(RewardType):
        errors.append(f"env.reward_type must be one of {list(RewardType)}, got '{cfg.env.reward_type}'")
    if cfg.env.reward_type == RewardType.DIFFERENTIAL_SHARPE and cfg.env.reward_eta <= 0:
        errors.append(f"env.reward_eta must be > 0 when using differential_sharpe, got {cfg.env.reward_eta}")
    if cfg.env.action_penalty_lambda < 0:
        errors.append(f"env.action_penalty_lambda must be >= 0, got {cfg.env.action_penalty_lambda}")
    if cfg.env.action_penalty_type not in set(ActionPenaltyType):
        errors.append(
            f"env.action_penalty_type must be one of {[t.value for t in ActionPenaltyType]}, "
            f"got '{cfg.env.action_penalty_type}'"
        )

    # Network
    if not cfg.network.actor_hidden_dims:
        errors.append("network.actor_hidden_dims must not be empty")
    if not cfg.network.value_hidden_dims:
        errors.append("network.value_hidden_dims must not be empty")
    if any(dim <= 0 for dim in cfg.network.actor_hidden_dims):
        errors.append("network.actor_hidden_dims must contain only positive integers")
    if any(dim <= 0 for dim in cfg.network.value_hidden_dims):
        errors.append("network.value_hidden_dims must contain only positive integers")

    # Algorithm enum
    if cfg.training.algorithm.upper() not in set(Algorithm):
        errors.append(
            f"training.algorithm must be one of {list(Algorithm)}, got '{cfg.training.algorithm}'"
        )

    if errors:
        raise ValueError("Configuration validation failed:\n  - " + "\n  - ".join(errors))


def _apply_dict_to_dataclass(target, d: dict) -> None:
    """Apply matching keys from d to a dataclass instance via setattr.

    Recurses into nested dataclass fields when the value is a dict.
    """
    for key, value in d.items():
        if not hasattr(target, key):
            continue
        existing = getattr(target, key)
        if isinstance(value, dict) and is_dataclass(existing):
            _apply_dict_to_dataclass(existing, value)
        else:
            setattr(target, key, value)



@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    benchmarks: BenchmarksConfig = field(default_factory=BenchmarksConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    statistical_testing: StatisticalTestingConfig = field(default_factory=StatisticalTestingConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)

    # Reproducibility
    seed: int | None = None
    device: str = "cpu"  # or "cuda"

    # Experiment metadata
    experiment_name: str = "ddpg_trading"

    # Per-check guardrail suppression (e.g. ["check_td3_policy_delay_too_small"])
    # Use training.skip_guardrails=true to disable all checks at once.
    disabled_guardrails: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate configuration after initialization."""
        _validate_experiment_config(self)

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
        # e.g. src/configs/scenarios/btc/td3_tradingenv/train.yaml -> btc_td3_tradingenv
        derived_name = _derive_experiment_name(yaml_path)
        _apply_derived_defaults(config_dict, derived_name)
        return cls.from_dict(config_dict)

    @classmethod
    def from_scenario(
        cls,
        scenario_dir: str | Path,
        command: str = "train",
        overrides: list[str] | None = None,
    ) -> "ExperimentConfig":
        """Load config by merging scenario component files.

        Merge order (later wins):
          1. observation.yaml        — feature pipeline + feature_columns
          2. train.yaml             — base params (always included, even for evaluate)
          3. {command}.yaml         — command-specific overrides (skipped for "train")
          4. feature_selection.yaml — IC-selected columns (only when
                                      data.automated_selection is True)
          5. CLI overrides

        Args:
            scenario_dir: Path to the scenario directory.
            command: Which command file to load ("train" or "evaluate").
            overrides: Optional OmegaConf dotlist overrides.
        """
        scenario_dir = Path(scenario_dir)
        if not scenario_dir.is_dir():
            raise NotADirectoryError(f"Scenario directory not found: {scenario_dir}")

        features_path = scenario_dir / "observation.yaml"
        train_path = scenario_dir / "train.yaml"
        command_path = scenario_dir / f"{command}.yaml"
        selection_path = scenario_dir / "feature_selection.yaml"

        if not train_path.exists():
            raise FileNotFoundError(
                f"train.yaml not found in scenario directory: {scenario_dir}"
            )

        overrides = overrides or []

        if OmegaConf is not None:
            layers = []
            if features_path.exists():
                layers.append(OmegaConf.load(str(features_path)))
            layers.append(OmegaConf.load(str(train_path)))
            # For evaluate, merge evaluate.yaml on top of train.yaml
            if command != "train" and command_path.exists():
                layers.append(OmegaConf.load(str(command_path)))
            cfg = OmegaConf.merge(*layers) if len(layers) > 1 else layers[0]

            automated = bool(OmegaConf.select(cfg, "data.automated_selection", default=False))
            if automated and selection_path.exists():
                cfg = OmegaConf.merge(cfg, OmegaConf.load(str(selection_path)))

            if overrides:
                cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
            config_dict = OmegaConf.to_container(cfg, resolve=True)
        else:
            if overrides:
                raise ImportError("OmegaConf is required for config overrides.")

            def _merge(base: dict, override: dict) -> dict:
                out = dict(base)
                for k, v in override.items():
                    if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                        out[k] = _merge(out[k], v)
                    else:
                        out[k] = v
                return out

            paths_to_load = [features_path, train_path]
            if command != "train" and command_path.exists():
                paths_to_load.append(command_path)

            config_dict: dict = {}
            for path in paths_to_load:
                if path.exists():
                    with open(path) as f:
                        config_dict = _merge(config_dict, yaml.safe_load(f) or {})

            if config_dict.get("data", {}).get("automated_selection") and selection_path.exists():
                with open(selection_path) as f:
                    config_dict = _merge(config_dict, yaml.safe_load(f) or {})

        config_dict = config_dict or {}

        # Derive experiment name from scenario directory (same logic as from_yaml)
        derived_name = _derive_experiment_name(command_path)
        _apply_derived_defaults(config_dict, derived_name)
        return cls.from_dict(config_dict)

    @classmethod
    def load(
        cls,
        path: str | Path,
        command: str = "train",
        overrides: list[str] | None = None,
    ) -> "ExperimentConfig":
        """Load configuration from a file or scenario directory.

        Dispatches to :meth:`from_scenario` when *path* is a directory and to
        :meth:`from_yaml` when it is a file, so callers never need to branch on
        the path type themselves.

        Args:
            path: Path to a YAML file **or** a scenario directory.
            command: Command-specific overlay to load from the directory
                (e.g. ``"evaluate"`` merges ``evaluate.yaml`` on top of
                ``train.yaml``). Ignored when *path* is a file.
            overrides: Optional OmegaConf dotlist overrides applied last.
        """
        path = Path(path)
        if path.is_dir():
            return cls.from_scenario(path, command=command, overrides=overrides)
        return cls.from_yaml(path, overrides=overrides)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ExperimentConfig":
        """Create config from dictionary.

        Args:
            config_dict: Dictionary with configuration values

        Returns:
            ExperimentConfig instance
        """
        config = cls()

        if "experiment_name" in config_dict:
            config.experiment_name = config_dict["experiment_name"]
        if "seed" in config_dict and config_dict["seed"] is not None:
            config.seed = config_dict["seed"]
        if "device" in config_dict:
            config.device = config_dict["device"]
        if "disabled_guardrails" in config_dict:
            config.disabled_guardrails = list(config_dict["disabled_guardrails"] or [])

        # Data config — needs datetime coercion for download_since
        if "data" in config_dict:
            data_dict = dict(config_dict["data"])
            if "download_since" in data_dict and isinstance(data_dict["download_since"], str):
                data_dict["download_since"] = datetime.datetime.fromisoformat(
                    data_dict["download_since"].replace("Z", "+00:00")
                )
            _apply_dict_to_dataclass(config.data, data_dict)

        # Simple sub-configs — no special type conversions needed
        for attr_name, key in [
            ("env", "env"),
            ("network", "network"),
            ("training", "training"),
            ("evaluation", "evaluation"),
            ("logging", "logging"),
            ("tracking", "tracking"),
            ("explainability", "explainability"),
            ("statistical_testing", "statistical_testing"),
            ("profiling", "profiling"),
        ]:
            if key in config_dict:
                _apply_dict_to_dataclass(getattr(config, attr_name), config_dict[key])

        # Legacy env key migration: price_columns (list) → price_column (str)
        if "env" in config_dict:
            env_dict = config_dict["env"]
            if "price_columns" in env_dict and not config.env.price_column:
                pc = env_dict["price_columns"]
                config.env.price_column = pc[0] if isinstance(pc, list) and pc else pc

        # Benchmarks — enabled list needs BenchmarkName enum conversion;
        # individual bool keys map to BenchmarkName values for backward compat.
        _BOOL_TO_BENCHMARK = {
            "buy_and_hold": BenchmarkName.BUY_AND_HOLD,
            "twap":         BenchmarkName.TWAP,
            "vwap":         BenchmarkName.VWAP,
            "random":       BenchmarkName.RANDOM_ACTIONS,
        }
        bench_dict = config_dict.get("benchmarks", {})
        if "enabled" in bench_dict:
            config.benchmarks.enabled = [BenchmarkName(v) for v in bench_dict["enabled"]]
        else:
            enabled: set[BenchmarkName] = set(config.benchmarks.enabled)
            for old_key, bname in _BOOL_TO_BENCHMARK.items():
                if old_key in bench_dict:
                    if bench_dict[old_key]:
                        enabled.add(bname)
                    else:
                        enabled.discard(bname)
            config.benchmarks.enabled = sorted(enabled, key=lambda b: list(BenchmarkName).index(b))
        for key, value in bench_dict.items():
            if key not in _BOOL_TO_BENCHMARK and key != "enabled" and hasattr(config.benchmarks, key):
                setattr(config.benchmarks, key, value)

        # Metrics — enabled list needs MetricName enum conversion
        if "metrics" in config_dict:
            metrics_dict = config_dict["metrics"]
            if "enabled" in metrics_dict:
                config.metrics.enabled = [MetricName(v) for v in metrics_dict["enabled"]]

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

        Includes every field from every sub-config dataclass so that
        ``ExperimentConfig.from_dict(config.to_dict())`` round-trips without
        data loss.  Sub-configs are serialised under their canonical field
        names (e.g. ``"env"``); ``from_dict`` already accepts both ``"env"``
        and the legacy ``"environment"`` key.

        Returns:
            Dictionary representation of the config
        """
        import dataclasses as _dc
        import enum

        def _ser(obj):
            if _dc.is_dataclass(obj) and not isinstance(obj, type):
                return {f.name: _ser(getattr(obj, f.name)) for f in _dc.fields(obj)}
            if isinstance(obj, datetime.datetime):
                return obj.isoformat()
            if isinstance(obj, enum.Enum):
                return obj.value
            if isinstance(obj, list):
                return [_ser(v) for v in obj]
            if isinstance(obj, dict):
                return {k: _ser(v) for k, v in obj.items()}
            return obj

        d = _ser(self)
        # Preserve "env" key name — from_dict accepts it directly.
        return d


# ---------------------------------------------------------------------------
# Narrow parameter objects for dependency injection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoggingParams:
    """Logging configuration parameters extracted from ExperimentConfig."""

    log_level: str
    log_dir: str
    log_file: str
    log_to_file: bool
    tensorboard_dir: str

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "LoggingParams":
        return cls(
            log_level=config.logging.log_level,
            log_dir=config.logging.log_dir,
            log_file=config.logging.log_file,
            log_to_file=config.logging.log_to_file,
            tensorboard_dir=config.logging.tensorboard_dir,
        )


@dataclass(frozen=True)
class MLflowCallbackParams:
    """Parameters for MLflowTrainingCallback construction."""

    tracking_uri: str | None
    total_episodes: int
    price_series: Any
    initial_portfolio_value: float
    reward_type: str
    action_positions: list[int]

    @classmethod
    def from_config(cls, config: ExperimentConfig, dataset: Any) -> "MLflowCallbackParams":
        from trading_rl.data.loading import PreparedDataset

        estimated_episodes = max(1, config.training.max_steps // config.data.train_size)
        price_series = (
            dataset.train_df[dataset.price_column]
            if isinstance(dataset, PreparedDataset)
            else None
        )
        return cls(
            tracking_uri=getattr(getattr(config, "tracking", None), "tracking_uri", None),
            total_episodes=estimated_episodes,
            price_series=price_series,
            initial_portfolio_value=config.env.initial_portfolio_value,
            reward_type=config.env.reward_type,
            action_positions=config.env.positions,
        )


@dataclass(frozen=True)
class TrainerConstructionParams:
    """Parameters for trainer construction."""

    config: TrainingConfig
    n_obs: int
    n_act: int
    actor_hidden_dims: list[int]
    value_hidden_dims: list[int]
    eval_config: EvaluationConfig
    checkpoint_dir: str
    checkpoint_prefix: str

    @classmethod
    def from_config(cls, config: ExperimentConfig, n_obs: int, n_act: int, checkpoint_prefix: str) -> "TrainerConstructionParams":
        return cls(
            config=config.training,
            n_obs=n_obs,
            n_act=n_act,
            actor_hidden_dims=config.network.actor_hidden_dims,
            value_hidden_dims=config.network.value_hidden_dims,
            eval_config=config.evaluation,
            checkpoint_dir=config.logging.log_dir,
            checkpoint_prefix=checkpoint_prefix,
        )
