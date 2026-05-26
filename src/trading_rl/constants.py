"""Shared string-valued enums for the trading_rl package.

Using StrEnum means enum members compare equal to their string values, so
YAML-loaded strings keep working in comparisons without any conversion:

    config.training.algorithm.upper() == Algorithm.PPO  # True when value is "PPO"
    reward_type == RewardType.LOG_RETURN                 # True when value is "log_return"
"""

from enum import Enum, IntEnum, StrEnum


# US equity market calendar assumptions used to derive periods-per-year below.
# 252 trading days, 6.5-hour session (09:30–16:00 ET).
_US_TRADING_DAYS: int = 252
_US_TRADING_HOURS: float = 6.5  # hours per trading day


def _ppy(bars_per_day: float) -> int:
    """Periods per year for a given bar size under the US equity calendar."""
    return round(_US_TRADING_DAYS * bars_per_day)


class ReportingFrequency(Enum):
    """Bar frequencies used for annualisation and Sharpe/Sortino computation.

    Each member carries the timeframe label and the corresponding
    periods-per-year count derived from the US equity trading calendar
    (_US_TRADING_DAYS days, _US_TRADING_HOURS hours/day).
    """

    WEEKLY  = ("1w",  52)                                          # calendar weeks
    DAILY   = ("1d",  _US_TRADING_DAYS)                            # 252
    HOUR_4  = ("4h",  _ppy(_US_TRADING_HOURS / 4))                 # 410
    HOURLY  = ("1h",  _ppy(_US_TRADING_HOURS))                     # 1_638
    MIN_30  = ("30m", _ppy(_US_TRADING_HOURS * 2))                 # 3_276
    MIN_15  = ("15m", _ppy(_US_TRADING_HOURS * 4))                 # 6_552
    MIN_5   = ("5m",  _ppy(_US_TRADING_HOURS * 12))                # 19_656
    MIN_1   = ("1m",  _ppy(_US_TRADING_HOURS * 60))                # 98_280
    SEC_15  = ("15s", _ppy(_US_TRADING_HOURS * 60 * 4))            # 393_120
    SEC_5   = ("5s",  _ppy(_US_TRADING_HOURS * 60 * 12))           # 1_179_360

    def __init__(self, label: str, periods_per_year: int) -> None:
        self.label = label
        self.periods_per_year = periods_per_year

    @classmethod
    def from_label(cls, label: str) -> "ReportingFrequency":
        """Return the member whose label matches (case-insensitive)."""
        target = label.lower()
        for member in cls:
            if member.label == target:
                return member
        raise ValueError(f"Unknown timeframe label: {label!r}")

    @classmethod
    def from_periods_per_year(cls, ppy: int) -> "ReportingFrequency | None":
        """Return the member with an exact periods_per_year match, or None."""
        for member in cls:
            if member.periods_per_year == ppy:
                return member
        return None


class Algorithm(StrEnum):
    """Supported RL training algorithms."""

    PPO = "PPO"
    TD3 = "TD3"
    DDPG = "DDPG"
    RANDOM = "RANDOM"


class RewardType(StrEnum):
    """Reward function variants for the trading environment."""

    LOG_RETURN = "log_return"
    DIFFERENTIAL_SHARPE = "differential_sharpe"


class EnvMode(StrEnum):
    """Experiment mode controlling which features are eligible.

    SHARED features are eligible in both MFT and HFT modes.
    """

    SHARED = "shared"
    MFT = "mft"
    HFT = "hft"


class EnvBackend(StrEnum):
    """Supported trading environment backends."""

    GYM_TRADING_DISCRETE = "gym_trading_env.discrete"
    GYM_TRADING_CONTINUOUS = "gym_trading_env.continuous"
    GYM_ANYTRADING_FOREX = "gym_anytrading.forex"
    GYM_ANYTRADING_STOCKS = "gym_anytrading.stocks"
    TRADINGENV = "tradingenv"


class Severity(StrEnum):
    """Validation issue severity levels."""

    ERROR = "error"
    WARNING = "warning"


class ExplainabilityMethod(StrEnum):
    """Feature importance methods for post-training explainability."""

    PERMUTATION = "permutation"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    MERGED = "merged"


class StatisticalTest(StrEnum):
    """Statistical significance tests for equity curve comparison."""

    T_TEST = "t_test"
    SHARPE_BOOTSTRAP = "sharpe_bootstrap"
    SORTINO_BOOTSTRAP = "sortino_bootstrap"
    MANN_WHITNEY = "mann_whitney"
    PERMUTATION_TEST = "permutation_test"


class DataFormat(StrEnum):
    """Supported file formats for persisting DataFrames."""

    PARQUET = "parquet"
    CSV = "csv"
    PICKLE = "pickle"


class EnsembleMethod(StrEnum):
    """Feature selection ensemble methods for cross-validation."""

    MAJORITY = "majority"
    RANK_AVERAGE = "rank_average"
    WEIGHTED = "weighted"


class LossFunction(StrEnum):
    """Critic loss function variants accepted by TorchRL loss modules."""

    L1 = "l1"
    L2 = "l2"
    SMOOTH_L1 = "smooth_l1"
    HUBER = "huber"


class EvalSymbolSelection(StrEnum):
    """Strategy for choosing the representative symbol in streaming/memmap eval mode."""

    FIRST = "first"
    RANDOM = "random"
    ROTATED = "rotated"


class SplitName(StrEnum):
    """Dataset split names."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class BenchmarkName(StrEnum):
    """Evaluation benchmark names used in reports and artifacts."""

    BUY_AND_HOLD = "buy_and_hold"
    TWAP = "twap"
    VWAP = "vwap"
    MAX_PROFIT = "max_profit"
    RANDOM_ACTIONS = "random_actions"


class MetricName(StrEnum):
    """Financial performance metric identifiers.

    Values match MetricReport field names exactly, enabling dict-based
    filtering without string literals scattered through calling code.

    Excluded from HFT configs: ANNUALIZED_RETURN_CAGR, CALMAR_RATIO —
    both annualise intra-session tick returns, producing misleading numbers
    on a data horizon of hours rather than years.
    """

    TOTAL_RETURN = "total_return"
    ANNUALIZED_RETURN_CAGR = "annualized_return_cagr"
    ANNUALIZED_VOLATILITY = "annualized_volatility"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    OMEGA_RATIO = "omega_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    AVERAGE_DRAWDOWN = "average_drawdown"
    MAX_DRAWDOWN_DURATION = "max_drawdown_duration"
    RECOVERY_TIME_FROM_MAX_DRAWDOWN = "recovery_time_from_max_drawdown"
    VAR_95 = "var_95"
    CVAR_95 = "cvar_95"
    DOWNSIDE_DEVIATION = "downside_deviation"
    RETURN_SKEWNESS = "return_skewness"
    RETURN_KURTOSIS = "return_kurtosis"
    WIN_RATE = "win_rate"
    LOSE_RATE = "lose_rate"
    PROFIT_FACTOR = "profit_factor"
    PAYOFF_RATIO = "payoff_ratio"
    EXPECTANCY_PER_PERIOD = "expectancy_per_period"
    TURNOVER = "turnover"
    AVERAGE_HOLDING_PERIOD = "average_holding_period"
    PCT_LONG = "pct_long"
    PCT_SHORT = "pct_short"
    BETA = "beta"
    ALPHA = "alpha"
    INFORMATION_RATIO = "information_ratio"
    TRACKING_ERROR = "tracking_error"


class MBOSide(StrEnum):
    """Order book side codes used in Databento MBO records."""

    ASK = "A"
    BID = "B"


class MBOAction(StrEnum):
    """Order book action codes used in Databento MBO records."""

    ADD = "A"
    CANCEL = "C"
    MODIFY = "M"
    CLEAR = "R"
    TRADE = "T"
    FILL = "F"
    NONE = "N"


class TradePosition(IntEnum):
    """Discrete trading position values used by the trading environment.

    These are the raw position integers passed to the environment and stored
    in the positions list.  The integer values (-1, 0, 1) are the canonical
    representation; the names are the human-readable labels used in plots.
    """

    SHORT = -1
    HOLD = 0
    LONG = 1
