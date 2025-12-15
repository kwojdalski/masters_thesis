"""Environment builders and factories."""

from trading_rl.envs.builder import (
    AlgorithmicEnvironmentBuilder,
    BaseEnvironmentBuilder,
)
from trading_rl.envs.trading_envs import (
    Backend,
    CustomTradingEnvironmentFactory,
    DiscreteActionWrapper,
    ForexEnvironmentFactory,
    StocksEnvironmentFactory,
    create_continuous_trading_environment,
    create_environment,
    get_environment_factory,
    validate_backend,
)
from trading_rl.envs.tradingenvxy_wrapper import (
    TradingEnvXYFactory,
    TradingEnvXYWrapper,
)

__all__ = [
    "AlgorithmicEnvironmentBuilder",
    "Backend",
    "BaseEnvironmentBuilder",
    "CustomTradingEnvironmentFactory",
    "DiscreteActionWrapper",
    "ForexEnvironmentFactory",
    "StocksEnvironmentFactory",
    "TradingEnvXYFactory",
    "TradingEnvXYWrapper",
    "create_continuous_trading_environment",
    "create_environment",
    "get_environment_factory",
    "validate_backend",
]
# The following shell command is an example for running training using a specific configuration.
# Move this to documentation or comments, not Python code.

# Example shell command (for README or comments):
