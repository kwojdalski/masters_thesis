"""Portfolio return extraction helpers for evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from logger import get_logger
from trading_rl.constants import RewardType

logger = get_logger(__name__)


class ReturnKind(StrEnum):
    """Semantic type for a numeric financial path."""

    SIMPLE = "simple"
    LOG = "log"
    CUMULATIVE_LOG = "cumulative_log"
    EQUITY = "equity"


@dataclass(frozen=True)
class ReturnSeries:
    """A return/equity path with explicit compounding semantics.

    ``includes_initial`` means ``values[0]`` is the pre-first-step baseline.
    It is meaningful for cumulative log and equity paths.
    """

    values: np.ndarray
    kind: ReturnKind
    name: str = "strategy"
    includes_initial: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            np.asarray(self.values, dtype=float).reshape(-1),
        )
        object.__setattr__(self, "kind", ReturnKind(self.kind))

    def to_simple(self) -> ReturnSeries:
        """Convert to per-step simple returns."""
        if self.kind == ReturnKind.SIMPLE:
            return self
        if self.kind == ReturnKind.LOG:
            return ReturnSeries(np.expm1(self.values), ReturnKind.SIMPLE, self.name)
        if self.kind == ReturnKind.CUMULATIVE_LOG:
            step_log = (
                np.diff(self.values)
                if self.includes_initial
                else np.diff(np.r_[0.0, self.values])
            )
            return ReturnSeries(np.expm1(step_log), ReturnKind.SIMPLE, self.name)
        if self.kind == ReturnKind.EQUITY:
            if self.values.size < 2:
                return ReturnSeries(np.array([], dtype=float), ReturnKind.SIMPLE, self.name)
            if not self.includes_initial:
                raise ValueError(
                    "equity ReturnSeries must include the initial value to convert to returns"
                )
            return ReturnSeries(
                self.values[1:] / self.values[:-1] - 1.0,
                ReturnKind.SIMPLE,
                self.name,
            )
        raise ValueError(f"Unsupported return kind: {self.kind}")

    def to_log(self) -> ReturnSeries:
        """Convert to per-step log returns."""
        if self.kind == ReturnKind.LOG:
            return self
        simple = self.to_simple().values
        return ReturnSeries(np.log1p(simple), ReturnKind.LOG, self.name)

    def to_cumulative_log(self, *, include_initial: bool = False) -> ReturnSeries:
        """Convert to cumulative log returns."""
        if self.kind == ReturnKind.CUMULATIVE_LOG:
            cumulative = self.values
            if include_initial and not self.includes_initial:
                cumulative = np.r_[0.0, cumulative]
            elif not include_initial and self.includes_initial:
                cumulative = cumulative[1:] if cumulative.size else cumulative
            return ReturnSeries(
                cumulative,
                ReturnKind.CUMULATIVE_LOG,
                self.name,
                includes_initial=include_initial,
            )

        cumulative = np.cumsum(self.to_log().values)
        if include_initial:
            cumulative = np.r_[0.0, cumulative]
        return ReturnSeries(
            cumulative,
            ReturnKind.CUMULATIVE_LOG,
            self.name,
            includes_initial=include_initial,
        )

    def to_equity(self, initial_value: float = 1.0) -> ReturnSeries:
        """Convert to an equity/value path including the initial value."""
        if initial_value <= 0:
            raise ValueError(f"initial_value must be > 0, got {initial_value}")
        if self.kind == ReturnKind.EQUITY:
            return self
        cumulative = self.to_cumulative_log(include_initial=True).values
        return ReturnSeries(
            initial_value * np.exp(cumulative),
            ReturnKind.EQUITY,
            self.name,
            includes_initial=True,
        )


@dataclass(frozen=True)
class RewardSeries:
    """RL reward stream; only log-return rewards can become returns."""

    values: np.ndarray
    reward_type: str
    name: str = "strategy"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            np.asarray(self.values, dtype=float).reshape(-1),
        )

    def to_return_series(self) -> ReturnSeries:
        if self.reward_type != RewardType.LOG_RETURN:
            raise ValueError(
                f"reward_type={self.reward_type!r} is a shaped reward, not a return"
            )
        return ReturnSeries(self.values, ReturnKind.LOG, self.name)


def calculate_actual_returns(rollout, env=None):
    """Calculate actual portfolio returns (log returns) from a rollout."""
    obs = rollout["next"]["observation"]
    n_steps = obs.shape[0]

    if hasattr(env, "_env") and hasattr(env._env, "_env"):
        try:
            trading_env = env._env._env._env
            if hasattr(trading_env, "broker"):
                pass
        except (AttributeError, IndexError):
            pass

    actions = rollout["action"].squeeze()
    if actions.ndim > 1 and actions.shape[-1] > 1:
        actions = actions.argmax(dim=-1) - 1

    rewards = rollout["next"]["reward"][:n_steps].detach().cpu().numpy()
    cumulative_returns = np.cumsum(rewards)
    logger.debug(
        "Calculated actual returns from rollout: %s steps, final return: %.4f",
        n_steps,
        cumulative_returns[-1],
    )
    return cumulative_returns


def _find_broker(env, max_depth: int = 8):
    """Traverse a TorchRL env stack to find the tradingenv broker.

    Different env topologies use either ._env (TransformedEnv) or .env
    (TorchRL GymWrapper) to reference the next layer, so both are tried.
    Returns the broker object or None if not found.
    """
    obj = env
    for _ in range(max_depth):
        if hasattr(obj, "broker"):
            broker = obj.broker
            if broker is not None:
                return broker
        next_obj = getattr(obj, "_env", None) or getattr(obj, "env", None)
        if next_obj is None or next_obj is obj:
            break
        obj = next_obj
    return None


def extract_tradingenv_return_series(env: Any, n_steps: int) -> ReturnSeries | None:
    """Extract the TradingEnv broker NLV path as an equity ReturnSeries."""
    try:
        broker = _find_broker(env)
        if broker is None:
            logger.debug("cannot find tradingenv broker in env stack")
            return None
        if not hasattr(broker, "track_record") or len(broker.track_record) == 0:
            logger.debug("broker has no track_record or it's empty")
            return None

        nlv_values = []
        max_records = min(n_steps, len(broker.track_record))
        for i in range(max_records):
            record = broker.track_record[i]
            if i == 0 and hasattr(record, "context_pre") and hasattr(record.context_pre, "nlv"):
                nlv_values.append(float(record.context_pre.nlv))
            if hasattr(record, "context_post") and hasattr(record.context_post, "nlv"):
                nlv_values.append(float(record.context_post.nlv))
            else:
                logger.warning("track record missing context_post.nlv")
                return None

        if len(nlv_values) < 2:
            logger.debug("insufficient nlv values count=%s", len(nlv_values))
            return None

        if np.any(np.asarray(nlv_values, dtype=float) <= 0):
            logger.warning("Invalid NLV path contains non-positive values: %s", nlv_values)
            return None

        series = ReturnSeries(
            np.asarray(nlv_values, dtype=float),
            ReturnKind.EQUITY,
            name="strategy",
            includes_initial=True,
        )
        logger.info(
            "Extracted %s NLV values from TradingEnv broker. Final value: %.6f",
            len(series.values),
            series.values[-1],
        )
        return series
    except Exception as exc:
        logger.warning("failed to extract TradingEnv returns: %s", exc)
        return None


def extract_tradingenv_returns(env, n_steps):
    """Extract cumulative log returns from TradingEnv broker.

    Backward-compatible wrapper. New code should prefer
    :func:`extract_tradingenv_return_series`.
    """
    series = extract_tradingenv_return_series(env, n_steps)
    if series is None:
        return None
    return series.to_cumulative_log(include_initial=True).values
