"""Portfolio return extraction helpers for evaluation."""

from __future__ import annotations

import numpy as np

from logger import get_logger

logger = get_logger(__name__)


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


def extract_tradingenv_returns(env, n_steps):
    """Extract actual portfolio returns from TradingEnv broker."""
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

        log_returns = []
        for i in range(1, len(nlv_values)):
            if nlv_values[i - 1] > 0 and nlv_values[i] > 0:
                log_returns.append(np.log(nlv_values[i] / nlv_values[i - 1]))
            else:
                logger.warning(
                    "Invalid NLV values: %s, %s",
                    nlv_values[i - 1],
                    nlv_values[i],
                )
                return None

        cumulative_returns = np.cumsum([0.0, *log_returns])
        logger.info(
            "Extracted %s actual returns from TradingEnv broker. Final return: %.6f",
            len(cumulative_returns),
            cumulative_returns[-1],
        )
        return cumulative_returns
    except Exception as exc:
        logger.warning("failed to extract TradingEnv returns: %s", exc)
        return None
