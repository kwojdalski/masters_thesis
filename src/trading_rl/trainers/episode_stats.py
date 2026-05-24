"""Episode statistics tracking for RL trainers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from logger import get_logger
from trading_rl.config import DEFAULT_INITIAL_PORTFOLIO_VALUE
from trading_rl.constants import RewardType, TradePosition

logger = get_logger(__name__)


class EpisodeStatsTracker:
    """Tracks and logs per-episode rewards, actions, and portfolio metrics.

    Extracted from BaseTrainer so episode bookkeeping can be unit-tested
    without constructing a full trainer with a live environment.

    Args:
        env: The training environment (used as fallback for DSR return extraction).
        logs: Shared trainer log dict (written to in-place).
        compute_exploration_ratio: Callable returning the current exploration ratio.
        get_last_episode_final_nlv: Callable returning (final_nlv, n_steps) for the
            last completed episode.
        get_current_episode_context: Callable returning (symbol, start_ts, end_ts)
            for the episode currently in progress.
    """

    def __init__(
        self,
        *,
        env: Any,
        logs: dict,
        compute_exploration_ratio: Callable[[], float],
        get_last_episode_final_nlv: Callable[[], tuple[float | None, int | None]],
        get_current_episode_context: Callable[[], tuple[str | None, str | None, str | None]],
    ) -> None:
        self._env = env
        self._logs = logs
        self._compute_exploration_ratio = compute_exploration_ratio
        self._get_last_episode_final_nlv = get_last_episode_final_nlv
        self._get_current_episode_context = get_current_episode_context
        self._pending_rewards: list[float] = []
        self._pending_actions: list[Any] = []

    def process_batch(self, data: Any, callback: Any) -> None:
        """Accumulate episode stats from a collector batch, flushing completed episodes."""
        rewards = (
            data["next", "reward"].detach().cpu().reshape(-1).to(torch.float64).tolist()
        )
        done_flags = (
            data["next", "done"].detach().cpu().reshape(-1).to(torch.bool).tolist()
        )
        actions = self.extract_logged_actions(data.get("action"), callback)

        pending_rewards = self._pending_rewards
        pending_actions = self._pending_actions
        segment_start = 0

        for idx, is_done in enumerate(done_flags):
            if not is_done:
                continue

            episode_rewards = [*pending_rewards, *rewards[segment_start : idx + 1]]
            episode_actions = [*pending_actions, *actions[segment_start : idx + 1]]
            self._log_completed_episode(
                episode_rewards=episode_rewards,
                actions=episode_actions,
                callback=callback,
            )
            pending_rewards = []
            pending_actions = []
            segment_start = idx + 1

        self._pending_rewards = [*pending_rewards, *rewards[segment_start:]]
        self._pending_actions = [*pending_actions, *actions[segment_start:]]

    def extract_logged_actions(
        self, actions_tensor: Any, callback: Any | None = None
    ) -> list[Any]:
        """Flatten action tensors into one logged action per transition."""
        if not isinstance(actions_tensor, torch.Tensor):
            return []
        if actions_tensor.ndim > 1 and actions_tensor.shape[-1] > 1:
            indices = actions_tensor.argmax(dim=-1).reshape(-1).tolist()
            positions = getattr(callback, "action_positions", None)
            if positions is None and actions_tensor.shape[-1] == len(TradePosition):
                positions = list(TradePosition)
            if positions and all(0 <= int(i) < len(positions) for i in indices):
                return [int(positions[int(i)]) for i in indices]
            return indices
        return actions_tensor.reshape(-1).tolist()

    def _log_completed_episode(
        self,
        *,
        episode_rewards: list[float],
        actions: list[Any],
        callback: Any,
    ) -> None:
        """Log one completed episode assembled from one or more collector batches."""
        if not episode_rewards:
            return
        episode_reward = float(np.sum(episode_rewards))
        initial_val = getattr(callback, "initial_portfolio_value", DEFAULT_INITIAL_PORTFOLIO_VALUE)
        reward_type = getattr(callback, "reward_type", RewardType.LOG_RETURN)

        episode_steps: int | None = None
        if reward_type == RewardType.LOG_RETURN:
            portfolio_valuation = initial_val * np.exp(episode_reward)
        elif reward_type == RewardType.DIFFERENTIAL_SHARPE:
            final_nlv, episode_steps = self._get_last_episode_final_nlv()
            if final_nlv is not None:
                portfolio_valuation = final_nlv
            else:
                episode_steps = None
                from trading_rl.evaluation.returns import extract_tradingenv_return_series
                actual_returns = extract_tradingenv_return_series(
                    self._env,
                    len(episode_rewards),
                )
                if actual_returns is not None and actual_returns.values.size > 0:
                    portfolio_valuation = float(actual_returns.values[-1])
                else:
                    logger.warning(
                        "failed to extract actual returns for dsr; "
                        "portfolio_valuation logged as initial capital (not a return)"
                    )
                    portfolio_valuation = initial_val
        else:
            logger.debug(
                "unrecognized reward_type=%s; portfolio_valuation logged as initial capital",
                reward_type,
            )
            portfolio_valuation = initial_val

        callback._portfolio_value = portfolio_valuation
        exploration_ratio = self._compute_exploration_ratio()

        callback.log_episode_stats(
            episode_reward=episode_reward,
            portfolio_valuation=portfolio_valuation,
            actions=actions,
            exploration_ratio=exploration_ratio,
        )

        if hasattr(callback, "_episode_count"):
            callback._episode_count += 1
        else:
            callback._episode_count = 1
        self._logs["episode_log_count"].append(int(callback._episode_count))

        portfolio_return = 100 * (portfolio_valuation / initial_val - 1)
        steps_key = "nlv_steps" if episode_steps is not None else "episode_steps"
        steps_val = episode_steps if episode_steps is not None else len(episode_rewards)
        steps_label = f" {steps_key}={steps_val}"
        symbol, start_ts, end_ts = self._get_current_episode_context()
        episode_ctx = (
            f" symbol={symbol} date_range=[{start_ts} {end_ts}]"
            if symbol else ""
        )
        if actions:
            arr = np.asarray(actions, dtype=float)
            n_act = len(arr)
            long_pct = 100.0 * np.sum(arr > 0) / n_act
            short_pct = 100.0 * np.sum(arr < 0) / n_act
            mean_exposure = float(np.mean(np.abs(arr)))
            position_str = (
                f" long_pct={long_pct:.1f}"
                f" short_pct={short_pct:.1f}"
                f" mean_exposure={mean_exposure:.3f}"
            )
        else:
            position_str = ""
        logger.info(
            "n_episode=%d portfolio_return_pct=%.2f portfolio_value=%.2f%s%s%s",
            callback._episode_count, portfolio_return, portfolio_valuation,
            steps_label, position_str, episode_ctx,
        )

    def log_sample_transitions(self, data: Any, n: int = 3) -> None:
        """Log n sample (s, a, r, s') tuples from a collected batch at DEBUG level."""
        try:
            size = data.numel()
            indices = np.linspace(0, size - 1, min(n, size), dtype=int)
            for idx in indices:
                td = data[int(idx)]
                obs = td.get("observation")
                next_obs = td.get(("next", "observation"))
                action = td.get("action")
                reward = td.get(("next", "reward"))

                def _fmt(t):
                    if t is None:
                        return "?"
                    arr = t.reshape(-1).cpu().tolist()
                    return "[" + ", ".join(f"{v:.4f}" for v in arr) + "]"

                logger.debug(
                    "transition idx=%d  s=%s  a=%s  r=%.6f  s'=%s",
                    idx, _fmt(obs), _fmt(action),
                    float(reward) if reward is not None else float("nan"),
                    _fmt(next_obs),
                )
        except Exception as exc:
            logger.debug("log_sample_transitions failed: %s", exc)
