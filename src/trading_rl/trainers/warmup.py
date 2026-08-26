"""Off-policy random warmup controller."""

from __future__ import annotations

from typing import Any

from logger import get_logger

logger = get_logger(__name__)


class WarmupController:
    """Manages the random-policy warmup phase for off-policy algorithms.

    Off-policy trainers (TD3, DDPG, SAC) fill the replay buffer with random
    actions before switching to the exploration policy.  This class owns that
    state transition so BaseTrainer does not.
    """

    def __init__(
        self,
        *,
        collector: Any,
        init_rand_steps: int,
        frames_per_batch: int = 0,
        replay_buffer: Any | None = None,
        use_replay_buffer: bool = True,
    ) -> None:
        self._collector = collector
        self._init_rand_steps = self._quantize_to_batch(
            init_rand_steps, frames_per_batch
        )
        self._replay_buffer = replay_buffer
        self._use_replay_buffer = use_replay_buffer
        self._exploration_policy: Any = None
        self.random_exploration_done: bool = False

    @staticmethod
    def _quantize_to_batch(init_rand_steps: int, frames_per_batch: int) -> int:
        """Round init_rand_steps up to the nearest multiple of frames_per_batch.

        The switch from random to exploration policy can only happen at a
        collector batch boundary -- a whole batch of frames_per_batch steps
        is always collected atomically under whichever policy was active
        when collection started, so a mid-batch crossing of init_rand_steps
        cannot be split. Rounding the target up to a batch boundary makes
        the random phase length an exact multiple of frames_per_batch
        instead of overshooting the configured value by up to
        frames_per_batch - 1 steps on every run.
        """
        if frames_per_batch <= 0 or init_rand_steps <= 0:
            return init_rand_steps
        remainder = init_rand_steps % frames_per_batch
        if remainder == 0:
            return init_rand_steps
        quantized = init_rand_steps + (frames_per_batch - remainder)
        logger.info(
            "warmup init_rand_steps={} is not a multiple of frames_per_batch={}; "
            "quantized up to {} so the random phase ends exactly on a batch boundary",
            init_rand_steps,
            frames_per_batch,
            quantized,
        )
        return quantized

    def initialize(
        self,
        exploration_policy: Any,
        action_spec: Any,
        *,
        total_count: int,
        algorithm_label: str = "Off-policy",
    ) -> None:
        """Configure the collector for warmup. Call once before training starts."""
        self._exploration_policy = exploration_policy

        if total_count >= self._init_rand_steps:
            self._collector.policy = exploration_policy
            self.random_exploration_done = True
            logger.info(
                "{} random warmup already complete at {} steps; starting with exploration policy.",
                algorithm_label,
                total_count,
            )
            return

        from torchrl.envs.utils import RandomPolicy

        self._collector.policy = RandomPolicy(action_spec)
        self.random_exploration_done = False
        logger.info(
            "{} using random policy for first {} steps",
            algorithm_label,
            self._init_rand_steps,
        )

    def maybe_switch(
        self, total_count: int, *, algorithm_label: str = "Off-policy"
    ) -> None:
        """Switch from random to exploration policy once warmup steps are complete."""
        if self.random_exploration_done:
            return
        if total_count < self._init_rand_steps:
            return
        if self._exploration_policy is None:
            logger.warning(
                "{} warmup threshold reached but no exploration policy configured.",
                algorithm_label,
            )
            return

        buffer_len = (
            len(self._replay_buffer)
            if (self._use_replay_buffer and self._replay_buffer is not None)
            else 0
        )
        logger.info(
            "{} random exploration finished at {} steps. Switching to exploration policy.",
            algorithm_label,
            total_count,
        )
        logger.trace("  Buffer now contains {} transitions", buffer_len)
        self._collector.policy = self._exploration_policy
        self.random_exploration_done = True
