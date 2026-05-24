from __future__ import annotations

from trading_rl.trainers.td3 import TD3Trainer


def test_td3_actor_update_waits_for_policy_delay_steps() -> None:
    trainer = TD3Trainer.__new__(TD3Trainer)
    trainer.policy_delay = 2

    assert [trainer._should_update_actor(step) for step in range(6)] == [
        False,
        False,
        True,
        False,
        True,
        False,
    ]


def test_td3_actor_update_respects_custom_policy_delay() -> None:
    trainer = TD3Trainer.__new__(TD3Trainer)
    trainer.policy_delay = 3

    assert [trainer._should_update_actor(step) for step in range(7)] == [
        False,
        False,
        False,
        True,
        False,
        False,
        True,
    ]
