"""Tests for explicit, immutable trainer dispatch."""

from __future__ import annotations

import importlib
import inspect

import pytest

import trading_rl.trainers.td3 as td3_module
from trading_rl.trainers.ddpg import DDPGTrainer
from trading_rl.trainers.ppo import PPOTrainer, PPOTrainerContinuous
from trading_rl.trainers.random_trainer import RandomTrainer
from trading_rl.trainers.recurrent_ppo import RecurrentPPOTrainer
from trading_rl.trainers.registry import TrainerRegistry
from trading_rl.trainers.sac import SACTrainer
from trading_rl.trainers.td3 import TD3Trainer


@pytest.mark.parametrize(
    ("algorithm", "continuous", "expected"),
    [
        ("DDPG", True, DDPGTrainer),
        ("PPO", False, PPOTrainer),
        ("PPO", True, PPOTrainerContinuous),
        ("RANDOM", False, RandomTrainer),
        ("RECURRENT_PPO", True, RecurrentPPOTrainer),
        ("SAC", True, SACTrainer),
        ("TD3", True, TD3Trainer),
    ],
)
def test_explicit_catalog_selects_expected_trainer(
    algorithm: str, continuous: bool, expected: type
) -> None:
    assert TrainerRegistry.get(algorithm, is_continuous=continuous) is expected


def test_catalog_is_immutable() -> None:
    with pytest.raises(TypeError):
        TrainerRegistry._continuous["MOCK"] = object  # type: ignore[index]

    assert "MOCK" not in TrainerRegistry.list_algorithms()


def test_catalog_lists_every_builtin_algorithm() -> None:
    assert TrainerRegistry.list_algorithms() == [
        "DDPG",
        "PPO",
        "RANDOM",
        "RECURRENT_PPO",
        "SAC",
        "TD3",
    ]


def test_random_trainer_satisfies_base_trainer_contract() -> None:
    assert not inspect.isabstract(RandomTrainer)
    trainer = RandomTrainer.__new__(RandomTrainer)
    assert trainer._algo_label == "random"
    assert trainer._get_checkpoint_network_state() == {}
    assert trainer._load_checkpoint_network_state({}) is None


def test_reimporting_trainer_module_does_not_mutate_catalog() -> None:
    before = (dict(TrainerRegistry._discrete), dict(TrainerRegistry._continuous))

    importlib.reload(td3_module)

    after = (dict(TrainerRegistry._discrete), dict(TrainerRegistry._continuous))
    assert after == before
