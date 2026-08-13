"""Regression tests: guardrail checks must read the nested TrainingConfig
sub-configs (training.td3.*, training.ppo.*) rather than pre-refactor flat
attributes that no longer exist. A check reading a removed flat attribute
raises AttributeError, which check_config_guardrails silently swallows
("never let a guardrail crash the run"), so a broken check just never fires
instead of erroring -- these tests confirm each check still fires.
"""

from __future__ import annotations

from trading_rl.config import ExperimentConfig
from trading_rl.config_guardrails_checks import Severity, check_config_guardrails


def _findings_for(config: ExperimentConfig, parameter: str) -> list:
    return [f for f in check_config_guardrails(config) if f.parameter == parameter]


def test_ppo_updates_per_rollout_check_reads_nested_epochs() -> None:
    config = ExperimentConfig()
    config.training.algorithm = "PPO"
    config.training.ppo.epochs = 50
    config.training.frames_per_batch = 64
    config.training.sample_size = 1

    findings = _findings_for(
        config, "training.ppo_epochs / training.frames_per_batch / training.sample_size"
    )
    assert len(findings) == 1
    assert findings[0].severity == Severity.WARN


def test_td3_noise_vs_clip_check_reads_nested_policy_noise_and_clip() -> None:
    config = ExperimentConfig()
    config.training.algorithm = "TD3"
    config.training.td3.policy_noise = 0.6
    config.training.td3.noise_clip = 0.3

    findings = _findings_for(config, "training.policy_noise / training.noise_clip")
    assert len(findings) == 1
    assert findings[0].severity == Severity.WARN


def test_ppo_clip_epsilon_check_reads_nested_clip_epsilon() -> None:
    config = ExperimentConfig()
    config.training.algorithm = "PPO"
    config.training.ppo.clip_epsilon = 0.9

    findings = _findings_for(config, "training.clip_epsilon")
    assert len(findings) == 1
    assert findings[0].severity == Severity.WARN


def test_no_exploration_check_reads_nested_exploration_noise_std() -> None:
    config = ExperimentConfig()
    config.training.algorithm = "TD3"
    config.training.td3.exploration_noise_std = 0.0
    config.training.init_rand_steps = 0

    findings = _findings_for(
        config, "training.exploration_noise_std / training.init_rand_steps"
    )
    assert len(findings) == 1
    assert findings[0].severity == Severity.WARN


def test_ppo_entropy_bonus_check_reads_nested_entropy_bonus() -> None:
    config = ExperimentConfig()
    config.training.algorithm = "PPO"
    config.training.ppo.entropy_bonus = 0.5

    findings = _findings_for(config, "training.entropy_bonus")
    assert len(findings) == 1
    assert findings[0].severity == Severity.WARN


def test_ppo_vf_coef_check_reads_nested_vf_coef() -> None:
    config = ExperimentConfig()
    config.training.algorithm = "PPO"
    config.training.ppo.vf_coef = 2.0

    findings = _findings_for(config, "training.vf_coef")
    assert len(findings) == 1
    assert findings[0].severity == Severity.WARN


def test_exploration_noise_too_large_check_reads_nested_exploration_noise_std() -> None:
    config = ExperimentConfig()
    config.training.algorithm = "TD3"
    config.training.td3.exploration_noise_std = 1.5

    findings = _findings_for(config, "training.exploration_noise_std")
    assert len(findings) == 1
    assert findings[0].severity == Severity.WARN
