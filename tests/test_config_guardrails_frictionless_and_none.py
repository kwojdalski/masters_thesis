"""Guardrail checks added / fixed for experiment-audit 2026-08-31 findings #17, #18.

#17 - `data.train_size: null` (a legal "use every row" setting) made five
size-ratio checks raise TypeError, which `check_config_guardrails` swallowed to a
log line while still printing "Guardrails passed".

#18 - nothing flagged the frictionless-microstructure combination
(trading_fees=0 + execution_price='mid' + exec_latency_ticks=0 on a microstructure
feature set) that lets an agent harvest the uncharged half-spread.
"""

from __future__ import annotations

import pytest

from trading_rl.config import ExperimentConfig
from trading_rl.config_guardrails_checks import (
    Severity,
    _check_frictionless_microstructure,
    check_config_guardrails,
)

_FRICTIONLESS_PARAM = "env.trading_fees / env.execution_price / env.exec_latency_ticks"


# ---------------------------------------------------------------------------
# #17 - train_size: None must not crash any check
# ---------------------------------------------------------------------------


def _config_that_would_trip_the_size_checks() -> ExperimentConfig:
    """train_size=None with every condition the 5 ratio checks look at set to a
    value that WOULD fire if train_size were an int."""
    config = ExperimentConfig()
    config.data.train_size = None
    config.data.memmap_dir = "memmap-dir-placeholder"  # defeats streaming early-returns
    config.data.warmup_rows = 10_000
    config.env.streaming_episode_length = 1_000_000
    config.training.frames_per_batch = 1_000_000
    return config


def test_train_size_none_does_not_crash_any_guardrail_check() -> None:
    findings = check_config_guardrails(_config_that_would_trip_the_size_checks())

    # A swallowed TypeError now surfaces as a WARN with a "guardrail:" parameter;
    # there must be none of those.
    crashed = [f for f in findings if f.parameter.startswith("guardrail:")]
    assert not crashed, [f.message for f in crashed]


@pytest.mark.parametrize(
    "check_name",
    [
        "_check_streaming_episode_vs_train_size",
        "_check_train_size_vs_warmup_rows",
        "_check_warmup_rows",
        "_check_frames_per_batch_vs_train_size",
        "_check_streaming_episode_too_long",
    ],
)
def test_each_size_check_returns_none_for_null_train_size(check_name: str) -> None:
    from trading_rl import config_guardrails_checks as m

    check = getattr(m, check_name)
    assert check(_config_that_would_trip_the_size_checks()) is None


def test_a_check_that_raises_is_surfaced_as_a_warn_finding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_rl import config_guardrails_checks as m

    def _boom(_config: ExperimentConfig):
        raise RuntimeError("simulated guardrail bug")

    _boom.__name__ = "_check_simulated_bug"
    monkeypatch.setattr(m, "_ALL_CHECKS", [_boom])

    findings = check_config_guardrails(ExperimentConfig())

    assert len(findings) == 1
    assert findings[0].severity == Severity.WARN
    assert findings[0].parameter == "guardrail:_check_simulated_bug"
    assert "simulated guardrail bug" in findings[0].message


# ---------------------------------------------------------------------------
# #18 - frictionless-microstructure combination
# ---------------------------------------------------------------------------


def _frictionless_micro_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.env.trading_fees = 0.0
    config.env.execution_price = "mid"
    config.env.exec_latency_ticks = 0
    config.env.feature_columns = [
        "feature_hft_microprice_divergence",
        "feature_hft_ofi",
        "feature_position",
    ]
    return config


def test_frictionless_microstructure_combination_warns() -> None:
    finding = _check_frictionless_microstructure(_frictionless_micro_config())
    assert finding is not None
    assert finding.severity == Severity.WARN
    assert finding.parameter == _FRICTIONLESS_PARAM


def test_opt_in_silences_the_frictionless_warning() -> None:
    config = _frictionless_micro_config()
    config.env.allow_frictionless = True
    assert _check_frictionless_microstructure(config) is None


@pytest.mark.parametrize(
    ("attr", "value"),
    [
        ("trading_fees", 1e-4),
        ("execution_price", "bid_ask"),
        ("exec_latency_ticks", 1),
    ],
)
def test_any_single_realistic_ingredient_clears_the_warning(
    attr: str, value: object
) -> None:
    config = _frictionless_micro_config()
    setattr(config.env, attr, value)
    assert _check_frictionless_microstructure(config) is None


def test_frictionless_without_microstructure_features_does_not_warn() -> None:
    config = _frictionless_micro_config()
    config.env.feature_columns = ["feature_log_return", "feature_position"]
    assert _check_frictionless_microstructure(config) is None


def test_the_four_h1_dsr_scenarios_opt_in_so_the_warning_is_silent() -> None:
    """The deliberate signal-ceiling arm sets env.allow_frictionless: true."""
    from pathlib import Path

    root = Path("src/configs/scenarios/pooled")
    for algo in ("td3", "ddpg", "ppo", "random"):
        scenario = root / f"{algo}_hft_lob_state_space_pooled_streaming_selected_dsr"
        config = ExperimentConfig.load(scenario, command="train")
        assert config.env.allow_frictionless is True, algo
        assert _check_frictionless_microstructure(config) is None, algo
