from __future__ import annotations

from pathlib import Path

from trading_rl.config import ExperimentConfig
from trading_rl.config_guardrails_checks import Severity, check_config_guardrails


def _oracle_feature_config(path: Path) -> Path:
    path.write_text(
        """
features:
  - name: future_close_vel
    feature_type: mid_price_future_velocity
    params:
      bid_price_col: bid_px_00
      ask_price_col: ask_px_00
    normalize: false
    domain: hft
""".lstrip()
    )
    return path


def test_oracle_feature_config_requires_explicit_opt_in(tmp_path: Path) -> None:
    config = ExperimentConfig()
    config.data.feature_config = str(_oracle_feature_config(tmp_path / "oracle.yaml"))

    findings = check_config_guardrails(config)

    oracle_findings = [
        finding
        for finding in findings
        if finding.parameter == "data.allow_oracle_features / data.feature_config / env.feature_columns"
    ]
    assert len(oracle_findings) == 1
    assert oracle_findings[0].severity == Severity.FATAL
    assert "future_close_vel" in oracle_findings[0].message


def test_oracle_feature_column_requires_explicit_opt_in() -> None:
    config = ExperimentConfig()
    config.env.feature_columns = ["feature_future_close_vel"]

    findings = check_config_guardrails(config)

    assert any(
        finding.severity == Severity.FATAL
        and finding.parameter == "data.allow_oracle_features / data.feature_config / env.feature_columns"
        for finding in findings
    )


def test_oracle_feature_guard_allows_declared_sanity_check(tmp_path: Path) -> None:
    config = ExperimentConfig()
    config.data.feature_config = str(_oracle_feature_config(tmp_path / "oracle.yaml"))
    config.env.feature_columns = ["feature_future_close_vel"]
    config.data.allow_oracle_features = True

    findings = check_config_guardrails(config)

    assert not any(
        finding.parameter == "data.allow_oracle_features / data.feature_config / env.feature_columns"
        for finding in findings
    )
