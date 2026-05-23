from __future__ import annotations

import pytest

from trading_rl.constants import EnvMode
from trading_rl.features.groups import FeatureGroupResolver


def _groups() -> dict:
    return {
        "imbalance": {
            "description": "Order book imbalance signals",
            "features": [
                {
                    "name": "book_pressure_l0",
                    "feature_type": "book_pressure",
                    "params": {"level": 0},
                    "output_name": "feature_book_pressure_l0",
                    "domain": "hft",
                },
                {
                    "name": "ofi",
                    "feature_type": "ofi",
                    "normalize": False,
                },
            ],
        },
        "flow": {
            "description": "Trade flow signals",
            "features": [
                {
                    "name": "ofi_duplicate",
                    "feature_type": "ofi",
                    "output_name": "feature_ofi",
                },
                {
                    "name": "signed_trade_flow",
                    "feature_type": "signed_trade_flow",
                    "params": {"window": 50},
                },
            ],
        },
    }


def test_list_groups_preserves_yaml_order() -> None:
    resolver = FeatureGroupResolver(_groups())

    assert resolver.list_groups() == ["imbalance", "flow"]


def test_get_group_description_returns_description() -> None:
    resolver = FeatureGroupResolver(_groups())

    assert resolver.get_group_description("flow") == "Trade flow signals"


def test_get_group_description_raises_for_unknown_group() -> None:
    resolver = FeatureGroupResolver(_groups())

    with pytest.raises(KeyError, match="Unknown feature group"):
        resolver.get_group_description("missing")


def test_get_group_features_converts_defaults() -> None:
    resolver = FeatureGroupResolver(_groups())

    configs = resolver.get_group_features("imbalance")

    assert configs[1].name == "ofi"
    assert configs[1].normalize is False
    assert configs[1].output_name == "feature_ofi"
    assert configs[1].domain == EnvMode.SHARED


def test_get_group_features_preserves_params_output_and_domain() -> None:
    resolver = FeatureGroupResolver(_groups())

    config = resolver.get_group_features("imbalance")[0]

    assert config.params == {"level": 0}
    assert config.output_name == "feature_book_pressure_l0"
    assert config.domain == EnvMode.HFT


def test_resolve_deduplicates_by_output_name() -> None:
    resolver = FeatureGroupResolver(_groups())

    configs = resolver.resolve(["imbalance", "flow"])

    output_names = [cfg.output_name or f"feature_{cfg.name}" for cfg in configs]
    assert output_names == [
        "feature_book_pressure_l0",
        "feature_ofi",
        "feature_signed_trade_flow",
    ]


def test_resolve_excludes_by_output_name() -> None:
    resolver = FeatureGroupResolver(_groups())

    configs = resolver.resolve(["imbalance", "flow"], exclude=["feature_ofi"])

    output_names = [cfg.output_name or f"feature_{cfg.name}" for cfg in configs]
    assert output_names == ["feature_book_pressure_l0", "feature_signed_trade_flow"]


def test_from_yaml_requires_existing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        FeatureGroupResolver.from_yaml(tmp_path / "missing.yaml")


def test_from_yaml_requires_groups_key(tmp_path) -> None:
    path = tmp_path / "groups.yaml"
    path.write_text("not_groups: {}\n")

    with pytest.raises(ValueError, match="top-level 'groups'"):
        FeatureGroupResolver.from_yaml(path)


def test_validation_rejects_feature_without_feature_type() -> None:
    groups = {"broken": {"features": [{"name": "x"}]}}

    with pytest.raises(ValueError, match="feature_type"):
        FeatureGroupResolver(groups)
