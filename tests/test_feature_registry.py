"""Tests for FeatureRegistry — create, list_features, and registration errors."""

from __future__ import annotations

import pytest

from trading_rl.features.base import Feature, FeatureConfig
from trading_rl.features.registry import FeatureRegistry, register_feature


def _cfg(feature_type: str, **params) -> FeatureConfig:
    return FeatureConfig(name=feature_type, feature_type=feature_type, params=params)


class TestFeatureRegistryCreate:
    def test_create_log_return_returns_feature_instance(self):
        import trading_rl.features.price_features  # noqa: F401 — side-effect: registers

        feature = FeatureRegistry.create(_cfg("log_return"))
        assert isinstance(feature, Feature)

    def test_create_realized_volatility_returns_feature_instance(self):
        import trading_rl.features.volatility_features  # noqa: F401

        feature = FeatureRegistry.create(_cfg("realized_volatility"))
        assert isinstance(feature, Feature)

    def test_create_log_volume_returns_feature_instance(self):
        import trading_rl.features.volume_features  # noqa: F401

        feature = FeatureRegistry.create(_cfg("log_volume"))
        assert isinstance(feature, Feature)

    def test_create_unknown_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown feature type"):
            FeatureRegistry.create(_cfg("nonexistent_feature_xyz"))

    def test_create_error_lists_available_types(self):
        with pytest.raises(ValueError, match="Available types"):
            FeatureRegistry.create(_cfg("nonexistent_feature_xyz"))

    def test_create_passes_config_to_feature(self):
        import trading_rl.features.volatility_features  # noqa: F401

        cfg = _cfg("realized_volatility", window=42)
        feature = FeatureRegistry.create(cfg)
        assert feature.config.params["window"] == 42

    def test_create_rsi_with_custom_period(self):
        import trading_rl.features.price_features  # noqa: F401

        cfg = _cfg("rsi", period=7)
        feature = FeatureRegistry.create(cfg)
        assert feature.config.params["period"] == 7


class TestFeatureRegistryList:
    def test_list_features_returns_list_of_strings(self):
        features = FeatureRegistry.list_features()
        assert isinstance(features, list)
        assert all(isinstance(f, str) for f in features)

    def test_list_features_includes_log_return(self):
        import trading_rl.features.price_features  # noqa: F401

        assert "log_return" in FeatureRegistry.list_features()

    def test_list_features_includes_log_volume(self):
        import trading_rl.features.volume_features  # noqa: F401

        assert "log_volume" in FeatureRegistry.list_features()

    def test_list_features_includes_realized_volatility(self):
        import trading_rl.features.volatility_features  # noqa: F401

        assert "realized_volatility" in FeatureRegistry.list_features()

    def test_newly_registered_feature_appears_in_list(self):
        @register_feature("_test_sentinel_feature_do_not_use")
        class _SentinelFeature(Feature):
            def required_columns(self):
                return []

            def compute(self, df):
                import pandas as pd

                return pd.Series([], dtype=float)

        assert "_test_sentinel_feature_do_not_use" in FeatureRegistry.list_features()

    def test_register_overwrites_existing_entry(self):
        import pandas as pd

        import trading_rl.features.price_features  # noqa: F401

        original_class = FeatureRegistry._registry.get("log_return")
        try:

            @register_feature("log_return")
            class _Dummy(Feature):
                def required_columns(self):
                    return []

                def compute(self, df):
                    return pd.Series([], dtype=float)

            feature = FeatureRegistry.create(_cfg("log_return"))
            assert isinstance(feature, _Dummy)
        finally:
            if original_class is not None:
                FeatureRegistry._registry["log_return"] = original_class


class TestFeatureConfigValidation:
    def test_invalid_domain_raises_value_error(self):
        with pytest.raises(ValueError, match="domain"):
            FeatureConfig(name="x", feature_type="x", domain="invalid_domain")

    def test_invalid_normalization_method_raises_value_error(self):
        with pytest.raises(ValueError, match="normalization_method"):
            FeatureConfig(name="x", feature_type="x", normalization_method="bogus")

    def test_valid_domain_shared_accepted(self):
        cfg = FeatureConfig(name="x", feature_type="x", domain="shared")
        assert cfg.domain == "shared"

    def test_valid_domain_hft_accepted(self):
        cfg = FeatureConfig(name="x", feature_type="x", domain="hft")
        assert cfg.domain == "hft"

    def test_valid_normalization_global_accepted(self):
        cfg = FeatureConfig(name="x", feature_type="x", normalization_method="global")
        assert cfg.normalization_method == "global"

    def test_default_output_name_is_feature_name(self):
        cfg = FeatureConfig(name="my_feat", feature_type="log_return")
        assert cfg.output_name == "feature_my_feat"

    def test_params_default_to_empty_dict(self):
        cfg = FeatureConfig(name="x", feature_type="x")
        assert cfg.params == {}
