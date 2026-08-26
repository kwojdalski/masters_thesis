"""Tests for enhanced feature selector capabilities."""

import numpy as np
import pandas as pd
import pytest
import yaml

from trading_rl.features.base import FeatureConfig
from trading_rl.features.selector import (
    FeatureSelectionResult,
    FeatureSelector,
    FeatureSelectorConfig,
    _build_multi_horizon_score_table,
    _build_proxy_target,
    _build_time_series_cv_splits,
    _compute_ic_series,
    _ensemble_select_features,
    _resolve_price_series,
    _select_features_conditional_ic,
)


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 5000

    dates = pd.date_range("2020-01-01", periods=n, freq="1min")
    price = 100 + np.cumsum(np.random.randn(n) * 0.1)
    volume = 1000 + np.random.randint(-100, 100, n)

    df = pd.DataFrame(
        {
            "close": price,
            "volume": volume,
            "high": price * 1.01,
            "low": price * 0.99,
            "open": price * 0.995,
        },
        index=dates,
    )
    return df


@pytest.fixture
def sample_features():
    """Create sample feature configurations."""
    return [
        FeatureConfig(
            name="log_return",
            feature_type="log_return",
            normalize=True,
        ),
        FeatureConfig(
            name="log_volume",
            feature_type="log_volume",
            normalize=True,
        ),
        FeatureConfig(
            name="high",
            feature_type="high",
            normalize=True,
        ),
        FeatureConfig(
            name="low",
            feature_type="low",
            normalize=True,
        ),
    ]


class TestTimeSeriesCV:
    """Tests for time-series cross-validation."""

    def test_cv_split_generation(self, sample_ohlcv_data):
        """Test that CV splits are generated correctly."""
        splits = _build_time_series_cv_splits(
            df=sample_ohlcv_data,
            n_splits=3,
            test_size=500,
            gap=50,
            min_train_size=1000,
        )

        assert len(splits) == 3

        # Check that splits don't overlap
        for i in range(len(splits) - 1):
            current_val_end = splits[i][3]
            next_train_end = splits[i + 1][1]
            assert current_val_end <= next_train_end, "Splits should not overlap"

        # Check that all indices are within bounds
        n_samples = len(sample_ohlcv_data)
        for train_start, train_end, val_start, val_end in splits:
            assert 0 <= train_start < train_end < n_samples
            assert 0 <= val_start < val_end < n_samples
            assert (
                train_end + 50 == val_start
            ), f"Gap of 50 expected: {train_end} -> {val_start}"

    def test_cv_split_with_insufficient_data(self, sample_ohlcv_data):
        """Test that CV splits handle insufficient data gracefully."""
        splits = _build_time_series_cv_splits(
            df=sample_ohlcv_data,
            n_splits=10,  # Too many for this dataset
            test_size=500,
            gap=50,
            min_train_size=1000,
        )

        # Should generate fewer splits than requested
        assert len(splits) < 10
        assert len(splits) > 0


class TestProxyTargetPriceResolution:
    def test_bid_ask_midpoint_is_used_when_close_is_absent(self):
        df = pd.DataFrame(
            {
                "bid_px_00": [99.0, 100.0, 102.0],
                "ask_px_00": [101.0, 102.0, 104.0],
            }
        )

        price = _resolve_price_series(df)
        target = _build_proxy_target(df, horizon=1)

        np.testing.assert_allclose(price.to_numpy(), [100.0, 101.0, 103.0])
        assert target.iloc[0] == pytest.approx(np.log(101.0 / 100.0))
        assert target.iloc[1] == pytest.approx(np.log(103.0 / 101.0))
        assert np.isnan(target.iloc[2])

    def test_missing_price_columns_raise_clear_error(self):
        with pytest.raises(ValueError, match="close"):
            _resolve_price_series(pd.DataFrame({"feature_x": [1.0, 2.0]}))


class TestMultiHorizonScoring:
    """Tests for multi-horizon composite scoring."""

    def test_multi_horizon_score_table(self, sample_ohlcv_data, sample_features):
        """Test that multi-horizon scoring produces composite scores."""
        train_size = 2000
        val_size = 1000

        train_df = sample_ohlcv_data.iloc[:train_size].copy()
        val_df = sample_ohlcv_data.iloc[train_size : train_size + val_size].copy()

        scores, _ic_series = _build_multi_horizon_score_table(
            train_frame=train_df,
            val_frame=val_df,
            feature_configs=sample_features,
            window_size=None,
            horizons=[1, 5, 10],
            horizon_weights=[0.5, 0.3, 0.2],
        )

        # Check that we have scores for each feature
        assert len(scores) > 0
        assert "feature" in scores.columns
        assert "icir" in scores.columns
        assert "horizon_ic_scores" in scores.columns

        # Check that horizon_ic_scores contains multiple horizons
        for _, row in scores.iterrows():
            horizon_scores = row["horizon_ic_scores"]
            assert isinstance(horizon_scores, str)
            assert "," in horizon_scores  # Multiple horizons

    def test_single_horizon_fallback(self, sample_ohlcv_data, sample_features):
        """Test that single horizon works as fallback."""
        train_size = 2000
        val_size = 1000

        train_df = sample_ohlcv_data.iloc[:train_size].copy()
        val_df = sample_ohlcv_data.iloc[train_size : train_size + val_size].copy()

        scores, _ic_series = _build_multi_horizon_score_table(
            train_frame=train_df,
            val_frame=val_df,
            feature_configs=sample_features,
            window_size=None,
            horizons=None,  # Should fallback to single horizon
            horizon_weights=None,
        )

        assert len(scores) > 0
        assert "icir" in scores.columns

    def test_multi_horizon_ic_series_uses_training_split(self):
        """Returned IC series should use train data, matching single-horizon mode."""

        class SignalPipeline:
            def transform(self, df):
                return df[["feature_signal"]]

        train_df = pd.DataFrame(
            {
                "close": np.linspace(100.0, 130.0, 30),
                "feature_signal": np.arange(30, dtype=float),
            },
            index=pd.date_range("2024-01-01", periods=30, freq="min"),
        )
        val_df = pd.DataFrame(
            {
                "close": np.linspace(200.0, 230.0, 30),
                "feature_signal": np.arange(30, dtype=float)[::-1],
            },
            index=pd.date_range("2024-02-01", periods=30, freq="min"),
        )

        _scores, ic_series = _build_multi_horizon_score_table(
            train_frame=train_df,
            val_frame=val_df,
            feature_pipeline=SignalPipeline(),
            window_size=None,
            horizons=[1],
            horizon_weights=[1.0],
        )

        first_ic_index = ic_series["feature_signal"].index[0]
        assert first_ic_index == train_df.index[0]
        assert first_ic_index != val_df.index[0]


class TestEnsembleSelection:
    """Tests for ensemble selection across CV splits."""

    def test_majority_voting(self):
        """Test majority voting ensemble."""
        split_selections = [
            ["feat_a", "feat_b", "feat_c"],
            ["feat_a", "feat_b", "feat_d"],
            ["feat_a", "feat_c", "feat_e"],
        ]

        ensemble = _ensemble_select_features(
            split_selections=split_selections,
            ensemble_method="majority",
        )

        # feat_a appears in all 3 splits, should be selected
        assert "feat_a" in ensemble

    def test_rank_averaging(self):
        """Test rank averaging ensemble."""
        split_selections = [
            ["feat_a", "feat_b", "feat_c"],
            ["feat_b", "feat_a", "feat_c"],  # feat_a and feat_b swapped
            ["feat_a", "feat_c", "feat_b"],  # feat_a first, others swapped
        ]

        ensemble = _ensemble_select_features(
            split_selections=split_selections,
            ensemble_method="rank_average",
        )

        # feat_a should have best average rank (always first)
        assert ensemble[0] == "feat_a"

    def test_rank_averaging_penalizes_missing_splits(self):
        """A one-fold top rank should not beat a consistently selected feature."""
        split_selections = [
            ["rare_spike", "stable"],
            ["stable", "other_a"],
            ["stable", "other_b"],
        ]

        ensemble = _ensemble_select_features(
            split_selections=split_selections,
            ensemble_method="rank_average",
        )

        assert ensemble[0] == "stable"
        assert ensemble.index("stable") < ensemble.index("rare_spike")

    def test_weighted_ensemble(self):
        """Test weighted ensemble using ICIR scores."""
        split_selections = [
            ["feat_a", "feat_b"],
            ["feat_a", "feat_b"],
        ]

        # Create fake scores with feat_a having higher ICIR
        scores_per_split = [
            pd.DataFrame(
                {
                    "feature": ["feat_a", "feat_b"],
                    "icir": [0.5, 0.1],
                }
            ),
            pd.DataFrame(
                {
                    "feature": ["feat_a", "feat_b"],
                    "icir": [0.4, 0.2],
                }
            ),
        ]

        ensemble = _ensemble_select_features(
            split_selections=split_selections,
            ensemble_method="weighted",
            scores_per_split=scores_per_split,
        )

        # feat_a should be first due to higher ICIR
        assert ensemble[0] == "feat_a"


class TestConditionalICSelection:
    def test_linear_duplicate_is_skipped_after_residualization(self):
        rng = np.random.default_rng(123)
        n = 200
        primary = rng.standard_normal(n)
        incremental = rng.standard_normal(n)
        target = pd.Series(primary + incremental, name="target")
        feature_data = pd.DataFrame(
            {
                "feature_primary": primary,
                "feature_duplicate": primary * 3.0,
                "feature_incremental": incremental,
            }
        )
        scores = pd.DataFrame(
            {
                "feature": [
                    "feature_primary",
                    "feature_duplicate",
                    "feature_incremental",
                ],
                "icir": [3.0, 2.9, 2.0],
            }
        )

        selected = _select_features_conditional_ic(
            scores=scores,
            feature_data=feature_data,
            target=target,
            top_k=2,
            icir_threshold=0.05,
        )

        assert selected == ["feature_primary", "feature_incremental"]


class TestFeatureSelector:
    """Tests for FeatureSelector with new capabilities."""

    def test_single_split_selection(self, sample_ohlcv_data, sample_features):
        """Test standard single-split selection."""
        train_size = 3000
        val_size = 1000

        train_df = sample_ohlcv_data.iloc[:train_size].copy()
        val_df = sample_ohlcv_data.iloc[train_size : train_size + val_size].copy()

        config = FeatureSelectorConfig(top_k=3, horizon=1)
        selector = FeatureSelector(config)

        result = selector.select(sample_features, train_df, val_df)

        assert len(result.selected_configs) <= 3
        assert len(result.selected_names) <= 3
        assert len(result.scores) > 0

    def test_multi_horizon_selection(self, sample_ohlcv_data, sample_features):
        """Test selection with multi-horizon scoring."""
        train_size = 3000
        val_size = 1000

        train_df = sample_ohlcv_data.iloc[:train_size].copy()
        val_df = sample_ohlcv_data.iloc[train_size : train_size + val_size].copy()

        config = FeatureSelectorConfig(
            top_k=3,
            use_multi_horizon=True,
            ic_decay_horizons=[1, 5, 10],
            horizon_weights=[0.5, 0.3, 0.2],
        )
        selector = FeatureSelector(config)

        result = selector.select(sample_features, train_df, val_df)

        assert len(result.selected_configs) <= 3
        # Check that scores include horizon information
        assert "horizon_ic_scores" in result.scores.columns

    def test_cross_validated_selection(self, sample_ohlcv_data, sample_features):
        """Test cross-validated selection."""
        config = FeatureSelectorConfig(
            top_k=3,
            use_cross_validation=True,
            n_cv_splits=3,
            cv_test_size=500,
            cv_gap=50,
            ensemble_method="rank_average",
        )
        selector = FeatureSelector(config)

        result = selector.select(sample_features, df=sample_ohlcv_data)

        assert len(result.selected_configs) <= 3
        assert len(result.scores) > 0

    def test_hyperparameter_search(self, sample_ohlcv_data, sample_features):
        """Test hyperparameter search."""
        config = FeatureSelectorConfig(
            enable_hyperparameter_search=True,
            hyperparameter_grid={
                "top_k": [2, 3],
                "icir_threshold": [0.01, 0.02],
            },
            n_cv_splits=2,
            cv_test_size=500,
            use_cross_validation=True,
        )
        selector = FeatureSelector(config)

        result = selector.select(sample_features, df=sample_ohlcv_data)

        assert len(result.selected_configs) <= 3
        assert len(result.selected_configs) >= 0

    def test_hyperparameter_search_scores_only_selected_features(self, monkeypatch):
        """Grid search objective should ignore unselected candidate scores."""
        weak = FeatureConfig(name="weak", feature_type="close")
        strong = FeatureConfig(name="strong", feature_type="close")
        score_table = pd.DataFrame(
            {
                "feature": ["weak", "strong"],
                "icir": [0.1, 0.9],
            }
        )

        def fake_select(self, feature_configs, train_df=None, val_df=None, df=None):
            selected = [weak] if self.config.top_k == 1 else [strong]
            return FeatureSelectionResult(
                selected_configs=selected,
                scores=score_table,
                ic_series={},
                correlation_matrix=pd.DataFrame(),
                selected_names=[selected[0].name],
                top_k=self.config.top_k,
                icir_threshold=self.config.icir_threshold,
            )

        monkeypatch.setattr(FeatureSelector, "select", fake_select)
        selector = FeatureSelector(
            FeatureSelectorConfig(
                enable_hyperparameter_search=True,
                hyperparameter_grid={"top_k": [1, 2]},
            )
        )

        result = selector._hyperparameter_search([weak, strong])

        assert result.selected_names == ["strong"]

    def test_combined_cv_and_multi_horizon(self, sample_ohlcv_data, sample_features):
        """Test combining CV and multi-horizon."""
        config = FeatureSelectorConfig(
            top_k=3,
            use_cross_validation=True,
            use_multi_horizon=True,
            n_cv_splits=3,
            cv_test_size=500,
            ic_decay_horizons=[1, 5],
            horizon_weights=[0.6, 0.4],
            ensemble_method="majority",
        )
        selector = FeatureSelector(config)

        result = selector.select(sample_features, df=sample_ohlcv_data)

        assert len(result.selected_configs) <= 3


class TestConfigurationValidation:
    """Tests for configuration validation."""

    def test_invalid_ensemble_method(self):
        """Test that invalid ensemble method raises error."""
        with pytest.raises(ValueError, match="ensemble_method"):
            FeatureSelectorConfig(ensemble_method="invalid")

    def test_invalid_multi_horizon_config(self):
        """Test that invalid multi-horizon config raises error."""
        # use_multi_horizon but no horizons
        with pytest.raises(ValueError, match="ic_decay_horizons"):
            FeatureSelectorConfig(
                use_multi_horizon=True,
                ic_decay_horizons=[],
            )

        # Mismatched weights length
        with pytest.raises(ValueError, match="horizon_weights"):
            FeatureSelectorConfig(
                use_multi_horizon=True,
                ic_decay_horizons=[1, 5, 10],
                horizon_weights=[0.5, 0.5],  # Only 2 weights for 3 horizons
            )

    def test_invalid_cv_config(self):
        """Test that invalid CV config raises error."""
        with pytest.raises(ValueError, match="n_cv_splits"):
            FeatureSelectorConfig(
                use_cross_validation=True,
                n_cv_splits=1,  # Must be >= 2
            )

        with pytest.raises(ValueError, match="cv_test_size"):
            FeatureSelectorConfig(
                use_cross_validation=True,
                cv_test_size=0,  # Must be > 0
            )

    def test_invalid_hyperparameter_grid(self):
        """Test that invalid hyperparameter grid raises error."""
        with pytest.raises(ValueError, match="Invalid hyperparameter"):
            FeatureSelectorConfig(
                enable_hyperparameter_search=True,
                hyperparameter_grid={
                    "invalid_param": [1, 2, 3],  # Invalid parameter name
                },
            )


class TestWriteSelectedYaml:
    def test_write_selected_yaml_preserves_selected_config_fields(self, tmp_path):
        selected = [
            FeatureConfig(
                name="log_return",
                feature_type="log_return",
                normalize=False,
                normalization_method="none",
                params={"column": "close"},
                output_name="feature_lr_custom",
                domain="mft",
            ),
            FeatureConfig(
                name="rsi",
                feature_type="rsi",
                normalize=True,
                params={"period": 3},
            ),
        ]
        result = FeatureSelectionResult(
            selected_configs=selected,
            scores=pd.DataFrame(),
            ic_series={},
            correlation_matrix=pd.DataFrame(),
            selected_names=["feature_lr_custom", "feature_rsi"],
            top_k=2,
            icir_threshold=0.02,
        )
        output_path = tmp_path / "selected_features.yaml"

        written = FeatureSelector.write_selected_yaml(result, output_path)

        assert written == output_path
        payload = yaml.safe_load(output_path.read_text(encoding="utf-8"))
        assert payload == {
            "features": [
                {
                    "name": "log_return",
                    "feature_type": "log_return",
                    "normalize": False,
                    "normalization_method": "none",
                    "rolling_window": 1000,
                    "reset_on_session_break": True,
                    "session_break_threshold_hours": 1.0,
                    "use_time_weights": False,
                    "domain": "mft",
                    "params": {"column": "close"},
                    "output_name": "feature_lr_custom",
                },
                {
                    "name": "rsi",
                    "feature_type": "rsi",
                    "normalize": True,
                    "normalization_method": "running",
                    "rolling_window": 1000,
                    "reset_on_session_break": True,
                    "session_break_threshold_hours": 1.0,
                    "use_time_weights": False,
                    "domain": "shared",
                    "params": {"period": 3},
                },
            ]
        }

    def test_write_selected_yaml_omits_empty_params_and_default_output_name(
        self, tmp_path
    ):
        result = FeatureSelectionResult(
            selected_configs=[
                FeatureConfig(name="high", feature_type="high", params={})
            ],
            scores=pd.DataFrame(),
            ic_series={},
            correlation_matrix=pd.DataFrame(),
            selected_names=["feature_high"],
            top_k=1,
            icir_threshold=0.02,
        )

        FeatureSelector.write_selected_yaml(result, tmp_path / "selected.yaml")

        payload = yaml.safe_load(
            (tmp_path / "selected.yaml").read_text(encoding="utf-8")
        )
        assert payload == {
            "features": [
                {
                    "name": "high",
                    "feature_type": "high",
                    "normalize": True,
                    "normalization_method": "running",
                    "rolling_window": 1000,
                    "reset_on_session_break": True,
                    "session_break_threshold_hours": 1.0,
                    "use_time_weights": False,
                    "domain": "shared",
                }
            ]
        }

    def test_write_selected_yaml_preserves_non_default_normalization_settings(
        self, tmp_path
    ):
        """Regression test: rolling-window normalization overrides must round-trip.

        Selecting from a config with normalization_method="rolling" and a custom
        rolling_window must not silently revert to the running-normalization
        defaults when the selected features are written back out to YAML.
        """
        result = FeatureSelectionResult(
            selected_configs=[
                FeatureConfig(
                    name="mid_price",
                    feature_type="mid_price",
                    normalize=True,
                    normalization_method="rolling",
                    rolling_window=500,
                    reset_on_session_break=False,
                    session_break_threshold_hours=2.0,
                    use_time_weights=True,
                )
            ],
            scores=pd.DataFrame(),
            ic_series={},
            correlation_matrix=pd.DataFrame(),
            selected_names=["feature_mid_price"],
            top_k=1,
            icir_threshold=0.02,
        )

        FeatureSelector.write_selected_yaml(result, tmp_path / "selected.yaml")

        payload = yaml.safe_load(
            (tmp_path / "selected.yaml").read_text(encoding="utf-8")
        )
        entry = payload["features"][0]
        assert entry["normalization_method"] == "rolling"
        assert entry["rolling_window"] == 500
        assert entry["reset_on_session_break"] is False
        assert entry["session_break_threshold_hours"] == 2.0
        assert entry["use_time_weights"] is True


class TestRollingICFormula:
    """_compute_ic_series must compute a true rolling Spearman correlation,
    not a rolling Pearson correlation of globally-assigned ranks."""

    def test_windowed_ic_matches_local_spearman_under_global_drift(self):
        n = 400
        window = 20
        base = np.linspace(0, 50, n)
        feature = base.copy()
        target = base.copy() * 0.5

        # Embed a window with perfect local rank anti-correlation inside an
        # overall upward-drifting series. A global-rank-then-Pearson
        # computation diverges from the true local Spearman here because the
        # window's globally-assigned ranks are not evenly spaced.
        w_start = 200
        idx = np.arange(w_start, w_start + window)
        feature[idx] = base[w_start] + np.arange(window)
        target[idx] = base[w_start] * 0.5 - np.arange(window)

        rolling_ic = _compute_ic_series(
            pd.Series(feature), pd.Series(target), window_size=window
        )

        assert rolling_ic.loc[idx[-1]] == pytest.approx(-1.0, abs=1e-9)
