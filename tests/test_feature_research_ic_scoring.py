"""Tests for IC/ICIR arithmetic and multi-file aggregation in feature research."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from trading_rl.feature_research.service import (
    _aggregate_symbol_scores,
    _compute_ic_series,
    _score_feature_at_horizon,
    run_feature_research,
)
from trading_rl.feature_research.config import FeatureResearchConfig


# ---------------------------------------------------------------------------
# Group 1: IC/ICIR arithmetic correctness
# ---------------------------------------------------------------------------


def _make_series(n: int, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.standard_normal(n), name="x")


class TestComputeIcSeries:
    def test_perfect_correlation_returns_one(self):
        feat = pd.Series(np.arange(50, dtype=float), name="f")
        target = pd.Series(np.arange(50, dtype=float), name="t")
        result = _compute_ic_series(feat, target, window_size=None)
        assert len(result) == 1
        assert pytest.approx(result.iloc[0], abs=1e-9) == 1.0

    def test_perfect_anti_correlation_returns_minus_one(self):
        feat = pd.Series(np.arange(50, dtype=float), name="f")
        target = pd.Series(-np.arange(50, dtype=float), name="t")
        result = _compute_ic_series(feat, target, window_size=None)
        assert pytest.approx(result.iloc[0], abs=1e-9) == -1.0

    def test_noise_feature_near_zero_ic(self):
        rng = np.random.default_rng(42)
        feat = pd.Series(rng.standard_normal(500), name="f")
        target = pd.Series(rng.standard_normal(500), name="t")
        result = _compute_ic_series(feat, target, window_size=None)
        assert abs(result.iloc[0]) < 0.15

    def test_matches_scipy_spearman(self):
        rng = np.random.default_rng(7)
        x = pd.Series(rng.standard_normal(80), name="f")
        y = pd.Series(rng.standard_normal(80), name="t")
        result = _compute_ic_series(x, y, window_size=None)
        expected, _ = spearmanr(x.values, y.values)
        assert pytest.approx(result.iloc[0], abs=1e-9) == expected

    def test_rolling_window_produces_multiple_values(self):
        # Need random data: perfectly sequential ranks become constant inside the
        # second rolling window, so all correlations are NaN after dropna().
        rng = np.random.default_rng(42)
        n = 200
        feat = pd.Series(rng.standard_normal(n), name="f")
        target = pd.Series(feat.values + rng.standard_normal(n) * 0.05, name="t")
        result = _compute_ic_series(feat, target, window_size=20)
        assert len(result) > 1
        assert result.mean() > 0.8

    def test_short_series_returns_zero_not_crash(self):
        feat = pd.Series([1.0, 2.0, 3.0], name="f")
        target = pd.Series([1.0, 2.0, 3.0], name="t")
        result = _compute_ic_series(feat, target, window_size=None)
        assert len(result) == 1
        assert result.iloc[0] == 0.0


class TestScoreFeatureAtHorizon:
    # window_size=1 triggers the global-Spearman path, which returns a 1-element
    # series; its std() is NaN, so the ICIR branch zeros everything out.
    # Use window_size=20 with enough data to get multiple IC windows.

    def test_high_ic_feature_has_positive_icir(self):
        rng = np.random.default_rng(0)
        n, window = 500, 20
        feat_train = pd.Series(rng.standard_normal(n), name="f_train")
        feat_val = pd.Series(rng.standard_normal(n // 5), name="f_val")
        train_target = pd.Series(feat_train.values + rng.standard_normal(n) * 0.1)
        val_target = pd.Series(feat_val.values + rng.standard_normal(n // 5) * 0.1)
        stats = _score_feature_at_horizon(
            feat_train, feat_val, train_target, val_target, window_size=window
        )
        assert stats["mean_ic"] > 0.5
        assert stats["icir"] > 0.0

    def test_noise_feature_has_low_icir(self):
        rng = np.random.default_rng(99)
        n, window = 500, 20
        feat_train = pd.Series(rng.standard_normal(n), name="f")
        feat_val = pd.Series(rng.standard_normal(n // 5), name="fv")
        train_target = pd.Series(rng.standard_normal(n), name="t")
        val_target = pd.Series(rng.standard_normal(n // 5), name="tv")
        stats = _score_feature_at_horizon(
            feat_train, feat_val, train_target, val_target, window_size=window
        )
        assert abs(stats["mean_ic"]) < 0.3

    def test_icir_equals_mean_ic_over_std_ic(self):
        rng = np.random.default_rng(3)
        n, window = 500, 30
        feat_train = pd.Series(rng.standard_normal(n), name="f")
        feat_val = pd.Series(rng.standard_normal(80), name="fv")
        train_target = pd.Series(feat_train.values * 0.5 + rng.standard_normal(n))
        val_target = pd.Series(feat_val.values * 0.5 + rng.standard_normal(80))
        stats = _score_feature_at_horizon(
            feat_train, feat_val, train_target, val_target, window_size=window
        )
        if stats["ic_std"] > 1e-10:
            expected_icir = stats["mean_ic"] / stats["ic_std"]
            assert pytest.approx(stats["icir"], rel=1e-6) == expected_icir

    def test_anti_correlated_feature_has_negative_mean_ic(self):
        rng = np.random.default_rng(5)
        n, window = 500, 20
        feat_train = pd.Series(rng.standard_normal(n), name="f")
        feat_val = pd.Series(rng.standard_normal(n // 5), name="fv")
        train_target = pd.Series(-feat_train.values + rng.standard_normal(n) * 0.1)
        val_target = pd.Series(-feat_val.values + rng.standard_normal(n // 5) * 0.1)
        stats = _score_feature_at_horizon(
            feat_train, feat_val, train_target, val_target, window_size=window
        )
        assert stats["mean_ic"] < -0.5
        assert stats["icir"] < 0.0

    def test_val_ic_is_computed_on_val_split_not_train(self):
        # Train: strong positive correlation. Val: independent noise.
        # mean_ic (from rolling train IC) should be high; val_mean_ic should be near 0.
        rng = np.random.default_rng(11)
        n, window = 500, 20
        feat_train = pd.Series(rng.standard_normal(n), name="f")
        train_target = pd.Series(feat_train.values + rng.standard_normal(n) * 0.05)
        feat_val = pd.Series(rng.standard_normal(100), name="fv")
        val_target = pd.Series(rng.standard_normal(100))
        stats = _score_feature_at_horizon(
            feat_train, feat_val, train_target, val_target, window_size=window
        )
        assert stats["mean_ic"] > 0.8
        assert abs(stats["val_mean_ic"]) < 0.5


# ---------------------------------------------------------------------------
# Group 2: Multi-file aggregation and significance counts
# ---------------------------------------------------------------------------


def _make_per_symbol_scores(
    icirs: list[float], threshold: float = 0.02
) -> list[pd.DataFrame]:
    """Build synthetic per-symbol score DataFrames with one feature each."""
    rows = []
    for icir in icirs:
        rows.append(
            pd.DataFrame(
                {
                    "feature": ["feature_a"],
                    "best_horizon": [1],
                    "mean_ic": [icir * 0.5],
                    "ic_std": [0.5 if abs(icir) > 0 else 0.0],
                    "icir": [icir],
                    "ic_tstat": [icir * 2],
                    "ic_positive_ratio": [0.6 if icir > 0 else 0.4],
                    "val_mean_ic": [icir * 0.4],
                    "ic_stability": [abs(icir) * 0.1],
                }
            )
        )
    return rows


class TestAggregateSymbolScores:
    def test_average_icir_is_mean_of_per_file_icirs(self):
        icirs = [0.5, 0.3, 0.7]
        per_sym = _make_per_symbol_scores(icirs)
        agg = _aggregate_symbol_scores(per_sym)
        assert pytest.approx(agg.loc[0, "icir"], rel=1e-6) == np.mean(icirs)

    def test_best_horizon_is_mode_across_files(self):
        rows = [
            pd.DataFrame({
                "feature": ["f"], "best_horizon": [5],
                "mean_ic": [0.1], "ic_std": [0.1], "icir": [1.0],
                "ic_tstat": [2.0], "ic_positive_ratio": [0.6],
                "val_mean_ic": [0.08], "ic_stability": [0.02],
            }),
            pd.DataFrame({
                "feature": ["f"], "best_horizon": [5],
                "mean_ic": [0.1], "ic_std": [0.1], "icir": [1.0],
                "ic_tstat": [2.0], "ic_positive_ratio": [0.6],
                "val_mean_ic": [0.08], "ic_stability": [0.02],
            }),
            pd.DataFrame({
                "feature": ["f"], "best_horizon": [10],
                "mean_ic": [0.1], "ic_std": [0.1], "icir": [1.0],
                "ic_tstat": [2.0], "ic_positive_ratio": [0.6],
                "val_mean_ic": [0.08], "ic_stability": [0.02],
            }),
        ]
        agg = _aggregate_symbol_scores(rows)
        assert int(agg.loc[0, "best_horizon"]) == 5

    def test_sorted_by_abs_icir_descending(self):
        rows = [
            pd.DataFrame({
                "feature": ["low", "high"],
                "best_horizon": [1, 1],
                "mean_ic": [0.1, 0.4],
                "ic_std": [0.1, 0.1],
                "icir": [0.3, 0.8],
                "ic_tstat": [1.0, 3.0],
                "ic_positive_ratio": [0.5, 0.7],
                "val_mean_ic": [0.05, 0.3],
                "ic_stability": [0.05, 0.1],
            })
        ]
        agg = _aggregate_symbol_scores(rows)
        assert list(agg["feature"]) == ["high", "low"]


class TestSignificanceCounts:
    """Test n_files_significant and pct_files_significant via run_feature_research."""

    def _write_parquet(self, path, n: int = 120, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2024-01-01", periods=n, freq="min")
        close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.1)
        pd.DataFrame({
            "open": close - 0.01,
            "high": close + 0.02,
            "low": close - 0.02,
            "close": close,
            "volume": 1000.0,
        }, index=idx).to_parquet(path)

    def _write_feature_config(self, path) -> None:
        path.write_text(
            "features:\n"
            "  - name: lag1\n"
            "    feature_type: return_lag\n"
            "    normalize: true\n"
            "    params:\n"
            "      column: close\n"
            "      lag: 1\n"
            "  - name: lag2\n"
            "    feature_type: return_lag\n"
            "    normalize: true\n"
            "    params:\n"
            "      column: close\n"
            "      lag: 2\n",
            encoding="utf-8",
        )

    def _make_config(self, tmp_path, data_paths: list, threshold: float = 0.0):
        feature_cfg = tmp_path / "features.yaml"
        self._write_feature_config(feature_cfg)
        output_dir = tmp_path / "out"
        paths_yaml = "\n".join(f"  - {p}" for p in data_paths)
        cfg_text = (
            "experiment_name: sig_count_test\n"
            "data:\n"
            f"  data_paths:\n{paths_yaml}\n"
            f"  feature_config: {feature_cfg}\n"
            "  train_size: 9999999\n"
            "research:\n"
            "  horizon: 1\n"
            "  top_k: 2\n"
            f"  icir_threshold: {threshold}\n"
            f"  output_dir: {output_dir}\n"
        )
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(cfg_text, encoding="utf-8")
        return FeatureResearchConfig.from_yaml(cfg_path)

    def test_n_files_significant_column_present(self, tmp_path):
        p1, p2 = tmp_path / "a.parquet", tmp_path / "b.parquet"
        self._write_parquet(p1, seed=1)
        self._write_parquet(p2, seed=2)
        config = self._make_config(tmp_path, [p1, p2], threshold=0.0)
        artifacts = run_feature_research(config=config)
        assert "n_files_significant" in artifacts.scores.columns
        assert "pct_files_significant" in artifacts.scores.columns

    def test_pct_significant_is_n_over_total_times_100(self, tmp_path):
        paths = []
        for i in range(3):
            p = tmp_path / f"sym_{i}.parquet"
            self._write_parquet(p, seed=i + 10)
            paths.append(p)
        config = self._make_config(tmp_path, paths, threshold=0.0)
        artifacts = run_feature_research(config=config)
        scores = artifacts.scores
        n_total = len(paths)
        for _, row in scores.iterrows():
            expected_pct = round(row["n_files_significant"] / n_total * 100, 1)
            assert pytest.approx(row["pct_files_significant"], abs=0.05) == expected_pct

    def test_n_files_significant_respects_threshold(self, tmp_path):
        # Build 3 parquets. Use icir_threshold=99 so nothing is significant.
        paths = []
        for i in range(3):
            p = tmp_path / f"sym_{i}.parquet"
            self._write_parquet(p, seed=i + 20)
            paths.append(p)
        config = self._make_config(tmp_path, paths, threshold=99.0)
        artifacts = run_feature_research(config=config)
        assert (artifacts.scores["n_files_significant"] == 0).all()
        assert (artifacts.scores["pct_files_significant"] == 0.0).all()

    def test_single_file_sets_counts_to_one_and_100(self, tmp_path):
        p = tmp_path / "single.parquet"
        self._write_parquet(p, seed=5)
        feature_cfg = tmp_path / "features.yaml"
        self._write_feature_config(feature_cfg)
        output_dir = tmp_path / "out"
        cfg_text = (
            "experiment_name: single_file_test\n"
            "data:\n"
            f"  data_path: {p}\n"
            f"  feature_config: {feature_cfg}\n"
            "  train_size: 80\n"
            "  validation_size: 20\n"
            "research:\n"
            "  horizon: 1\n"
            "  top_k: 2\n"
            "  icir_threshold: 0.0\n"
            f"  output_dir: {output_dir}\n"
        )
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(cfg_text, encoding="utf-8")
        config = FeatureResearchConfig.from_yaml(cfg_path)
        artifacts = run_feature_research(config=config)
        assert (artifacts.scores["n_files_significant"] == 1).all()
        assert (artifacts.scores["pct_files_significant"] == 100.0).all()
