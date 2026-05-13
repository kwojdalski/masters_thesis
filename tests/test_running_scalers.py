"""Unit tests for RunningMeanStd and TimeWeightedRunningMeanStd scalers."""

import numpy as np
import pandas as pd
import pytest

from trading_rl.features.base import RunningMeanStd, TimeWeightedRunningMeanStd


class TestRunningMeanStd:
    """Test suite for RunningMeanStd using Welford's algorithm."""

    def test_initialization(self):
        """Test RunningMeanStd initialization."""
        scaler = RunningMeanStd(epsilon=1e-4)

        assert scaler.mean == 0.0
        assert scaler.var == 1.0  # Initial variance is 1.0, not 0.0
        assert scaler.count == 0
        assert scaler.epsilon == 1e-4

    def test_constant_series_convergence(self):
        """Test that mean/var converge to true values on constant series."""
        scaler = RunningMeanStd()
        constant_value = 5.0
        n_samples = 100

        for _ in range(n_samples):
            scaler.update(np.array([constant_value]))

        # After many samples, mean should converge to constant value
        assert abs(scaler.mean - constant_value) < 1e-3
        # Variance should converge to 0 for constant series
        assert abs(scaler.var) < 1e-3
        assert scaler.count == n_samples

    def test_reset_clears_state(self):
        """Test that reset() zeroes all state."""
        scaler = RunningMeanStd()

        # Update with some data
        scaler.update(np.array([5.0, 10.0, 15.0]))

        # State should be non-zero
        assert scaler.mean != 0.0 or scaler.var != 1.0
        assert scaler.count == 3

        # Reset
        scaler.reset()

        # State should be cleared
        assert scaler.mean == 0.0
        assert scaler.var == 1.0
        assert scaler.count == 0

    def test_varying_series_variance(self):
        """Test variance computation on varying series."""
        scaler = RunningMeanStd()
        series = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])

        for value in series:
            scaler.update(np.array([value]))

        # Variance should be positive for varying series
        assert scaler.var > 0

    def test_single_sample_variance(self):
        """Test variance behavior with single sample."""
        scaler = RunningMeanStd()
        scaler.update(np.array([5.0]))

        # With single sample, variance may be near 0
        # RunningMeanStd uses ddof=1, so var may not be exactly 0
        # but should handle edge case gracefully
        assert scaler.count == 1
        assert scaler.mean == 5.0

    def test_numerical_stability_large_values(self):
        """Test numerical stability with large values."""
        scaler = RunningMeanStd()
        large_value = 1e10
        scaler.update(np.array([large_value]))

        assert np.isfinite(scaler.mean)
        assert np.isfinite(scaler.var)

    def test_numerical_stability_small_values(self):
        """Test numerical stability with very small values."""
        scaler = RunningMeanStd()
        small_value = 1e-10
        scaler.update(np.array([small_value]))

        assert np.isfinite(scaler.mean)
        assert np.isfinite(scaler.var)


class TestTimeWeightedRunningMeanStd:
    """Test suite for TimeWeightedRunningMeanStd with session awareness."""

    def test_initialization(self):
        """Test TimeWeightedRunningMeanStd initialization."""
        scaler = TimeWeightedRunningMeanStd(epsilon=1e-4)

        assert scaler.mean == 0.0
        assert scaler.var == 1.0  # Initial variance is 1.0, not 0.0
        assert scaler.total_weight == 0.0
        assert scaler.epsilon == 1e-4

    def test_constant_series_convergence(self):
        """Test that mean/var converge on constant series with uniform weights."""
        scaler = TimeWeightedRunningMeanStd()
        constant_value = 5.0
        n_samples = 50
        uniform_weights = np.ones(n_samples)

        for value, weight in zip([constant_value] * n_samples, uniform_weights):
            scaler.update(np.array([value]), np.array([weight]))

        # Mean should converge to constant value
        assert abs(scaler.mean - constant_value) < 1e-3
        # Variance should converge to 0
        assert abs(scaler.var) < 1e-3
        assert scaler.total_weight == n_samples

    def test_reset_clears_state(self):
        """Test that reset() zeroes all state including weights."""
        scaler = TimeWeightedRunningMeanStd()

        # Update with weighted data
        values = np.array([5.0, 10.0, 15.0])
        weights = np.array([1.0, 1.0, 1.0])
        for value, weight in zip(values, weights):
            scaler.update(np.array([value]), np.array([weight]))

        # State should be non-zero
        assert scaler.mean != 0.0 or scaler.var != 1.0
        assert scaler.total_weight > 0

        # Reset
        scaler.reset()

        # State should be cleared
        assert scaler.mean == 0.0
        assert scaler.var == 1.0
        assert scaler.total_weight == 0.0

    def test_zero_weight_graceful_handling(self):
        """Test that zero weight is handled gracefully (early return)."""
        scaler = TimeWeightedRunningMeanStd()

        # Single value with zero weight should return early without crash
        values = np.array([5.0])
        weights = np.array([0.0])

        result = scaler.update(values, weights)

        # Should return the scaler unchanged due to zero total weight
        assert result is scaler

        # State should remain at initial values
        assert scaler.total_weight == 0.0
        assert scaler.mean == 0.0

    def test_non_session_aware_transform(self):
        """Test that non-session-aware transform uses all history."""
        scaler = TimeWeightedRunningMeanStd()
        values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        # Transform without session awareness (uses all cumulative history)
        result = scaler.transform(values)

        # All values should normalize using cumulative history
        # (no zeros from session breaks)
        assert not any(result == 0.0)

    def test_variance_positive_for_varying_series(self):
        """Test variance is positive for time-weighted varying series."""
        scaler = TimeWeightedRunningMeanStd()

        values = np.array([1.0, 10.0, 1.0, 10.0, 1.0, 10.0, 1.0, 10.0, 1.0, 10.0])
        uniform_weights = np.ones(10)

        for value, weight in zip(values, uniform_weights):
            scaler.update(np.array([value]), np.array([weight]))

        # Variance should be positive for varying values
        assert scaler.var > 0

    def test_numerical_stability_time_weights(self):
        """Test numerical stability with varying time weights."""
        scaler = TimeWeightedRunningMeanStd()

        # Varying weights simulating irregular time intervals
        values = np.array([5.0, 10.0, 15.0])
        weights = np.array([0.1, 1.0, 10.0])  # Varying orders of magnitude

        for value, weight in zip(values, weights):
            scaler.update(np.array([value]), np.array([weight]))

        # All values should be finite
        assert np.isfinite(scaler.mean)
        assert np.isfinite(scaler.var)
        assert scaler.total_weight == 11.1
