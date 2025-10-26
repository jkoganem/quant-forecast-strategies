"""Test Diebold-Mariano test on synthetic data with known superiority."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multi_ts.diagnostics.diebold_mariano import diebold_mariano_test


def test_dm_detects_superiority():
    """Test that DM test detects when one model is clearly superior."""
    np.random.seed(42)
    n = 200

    # Create synthetic actual values
    actual = pd.Series(np.random.randn(n) * 0.01, name='actual')

    # Model 1: Good forecasts (small errors)
    forecast1 = actual + np.random.randn(n) * 0.002

    # Model 2: Poor forecasts (large errors)
    forecast2 = actual + np.random.randn(n) * 0.01

    # Run DM test
    result = diebold_mariano_test(actual, forecast1, forecast2)

    # Model 1 should be significantly better (negative DM stat, low p-value)
    assert result['dm_statistic'] < 0, "DM statistic should be negative (forecast1 better)"
    assert result['p_value'] < 0.05, f"Should be significant at 5% level, got p={result['p_value']:.4f}"
    assert result['better_model'] == 'forecast1'
    assert result['significant'] is True


def test_dm_no_difference():
    """Test that DM test does not find difference when models are equal."""
    np.random.seed(42)
    n = 200

    # Create synthetic actual values
    actual = pd.Series(np.random.randn(n) * 0.01, name='actual')

    # Both models have similar error distributions
    forecast1 = actual + np.random.randn(n) * 0.005
    forecast2 = actual + np.random.randn(n) * 0.005

    # Run DM test
    result = diebold_mariano_test(actual, forecast1, forecast2)

    # Should not be significant
    assert result['p_value'] > 0.05, f"Should not be significant, got p={result['p_value']:.4f}"
    assert result['significant'] is False


def test_dm_harvey_adjustment():
    """Test that Harvey-Leybourne-Newbold adjustment is applied for small samples."""
    np.random.seed(42)
    n = 50  # Small sample

    actual = pd.Series(np.random.randn(n) * 0.01, name='actual')
    forecast1 = actual + np.random.randn(n) * 0.002
    forecast2 = actual + np.random.randn(n) * 0.01

    # Run DM test
    result = diebold_mariano_test(actual, forecast1, forecast2)

    # Check that Harvey-adjusted p-value exists
    assert 'harvey_pvalue' in result
    assert isinstance(result['harvey_pvalue'], float)

    # Harvey p-value should be larger (more conservative) than standard
    assert result['harvey_pvalue'] >= result['p_value']


def test_dm_with_nans():
    """Test that DM test handles NaN values gracefully."""
    np.random.seed(42)
    n = 200

    actual = pd.Series(np.random.randn(n) * 0.01, name='actual')
    forecast1 = actual + np.random.randn(n) * 0.002
    forecast2 = actual + np.random.randn(n) * 0.01

    # Introduce some NaNs
    forecast1.iloc[10:15] = np.nan
    forecast2.iloc[20:25] = np.nan

    # Should still work after dropping NaNs
    result = diebold_mariano_test(actual, forecast1, forecast2)

    assert result['dm_statistic'] is not None
    assert not np.isnan(result['dm_statistic'])


def test_dm_bias_detection():
    """Test that DM test detects systematic bias."""
    np.random.seed(42)
    n = 200

    actual = pd.Series(np.random.randn(n) * 0.01, name='actual')

    # Model 1: Unbiased small errors
    forecast1 = actual + np.random.randn(n) * 0.003

    # Model 2: Biased forecasts (systematically too high)
    forecast2 = actual + 0.005 + np.random.randn(n) * 0.003

    # Run DM test
    result = diebold_mariano_test(actual, forecast1, forecast2)

    # Model 1 should be significantly better due to lack of bias
    assert result['dm_statistic'] < 0
    assert result['p_value'] < 0.05
    assert result['mse_forecast1'] < result['mse_forecast2']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
