"""Test model output shapes, alignments, and deterministic behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multi_ts.models.garch import rolling_garch_forecast
from multi_ts.models.sarima import rolling_sarima_forecast
from multi_ts.models.dhr import rolling_dhr_forecast
from multi_ts.models.linear import rolling_ridge_forecast, rolling_lasso_forecast
from multi_ts.models.xgb import rolling_xgboost_forecast


@pytest.fixture
def synthetic_returns():
    """Generate synthetic return series for testing."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    returns = pd.Series(
        np.random.randn(n) * 0.01 + 0.0002,
        index=dates,
        name='returns'
    )
    return returns


def test_garch_output_shape(synthetic_returns):
    """Test that GARCH model returns correct shape."""
    result = rolling_garch_forecast(
        synthetic_returns,
        window_init=250,
        refit_freq=20
    )

    # Check result structure
    assert 'yhat' in result
    assert 'resid' in result
    assert 'meta' in result

    # Check alignment
    assert len(result['yhat']) == len(result['resid'])
    assert result['yhat'].index.equals(result['resid'].index)

    # Check forecast count (should be n - window_init)
    expected_forecasts = len(synthetic_returns) - 250
    assert len(result['yhat']) == expected_forecasts

    # Check all forecasts are positive (variance must be positive)
    assert (result['yhat'] > 0).all()


def test_sarima_output_shape(synthetic_returns):
    """Test that SARIMA model returns correct shape."""
    result = rolling_sarima_forecast(
        synthetic_returns,
        window_init=250,
        refit_freq=20
    )

    assert 'yhat' in result
    assert 'resid' in result
    assert len(result['yhat']) == len(result['resid'])

    # Check that we have forecasts
    expected_forecasts = len(synthetic_returns) - 250
    assert len(result['yhat']) == expected_forecasts


def test_dhr_output_shape(synthetic_returns):
    """Test that DHR model returns correct shape."""
    result = rolling_dhr_forecast(
        synthetic_returns,
        window_init=250,
        refit_freq=20
    )

    assert 'yhat' in result
    assert 'resid' in result
    assert len(result['yhat']) == len(result['resid'])

    expected_forecasts = len(synthetic_returns) - 250
    assert len(result['yhat']) == expected_forecasts


def test_ridge_output_shape(synthetic_returns):
    """Test that Ridge regression returns correct shape."""
    result = rolling_ridge_forecast(
        synthetic_returns,
        window_init=250,
        refit_freq=20
    )

    assert 'yhat' in result
    assert 'resid' in result
    assert len(result['yhat']) == len(result['resid'])

    expected_forecasts = len(synthetic_returns) - 250
    assert len(result['yhat']) == expected_forecasts


def test_lasso_output_shape(synthetic_returns):
    """Test that Lasso regression returns correct shape."""
    result = rolling_lasso_forecast(
        synthetic_returns,
        window_init=250,
        refit_freq=20
    )

    assert 'yhat' in result
    assert 'resid' in result
    assert len(result['yhat']) == len(result['resid'])

    expected_forecasts = len(synthetic_returns) - 250
    assert len(result['yhat']) == expected_forecasts


def test_xgboost_output_shape(synthetic_returns):
    """Test that XGBoost returns correct shape."""
    result = rolling_xgboost_forecast(
        synthetic_returns,
        window_init=250,
        refit_freq=20
    )

    assert 'yhat' in result
    assert 'resid' in result
    assert len(result['yhat']) == len(result['resid'])

    expected_forecasts = len(synthetic_returns) - 250
    assert len(result['yhat']) == expected_forecasts


def test_xgboost_deterministic():
    """Test that XGBoost produces deterministic results with same seed."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    returns = pd.Series(
        np.random.randn(n) * 0.01,
        index=dates,
        name='returns'
    )

    # Run twice with same seed
    result1 = rolling_xgboost_forecast(
        returns,
        window_init=150,
        refit_freq=20,
        seed=42
    )

    result2 = rolling_xgboost_forecast(
        returns,
        window_init=150,
        refit_freq=20,
        seed=42
    )

    # Results should be identical
    np.testing.assert_array_almost_equal(
        result1['yhat'].values,
        result2['yhat'].values,
        decimal=10
    )


def test_model_alignment_consistency():
    """Test that all models have consistent index alignment."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    returns = pd.Series(
        np.random.randn(n) * 0.01,
        index=dates,
        name='returns'
    )

    window_init = 150

    # Get results from all models
    garch_result = rolling_garch_forecast(returns, window_init=window_init, refit_freq=20)
    ridge_result = rolling_ridge_forecast(returns, window_init=window_init, refit_freq=20)
    xgb_result = rolling_xgboost_forecast(returns, window_init=window_init, refit_freq=20)

    # All forecasts should have same index
    assert garch_result['yhat'].index.equals(ridge_result['yhat'].index)
    assert ridge_result['yhat'].index.equals(xgb_result['yhat'].index)

    # All forecasts should start at the same date
    expected_start = returns.index[window_init]
    assert garch_result['yhat'].index[0] == expected_start
    assert ridge_result['yhat'].index[0] == expected_start
    assert xgb_result['yhat'].index[0] == expected_start


def test_no_lookahead_bias():
    """Test that models don't use future information."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range('2020-01-01', periods=n, freq='D')

    # Create a series with a known future spike
    returns = pd.Series(np.random.randn(n) * 0.01, index=dates, name='returns')
    returns.iloc[200] = 0.05  # Large spike at position 200

    result = rolling_ridge_forecast(returns, window_init=150, refit_freq=20)

    # Forecast at position 199 should NOT anticipate the spike
    # (It should be based only on data up to position 198)
    forecast_before_spike = result['yhat'].iloc[199 - 150]  # Forecast for position 199

    # The forecast should be close to historical mean, not close to the spike
    historical_mean = returns.iloc[:199].mean()
    assert abs(forecast_before_spike - historical_mean) < 0.01
    assert abs(forecast_before_spike - 0.05) > 0.02


def test_residuals_calculation():
    """Test that residuals are correctly calculated as actual - forecast."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    returns = pd.Series(
        np.random.randn(n) * 0.01,
        index=dates,
        name='returns'
    )

    result = rolling_ridge_forecast(returns, window_init=150, refit_freq=20)

    # Get overlapping period
    forecast_dates = result['yhat'].index
    actual_values = returns.loc[forecast_dates]

    # Manually calculate residuals
    expected_resid = actual_values - result['yhat']

    # Compare with returned residuals
    np.testing.assert_array_almost_equal(
        result['resid'].values,
        expected_resid.values,
        decimal=10
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
