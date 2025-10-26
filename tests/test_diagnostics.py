"""Tests for statistical diagnostic functions.

Tests ADF, Ljung-Box, and ARCH LM tests.
"""

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.arima_process import ArmaProcess

from multi_ts.diagnostics.stationarity import adf_test, kpss_test
from multi_ts.diagnostics.whiteness import ljung_box_test
from multi_ts.diagnostics.arch_lm import arch_lm_test


def test_adf_on_stationary_series():
    """Test ADF on known stationary series."""
    # Generate stationary AR(1) process
    np.random.seed(42)
    ar = np.array([1, -0.5])  # AR(1) with coefficient 0.5
    ma = np.array([1])
    arma_process = ArmaProcess(ar, ma)
    y = arma_process.generate_sample(nsample=500)
    series = pd.Series(y, index=pd.date_range("2020-01-01", periods=500))

    # Run ADF test
    result = adf_test(series)

    # Should detect stationarity
    assert result["is_stationary"], "ADF should detect AR(1) as stationary"
    assert result["p_value"] < 0.05


def test_adf_on_nonstationary_series():
    """Test ADF on known non-stationary series."""
    # Generate random walk (non-stationary)
    np.random.seed(42)
    n = 500
    y = np.cumsum(np.random.randn(n))
    series = pd.Series(y, index=pd.date_range("2020-01-01", periods=n))

    # Run ADF test
    result = adf_test(series)

    # Should detect non-stationarity
    assert not result["is_stationary"], "ADF should detect random walk as non-stationary"
    assert result["p_value"] > 0.05


def test_ljung_box_on_white_noise():
    """Test Ljung-Box on white noise."""
    # Generate white noise
    np.random.seed(42)
    n = 500
    white_noise = np.random.randn(n)
    series = pd.Series(white_noise, index=pd.date_range("2020-01-01", periods=n))

    # Run Ljung-Box test
    result = ljung_box_test(series, lags=10)

    # Should not reject white noise hypothesis
    assert result["is_white"], "Ljung-Box should accept white noise"
    assert result["p_value"] > 0.05


def test_ljung_box_on_autocorrelated():
    """Test Ljung-Box on autocorrelated series."""
    # Generate AR(1) process
    np.random.seed(42)
    n = 500
    ar_coef = 0.8
    y = np.zeros(n)
    y[0] = np.random.randn()

    for t in range(1, n):
        y[t] = ar_coef * y[t-1] + np.random.randn()

    series = pd.Series(y, index=pd.date_range("2020-01-01", periods=n))

    # Run Ljung-Box test
    result = ljung_box_test(series, lags=10)

    # Should reject white noise hypothesis
    assert not result["is_white"], "Ljung-Box should reject for AR(1)"
    assert result["p_value"] < 0.05


def test_arch_lm_on_homoskedastic():
    """Test ARCH LM on homoskedastic series."""
    # Generate homoskedastic white noise
    np.random.seed(42)
    n = 500
    series = pd.Series(
        np.random.randn(n),
        index=pd.date_range("2020-01-01", periods=n)
    )

    # Run ARCH LM test
    result = arch_lm_test(series, lags=5)

    # Should not detect ARCH effects
    assert not result["has_arch"], "Should not detect ARCH in white noise"
    assert result["lm_pvalue"] > 0.05


def test_arch_lm_on_heteroskedastic():
    """Test ARCH LM on series with ARCH effects."""
    # Generate ARCH(1) process
    np.random.seed(42)
    n = 500
    omega = 0.01
    alpha = 0.5

    returns = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha)

    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t-1]**2
        returns[t] = np.sqrt(sigma2[t]) * np.random.randn()

    series = pd.Series(returns, index=pd.date_range("2020-01-01", periods=n))

    # Run ARCH LM test
    result = arch_lm_test(series, lags=5)

    # Should detect ARCH effects
    assert result["has_arch"], "Should detect ARCH effects"
    assert result["lm_pvalue"] < 0.05


def test_diagnostic_edge_cases():
    """Test diagnostic functions with edge cases."""
    # Very short series
    short_series = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3))

    adf_result = adf_test(short_series)
    assert not adf_result["is_stationary"]
    assert "Insufficient" in adf_result["conclusion"]

    lb_result = ljung_box_test(short_series, lags=5)
    assert not lb_result["is_white"]
    assert "Insufficient" in lb_result["conclusion"]

    arch_result = arch_lm_test(short_series, lags=2)
    assert not arch_result["has_arch"]
    assert "Insufficient" in arch_result["conclusion"]

    # Series with NaN
    nan_series = pd.Series(
        [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10],
        index=pd.date_range("2020-01-01", periods=10)
    )

    # Should handle NaN gracefully
    adf_result = adf_test(nan_series)
    assert "p_value" in adf_result

    lb_result = ljung_box_test(nan_series, lags=2)
    assert "p_value" in lb_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])