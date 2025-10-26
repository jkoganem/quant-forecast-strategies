"""Tests for GARCH model constraints.

Verifies positivity and stationarity constraints.
"""

import numpy as np
import pandas as pd
import pytest

from multi_ts.models.garch import (
    fit_garch11,
    validate_garch_constraints,
    rolling_garch_forecast
)


def test_garch_constraints_validation():
    """Test GARCH parameter constraint validation."""
    # Valid parameters
    valid_params = {"omega": 0.01, "alpha": 0.1, "beta": 0.8}
    is_valid, msg = validate_garch_constraints(valid_params)
    assert is_valid, f"Valid params rejected: {msg}"

    # Invalid omega
    invalid_omega = {"omega": -0.01, "alpha": 0.1, "beta": 0.8}
    is_valid, msg = validate_garch_constraints(invalid_omega)
    assert not is_valid
    assert "omega" in msg.lower()

    # Invalid alpha
    invalid_alpha = {"omega": 0.01, "alpha": -0.1, "beta": 0.8}
    is_valid, msg = validate_garch_constraints(invalid_alpha)
    assert not is_valid
    assert "alpha" in msg.lower()

    # Non-stationary
    non_stationary = {"omega": 0.01, "alpha": 0.5, "beta": 0.6}
    is_valid, msg = validate_garch_constraints(non_stationary)
    assert not is_valid
    assert "stationarity" in msg.lower()


def test_garch_fitting_respects_constraints():
    """Test that GARCH fitting produces valid parameters."""
    # Generate synthetic GARCH data
    np.random.seed(42)
    n = 1000
    omega_true = 0.01
    alpha_true = 0.15
    beta_true = 0.80

    # Simulate GARCH(1,1) process
    returns = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega_true / (1 - alpha_true - beta_true)

    for t in range(1, n):
        sigma2[t] = omega_true + alpha_true * returns[t-1]**2 + beta_true * sigma2[t-1]
        returns[t] = np.sqrt(sigma2[t]) * np.random.randn()

    returns_series = pd.Series(returns, index=pd.date_range("2020-01-01", periods=n))

    # Fit GARCH
    result = fit_garch11(returns_series, seed=42)

    # Check constraints
    assert result["omega"] > 0, "Omega must be positive"
    assert result["alpha"] >= 0, "Alpha must be non-negative"
    assert result["beta"] >= 0, "Beta must be non-negative"
    assert result["persistence"] < 1, "Persistence must be < 1"

    # Validate using constraint function
    params = {
        "omega": result["omega"],
        "alpha": result["alpha"],
        "beta": result["beta"],
    }
    is_valid, msg = validate_garch_constraints(params)
    assert is_valid, f"Fitted params invalid: {msg}"


def test_garch_rolling_forecast():
    """Test rolling GARCH forecast."""
    # Generate simple returns data
    np.random.seed(42)
    n = 600
    returns = np.random.randn(n) * 0.01
    returns_series = pd.Series(returns, index=pd.date_range("2020-01-01", periods=n))

    # Run rolling forecast
    result = rolling_garch_forecast(returns_series, window=100, seed=42)

    # Check output structure
    assert len(result.yhat) == len(returns_series)
    assert len(result.resid) == len(returns_series)
    assert "model" in result.meta
    assert result.meta["model"] == "GARCH(1,1)"

    # Check that forecasts are positive (variance)
    valid_forecasts = result.yhat.dropna()
    assert (valid_forecasts > 0).all(), "Variance forecasts must be positive"

    # Check that we have reasonable number of forecasts
    n_forecasts = result.meta["n_forecasts"]
    assert n_forecasts > 0
    assert n_forecasts <= n - 100  # Should not exceed available data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])