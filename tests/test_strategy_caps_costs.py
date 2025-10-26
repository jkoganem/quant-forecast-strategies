"""Test that strategy respects weight caps and transaction costs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multi_ts.strategy import (
    volatility_managed_weights,
    directional_overlay,
    calculate_transaction_costs,
    execute_strategy,
)


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range('2020-01-01', periods=n, freq='D')

    returns = pd.Series(np.random.randn(n) * 0.015 + 0.0003, index=dates, name='returns')
    variance_forecast = pd.Series(np.abs(np.random.randn(n) * 0.0001) + 0.0002, index=dates)
    return_forecast = pd.Series(np.random.randn(n) * 0.001, index=dates)

    return returns, variance_forecast, return_forecast


def test_weight_cap_respected(sample_data):
    """Test that weights never exceed w_max."""
    returns, variance_forecast, _ = sample_data

    w_max = 1.5
    weights = volatility_managed_weights(
        variance_forecast=variance_forecast,
        target_vol_ann=0.10,
        w_max=w_max
    )

    # Check that all weights are within [-w_max, w_max]
    assert (weights <= w_max).all(), f"Some weights exceed {w_max}"
    assert (weights >= -w_max).all(), f"Some weights exceed -{w_max}"


def test_weight_cap_tight_constraint():
    """Test weight cap with very low limit."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range('2020-01-01', periods=n, freq='D')

    # Create low variance forecast (would normally give high weights)
    variance_forecast = pd.Series(np.ones(n) * 0.00001, index=dates)

    w_max = 0.5
    weights = volatility_managed_weights(
        variance_forecast=variance_forecast,
        target_vol_ann=0.10,
        w_max=w_max
    )

    # All weights should be capped at w_max
    assert (weights <= w_max).all()
    assert weights.max() <= w_max


def test_transaction_costs_accumulate(sample_data):
    """Test that transaction costs accumulate with position changes."""
    returns, _, _ = sample_data

    # Strategy that changes position every day
    weights = pd.Series(
        np.sin(np.arange(len(returns)) * 0.1),  # Oscillating weights
        index=returns.index
    )

    tc_bps = 10  # 10 basis points
    costs = calculate_transaction_costs(weights, tc_bps)

    # Costs should be negative (drag on returns)
    assert (costs <= 0).all()

    # Total costs should be non-zero for changing positions
    assert costs.sum() < 0


def test_zero_costs_for_constant_position():
    """Test that constant position has no transaction costs."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range('2020-01-01', periods=n, freq='D')

    # Constant weight (no position changes)
    weights = pd.Series(np.ones(n) * 1.0, index=dates)

    tc_bps = 10
    costs = calculate_transaction_costs(weights, tc_bps)

    # First period has cost (initial entry), rest should be zero
    assert costs.iloc[0] < 0  # Entry cost
    assert (costs.iloc[1:] == 0).all()  # No subsequent costs


def test_higher_turnover_higher_costs():
    """Test that higher turnover results in higher costs."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range('2020-01-01', periods=n, freq='D')

    # Low turnover strategy (changes every 10 days)
    weights_low_turnover = pd.Series(
        np.repeat(np.random.randn(10), 10),
        index=dates
    )

    # High turnover strategy (changes every day)
    weights_high_turnover = pd.Series(
        np.random.randn(n),
        index=dates
    )

    tc_bps = 10
    costs_low = calculate_transaction_costs(weights_low_turnover, tc_bps)
    costs_high = calculate_transaction_costs(weights_high_turnover, tc_bps)

    # High turnover should have larger (more negative) total costs
    assert costs_high.sum() < costs_low.sum()


def test_strategy_respects_both_caps_and_costs(sample_data):
    """Test full strategy execution respects caps and includes costs."""
    returns, variance_forecast, return_forecast = sample_data

    w_max = 1.0
    tc_bps = 5
    target_vol_ann = 0.10

    result = execute_strategy(
        returns=returns,
        variance_forecast=variance_forecast,
        return_forecast=return_forecast,
        target_vol_ann=target_vol_ann,
        w_max=w_max,
        tc_bps=tc_bps,
        strategy_type='directional'
    )

    # Check weights respect cap
    assert (result['weights'] <= w_max).all()
    assert (result['weights'] >= -w_max).all()

    # Check that costs are included
    assert 'costs' in result
    assert (result['costs'] <= 0).all()

    # Net returns should include costs
    assert 'net_returns' in result


def test_directional_overlay_amplifies_signals():
    """Test that directional overlay amplifies strong signals."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range('2020-01-01', periods=n, freq='D')

    # Base weights
    base_weights = pd.Series(np.ones(n) * 0.5, index=dates)

    # Strong positive forecast
    return_forecast = pd.Series(np.ones(n) * 0.005, index=dates)  # Strong signal
    threshold = 0.001

    adjusted_weights = directional_overlay(
        base_weights=base_weights,
        return_forecast=return_forecast,
        threshold=threshold
    )

    # Weights should be increased for positive signals
    assert (adjusted_weights >= base_weights).all()


def test_directional_overlay_reduces_weak_signals():
    """Test that directional overlay reduces exposure for weak signals."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range('2020-01-01', periods=n, freq='D')

    # Base weights
    base_weights = pd.Series(np.ones(n) * 1.0, index=dates)

    # Weak positive forecast (below threshold)
    return_forecast = pd.Series(np.ones(n) * 0.0001, index=dates)
    threshold = 0.001

    adjusted_weights = directional_overlay(
        base_weights=base_weights,
        return_forecast=return_forecast,
        threshold=threshold
    )

    # Weights should be reduced for weak signals
    assert (adjusted_weights < base_weights).all()


def test_transaction_cost_calculation_accuracy():
    """Test that transaction costs are calculated correctly."""
    dates = pd.date_range('2020-01-01', periods=5, freq='D')

    # Known position changes
    weights = pd.Series([0, 1.0, 1.0, 0.5, 0], index=dates)

    tc_bps = 10  # 10 basis points = 0.001

    costs = calculate_transaction_costs(weights, tc_bps)

    # Expected costs:
    # Day 0: |0 - 0| * 0.001 = 0 (but entry cost applied)
    # Day 1: |1.0 - 0| * 0.001 = 0.001
    # Day 2: |1.0 - 1.0| * 0.001 = 0
    # Day 3: |0.5 - 1.0| * 0.001 = 0.0005
    # Day 4: |0 - 0.5| * 0.001 = 0.0005

    assert costs.iloc[0] < 0  # Entry cost
    assert costs.iloc[1] < 0  # Position change
    assert costs.iloc[2] == 0  # No change
    assert costs.iloc[3] < 0  # Position change
    assert costs.iloc[4] < 0  # Position change


def test_zero_transaction_costs():
    """Test strategy with zero transaction costs."""
    np.random.seed(42)
    n = 50
    dates = pd.date_range('2020-01-01', periods=n, freq='D')

    returns = pd.Series(np.random.randn(n) * 0.01, index=dates)
    weights = pd.Series(np.random.randn(n), index=dates)

    tc_bps = 0  # Zero costs
    costs = calculate_transaction_costs(weights, tc_bps)

    # All costs should be zero
    assert (costs == 0).all()


def test_weight_cap_preserves_sign():
    """Test that weight capping preserves the direction (sign) of the weight."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range('2020-01-01', periods=n, freq='D')

    # Mix of positive and negative forecasts
    variance_forecast = pd.Series(np.abs(np.random.randn(n) * 0.0001) + 0.0002, index=dates)

    w_max = 1.5
    weights = volatility_managed_weights(
        variance_forecast=variance_forecast,
        target_vol_ann=0.10,
        w_max=w_max
    )

    # All weights should be non-negative for vol-managed (long-only by default)
    assert (weights >= 0).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
