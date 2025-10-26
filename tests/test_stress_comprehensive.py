"""Comprehensive stress tests to catch bugs introduced by code reorganization.

This test suite specifically targets bugs that can be introduced when code
is refactored or reorganized, including:
- Index alignment issues
- NaN propagation
- Incorrect calculations on sliced data
- Mismatched date ranges
- Off-by-one errors
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multi_ts.strategy import execute_strategy, run_multiple_strategies
from multi_ts.evaluation import evaluate_strategy, evaluate_forecast
from multi_ts.datatypes import TSConfig, ForecastResult


class TestStrategyAlignment:
    """Test that strategies properly align data and handle date ranges."""

    def test_benchmark_alignment(self):
        """Test that benchmark returns are correctly aligned with strategy returns."""
        # Create test data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
        variance_forecast = pd.Series(np.ones(100) * 0.0001, index=dates)

        config = TSConfig(
            start_date='2023-01-01',
            end_date='2023-04-10',
            target_vol_ann=0.10,
            w_max=2.0,
            tc_bps=1.0,
            window_init_days=20,
            out_dir='outputs',
            plots=False,
            json_bundle=False,
            seed=42,
            iv_csv=None,
        )

        result = execute_strategy(returns, variance_forecast, None, config, use_directional=False)

        # Check that benchmark has same length as strategy
        assert len(result['cum_benchmark']) == len(result['cum_net']), \
            "Benchmark and strategy cumulative returns must have same length"

        # Check that indices match
        assert (result['cum_benchmark'].index == result['cum_net'].index).all(), \
            "Benchmark and strategy must have matching indices"

        # Check no NaN values in benchmark
        assert not result['cum_benchmark'].isna().any(), \
            "Benchmark should not contain NaN values"

    def test_costs_alignment(self):
        """Test that transaction costs align with returns."""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        returns = pd.Series(np.random.randn(50) * 0.01, index=dates)
        variance_forecast = pd.Series(np.ones(50) * 0.0001, index=dates)

        config = TSConfig(
            start_date='2023-01-01',
            end_date='2023-02-19',
            target_vol_ann=0.10,
            w_max=2.0,
            tc_bps=10.0,  # High costs to make them visible
            window_init_days=10,
            out_dir='outputs',
            plots=False,
            json_bundle=False,
            seed=42,
            iv_csv=None,
        )

        result = execute_strategy(returns, variance_forecast, None, config, use_directional=False)

        # Net returns should equal gross returns + costs
        diff = result['net_returns'] - (result['gross_returns'] + result['costs'])
        assert np.abs(diff).max() < 1e-10, \
            "Net returns must equal gross returns + costs"

        # Costs should be negative or zero
        assert (result['costs'] <= 0).all(), \
            "Transaction costs should be negative or zero"


class TestDrawdownCalculations:
    """Test that drawdown calculations are correct."""

    def test_drawdown_formula(self):
        """Test drawdown calculation matches mathematical definition."""
        # Create known cumulative returns with sufficient data points (>= 20)
        values = [1.0, 1.1, 1.2, 1.15, 1.25, 1.20, 1.30, 1.28, 1.35, 1.33,
                  1.40, 1.38, 1.45, 1.43, 1.50, 1.48, 1.55, 1.53, 1.60, 1.58,
                  1.65, 1.63, 1.70, 1.68, 1.75]
        cum_returns = pd.Series(values)

        # Calculate drawdown manually
        running_max = cum_returns.cummax()
        expected_dd = ((cum_returns - running_max) / running_max * 100)

        # Test that our calculation matches
        from multi_ts.evaluation import evaluate_strategy

        # Create mock strategy data
        returns = cum_returns.pct_change().fillna(0)
        weights = pd.Series(1.0, index=returns.index)
        costs = pd.Series(0.0, index=returns.index)

        result = evaluate_strategy(returns, weights, costs, "test")

        # Max drawdown at index 3 and 5
        assert expected_dd[3] == pytest.approx(-4.1667, rel=1e-3), \
            "Drawdown at index 3 should be -4.17%"
        assert expected_dd[5] == pytest.approx(-4.0, rel=1e-3), \
            "Drawdown at index 5 should be -4.0%"

        # Max drawdown should be minimum of drawdown series
        assert result['max_drawdown'] == pytest.approx(expected_dd.min(), rel=1e-3), \
            "Max drawdown should match minimum of drawdown series"

    def test_no_drawdown_on_monotonic_growth(self):
        """Test that monotonically increasing returns have zero drawdown."""
        # Create sufficient data points (>= 20)
        values = [1.0 + i * 0.05 for i in range(25)]
        cum_returns = pd.Series(values)
        returns = cum_returns.pct_change().fillna(0)
        weights = pd.Series(1.0, index=returns.index)
        costs = pd.Series(0.0, index=returns.index)

        result = evaluate_strategy(returns, weights, costs, "test")

        assert result['max_drawdown'] == pytest.approx(0.0, abs=1e-6), \
            "Monotonically increasing returns should have zero drawdown"


class TestRollingCalculations:
    """Test that rolling calculations handle slicing correctly."""

    def test_realized_variance_on_sliced_data(self):
        """Test that realized variance is calculated on full data before slicing."""
        # Create returns with known variance structure
        np.random.seed(42)
        n = 300
        dates = pd.date_range('2023-01-01', periods=n, freq='D')

        # Create returns with increasing volatility
        vol_schedule = np.linspace(0.01, 0.03, n)
        returns = pd.Series(np.random.randn(n) * vol_schedule, index=dates)

        # Calculate realized variance on full series
        realized_var_full = returns.rolling(20).var()

        # Slice to last 100 days
        realized_var_sliced = realized_var_full.iloc[-100:]

        # The sliced version should have no NaN if calculated on full data first
        assert realized_var_sliced.isna().sum() == 0, \
            "Realized variance sliced from full calculation should have no NaN"

        # Compare to incorrect method (calculate after slicing)
        returns_sliced = returns.iloc[-100:]
        realized_var_wrong = returns_sliced.rolling(20).var()

        # The incorrect method will have NaN at the start
        assert realized_var_wrong.isna().sum() == 19, \
            "Realized variance calculated on sliced data should have 19 NaN"

    def test_rolling_window_requires_sufficient_data(self):
        """Test that rolling calculations fail gracefully with insufficient data."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        returns = pd.Series(np.random.randn(10) * 0.01, index=dates)

        # Rolling window of 20 on 10 observations
        rolling_var = returns.rolling(20).var()

        # Should have all NaN
        assert rolling_var.isna().all(), \
            "Rolling window larger than data should produce all NaN"


class TestForecastAlignment:
    """Test that forecasts align properly with actuals."""

    def test_forecast_evaluation_alignment(self):
        """Test that forecast evaluation handles index alignment."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        actual = pd.Series(np.random.randn(100) * 0.01, index=dates)

        # Forecast with partial overlap
        forecast_dates = dates[10:90]
        forecast = pd.Series(np.random.randn(80) * 0.01, index=forecast_dates)

        result = evaluate_forecast(actual, forecast, "test_model")

        # Should evaluate on the overlapping region only
        assert result['n_obs'] == 80, \
            "Should evaluate on 80 overlapping observations"

        # MSE should be computable
        assert not np.isnan(result['mse']), \
            "MSE should be computable on aligned data"

    def test_forecast_with_no_overlap(self):
        """Test forecast evaluation when there's no overlap."""
        dates1 = pd.date_range('2023-01-01', periods=50, freq='D')
        dates2 = pd.date_range('2023-03-01', periods=50, freq='D')

        actual = pd.Series(np.random.randn(50) * 0.01, index=dates1)
        forecast = pd.Series(np.random.randn(50) * 0.01, index=dates2)

        result = evaluate_forecast(actual, forecast, "test_model")

        # Should return NaN metrics
        assert np.isnan(result['mse']), \
            "MSE should be NaN when there's no overlap"
        assert result['n_obs'] == 0, \
            "Should have 0 observations when there's no overlap"


class TestNaNPropagation:
    """Test that NaN values are handled correctly and don't propagate unexpectedly."""

    def test_nan_in_returns(self):
        """Test that NaN in returns is handled gracefully."""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        returns = pd.Series(np.random.randn(50) * 0.01, index=dates)

        # Introduce NaN
        returns.iloc[25] = np.nan

        variance_forecast = pd.Series(np.ones(50) * 0.0001, index=dates)

        config = TSConfig(
            start_date='2023-01-01',
            end_date='2023-02-19',
            target_vol_ann=0.10,
            w_max=2.0,
            tc_bps=1.0,
            window_init_days=10,
            out_dir='outputs',
            plots=False,
            json_bundle=False,
            seed=42,
            iv_csv=None,
        )

        result = execute_strategy(returns, variance_forecast, None, config, use_directional=False)

        # Check that NaN is handled (filled with 0 in strategy)
        assert not result['gross_returns'].isna().all(), \
            "Strategy should handle NaN in returns"

    def test_nan_in_variance_forecast(self):
        """Test that NaN in variance forecast is handled."""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        returns = pd.Series(np.random.randn(50) * 0.01, index=dates)
        variance_forecast = pd.Series(np.ones(50) * 0.0001, index=dates)

        # Introduce NaN
        variance_forecast.iloc[20:25] = np.nan

        config = TSConfig(
            start_date='2023-01-01',
            end_date='2023-02-19',
            target_vol_ann=0.10,
            w_max=2.0,
            tc_bps=1.0,
            window_init_days=10,
            out_dir='outputs',
            plots=False,
            json_bundle=False,
            seed=42,
            iv_csv=None,
        )

        result = execute_strategy(returns, variance_forecast, None, config, use_directional=False)

        # Weights should handle NaN (fillna with 1.0)
        assert not result['weights'].isna().all(), \
            "Weights should handle NaN in variance forecast"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_observation(self):
        """Test that single observation doesn't crash."""
        dates = pd.date_range('2023-01-01', periods=1, freq='D')
        returns = pd.Series([0.01], index=dates)
        weights = pd.Series([1.0], index=dates)
        costs = pd.Series([0.0], index=dates)

        result = evaluate_strategy(returns, weights, costs, "test")

        # Should return NaN for most metrics due to insufficient data
        assert np.isnan(result['sharpe_ratio']), \
            "Sharpe ratio should be NaN for single observation"

    def test_zero_volatility(self):
        """Test that zero volatility is handled."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        returns = pd.Series(np.zeros(100), index=dates)  # Zero returns
        weights = pd.Series(1.0, index=dates)
        costs = pd.Series(0.0, index=dates)

        result = evaluate_strategy(returns, weights, costs, "test")

        # Sharpe should be NaN (0 vol)
        assert np.isnan(result['sharpe_ratio']) or result['sharpe_ratio'] == 0, \
            "Sharpe ratio should be NaN or 0 for zero volatility"

    def test_all_losses(self):
        """Test strategy with all negative returns."""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        # Ensure all returns are negative by using absolute value
        returns = pd.Series(-np.abs(np.random.randn(50)) * 0.01, index=dates)
        weights = pd.Series(1.0, index=dates)
        costs = pd.Series(0.0, index=dates)

        result = evaluate_strategy(returns, weights, costs, "test")

        # Win rate should be 0
        assert result['win_rate'] == 0, \
            "Win rate should be 0 for all losses"

        # Total return should be negative
        assert result['total_return'] < 0, \
            "Total return should be negative for all losses"


class TestCumulativeReturnsRebasing:
    """Test that cumulative returns are rebased correctly for plotting."""

    def test_rebase_to_start_at_one(self):
        """Test rebasing cumulative returns to start at 1.0."""
        cum_returns = pd.Series([1.5, 1.6, 1.7, 1.65, 1.75])

        # Rebase to start at 1.0
        rebased = cum_returns / cum_returns.iloc[0]

        assert rebased.iloc[0] == pytest.approx(1.0, abs=1e-6), \
            "Rebased series should start at 1.0"

        # Relative changes should be preserved
        original_change = (cum_returns.iloc[-1] - cum_returns.iloc[0]) / cum_returns.iloc[0]
        rebased_change = rebased.iloc[-1] - 1.0

        assert original_change == pytest.approx(rebased_change, rel=1e-6), \
            "Relative changes should be preserved after rebasing"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
