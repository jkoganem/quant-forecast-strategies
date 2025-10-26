"""Test IV feature joins and VRP computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tempfile
import os

from multi_ts.features.iv_features import join_atm_iv, variance_risk_premium


@pytest.fixture
def sample_returns():
    """Generate sample returns data."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    returns = pd.DataFrame({
        'date': dates,
        'returns': np.random.randn(n) * 0.01
    })
    return returns


@pytest.fixture
def sample_iv_csv():
    """Create a temporary IV CSV file."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    iv_data = pd.DataFrame({
        'date': dates,
        'atm_iv': np.random.uniform(0.15, 0.35, n)
    })

    # Write to temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
    iv_data.to_csv(temp_file.name, index=False)
    temp_file.close()

    yield temp_file.name

    # Cleanup
    os.unlink(temp_file.name)


def test_join_atm_iv_success(sample_returns, sample_iv_csv):
    """Test that join_atm_iv successfully joins IV data to returns."""
    result = join_atm_iv(sample_iv_csv, sample_returns)

    # Check that result has both returns and IV columns
    assert 'returns' in result.columns
    assert 'atm_iv' in result.columns

    # Check that no extra rows were added
    assert len(result) == len(sample_returns)

    # Check that IV values are reasonable
    assert (result['atm_iv'] >= 0.10).all()
    assert (result['atm_iv'] <= 0.50).all()


def test_join_atm_iv_alignment(sample_returns, sample_iv_csv):
    """Test that dates are properly aligned after join."""
    result = join_atm_iv(sample_iv_csv, sample_returns)

    # Original dates should be preserved
    assert result['date'].equals(sample_returns['date'])


def test_join_atm_iv_missing_dates(sample_returns):
    """Test join behavior when some dates are missing in IV data."""
    np.random.seed(42)

    # Create IV data with gaps (only every other day)
    dates = sample_returns['date'].iloc[::2]
    iv_data = pd.DataFrame({
        'date': dates,
        'atm_iv': np.random.uniform(0.15, 0.35, len(dates))
    })

    # Write to temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
    iv_data.to_csv(temp_file.name, index=False)
    temp_file.close()

    try:
        result = join_atm_iv(temp_file.name, sample_returns)

        # Result should have some NaN values for missing IV dates
        assert result['atm_iv'].isna().any()

        # But returns should still be present
        assert not result['returns'].isna().any()
    finally:
        os.unlink(temp_file.name)


def test_variance_risk_premium_calculation():
    """Test VRP calculation: IV^2 - RV."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range('2020-01-01', periods=n, freq='D')

    # Create IV and realized variance series
    iv_series = pd.Series(np.random.uniform(0.15, 0.30, n), index=dates)
    realized_var = pd.Series(np.random.uniform(0.01, 0.04, n), index=dates)

    vrp = variance_risk_premium(iv_series, realized_var)

    # VRP should be IV^2 - RV
    expected_vrp = iv_series ** 2 - realized_var

    pd.testing.assert_series_equal(vrp, expected_vrp)


def test_vrp_sign_interpretation():
    """Test VRP interpretation: positive when IV > realized vol."""
    dates = pd.date_range('2020-01-01', periods=10, freq='D')

    # High IV, low realized volatility -> positive VRP
    iv_series = pd.Series(np.ones(10) * 0.30, index=dates)
    realized_var = pd.Series(np.ones(10) * 0.02, index=dates)

    vrp = variance_risk_premium(iv_series, realized_var)

    # VRP should be positive (IV^2 = 0.09, RV = 0.02, VRP = 0.07)
    assert (vrp > 0).all()


def test_vrp_negative_when_vol_spike():
    """Test that VRP can be negative during volatility spikes."""
    dates = pd.date_range('2020-01-01', periods=10, freq='D')

    # Low IV, high realized volatility -> negative VRP
    iv_series = pd.Series(np.ones(10) * 0.15, index=dates)  # IV = 15%
    realized_var = pd.Series(np.ones(10) * 0.05, index=dates)  # RV = 5% (vol spike)

    vrp = variance_risk_premium(iv_series, realized_var)

    # VRP should be negative (IV^2 = 0.0225, RV = 0.05, VRP = -0.0275)
    assert (vrp < 0).all()


def test_join_preserves_data_types(sample_returns, sample_iv_csv):
    """Test that join preserves numeric data types."""
    result = join_atm_iv(sample_iv_csv, sample_returns)

    # Check data types
    assert pd.api.types.is_numeric_dtype(result['returns'])
    assert pd.api.types.is_numeric_dtype(result['atm_iv'])


def test_iv_features_no_lookahead_bias(sample_returns):
    """Test that IV features don't introduce lookahead bias."""
    np.random.seed(42)

    # Create IV data where IV at time t should align with returns at time t
    dates = sample_returns['date']
    iv_data = pd.DataFrame({
        'date': dates,
        'atm_iv': np.random.uniform(0.15, 0.35, len(dates))
    })

    # Write to temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
    iv_data.to_csv(temp_file.name, index=False)
    temp_file.close()

    try:
        result = join_atm_iv(temp_file.name, sample_returns)

        # For each date, IV should be from the same date or earlier
        # (Not from the future)
        for i in range(len(result)):
            date = result.iloc[i]['date']
            iv_value = result.iloc[i]['atm_iv']

            # IV at this date should match the IV data for this date
            matching_iv = iv_data[iv_data['date'] == date]['atm_iv'].values
            if len(matching_iv) > 0:
                assert iv_value == matching_iv[0]
    finally:
        os.unlink(temp_file.name)


def test_vrp_with_nan_handling():
    """Test VRP calculation handles NaN values."""
    dates = pd.date_range('2020-01-01', periods=10, freq='D')

    iv_series = pd.Series([0.2, 0.21, np.nan, 0.22, 0.23, np.nan, 0.24, 0.25, 0.26, 0.27], index=dates)
    realized_var = pd.Series([0.02, 0.021, 0.022, np.nan, 0.023, 0.024, np.nan, 0.025, 0.026, 0.027], index=dates)

    vrp = variance_risk_premium(iv_series, realized_var)

    # VRP should have NaN where either input has NaN
    assert vrp.isna().sum() >= 3  # At least 3 NaN values from inputs


def test_empty_iv_csv_handling(sample_returns):
    """Test behavior with empty IV CSV."""
    # Create empty CSV
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
    temp_file.write('date,atm_iv\n')  # Header only
    temp_file.close()

    try:
        result = join_atm_iv(temp_file.name, sample_returns)

        # Should return returns with NaN for IV
        assert len(result) == len(sample_returns)
        assert result['atm_iv'].isna().all()
    finally:
        os.unlink(temp_file.name)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
