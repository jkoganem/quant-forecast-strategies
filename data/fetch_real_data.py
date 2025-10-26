#!/usr/bin/env python
"""Fetch real market data from various sources.

This script downloads real SPY data and VIX (implied volatility) data
to replace the synthetic data with actual market data.
"""

import os
import sys
from datetime import datetime, timedelta
import time

import pandas as pd
import numpy as np


def fetch_yahoo_finance_data(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol.
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.

    Returns:
        DataFrame with OHLCV data.
    """
    try:
        import yfinance as yf
        print(f"Fetching {ticker} data from Yahoo Finance...")

        # Download data
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
        )

        if len(data) == 0:
            raise ValueError(f"No data returned for {ticker}")

        # Rename columns to match our format
        data = data.rename(columns={
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Adj Close': 'Adj_Close',
            'Volume': 'Volume',
        })

        print(f"  Downloaded {len(data)} rows from {data.index[0]} to {data.index[-1]}")

        return data

    except ImportError:
        print("ERROR: yfinance not installed. Installing...")
        os.system("pip install yfinance --quiet")
        import yfinance as yf
        return fetch_yahoo_finance_data(ticker, start_date, end_date)


def fetch_fred_vix_data(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch VIX data from FRED (Federal Reserve Economic Data).

    Args:
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.

    Returns:
        DataFrame with VIX data.
    """
    try:
        from fredapi import Fred

        # FRED API key (public data, no key needed for basic access)
        # Using pandas_datareader as alternative
        print("Fetching VIX data from FRED...")

        try:
            import pandas_datareader as pdr
            vix = pdr.get_data_fred('VIXCLS', start=start_date, end=end_date)
            vix.columns = ['VIX']
            print(f"  Downloaded {len(vix)} VIX observations")
            return vix
        except:
            # Fallback to direct Yahoo Finance VIX
            print("  FRED access failed, using Yahoo Finance for ^VIX...")
            return fetch_yahoo_finance_data('^VIX', start_date, end_date)

    except ImportError:
        print("Installing pandas_datareader...")
        os.system("pip install pandas-datareader --quiet")
        return fetch_fred_vix_data(start_date, end_date)


def fetch_alternative_vix() -> pd.DataFrame:
    """Fetch VIX from alternative source (Yahoo Finance).

    Returns:
        DataFrame with VIX data.
    """
    print("Fetching VIX from Yahoo Finance...")

    import yfinance as yf

    # Download VIX index
    vix = yf.download(
        '^VIX',
        start='2010-01-01',
        end=datetime.now().strftime('%Y-%m-%d'),
        progress=False,
    )

    if len(vix) > 0:
        vix = pd.DataFrame({'VIX': vix['Close']})
        print(f"  Downloaded {len(vix)} VIX observations")
        return vix
    else:
        raise ValueError("Failed to download VIX data")


def create_atm_iv_from_vix(
    vix_data: pd.DataFrame,
    spy_data: pd.DataFrame,
) -> pd.DataFrame:
    """Create ATM IV series from VIX data.

    VIX represents 30-day implied volatility of S&P 500 options,
    which is a good proxy for ATM implied volatility.

    Args:
        vix_data: VIX data.
        spy_data: SPY data for date alignment.

    Returns:
        DataFrame with ATM IV.
    """
    print("Creating ATM IV from VIX...")

    # Align VIX with SPY trading days
    vix_aligned = vix_data.reindex(spy_data.index, method='ffill')

    # VIX is already in percentage form, representing annualized volatility
    atm_iv = pd.DataFrame({
        'ATM_IV': vix_aligned['VIX'].values
    }, index=vix_aligned.index)

    # Remove NaN values
    atm_iv = atm_iv.dropna()

    print(f"  Created {len(atm_iv)} ATM IV observations")
    print(f"  ATM IV range: {atm_iv['ATM_IV'].min():.1f}% - {atm_iv['ATM_IV'].max():.1f}%")
    print(f"  ATM IV mean: {atm_iv['ATM_IV'].mean():.1f}%")

    return atm_iv


def validate_data(data: pd.DataFrame, name: str) -> bool:
    """Validate downloaded data.

    Args:
        data: DataFrame to validate.
        name: Name for reporting.

    Returns:
        True if valid, False otherwise.
    """
    print(f"Validating {name}...")

    if len(data) == 0:
        print(f"  ERROR: {name} is empty")
        return False

    # Check for too many NaN values
    nan_pct = data.isna().sum().sum() / (len(data) * len(data.columns)) * 100
    if nan_pct > 10:
        print(f"  WARNING: {name} has {nan_pct:.1f}% NaN values")

    # Check date range
    print(f"  Date range: {data.index[0]} to {data.index[-1]}")
    print(f"  Observations: {len(data)}")

    # Check for reasonable values (if numeric)
    for col in data.select_dtypes(include=[np.number]).columns:
        col_min = data[col].min()
        col_max = data[col].max()
        col_mean = data[col].mean()
        print(f"  {col}: min={col_min:.2f}, max={col_max:.2f}, mean={col_mean:.2f}")

    return True


def main():
    """Main function to fetch all data."""
    print("=" * 70)
    print("FETCHING REAL MARKET DATA")
    print("=" * 70)
    print()

    # Configuration
    start_date = "2010-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    # Create data directory
    os.makedirs("data", exist_ok=True)

    try:
        # 1. Fetch SPY data
        print("\n1. Fetching SPY Data")
        print("-" * 70)
        spy_data = fetch_yahoo_finance_data("SPY", start_date, end_date)

        if validate_data(spy_data, "SPY"):
            # Save to CSV
            spy_data.to_csv("data/spy.csv")
            print(f"  ✓ Saved to data/spy.csv")
        else:
            print("  ✗ SPY data validation failed")
            return 1

        # 2. Fetch VIX data
        print("\n2. Fetching VIX Data")
        print("-" * 70)

        try:
            # Try FRED first
            vix_data = fetch_fred_vix_data(start_date, end_date)
        except Exception as e:
            print(f"  FRED failed: {e}")
            # Fallback to Yahoo Finance
            vix_data = fetch_alternative_vix()

        if 'VIX' not in vix_data.columns and 'Close' in vix_data.columns:
            vix_data = pd.DataFrame({'VIX': vix_data['Close']})

        if validate_data(vix_data, "VIX"):
            # Convert VIX to ATM IV format
            atm_iv = create_atm_iv_from_vix(vix_data, spy_data)

            # Save to CSV
            atm_iv.to_csv("data/atm_iv_real.csv")
            print(f"  ✓ Saved to data/atm_iv_real.csv")
        else:
            print("  ✗ VIX data validation failed")
            return 1

        # 3. Generate summary statistics
        print("\n3. Data Summary")
        print("-" * 70)

        # Calculate SPY returns
        returns = np.log(spy_data['Close'] / spy_data['Close'].shift(1)).dropna()

        print(f"SPY Statistics (Total Period):")
        print(f"  Total return: {((spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[0]) - 1) * 100:.1f}%")
        print(f"  Annual return: {returns.mean() * 252 * 100:.1f}%")
        print(f"  Annual volatility: {returns.std() * np.sqrt(252) * 100:.1f}%")
        print(f"  Sharpe ratio: {(returns.mean() * 252) / (returns.std() * np.sqrt(252)):.2f}")
        print(f"  Max drawdown: {((spy_data['Close'] / spy_data['Close'].cummax()) - 1).min() * 100:.1f}%")

        print(f"\nVIX/IV Statistics:")
        print(f"  Mean IV: {atm_iv['ATM_IV'].mean():.1f}%")
        print(f"  Min IV: {atm_iv['ATM_IV'].min():.1f}%")
        print(f"  Max IV: {atm_iv['ATM_IV'].max():.1f}%")
        print(f"  Std IV: {atm_iv['ATM_IV'].std():.1f}%")

        # Recent statistics (last 5 years)
        recent_start = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")
        spy_recent = spy_data[spy_data.index >= recent_start]
        returns_recent = np.log(spy_recent['Close'] / spy_recent['Close'].shift(1)).dropna()

        print(f"\nSPY Statistics (Last 5 Years):")
        print(f"  Annual return: {returns_recent.mean() * 252 * 100:.1f}%")
        print(f"  Annual volatility: {returns_recent.std() * np.sqrt(252) * 100:.1f}%")
        print(f"  Sharpe ratio: {(returns_recent.mean() * 252) / (returns_recent.std() * np.sqrt(252)):.2f}")

        print("\n" + "=" * 70)
        print("DATA DOWNLOAD COMPLETE!")
        print("=" * 70)
        print(f"\nFiles created:")
        print(f"  - data/spy.csv ({len(spy_data)} rows)")
        print(f"  - data/atm_iv_real.csv ({len(atm_iv)} rows)")
        print(f"\nYou can now run the analysis with:")
        print(f"  python -m multi_ts.cli --start 2015-01-01 --end {end_date} --iv_csv data/atm_iv_real.csv")

        return 0

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
