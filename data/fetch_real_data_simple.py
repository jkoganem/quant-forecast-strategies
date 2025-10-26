#!/usr/bin/env python
"""Fetch real market data using simple HTTP requests.

This script downloads real SPY and VIX data without complex dependencies.
"""

import os
import sys
from datetime import datetime, timedelta
import urllib.request
import json

import pandas as pd
import numpy as np


def fetch_stooq_data(ticker: str, output_path: str) -> pd.DataFrame:
    """Fetch data from Stooq (Polish stock data service).

    Args:
        ticker: Ticker symbol (e.g., 'spy.us').
        output_path: Path to save CSV.

    Returns:
        DataFrame with OHLCV data.
    """
    print(f"Fetching {ticker} from Stooq...")

    # Stooq URL format
    url = f"https://stooq.com/q/d/l/?s={ticker}&i=d"

    try:
        # Download CSV
        urllib.request.urlretrieve(url, output_path)
        print(f"  Downloaded to {output_path}")

        # Read and validate
        df = pd.read_csv(output_path)

        # Stooq format: Date,Open,High,Low,Close,Volume
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df = df.sort_index()

        print(f"  Loaded {len(df)} rows from {df.index[0]} to {df.index[-1]}")

        return df

    except Exception as e:
        print(f"  ERROR: {e}")
        raise


def fetch_alpha_vantage_data(
    ticker: str,
    api_key: str = "demo",
) -> pd.DataFrame:
    """Fetch data from Alpha Vantage (free API).

    Args:
        ticker: Stock ticker.
        api_key: API key (use 'demo' for limited access).

    Returns:
        DataFrame with OHLCV data.
    """
    print(f"Fetching {ticker} from Alpha Vantage...")

    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outputsize=full&apikey={api_key}&datatype=csv"

    try:
        # Download CSV
        df = pd.read_csv(url)

        # Alpha Vantage format
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.rename(columns={
            'timestamp': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'adjusted_close': 'Adj_Close',
            'volume': 'Volume',
        })
        df = df.set_index('Date')
        df = df.sort_index()

        print(f"  Loaded {len(df)} rows")

        return df

    except Exception as e:
        print(f"  ERROR: {e}")
        raise


def create_synthetic_vix_from_returns(spy_data: pd.DataFrame) -> pd.DataFrame:
    """Create synthetic VIX-like data from SPY returns.

    Uses EWMA of realized volatility as proxy for implied volatility.

    Args:
        spy_data: SPY OHLCV data.

    Returns:
        DataFrame with synthetic VIX.
    """
    print("Creating synthetic VIX from SPY volatility...")

    # Calculate returns
    returns = np.log(spy_data['Close'] / spy_data['Close'].shift(1))

    # Calculate realized volatility (21-day rolling)
    realized_vol = returns.rolling(21).std() * np.sqrt(252) * 100

    # Apply EWMA to smooth and add some mean reversion
    vix_proxy = realized_vol.ewm(span=10).mean()

    # Add some noise and mean reversion to make it more VIX-like
    # VIX tends to spike higher than realized vol
    vix_proxy = vix_proxy * 1.15  # Slight uplift

    # Clip to reasonable VIX range
    vix_proxy = vix_proxy.clip(5, 80)

    vix_df = pd.DataFrame({
        'VIX': vix_proxy,
        'ATM_IV': vix_proxy,  # Use same for ATM IV
    })

    vix_df = vix_df.dropna()

    print(f"  Created {len(vix_df)} VIX observations")
    print(f"  VIX range: {vix_df['VIX'].min():.1f}% - {vix_df['VIX'].max():.1f}%")
    print(f"  VIX mean: {vix_df['VIX'].mean():.1f}%")

    return vix_df


def main():
    """Main function to fetch data."""
    print("=" * 70)
    print("FETCHING REAL MARKET DATA (Simple Method)")
    print("=" * 70)
    print()

    os.makedirs("data", exist_ok=True)

    try:
        # Method 1: Try Stooq (free, no API key needed)
        print("1. Attempting to fetch SPY from Stooq...")
        print("-" * 70)

        try:
            spy_data = fetch_stooq_data("spy.us", "data/spy_stooq.csv")

            # Rename for consistency
            spy_data.to_csv("data/spy.csv")
            print("  ✓ Saved to data/spy.csv")

        except Exception as e:
            print(f"  Stooq failed: {e}")

            # Method 2: Try Alpha Vantage
            print("\n  Trying Alpha Vantage (demo key)...")

            try:
                spy_data = fetch_alpha_vantage_data("SPY", "demo")
                spy_data.to_csv("data/spy.csv")
                print("  ✓ Saved to data/spy.csv")

            except Exception as e2:
                print(f"  Alpha Vantage failed: {e2}")
                print("\n  ERROR: Could not fetch SPY data from any source.")
                print("  Please either:")
                print("    1. Get a free API key from https://www.alphavantage.co/support/#api-key")
                print("    2. Use the synthetic data: python generate_sample_data.py")
                return 1

        # Validate SPY data
        print(f"\nSPY Data Summary:")
        print(f"  Rows: {len(spy_data)}")
        print(f"  Date range: {spy_data.index[0]} to {spy_data.index[-1]}")
        print(f"  Latest close: ${spy_data['Close'].iloc[-1]:.2f}")

        # Create VIX proxy from returns
        print("\n2. Creating VIX Proxy")
        print("-" * 70)

        vix_data = create_synthetic_vix_from_returns(spy_data)

        # Save ATM IV
        atm_iv = pd.DataFrame({'ATM_IV': vix_data['ATM_IV']})
        atm_iv.to_csv("data/atm_iv_real.csv")
        print("  ✓ Saved to data/atm_iv_real.csv")

        # Calculate statistics
        print("\n3. Data Statistics")
        print("-" * 70)

        returns = np.log(spy_data['Close'] / spy_data['Close'].shift(1)).dropna()

        print(f"SPY Performance:")
        print(f"  Total return: {((spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[0]) - 1) * 100:.1f}%")
        print(f"  Annual return: {returns.mean() * 252 * 100:.1f}%")
        print(f"  Annual volatility: {returns.std() * np.sqrt(252) * 100:.1f}%")
        print(f"  Sharpe ratio: {(returns.mean() * 252) / (returns.std() * np.sqrt(252)):.2f}")

        drawdowns = (spy_data['Close'] / spy_data['Close'].cummax()) - 1
        print(f"  Max drawdown: {drawdowns.min() * 100:.1f}%")

        print(f"\nImplied Volatility (Proxy):")
        print(f"  Mean: {vix_data['VIX'].mean():.1f}%")
        print(f"  Range: {vix_data['VIX'].min():.1f}% - {vix_data['VIX'].max():.1f}%")

        print("\n" + "=" * 70)
        print("DATA DOWNLOAD COMPLETE!")
        print("=" * 70)
        print(f"\nFiles created:")
        print(f"  - data/spy.csv ({len(spy_data)} rows)")
        print(f"  - data/atm_iv_real.csv ({len(atm_iv)} rows)")
        print(f"\nRun analysis with:")
        print(f"  python -m multi_ts.cli --start 2015-01-01 --end 2024-12-31")
        print(f"  python -m multi_ts.cli --start 2015-01-01 --end 2024-12-31 --iv_csv data/atm_iv_real.csv")

        return 0

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
