"""Generate sample data for testing the system.

Creates synthetic market data and implied volatility data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_synthetic_ohlcv(
    start_date: str = "2015-01-01",
    end_date: str = "2024-12-31",
    initial_price: float = 100.0,
    volatility: float = 0.20,
    drift: float = 0.08,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data.

    Args:
        start_date: Start date.
        end_date: End date.
        initial_price: Initial price level.
        volatility: Annual volatility.
        drift: Annual drift (expected return).
        seed: Random seed.

    Returns:
        DataFrame with OHLCV data.
    """
    np.random.seed(seed)

    # Create date range (business days only)
    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)

    # Daily parameters
    daily_drift = drift / 252
    daily_vol = volatility / np.sqrt(252)

    # Generate returns with GARCH-like volatility clustering
    returns = np.zeros(n_days)
    conditional_vol = np.zeros(n_days)
    conditional_vol[0] = daily_vol

    # GARCH(1,1) parameters for volatility clustering
    omega = daily_vol**2 * 0.05
    alpha = 0.1
    beta = 0.85

    for t in range(1, n_days):
        # Update conditional volatility
        conditional_vol[t] = np.sqrt(
            omega + alpha * returns[t-1]**2 + beta * conditional_vol[t-1]**2
        )

        # Generate return
        returns[t] = daily_drift + conditional_vol[t] * np.random.randn()

    # Generate prices
    prices = initial_price * np.exp(np.cumsum(returns))

    # Generate OHLC from close prices
    data = []
    for i, (date, close_price) in enumerate(zip(dates, prices)):
        # Add intraday noise
        daily_range = close_price * daily_vol * np.abs(np.random.randn()) * 0.5

        high = close_price + daily_range * np.random.uniform(0, 1)
        low = close_price - daily_range * np.random.uniform(0, 1)
        open_price = np.random.uniform(low, high)

        # Volume (random with trend)
        base_volume = 1000000
        volume = int(base_volume * (1 + 0.5 * np.random.randn()) * (1 + i / n_days))

        data.append({
            "Date": date,
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close_price,
            "Volume": volume,
        })

    df = pd.DataFrame(data)
    df = df.set_index("Date")

    return df


def generate_implied_volatility(
    dates: pd.DatetimeIndex,
    base_iv: float = 0.18,
    mean_reversion_speed: float = 0.1,
    vol_of_vol: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic implied volatility data.

    Args:
        dates: Date index.
        base_iv: Long-term mean IV.
        mean_reversion_speed: Speed of mean reversion.
        vol_of_vol: Volatility of implied volatility.
        seed: Random seed.

    Returns:
        DataFrame with ATM IV data.
    """
    np.random.seed(seed)

    n_days = len(dates)
    iv = np.zeros(n_days)
    iv[0] = base_iv

    # Generate IV with mean reversion
    daily_speed = mean_reversion_speed / 252
    daily_vol = vol_of_vol / np.sqrt(252)

    for t in range(1, n_days):
        # Mean reverting process with stochastic volatility
        drift = daily_speed * (base_iv - iv[t-1])
        diffusion = daily_vol * np.sqrt(iv[t-1]) * np.random.randn()
        iv[t] = iv[t-1] + drift + diffusion

        # Keep IV positive and reasonable
        iv[t] = np.clip(iv[t], 0.05, 0.80)

    # Create DataFrame
    df = pd.DataFrame({
        "Date": dates,
        "ATM_IV": iv * 100,  # Convert to percentage
    })
    df = df.set_index("Date")

    return df


def main():
    """Generate all sample data files."""
    print("Generating sample data files...")

    # Generate SPY-like data
    print("  Generating SPY data...")
    spy_data = generate_synthetic_ohlcv(
        start_date="2015-01-01",
        end_date="2024-12-31",
        initial_price=200.0,
        volatility=0.16,
        drift=0.10,
        seed=42,
    )
    spy_data.to_csv("data/spy.csv")
    print(f"    Created data/spy.csv with {len(spy_data)} rows")

    # Generate implied volatility data
    print("  Generating IV data...")
    iv_data = generate_implied_volatility(
        dates=spy_data.index,
        base_iv=0.16,
        mean_reversion_speed=0.2,
        vol_of_vol=0.4,
        seed=42,
    )
    iv_data.to_csv("data/atm_iv_sample.csv")
    print(f"    Created data/atm_iv_sample.csv with {len(iv_data)} rows")

    # Print sample statistics
    returns = np.log(spy_data["Close"] / spy_data["Close"].shift(1))
    print("\nSample Statistics:")
    print(f"  SPY Annual Return: {returns.mean() * 252:.1%}")
    print(f"  SPY Annual Volatility: {returns.std() * np.sqrt(252):.1%}")
    print(f"  SPY Sharpe Ratio: {(returns.mean() * 252) / (returns.std() * np.sqrt(252)):.2f}")
    print(f"  Average ATM IV: {iv_data['ATM_IV'].mean():.1f}%")
    print(f"  IV Range: {iv_data['ATM_IV'].min():.1f}% - {iv_data['ATM_IV'].max():.1f}%")

    print("\nSample data generation complete!")


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    main()