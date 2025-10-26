"""Main forecasting orchestrator.

Coordinates all models and generates ensemble forecasts.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from multi_ts.datatypes import ForecastResult, TSConfig
from multi_ts.models.garch import rolling_garch_forecast
from multi_ts.models.sarima import rolling_sarima_forecast
from multi_ts.models.dhr import rolling_dhr_forecast
from multi_ts.models.linear import rolling_linear_forecast
from multi_ts.models.xgb import rolling_xgboost_forecast
from multi_ts.features.iv_features import join_atm_iv


def naive_forecast(
    returns: pd.Series,
    window: int = 500,
) -> ForecastResult:
    """Generate naive forecast using historical mean.

    Args:
        returns: Returns series.
        window: Window size for rolling mean.

    Returns:
        ForecastResult with naive forecasts.
    """
    n = len(returns)
    forecasts = np.full(n, np.nan)

    for t in range(window, n):
        # Use expanding mean up to t-1
        forecasts[t] = returns.iloc[:t].mean()

    forecast_series = pd.Series(forecasts, index=returns.index)
    residuals = returns - forecast_series

    meta = {
        "model": "Naive-Mean",
        "window": window,
        "n_forecasts": np.sum(~np.isnan(forecasts)),
    }

    return ForecastResult(
        yhat=forecast_series,
        resid=residuals,
        meta=meta,
    )


def seasonal_naive_forecast(
    returns: pd.Series,
    window: int = 500,
    season_length: int = 5,
) -> ForecastResult:
    """Generate seasonal naive forecast.

    Uses same day-of-week from previous periods.

    Args:
        returns: Returns series.
        window: Minimum window size.
        season_length: Seasonal period (5 for weekly).

    Returns:
        ForecastResult with seasonal naive forecasts.
    """
    n = len(returns)
    forecasts = np.full(n, np.nan)

    for t in range(window, n):
        # Get same season from previous periods
        season_values = []
        for back_period in range(1, min(10, t // season_length)):
            idx = t - back_period * season_length
            if idx >= 0:
                season_values.append(returns.iloc[idx])

        if season_values:
            forecasts[t] = np.mean(season_values)
        else:
            forecasts[t] = returns.iloc[:t].mean()

    forecast_series = pd.Series(forecasts, index=returns.index)
    residuals = returns - forecast_series

    meta = {
        "model": "Seasonal-Naive",
        "window": window,
        "season_length": season_length,
        "n_forecasts": np.sum(~np.isnan(forecasts)),
    }

    return ForecastResult(
        yhat=forecast_series,
        resid=residuals,
        meta=meta,
    )


def rolling_variance_baseline(
    returns: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Calculate rolling variance baseline.

    Args:
        returns: Returns series.
        window: Rolling window size.

    Returns:
        Series of rolling variances.
    """
    return returns.rolling(window=window).var()


def run_all_models(
    returns: pd.Series,
    config: TSConfig,
    iv_data: Optional[pd.DataFrame] = None,
) -> Dict[str, ForecastResult]:
    """Run all forecasting models.

    Args:
        returns: Returns series.
        config: Configuration object.
        iv_data: Optional implied volatility data.

    Returns:
        Dictionary mapping model names to ForecastResults.
    """
    results = {}

    # Set common parameters
    window = config.window_init_days
    seed = config.seed
    model_filter_env = os.environ.get("MODEL_FILTER")
    model_whitelist = None
    if model_filter_env:
        model_whitelist = {
            name.strip().lower()
            for name in model_filter_env.split(",")
            if name.strip()
        }

    print("Running forecasting models...")

    # 1. Naive models
    if model_whitelist is None or "naive" in model_whitelist or "naive_mean" in model_whitelist:
        print("  - Naive mean...")
        results["naive_mean"] = naive_forecast(returns, window)

    if model_whitelist is None or "seasonal_naive" in model_whitelist:
        print("  - Seasonal naive...")
        results["seasonal_naive"] = seasonal_naive_forecast(returns, window, season_length=5)

    # 2. GARCH for variance
    if model_whitelist is None or "garch" in model_whitelist or "garch_variance" in model_whitelist:
        print("  - GARCH(1,1)...")
        results["garch_variance"] = rolling_garch_forecast(returns, window, seed)

    # 3. SARIMA
    skip_sarima = os.environ.get("SKIP_SARIMA") == "1"
    if not skip_sarima and (model_whitelist is None or "sarima" in model_whitelist):
        print("  - SARIMA...")
        results["sarima"] = rolling_sarima_forecast(
            returns, window, seasonal_period=5, refit_freq=20
        )

    # 4. DHR
    if model_whitelist is None or "dhr" in model_whitelist:
        print("  - DHR...")
        results["dhr"] = rolling_dhr_forecast(
            returns,
            window,
            periods=[5, 21, 252],
            n_terms=[2, 2, 4],
            arima_order=(1, 0, 1),
            refit_freq=20,
        )

    # 5. Linear models
    if model_whitelist is None or "ridge" in model_whitelist:
        print("  - Ridge regression...")
        results["ridge"] = rolling_linear_forecast(
            returns,
            window,
            model_type="ridge",
            n_lags=10,
            refit_freq=20,
            include_iv=(iv_data is not None),
            iv_data=iv_data,
            seed=seed,
        )

    if model_whitelist is None or "lasso" in model_whitelist:
        print("  - Lasso regression...")
        results["lasso"] = rolling_linear_forecast(
            returns,
            window,
            model_type="lasso",
            n_lags=10,
            refit_freq=20,
            include_iv=(iv_data is not None),
            iv_data=iv_data,
            seed=seed,
        )

    # 6. XGBoost
    if model_whitelist is None or "xgboost" in model_whitelist or "xgb" in model_whitelist:
        print("  - XGBoost...")
        results["xgboost"] = rolling_xgboost_forecast(
            returns,
            window,
            n_lags=10,
            refit_freq=20,
            num_rounds=100,
            early_stopping_rounds=20,
            include_iv=(iv_data is not None),
            iv_data=iv_data,
            seed=seed,
        )

    return results


def create_ensemble_forecast(
    forecasts: Dict[str, ForecastResult],
    weights: Optional[Dict[str, float]] = None,
    method: str = "equal",
) -> ForecastResult:
    """Create ensemble forecast from multiple models.

    Args:
        forecasts: Dictionary of model forecasts.
        weights: Optional weights for models.
        method: Ensembling method ("equal", "inverse_mse", "trimmed").

    Returns:
        Ensemble ForecastResult.
    """
    # Extract forecast series
    forecast_df = pd.DataFrame({
        name: result.yhat
        for name, result in forecasts.items()
    })

    # Drop variance models (different scale)
    forecast_df = forecast_df.drop(columns=[col for col in forecast_df.columns if "variance" in col], errors="ignore")

    if method == "equal":
        # Equal weighting
        ensemble = forecast_df.mean(axis=1)
        used_weights = {col: 1/len(forecast_df.columns) for col in forecast_df.columns}

    elif method == "inverse_mse":
        # Weight by inverse MSE
        mse_dict = {}
        for name, result in forecasts.items():
            if "variance" not in name:
                mse = np.nanmean(result.resid ** 2)
                mse_dict[name] = mse

        # Calculate weights
        inv_mse = {name: 1/mse for name, mse in mse_dict.items() if mse > 0}
        total = sum(inv_mse.values())
        used_weights = {name: w/total for name, w in inv_mse.items()}

        # Apply weights
        ensemble = sum(forecast_df[name] * weight
                      for name, weight in used_weights.items()
                      if name in forecast_df.columns)

    elif method == "trimmed":
        # Trimmed mean (remove best and worst at each point)
        ensemble = forecast_df.apply(
            lambda row: row.sort_values().iloc[1:-1].mean()
            if len(row.dropna()) > 2 else row.mean(),
            axis=1,
        )
        used_weights = None

    else:
        # Custom weights
        if weights is None:
            weights = {col: 1/len(forecast_df.columns) for col in forecast_df.columns}
        ensemble = sum(forecast_df[name] * weights.get(name, 0)
                      for name in forecast_df.columns)
        used_weights = weights

    # Get actual returns for residuals
    # Use the first model's residuals to get actual values
    first_model = list(forecasts.values())[0]
    actual = first_model.yhat + first_model.resid
    residuals = actual - ensemble

    meta = {
        "model": f"Ensemble-{method}",
        "n_models": len(forecast_df.columns),
        "models": list(forecast_df.columns),
        "weights": used_weights,
        "method": method,
    }

    return ForecastResult(
        yhat=ensemble,
        resid=residuals,
        meta=meta,
    )


def prepare_forecast_data(
    returns: pd.Series,
    config: TSConfig,
) -> tuple[pd.Series, Optional[pd.DataFrame]]:
    """Prepare data for forecasting.

    Args:
        returns: Raw returns series.
        config: Configuration object.

    Returns:
        Tuple of (filtered returns, optional IV data).
    """
    # Filter by date range
    returns = returns.loc[config.start_date:config.end_date]

    # Load IV data if provided
    iv_data = None
    if config.iv_csv:
        # Create DataFrame with returns
        data_df = pd.DataFrame({"returns": returns})

        # Join IV data
        iv_data = join_atm_iv(config.iv_csv, data_df)

    return returns, iv_data
