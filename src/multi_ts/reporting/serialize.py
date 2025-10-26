"""Serialization for UI and data export.

Creates JSON bundles for frontend consumption.
"""

import json
from typing import Any, Dict, List
from datetime import datetime

import numpy as np
import pandas as pd

from multi_ts.datatypes import ForecastResult


def series_to_json(series: pd.Series, name: str = "value") -> List[Dict]:
    """Convert pandas Series to JSON-serializable format.

    Args:
        series: Pandas Series to convert.
        name: Name for the value field.

    Returns:
        List of dictionaries with date and value.
    """
    clean_series = series.dropna()
    data = []

    for date, value in clean_series.items():
        data.append({
            "date": date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date),
            name: float(value) if not np.isnan(value) else None,
        })

    return data


def create_results_bundle(
    returns: pd.Series,
    forecasts: Dict[str, ForecastResult],
    strategies: Dict[str, Dict[str, pd.Series]],
    forecast_eval: pd.DataFrame,
    strategy_eval: pd.DataFrame,
    diagnostics: Dict[str, Dict],
    config: Any,
) -> Dict:
    """Create comprehensive results bundle in JSON format.

    Args:
        returns: Actual returns series.
        forecasts: Dictionary of forecast results.
        strategies: Dictionary of strategy results.
        forecast_eval: Forecast evaluation DataFrame.
        strategy_eval: Strategy evaluation DataFrame.
        diagnostics: Diagnostic test results.
        config: Configuration object.

    Returns:
        Dictionary ready for JSON serialization.
    """
    bundle = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "start_date": config.start_date,
            "end_date": config.end_date,
            "n_observations": len(returns),
            "target_vol_ann": config.target_vol_ann,
            "w_max": config.w_max,
            "tc_bps": config.tc_bps,
        },
        "returns": {
            "actual": series_to_json(returns, "return"),
        },
        "forecasts": {},
        "strategies": {},
        "evaluation": {
            "forecasts": forecast_eval.to_dict("records"),
            "strategies": strategy_eval.to_dict("records"),
        },
        "diagnostics": {},
    }

    # Add forecast data
    for name, result in forecasts.items():
        bundle["forecasts"][name] = {
            "predictions": series_to_json(result.yhat, "forecast"),
            "residuals": series_to_json(result.resid, "residual"),
            "meta": result.meta,
        }

    # Add strategy data
    for name, data in strategies.items():
        bundle["strategies"][name] = {
            "returns": series_to_json(data["net_returns"], "return"),
            "cumulative": series_to_json(data["cum_net"], "cumulative"),
            "weights": series_to_json(data["weights"], "weight"),
            "costs": series_to_json(data["costs"], "cost"),
        }

    # Add diagnostic summaries
    # Note: diagnostics dict has keys like "dhr_ljung_box", "returns_adf", etc.
    # Each value is the direct output from the test function
    for key, test_result in diagnostics.items():
        # Skip error keys
        if "_error" in key:
            continue
        # Add all test results directly
        if isinstance(test_result, dict):
            bundle["diagnostics"][key] = test_result

    return bundle


def save_json_bundle(
    bundle: Dict,
    filepath: str,
) -> None:
    """Save JSON bundle to file.

    Args:
        bundle: Dictionary to save.
        filepath: Path to save the JSON file.
    """
    # Convert NaN and inf to null
    def clean_value(v):
        if isinstance(v, float):
            if np.isnan(v) or np.isinf(v):
                return None
        return v

    def clean_dict(d):
        if isinstance(d, dict):
            return {k: clean_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [clean_dict(item) for item in d]
        else:
            return clean_value(d)

    clean_bundle = clean_dict(bundle)

    def convert_to_json_type(obj):
        """Convert numpy/pandas types to JSON-serializable types."""
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        return str(obj)

    with open(filepath, 'w') as f:
        json.dump(clean_bundle, f, indent=2, default=convert_to_json_type)


def export_results_to_csv(
    returns: pd.Series,
    forecasts: Dict[str, ForecastResult],
    strategies: Dict[str, Dict[str, pd.Series]],
    output_dir: str,
) -> None:
    """Export results to CSV files.

    Args:
        returns: Actual returns.
        forecasts: Dictionary of forecast results.
        strategies: Dictionary of strategy results.
        output_dir: Directory to save CSV files.
    """
    import os

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Export forecasts
    forecast_df = pd.DataFrame({"actual": returns})
    for name, result in forecasts.items():
        forecast_df[f"{name}_forecast"] = result.yhat
        forecast_df[f"{name}_residual"] = result.resid

    forecast_df.to_csv(os.path.join(output_dir, "forecasts.csv"))

    # Export strategies
    strategy_returns = pd.DataFrame()
    strategy_cumulative = pd.DataFrame()

    for name, data in strategies.items():
        strategy_returns[name] = data["net_returns"]
        strategy_cumulative[name] = data["cum_net"]

    strategy_returns.to_csv(os.path.join(output_dir, "strategy_returns.csv"))
    strategy_cumulative.to_csv(os.path.join(output_dir, "strategy_cumulative.csv"))
