"""Table generation for reporting.

Creates formatted tables for results display.
"""

from typing import Dict, Optional

import pandas as pd


def create_forecast_summary_table(
    forecast_eval: pd.DataFrame,
) -> str:
    """Create formatted forecast evaluation summary table.

    Args:
        forecast_eval: Forecast evaluation DataFrame.

    Returns:
        Formatted string table.
    """
    # Select columns to display
    display_cols = ['model', 'mse', 'mae', 'directional_accuracy', 'r2']

    # Format the table
    summary = forecast_eval[display_cols].head(10).copy()

    # Format numeric columns
    summary['mse'] = summary['mse'].apply(lambda x: f"{x:.2e}")
    summary['mae'] = summary['mae'].apply(lambda x: f"{x:.2e}")
    summary['directional_accuracy'] = summary['directional_accuracy'].apply(lambda x: f"{x:.1f}%")
    summary['r2'] = summary['r2'].apply(lambda x: f"{x:.3f}")

    # Create string representation
    table_str = "=" * 70 + "\n"
    table_str += "FORECAST MODEL EVALUATION\n"
    table_str += "=" * 70 + "\n\n"
    table_str += summary.to_string(index=False)
    table_str += "\n" + "=" * 70 + "\n"

    return table_str


def create_strategy_summary_table(
    strategy_eval: pd.DataFrame,
) -> str:
    """Create formatted strategy evaluation summary table.

    Args:
        strategy_eval: Strategy evaluation DataFrame.

    Returns:
        Formatted string table.
    """
    # Select columns to display
    display_cols = ['strategy', 'annual_return', 'annual_volatility',
                   'sharpe_ratio', 'max_drawdown']

    # Format the table
    summary = strategy_eval[display_cols].head(10).copy()

    # Format numeric columns
    summary['annual_return'] = summary['annual_return'].apply(lambda x: f"{x:.1f}%")
    summary['annual_volatility'] = summary['annual_volatility'].apply(lambda x: f"{x:.1f}%")
    summary['sharpe_ratio'] = summary['sharpe_ratio'].apply(lambda x: f"{x:.2f}")
    summary['max_drawdown'] = summary['max_drawdown'].apply(lambda x: f"{x:.1f}%")

    # Create string representation
    table_str = "=" * 70 + "\n"
    table_str += "STRATEGY PERFORMANCE EVALUATION\n"
    table_str += "=" * 70 + "\n\n"
    table_str += summary.to_string(index=False)
    table_str += "\n" + "=" * 70 + "\n"

    return table_str


def create_diagnostic_summary_table(
    diagnostics: Dict[str, Dict],
) -> str:
    """Create formatted diagnostic test summary table.

    Args:
        diagnostics: Dictionary of diagnostic results.

    Returns:
        Formatted string table.
    """
    # Build summary data
    summary_data = []
    for model, tests in diagnostics.items():
        row = {
            'Model': model,
            'Stationary (ADF)': 'Yes' if tests.get('adf', {}).get('is_stationary', False) else 'No',
            'White Noise (LB)': 'Yes' if tests.get('ljung_box', {}).get('is_white', False) else 'No',
            'No ARCH': 'Yes' if not tests.get('arch_lm', {}).get('has_arch', True) else 'No',
        }
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)

    # Create string representation
    table_str = "=" * 70 + "\n"
    table_str += "DIAGNOSTIC TEST SUMMARY\n"
    table_str += "=" * 70 + "\n\n"
    table_str += summary_df.to_string(index=False)
    table_str += "\n" + "=" * 70 + "\n"

    return table_str


def create_diebold_mariano_table(
    dm_results: pd.DataFrame,
) -> str:
    """Create formatted Diebold-Mariano test results table.

    Args:
        dm_results: DataFrame with DM test results.

    Returns:
        Formatted string table.
    """
    # Format the table
    summary = dm_results.copy()

    # Format numeric columns
    if 'dm_statistic' in summary.columns:
        summary['dm_statistic'] = summary['dm_statistic'].apply(lambda x: f"{x:.3f}")
    if 'p_value' in summary.columns:
        summary['p_value'] = summary['p_value'].apply(lambda x: f"{x:.4f}")
    if 'significant' in summary.columns:
        summary['significant'] = summary['significant'].apply(lambda x: 'Yes' if x else 'No')

    # Create string representation
    table_str = "=" * 70 + "\n"
    table_str += "DIEBOLD-MARIANO TEST RESULTS\n"
    table_str += "=" * 70 + "\n\n"
    table_str += summary.to_string(index=False)
    table_str += "\n" + "=" * 70 + "\n"

    return table_str


def create_comprehensive_report(
    forecast_eval: pd.DataFrame,
    strategy_eval: pd.DataFrame,
    diagnostics: Dict[str, Dict],
    dm_results: Optional[pd.DataFrame] = None,
) -> str:
    """Create comprehensive text report.

    Args:
        forecast_eval: Forecast evaluation DataFrame.
        strategy_eval: Strategy evaluation DataFrame.
        diagnostics: Dictionary of diagnostic results.
        dm_results: Optional Diebold-Mariano results.

    Returns:
        Complete formatted report string.
    """
    report = "=" * 80 + "\n"
    report += " " * 20 + "MULTI-MODEL TIME SERIES ANALYSIS REPORT\n"
    report += "=" * 80 + "\n\n"

    # Add forecast evaluation
    report += create_forecast_summary_table(forecast_eval)
    report += "\n"

    # Add strategy evaluation
    report += create_strategy_summary_table(strategy_eval)
    report += "\n"

    # Add diagnostic summary
    report += create_diagnostic_summary_table(diagnostics)
    report += "\n"

    # Add Diebold-Mariano if available
    if dm_results is not None and len(dm_results) > 0:
        report += create_diebold_mariano_table(dm_results)
        report += "\n"

    # Add footer
    report += "=" * 80 + "\n"
    report += "Report generated with Multi-TS Forecasting System\n"
    report += "=" * 80 + "\n"

    return report
