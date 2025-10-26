"""Plotting functions for visualization.

Creates comprehensive plots for analysis results.
"""

import matplotlib

matplotlib.use("Agg")

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from multi_ts.datatypes import ForecastResult


def setup_plot_style():
    """Set up consistent plot styling."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10


def plot_forecasts(
    returns: pd.Series,
    forecasts: Dict[str, ForecastResult],
    last_n_days: int = 252,
    save_path: Optional[str] = None,
) -> None:
    """Plot actual vs forecasted returns.

    Args:
        returns: Actual returns.
        forecasts: Dictionary of forecast results.
        last_n_days: Number of recent days to plot.
        save_path: Optional path to save the plot.
    """
    setup_plot_style()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Filter to last N days
    plot_returns = returns.iloc[-last_n_days:]

    # Plot 1: Return forecasts
    ax1 = axes[0]
    ax1.plot(plot_returns.index, plot_returns.values, 'k-', alpha=0.7, label='Actual', linewidth=0.8)

    colors = plt.cm.tab10(np.linspace(0, 1, len(forecasts)))
    for (name, result), color in zip(forecasts.items(), colors):
        if "variance" not in name.lower():
            forecast_slice = result.yhat.iloc[-last_n_days:]
            ax1.plot(forecast_slice.index, forecast_slice.values,
                    alpha=0.6, label=name, color=color, linewidth=0.8)

    ax1.set_title('Return Forecasts vs Actual', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Returns')
    ax1.legend(loc='upper left', ncol=3, fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Variance forecasts
    ax2 = axes[1]
    # Calculate realized variance on full returns series, then slice
    realized_var_full = returns.rolling(20).var()
    realized_var = realized_var_full.iloc[-last_n_days:]
    ax2.plot(realized_var.index, realized_var.values, 'k-', alpha=0.7,
            label='Realized (20d)', linewidth=1)

    for name, result in forecasts.items():
        if "variance" in name.lower() or "garch" in name.lower():
            var_slice = result.yhat.iloc[-last_n_days:]
            ax2.plot(var_slice.index, var_slice.values,
                    alpha=0.7, label=name, linewidth=1)

    ax2.set_title('Variance Forecasts', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Variance')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_diagnostics(
    diagnostics: Dict[str, Dict],
    save_path: Optional[str] = None,
) -> None:
    """Plot diagnostic test results - handles flat structure from CLI.

    Args:
        diagnostics: Dictionary of diagnostic results (flat structure).
        save_path: Optional path to save the plot.
    """
    setup_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Extract model names from flat keys
    model_names = set()
    for key in diagnostics.keys():
        if '_' in key:
            if key.endswith('_ljung_box'):
                model = key.rsplit('_ljung_box', 1)[0]
                if model not in ['returns', 'diebold_mariano']:
                    model_names.add(model)

    models = sorted(list(model_names))[:10]  # Limit to 10 models for readability

    # Plot 1: Ljung-Box p-values (white noise test)
    ax1 = axes[0]
    test_items = []
    pvals = []

    # Add model residuals Ljung-Box tests
    for model in models[:10]:  # Show up to 10 models
        lb_key = f"{model}_ljung_box"
        if lb_key in diagnostics:
            lb_p = diagnostics[lb_key].get('p_value')  # Use 'p_value' not 'ljungbox_pvalue'
            if lb_p is not None:
                test_items.append(model.replace('_', ' ').title()[:15])
                pvals.append(lb_p)

    if test_items:
        # For Ljung-Box (white noise test): HIGH p-value (> 0.05) = white noise = GOOD (green)
        # LOW p-value (< 0.05) = autocorrelation present = BAD (red)
        colors = ['green' if p > 0.05 else 'red' if not np.isnan(p) else 'gray' for p in pvals]
        bars = ax1.barh(test_items, [min(p, 1.0) if not np.isnan(p) else 0.01 for p in pvals], color=colors)
        ax1.axvline(x=0.05, color='black', linestyle='--', alpha=0.5, linewidth=2)
        ax1.set_xlabel('p-value', fontsize=12)
        ax1.set_title('White Noise Test (Ljung-Box)\nGreen = White Noise (p > 0.05)', fontweight='bold', fontsize=13)
        ax1.set_xlim([0, 1.0])
        ax1.grid(True, alpha=0.3, axis='x')

    # Plot 2: Summary table with p-values
    ax2 = axes[1]
    ax2.axis('tight')
    ax2.axis('off')

    # Create detailed summary table
    summary_data = []
    for model in models[:10]:  # Show up to 10 models
        lb_key = f"{model}_ljung_box"

        lb_data = diagnostics.get(lb_key, {})
        lb_p = lb_data.get('p_value')  # Use 'p_value' not 'ljungbox_pvalue'
        lb_stat = lb_data.get('test_statistic')

        # Format values
        lb_p_str = f"{lb_p:.3f}" if lb_p is not None else "N/A"
        lb_stat_str = f"{lb_stat:.2f}" if lb_stat is not None else "N/A"

        # Determine white noise status
        white_noise = 'Yes' if lb_p is not None and lb_p > 0.05 else 'No'

        summary_data.append([
            model.replace('_', ' ').title()[:18],
            lb_stat_str,
            lb_p_str,
            white_noise
        ])

    if summary_data:
        table = ax2.table(cellText=summary_data,
                         colLabels=['Model', 'LB Statistic', 'p-value', 'White Noise'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.35, 0.20, 0.20, 0.20])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.0, 2.2)

        # Color the header row
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

    ax2.set_title('Ljung-Box Test Results\np > 0.05 = White Noise (Good) | p < 0.05 = Autocorrelation (Bad)',
                  fontweight='bold', fontsize=13, pad=20)

    plt.suptitle('Model Diagnostic Tests', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_equity_curves(
    strategies: Dict[str, Dict[str, pd.Series]],
    save_path: Optional[str] = None,
) -> None:
    """Plot strategy equity curves.

    Args:
        strategies: Dictionary of strategy results.
        save_path: Optional path to save the plot.
    """
    setup_plot_style()

    # Find common date range across all strategies
    common_start = None
    common_end = None

    for name, data in strategies.items():
        if "cum_net" in data:
            series = data["cum_net"].dropna()
            if len(series) > 0:
                start = series.index[0]
                end = series.index[-1]

                if common_start is None or start > common_start:
                    common_start = start
                if common_end is None or end < common_end:
                    common_end = end

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Cumulative returns (only common period)
    ax1 = axes[0]
    # Plot all strategies first (lower z-order)
    for name, data in strategies.items():
        if name == "vol_managed":
            continue  # Plot vol_managed last to be on top
        if "cum_net" in data:
            series = data["cum_net"].dropna()
            if len(series) > 0:
                # Filter to common period
                mask = (series.index >= common_start) & (series.index <= common_end)
                series_common = series[mask]

                # Rebase to start at 1.0 for fair comparison
                if len(series_common) > 0:
                    series_rebased = series_common / series_common.iloc[0]
                    ax1.plot(series_rebased.index, series_rebased.values,
                            label=name, alpha=0.5, linewidth=1.2)

    # Plot vol_managed strategy on top with emphasis
    if "vol_managed" in strategies and "cum_net" in strategies["vol_managed"]:
        series = strategies["vol_managed"]["cum_net"].dropna()
        if len(series) > 0:
            mask = (series.index >= common_start) & (series.index <= common_end)
            series_common = series[mask]
            if len(series_common) > 0:
                series_rebased = series_common / series_common.iloc[0]
                ax1.plot(series_rebased.index, series_rebased.values,
                        label="vol_managed", alpha=1.0, linewidth=3.5,
                        color='#FF6B35', zorder=10)  # Bright orange, on top

    ax1.set_title('Cumulative Returns (Net of Costs) - Common Period', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend(loc='best', ncol=2, fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: Drawdowns (only common period)
    ax2 = axes[1]
    # Plot all strategies first (lower z-order)
    for name, data in strategies.items():
        if name == "vol_managed":
            continue  # Plot vol_managed last to be on top
        if "cum_net" in data:
            series = data["cum_net"].dropna()
            if len(series) > 0:
                # Filter to common period
                mask = (series.index >= common_start) & (series.index <= common_end)
                cum_returns = series[mask]

                if len(cum_returns) > 0:
                    # Rebase for drawdown calculation
                    cum_returns_rebased = cum_returns / cum_returns.iloc[0]
                    running_max = cum_returns_rebased.cummax()
                    drawdown = ((cum_returns_rebased - running_max) / running_max * 100)
                    ax2.plot(drawdown.index, drawdown.values,
                            alpha=0.5, label=name, linewidth=1.2)

    # Plot vol_managed drawdown on top with emphasis
    if "vol_managed" in strategies and "cum_net" in strategies["vol_managed"]:
        series = strategies["vol_managed"]["cum_net"].dropna()
        if len(series) > 0:
            mask = (series.index >= common_start) & (series.index <= common_end)
            cum_returns = series[mask]
            if len(cum_returns) > 0:
                cum_returns_rebased = cum_returns / cum_returns.iloc[0]
                running_max = cum_returns_rebased.cummax()
                drawdown = ((cum_returns_rebased - running_max) / running_max * 100)
                ax2.plot(drawdown.index, drawdown.values,
                        alpha=1.0, label="vol_managed", linewidth=3.5,
                        color='#FF6B35', zorder=10)  # Bright orange, on top

    ax2.set_title('Drawdowns (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend(loc='best', ncol=2, fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(top=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_model_comparison(
    forecast_eval: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """Plot model comparison metrics.

    Args:
        forecast_eval: Forecast evaluation DataFrame.
        save_path: Optional path to save the plot.
    """
    setup_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Select top models
    top_models = forecast_eval.head(10)

    # Plot 1: MSE comparison
    ax1 = axes[0, 0]
    ax1.barh(top_models['model'], top_models['mse'])
    ax1.set_xlabel('MSE')
    ax1.set_title('Mean Squared Error', fontweight='bold')
    ax1.invert_yaxis()

    # Plot 2: MAE comparison
    ax2 = axes[0, 1]
    ax2.barh(top_models['model'], top_models['mae'])
    ax2.set_xlabel('MAE')
    ax2.set_title('Mean Absolute Error', fontweight='bold')
    ax2.invert_yaxis()

    # Plot 3: Directional accuracy
    ax3 = axes[1, 0]
    ax3.barh(top_models['model'], top_models['directional_accuracy'])
    ax3.set_xlabel('Accuracy (%)')
    ax3.set_title('Directional Accuracy', fontweight='bold')
    ax3.invert_yaxis()

    # Plot 4: Leaderboard table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')

    # Create leaderboard
    leaderboard_data = []
    for idx, row in top_models.head(5).iterrows():
        leaderboard_data.append([
            row['model'][:15],
            f"{row['mse']:.2e}",
            f"{row['mae']:.2e}",
            f"{row['directional_accuracy']:.1f}%"
        ])

    table = ax4.table(cellText=leaderboard_data,
                     colLabels=['Model', 'MSE', 'MAE', 'Dir. Acc.'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    ax4.set_title('Top 5 Models', fontweight='bold', pad=20)

    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_iv_analysis(
    iv_data: pd.DataFrame,
    vrp: pd.Series,
    save_path: Optional[str] = None,
) -> None:
    """Plot implied volatility analysis.

    Args:
        iv_data: DataFrame with IV data.
        vrp: Variance risk premium series.
        save_path: Optional path to save the plot.
    """
    setup_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: IV time series
    ax1 = axes[0, 0]
    if 'atm_iv' in iv_data.columns:
        ax1.plot(iv_data.index, iv_data['atm_iv'] * 100, 'b-', alpha=0.7)
        ax1.set_title('ATM Implied Volatility', fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('IV (%)')
        ax1.grid(True, alpha=0.3)

    # Plot 2: VRP time series
    ax2 = axes[0, 1]
    ax2.plot(vrp.index, vrp.values, 'r-', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('Variance Risk Premium', fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('VRP')
    ax2.grid(True, alpha=0.3)

    # Plot 3: IV vs Realized Vol
    ax3 = axes[1, 0]
    if 'returns' in iv_data.columns and 'atm_iv' in iv_data.columns:
        rv = iv_data['returns'].rolling(20).std() * np.sqrt(252) * 100
        valid_idx = ~(iv_data['atm_iv'].isna() | rv.isna())
        ax3.scatter(iv_data.loc[valid_idx, 'atm_iv'] * 100,
                   rv[valid_idx], alpha=0.3, s=10)

        # Add 45-degree line
        min_val = min(iv_data.loc[valid_idx, 'atm_iv'].min() * 100, rv[valid_idx].min())
        max_val = max(iv_data.loc[valid_idx, 'atm_iv'].max() * 100, rv[valid_idx].max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)

        ax3.set_title('IV vs Realized Volatility', fontweight='bold')
        ax3.set_xlabel('Implied Volatility (%)')
        ax3.set_ylabel('Realized Volatility (%)')
        ax3.grid(True, alpha=0.3)

    # Plot 4: VRP distribution
    ax4 = axes[1, 1]
    clean_vrp = vrp.dropna()
    ax4.hist(clean_vrp, bins=50, alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax4.axvline(x=clean_vrp.mean(), color='blue', linestyle='--',
               alpha=0.7, label=f'Mean: {clean_vrp.mean():.4f}')
    ax4.set_title('VRP Distribution', fontweight='bold')
    ax4.set_xlabel('VRP')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Implied Volatility Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_residual_diagnostics(
    residuals: Dict[str, pd.Series],
    save_path: Optional[str] = None,
) -> None:
    """Plot comprehensive residual diagnostics.

    Args:
        residuals: Dictionary of residual series by model.
        save_path: Optional path to save the plot.
    """
    setup_plot_style()

    # Select up to 4 models for detailed analysis
    models = list(residuals.keys())[:4]

    fig, axes = plt.subplots(len(models), 4, figsize=(16, 4 * len(models)))
    if len(models) == 1:
        axes = axes.reshape(1, -1)

    for i, model in enumerate(models):
        resid = residuals[model].dropna()

        # 1. Residuals time series
        ax1 = axes[i, 0]
        ax1.plot(resid.index, resid.values, alpha=0.7, linewidth=0.8)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax1.set_title(f'{model}: Residuals', fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Residual')
        ax1.grid(True, alpha=0.3)

        # 2. ACF plot
        ax2 = axes[i, 1]
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(resid.values, lags=20, ax=ax2, alpha=0.05)
        ax2.set_title(f'{model}: ACF', fontweight='bold')

        # 3. Q-Q plot
        ax3 = axes[i, 2]
        from scipy import stats
        stats.probplot(resid.values, dist="norm", plot=ax3)
        ax3.set_title(f'{model}: Q-Q Plot', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 4. Histogram
        ax4 = axes[i, 3]
        ax4.hist(resid.values, bins=50, alpha=0.7, edgecolor='black', density=True)

        # Overlay normal distribution
        x = np.linspace(resid.min(), resid.max(), 100)
        ax4.plot(x, stats.norm.pdf(x, resid.mean(), resid.std()), 'r-', alpha=0.7)
        ax4.set_title(f'{model}: Distribution', fontweight='bold')
        ax4.set_xlabel('Residual')
        ax4.set_ylabel('Density')
        ax4.grid(True, alpha=0.3)

    plt.suptitle('Residual Diagnostics', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_rolling_performance(
    strategies: Dict[str, Dict[str, pd.Series]],
    window: int = 252,
    save_path: Optional[str] = None,
) -> None:
    """Plot rolling performance metrics.

    Args:
        strategies: Dictionary of strategy results.
        window: Rolling window size (default 252 for annual).
        save_path: Optional path to save the plot.
    """
    setup_plot_style()

    # Find common date range across all strategies (same logic as equity_curves)
    common_start = None
    common_end = None

    for name, data in strategies.items():
        if "net_returns" in data:
            series = data["net_returns"].dropna()
            if len(series) > 0:
                start = series.index[0]
                end = series.index[-1]

                if common_start is None or start > common_start:
                    common_start = start
                if common_end is None or end < common_end:
                    common_end = end

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    ax1, ax2, ax3 = axes[0], axes[1], axes[2]

    for name, data in strategies.items():
        if "net_returns" in data:
            returns = data["net_returns"].dropna()

            # Filter to common period
            if len(returns) > 0 and common_start is not None and common_end is not None:
                mask = (returns.index >= common_start) & (returns.index <= common_end)
                returns_common = returns[mask]

                if len(returns_common) > window:
                    # Calculate rolling metrics on common period
                    rolling_return = returns_common.rolling(window).mean() * 252
                    rolling_vol = returns_common.rolling(window).std() * np.sqrt(252)
                    rolling_sharpe = rolling_return / rolling_vol

                    # Plot 1: Rolling returns
                    ax1.plot(rolling_return.index, rolling_return.values * 100,
                            label=name, alpha=0.7, linewidth=1)

                    # Plot 2: Rolling volatility
                    ax2.plot(rolling_vol.index, rolling_vol.values * 100,
                            label=name, alpha=0.7, linewidth=1)

                    # Plot 3: Rolling Sharpe
                    ax3.plot(rolling_sharpe.index, rolling_sharpe.values,
                            label=name, alpha=0.7, linewidth=1)

    # Format plots
    ax1.set_title(f'{window}-Day Rolling Annualized Return', fontweight='bold')
    ax1.set_ylabel('Return (%)')
    ax1.legend(loc='upper left', ncol=3, fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    ax2.set_title(f'{window}-Day Rolling Annualized Volatility', fontweight='bold')
    ax2.set_ylabel('Volatility (%)')
    ax2.legend(loc='upper left', ncol=3, fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3.set_title(f'{window}-Day Rolling Sharpe Ratio', fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.legend(loc='upper left', ncol=3, fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    plt.suptitle('Rolling Performance Metrics', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_correlation_heatmap(
    forecasts: Dict[str, ForecastResult],
    save_path: Optional[str] = None,
) -> None:
    """Plot correlation heatmap between model forecasts.

    Args:
        forecasts: Dictionary of forecast results.
        save_path: Optional path to save the plot.
    """
    setup_plot_style()

    # Create DataFrame of forecasts
    forecast_df = pd.DataFrame()
    for name, result in forecasts.items():
        if "variance" not in name.lower():
            forecast_df[name] = result.yhat

    # Calculate correlation matrix
    corr_matrix = forecast_df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0.5, square=True, linewidths=1,
                cbar_kws={"shrink": 0.8}, ax=ax)

    ax.set_title('Model Forecast Correlation Matrix', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_forecast_errors(
    returns: pd.Series,
    forecasts: Dict[str, ForecastResult],
    save_path: Optional[str] = None,
) -> None:
    """Plot forecast errors analysis.

    Args:
        returns: Actual returns.
        forecasts: Dictionary of forecast results.
        save_path: Optional path to save the plot.
    """
    setup_plot_style()

    # Calculate errors for each model
    errors = {}
    for name, result in forecasts.items():
        if "variance" not in name.lower():
            common_idx = returns.index.intersection(result.yhat.index)
            errors[name] = returns[common_idx] - result.yhat[common_idx]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Error distributions
    ax1 = axes[0, 0]
    error_data = [e.dropna().values for e in errors.values()]
    bp = ax1.boxplot(error_data, labels=list(errors.keys()), patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax1.set_title('Forecast Error Distributions', fontweight='bold')
    ax1.set_ylabel('Forecast Error')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

    # Plot 2: Cumulative absolute errors
    ax2 = axes[0, 1]
    for name, error in errors.items():
        cum_abs_error = error.abs().cumsum()
        ax2.plot(cum_abs_error.index, cum_abs_error.values,
                label=name, alpha=0.7, linewidth=1)
    ax2.set_title('Cumulative Absolute Errors', fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cumulative |Error|')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Error autocorrelation
    ax3 = axes[1, 0]
    for i, (name, error) in enumerate(errors.items()):
        clean_error = error.dropna()
        if len(clean_error) > 10:
            autocorr = [clean_error.autocorr(lag=j) for j in range(1, 11)]
            ax3.plot(range(1, 11), autocorr, marker='o',
                    label=name, alpha=0.7)
    ax3.set_title('Error Autocorrelation', fontweight='bold')
    ax3.set_xlabel('Lag')
    ax3.set_ylabel('Autocorrelation')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Error statistics table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')

    # Calculate statistics
    stats_data = []
    for name, error in errors.items():
        clean_error = error.dropna()
        stats_data.append([
            name[:12],
            f"{clean_error.mean():.2e}",
            f"{clean_error.std():.2e}",
            f"{clean_error.skew():.2f}",
            f"{clean_error.kurtosis():.2f}"
        ])

    table = ax4.table(cellText=stats_data,
                     colLabels=['Model', 'Mean', 'Std', 'Skew', 'Kurt'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title('Error Statistics', fontweight='bold', pad=20)

    plt.suptitle('Forecast Error Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
