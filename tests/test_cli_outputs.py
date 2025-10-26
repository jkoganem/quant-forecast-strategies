"""Test that CLI creates expected output files."""

from __future__ import annotations

import subprocess
import os
import json
import pytest
import tempfile
import shutil


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_cli_creates_basic_outputs(temp_output_dir):
    """Test that CLI creates basic output files."""
    # Run CLI with minimal settings
    result = subprocess.run(
        [
            'python', '-m', 'multi_ts.cli',
            '--start', '2023-01-01',
            '--end', '2023-06-30',
            '--out', temp_output_dir,
            '--plots', 'true',
            '--json_bundle', 'true'
        ],
        capture_output=True,
        text=True,
        timeout=300
    )

    # Check that command succeeded
    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    # Check that expected files were created
    expected_files = [
        'forecasts.csv',
        'strategy_returns.csv',
        'strategy_cumulative.csv',
        'forecasts.png',
        'diagnostics.png',
        'equity_curves.png',
        'model_compare.png',
        'results_bundle.json',
        'report.txt'
    ]

    for filename in expected_files:
        filepath = os.path.join(temp_output_dir, filename)
        assert os.path.exists(filepath), f"Expected file not created: {filename}"
        assert os.path.getsize(filepath) > 0, f"File is empty: {filename}"


def test_cli_results_json_structure(temp_output_dir):
    """Test that results_bundle.json has correct structure."""
    # Run CLI
    result = subprocess.run(
        [
            'python', '-m', 'multi_ts.cli',
            '--start', '2023-01-01',
            '--end', '2023-03-31',
            '--out', temp_output_dir,
            '--plots', 'false',
            '--json_bundle', 'true'
        ],
        capture_output=True,
        text=True,
        timeout=300
    )

    assert result.returncode == 0

    # Load and check JSON structure
    json_path = os.path.join(temp_output_dir, 'results_bundle.json')
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Check top-level keys
    assert 'metadata' in data
    assert 'returns' in data
    assert 'forecasts' in data
    assert 'strategies' in data
    assert 'evaluation' in data
    assert 'diagnostics' in data

    # Check metadata structure
    assert 'generated_at' in data['metadata']
    assert 'start_date' in data['metadata']
    assert 'end_date' in data['metadata']

    # Check that forecasts is a dictionary with model names
    assert isinstance(data['forecasts'], dict)
    assert len(data['forecasts']) > 0

    # Check that strategies is a dictionary
    assert isinstance(data['strategies'], dict)
    assert len(data['strategies']) > 0


def test_cli_lightweight_mode(temp_output_dir):
    """Test that lightweight mode runs faster and skips SARIMA."""
    # Run with lightweight mode
    result = subprocess.run(
        [
            'python', '-m', 'multi_ts.cli',
            '--start', '2023-01-01',
            '--end', '2023-03-31',
            '--out', temp_output_dir,
            '--lightweight', 'true',
            '--plots', 'false',
            '--json_bundle', 'false'
        ],
        capture_output=True,
        text=True,
        timeout=180
    )

    assert result.returncode == 0

    # Check output mentions lightweight mode
    assert 'lightweight' in result.stdout.lower() or 'skip' in result.stdout.lower()


def test_cli_with_iv_csv(temp_output_dir):
    """Test CLI with implied volatility CSV."""
    # Check if IV file exists
    iv_path = './data/atm_iv_real.csv'
    if not os.path.exists(iv_path):
        pytest.skip("IV CSV not available")

    result = subprocess.run(
        [
            'python', '-m', 'multi_ts.cli',
            '--start', '2023-01-01',
            '--end', '2023-03-31',
            '--out', temp_output_dir,
            '--iv_csv', iv_path,
            '--plots', 'false',
            '--json_bundle', 'false'
        ],
        capture_output=True,
        text=True,
        timeout=300
    )

    assert result.returncode == 0


def test_cli_plots_flag_false(temp_output_dir):
    """Test that plots=false doesn't create plot files."""
    result = subprocess.run(
        [
            'python', '-m', 'multi_ts.cli',
            '--start', '2023-01-01',
            '--end', '2023-02-28',
            '--out', temp_output_dir,
            '--plots', 'false',
            '--json_bundle', 'false'
        ],
        capture_output=True,
        text=True,
        timeout=180
    )

    assert result.returncode == 0

    # CSV files should exist
    assert os.path.exists(os.path.join(temp_output_dir, 'forecasts.csv'))

    # Plot files should NOT exist
    plot_files = ['forecasts.png', 'diagnostics.png', 'equity_curves.png']
    for plot_file in plot_files:
        plot_path = os.path.join(temp_output_dir, plot_file)
        # Either doesn't exist or was created elsewhere
        if os.path.exists(plot_path):
            # If it exists in temp dir, that's an error
            pytest.fail(f"Plot file should not be created with --plots false: {plot_file}")


def test_cli_json_bundle_flag_false(temp_output_dir):
    """Test that json_bundle=false doesn't create JSON file."""
    result = subprocess.run(
        [
            'python', '-m', 'multi_ts.cli',
            '--start', '2023-01-01',
            '--end', '2023-02-28',
            '--out', temp_output_dir,
            '--plots', 'false',
            '--json_bundle', 'false'
        ],
        capture_output=True,
        text=True,
        timeout=180
    )

    assert result.returncode == 0

    # CSV files should exist
    assert os.path.exists(os.path.join(temp_output_dir, 'forecasts.csv'))

    # JSON should NOT exist
    json_path = os.path.join(temp_output_dir, 'results_bundle.json')
    assert not os.path.exists(json_path), "Results JSON should not be created with --json_bundle false"


def test_cli_forecast_csv_structure(temp_output_dir):
    """Test that forecast CSV has expected columns."""
    result = subprocess.run(
        [
            'python', '-m', 'multi_ts.cli',
            '--start', '2023-01-01',
            '--end', '2023-02-28',
            '--out', temp_output_dir,
            '--plots', 'false',
            '--json_bundle', 'false'
        ],
        capture_output=True,
        text=True,
        timeout=180
    )

    assert result.returncode == 0

    # Read and check CSV
    import pandas as pd
    csv_path = os.path.join(temp_output_dir, 'forecasts.csv')
    df = pd.read_csv(csv_path)

    # Check expected columns
    assert 'Date' in df.columns or 'date' in df.columns
    assert 'actual' in df.columns
    assert any('forecast' in col.lower() for col in df.columns)


def test_cli_report_txt_content(temp_output_dir):
    """Test that report.txt has expected content."""
    result = subprocess.run(
        [
            'python', '-m', 'multi_ts.cli',
            '--start', '2023-01-01',
            '--end', '2023-02-28',
            '--out', temp_output_dir,
            '--plots', 'false',
            '--json_bundle', 'false'
        ],
        capture_output=True,
        text=True,
        timeout=180
    )

    assert result.returncode == 0

    # Read report
    report_path = os.path.join(temp_output_dir, 'report.txt')
    with open(report_path, 'r') as f:
        report_content = f.read()

    # Check for expected sections
    assert 'FORECAST MODEL EVALUATION' in report_content
    assert 'STRATEGY PERFORMANCE' in report_content
    assert 'DIAGNOSTIC' in report_content


def test_cli_error_handling_invalid_dates():
    """Test that CLI handles invalid date ranges gracefully."""
    result = subprocess.run(
        [
            'python', '-m', 'multi_ts.cli',
            '--start', '2025-01-01',  # Future date
            '--end', '2024-01-01',    # End before start
            '--plots', 'false',
            '--json_bundle', 'false'
        ],
        capture_output=True,
        text=True,
        timeout=60
    )

    # Should fail with non-zero exit code
    assert result.returncode != 0


def test_cli_selective_models(temp_output_dir):
    """Test running only specific models."""
    result = subprocess.run(
        [
            'python', '-m', 'multi_ts.cli',
            '--start', '2023-01-01',
            '--end', '2023-02-28',
            '--out', temp_output_dir,
            '--models', 'garch,ridge',
            '--plots', 'false',
            '--json_bundle', 'false'
        ],
        capture_output=True,
        text=True,
        timeout=120
    )

    assert result.returncode == 0

    # Check that output mentions only selected models
    csv_path = os.path.join(temp_output_dir, 'forecasts.csv')
    import pandas as pd
    df = pd.read_csv(csv_path)

    # Should have garch and ridge forecasts
    assert any('garch' in col.lower() for col in df.columns)
    assert any('ridge' in col.lower() for col in df.columns)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
