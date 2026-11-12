"""
Unit tests for strategy backtester calculations and quantitative metrics.
"""

import pytest
import pandas as pd
import numpy as np

from src.backtester import run_backtest


def test_backtester_execution_and_metrics():
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    # Steady upward trending prices
    prices = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0]
    # In position for days 2 to 6, then flat
    positions = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    signals = [0, 0, 1, 0, 0, 0, -1, 0, 0, 0]

    df = pd.DataFrame(
        {
            "Close": prices,
            "Position": positions,
            "Signal": signals,
        },
        index=dates,
    )

    out = run_backtest(df, initial_capital=10_000.0, transaction_cost_pct=0.0)

    results_df = out["results_df"]
    metrics = out["metrics"]
    trades_df = out["trades_df"]

    assert "Strategy_Equity" in results_df.columns
    assert "Benchmark_Equity" in results_df.columns
    assert "Strategy_Drawdown" in results_df.columns

    # Verify trades extracted
    assert len(trades_df) == 1
    assert trades_df.iloc[0]["Result"] == "WIN"

    # Verify metrics structure
    assert "Strategy Total Return (%)" in metrics
    assert "Benchmark Total Return (%)" in metrics
    assert "Max Drawdown (%)" in metrics
    assert "Sharpe Ratio" in metrics
    assert "Win Rate (%)" in metrics
    assert metrics["Winning Trades"] == 1
    assert metrics["Losing Trades"] == 0
    assert metrics["Win Rate (%)"] == 100.0


def test_backtester_zero_trades():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    df = pd.DataFrame(
        {
            "Close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "Position": [0, 0, 0, 0, 0],
            "Signal": [0, 0, 0, 0, 0],
        },
        index=dates,
    )

    out = run_backtest(df, initial_capital=50_000.0)
    assert out["metrics"]["Total Round-Trip Trades"] == 0
    assert out["metrics"]["Ending Portfolio Value (₹)"] == 50_000.0

